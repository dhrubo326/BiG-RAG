"""
Educational Domain Knowledge Graph Builder

End-to-end wrapper combining:
- ProductionKGPipeline (extraction + validation)
- BipartiteGraphBuilder (graph construction)
- BiGRAG (storage and retrieval)

Designed for educational admission documents (KUET, BUET, RUET, CUET, etc.)
with 99%+ accuracy requirements.

Usage:
    from bigrag.educational_pipeline import build_educational_kg

    # Build knowledge graph from multiple documents
    rag, results = await build_educational_kg(
        markdown_documents=[kuet_doc, buet_doc],
        document_metadata=[
            {'title': 'KUET Admission 2024-25', 'category': 'university'},
            {'title': 'BUET Admission 2024-25', 'category': 'university'}
        ],
        api_key="your-openai-key",
        working_dir="./expr/educational_kg"
    )

    # Query the knowledge graph
    contexts = rag.query("কুয়েটে CSE বিভাগে কতটি আসন আছে?")
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from bigrag import BiGRAG
from bigrag.production_pipeline import ProductionKGPipeline
from bigrag.builders.bipartite_graph_builder import build_bipartite_graph_from_pipeline
from bigrag.utils import logger


async def build_educational_kg(
    markdown_documents: List[str],
    document_metadata: List[Dict],
    api_key: str,
    working_dir: str = "./expr/educational_kg",
    validation_level: str = "STRICT",
    enable_entity_linking: bool = True,
    extraction_mode: str = "semi_structured",
    chunk_token_size: int = 1200,
    chunk_overlap: int = 100,
) -> Tuple[BiGRAG, List[Dict]]:
    """
    Build knowledge graph from educational documents (KUET, BUET, etc.).

    This function orchestrates the complete pipeline:
    1. Initialize BiGRAG storage
    2. For each document:
       a. Extract entities/relations using ProductionKGPipeline
       b. Validate extraction (numeric accuracy + consistency)
       c. Build bipartite graph structure
       d. Index to vector DBs
    3. Save graph to disk

    Args:
        markdown_documents: List of markdown document contents
        document_metadata: List of metadata dicts (title, category, tags)
            Example: {'title': 'KUET Admission', 'category': 'university', 'tags': ['engineering']}
        api_key: OpenAI API key for GPT-4o extraction
        working_dir: Directory to save graph files (default: ./expr/educational_kg)
        validation_level: "STRICT" (99%+), "MODERATE" (95%+), "LENIENT" (90%+)
        enable_entity_linking: Whether to merge duplicate entities (recommended: True)
        extraction_mode: Validation mode (structured/semi_structured/unstructured) [DEFAULT: semi_structured]
            - structured: 99%+ accuracy for tables
            - semi_structured: 95%+ accuracy for mixed content [DEFAULT]
            - unstructured: 80%+ accuracy for narrative text
        chunk_token_size: Maximum chunk size in tokens (default: 1200)
        chunk_overlap: Overlap between chunks in tokens (default: 100)

    Returns:
        (rag_instance, pipeline_results)
        - rag_instance: BiGRAG instance with populated knowledge graph
        - pipeline_results: List of pipeline results for each document

    Example:
        import asyncio
        from bigrag.educational_pipeline import build_educational_kg

        # Load documents
        kuet_doc = open('datasets/KUET/admission.md').read()
        buet_doc = open('datasets/BUET/admission.md').read()

        documents = [kuet_doc, buet_doc]
        metadata = [
            {'title': 'KUET Admission 2024-25', 'category': 'university', 'tags': ['engineering']},
            {'title': 'BUET Admission 2024-25', 'category': 'university', 'tags': ['engineering']}
        ]

        # Build knowledge graph
        rag, results = asyncio.run(build_educational_kg(
            documents,
            metadata,
            api_key="your-openai-key",
            validation_level="STRICT"
        ))

        # Check validation results
        for i, result in enumerate(results):
            status = result['validation']['overall_status']
            print(f"Document {i+1}: {status}")

        # Query the graph
        contexts = rag.query("কুয়েটে CSE বিভাগে কতটি আসন আছে?")
        print(contexts)
    """

    # Validate inputs
    if len(markdown_documents) != len(document_metadata):
        raise ValueError(
            f"Number of documents ({len(markdown_documents)}) must match "
            f"metadata ({len(document_metadata)})"
        )

    logger.info("=" * 80)
    logger.info("Educational Knowledge Graph Builder")
    logger.info("=" * 80)
    logger.info(f"Documents: {len(markdown_documents)}")
    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Validation level: {validation_level}")
    logger.info(f"Entity linking: {enable_entity_linking}")
    logger.info("=" * 80)

    # Initialize BiGRAG instance
    rag = BiGRAG(
        working_dir=working_dir,
        enable_llm_cache=True,
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap,
        addon_params={
            "language": "English",
            "entity_types": [
                "department", "faculty", "university",
                "department_code", "seat_count", "gpa_requirement",
                "fee", "deadline", "time", "event", "location",
                "person", "organization", "concept", "number"
            ]
        }
    )

    # Initialize production pipeline
    pipeline = ProductionKGPipeline(
        api_key=api_key,
        validation_level=validation_level,
        enable_entity_linking=enable_entity_linking,
        extraction_mode=extraction_mode
    )

    # Process each document
    all_results = []
    total_entities = 0
    total_relations = 0
    total_edges = 0
    failed_docs = []

    for i, (doc_text, doc_meta) in enumerate(zip(markdown_documents, document_metadata)):
        doc_title = doc_meta.get('title', f'Document {i+1}')
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Document {i+1}/{len(markdown_documents)}: {doc_title}")
        logger.info(f"{'='*80}")

        try:
            # Run extraction pipeline
            result = await pipeline.process_document(
                doc_text,
                metadata=doc_meta,
                language="English"
            )

            # Check validation status
            validation_status = result['validation']['overall_status']
            numeric_coverage = result['validation']['numeric']['numeric_coverage']
            consistency_score = result['validation']['consistency']['consistency_score']
            extraction_quality = result['validation'].get('extraction_quality', {})

            logger.info(f"\n[Validation Results]")
            logger.info(f"  Status: {validation_status}")
            logger.info(f"  Numeric Coverage: {numeric_coverage:.2%}")
            logger.info(f"  Consistency Score: {consistency_score:.2%}")

            # Handle WARNING status separately
            if validation_status == 'WARNING':
                logger.warning(f"\n[WARNING] Document validation completed with warnings:")
                logger.warning(f"  Mode: {extraction_quality.get('extraction_mode', 'unknown')}")
                for reason in extraction_quality.get('warning_reasons', []):
                    logger.warning(f"  - {reason}")
                logger.warning(f"  This extraction will be included but may need review.")

            elif validation_status == 'FAIL':
                logger.error(
                    f"  [FAIL] Document failed validation. "
                    f"Consider reviewing extraction quality."
                )
                failed_docs.append({
                    'title': doc_title,
                    'index': i,
                    'reason': 'validation_failed',
                    'numeric_coverage': numeric_coverage,
                    'consistency_score': consistency_score
                })
                # Continue anyway (user can review later)

            # Build bipartite graph
            logger.info(f"\n[Building Graph]")
            graph_stats = await build_bipartite_graph_from_pipeline(
                result,
                rag.chunk_entity_relation_graph,
                rag.vdb_entities,
                rag.vdb_relations
            )

            logger.info(f"  Entity nodes: {graph_stats['entity_nodes']}")
            logger.info(f"  Relation nodes: {graph_stats['relation_nodes']}")
            logger.info(f"  Bipartite edges: {graph_stats['bipartite_edges']}")

            if graph_stats['orphan_relations'] > 0:
                logger.warning(
                    f"  [WARN] {graph_stats['orphan_relations']} orphan relations "
                    f"(relations with no linked entities)"
                )

            # Accumulate stats
            total_entities += graph_stats['entity_nodes']
            total_relations += graph_stats['relation_nodes']
            total_edges += graph_stats['bipartite_edges']

            # Store result
            all_results.append(result)

        except Exception as e:
            logger.error(f"[ERROR] Failed to process document {i+1}: {e}")
            failed_docs.append({
                'title': doc_title,
                'index': i,
                'reason': 'exception',
                'error': str(e)
            })
            # Continue with next document

    # Save graph to disk
    logger.info(f"\n{'='*80}")
    logger.info("Saving graph to disk...")
    logger.info(f"{'='*80}")
    await rag._insert_done()

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("Knowledge Graph Construction Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Total documents processed: {len(all_results)}/{len(markdown_documents)}")
    logger.info(f"Total entity nodes: {total_entities}")
    logger.info(f"Total relation nodes: {total_relations}")
    logger.info(f"Total bipartite edges: {total_edges}")

    if failed_docs:
        logger.warning(f"\n[WARN] {len(failed_docs)} documents failed:")
        for doc in failed_docs:
            reason = doc.get('reason', 'unknown')
            if reason == 'validation_failed':
                logger.warning(
                    f"  - {doc['title']}: Validation failed "
                    f"(coverage: {doc['numeric_coverage']:.2%}, "
                    f"consistency: {doc['consistency_score']:.2%})"
                )
            else:
                logger.warning(f"  - {doc['title']}: {doc.get('error', 'Unknown error')}")

    logger.info(f"\nGraph saved to: {working_dir}")
    logger.info(f"{'='*80}\n")

    return rag, all_results


async def process_single_document(
    markdown_text: str,
    metadata: Dict,
    api_key: str,
    rag_instance: Optional[BiGRAG] = None,
    working_dir: str = "./expr/educational_kg",
    validation_level: str = "STRICT"
) -> Tuple[BiGRAG, Dict]:
    """
    Process single document and add to knowledge graph.

    This is a convenience function for adding documents one at a time
    to an existing or new knowledge graph.

    Args:
        markdown_text: Document content in markdown format
        metadata: Document metadata (title, category, tags)
        api_key: OpenAI API key
        rag_instance: Existing BiGRAG instance (or create new one)
        working_dir: Directory to save graph (if creating new instance)
        validation_level: Validation strictness level

    Returns:
        (rag_instance, pipeline_result)

    Example:
        from bigrag.educational_pipeline import process_single_document

        # Process first document
        rag, result1 = await process_single_document(
            doc1_text,
            {'title': 'KUET Admission', 'category': 'university'},
            api_key="your-key"
        )

        # Add second document to same graph
        rag, result2 = await process_single_document(
            doc2_text,
            {'title': 'BUET Admission', 'category': 'university'},
            api_key="your-key",
            rag_instance=rag  # Reuse existing instance
        )

        # Query combined graph
        contexts = rag.query("কুয়েটে CSE বিভাগে কতটি আসন আছে?")
    """

    # Create new RAG instance if not provided
    if rag_instance is None:
        logger.info(f"Creating new BiGRAG instance at {working_dir}")
        rag_instance = BiGRAG(
            working_dir=working_dir,
            enable_llm_cache=True,
            addon_params={
                "language": "English",
                "entity_types": [
                    "department", "faculty", "university",
                    "department_code", "seat_count", "gpa_requirement",
                    "fee", "deadline", "time", "event", "location",
                    "person", "organization", "concept", "number"
                ]
            }
        )

    # Initialize pipeline
    pipeline = ProductionKGPipeline(
        api_key=api_key,
        validation_level=validation_level,
        extraction_mode="semi_structured"  # Default to semi_structured for flexibility
    )

    # Extract
    logger.info(f"Processing: {metadata.get('title', 'Untitled')}")
    result = await pipeline.process_document(markdown_text, metadata)

    # Build graph
    graph_stats = await build_bipartite_graph_from_pipeline(
        result,
        rag_instance.chunk_entity_relation_graph,
        rag_instance.vdb_entities,
        rag_instance.vdb_relations
    )

    logger.info(
        f"Added {graph_stats['entity_nodes']} entities, "
        f"{graph_stats['relation_nodes']} relations, "
        f"{graph_stats['bipartite_edges']} edges"
    )

    # Save
    await rag_instance._insert_done()

    return rag_instance, result


def sync_build_educational_kg(
    markdown_documents: List[str],
    document_metadata: List[Dict],
    api_key: str,
    working_dir: str = "./expr/educational_kg",
    validation_level: str = "STRICT"
) -> Tuple[BiGRAG, List[Dict]]:
    """
    Synchronous wrapper for build_educational_kg().

    Use this if you prefer synchronous code instead of async/await.

    Args:
        Same as build_educational_kg()

    Returns:
        Same as build_educational_kg()

    Example:
        from bigrag.educational_pipeline import sync_build_educational_kg

        # Synchronous usage (no asyncio.run needed)
        rag, results = sync_build_educational_kg(
            documents,
            metadata,
            api_key="your-key"
        )
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        build_educational_kg(
            markdown_documents,
            document_metadata,
            api_key,
            working_dir,
            validation_level
        )
    )
