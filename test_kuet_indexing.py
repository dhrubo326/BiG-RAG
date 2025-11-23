"""
KUET Admission Document Indexing Test

This script tests the complete production KG building pipeline:
1. Loads KUET_Admission_info.md
2. Processes through all 5 phases
3. Builds BiG-RAG knowledge graph
4. Saves to demo_test dataset
5. Reports detailed statistics

Usage:
    python test_kuet_indexing.py
"""

import asyncio
import os
import json
from pathlib import Path
from bigrag import BiGRAG
from bigrag.production_pipeline import ProductionKGPipeline
from bigrag.builders.bipartite_graph_builder import build_bipartite_graph_from_pipeline


async def test_kuet_indexing():
    """Test complete KUET admission document indexing."""

    print("\n" + "="*80)
    print("KUET ADMISSION DOCUMENT - PRODUCTION KG INDEXING TEST")
    print("="*80)

    # Step 1: Load API key
    print("\n[Step 1] Loading OpenAI API key...")
    api_key_file = "openai_api_key.txt"
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        print(f"  [OK] API key loaded from {api_key_file}")
    else:
        print(f"  [ERROR] {api_key_file} not found!")
        return False

    # Step 2: Load KUET document
    print("\n[Step 2] Loading KUET admission document...")
    kuet_file = "KUET_Admission_info.md"
    if not os.path.exists(kuet_file):
        print(f"  [ERROR] {kuet_file} not found!")
        return False

    with open(kuet_file, 'r', encoding='utf-8') as f:
        kuet_doc = f.read()

    print(f"  [OK] Document loaded")
    print(f"  Length: {len(kuet_doc)} characters")
    print(f"  Lines: {len(kuet_doc.splitlines())}")

    # Step 3: Prepare metadata
    print("\n[Step 3] Preparing document metadata...")
    metadata = {
        'title': 'KUET Admission 2024-25',
        'category': 'university_admission',
        'tags': ['engineering', 'admission', 'KUET', 'Bangladesh'],
        'year': '2024-2025',
        'university': 'Khulna University of Engineering and Technology'
    }
    print(f"  [OK] Metadata: {metadata}")

    # Step 4: Initialize Production Pipeline
    print("\n[Step 4] Initializing Production KG Pipeline...")
    print("  Configuration:")
    print("    - Model: gpt-4o (advanced LLM for best results)")
    print("    - Validation: MODERATE (95%+ accuracy)")
    print("    - Extraction mode: semi_structured (default)")
    print("    - Entity linking: Enabled")

    pipeline = ProductionKGPipeline(
        api_key=api_key,
        model="gpt-4o",  # Using advanced LLM as requested
        validation_level="MODERATE",
        extraction_mode="semi_structured",
        enable_entity_linking=True
    )
    print("  [OK] Pipeline initialized")

    # Step 5: Process document through all 5 phases
    print("\n[Step 5] Processing document through production pipeline...")
    print("  This will take several minutes (LLM calls + validation)...")
    print("  " + "-"*76)

    try:
        result = await pipeline.process_document(
            markdown_text=kuet_doc,
            metadata=metadata,
            language="English"  # Output in English for consistency
        )
        print("\n  [OK] Document processing complete!")

    except Exception as e:
        print(f"\n  [ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Analyze results
    print("\n[Step 6] Analyzing extraction results...")
    print("  " + "-"*76)

    entities = result['entities']
    relations = result['relations']
    chunks = result['chunks']
    validation = result['validation']
    stats = result['statistics']

    print(f"\n  Extraction Statistics:")
    print(f"    Total chunks: {stats['total_chunks']}")
    print(f"    - Table chunks: {stats['table_chunks']}")
    print(f"    - Paragraph chunks: {stats['paragraph_chunks']}")
    print(f"    Total entities extracted: {len(entities)}")
    print(f"    Total relations extracted: {len(relations)}")
    if pipeline.enable_entity_linking:
        print(f"    Entity merge reduction: {stats['entity_merge_reduction']} duplicates removed")

    print(f"\n  Validation Results:")
    print(f"    Overall status: {validation['overall_status']}")
    print(f"    Numeric coverage: {validation['numeric']['numeric_coverage']:.2%}")
    print(f"    Hallucination rate: {validation['numeric']['hallucination_rate']:.2%}")
    print(f"    Consistency score: {validation['consistency']['consistency_score']:.2%}")

    extraction_quality = validation.get('extraction_quality', {})
    print(f"\n  Extraction Quality:")
    print(f"    Mode: {extraction_quality.get('extraction_mode', 'N/A')}")
    print(f"    Numeric status: {extraction_quality.get('numeric_status', 'N/A')}")
    print(f"    Consistency status: {extraction_quality.get('consistency_status', 'N/A')}")

    if extraction_quality.get('warning_reasons'):
        print(f"\n  [WARNING] Extraction warnings:")
        for reason in extraction_quality['warning_reasons']:
            print(f"    - {reason}")

    # Step 7: Save detailed results to JSON (avoid Unicode print issues)
    print(f"\n  Saving detailed results to JSON...")
    result_file = "kuet_indexing_results.json"

    # Prepare summary
    summary = {
        'document': 'KUET_Admission_info.md',
        'status': validation['overall_status'],
        'statistics': {
            'entities': len(entities),
            'relations': len(relations),
            'chunks': len(chunks),
            'numeric_coverage': validation['numeric']['numeric_coverage'],
            'hallucination_rate': validation['numeric']['hallucination_rate'],
            'consistency_score': validation['consistency']['consistency_score']
        },
        'sample_entities': entities[:10],
        'sample_relations': relations[:5],
        'validation_details': validation
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"    [OK] Results saved to {result_file}")
    print(f"    Review this file to see entity names and descriptions")

    # Step 9: Check validation status
    print("\n[Step 7] Checking validation status...")
    overall_status = validation['overall_status']

    if overall_status == 'PASS':
        print("  [OK] Validation PASSED - Data is production ready!")
    elif overall_status == 'WARNING':
        print("  [WARNING] Validation completed with warnings - Data is usable but review recommended")
    else:
        print("  [FAIL] Validation FAILED - Data quality below threshold")
        print(f"  Numeric coverage: {validation['numeric']['numeric_coverage']:.2%} (need 95%+)")
        print(f"  Consistency: {validation['consistency']['consistency_score']:.2%} (need 95%+)")

        # Show what failed
        if validation['numeric']['missing_numbers']:
            print(f"\n  Missing numbers: {validation['numeric']['missing_numbers'][:5]}")
        if validation['numeric']['hallucinated_numbers']:
            print(f"  Hallucinated numbers: {validation['numeric']['hallucinated_numbers'][:5]}")

    # Step 10: Initialize BiGRAG and build graph
    print("\n[Step 8] Building BiG-RAG knowledge graph...")
    print("  Working directory: ./expr/kuet_test")

    working_dir = "./expr/kuet_test"
    os.makedirs(working_dir, exist_ok=True)

    # Initialize BiGRAG
    rag = BiGRAG(
        working_dir=working_dir
        # Uses default embedding_func (openai_embedding)
    )

    print("  [OK] BiGRAG instance created")

    # Build bipartite graph from pipeline results
    print("  Building bipartite graph structure...")

    try:
        graph_stats = await build_bipartite_graph_from_pipeline(
            pipeline_result=result,
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            vdb_entities=rag.vdb_entities,
            vdb_relations=rag.vdb_relations
        )

        print("\n  [OK] Graph construction complete!")

        # MISSING STEP 1: Store chunks to KV storage
        print("  Storing chunks to KV storage...")
        from bigrag.utils import compute_mdhash_id

        bigrag_chunks = {}
        production_chunk_to_bigrag_id = {}

        for prod_chunk in chunks:
            # Create BiGRAG chunk ID (hash of content)
            chunk_id = compute_mdhash_id(prod_chunk['content'], prefix='chunk-')

            bigrag_chunks[chunk_id] = {
                "content": prod_chunk['content'],
                "tokens": prod_chunk.get('tokens', []),
                "chunk_order_index": prod_chunk.get('chunk_order_index', 0),
                "full_doc_id": "doc-kuet-admission",
                "doc_title": metadata.get("title", ""),
                "doc_metadata": metadata,
            }

            # Map ProductionPipeline chunk ID → BiGRAG chunk ID
            prod_chunk_id = prod_chunk.get('chunk_id') or prod_chunk.get('source_id')
            if prod_chunk_id:
                production_chunk_to_bigrag_id[prod_chunk_id] = chunk_id

        await rag.text_chunks.upsert(bigrag_chunks)
        print(f"    [OK] Stored {len(bigrag_chunks)} chunks to KV storage")

        # MISSING STEP 2: Store full document
        print("  Storing full document...")
        doc_id = "doc-kuet-admission"
        await rag.full_docs.upsert({
            doc_id: {
                "content": kuet_doc,
                "title": metadata.get("title", ""),
                "metadata": metadata
            }
        })
        print(f"    [OK] Stored full document")

        # MISSING STEP 3: Index chunks to vdb_chunks (Path C retrieval)
        print("  Indexing chunks to vector DB...")
        chunks_for_vdb = {
            chunk_id: {
                "content": f"[{metadata['title']}] {chunk_data['content']}",
                "full_doc_id": doc_id
            }
            for chunk_id, chunk_data in bigrag_chunks.items()
        }
        await rag.vdb_chunks.upsert(chunks_for_vdb)
        print(f"    [OK] Indexed {len(chunks_for_vdb)} chunks to vector DB")

        # Update graph stats with chunk count
        graph_stats['chunk_nodes'] = len(bigrag_chunks)

        print(f"\n  Graph Statistics:")
        print(f"    Entity nodes created: {graph_stats.get('entity_nodes', 0)}")
        print(f"    Relation nodes created: {graph_stats.get('relation_nodes', 0)}")
        print(f"    Chunk nodes created: {graph_stats.get('chunk_nodes', 0)}")
        print(f"    Bipartite edges created: {graph_stats.get('bipartite_edges', 0)}")

    except Exception as e:
        print(f"\n  [ERROR] Graph building failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 11: Save graph to disk
    print("\n[Step 9] Saving knowledge graph to disk...")

    try:
        # CRITICAL: Call index_done_callback() to persist data
        # The graph builder only loads data in memory - it doesn't auto-save!
        print("  Calling index_done_callback() on all storage instances...")

        await rag.chunk_entity_relation_graph.index_done_callback()
        print("  [OK] Graph saved to disk")

        await rag.vdb_entities.index_done_callback()
        print("  [OK] Entity vector DB saved")

        await rag.vdb_relations.index_done_callback()
        print("  [OK] Relation vector DB saved")

        await rag.vdb_chunks.index_done_callback()
        print("  [OK] Chunk vector DB saved")

        await rag.full_docs.index_done_callback()
        print("  [OK] Full docs KV storage saved")

        await rag.text_chunks.index_done_callback()
        print("  [OK] Text chunks KV storage saved")

        # List saved files
        saved_files = list(Path(working_dir).glob("*.json")) + list(Path(working_dir).glob("*.graphml"))
        print(f"\n  Saved files ({len(saved_files)}):")
        for file in sorted(saved_files):
            size = file.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
            print(f"    - {file.name} ({size_str})")

    except Exception as e:
        print(f"\n  [ERROR] Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 12: Test query (save results to avoid Unicode print issues)
    print("\n[Step 10] Testing query functionality...")

    from bigrag.base import QueryParam

    test_queries = [
        "How many seats in CSE department?",
        "What are the admission requirements?",
        "When is the admission test?"
    ]

    query_results = []
    for query in test_queries:
        print(f"\n  Query: {query}")
        try:
            contexts = await rag.aquery(
                query=query,
                param=QueryParam(mode="hybrid", top_k=3)
            )
            print(f"  Retrieved {len(contexts)} contexts (saved to JSON)")
            query_results.append({
                'query': query,
                'contexts': contexts
            })
        except Exception as e:
            print(f"  [ERROR] Query failed: {e}")

    # Save query results
    with open("kuet_query_results.json", 'w', encoding='utf-8') as f:
        json.dump(query_results, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] Query results saved to kuet_query_results.json")

    # Step 13: Final summary
    print("\n" + "="*80)
    print("INDEXING TEST SUMMARY")
    print("="*80)

    print(f"\nDocument: KUET_Admission_info.md")
    print(f"Status: {validation['overall_status']}")
    print(f"\nData Quality:")
    print(f"  Entities: {len(entities)}")
    print(f"  Relations: {len(relations)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Numeric coverage: {validation['numeric']['numeric_coverage']:.2%}")
    print(f"  Consistency: {validation['consistency']['consistency_score']:.2%}")

    print(f"\nGraph Structure:")
    print(f"  Entity nodes: {graph_stats.get('entity_nodes', 0)}")
    print(f"  Relation nodes: {graph_stats.get('relation_nodes', 0)}")
    print(f"  Chunk nodes: {graph_stats.get('chunk_nodes', 0)}")
    print(f"  Bipartite edges: {graph_stats.get('bipartite_edges', 0)}")

    print(f"\nSaved to: {working_dir}")

    if validation['overall_status'] in ['PASS', 'WARNING']:
        print("\n[SUCCESS] Knowledge graph built successfully!")
        print("You can now start the server with: python backend/server.py")
        return True
    else:
        print("\n[WARNING] Knowledge graph built but validation failed")
        print("Review the validation errors above before using in production")
        return False


async def main():
    """Main entry point."""

    try:
        success = await test_kuet_indexing()
        return 0 if success else 1

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Unexpected failure: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
