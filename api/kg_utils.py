"""
Knowledge Graph Utility Functions

Helper functions for querying and manipulating the knowledge graph
"""

import json
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import GRAPH_FIELD_SEP from bigrag.prompt
try:
    from bigrag.prompt import GRAPH_FIELD_SEP
except ImportError:
    GRAPH_FIELD_SEP = "<SEP>"


async def get_document_content_from_corpus(
    data_source: str,
    document_id: str
) -> Optional[str]:
    """
    Retrieve document content from corpus.jsonl

    Args:
        data_source: Dataset name
        document_id: Document ID

    Returns:
        Document content or None if not found
    """
    corpus_file = f"datasets/{data_source}/raw/corpus.jsonl"

    if not os.path.exists(corpus_file):
        logger.warning(f"Corpus file not found: {corpus_file}")
        return None

    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                if doc.get("id") == document_id:
                    return doc.get("contents", "")
    except Exception as e:
        logger.error(f"Error reading corpus file: {e}")

    return None


async def get_document_stats_from_kg(
    data_source: str,
    document_id: str
) -> Dict:
    """
    Get statistics about a document in the knowledge graph

    Args:
        data_source: Dataset name
        document_id: Document ID

    Returns:
        {chunks, entities, edges, tokens}
    """
    chunks_file = f"expr/{data_source}/kv_store_text_chunks.json"
    entities_file = f"expr/{data_source}/kv_store_entities.json"
    edges_file = f"expr/{data_source}/kv_store_bipartite_edges.json"

    stats = {
        "chunks": 0,
        "entities": 0,
        "edges": 0,
        "tokens": 0
    }

    # Count chunks
    if os.path.exists(chunks_file):
        try:
            with open(chunks_file, encoding='utf-8') as f:
                chunks = json.load(f)

            # Filter by document ID
            doc_chunks = [
                c for c_id, c in chunks.items()
                if document_id in c_id or c.get("doc_id") == document_id
            ]
            stats["chunks"] = len(doc_chunks)
            stats["tokens"] = sum(c.get("tokens", 0) for c in doc_chunks)
        except Exception as e:
            logger.error(f"Error reading chunks file: {e}")

    # Count entities (by source_id)
    if os.path.exists(entities_file):
        try:
            with open(entities_file, encoding='utf-8') as f:
                entities = json.load(f)

            doc_entities = [
                e for e_id, e in entities.items()
                if document_id in str(e.get("source_id", ""))
            ]
            stats["entities"] = len(doc_entities)
        except Exception as e:
            logger.error(f"Error reading entities file: {e}")

    # Count edges (by source_id)
    if os.path.exists(edges_file):
        try:
            with open(edges_file, encoding='utf-8') as f:
                edges = json.load(f)

            doc_edges = [
                edge for edge_id, edge in edges.items()
                if document_id in str(edge.get("source_id", ""))
            ]
            stats["edges"] = len(doc_edges)
        except Exception as e:
            logger.error(f"Error reading edges file: {e}")

    return stats


async def get_document_entities(
    data_source: str,
    document_id: str,
    top_k: int = 10
) -> List[Dict]:
    """
    Get top entities for a document

    Args:
        data_source: Dataset name
        document_id: Document ID
        top_k: Number of top entities to return

    Returns:
        List of {name, type, weight} dicts
    """
    entities_file = f"expr/{data_source}/kv_store_entities.json"

    if not os.path.exists(entities_file):
        return []

    try:
        with open(entities_file, encoding='utf-8') as f:
            entities = json.load(f)

        # Filter by document
        doc_entities = [
            {
                "name": e.get("entity_name"),
                "type": e.get("entity_type"),
                "weight": e.get("weight", 0)
            }
            for e_id, e in entities.items()
            if document_id in str(e.get("source_id", ""))
        ]

        # Sort by weight
        doc_entities.sort(key=lambda x: x["weight"], reverse=True)

        return doc_entities[:top_k]
    except Exception as e:
        logger.error(f"Error getting document entities: {e}")
        return []


async def find_related_documents(
    data_source: str,
    document_id: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Find documents related by entity overlap

    Args:
        data_source: Dataset name
        document_id: Document ID
        top_k: Number of related documents to return

    Returns:
        List of {id, title, similarity} dicts
    """
    entities_file = f"expr/{data_source}/kv_store_entities.json"

    if not os.path.exists(entities_file):
        return []

    try:
        with open(entities_file, encoding='utf-8') as f:
            entities = json.load(f)

        # Get this document's entities
        doc_entities = set([
            e.get("entity_name")
            for e_id, e in entities.items()
            if document_id in str(e.get("source_id", ""))
        ])

        if not doc_entities:
            return []

        # Find other documents with overlapping entities
        doc_scores = {}

        for e_id, e in entities.items():
            entity_name = e.get("entity_name")

            if entity_name not in doc_entities:
                continue

            # Get all documents mentioning this entity
            source_id_str = str(e.get("source_id", ""))
            source_ids = source_id_str.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in source_id_str else [source_id_str]

            for source_id in source_ids:
                if not source_id or source_id == document_id:
                    continue

                if source_id not in doc_scores:
                    doc_scores[source_id] = 0

                doc_scores[source_id] += 1

        # Normalize scores
        max_score = max(doc_scores.values()) if doc_scores else 1

        related = [
            {
                "id": doc_id,
                "title": await get_document_title(data_source, doc_id),
                "similarity": score / max_score
            }
            for doc_id, score in doc_scores.items()
        ]

        # Sort by similarity
        related.sort(key=lambda x: x["similarity"], reverse=True)

        return related[:top_k]
    except Exception as e:
        logger.error(f"Error finding related documents: {e}")
        return []


async def get_document_title(data_source: str, document_id: str) -> str:
    """
    Get document title from registry or corpus

    Args:
        data_source: Dataset name
        document_id: Document ID

    Returns:
        Document title or document_id if not found
    """
    # Try registry first
    try:
        from api.registry import registry
        doc = await registry.get_document(document_id, dataset=data_source)

        if doc:
            return doc.get("title", document_id)
    except Exception as e:
        logger.warning(f"Error accessing registry: {e}")

    # Fallback to corpus
    corpus_file = f"datasets/{data_source}/raw/corpus.jsonl"

    if os.path.exists(corpus_file):
        try:
            with open(corpus_file, encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    if doc.get("id") == document_id:
                        return doc.get("title", document_id)
        except Exception as e:
            logger.error(f"Error reading corpus: {e}")

    return document_id


async def remove_from_corpus(data_source: str, document_id: str):
    """
    Remove document from corpus.jsonl (hard delete)

    Rewrites corpus without the specified document

    Args:
        data_source: Dataset name
        document_id: Document ID to remove
    """
    corpus_file = f"datasets/{data_source}/raw/corpus.jsonl"
    temp_file = f"{corpus_file}.tmp"

    if not os.path.exists(corpus_file):
        logger.warning(f"Corpus file not found: {corpus_file}")
        return

    try:
        # Rewrite corpus without this document
        with open(corpus_file, 'r', encoding='utf-8') as f_in:
            with open(temp_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    doc = json.loads(line)
                    if doc.get("id") != document_id:
                        f_out.write(line)

        # Replace original
        os.replace(temp_file, corpus_file)
        logger.info(f"Removed document {document_id} from corpus")
    except Exception as e:
        logger.error(f"Error removing from corpus: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


async def rebuild_entire_graph(dataset: str, job_id: str, rag_instance, processing_jobs_dict):
    """
    Rebuild entire knowledge graph from corpus (for hard deletes)

    WARNING: This clears existing graph and rebuilds from scratch

    Args:
        dataset: Dataset name
        job_id: Job ID for tracking
        rag_instance: BiGRAG instance
        processing_jobs_dict: Reference to processing_jobs dict
    """
    from api.jobs import ProcessingJob, JobStatus
    from datetime import datetime

    job = ProcessingJob(
        job_id=job_id,
        document_id="rebuild",
        dataset=dataset,
        status=JobStatus.PROCESSING
    )
    processing_jobs_dict[job_id] = job

    try:
        # Clear existing graph files
        working_dir = f"expr/{dataset}"

        for file in [
            "kv_store_entities.json",
            "kv_store_bipartite_edges.json",
            "kv_store_text_chunks.json",
            "kv_store_full_docs.json",
            "index_entity.bin",
            "index_bipartite_edge.bin",
            "index.bin",
            "corpus.npy",
            "corpus_entity.npy",
            "corpus_bipartite_edge.npy"
        ]:
            path = os.path.join(working_dir, file)
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed {file}")

        # Reload all documents from corpus
        corpus_file = f"datasets/{dataset}/raw/corpus.jsonl"

        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        documents = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents.append({
                    "content": doc.get("contents", ""),
                    "title": doc.get("title", "")
                })

        logger.info(f"Rebuilding graph with {len(documents)} documents")

        # Rebuild graph in batches
        batch_size = 10

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            await rag_instance.ainsert(batch)

            progress = (i + len(batch)) / len(documents)
            job.update(progress=progress)
            logger.info(f"Rebuild progress: {progress*100:.1f}%")

        job.update(
            status=JobStatus.COMPLETED,
            progress=1.0,
            completed_at=datetime.now()
        )
        logger.info(f"Graph rebuild completed successfully")

    except Exception as e:
        logger.error(f"Graph rebuild failed: {e}")
        job.update(
            status=JobStatus.FAILED,
            error=str(e),
            completed_at=datetime.now()
        )
        raise
