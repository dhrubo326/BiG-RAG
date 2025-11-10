"""
Knowledge Graph Utility Functions

Helper functions for querying and manipulating the knowledge graph
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Get project root directory (3 levels up from api/services/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Load environment variables from root .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

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
    input_dir_base = os.getenv('INPUT_DIR', './datasets').lstrip('./')
    corpus_file = PROJECT_ROOT / input_dir_base / data_source / "raw" / "corpus.jsonl"

    if not corpus_file.exists():
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
    working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
    chunks_file = PROJECT_ROOT / working_dir_base / data_source / "kv_store_text_chunks.json"
    graph_file = PROJECT_ROOT / working_dir_base / data_source / "graph_chunk_entity_relation.graphml"

    stats = {
        "chunks": 0,
        "entities": 0,
        "edges": 0,
        "tokens": 0
    }

    # Count chunks - use correct field name "full_doc_id"
    doc_chunk_ids = set()  # Track chunk IDs for this document
    if chunks_file.exists():
        try:
            with open(chunks_file, encoding='utf-8') as f:
                chunks = json.load(f)

            # Filter by document ID using correct field name
            doc_chunks = [
                (c_id, c) for c_id, c in chunks.items()
                if c.get("full_doc_id") == document_id
            ]

            # Store chunk IDs for entity/relation lookup
            doc_chunk_ids = set(c_id for c_id, _ in doc_chunks)

            stats["chunks"] = len(doc_chunks)
            stats["tokens"] = sum(c.get("tokens", 0) for _, c in doc_chunks)
        except Exception as e:
            logger.error(f"Error reading chunks file: {e}")

    # Count entities and edges from GraphML file (Bug Fix: was looking for non-existent KV files)
    if graph_file.exists() and doc_chunk_ids:
        try:
            import networkx as nx
            G = nx.read_graphml(graph_file)

            doc_entities = []
            doc_edges = []

            # Iterate through all nodes in the graph
            for node, attrs in G.nodes(data=True):
                source_id_str = str(attrs.get("source_id", ""))
                role = attrs.get("role", "")

                # Split source_id by GRAPH_FIELD_SEP
                source_ids = source_id_str.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in source_id_str else [source_id_str]

                # Check if any source_id matches our document's chunks
                if any(sid in doc_chunk_ids for sid in source_ids):
                    if role == "entity":
                        doc_entities.append(node)
                    elif role == "relation":
                        doc_edges.append(node)

            stats["entities"] = len(doc_entities)
            stats["relations"] = len(doc_edges)
            stats["edges"] = len(doc_edges)  # Deprecated: alias for backward compatibility

            logger.debug(f"Document {document_id}: Found {stats['entities']} entities and {stats['relations']} relations")

        except Exception as e:
            logger.error(f"Error reading graph file: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
    working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
    graph_file = PROJECT_ROOT / working_dir_base / data_source / "graph_chunk_entity_relation.graphml"
    chunks_file = PROJECT_ROOT / working_dir_base / data_source / "kv_store_text_chunks.json"

    if not graph_file.exists():
        return []

    try:
        # First, get all chunk IDs for this document
        doc_chunk_ids = set()
        if chunks_file.exists():
            with open(chunks_file, encoding='utf-8') as f:
                chunks = json.load(f)
            doc_chunk_ids = set(
                c_id for c_id, c in chunks.items()
                if c.get("full_doc_id") == document_id
            )

        if not doc_chunk_ids:
            return []

        # Read entities from GraphML (Bug Fix: was looking for non-existent KV file)
        import networkx as nx
        G = nx.read_graphml(graph_file)

        doc_entities = []
        for node, attrs in G.nodes(data=True):
            if attrs.get("role") != "entity":
                continue

            source_id_str = str(attrs.get("source_id", ""))
            source_ids = source_id_str.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in source_id_str else [source_id_str]

            # Check if any source_id matches our document's chunks
            if any(sid in doc_chunk_ids for sid in source_ids):
                doc_entities.append({
                    "name": node,  # node name is the entity name
                    "type": attrs.get("entity_type", "UNKNOWN"),
                    "weight": float(attrs.get("weight", 0))
                })

        # Sort by weight
        doc_entities.sort(key=lambda x: x["weight"], reverse=True)

        return doc_entities[:top_k]
    except Exception as e:
        logger.error(f"Error getting document entities: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
    working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
    graph_file = PROJECT_ROOT / working_dir_base / data_source / "graph_chunk_entity_relation.graphml"
    chunks_file = PROJECT_ROOT / working_dir_base / data_source / "kv_store_text_chunks.json"

    if not graph_file.exists() or not chunks_file.exists():
        return []

    try:
        # Load chunks to build chunk_id -> document_id mapping
        with open(chunks_file, encoding='utf-8') as f:
            chunks = json.load(f)

        chunk_to_doc = {
            c_id: c.get("full_doc_id")
            for c_id, c in chunks.items()
            if c.get("full_doc_id")
        }

        # Get chunk IDs for this document
        doc_chunk_ids = set(
            c_id for c_id, c in chunks.items()
            if c.get("full_doc_id") == document_id
        )

        if not doc_chunk_ids:
            return []

        # Load entities from GraphML (Bug Fix: was looking for non-existent KV file)
        import networkx as nx
        G = nx.read_graphml(graph_file)

        # Get this document's entities
        doc_entities = set()
        for node, attrs in G.nodes(data=True):
            if attrs.get("role") != "entity":
                continue

            source_id_str = str(attrs.get("source_id", ""))
            source_ids = source_id_str.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in source_id_str else [source_id_str]

            # Check if any source_id (chunk) belongs to this document
            if any(sid in doc_chunk_ids for sid in source_ids):
                doc_entities.add(node)  # node name is the entity name

        if not doc_entities:
            return []

        # Find other documents with overlapping entities
        doc_scores = {}

        for node, attrs in G.nodes(data=True):
            if attrs.get("role") != "entity":
                continue

            entity_name = node

            if entity_name not in doc_entities:
                continue

            # Get all chunk IDs mentioning this entity
            source_id_str = str(attrs.get("source_id", ""))
            source_ids = source_id_str.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in source_id_str else [source_id_str]

            for chunk_id in source_ids:
                if not chunk_id or chunk_id in doc_chunk_ids:
                    continue  # Skip our own document's chunks

                # Map chunk_id to document_id
                related_doc_id = chunk_to_doc.get(chunk_id)
                if not related_doc_id:
                    continue

                if related_doc_id not in doc_scores:
                    doc_scores[related_doc_id] = 0

                doc_scores[related_doc_id] += 1

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
        import traceback
        logger.error(traceback.format_exc())
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
        from .registry import registry
        doc = await registry.get_document(document_id, dataset=data_source)

        if doc:
            return doc.get("title", document_id)
    except Exception as e:
        logger.warning(f"Error accessing registry: {e}")

    # Fallback to corpus
    input_dir_base = os.getenv('INPUT_DIR', './datasets').lstrip('./')
    corpus_file = PROJECT_ROOT / input_dir_base / data_source / "raw" / "corpus.jsonl"

    if corpus_file.exists():
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
    input_dir_base = os.getenv('INPUT_DIR', './datasets').lstrip('./')
    corpus_file = PROJECT_ROOT / input_dir_base / data_source / "raw" / "corpus.jsonl"
    temp_file = corpus_file.with_suffix('.jsonl.tmp')

    if not corpus_file.exists():
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
        temp_file.replace(corpus_file)
        logger.info(f"Removed document {document_id} from corpus")
    except Exception as e:
        logger.error(f"Error removing from corpus: {e}")
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
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
    from .jobs import ProcessingJob, JobStatus
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
        working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
        working_dir = PROJECT_ROOT / working_dir_base / dataset

        for file in [
            # KV Storage files (JSON-based metadata)
            "kv_store_text_chunks.json",
            "kv_store_full_docs.json",
            "kv_store_llm_response_cache.json",

            # Vector DB files (embeddings for similarity search)
            "vdb_entities.json",
            "vdb_relations.json",
            "vdb_chunks.json",

            # Graph structure file (NetworkX GraphML)
            "graph_chunk_entity_relation.graphml",

            # Legacy FAISS index files (kept for backward compatibility)
            "index_entity.bin",
            "index_relation.bin",
            "index.bin",
            "corpus.npy",
            "corpus_entity.npy",
            "corpus_relation.npy"
        ]:
            path = working_dir / file
            if path.exists():
                path.unlink()
                logger.info(f"Removed {file}")

        # Reload all documents from corpus
        input_dir_base = os.getenv('INPUT_DIR', './datasets').lstrip('./')
        corpus_file = PROJECT_ROOT / input_dir_base / dataset / "raw" / "corpus.jsonl"

        if not corpus_file.exists():
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


# ==============================================================================
# Document Management Helper Functions
# ==============================================================================

def compute_doc_id(content: str, prefix: str = "doc") -> str:
    """
    Generate unique ID from content hash

    Uses BiGRAG's compute_mdhash_id to ensure document IDs match
    between upload system and knowledge graph.
    """
    from bigrag.utils import compute_mdhash_id
    return compute_mdhash_id(content.strip(), prefix=f"{prefix}-")


async def add_document_to_corpus(
    data_source: str,
    doc_id: str,
    content: str,
    title: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Add a document to the corpus.jsonl file"""
    from pathlib import Path
    import os
    import json
    from datetime import datetime
    from bigrag.utils import logger
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    
    # Use configurable input directory from environment
    input_dir_base = os.getenv('INPUT_DIR', './datasets').lstrip('./')
    corpus_file = PROJECT_ROOT / input_dir_base / data_source / "raw" / "corpus.jsonl"

    # Create directory if doesn't exist
    corpus_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare document
    doc = {
        "id": doc_id,
        "contents": content,
        "title": title,
        "upload_date": datetime.now().isoformat(),
        "source": "upload"
    }

    # Add metadata if provided
    if metadata:
        doc["metadata"] = metadata

    # Append to corpus
    with open(corpus_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    logger.info(f"Added document {doc_id} to corpus")
    return doc
