"""
Build Knowledge Graph from Corpus

This script builds a BiG-RAG knowledge graph from a corpus.jsonl file.
It handles the complete pipeline:
1. Load corpus from JSONL
2. Chunk documents
3. Extract entities and relations (using GPT-4o-mini)
4. Build bipartite graph structure
5. Create vector embeddings
6. Save to disk (graph + vector storage)

Usage:
    python build_kg_from_corpus.py --data-source my_data
    python build_kg_from_corpus.py --data-source my_data --batch-size 10

Prerequisites:
    - OpenAI API key in openai_api_key.txt
    - Corpus file at datasets/{data_source}/raw/corpus.jsonl
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

from bigrag import BiGRAG
from bigrag.llm import gpt_4o_mini_complete, openai_embedding
from bigrag.utils import logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_kg.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Windows console UTF-8 support
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_api_key():
    """Load OpenAI API key from file"""
    api_key_file = Path("openai_api_key.txt")

    if api_key_file.exists():
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info("✓ Loaded OpenAI API key")
        return api_key
    else:
        logger.error("✗ openai_api_key.txt not found!")
        logger.error("  Create this file with your OpenAI API key")
        sys.exit(1)


def load_corpus(data_source: str):
    """Load corpus from JSONL file"""
    corpus_path = Path(f"datasets/{data_source}/raw/corpus.jsonl")

    if not corpus_path.exists():
        logger.error(f"✗ Corpus not found: {corpus_path}")
        logger.error("  Create corpus using: python convert_text_to_corpus.py")
        sys.exit(1)

    documents = []
    with open(corpus_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                documents.append({
                    "id": data.get("id", f"doc-{line_num}"),
                    "content": data.get("contents", ""),
                    "title": data.get("title", f"Document {line_num}")
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue

    logger.info(f"✓ Loaded {len(documents)} documents from {corpus_path}")
    return documents


def build_knowledge_graph(rag, documents, batch_size=5):
    """
    Build knowledge graph from documents

    This function:
    1. Chunks documents into smaller pieces (1200 tokens, 100 overlap)
    2. Extracts entities and relations using GPT-4o-mini
    3. Builds bipartite graph (entities ↔ relations)
    4. Creates vector embeddings for all components
    5. Saves everything to disk

    Args:
        rag: BiGRAG instance
        documents: List of document dicts
        batch_size: Number of documents to process per batch
    """
    logger.info("="*80)
    logger.info("BUILDING KNOWLEDGE GRAPH")
    logger.info("="*80)
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("")
    logger.info("This process:")
    logger.info("  1. Chunks documents (1200 tokens, 100 overlap)")
    logger.info("  2. Extracts entities using GPT-4o-mini")
    logger.info("  3. Extracts n-ary relations")
    logger.info("  4. Builds bipartite graph structure")
    logger.info("  5. Creates vector embeddings")
    logger.info("  6. Saves to disk (KV storage + vector DB + graph)")
    logger.info("")
    logger.info("Estimated time: 2-5 minutes per 10 documents")
    logger.info("")

    # Extract content
    contents = [doc["content"] for doc in documents]
    total_batches = (len(contents) + batch_size - 1) // batch_size

    logger.info(f"Processing in {total_batches} batches...")
    logger.info("")

    # Process in batches
    for i in range(0, len(contents), batch_size):
        batch_num = i // batch_size + 1
        batch = contents[i:i+batch_size]

        logger.info(f"[Batch {batch_num}/{total_batches}] Processing documents {i+1} to {i+len(batch)}...")

        retries = 0
        max_retries = 3

        while retries < max_retries:
            try:
                # Insert batch (handles chunking, extraction, graph building, embedding)
                rag.insert(batch)
                logger.info(f"[Batch {batch_num}/{total_batches}] ✓ Success")
                break

            except Exception as e:
                retries += 1
                logger.warning(f"[Batch {batch_num}/{total_batches}] ✗ Error (attempt {retries}/{max_retries}): {e}")

                if retries < max_retries:
                    wait_time = 5 * retries
                    logger.info(f"[Batch {batch_num}/{total_batches}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[Batch {batch_num}/{total_batches}] ✗ Failed after {max_retries} attempts")
                    raise

    logger.info("")
    logger.info("✓ Knowledge graph construction complete!")
    logger.info("")


def verify_output(working_dir: str):
    """Verify that all required files were created"""
    logger.info("="*80)
    logger.info("VERIFYING OUTPUT")
    logger.info("="*80)

    output_dir = Path(working_dir)

    # Expected files
    expected_files = [
        "kv_store_text_chunks.json",        # Text chunks
        "vdb_entities.json",                # Entity vectors
        "vdb_bipartite_edges.json",         # Relation vectors
        "graph_chunk_entity_relation.graphml",  # Graph structure
    ]

    logger.info("Checking for output files:")
    logger.info("")

    all_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            logger.info(f"  ✓ {filename} ({size:,} bytes)")
        else:
            logger.error(f"  ✗ {filename} - NOT FOUND")
            all_exist = False

    logger.info("")

    if all_exist:
        # Load and display statistics
        try:
            with open(output_dir / "kv_store_text_chunks.json", 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            with open(output_dir / "vdb_entities.json", 'r', encoding='utf-8') as f:
                entities_vdb = json.load(f)
            with open(output_dir / "vdb_bipartite_edges.json", 'r', encoding='utf-8') as f:
                edges_vdb = json.load(f)

            logger.info("="*80)
            logger.info("KNOWLEDGE GRAPH STATISTICS")
            logger.info("="*80)
            logger.info(f"  Text Chunks:      {len(chunks)}")

            # NanoVectorDB stores data in 'data' key
            entities_count = len(entities_vdb.get('data', []))
            edges_count = len(edges_vdb.get('data', []))

            logger.info(f"  Entities:         {entities_count}")
            logger.info(f"  Relations:        {edges_count}")
            logger.info("="*80)
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"Error reading output files: {e}")
            return False
    else:
        logger.error("✗ Some output files are missing")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build BiG-RAG knowledge graph from corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from datasets/my_data/raw/corpus.jsonl
  python build_kg_from_corpus.py --data-source my_data

  # Custom batch size (larger = faster but more API requests)
  python build_kg_from_corpus.py --data-source my_data --batch-size 10

  # Use different embedding model
  python build_kg_from_corpus.py --data-source my_data --embedding-model text-embedding-3-small

Output location:
  expr/{data_source}/
    ├── kv_store_text_chunks.json       # Text chunks
    ├── vdb_entities.json               # Entity embeddings
    ├── vdb_bipartite_edges.json        # Relation embeddings
    └── graph_chunk_entity_relation.graphml  # Graph structure
        """
    )

    parser.add_argument('--data-source', required=True,
                       help='Dataset name (folder in datasets/)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Documents per batch (default: 5)')
    parser.add_argument('--chunk-size', type=int, default=1200,
                       help='Token size per chunk (default: 1200)')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                       help='Overlap between chunks (default: 100)')
    parser.add_argument('--embedding-model', default='text-embedding-3-large',
                       help='OpenAI embedding model (default: text-embedding-3-large)')

    args = parser.parse_args()

    print("="*80)
    print("BiG-RAG Knowledge Graph Builder")
    print("="*80)
    print()

    data_source = args.data_source
    working_dir = f"expr/{data_source}"

    # Step 1: Load API key
    logger.info("Step 1: Loading OpenAI API key...")
    load_api_key()
    print()

    # Step 2: Load corpus
    logger.info("Step 2: Loading corpus...")
    documents = load_corpus(data_source)
    print()

    # Step 3: Initialize BiGRAG
    logger.info("Step 3: Initializing BiG-RAG...")
    logger.info(f"  Working directory: {working_dir}")
    logger.info(f"  LLM: gpt-4o-mini (entity extraction)")
    logger.info(f"  Embedding: {args.embedding_model}")
    logger.info(f"  Chunk size: {args.chunk_size} tokens")
    logger.info(f"  Chunk overlap: {args.chunk_overlap} tokens")
    print()

    rag = BiGRAG(
        working_dir=working_dir,
        # Models
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding,
        # Chunking
        chunk_token_size=args.chunk_size,
        chunk_overlap_token_size=args.chunk_overlap,
        # Entity extraction
        entity_extract_max_gleaning=1,  # 1 pass (faster)
        entity_summary_to_max_tokens=500,
        # Enable caching
        enable_llm_cache=True,
    )

    logger.info("✓ BiG-RAG initialized")
    print()

    # Step 4: Build knowledge graph
    try:
        build_knowledge_graph(rag, documents, batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"✗ Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Step 5: Verify output
    success = verify_output(working_dir)

    if success:
        logger.info("="*80)
        logger.info("BUILD SUCCESSFUL!")
        logger.info("="*80)
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Start API server:")
        logger.info(f"     python script_api.py --data_source {data_source}")
        logger.info("")
        logger.info(f"  2. Test queries:")
        logger.info(f"     curl -X POST http://localhost:8001/search \\")
        logger.info(f"       -H 'Content-Type: application/json' \\")
        logger.info(f"       -d '{{\"queries\": [\"your question here\"]}}'")
        logger.info("")
        logger.info(f"  3. Or use in Python:")
        logger.info(f"     from bigrag import BiGRAG, QueryParam")
        logger.info(f"     rag = BiGRAG(working_dir='expr/{data_source}')")
        logger.info(f"     result = rag.query('your question')")
        logger.info("")
    else:
        logger.error("✗ Build incomplete")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n✗ Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n✗ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
