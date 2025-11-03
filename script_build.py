"""
Build BiG-RAG Knowledge Graph using OpenAI Models
- GPT-4o-mini for entity extraction
- text-embedding-3-large for embeddings (3072 dimensions)
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

# Add bigrag to path
sys.path.insert(0, str(Path(__file__).parent))

from bigrag import BiGRAG
from bigrag.llm import gpt_4o_mini_complete, openai_embedding
from bigrag.utils import logger
from bigrag.config import config

# Configure logging (UTF-8 encoding for Windows console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_graph.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Fix Windows console encoding for Unicode characters
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_api_key():
    """
    Load OpenAI API key from configuration.
    Config automatically handles: .env file → openai_api_key.txt → environment variable
    """
    if not config.openai_api_key:
        logger.error("ERROR: OpenAI API key not found!")
        logger.error("Please set it in one of these ways:")
        logger.error("  1. Create .env file with: OPENAI_API_KEY=sk-your-key")
        logger.error("  2. Create openai_api_key.txt file")
        logger.error("  3. Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    logger.info(f"Using OpenAI API key: {config.openai_api_key[:12]}...")
    return config.openai_api_key


def load_corpus(data_source: str):
    """Load corpus from JSONL file"""
    corpus_path = Path(f"{config.input_dir}/{data_source}/raw/corpus.jsonl")

    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        logger.error("Run convert_csv_to_corpus.py first!")
        sys.exit(1)

    documents = []
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            documents.append({
                "id": data.get("id", ""),
                "content": data.get("contents", ""),
                "title": data.get("title", "")
            })

    logger.info(f"Loaded {len(documents)} documents from {corpus_path}")
    return documents


def extract_knowledge(rag, documents, batch_size=5):
    """
    Extract entities and build bipartite graph with metadata preservation
    Uses BiGRAG's insert method which handles:
    1. Chunking (with metadata preservation)
    2. Entity extraction via GPT-4o-mini (context-enhanced with metadata)
    3. Bipartite graph construction
    4. Embeddings via text-embedding-3-large
    """
    logger.info("="*80)
    logger.info("BUILDING KNOWLEDGE GRAPH")
    logger.info("="*80)
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("")
    logger.info("This will take 10-30 minutes depending on:")
    logger.info("  - Number of documents")
    logger.info("  - OpenAI API rate limits")
    logger.info("  - Entity extraction complexity")
    logger.info("")

    # Extract content AND metadata for BiGRAG (Phase 2.1 improvement)
    contents = [doc["content"] for doc in documents]
    metadata_list = [
        {
            "title": doc.get("title", ""),
            "id": doc.get("id", ""),
        }
        for doc in documents
    ]

    # Process in batches to avoid overwhelming the API
    total_batches = (len(contents) + batch_size - 1) // batch_size

    logger.info(f"Processing in {total_batches} batches...")
    logger.info("(Now with metadata preservation for improved entity extraction)")
    logger.info("")

    for i in range(0, len(contents), batch_size):
        batch_num = i // batch_size + 1
        batch_contents = contents[i:i+batch_size]
        batch_metadata = metadata_list[i:i+batch_size]

        logger.info(f"[Batch {batch_num}/{total_batches}] Processing documents {i+1} to {min(i+len(batch_contents), len(contents))}...")

        retries = 0
        max_retries = 3

        while retries < max_retries:
            try:
                # Insert batch into BiGRAG with metadata
                rag.insert(batch_contents, metadata=batch_metadata)
                logger.info(f"[Batch {batch_num}/{total_batches}] Successfully inserted")
                break
            except Exception as e:
                retries += 1
                logger.warning(f"[Batch {batch_num}/{total_batches}] Error (attempt {retries}/{max_retries}): {e}")
                if retries < max_retries:
                    wait_time = 5 * retries
                    logger.info(f"[Batch {batch_num}/{total_batches}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[Batch {batch_num}/{total_batches}] Failed after {max_retries} attempts")
                    raise

        # Small delay between batches to respect rate limits
        if batch_num < total_batches:
            time.sleep(2)

    logger.info("")
    logger.info("Knowledge graph built successfully!")
    logger.info("(Embeddings are automatically created by NanoVectorDB during insert)")
    logger.info("")


def verify_output(working_dir: str):
    """Verify output files were created"""
    logger.info("="*80)
    logger.info("VERIFYING OUTPUT FILES")
    logger.info("="*80)

    output_dir = Path(working_dir)

    expected_files = [
        "kv_store_text_chunks.json",
        "vdb_entities.json",
        "vdb_bipartite_edges.json",
        "graph_chunk_entity_relation.graphml",
    ]

    all_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            logger.info(f"  [OK] {filename} ({size:,} bytes)")
        else:
            logger.error(f"  [ERROR] {filename} - NOT FOUND")
            all_exist = False

    logger.info("")

    if all_exist:
        # Load and show statistics
        with open(output_dir / "kv_store_text_chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        with open(output_dir / "vdb_entities.json", 'r', encoding='utf-8') as f:
            vdb_entities = json.load(f)
        with open(output_dir / "vdb_bipartite_edges.json", 'r', encoding='utf-8') as f:
            vdb_bipartite_edges = json.load(f)

        logger.info("="*80)
        logger.info("GRAPH STATISTICS")
        logger.info("="*80)
        logger.info(f"  Text Chunks: {len(chunks)}")
        entities_count = len(vdb_entities.get('data', []))
        edges_count = len(vdb_bipartite_edges.get('data', []))
        logger.info(f"  Entities: {entities_count}")
        logger.info(f"  Relations (Bipartite Edges): {edges_count}")
        logger.info("="*80)
        logger.info("")
        return True
    else:
        logger.error("Build incomplete - some files are missing")
        return False


def main():
    """Main build pipeline"""
    parser = argparse.ArgumentParser(description="Build BiG-RAG knowledge graph with OpenAI models")
    parser.add_argument("--data_source", type=str, default="Single-Topic",
                        help="Dataset name (default: Single-Topic)")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size for processing (default: 5)")
    parser.add_argument("--chunk_size", type=int, default=1200,
                        help="Chunk token size (default: 1200)")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="Chunk overlap token size (default: 100)")

    args = parser.parse_args()

    print("="*80)
    print("BiG-RAG Knowledge Graph Builder (OpenAI Models)")
    print("="*80)
    print(f"Dataset: {args.data_source}")
    print(f"LLM: gpt-4o-mini (entity extraction)")
    print(f"Embedding: text-embedding-3-large (3072 dimensions)")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Chunk overlap: {args.chunk_overlap} tokens")
    print("="*80)
    print("")

    working_dir = f"{config.working_dir}/{args.data_source}"

    # Step 1: Load API key
    logger.info("Step 1: Loading OpenAI API key...")
    load_api_key()
    print("")

    # Step 2: Load corpus
    logger.info("Step 2: Loading corpus...")
    documents = load_corpus(args.data_source)
    print("")

    # Step 3: Initialize BiGRAG with OpenAI models
    logger.info("Step 3: Initializing BiG-RAG...")
    logger.info(f"  Working directory: {working_dir}")
    logger.info(f"  Chunk size: {config.chunk_size} tokens")
    logger.info(f"  Chunk overlap: {config.chunk_overlap_size} tokens")
    logger.info(f"  LLM cache: {config.enable_llm_cache}")
    print("")

    rag = BiGRAG(
        working_dir=working_dir,
        # Use OpenAI models
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding,
        # Chunking parameters from config (can be overridden by CLI args)
        chunk_token_size=args.chunk_size if args.chunk_size != 1200 else config.chunk_size,
        chunk_overlap_token_size=args.chunk_overlap if args.chunk_overlap != 100 else config.chunk_overlap_size,
        # Entity extraction parameters
        entity_extract_max_gleaning=1,
        entity_summary_to_max_tokens=500,
        # Caching from config
        enable_llm_cache=config.enable_llm_cache,
    )

    logger.info("BiG-RAG initialized successfully")
    print("")

    # Step 4: Extract knowledge and build graph
    try:
        extract_knowledge(rag, documents, batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Step 5: Verify output
    if verify_output(working_dir):
        logger.info("="*80)
        logger.info("BUILD SUCCESSFUL!")
        logger.info("="*80)
        logger.info("")
        logger.info("Next step: Run evaluation")
        logger.info(f"  python script_quick_eval.py --sample 20")
        logger.info("")
    else:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nBuild cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
