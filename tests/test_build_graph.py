"""
Build Knowledge Graph with OpenAI Models (Test Script)
This script builds a BiG-RAG knowledge graph using:
- gpt-4o-mini for entity extraction
- text-embedding-3-large for embeddings
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

# Add bigrag to path
sys.path.insert(0, str(Path(__file__).parent))

from bigrag import BiGRAG
from bigrag.llm import gpt_4o_mini_complete, openai_embedding
from bigrag.utils import logger

# Configure logging (use UTF-8 encoding for file handler to support emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_graph.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# For Windows console, replace sys.stdout to handle Unicode
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load OpenAI API key
def load_api_key():
    """Load OpenAI API key from file"""
    api_key_file = Path("openai_api_key.txt")
    if api_key_file.exists():
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info(" Loaded OpenAI API key from openai_api_key.txt")
        return api_key
    else:
        logger.error(" openai_api_key.txt not found!")
        sys.exit(1)


def load_corpus(data_source: str):
    """Load corpus from JSONL file"""
    corpus_path = Path(f"datasets/{data_source}/raw/corpus.jsonl")

    if not corpus_path.exists():
        logger.error(f" Corpus not found: {corpus_path}")
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

    logger.info(f" Loaded {len(documents)} documents from {corpus_path}")
    return documents


def extract_knowledge(rag, documents):
    """
    Extract entities and build bipartite graph
    Uses BiGRAG's insert method which handles:
    1. Chunking
    2. Entity extraction via LLM
    3. Graph construction
    """
    logger.info("="*80)
    logger.info("PHASE 1: Extracting Knowledge (Entities & Relations)")
    logger.info("="*80)
    logger.info(f"Total documents to process: {len(documents)}")
    logger.info("This will take 5-15 minutes depending on document size...")
    logger.info("")

    # Extract content for BiGRAG
    contents = [doc["content"] for doc in documents]

    # Process in batches to avoid overwhelming the API
    batch_size = 5  # Small batches for demo
    total_batches = (len(contents) + batch_size - 1) // batch_size

    logger.info(f"Processing in {total_batches} batches (batch_size={batch_size})")

    for i in range(0, len(contents), batch_size):
        batch_num = i // batch_size + 1
        batch = contents[i:i+batch_size]

        logger.info(f"[Batch {batch_num}/{total_batches}] Processing documents {i+1} to {i+len(batch)}...")

        retries = 0
        max_retries = 3

        while retries < max_retries:
            try:
                # Insert batch into BiGRAG
                rag.insert(batch)
                logger.info(f"[Batch {batch_num}/{total_batches}]  Successfully inserted")
                break
            except Exception as e:
                retries += 1
                logger.warning(f"[Batch {batch_num}/{total_batches}]  Error (attempt {retries}/{max_retries}): {e}")
                if retries < max_retries:
                    wait_time = 5 * retries
                    logger.info(f"[Batch {batch_num}/{total_batches}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[Batch {batch_num}/{total_batches}]  Failed after {max_retries} attempts")
                    raise

    logger.info("")
    logger.info(" Knowledge extraction complete!")
    logger.info("")
    logger.info("Note: Embeddings are automatically created by NanoVectorDB during insert()")
    logger.info("")


def main():
    """Main build pipeline"""
    print("="*80)
    print("BiG-RAG Knowledge Graph Builder (OpenAI Models)")
    print("="*80)
    print("")

    # Configuration
    data_source = "demo_test"
    working_dir = f"expr/{data_source}"

    # Step 1: Load API key
    logger.info("Step 1: Loading OpenAI API key...")
    load_api_key()
    print("")

    # Step 2: Load corpus
    logger.info("Step 2: Loading corpus...")
    documents = load_corpus(data_source)
    print("")

    # Step 3: Initialize BiGRAG with OpenAI models
    logger.info("Step 3: Initializing BiG-RAG...")
    logger.info(f"  - Working directory: {working_dir}")
    logger.info(f"  - LLM: gpt-4o-mini (entity extraction)")
    logger.info(f"  - Embedding: text-embedding-3-large (3072 dimensions)")
    logger.info(f"  - Chunk size: 1200 tokens")
    logger.info(f"  - Chunk overlap: 100 tokens")
    print("")

    rag = BiGRAG(
        working_dir=working_dir,
        # Use OpenAI models
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding,
        # Chunking parameters
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        # Entity extraction parameters
        entity_extract_max_gleaning=1,  # 1 pass for demo (faster)
        entity_summary_to_max_tokens=500,
        # Enable caching to reduce costs during testing
        enable_llm_cache=True,
    )

    logger.info(" BiG-RAG initialized successfully")
    print("")

    # Step 4: Extract knowledge and build graph
    # Note: Embeddings are automatically created during insert() via NanoVectorDB
    try:
        extract_knowledge(rag, documents)
    except Exception as e:
        logger.error(f" Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Step 5: Verify output
    logger.info("="*80)
    logger.info("PHASE 2: Verifying Output")
    logger.info("="*80)

    output_dir = Path(working_dir)

    # After bug fixes: entities and bipartite_edges are stored in VectorDB (not KV storage)
    expected_files = [
        "kv_store_text_chunks.json",  # Text chunks KV storage
        "vdb_entities.json",           # Entities vector database
        "vdb_bipartite_edges.json",    # Bipartite edges vector database
        "graph_chunk_entity_relation.graphml",  # NetworkX graph
    ]

    all_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            logger.info(f" {filename} ({size:,} bytes)")
        else:
            logger.error(f" {filename} - NOT FOUND")
            all_exist = False

    print("")

    if all_exist:
        # Load and show statistics
        with open(output_dir / "kv_store_text_chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        with open(output_dir / "vdb_entities.json", 'r', encoding='utf-8') as f:
            entities_vdb = json.load(f)
        with open(output_dir / "vdb_bipartite_edges.json", 'r', encoding='utf-8') as f:
            edges_vdb = json.load(f)

        logger.info("="*80)
        logger.info("GRAPH STATISTICS")
        logger.info("="*80)
        logger.info(f"  Text Chunks: {len(chunks)}")
        # NanoVectorDB stores vectors in 'data' key
        entities_count = len(entities_vdb.get('data', []))
        edges_count = len(edges_vdb.get('data', []))
        logger.info(f"  Entities: {entities_count}")
        logger.info(f"  Relations (Bipartite Edges): {edges_count}")
        logger.info("="*80)
        print("")

        logger.info(" BUILD SUCCESSFUL!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run test_retrieval.py to test query functionality")
        logger.info("  2. Run test_end_to_end.py for complete pipeline test")
        print("")
    else:
        logger.error(" Build incomplete - some files are missing")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
