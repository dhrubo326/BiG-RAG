"""
Test BiG-RAG Retrieval Pipeline
Tests the knowledge graph query functionality
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add bigrag to path
sys.path.insert(0, str(Path(__file__).parent))

from bigrag import BiGRAG, QueryParam
from bigrag.llm import gpt_4o_mini_complete, openai_embedding
from bigrag.utils import logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_retrieval.log'),
        logging.StreamHandler()
    ]
)


def load_api_key():
    """Load OpenAI API key"""
    api_key_file = Path("openai_api_key.txt")
    if api_key_file.exists():
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info("✓ Loaded OpenAI API key")
        return api_key
    else:
        logger.error("❌ openai_api_key.txt not found!")
        sys.exit(1)


def load_test_questions(data_source: str):
    """Load test questions from qa_test.json"""
    qa_path = Path(f"datasets/{data_source}/raw/qa_test.json")

    if not qa_path.exists():
        logger.error(f"❌ Test questions not found: {qa_path}")
        sys.exit(1)

    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    logger.info(f"✓ Loaded {len(qa_data)} test questions")
    return qa_data


def initialize_rag(data_source: str):
    """Initialize BiG-RAG for querying"""
    working_dir = f"expr/{data_source}"

    # Check if graph exists
    graph_dir = Path(working_dir)
    if not graph_dir.exists():
        logger.error(f"❌ Knowledge graph not found at {working_dir}")
        logger.error("Please run test_build_graph.py first!")
        sys.exit(1)

    required_files = [
        "kv_store_text_chunks.json",
        "kv_store_entities.json",
        "kv_store_bipartite_edges.json",
    ]

    for filename in required_files:
        if not (graph_dir / filename).exists():
            logger.error(f"❌ Missing file: {filename}")
            logger.error("Please run test_build_graph.py first!")
            sys.exit(1)

    logger.info(f"✓ Knowledge graph found at {working_dir}")

    # Initialize BiGRAG
    rag = BiGRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding,
        enable_llm_cache=True,
    )

    logger.info("✓ BiG-RAG initialized for querying")
    return rag


def test_single_query(rag, query: str, mode: str = "hybrid", top_k: int = 5):
    """Test a single query"""
    logger.info("")
    logger.info("="*80)
    logger.info(f"Query: {query}")
    logger.info(f"Mode: {mode} | Top-K: {top_k}")
    logger.info("="*80)

    try:
        # Create query parameters
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            max_token_for_text_unit=4000,
            max_token_for_local_context=4000,
            max_token_for_global_context=4000,
        )

        # Execute query
        logger.info("Executing query...")
        results = rag.query(query, param=param)

        # Display results
        logger.info(f"✓ Retrieved {len(results)} results")
        logger.info("")

        if results:
            for i, result in enumerate(results, 1):
                knowledge = result.get("<knowledge>", "")
                coherence = result.get("<coherence>", 0.0)

                # Truncate long results
                if len(knowledge) > 300:
                    knowledge = knowledge[:300] + "..."

                logger.info(f"Result {i} (Coherence: {coherence:.4f}):")
                logger.info(f"  {knowledge}")
                logger.info("")
        else:
            logger.warning("⚠ No results found")

        return results

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def test_all_modes(rag, query: str):
    """Test query with all retrieval modes"""
    modes = ["hybrid", "local", "global", "naive"]

    logger.info("")
    logger.info("="*80)
    logger.info("TESTING ALL RETRIEVAL MODES")
    logger.info("="*80)
    logger.info(f"Query: {query}")
    logger.info("")

    results_summary = {}

    for mode in modes:
        logger.info(f"Testing mode: {mode}...")
        results = test_single_query(rag, query, mode=mode, top_k=3)
        results_summary[mode] = len(results)
        logger.info(f"Mode '{mode}': {len(results)} results retrieved")
        logger.info("")

    # Summary
    logger.info("="*80)
    logger.info("RETRIEVAL MODE COMPARISON")
    logger.info("="*80)
    for mode, count in results_summary.items():
        logger.info(f"  {mode:12s}: {count} results")
    logger.info("="*80)
    logger.info("")


def run_qa_tests(rag, qa_data):
    """Run all QA test questions"""
    logger.info("")
    logger.info("="*80)
    logger.info("RUNNING QA TESTS")
    logger.info("="*80)
    logger.info(f"Total questions: {len(qa_data)}")
    logger.info("")

    results = []

    for i, qa in enumerate(qa_data, 1):
        question = qa["question"]
        golden_answers = qa["golden_answers"]

        logger.info(f"[{i}/{len(qa_data)}] Question: {question}")
        logger.info(f"         Expected: {golden_answers[0]}")

        # Query BiGRAG
        param = QueryParam(mode="hybrid", top_k=5)
        retrieved = rag.query(question, param=param)

        if retrieved:
            top_result = retrieved[0].get("<knowledge>", "")
            coherence = retrieved[0].get("<coherence>", 0.0)

            # Truncate for display
            display_result = top_result[:200] + "..." if len(top_result) > 200 else top_result

            logger.info(f"         Retrieved: {display_result}")
            logger.info(f"         Coherence: {coherence:.4f}")
            logger.info(f"         ✓ Success")

            results.append({
                "question": question,
                "success": True,
                "num_results": len(retrieved),
                "coherence": coherence
            })
        else:
            logger.warning(f"         ⚠ No results")
            results.append({
                "question": question,
                "success": False,
                "num_results": 0,
                "coherence": 0.0
            })

        logger.info("")

    # Summary
    logger.info("="*80)
    logger.info("QA TEST SUMMARY")
    logger.info("="*80)
    successful = sum(1 for r in results if r["success"])
    avg_coherence = sum(r["coherence"] for r in results if r["success"]) / max(successful, 1)
    logger.info(f"  Total questions: {len(results)}")
    logger.info(f"  Successful retrievals: {successful}/{len(results)}")
    logger.info(f"  Success rate: {successful/len(results)*100:.1f}%")
    logger.info(f"  Average coherence: {avg_coherence:.4f}")
    logger.info("="*80)
    logger.info("")

    return results


def main():
    """Main test pipeline"""
    print("="*80)
    print("BiG-RAG Retrieval Pipeline Test")
    print("="*80)
    print("")

    data_source = "demo_test"

    # Step 1: Load API key
    logger.info("Step 1: Loading OpenAI API key...")
    load_api_key()
    print("")

    # Step 2: Initialize BiG-RAG
    logger.info("Step 2: Initializing BiG-RAG...")
    rag = initialize_rag(data_source)
    print("")

    # Step 3: Test single query
    logger.info("Step 3: Testing single query...")
    test_query = "What is Artificial Intelligence?"
    test_single_query(rag, test_query, mode="hybrid", top_k=5)

    # Step 4: Test all retrieval modes
    logger.info("Step 4: Testing all retrieval modes...")
    test_all_modes(rag, "What is machine learning?")

    # Step 5: Run full QA test suite
    logger.info("Step 5: Running full QA test suite...")
    qa_data = load_test_questions(data_source)
    run_qa_tests(rag, qa_data)

    logger.info("✅ ALL TESTS COMPLETE!")
    logger.info("")
    logger.info("Next step: Run test_end_to_end.py for complete RAG pipeline test")
    print("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n❌ Tests cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
