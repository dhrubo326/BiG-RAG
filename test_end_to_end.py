"""
End-to-End BiG-RAG Test with LLM Integration
Tests complete RAG pipeline: Retrieval + LLM Generation
Uses gpt-4o-mini for answer synthesis
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict

# Add bigrag to path
sys.path.insert(0, str(Path(__file__).parent))

from bigrag import BiGRAG, QueryParam
from bigrag.llm import gpt_4o_mini_complete, gpt_4o_complete, openai_embedding
from bigrag.utils import logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_end_to_end.log'),
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

    logger.info(f"✓ Knowledge graph found at {working_dir}")

    # Initialize BiGRAG
    rag = BiGRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding,
        enable_llm_cache=True,
    )

    logger.info("✓ BiG-RAG initialized")
    return rag


async def generate_answer_with_llm(
    question: str,
    context: str,
    llm_func,
    use_advanced_model: bool = False
):
    """
    Generate answer using LLM with retrieved context

    Args:
        question: User question
        context: Retrieved knowledge from BiGRAG
        llm_func: LLM completion function
        use_advanced_model: If True, use gpt-4o instead of gpt-4o-mini
    """
    # Construct RAG prompt
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.

Instructions:
- Use ONLY the information from the provided context to answer
- Be concise and accurate
- If the context doesn't contain enough information, say so
- Cite specific facts from the context when possible"""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    try:
        # Select model
        if use_advanced_model:
            response = await gpt_4o_complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )
        else:
            response = await llm_func(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )

        return response.strip()

    except Exception as e:
        logger.error(f"❌ LLM generation failed: {e}")
        return None


def retrieve_context(rag, question: str, top_k: int = 3) -> str:
    """
    Retrieve relevant context from BiGRAG
    """
    param = QueryParam(
        mode="hybrid",
        top_k=top_k,
        max_token_for_text_unit=4000,
        max_token_for_local_context=4000,
        max_token_for_global_context=4000,
    )

    results = rag.query(question, param=param)

    if not results:
        return ""

    # Combine top results into context
    context_parts = []
    for i, result in enumerate(results[:top_k], 1):
        knowledge = result.get("<knowledge>", "")
        if knowledge:
            context_parts.append(f"[Source {i}]\n{knowledge}")

    return "\n\n".join(context_parts)


def simple_match(answer: str, golden_answers: List[str]) -> bool:
    """
    Simple matching: check if any golden answer appears in generated answer
    (case-insensitive, normalized)
    """
    answer_lower = answer.lower()

    for golden in golden_answers:
        golden_lower = golden.lower()

        # Remove punctuation for better matching
        import re
        answer_normalized = re.sub(r'[^\w\s]', '', answer_lower)
        golden_normalized = re.sub(r'[^\w\s]', '', golden_lower)

        if golden_normalized in answer_normalized:
            return True

    return False


async def test_rag_qa(
    rag,
    qa_data: List[Dict],
    llm_func,
    use_advanced_model: bool = False
):
    """
    Test complete RAG pipeline on QA dataset
    """
    model_name = "gpt-4o" if use_advanced_model else "gpt-4o-mini"

    logger.info("")
    logger.info("="*80)
    logger.info(f"TESTING COMPLETE RAG PIPELINE (LLM: {model_name})")
    logger.info("="*80)
    logger.info(f"Total questions: {len(qa_data)}")
    logger.info("")

    results = []

    for i, qa in enumerate(qa_data, 1):
        question = qa["question"]
        golden_answers = qa["golden_answers"]

        logger.info("="*80)
        logger.info(f"[{i}/{len(qa_data)}] Question: {question}")
        logger.info(f"Expected answer: {golden_answers[0]}")
        logger.info("")

        # Step 1: Retrieve context
        logger.info("Step 1: Retrieving context from BiGRAG...")
        context = retrieve_context(rag, question, top_k=3)

        if not context:
            logger.warning("⚠ No context retrieved")
            results.append({
                "question": question,
                "success": False,
                "retrieval_success": False,
                "answer": None,
                "match": False
            })
            logger.info("")
            continue

        # Show retrieved context (truncated)
        context_preview = context[:300] + "..." if len(context) > 300 else context
        logger.info(f"✓ Retrieved context ({len(context)} chars):")
        logger.info(f"  {context_preview}")
        logger.info("")

        # Step 2: Generate answer with LLM
        logger.info("Step 2: Generating answer with LLM...")
        answer = await generate_answer_with_llm(
            question,
            context,
            llm_func,
            use_advanced_model
        )

        if not answer:
            logger.warning("⚠ Answer generation failed")
            results.append({
                "question": question,
                "success": False,
                "retrieval_success": True,
                "answer": None,
                "match": False
            })
            logger.info("")
            continue

        logger.info(f"✓ Generated answer: {answer}")
        logger.info("")

        # Step 3: Check if answer matches expected
        match = simple_match(answer, golden_answers)

        if match:
            logger.info("✓ Answer matches expected response!")
        else:
            logger.info("⚠ Answer does not match expected response")

        results.append({
            "question": question,
            "success": True,
            "retrieval_success": True,
            "answer": answer,
            "match": match,
            "expected": golden_answers[0]
        })

        logger.info("")

    # Summary
    logger.info("="*80)
    logger.info("END-TO-END TEST SUMMARY")
    logger.info("="*80)

    total = len(results)
    retrieval_success = sum(1 for r in results if r["retrieval_success"])
    generation_success = sum(1 for r in results if r["success"])
    answer_match = sum(1 for r in results if r["match"])

    logger.info(f"  Total questions: {total}")
    logger.info(f"  Retrieval success: {retrieval_success}/{total} ({retrieval_success/total*100:.1f}%)")
    logger.info(f"  Generation success: {generation_success}/{total} ({generation_success/total*100:.1f}%)")
    logger.info(f"  Answer matches: {answer_match}/{total} ({answer_match/total*100:.1f}%)")
    logger.info("="*80)
    logger.info("")

    return results


async def demo_interactive_query(rag, llm_func):
    """
    Interactive demo: user can ask questions
    """
    logger.info("")
    logger.info("="*80)
    logger.info("INTERACTIVE DEMO MODE")
    logger.info("="*80)
    logger.info("Ask questions about AI and ML (type 'quit' to exit)")
    logger.info("")

    example_questions = [
        "What is deep learning?",
        "Which companies developed TensorFlow and PyTorch?",
        "What are the applications of computer vision?",
    ]

    logger.info("Example questions:")
    for q in example_questions:
        logger.info(f"  - {q}")
    logger.info("")

    # For testing, use example questions instead of interactive input
    logger.info("Running example questions...\n")

    for question in example_questions:
        logger.info("="*80)
        logger.info(f"Question: {question}")
        logger.info("="*80)

        # Retrieve
        context = retrieve_context(rag, question, top_k=3)
        if not context:
            logger.warning("⚠ No context found")
            continue

        # Generate
        answer = await generate_answer_with_llm(question, context, llm_func)
        if answer:
            logger.info(f"\nAnswer: {answer}\n")
        else:
            logger.warning("⚠ Failed to generate answer\n")


async def main():
    """Main test pipeline"""
    print("="*80)
    print("BiG-RAG End-to-End Test (Retrieval + LLM)")
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

    # Step 3: Load test questions
    logger.info("Step 3: Loading test questions...")
    qa_data = load_test_questions(data_source)
    print("")

    # Step 4: Run end-to-end tests with gpt-4o-mini
    logger.info("Step 4: Running end-to-end tests (gpt-4o-mini)...")
    results = await test_rag_qa(rag, qa_data, gpt_4o_mini_complete, use_advanced_model=False)
    print("")

    # Step 5: Interactive demo
    logger.info("Step 5: Running interactive demo...")
    await demo_interactive_query(rag, gpt_4o_mini_complete)
    print("")

    logger.info("✅ ALL END-TO-END TESTS COMPLETE!")
    logger.info("")
    logger.info("Test logs saved to: test_end_to_end.log")
    print("")


if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\n❌ Tests cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
