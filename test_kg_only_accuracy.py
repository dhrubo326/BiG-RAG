"""
KG-Only Retrieval Accuracy Test

This script tests the knowledge graph quality by:
1. Creating test questions from KUET document
2. Retrieving ONLY from KG (Entity + Relation paths, NO chunks)
3. Verifying if KG contains accurate answers

This validates the graph structure itself, not chunk retrieval.
"""

import asyncio
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bigrag import BiGRAG, QueryParam


# Test questions with expected answers
TEST_QUESTIONS = [
    {
        "question": "KUET CSE department has how many seats?",
        "expected_answer": "120",
        "keywords": ["CSE", "120", "seats", "Computer Science"]
    },
    {
        "question": "When is the KUET admission test?",
        "expected_answer": "January 11, 2025",
        "keywords": ["January", "11", "2025", "admission test"]
    },
    {
        "question": "How many questions are in Physics in the admission test?",
        "expected_answer": "15 questions",
        "keywords": ["Physics", "15", "questions"]
    },
    {
        "question": "What is the total number of marks in the admission test?",
        "expected_answer": "500 marks",
        "keywords": ["500", "marks", "total"]
    },
    {
        "question": "How many seats does EEE department have?",
        "expected_answer": "120 seats",
        "keywords": ["EEE", "120", "seats", "Electrical"]
    },
    {
        "question": "What is the minimum GPA required for SSC?",
        "expected_answer": "4.00",
        "keywords": ["GPA", "4.00", "SSC", "secondary"]
    },
    {
        "question": "How many seats does BME department have?",
        "expected_answer": "30 seats",
        "keywords": ["BME", "30", "Biomedical"]
    },
    {
        "question": "What is the total number of seats in KUET?",
        "expected_answer": "1065 seats",
        "keywords": ["1065", "total", "seats"]
    },
    {
        "question": "How many questions are in English in the admission test?",
        "expected_answer": "10 questions",
        "keywords": ["English", "10", "questions"]
    },
    {
        "question": "When does the application deadline end?",
        "expected_answer": "December 14, 2024",
        "keywords": ["December", "14", "2024", "deadline", "application"]
    }
]


async def test_kg_only_retrieval(data_source: str = "demo_test"):
    """
    Test KG-only retrieval accuracy
    """

    print("="*80)
    print("KG-ONLY RETRIEVAL ACCURACY TEST")
    print("="*80)
    print("Testing Knowledge Graph quality WITHOUT chunk retrieval")
    print("Mode: Dual-path (Entity + Relation only)")
    print("="*80)
    print()

    # Read API key
    api_key_file = "openai_api_key.txt"
    if not os.path.exists(api_key_file):
        print(f"[ERROR] {api_key_file} not found!")
        return

    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()

    os.environ['OPENAI_API_KEY'] = api_key

    # Initialize BiGRAG
    working_dir = f"expr/{data_source}"

    print(f"[INFO] Loading BiGRAG from: {working_dir}")
    rag = BiGRAG(
        working_dir=working_dir,
        enable_llm_cache=True
    )

    print(f"[INFO] Graph loaded successfully")
    print()

    # Test each question
    results = []
    correct_count = 0
    partial_count = 0
    missing_count = 0

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected = test["expected_answer"]
        keywords = test["keywords"]

        print(f"[{i}/{len(TEST_QUESTIONS)}] Testing: {question}")
        print(f"      Expected: {expected}")

        # Query with KG-ONLY (no chunks)
        # Mode: hybrid (Entity + Relation paths)
        # num_chunks_in_context: 0 (disable chunk retrieval - Path C)
        result = await rag.aquery(
            question,
            param=QueryParam(
                mode="hybrid",  # Use Entity + Relation paths
                only_need_context=True,
                top_k=60,
                num_kg_in_context=15,  # Get 15 KG items (relations)
                num_chunks_in_context=0,  # DISABLE chunks (Path C)
                enable_reranking=False
            )
        )

        if not result:
            print(f"      Result: [MISSING] No KG context retrieved")
            missing_count += 1
            results.append({
                "question": question,
                "expected": expected,
                "status": "MISSING",
                "contexts": []
            })
            print()
            continue

        # Check if any context contains the expected answer
        contexts = []
        found_exact = False
        found_partial = False

        for item in result:
            if isinstance(item, dict):
                context = item.get("<knowledge>", str(item))
                item_type = item.get("<type>", "unknown")
                score = item.get("<coherence>", 0.0)
            else:
                context = str(item)
                item_type = "unknown"
                score = 0.0

            contexts.append({
                "type": item_type,
                "content": context,
                "score": score
            })

            # Check for exact match
            if expected.lower() in context.lower():
                found_exact = True

            # Check for partial match (all keywords present)
            if all(kw.lower() in context.lower() for kw in keywords):
                found_partial = True

        # Determine status
        if found_exact:
            status = "CORRECT"
            correct_count += 1
            print(f"      Result: [OK] Found exact answer in KG")
        elif found_partial:
            status = "PARTIAL"
            partial_count += 1
            print(f"      Result: [PARTIAL] Found related info in KG")
        else:
            status = "INCORRECT"
            missing_count += 1
            print(f"      Result: [MISSING] Answer not found in KG")

        # Show top 2 contexts
        print(f"      Top KG contexts:")
        for j, ctx in enumerate(contexts[:2], 1):
            print(f"        {j}. [{ctx['type']}] {ctx['content'][:80]}...")

        results.append({
            "question": question,
            "expected": expected,
            "status": status,
            "contexts": contexts
        })

        print()

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Questions: {len(TEST_QUESTIONS)}")
    print(f"Correct (Exact Match): {correct_count} ({correct_count/len(TEST_QUESTIONS)*100:.1f}%)")
    print(f"Partial (Keywords Found): {partial_count} ({partial_count/len(TEST_QUESTIONS)*100:.1f}%)")
    print(f"Missing (No Answer): {missing_count} ({missing_count/len(TEST_QUESTIONS)*100:.1f}%)")
    print()

    # Calculate accuracy
    total_useful = correct_count + partial_count
    accuracy = (total_useful / len(TEST_QUESTIONS)) * 100

    print(f"KG Coverage: {accuracy:.1f}% (Correct + Partial)")
    print()

    if accuracy >= 80:
        print("[PASS] KG quality is GOOD - contains most critical information")
    elif accuracy >= 60:
        print("[WARNING] KG quality is MODERATE - some information missing")
    else:
        print("[FAIL] KG quality is POOR - significant information missing")

    print("="*80)

    # Save detailed results
    import json
    results_file = "kg_only_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_type": "kg_only_retrieval",
            "mode": "hybrid (Entity + Relation, NO chunks)",
            "total_questions": len(TEST_QUESTIONS),
            "correct": correct_count,
            "partial": partial_count,
            "missing": missing_count,
            "accuracy": accuracy,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Detailed results saved to: {results_file}")

    return results, accuracy


async def main():
    results, accuracy = await test_kg_only_retrieval("demo_test")

    print("\n[INFO] Test completed")
    print(f"[INFO] KG Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
