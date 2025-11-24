"""
Test retrieval from KUET Standard Pipeline Knowledge Graph

This script tests how well the standard pipeline KG retrieves information
for various types of queries.
"""

import sys
import os
sys.path.insert(0, 'D:/BiG-RAG')

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
from bigrag import BiGRAG
from bigrag.llm import gpt_4o_mini_complete
from bigrag.base import QueryParam
from bigrag.config import config
import json
from datetime import datetime

# Test queries covering different information types
TEST_QUERIES = [
    # Structured data queries (table-based)
    {
        "query": "CSE তে কত আসন আছে?",
        "type": "structured",
        "expected": "CSE department has 120 seats"
    },
    {
        "query": "বায়োমেডিকেল ইঞ্জিনিয়ারিং বিভাগে আসন সংখ্যা কত?",
        "type": "structured",
        "expected": "BME department has 30 seats"
    },
    {
        "query": "আবেদনপত্র জমা দেওয়ার শেষ তারিখ কবে?",
        "type": "structured",
        "expected": "Application submission deadline is 14 December 2024"
    },

    # Narrative/procedural queries
    {
        "query": "ভর্তি পরীক্ষার যোগ্যতা কি কি?",
        "type": "narrative",
        "expected": "Eligibility requirements for admission test"
    },
    {
        "query": "প্রথম ২০,০০০ প্রার্থী কিভাবে নির্বাচন করা হবে?",
        "type": "narrative",
        "expected": "Selection process for first 20,000 candidates"
    },
    {
        "query": "মেধা তালিকা কিভাবে তৈরি করা হবে?",
        "type": "narrative",
        "expected": "Merit list preparation process"
    },

    # Multi-hop queries
    {
        "query": "সিভিল ইঞ্জিনিয়ারিং অনুষদে কোন কোন বিভাগ আছে এবং প্রতিটিতে কত আসন?",
        "type": "multi-hop",
        "expected": "Civil Engineering faculty departments and their seat counts"
    },
    {
        "query": "ভর্তি পরীক্ষার বিষয় এবং নম্বর বিন্যাস কি?",
        "type": "multi-hop",
        "expected": "Admission test subjects and mark distribution"
    },
]

async def test_standard_pipeline():
    print("="*80)
    print("KUET STANDARD PIPELINE RETRIEVAL TEST")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize BiGRAG with standard pipeline
    working_dir = "D:/BiG-RAG/expr/kuet_standard"

    print(f"Loading knowledge graph from: {working_dir}")
    rag = BiGRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        chunk_token_size=config.chunk_size,
        chunk_overlap_token_size=config.chunk_overlap_size,
        enable_llm_cache=config.enable_llm_cache,
        addon_params={"language": "Bangla"}
    )

    print("Knowledge graph loaded successfully!")
    print()

    # Test each query
    results = []

    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(TEST_QUERIES)}: {test_case['type'].upper()} QUERY")
        print(f"{'='*80}")
        print(f"Query: {test_case['query']}")
        print(f"Expected: {test_case['expected']}")
        print()

        try:
            # Query the KG
            query_param = QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=5
            )

            contexts = await rag.aquery(test_case['query'], query_param)

            # Analyze results
            result = {
                "query": test_case['query'],
                "type": test_case['type'],
                "expected": test_case['expected'],
                "num_contexts": len(contexts),
                "contexts": []
            }

            print(f"Retrieved {len(contexts)} context items:")
            print("-" * 80)

            for j, ctx in enumerate(contexts, 1):
                print(f"\nContext {j}:")
                print(f"  Content: {ctx[:200]}..." if len(ctx) > 200 else f"  Content: {ctx}")

                result["contexts"].append({
                    "rank": j,
                    "content": ctx,
                    "length": len(ctx)
                })

            results.append(result)

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "query": test_case['query'],
                "type": test_case['type'],
                "error": str(e)
            })

    # Save results
    output_file = "D:/BiG-RAG/test_scripts/kuet_standard_retrieval_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "pipeline": "standard",
            "test_date": datetime.now().isoformat(),
            "total_queries": len(TEST_QUERIES),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"Successful: {len([r for r in results if 'error' not in r])}")
    print(f"Failed: {len([r for r in results if 'error' in r])}")
    print(f"\nResults saved to: {output_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_standard_pipeline())
