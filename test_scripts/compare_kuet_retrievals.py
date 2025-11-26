"""
Compare retrieval effectiveness between Standard and Production pipelines

Simplified test with concise output
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

# Test queries
QUERIES = [
    "CSE তে কত আসন আছে?",  # Structured - table data
    "বায়োমেডিকেল ইঞ্জিনিয়ারিং বিভাগে আসন সংখ্যা কত?",  # Structured
    "আবেদনপত্র জমা দেওয়ার শেষ তারিখ কবে?",  # Structured - dates
    "ভর্তি পরীক্ষার যোগ্যতা কি কি?",  # Narrative
    "প্রথম ২০,০০০ প্রার্থী কিভাবে নির্বাচন করা হবে?",  # Narrative
]

async def test_pipeline(pipeline_name, working_dir):
    print(f"\n{'='*80}")
    print(f"{pipeline_name.upper()} PIPELINE TEST")
    print(f"{'='*80}\n")

    rag = BiGRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        chunk_token_size=config.chunk_size,
        chunk_overlap_token_size=config.chunk_overlap_size,
        enable_llm_cache=config.enable_llm_cache,
        addon_params={"language": "Bangla"}
    )

    results = []

    for i, query in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] Testing: {query[:50]}...")

        try:
            query_param = QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=5
            )

            contexts = await rag.aquery(query, query_param)

            # Extract just the content from knowledge tags
            clean_contexts = []
            for ctx in contexts:
                if isinstance(ctx, dict) and '<knowledge>' in ctx:
                    clean_contexts.append(ctx['<knowledge>'][:300])  # First 300 chars
                elif isinstance(ctx, str):
                    clean_contexts.append(ctx[:300])

            result = {
                "query": query,
                "num_contexts": len(contexts),
                "contexts": clean_contexts
            }

            print(f"  -> Retrieved {len(contexts)} contexts")
            results.append(result)

        except Exception as e:
            print(f"  -> ERROR: {str(e)[:100]}")
            results.append({
                "query": query,
                "error": str(e)
            })

    return results

async def main():
    print("\n" + "="*80)
    print("KUET RETRIEVAL COMPARISON: STANDARD VS PRODUCTION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Test standard pipeline
    std_results = await test_pipeline("standard", "D:/BiG-RAG/expr/keut_unfied")

    # Test production pipeline
    prod_results = await test_pipeline("production", "D:/BiG-RAG/expr/kuet_production")

    # Save combined results
    output = {
        "test_date": datetime.now().isoformat(),
        "unfied": std_results,
        "production": prod_results
    }

    output_file = "D:/BiG-RAG/test_scripts/kuet_retrieval_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    for i, query in enumerate(QUERIES):
        std_count = std_results[i].get('num_contexts', 0) if i < len(std_results) else 0
        prod_count = prod_results[i].get('num_contexts', 0) if i < len(prod_results) else 0

        print(f"Query {i+1}: {query[:50]}...")
        print(f"  Standard:   {std_count} contexts")
        print(f"  Production: {prod_count} contexts")

        if std_count > prod_count:
            print(f"  Winner: STANDARD (+{std_count - prod_count})")
        elif prod_count > std_count:
            print(f"  Winner: PRODUCTION (+{prod_count - std_count})")
        else:
            print(f"  Winner: TIE")
        print()

    print(f"Detailed results saved to: {output_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
