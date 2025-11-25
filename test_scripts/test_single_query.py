"""
Test single query with detailed logging
"""

import sys
sys.path.insert(0, 'D:/BiG-RAG')

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
from bigrag import BiGRAG
from bigrag.base import QueryParam
import traceback

async def test_query():
    print("Testing single query with production pipeline...")

    rag = BiGRAG(
        working_dir="D:/BiG-RAG/expr/kuet_production",
        addon_params={"language": "Bangla"}
    )

    query = "CSE তে কত আসন আছে?"

    try:
        query_param = QueryParam(
            mode="hybrid",
            only_need_context=True,
            top_k=5
        )

        print(f"\nQuery: {query}")
        print(f"QueryParam: mode={query_param.mode}, top_k={query_param.top_k}")

        contexts = await rag.aquery(query, query_param)

        print(f"\n[SUCCESS] Retrieved {len(contexts)} contexts")
        for i, ctx in enumerate(contexts, 1):
            if isinstance(ctx, dict):
                print(f"\nContext {i}:")
                print(f"  Type: {type(ctx)}")
                print(f"  Keys: {list(ctx.keys())}")
                if '<knowledge>' in ctx:
                    print(f"  Knowledge: {ctx['<knowledge>'][:200]}...")
            else:
                print(f"\nContext {i}: {type(ctx)}")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_query())
