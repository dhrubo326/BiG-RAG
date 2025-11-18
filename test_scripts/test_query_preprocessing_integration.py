import asyncio
from bigrag import BiGRAG
from bigrag.base import QueryParam

async def test_full_pipeline():
    rag = BiGRAG(working_dir="./expr/demo_test", enable_llm_cache=True)

    results = await rag.aquery(
        "who is messi",
        param=QueryParam(mode="hybrid", top_k=60, only_need_context=True)
    )

    print(f"[OK] Retrieved {len(results)} items")
    for item in results[:3]:
        print(f"  - {item['<knowledge>'][:80]}...")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
