"""Quick test of unified system without caching"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from bigrag.unified import UnifiedQueryExecutor
from bigrag.llm import gpt_4o_mini_complete
from bigrag.base import QueryParam

async def test():
    print("Initializing unified executor...")
    executor = UnifiedQueryExecutor(
        registry_path="expr/subgraph_registry.json",
        llm_func=gpt_4o_mini_complete,
        max_cached_subgraphs=5,
        enable_parallel=True
    )

    print("\nTesting query: 'Who won the 2022 World Cup?'")
    result = await executor.query(
        query="Who won the 2022 World Cup?",
        query_param=QueryParam(only_need_context=True, top_k=3),
        include_metadata=True
    )

    print(f"\nRouting: {result['routing']}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    print(f"Results: {len(result['results'])} items")

    if result['results']:
        print("\nFirst result:")
        print(result['results'][0])
    else:
        print("\nSubgraph results:")
        for sg, sg_result in result['subgraph_results'].items():
            print(f"  {sg}: success={sg_result['success']}, error={sg_result.get('error', 'None')}")

asyncio.run(test())
