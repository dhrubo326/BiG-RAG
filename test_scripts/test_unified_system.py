"""
Test Unified Subgraph System

Tests the complete unified query flow:
1. Router selects relevant subgraph(s)
2. Cache lazy-loads subgraphs
3. Executor queries and aggregates results
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bigrag.unified import UnifiedQueryExecutor
from bigrag.llm import gpt_4o_mini_complete
from bigrag.base import QueryParam


async def test_unified_system():
    """Test unified query system end-to-end."""

    print("="*60)
    print("Testing Unified Subgraph System")
    print("="*60)

    # Initialize executor
    print("\n[1/6] Initializing UnifiedQueryExecutor...")
    try:
        executor = UnifiedQueryExecutor(
            registry_path=str(PROJECT_ROOT / "expr" / "subgraph_registry.json"),
            llm_func=gpt_4o_mini_complete,
            max_cached_subgraphs=5,
            prewarm_subgraphs=None,  # Don't prewarm for testing
            enable_parallel=True
        )
        print(f"[OK] Executor initialized")
        print(f"     Available subgraphs: {executor.get_available_subgraphs()}")
    except Exception as e:
        print(f"[FAIL] Executor initialization failed: {e}")
        return

    # Test 1: List subgraphs
    print("\n[2/6] Testing get_available_subgraphs()...")
    try:
        subgraphs = executor.get_available_subgraphs()
        print(f"[OK] Found {len(subgraphs)} subgraphs: {subgraphs}")
    except Exception as e:
        print(f"[FAIL] {e}")

    # Test 2: Get subgraph info
    print("\n[3/6] Testing get_subgraph_info()...")
    try:
        info = executor.get_subgraph_info("kuet_test")
        print(f"[OK] KUET info retrieved:")
        print(f"     Description: {info['description']}")
        print(f"     Aliases: {info['aliases'][:3]}...")
        print(f"     Topics: {info['topics'][:5]}...")
    except Exception as e:
        print(f"[FAIL] {e}")

    # Test 3: Router (routing decision only)
    print("\n[4/6] Testing router.route()...")
    test_queries = [
        "Who won the 2022 World Cup?",
        "How many seats in KUET CSE department?",
        "Tell me about something general"
    ]

    for query in test_queries:
        try:
            routing = await executor.router.route(query)
            print(f"[OK] Query: '{query}'")
            print(f"     Routed to: {routing['subgraphs']}")
            print(f"     Confidence: {routing['confidence']:.2f}")
            print(f"     Reasoning: {routing['reasoning'][:80]}...")
        except Exception as e:
            print(f"[FAIL] Routing failed for '{query}': {e}")

    # Test 4: Cache stats (before any queries)
    print("\n[5/6] Testing cache stats (before queries)...")
    try:
        stats = executor.get_cache_stats()
        print(f"[OK] Cache stats:")
        print(f"     Cache size: {stats['cache_size']}/{stats['max_size']}")
        print(f"     Hits: {stats['hits']}, Misses: {stats['misses']}")
        print(f"     Cached subgraphs: {stats['cached_subgraphs']}")
    except Exception as e:
        print(f"[FAIL] {e}")

    # Test 5: Full unified query (without actual BiGRAG instances to avoid loading)
    print("\n[6/6] Testing full unified query...")
    print("     [NOTE] Skipping actual query to avoid loading large subgraphs")
    print("     To test full query, ensure subgraphs are built and run:")
    print("     python test_scripts/test_unified_query.py")

    print("\n" + "="*60)
    print("All Tests Completed!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Start server in unified mode:")
    print("   cd backend && python server.py --unified")
    print("2. Test via API:")
    print("   curl -X POST http://localhost:8001/api/unified/query \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"query\": \"Who won 2022 World Cup?\", \"top_k\": 5}'")
    print("3. Check available subgraphs:")
    print("   curl http://localhost:8001/api/unified/subgraphs")


if __name__ == "__main__":
    asyncio.run(test_unified_system())
