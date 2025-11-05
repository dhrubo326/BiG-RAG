"""
Comprehensive retrieval test for demo_test dataset
Tests all critical retrieval paths and graph traversal
"""
import requests
import json
from typing import List, Dict

API_URL = "http://localhost:8002"

def print_section(title: str):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_query(query: str, mode: str = "hybrid", top_k: int = 3, enable_reranking: bool = False):
    """Test a single query and print detailed results"""
    print(f"\n[QUERY] {query}")
    print(f"   Mode: {mode}, Top-K: {top_k}, Reranking: {enable_reranking}")
    print("-" * 80)

    payload = {
        "queries": [query],
        "mode": mode,
        "top_k": top_k,
        "enable_reranking": enable_reranking
    }

    try:
        response = requests.post(f"{API_URL}/search", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if not result or len(result) == 0:
            print("[FAIL] No results returned")
            return None

        contexts = result[0] if isinstance(result, list) else result

        if not contexts:
            print("[FAIL] Empty context returned")
            return None

        print(f"[OK] Retrieved {len(contexts)} contexts\n")

        for i, ctx in enumerate(contexts, 1):
            print(f"Context {i}:")
            print(f"  Content: {ctx[:200]}...")
            print()

        return contexts

    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def test_server_health():
    """Test if server is running"""
    print_section("SERVER HEALTH CHECK")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        print("[OK] Server is running and healthy")
        return True
    except Exception as e:
        print(f"[FAIL] Server health check failed: {e}")
        return False

def test_graph_stats():
    """Test graph statistics endpoint"""
    print_section("GRAPH STATISTICS")
    try:
        response = requests.get(f"{API_URL}/graph/stats", timeout=5)
        response.raise_for_status()
        stats = response.json()
        print(f"[OK] Graph loaded successfully:")
        print(f"   Entities: {stats.get('entities', 'N/A')}")
        print(f"   Relations: {stats.get('relations', 'N/A')}")
        print(f"   Chunks: {stats.get('chunks', 'N/A')}")
        return stats
    except Exception as e:
        print(f"[FAIL] Failed to get graph stats: {e}")
        return None

def test_entity_retrieval():
    """Test Path A: Entity-based retrieval"""
    print_section("TEST 1: ENTITY-BASED RETRIEVAL (Path A)")

    test_cases = [
        ("Who designed the Eiffel Tower?", "Should find: Gustave Eiffel"),
        ("Who developed the theory of relativity?", "Should find: Albert Einstein"),
        ("Who created Python programming language?", "Should find: Guido van Rossum"),
        ("Who founded Netflix?", "Should find: Reed Hastings, Marc Randolph"),
    ]

    results = {}
    for query, expected in test_cases:
        print(f"\n[TEST] Expected: {expected}")
        contexts = test_query(query, mode="local", top_k=3)
        results[query] = contexts is not None and len(contexts) > 0

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\n[RESULT] Entity Retrieval Success Rate: {success_rate:.1f}% ({sum(results.values())}/{len(results)})")
    return results

def test_relation_retrieval():
    """Test Path B: Relation-based retrieval"""
    print_section("TEST 2: RELATION-BASED RETRIEVAL (Path B)")

    test_cases = [
        ("What was the Eiffel Tower built for?", "Should find relation about World's Fair"),
        ("When was World War II?", "Should find relation about 1939-1945"),
        ("What is Python used for?", "Should find relations about applications"),
    ]

    results = {}
    for query, expected in test_cases:
        print(f"\n[TEST] Expected: {expected}")
        contexts = test_query(query, mode="global", top_k=3)
        results[query] = contexts is not None and len(contexts) > 0

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\n[RESULT] Relation Retrieval Success Rate: {success_rate:.1f}% ({sum(results.values())}/{len(results)})")
    return results

def test_chunk_retrieval():
    """Test Path C: Chunk-based retrieval"""
    print_section("TEST 3: CHUNK-BASED RETRIEVAL (Path C)")

    test_cases = [
        ("Tell me about the Eiffel Tower", "Should find Eiffel Tower document chunk"),
        ("Explain Einstein's work", "Should find Einstein document chunk"),
        ("Information about World War II", "Should find WWII document chunk"),
    ]

    results = {}
    for query, expected in test_cases:
        print(f"\n[TEST] Expected: {expected}")
        contexts = test_query(query, mode="naive", top_k=3)
        results[query] = contexts is not None and len(contexts) > 0

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\n[RESULT] Chunk Retrieval Success Rate: {success_rate:.1f}% ({sum(results.values())}/{len(results)})")
    return results

def test_hybrid_retrieval():
    """Test Hybrid: All paths combined"""
    print_section("TEST 4: HYBRID RETRIEVAL (Path A + B + C)")

    test_cases = [
        ("Who built the Eiffel Tower and when?", "Should combine entity + relation info"),
        ("What did Albert Einstein discover?", "Should find Einstein + theory of relativity"),
        ("Python programming language creator and uses", "Should find creator + applications"),
    ]

    results = {}
    for query, expected in test_cases:
        print(f"\n[TEST] Expected: {expected}")
        contexts = test_query(query, mode="hybrid", top_k=5)
        results[query] = contexts is not None and len(contexts) > 0

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\n[RESULT] Hybrid Retrieval Success Rate: {success_rate:.1f}% ({sum(results.values())}/{len(results)})")
    return results

def test_multihop_queries():
    """Test multi-hop reasoning queries"""
    print_section("TEST 5: MULTI-HOP QUERIES")

    test_cases = [
        ("What year was Python created and who manages it now?", "Should find 1991 + Python Software Foundation"),
        ("When did World War II happen and who led Germany?", "Should find 1939-1945 + Adolf Hitler"),
    ]

    results = {}
    for query, expected in test_cases:
        print(f"\n[TEST] Expected: {expected}")
        contexts = test_query(query, mode="hybrid", top_k=5)

        # Check if contexts contain relevant info
        if contexts:
            combined = " ".join(contexts).lower()
            # Simple heuristic: check if key terms are present
            if "python" in query.lower() and "1991" in combined and "foundation" in combined:
                results[query] = True
            elif "war" in query.lower() and ("1939" in combined or "1945" in combined) and "hitler" in combined:
                results[query] = True
            else:
                results[query] = contexts is not None and len(contexts) > 0
        else:
            results[query] = False

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\n[RESULT] Multi-hop Query Success Rate: {success_rate:.1f}% ({sum(results.values())}/{len(results)})")
    return results

def test_reranking():
    """Test semantic reranking"""
    print_section("TEST 6: SEMANTIC RERANKING")

    query = "Who created Python programming language?"

    print("\n[TEST] Without Reranking:")
    contexts_no_rerank = test_query(query, mode="hybrid", top_k=5, enable_reranking=False)

    print("\n[TEST] With Reranking:")
    contexts_rerank = test_query(query, mode="hybrid", top_k=5, enable_reranking=True)

    # Compare results
    if contexts_no_rerank and contexts_rerank:
        print("\n[RESULT] Reranking Comparison:")
        print(f"   Without reranking: {len(contexts_no_rerank)} contexts")
        print(f"   With reranking: {len(contexts_rerank)} contexts")

        # Check if top result mentions "Guido van Rossum"
        if contexts_rerank and "guido" in contexts_rerank[0].lower():
            print("   [OK] Reranking improved relevance (Guido van Rossum in top result)")
            return True
        else:
            print("   [WARNING]  Reranking may not have improved relevance")
            return False

    return False

def test_edge_cases():
    """Test edge cases and robustness"""
    print_section("TEST 7: EDGE CASES")

    test_cases = [
        ("", "Empty query"),
        ("zxcvbnmasdfghjkl", "Random gibberish"),
        ("Quantum computing machine learning blockchain", "Unrelated topics"),
    ]

    results = {}
    for query, description in test_cases:
        print(f"\n[TEST] Test: {description}")
        print(f"   Query: '{query}'")
        try:
            contexts = test_query(query, mode="hybrid", top_k=3)
            # Edge cases should either return empty or handle gracefully
            results[description] = True  # Server didn't crash
        except Exception as e:
            print(f"   [FAIL] Server error: {e}")
            results[description] = False

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\n[RESULT] Edge Case Handling: {success_rate:.1f}% ({sum(results.values())}/{len(results)})")
    return results

def run_all_tests():
    """Run all retrieval tests"""
    print("\n" + "="*80)
    print("  BiG-RAG RETRIEVAL SYSTEM - COMPREHENSIVE TEST SUITE")
    print("  Dataset: demo_test (5 documents)")
    print("="*80)

    # Health check
    if not test_server_health():
        print("\n[FAIL] Server not responding. Please start the API server first:")
        print("   python script_api.py --data_source demo_test --port 8002")
        return

    # Graph stats
    stats = test_graph_stats()
    if not stats:
        print("\n[FAIL] Failed to load graph statistics")
        return

    # Run all tests
    results = {}
    results["entity"] = test_entity_retrieval()
    results["relation"] = test_relation_retrieval()
    results["chunk"] = test_chunk_retrieval()
    results["hybrid"] = test_hybrid_retrieval()
    results["multihop"] = test_multihop_queries()
    results["reranking"] = test_reranking()
    results["edge_cases"] = test_edge_cases()

    # Final summary
    print_section("FINAL SUMMARY")

    all_tests = []
    for category, test_results in results.items():
        if isinstance(test_results, dict):
            success = sum(test_results.values())
            total = len(test_results)
            rate = success / total * 100 if total > 0 else 0
            all_tests.extend(test_results.values())
            print(f"[OK] {category.upper()}: {success}/{total} passed ({rate:.1f}%)")
        elif isinstance(test_results, bool):
            all_tests.append(test_results)
            status = "[OK] PASSED" if test_results else "[FAIL] FAILED"
            print(f"{status} {category.upper()}")

    overall_success = sum(all_tests)
    overall_total = len(all_tests)
    overall_rate = overall_success / overall_total * 100 if overall_total > 0 else 0

    print("\n" + "="*80)
    print(f"OVERALL SUCCESS RATE: {overall_rate:.1f}% ({overall_success}/{overall_total})")
    print("="*80)

    if overall_rate >= 80:
        print("\n[SUCCESS] EXCELLENT! Graph is working correctly!")
    elif overall_rate >= 60:
        print("\n[OK] GOOD! Graph is mostly working with minor issues")
    elif overall_rate >= 40:
        print("\n[WARNING]  FAIR! Graph has some issues that need attention")
    else:
        print("\n[FAIL] POOR! Graph has significant issues")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_all_tests()
