"""
Comprehensive retrieval test for all modes
Tests entity-based, relation-based, chunk-based, and hybrid retrieval
"""
import requests
import json

API_URL = "http://localhost:8002"

def test_retrieval(query, mode, expected_keywords, top_k=5):
    """Test a single query and check if expected keywords are found"""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Mode: {mode} | Top-K: {top_k}")
    print(f"Expected keywords: {expected_keywords}")
    print("-"*80)

    payload = {
        "queries": [query],
        "mode": mode,
        "top_k": top_k
    }

    try:
        response = requests.post(f"{API_URL}/search", json=payload, timeout=30)
        response.raise_for_status()
        result_list = response.json()

        if not result_list:
            print("[FAIL] No results returned")
            return False

        # Parse the JSON string
        result_str = result_list[0]
        result_dict = json.loads(result_str)

        results = result_dict.get('results', [])
        print(f"[OK] Retrieved {len(results)} results")

        # Combine all knowledge
        combined_knowledge = " ".join([r.get('<knowledge>', '') for r in results])

        # Show top 3 results
        print("\nTop 3 Results:")
        for i, r in enumerate(results[:3], 1):
            knowledge = r.get('<knowledge>', 'N/A')
            print(f"  {i}. {knowledge[:150]}...")

        # Check for expected keywords
        print("\nKeyword Check:")
        found_keywords = []
        missing_keywords = []
        for keyword in expected_keywords:
            if keyword.lower() in combined_knowledge.lower():
                found_keywords.append(keyword)
                print(f"  [OK] Found: '{keyword}'")
            else:
                missing_keywords.append(keyword)
                print(f"  [MISS] Not found: '{keyword}'")

        success = len(found_keywords) > 0
        print(f"\nResult: {'[OK] PASS' if success else '[FAIL] FAIL'} - Found {len(found_keywords)}/{len(expected_keywords)} keywords")
        return success

    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def run_comprehensive_tests():
    """Run all retrieval tests"""
    print("\n" + "="*80)
    print("  BiG-RAG COMPREHENSIVE RETRIEVAL TEST")
    print("  Dataset: demo_test (5 documents)")
    print("="*80)

    # Check server health
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        print("[OK] Server is running\n")
    except:
        print("[FAIL] Server not responding. Start it with:")
        print("  python script_api.py --data_source demo_test --port 8002")
        return

    results = {}

    # =============================================================================
    # TEST 1: Entity-Based Retrieval (Path A - "local" mode)
    # =============================================================================
    print("\n" + "#"*80)
    print("# TEST 1: ENTITY-BASED RETRIEVAL (Path A)")
    print("# Mode: 'local' - Searches entities in the graph")
    print("#"*80)

    results['entity_1'] = test_retrieval(
        "Who designed the Eiffel Tower?",
        "local",
        ["Gustave Eiffel", "engineer", "1887"]
    )

    results['entity_2'] = test_retrieval(
        "Who developed the theory of relativity?",
        "local",
        ["Albert Einstein", "physicist", "relativity"]
    )

    results['entity_3'] = test_retrieval(
        "Who created Python?",
        "local",
        ["Guido van Rossum", "Python", "1991"]
    )

    results['entity_4'] = test_retrieval(
        "Who founded Netflix?",
        "local",
        ["Reed Hastings", "Marc Randolph", "1997"]
    )

    # =============================================================================
    # TEST 2: Relation-Based Retrieval (Path B - "global" mode)
    # =============================================================================
    print("\n" + "#"*80)
    print("# TEST 2: RELATION-BASED RETRIEVAL (Path B)")
    print("# Mode: 'global' - Searches bipartite edge nodes (relations)")
    print("#"*80)

    results['relation_1'] = test_retrieval(
        "What was the purpose of the Eiffel Tower?",
        "global",
        ["World's Fair", "1889", "centennial"]
    )

    results['relation_2'] = test_retrieval(
        "When did World War II happen?",
        "global",
        ["1939", "1945", "global conflict"]
    )

    results['relation_3'] = test_retrieval(
        "What is Python used for?",
        "global",
        ["programming", "development", "web"]
    )

    # =============================================================================
    # TEST 3: Chunk-Based Retrieval (Path C - "naive" mode)
    # =============================================================================
    print("\n" + "#"*80)
    print("# TEST 3: CHUNK-BASED RETRIEVAL (Path C)")
    print("# Mode: 'naive' - Direct text chunk retrieval")
    print("#"*80)

    results['chunk_1'] = test_retrieval(
        "Tell me about the Eiffel Tower",
        "naive",
        ["Eiffel Tower", "wrought-iron", "Paris"]
    )

    results['chunk_2'] = test_retrieval(
        "Explain Einstein's contributions",
        "naive",
        ["Einstein", "theoretical physicist", "relativity"]
    )

    results['chunk_3'] = test_retrieval(
        "Netflix streaming service",
        "naive",
        ["Netflix", "streaming", "subscription"]
    )

    # =============================================================================
    # TEST 4: Hybrid Retrieval (Paths A + B + C - "hybrid" mode)
    # =============================================================================
    print("\n" + "#"*80)
    print("# TEST 4: HYBRID RETRIEVAL (All Paths Combined)")
    print("# Mode: 'hybrid' - Combines entity, relation, and chunk retrieval")
    print("#"*80)

    results['hybrid_1'] = test_retrieval(
        "Who built the Eiffel Tower and when?",
        "hybrid",
        ["Gustave Eiffel", "1887", "1889"]
    )

    results['hybrid_2'] = test_retrieval(
        "What is Albert Einstein known for?",
        "hybrid",
        ["Einstein", "relativity", "physicist"]
    )

    results['hybrid_3'] = test_retrieval(
        "Python programming language history",
        "hybrid",
        ["Guido van Rossum", "1991", "programming"]
    )

    # =============================================================================
    # TEST 5: Multi-Hop Queries
    # =============================================================================
    print("\n" + "#"*80)
    print("# TEST 5: MULTI-HOP QUERIES")
    print("# Testing complex queries that require multiple pieces of information")
    print("#"*80)

    results['multihop_1'] = test_retrieval(
        "When was Python created and what is it used for?",
        "hybrid",
        ["1991", "Guido van Rossum", "web development", "data analysis"]
    )

    results['multihop_2'] = test_retrieval(
        "World War II duration and leader of Germany",
        "hybrid",
        ["1939", "1945", "Adolf Hitler"]
    )

    # =============================================================================
    # TEST 6: Edge Cases
    # =============================================================================
    print("\n" + "#"*80)
    print("# TEST 6: EDGE CASES")
    print("# Testing robustness with difficult queries")
    print("#"*80)

    # Query about something not in the data
    print(f"\n{'='*80}")
    print(f"Query: Who invented the telephone?")
    print(f"Mode: hybrid | Top-K: 5")
    print(f"Expected: Should return empty or unrelated results (not in dataset)")
    print("-"*80)

    payload = {"queries": ["Who invented the telephone?"], "mode": "hybrid", "top_k": 5}
    response = requests.post(f"{API_URL}/search", json=payload)
    result_list = response.json()
    result_dict = json.loads(result_list[0])
    results_count = len(result_dict.get('results', []))
    print(f"Retrieved {results_count} results")
    if results_count == 0:
        print("[OK] Correctly returned no results for out-of-domain query")
        results['edge_case_1'] = True
    else:
        combined = " ".join([r.get('<knowledge>', '') for r in result_dict.get('results', [])])
        if "telephone" in combined.lower():
            print("[WARNING] Found 'telephone' but shouldn't (not in dataset)")
            results['edge_case_1'] = False
        else:
            print("[OK] Returned some results but they don't mention telephone (acceptable)")
            results['edge_case_1'] = True

    # =============================================================================
    # SUMMARY
    # =============================================================================
    print("\n" + "="*80)
    print("  FINAL RESULTS")
    print("="*80)

    # Calculate success rate by category
    categories = {
        'Entity-Based': [k for k in results.keys() if k.startswith('entity_')],
        'Relation-Based': [k for k in results.keys() if k.startswith('relation_')],
        'Chunk-Based': [k for k in results.keys() if k.startswith('chunk_')],
        'Hybrid': [k for k in results.keys() if k.startswith('hybrid_')],
        'Multi-Hop': [k for k in results.keys() if k.startswith('multihop_')],
        'Edge Cases': [k for k in results.keys() if k.startswith('edge_case_')]
    }

    print()
    for category, keys in categories.items():
        if not keys:
            continue
        passed = sum([results[k] for k in keys if k in results])
        total = len(keys)
        percentage = (passed / total * 100) if total > 0 else 0
        status = "[OK]" if percentage >= 75 else ("[WARNING]" if percentage >= 50 else "[FAIL]")
        print(f"{status} {category:20s}: {passed}/{total} passed ({percentage:.1f}%)")

    # Overall
    total_passed = sum(results.values())
    total_tests = len(results)
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print()
    print("="*80)
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({overall_percentage:.1f}%)")
    print("="*80)

    if overall_percentage >= 80:
        print("\n[SUCCESS] Excellent! Graph is working correctly!")
        print("The bipartite graph structure is properly built and retrieval is functional.")
    elif overall_percentage >= 60:
        print("\n[OK] Good! Graph is mostly working with minor issues.")
    else:
        print("\n[WARNING] Graph has some issues that need investigation.")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_comprehensive_tests()
