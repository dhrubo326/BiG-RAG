"""
Debug script to trace why indirect chunks are not being retrieved
"""
import requests
import json

API_URL = "http://localhost:8002"

def test_chunk_retrieval_debug():
    """Test to understand chunk retrieval behavior"""

    query = "When was Python created and what is it used for?"

    print("="*80)
    print("DEBUGGING CHUNK RETRIEVAL")
    print("="*80)
    print(f"Query: {query}")
    print()

    # Test with hybrid mode (should return entities + relations + chunks)
    payload = {
        "queries": [query],
        "mode": "hybrid",
        "top_k": 5
    }

    response = requests.post(f"{API_URL}/search", json=payload)
    result_list = response.json()
    result_dict = json.loads(result_list[0])
    results = result_dict.get('results', [])

    print(f"Total results: {len(results)}")
    print()

    # Separate by type
    entity_results = []
    relation_results = []
    chunk_results = []

    for r in results:
        knowledge = r.get('<knowledge>', '')
        source_ids = r.get('<source_ids>', [])
        rtype = r.get('<type>', 'unknown')

        if rtype == 'entity':
            entity_results.append((knowledge, source_ids))
        elif rtype == 'bipartite_edge':
            relation_results.append((knowledge, source_ids))
        elif 'chunk' in rtype:
            chunk_results.append((knowledge, source_ids))

    print("BREAKDOWN BY TYPE:")
    print("-"*80)
    print(f"Entity results (Path A): {len(entity_results)}")
    for i, (k, s) in enumerate(entity_results[:3], 1):
        print(f"  {i}. {k[:80]}...")
        print(f"     Source IDs: {s}")

    print()
    print(f"Relation results (Path B): {len(relation_results)}")
    for i, (k, s) in enumerate(relation_results[:3], 1):
        print(f"  {i}. {k[:80]}...")
        print(f"     Source IDs: {s}")

    print()
    print(f"Chunk results (Path C): {len(chunk_results)}")
    for i, (k, s) in enumerate(chunk_results, 1):
        print(f"  {i}. {k[:80]}...")
        print(f"     Source IDs: {s}")

    print()
    print("="*80)
    print("ANALYSIS:")
    print("="*80)

    # Collect all source IDs from entities and relations
    all_entity_source_ids = set()
    for _, source_ids in entity_results:
        if source_ids:
            all_entity_source_ids.update(source_ids)

    all_relation_source_ids = set()
    for _, source_ids in relation_results:
        if source_ids:
            all_relation_source_ids.update(source_ids)

    combined_source_ids = all_entity_source_ids.union(all_relation_source_ids)

    print(f"Entity source IDs collected: {len(all_entity_source_ids)}")
    print(f"  IDs: {list(all_entity_source_ids)[:5]}")
    print()
    print(f"Relation source IDs collected: {len(all_relation_source_ids)}")
    print(f"  IDs: {list(all_relation_source_ids)[:5]}")
    print()
    print(f"Combined unique source IDs: {len(combined_source_ids)}")
    print(f"  IDs: {list(combined_source_ids)[:5]}")
    print()

    # Check chunk source IDs
    chunk_source_ids = set()
    for _, source_ids in chunk_results:
        if source_ids:
            chunk_source_ids.update(source_ids)

    print(f"Chunk source IDs returned: {len(chunk_source_ids)}")
    print(f"  IDs: {list(chunk_source_ids)}")
    print()

    # Check overlap
    overlap = combined_source_ids.intersection(chunk_source_ids)
    print(f"Overlap between entity/relation sources and chunk sources: {len(overlap)}")
    print(f"  Overlapping IDs: {list(overlap)}")
    print()

    # Check if there are source IDs that should be retrieved but aren't
    missing = combined_source_ids - chunk_source_ids
    print(f"Source IDs from entities/relations NOT returned as chunks: {len(missing)}")
    if missing:
        print(f"  Missing chunk IDs: {list(missing)[:5]}")
        print()
        print("  [ISSUE] These chunks should have been retrieved via indirect path!")
        print("  The system found entities/relations from these chunks,")
        print("  but did not retrieve the actual chunk content.")
    else:
        print("  [OK] All source chunks are present")

    print()
    print("="*80)
    print("EXPECTED BEHAVIOR:")
    print("="*80)
    print("If entities are found from chunk-123, then:")
    print("  1. Path A returns entities with source_ids=['chunk-123']")
    print("  2. Path C should retrieve chunk-123 content (indirect)")
    print("  3. Result should include the full chunk text")
    print()
    print(f"In your case:")
    print(f"  - Found {len(entity_results)} entities")
    print(f"  - Found {len(relation_results)} relations")
    print(f"  - Those came from {len(combined_source_ids)} unique chunks")
    print(f"  - But only {len(chunk_source_ids)} chunks were returned")
    print(f"  - Missing {len(missing)} chunks that should be in Path C")
    print()

    if len(missing) > 0:
        print("[DIAGNOSIS] Indirect chunk retrieval is NOT working properly!")
        print("The system is finding entities but not returning their source chunks.")
    else:
        print("[DIAGNOSIS] Chunk retrieval is working correctly!")

    print("="*80)

if __name__ == "__main__":
    test_chunk_retrieval_debug()
