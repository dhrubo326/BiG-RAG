"""
Test query that spans multiple documents to verify indirect chunk retrieval
"""
import requests
import json

API_URL = "http://localhost:8002"

def test_multi_document_query():
    """Test a query that should retrieve from multiple documents"""

    # Query mentioning both Eiffel Tower AND Netflix
    query = "Who designed the Eiffel Tower and who founded Netflix?"

    print("="*80)
    print("TESTING MULTI-DOCUMENT QUERY")
    print("="*80)
    print(f"Query: {query}")
    print()
    print("Expected: Should find entities from BOTH Eiffel Tower doc AND Netflix doc")
    print("          Should return 2+ chunks (one from each document)")
    print("="*80)

    payload = {
        "queries": [query],
        "mode": "hybrid",
        "top_k": 5
    }

    response = requests.post(f"{API_URL}/search", json=payload)
    result_list = response.json()
    result_dict = json.loads(result_list[0])
    results = result_dict.get('results', [])

    print(f"\nTotal results: {len(results)}")
    print()

    # Analyze by type and source
    chunks_by_source = {}
    entities_by_source = {}

    for r in results:
        knowledge = r.get('<knowledge>', '')
        source_ids = r.get('<source_ids>', [])
        rtype = r.get('<type>', 'unknown')

        if 'chunk' in rtype:
            for sid in source_ids:
                if sid not in chunks_by_source:
                    chunks_by_source[sid] = []
                chunks_by_source[sid].append(knowledge[:80])
        else:
            for sid in source_ids:
                if sid not in entities_by_source:
                    entities_by_source[sid] = []
                entities_by_source[sid].append((rtype, knowledge[:80]))

    print("ENTITIES BY SOURCE DOCUMENT:")
    print("-"*80)
    for sid, entities in entities_by_source.items():
        print(f"\n{sid}:")
        for etype, content in entities[:3]:
            print(f"  [{etype}] {content}...")

    print("\n")
    print("CHUNKS BY SOURCE DOCUMENT:")
    print("-"*80)
    for sid, chunks in chunks_by_source.items():
        print(f"\n{sid}:")
        for chunk in chunks:
            print(f"  {chunk}...")

    print("\n")
    print("="*80)
    print("ANALYSIS:")
    print("="*80)
    print(f"Unique source documents (from entities): {len(entities_by_source)}")
    print(f"Unique source documents (from chunks): {len(chunks_by_source)}")
    print()

    if len(chunks_by_source) >= 2:
        print("[OK] Multiple chunks returned from different documents!")
        print("     Indirect chunk retrieval is working correctly.")
    elif len(chunks_by_source) == 1:
        print("[INFO] Only 1 chunk returned")
        if len(entities_by_source) >= 2:
            print("[ISSUE] But entities came from 2+ documents!")
            print("        Expected indirect chunks to fill the gap.")
        else:
            print("[OK] Entities also came from 1 document (expected)")
    else:
        print("[ISSUE] No chunks returned!")

    print("="*80)

if __name__ == "__main__":
    test_multi_document_query()
