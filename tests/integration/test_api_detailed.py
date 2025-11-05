import requests
import json

# Test the API and parse properly
url = "http://localhost:8002/search"
payload = {
    "queries": ["Who designed the Eiffel Tower?"],
    "mode": "local",
    "top_k": 5
}

print("Testing API Retrieval...")
print("="*80)

response = requests.post(url, json=payload)
result = response.json()

# The API returns a list with one item per query
# Each item is a JSON string that needs to be parsed
print(f"Response is a list with {len(result)} items")
first_result_str = result[0]
print(f"First item is a string of length {len(first_result_str)}")
print("="*80)

# Parse the JSON string
first_result = json.loads(first_result_str)
print(f"Parsed result type: {type(first_result)}")
print(f"Parsed result keys: {list(first_result.keys())}")
print("="*80)

# Show the query
print(f"Query: {first_result['query']}")
print("="*80)

# Show the results
results_list = first_result['results']
print(f"Retrieved {len(results_list)} results")
print("="*80)

for i, result_item in enumerate(results_list[:5], 1):
    print(f"\nResult {i}:")
    print(f"  Knowledge: {result_item.get('<knowledge>', 'N/A')[:200]}...")
    print(f"  Coherence: {result_item.get('<coherence>', 'N/A')}")
    print(f"  Source IDs: {result_item.get('<source_ids>', 'N/A')}")
    print()

print("="*80)
print("ASSESSMENT:")
print("="*80)

# Check if "Gustave Eiffel" is mentioned in any result
combined_knowledge = " ".join([r.get('<knowledge>', '') for r in results_list])
if "gustave eiffel" in combined_knowledge.lower():
    print("[OK] Found 'Gustave Eiffel' in retrieval results!")
else:
    print("[WARNING] 'Gustave Eiffel' not found in retrieval results")

if "eiffel" in combined_knowledge.lower():
    print("[OK] Found 'Eiffel' in retrieval results")
else:
    print("[FAIL] 'Eiffel' not found at all")

print("="*80)
