import requests
import json

# Test the API directly
url = "http://localhost:8002/search"
payload = {
    "queries": ["Who designed the Eiffel Tower?"],
    "mode": "local",
    "top_k": 3
}

print("Testing API...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print("="*80)

response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response Type: {type(response.json())}")
print("="*80)

result = response.json()
print(f"Result type: {type(result)}")
print(f"Result length: {len(result) if isinstance(result, (list, dict, str)) else 'N/A'}")
print("="*80)

if isinstance(result, list):
    print(f"List with {len(result)} items")
    if len(result) > 0:
        first_item = result[0]
        print(f"First item type: {type(first_item)}")
        print(f"First item length: {len(first_item) if isinstance(first_item, (list, dict, str)) else 'N/A'}")

        if isinstance(first_item, str):
            print(f"First item (string): {first_item[:200]}...")
        elif isinstance(first_item, list):
            print(f"First item is list with {len(first_item)} items")
            if len(first_item) > 0:
                print(f"First sub-item type: {type(first_item[0])}")
                print(f"First sub-item: {first_item[0][:200] if isinstance(first_item[0], str) else first_item[0]}")
        elif isinstance(first_item, dict):
            print(f"First item (dict keys): {list(first_item.keys())}")
            print(f"First item content: {json.dumps(first_item, indent=2)[:500]}")
