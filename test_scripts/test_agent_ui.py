"""
Test Agent UI endpoint with exact frontend request format.

This script mimics what the frontend sends to diagnose 422 errors.
"""

import requests
import json

# Test payload (matching frontend defaults)
payload = {
    "question": "Who is the captain of the team that won the 2022 FIFA World Cup?",
    "language": "auto",
    "max_iterations": 3,
    "agent_model": "gpt-4o",
    "enable_parallel": True,
    "top_k_per_query": 60,
    "num_kg_in_context": 15,
    "num_chunks_in_context": 5,
    "enable_reranking": False,
    "enable_variable_storage": True,
    "confidence_threshold": 0.8
}

print("Testing /agent/query endpoint...")
print("\nPayload:")
print(json.dumps(payload, indent=2))

url = "http://localhost:8001/agent/query"

try:
    response = requests.post(url, json=payload, timeout=60)

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 422:
        print("\nValidation Error:")
        print(json.dumps(response.json(), indent=2))
    elif response.status_code == 200:
        print("\nSuccess!")
        result = response.json()
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Iterations: {result['total_iterations']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"\nUnexpected status code: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\nERROR: Could not connect to backend. Is the server running on port 8001?")
except Exception as e:
    print(f"\nERROR: {e}")
