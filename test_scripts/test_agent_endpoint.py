"""
Test script for the agent endpoint.

Tests the multi-hop reasoning agent with various query types.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import time


def test_agent_health():
    """Test agent health endpoint."""
    print("[TEST] Checking agent health...")
    response = requests.get("http://localhost:8001/agent/health")

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Agent status: {data['status']}")
        print(f"[OK] Model: {data.get('model', 'N/A')}")
        return data['ready']
    else:
        print(f"[FAIL] Health check failed: {response.status_code}")
        return False


def test_agent_info():
    """Test agent info endpoint."""
    print("\n[TEST] Getting agent info...")
    response = requests.get("http://localhost:8001/agent/info")

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Agent name: {data['name']}")
        print(f"[OK] Capabilities: {len(data['capabilities'])}")
        print(f"[OK] Supported languages: {len(data['supported_languages'])}")
        return True
    else:
        print(f"[FAIL] Info request failed: {response.status_code}")
        return False


def test_simple_query():
    """Test with a simple, single-hop query."""
    print("\n[TEST] Testing simple query...")

    payload = {
        "question": "What is Bangladesh?",
        "max_iterations": 1,
        "agent_model": "gpt-4o-mini",  # Use cheaper model for testing
        "top_k_per_query": 3
    }

    start_time = time.time()
    response = requests.post(
        "http://localhost:8001/agent/query",
        json=payload,
        timeout=60
    )
    execution_time = time.time() - start_time

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Simple query completed in {execution_time:.1f}s")
        print(f"[OK] Answer: {data['answer'][:200]}...")
        print(f"[OK] Iterations: {data['total_iterations']}")
        print(f"[OK] Queries executed: {data['metadata']['queries_executed']}")
        print(f"[OK] Confidence: {data['confidence']:.2f}")
        return True
    else:
        print(f"[FAIL] Query failed: {response.status_code}")
        print(f"[FAIL] Error: {response.text}")
        return False


def test_sequential_multihop():
    """Test sequential multi-hop reasoning (like Scenario 1)."""
    print("\n[TEST] Testing sequential multi-hop query...")

    # Note: This is a hypothetical example
    # Actual results depend on your knowledge graph content
    payload = {
        "question": "Who is the captain of the 2022 World Cup winner?",
        "max_iterations": 3,
        "agent_model": "gpt-4o-mini",
        "enable_parallel": False,  # Force sequential
        "top_k_per_query": 5
    }

    start_time = time.time()
    response = requests.post(
        "http://localhost:8001/agent/query",
        json=payload,
        timeout=120
    )
    execution_time = time.time() - start_time

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Multi-hop query completed in {execution_time:.1f}s")
        print(f"[OK] Answer: {data['answer'][:200]}...")
        print(f"[OK] Iterations: {data['total_iterations']}")
        print(f"[OK] Total queries: {data['metadata']['queries_executed']}")

        # Print reasoning trace
        print("\n[OK] Reasoning trace:")
        for step in data['reasoning_trace']:
            print(f"  Step {step['step']}: {step['thought'][:100]}...")
            print(f"    Queries: {len(step['planned_queries'])}")
            print(f"    Confidence: {step['confidence']:.2f}")

        return True
    else:
        print(f"[FAIL] Query failed: {response.status_code}")
        print(f"[FAIL] Error: {response.text}")
        return False


def test_parallel_multihop():
    """Test parallel multi-hop reasoning (like Scenario 2)."""
    print("\n[TEST] Testing parallel multi-hop query...")

    payload = {
        "question": "Compare Bangladesh and India",
        "max_iterations": 2,
        "agent_model": "gpt-4o-mini",
        "enable_parallel": True,
        "top_k_per_query": 5
    }

    start_time = time.time()
    response = requests.post(
        "http://localhost:8001/agent/query",
        json=payload,
        timeout=120
    )
    execution_time = time.time() - start_time

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Parallel query completed in {execution_time:.1f}s")
        print(f"[OK] Answer: {data['answer'][:300]}...")
        print(f"[OK] Iterations: {data['total_iterations']}")
        print(f"[OK] Total queries: {data['metadata']['queries_executed']}")
        print(f"[OK] Stop reason: {data['metadata']['stopped_reason']}")

        return True
    else:
        print(f"[FAIL] Query failed: {response.status_code}")
        print(f"[FAIL] Error: {response.text}")
        return False


def test_multilingual():
    """Test multilingual query."""
    print("\n[TEST] Testing multilingual query...")

    payload = {
        "question": "বাংলাদেশের রাজধানী কোথায়?",  # "Where is the capital of Bangladesh?"
        "language": "Bangla",
        "max_iterations": 1,
        "agent_model": "gpt-4o-mini",
        "top_k_per_query": 3
    }

    start_time = time.time()
    response = requests.post(
        "http://localhost:8001/agent/query",
        json=payload,
        timeout=60
    )
    execution_time = time.time() - start_time

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Multilingual query completed in {execution_time:.1f}s")
        print(f"[OK] Answer: {data['answer'][:200]}...")
        return True
    else:
        print(f"[FAIL] Query failed: {response.status_code}")
        print(f"[FAIL] Error: {response.text}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Agent Endpoint Test Suite")
    print("="*60)

    # Check if server is running
    print("\n[INFO] Checking if server is running...")
    try:
        response = requests.get("http://localhost:8001/docs", timeout=5)
        if response.status_code != 200:
            print("[ERROR] Server not responding. Please start the server first:")
            print("  cd backend")
            print("  python server.py --data_source SingleTopic")
            return
    except requests.exceptions.RequestException:
        print("[ERROR] Cannot connect to server. Please start it first:")
        print("  cd backend")
        print("  python server.py --data_source SingleTopic")
        return

    results = []

    # Run tests
    results.append(("Health Check", test_agent_health()))
    results.append(("Agent Info", test_agent_info()))

    # Only run query tests if agent is ready
    if results[0][1]:
        results.append(("Simple Query", test_simple_query()))
        results.append(("Sequential Multi-hop", test_sequential_multihop()))
        results.append(("Parallel Multi-hop", test_parallel_multihop()))
        # results.append(("Multilingual Query", test_multilingual()))  # Uncomment if you have multilingual data
    else:
        print("\n[WARN] Agent not ready, skipping query tests")

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
