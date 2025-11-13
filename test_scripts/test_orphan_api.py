"""
Test script for orphan node API enhancements.

Before running:
1. Start backend: cd backend && python server.py --data_source football
2. Run this script: python test_scripts/test_orphan_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8001"

def test_normal_mode():
    """Test with include_all_orphans=false (default)"""
    print("\n" + "="*70)
    print("[TEST 1: Normal Mode - include_all_orphans=false]")
    print("="*70)

    response = requests.get(f"{BASE_URL}/graph/export", params={
        "data_source": "football",
        "limit": 1000,
        "include_all_orphans": False
    })

    data = response.json()
    orphan_breakdown = data.get("orphan_breakdown", {})

    print(f"\nOrphan Breakdown:")
    print(f"  Total orphan nodes in graph: {orphan_breakdown.get('total', 0)}")
    print(f"    - Orphan entities: {orphan_breakdown.get('entities', 0)}")
    print(f"    - Orphan relations: {orphan_breakdown.get('relations', 0)}")
    print(f"    - Orphan chunks: {orphan_breakdown.get('chunks', 0)}")
    print(f"  Orphan nodes included in response: {orphan_breakdown.get('included_in_response', 0)}")
    print(f"  Include all orphans mode: {orphan_breakdown.get('include_all_orphans_mode', False)}")

    print(f"\nResponse Summary:")
    print(f"  Total nodes returned: {len(data.get('nodes', []))}")
    print(f"  Total edges returned: {len(data.get('edges', []))}")

    return data

def test_debug_mode():
    """Test with include_all_orphans=true"""
    print("\n" + "="*70)
    print("[TEST 2: Debug Mode - include_all_orphans=true]")
    print("="*70)

    response = requests.get(f"{BASE_URL}/graph/export", params={
        "data_source": "football",
        "limit": 1000,
        "include_all_orphans": True
    })

    data = response.json()
    orphan_breakdown = data.get("orphan_breakdown", {})

    print(f"\nOrphan Breakdown:")
    print(f"  Total orphan nodes in graph: {orphan_breakdown.get('total', 0)}")
    print(f"    - Orphan entities: {orphan_breakdown.get('entities', 0)}")
    print(f"    - Orphan relations: {orphan_breakdown.get('relations', 0)}")
    print(f"    - Orphan chunks: {orphan_breakdown.get('chunks', 0)}")
    print(f"  Orphan nodes included in response: {orphan_breakdown.get('included_in_response', 0)}")
    print(f"  Include all orphans mode: {orphan_breakdown.get('include_all_orphans_mode', False)}")

    print(f"\nResponse Summary:")
    print(f"  Total nodes returned: {len(data.get('nodes', []))}")
    print(f"  Total edges returned: {len(data.get('edges', []))}")

    # List all orphan nodes
    orphan_nodes = [n for n in data.get('nodes', []) if n.get('connections', 0) == 0]
    print(f"\n[ORPHAN NODES IN RESPONSE]")
    for i, node in enumerate(orphan_nodes[:20], 1):  # Show first 20
        print(f"  {i}. [{node['type']}] {node['label'][:60]}... (weight: {node['weight']})")

    if len(orphan_nodes) > 20:
        print(f"  ... and {len(orphan_nodes) - 20} more")

    return data

def test_large_graph_simulation():
    """Test with SingleTopic dataset (larger graph)"""
    print("\n" + "="*70)
    print("[TEST 3: Large Graph - SingleTopic dataset]")
    print("="*70)

    try:
        # Test normal mode
        response = requests.get(f"{BASE_URL}/graph/export", params={
            "data_source": "SingleTopic",
            "limit": 500,  # Reduced limit to force sampling
            "include_all_orphans": False
        })

        data = response.json()
        orphan_breakdown = data.get("orphan_breakdown", {})

        print(f"\n[Normal Mode]")
        print(f"  Total orphan nodes: {orphan_breakdown.get('total', 0)}")
        print(f"  Orphan nodes included (20% cap): {orphan_breakdown.get('included_in_response', 0)}")
        print(f"  Total nodes returned: {len(data.get('nodes', []))}")

        # Test debug mode
        response2 = requests.get(f"{BASE_URL}/graph/export", params={
            "data_source": "SingleTopic",
            "limit": 500,
            "include_all_orphans": True
        })

        data2 = response2.json()
        orphan_breakdown2 = data2.get("orphan_breakdown", {})

        print(f"\n[Debug Mode - include_all_orphans=true]")
        print(f"  Total orphan nodes: {orphan_breakdown2.get('total', 0)}")
        print(f"  Orphan nodes included (ALL): {orphan_breakdown2.get('included_in_response', 0)}")
        print(f"  Total nodes returned: {len(data2.get('nodes', []))}")

        print(f"\n[COMPARISON]")
        print(f"  Normal mode orphans: {orphan_breakdown.get('included_in_response', 0)}")
        print(f"  Debug mode orphans: {orphan_breakdown2.get('included_in_response', 0)}")
        print(f"  Difference: +{orphan_breakdown2.get('included_in_response', 0) - orphan_breakdown.get('included_in_response', 0)}")

    except Exception as e:
        print(f"  SingleTopic dataset not available or error: {e}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ORPHAN NODE API TESTING")
    print("="*70)

    try:
        # Test 1: Normal mode with football
        test_normal_mode()

        # Test 2: Debug mode with football
        test_debug_mode()

        # Test 3: Large graph (if available)
        test_large_graph_simulation()

        print("\n" + "="*70)
        print("[ALL TESTS COMPLETED]")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to backend server!")
        print("Please start the backend server first:")
        print("  cd backend && python server.py --data_source football")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
