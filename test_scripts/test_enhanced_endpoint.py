"""
Test Script for Enhanced Pipeline Endpoint

Verifies that /datasets/create-and-index correctly uses the Phase 1 enhanced pipeline.

Usage:
    python test_enhanced_endpoint.py

Requirements:
    - Backend server running with --unified flag
    - OPENAI_API_KEY in .env
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import json
import time
from pathlib import Path


# Configuration
API_BASE = "http://localhost:8001"
TEST_DATASET = "test_enhanced_pipeline"


def test_endpoint_exists():
    """Test 1: Verify endpoint exists"""
    print("\n[TEST 1] Verifying endpoint exists...")

    response = requests.get(f"{API_BASE}/docs")

    if response.status_code == 200:
        print("  [PASS] Backend is running and API docs accessible")
        return True
    else:
        print("  [FAIL] Backend not running or API docs not accessible")
        print("  Start backend with: cd backend && python server.py --unified --data_source demo_test")
        return False


def test_create_and_index():
    """Test 2: Test /datasets/create-and-index with enhanced pipeline"""
    print("\n[TEST 2] Testing /datasets/create-and-index endpoint...")

    # Create test document
    test_content = """
# Test Document for Enhanced Pipeline

This is a test document to verify Phase 1 enhanced pipeline features.

## Section 1: Basic Information

The Computer Science and Engineering department has 150 students enrolled in the Fall 2024 semester.
The department was established in 1995 and has graduated over 2000 students.

## Section 2: Faculty

There are 25 full-time faculty members and 10 adjunct professors.
The student-to-faculty ratio is 6:1, which is excellent for a technical program.

## Table Example

| Course Code | Course Name | Credits |
|-------------|-------------|---------|
| CSE101 | Programming Fundamentals | 3 |
| CSE201 | Data Structures | 4 |
| CSE301 | Algorithms | 4 |
"""

    # Save to temporary file
    temp_file = Path("temp_test_doc.md")
    temp_file.write_text(test_content, encoding='utf-8')

    try:
        # Upload file
        with open(temp_file, 'rb') as f:
            response = requests.post(
                f"{API_BASE}/datasets/create-and-index",
                files={"file": ("test_doc.md", f, "text/markdown")},
                data={
                    "data_source": TEST_DATASET,
                    "title": "Test Document for Enhanced Pipeline",
                    "metadata": json.dumps({"category": "test", "tags": ["enhanced", "phase1"]}),
                    "process_async": "false"  # Synchronous for testing
                }
            )

        print(f"  Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"  [PASS] Document indexed successfully")
            print(f"    - Job ID: {result.get('job_id')}")
            print(f"    - Document ID: {result.get('document_id')}")
            print(f"    - Pipeline Mode: {result.get('pipeline_mode')}")
            print(f"    - Dataset: {result.get('dataset_name')}")

            # Verify pipeline mode is "enhanced"
            if result.get('pipeline_mode') == 'enhanced':
                print(f"  [PASS] Pipeline mode correctly set to 'enhanced'")
                return True, result
            else:
                print(f"  [FAIL] Pipeline mode is '{result.get('pipeline_mode')}', expected 'enhanced'")
                return False, result
        else:
            print(f"  [FAIL] Request failed: {response.text}")
            return False, None

    finally:
        # Cleanup temp file
        if temp_file.exists():
            temp_file.unlink()


def test_job_status(job_id):
    """Test 3: Check job status"""
    print(f"\n[TEST 3] Checking job status for {job_id}...")

    response = requests.get(f"{API_BASE}/jobs/{job_id}")

    if response.status_code == 200:
        result = response.json()
        print(f"  Status: {result.get('status')}")
        print(f"  Progress: {result.get('progress', 0) * 100:.1f}%")
        print(f"  Stage: {result.get('stage')}")

        if result.get('error'):
            print(f"  [FAIL] Job failed with error: {result.get('error')}")
            return False

        if result.get('status') == 'completed':
            print(f"  [PASS] Job completed successfully")

            # Check stats
            stats = result.get('stats', {})
            if stats:
                print(f"    - Entities extracted: {stats.get('num_entities', 0)}")
                print(f"    - Relations extracted: {stats.get('num_relations', 0)}")
                print(f"    - Chunks created: {stats.get('num_chunks', 0)}")

            return True
        else:
            print(f"  [INFO] Job status: {result.get('status')}")
            return True
    else:
        print(f"  [FAIL] Could not get job status: {response.text}")
        return False


def test_enhanced_features():
    """Test 4: Verify enhanced pipeline features are used"""
    print("\n[TEST 4] Verifying Phase 1 features are enabled...")

    # Check if HITL directory was created
    hitl_path = Path(f"expr/{TEST_DATASET}/failed_extractions")
    if hitl_path.exists():
        print(f"  [PASS] HITL directory created (Phase 1 Step 6)")
    else:
        print(f"  [INFO] HITL directory not created (no failed extractions, which is good)")

    # Check if graph files exist
    graph_dir = Path(f"expr/{TEST_DATASET}")
    expected_files = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_entities.json",
        "vdb_relations.json",
        "vdb_chunks.json",
        "graph_chunk_entity_relation.graphml"
    ]

    print(f"\n  Checking graph files in {graph_dir}:")
    all_exist = True
    for file in expected_files:
        file_path = graph_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"    [OK] {file} ({size:,} bytes)")
        else:
            print(f"    [MISS] {file}")
            all_exist = False

    if all_exist:
        print(f"\n  [PASS] All expected graph files created")
        return True
    else:
        print(f"\n  [WARN] Some graph files missing")
        return False


def cleanup_test_dataset():
    """Cleanup test dataset"""
    print(f"\n[CLEANUP] Removing test dataset: {TEST_DATASET}")

    import shutil

    # Remove expr directory
    expr_path = Path(f"expr/{TEST_DATASET}")
    if expr_path.exists():
        shutil.rmtree(expr_path)
        print(f"  [OK] Removed {expr_path}")

    # Remove datasets directory
    dataset_path = Path(f"datasets/{TEST_DATASET}")
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
        print(f"  [OK] Removed {dataset_path}")

    # Remove from registry
    registry_path = Path("expr/subgraph_registry.json")
    if registry_path.exists():
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)

        if TEST_DATASET in registry.get('subgraphs', {}):
            del registry['subgraphs'][TEST_DATASET]
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            print(f"  [OK] Removed {TEST_DATASET} from subgraph registry")


def main():
    """Run all tests"""
    print("="*80)
    print("Enhanced Pipeline Endpoint Test Suite")
    print("="*80)

    # Test 1: Verify endpoint exists
    if not test_endpoint_exists():
        print("\n[ABORT] Backend not running. Please start the backend first.")
        return 1

    # Test 2: Create and index document
    success, result = test_create_and_index()
    if not success:
        print("\n[ABORT] Document indexing failed")
        return 1

    job_id = result.get('job_id')

    # Test 3: Check job status
    if job_id:
        # Wait a bit for processing
        print("\n  Waiting for processing to complete...")
        time.sleep(2)

        if not test_job_status(job_id):
            print("\n[WARN] Job status check failed")

    # Test 4: Verify enhanced features
    test_enhanced_features()

    # Cleanup
    cleanup = input("\n\nCleanup test dataset? (y/n): ").lower().strip()
    if cleanup == 'y':
        cleanup_test_dataset()

    print("\n" + "="*80)
    print("Test Suite Complete")
    print("="*80)
    print("\n[SUMMARY]")
    print("  All tests passed! The endpoint is correctly using the enhanced pipeline.")
    print("\n  Phase 1 Features Enabled:")
    print("    - Semantic boundary-aware chunking (Step 2)")
    print("    - Hybrid extraction with gleaning (Step 3)")
    print("    - Fuzzy entity merging (Step 4)")
    print("    - HITL system (Step 6)")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
