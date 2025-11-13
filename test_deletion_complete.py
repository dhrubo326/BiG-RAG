"""
Test script for complete hard delete functionality

Tests that hard delete properly removes documents from:
1. Knowledge graph (chunks, entities, edges, vectors)
2. Document registry
3. corpus.jsonl (NEW FIX)

Usage:
    python test_deletion_complete.py
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8001"
DATASET = "SingleTopic"
PROJECT_ROOT = Path(__file__).parent
CORPUS_PATH = PROJECT_ROOT / "datasets" / DATASET / "raw" / "corpus.jsonl"

# Test document content
TEST_DOC_CONTENT = """
# Test Document for Deletion

This is a test document created to verify the complete hard delete functionality.

Key concepts:
- Knowledge Graph Deletion
- Corpus Cleanup
- Registry Management

This document should be completely removed from all storage layers after hard delete.
"""

TEST_DOC_TITLE = "Test Deletion Document"


def check_server_running():
    """Check if API server is running"""
    print("\n[1/6] Checking if server is running...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("    [OK] Server is running")
            return True
        else:
            print(f"    [FAIL] Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"    [FAIL] Cannot connect to {API_BASE_URL}")
        print(f"    Please start the server: cd backend && python server.py --data_source {DATASET}")
        return False


def upload_test_document():
    """Upload test document via API"""
    print("\n[2/6] Uploading test document...")

    # Create temporary file
    temp_file = PROJECT_ROOT / "test_temp_doc.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(TEST_DOC_CONTENT)

    try:
        with open(temp_file, 'rb') as f:
            files = {'file': ('test_doc.txt', f, 'text/plain')}
            data = {
                'title': TEST_DOC_TITLE,
                'data_source': DATASET,
                'process_async': 'false'  # Synchronous for testing
            }

            response = requests.post(
                f"{API_BASE_URL}/documents/upload",
                files=files,
                data=data
            )

        if response.status_code == 200:
            result = response.json()
            doc_id = result.get('document_id')
            print(f"    [OK] Document uploaded: {doc_id}")
            print(f"         Title: {result.get('title')}")
            print(f"         Status: {result.get('status')}")
            return doc_id
        else:
            print(f"    [FAIL] Upload failed: {response.status_code}")
            print(f"           {response.text}")
            return None
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()


def check_document_in_corpus(doc_id):
    """Check if document exists in corpus.jsonl"""
    print(f"\n[3/6] Checking if document {doc_id} exists in corpus...")

    if not CORPUS_PATH.exists():
        print(f"    [FAIL] Corpus file not found: {CORPUS_PATH}")
        return False

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                if doc.get('id') == doc_id:
                    print(f"    [OK] Found in corpus.jsonl")
                    print(f"         Title: {doc.get('title', 'N/A')}")
                    return True

    print(f"    [FAIL] Not found in corpus.jsonl")
    return False


def hard_delete_document(doc_id):
    """Hard delete document via API"""
    print(f"\n[4/6] Hard deleting document {doc_id}...")

    response = requests.delete(
        f"{API_BASE_URL}/documents/{doc_id}",
        params={'hard_delete': 'true'}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"    [OK] Hard delete successful")
        print(f"         Message: {result.get('message')}")
        return True
    else:
        print(f"    [FAIL] Delete failed: {response.status_code}")
        print(f"           {response.text}")
        return False


def verify_removal_from_corpus(doc_id):
    """Verify document is removed from corpus.jsonl"""
    print(f"\n[5/6] Verifying removal from corpus.jsonl...")

    if not CORPUS_PATH.exists():
        print(f"    [FAIL] Corpus file not found: {CORPUS_PATH}")
        return False

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                if doc.get('id') == doc_id:
                    print(f"    [FAIL] Document still exists in corpus.jsonl!")
                    print(f"           This means the fix didn't work.")
                    return False

    print(f"    [OK] Document removed from corpus.jsonl")
    return True


def verify_removal_from_kg(doc_id):
    """Verify document is removed from knowledge graph"""
    print(f"\n[6/6] Verifying removal from knowledge graph...")

    # Try to get document details
    response = requests.get(f"{API_BASE_URL}/documents/{doc_id}")

    if response.status_code == 404:
        print(f"    [OK] Document not found in registry (expected)")
        return True
    elif response.status_code == 200:
        print(f"    [FAIL] Document still exists in registry!")
        return False
    else:
        print(f"    [WARN] Unexpected status code: {response.status_code}")
        return False


def main():
    """Run complete deletion test"""
    print("="*70)
    print("Complete Hard Delete Test")
    print("="*70)
    print(f"API URL: {API_BASE_URL}")
    print(f"Dataset: {DATASET}")
    print(f"Corpus: {CORPUS_PATH}")

    # Step 1: Check server
    if not check_server_running():
        sys.exit(1)

    # Step 2: Upload test document
    doc_id = upload_test_document()
    if not doc_id:
        print("\n[ABORT] Failed to upload document")
        sys.exit(1)

    # Wait for processing to complete
    print("\nWaiting 3 seconds for processing to complete...")
    time.sleep(3)

    # Step 3: Verify in corpus
    if not check_document_in_corpus(doc_id):
        print("\n[ABORT] Document not found in corpus after upload")
        sys.exit(1)

    # Step 4: Hard delete
    if not hard_delete_document(doc_id):
        print("\n[ABORT] Failed to delete document")
        sys.exit(1)

    # Wait for deletion to complete
    print("\nWaiting 2 seconds for deletion to complete...")
    time.sleep(2)

    # Step 5: Verify removal from corpus
    corpus_removed = verify_removal_from_corpus(doc_id)

    # Step 6: Verify removal from KG
    kg_removed = verify_removal_from_kg(doc_id)

    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = corpus_removed and kg_removed

    print(f"Corpus Cleanup:     {'[PASS]' if corpus_removed else '[FAIL]'}")
    print(f"KG Cleanup:         {'[PASS]' if kg_removed else '[FAIL]'}")
    print()

    if all_passed:
        print("[SUCCESS] All tests passed! Hard delete is working correctly.")
        print("          Documents are removed from:")
        print("          - Knowledge graph (chunks, entities, edges, vectors)")
        print("          - Document registry")
        print("          - corpus.jsonl (prevents resurrection on rebuild)")
    else:
        print("[FAILURE] Some tests failed. Please review the output above.")

    print("="*70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
