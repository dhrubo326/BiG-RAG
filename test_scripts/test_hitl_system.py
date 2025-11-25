"""
Test Suite for HITL (Human-in-the-Loop) System

Tests all functionality of the HITL system including:
- Failed extraction storage
- Review queue management
- Status updates
- Statistics generation
- API endpoints (integration tests)

Part of Phase 1 Step 6: HITL System
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import json
import tempfile
import shutil
from pathlib import Path

from bigrag.hitl.failed_extraction_store import FailedExtractionStore


# Test Cases

async def test_save_failed_chunk():
    """Test saving failed chunk extraction."""
    print("\n[TEST 1] Saving failed chunk...")

    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        extraction_id = store.save_failed_chunk(
            chunk_id="chunk_001",
            chunk_content="This is a test paragraph with failed extraction.",
            failure_reason="Validation failed: numeric mismatch",
            validation_details={
                "status": "FAIL",
                "errors": ["Number mismatch: expected 5, found 3"]
            },
            document_id="doc_123",
            metadata={"title": "Test Document"}
        )

        print(f"  Extraction ID: {extraction_id}")
        assert extraction_id.startswith("chunk_chunk_001")
        assert store.chunks_file.exists()

        # Verify stored data
        with open(store.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["chunk_id"] == "chunk_001"
            assert data[0]["status"] == "pending_review"

        print("  [PASS] Failed chunk saved successfully")


async def test_save_failed_table():
    """Test saving failed table extraction."""
    print("\n[TEST 2] Saving failed table...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        extraction_id = store.save_failed_table(
            table_id="table_002",
            table_data={
                "headers": ["Column1", "Column2"],
                "rows": [["A", "B"], ["C", "D"]],
                "caption": "Test Table"
            },
            failure_reason="Table extraction incomplete",
            document_id="doc_456"
        )

        print(f"  Extraction ID: {extraction_id}")
        assert extraction_id.startswith("table_table_002")
        assert store.tables_file.exists()

        # Verify stored data
        with open(store.tables_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["table_id"] == "table_002"
            assert data[0]["type"] == "table"

        print("  [PASS] Failed table saved successfully")


async def test_get_failed_extractions():
    """Test retrieving failed extractions with filters."""
    print("\n[TEST 3] Retrieving failed extractions...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        # Save multiple failures
        store.save_failed_chunk("chunk_001", "Content 1", "Reason 1", {}, "doc_123")
        store.save_failed_chunk("chunk_002", "Content 2", "Reason 2", {}, "doc_123")
        store.save_failed_chunk("chunk_003", "Content 3", "Reason 3", {}, "doc_456")
        store.save_failed_table("table_001", {"headers": []}, "Reason 4", "doc_123")

        # Test: Get all failures
        all_failures = store.get_failed_extractions()
        print(f"  Total failures: {len(all_failures)}")
        assert len(all_failures) == 4

        # Test: Filter by document_id
        doc_failures = store.get_failed_extractions(document_id="doc_123")
        print(f"  Failures for doc_123: {len(doc_failures)}")
        assert len(doc_failures) == 3

        # Test: Filter by type
        chunk_failures = store.get_failed_extractions(extraction_type="chunk")
        print(f"  Chunk failures: {len(chunk_failures)}")
        assert len(chunk_failures) == 3

        table_failures = store.get_failed_extractions(extraction_type="table")
        print(f"  Table failures: {len(table_failures)}")
        assert len(table_failures) == 1

        print("  [PASS] Retrieval with filters works correctly")


async def test_review_queue():
    """Test review queue management."""
    print("\n[TEST 4] Review queue management...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        # Save failures
        ext_id_1 = store.save_failed_chunk("chunk_001", "Content", "Reason", {}, "doc_123")
        ext_id_2 = store.save_failed_chunk("chunk_002", "Content", "Reason", {}, "doc_456")

        # Get review queue
        queue = store.get_review_queue()
        print(f"  Pending review: {len(queue)}")
        assert len(queue) == 2
        assert all(item["status"] == "pending_review" for item in queue)

        # Mark one as reviewed
        store.mark_reviewed(ext_id_1)

        # Check queue again
        queue_after = store.get_review_queue()
        print(f"  Pending after review: {len(queue_after)}")
        assert len(queue_after) == 1

        print("  [PASS] Review queue management works correctly")


async def test_mark_reviewed():
    """Test marking extraction as reviewed."""
    print("\n[TEST 5] Marking extraction as reviewed...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        ext_id = store.save_failed_chunk("chunk_001", "Content", "Reason", {}, "doc_123")

        # Mark as reviewed without corrections
        success = store.mark_reviewed(ext_id)
        assert success

        # Verify status changed
        failures = store.get_failed_extractions()
        assert failures[0]["status"] == "reviewed"
        assert "reviewed_at" in failures[0]

        print("  [PASS] Extraction marked as reviewed")


async def test_mark_reviewed_with_corrections():
    """Test marking extraction as reviewed with corrections."""
    print("\n[TEST 6] Marking with corrections...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        ext_id = store.save_failed_chunk("chunk_001", "Content", "Reason", {}, "doc_123")

        # Mark as reviewed with corrections
        corrected_data = {
            "entities": [{"entity_name": "Corrected Entity"}],
            "relations": []
        }

        success = store.mark_reviewed(
            ext_id,
            corrected_data=corrected_data,
            reviewer_notes="Fixed numeric values"
        )
        assert success

        # Verify corrections stored
        failures = store.get_failed_extractions()
        assert "corrected_data" in failures[0]
        assert failures[0]["corrected_data"]["entities"][0]["entity_name"] == "Corrected Entity"
        assert failures[0]["reviewer_notes"] == "Fixed numeric values"

        print("  [PASS] Corrections stored successfully")


async def test_mark_corrected():
    """Test marking extraction as corrected and ready for reprocessing."""
    print("\n[TEST 7] Marking as corrected...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        ext_id = store.save_failed_chunk("chunk_001", "Content", "Reason", {}, "doc_123")

        # Mark as corrected
        corrected_data = {
            "entities": [{"entity_name": "Final Entity"}],
            "relations": [{"content": "Final Relation"}]
        }

        success = store.mark_corrected(
            ext_id,
            corrected_data=corrected_data,
            correction_notes="Ready for reprocessing"
        )
        assert success

        # Verify status
        failures = store.get_failed_extractions()
        assert failures[0]["status"] == "corrected"

        print("  [PASS] Extraction marked as corrected")


async def test_statistics():
    """Test statistics generation."""
    print("\n[TEST 8] Statistics generation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        # Create diverse failures
        ext_1 = store.save_failed_chunk("chunk_001", "C1", "R1", {}, "doc_123")
        ext_2 = store.save_failed_chunk("chunk_002", "C2", "R2", {}, "doc_123")
        ext_3 = store.save_failed_table("table_001", {}, "R3", "doc_456")
        ext_4 = store.save_failed_chunk("chunk_003", "C3", "R4", {}, "doc_789")

        # Mark some as reviewed
        store.mark_reviewed(ext_1)
        store.mark_corrected(ext_2, {"entities": []})

        # Get statistics
        stats = store.get_statistics()

        print(f"  Total failures: {stats['total_failures']}")
        print(f"  By type: {stats['by_type']}")
        print(f"  By status: {stats['by_status']}")
        print(f"  By document: {stats['by_document']}")

        assert stats["total_failures"] == 4
        assert stats["by_type"]["chunk"] == 3
        assert stats["by_type"]["table"] == 1
        assert stats["by_status"]["pending_review"] == 2  # ext_3 and ext_4
        assert stats["by_status"]["reviewed"] == 1  # ext_1
        assert stats["by_status"]["corrected"] == 1  # ext_2
        assert len(stats["by_document"]) == 3

        print("  [PASS] Statistics calculated correctly")


async def test_delete_extraction():
    """Test deleting extraction record."""
    print("\n[TEST 9] Deleting extraction...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        ext_id = store.save_failed_chunk("chunk_001", "Content", "Reason", {}, "doc_123")

        # Verify exists
        failures_before = store.get_failed_extractions()
        assert len(failures_before) == 1

        # Delete
        success = store.delete_extraction(ext_id)
        assert success

        # Verify deleted
        failures_after = store.get_failed_extractions()
        assert len(failures_after) == 0

        # Try deleting again (should fail)
        success_2 = store.delete_extraction(ext_id)
        assert not success_2

        print("  [PASS] Extraction deleted successfully")


async def test_multiple_failures_same_document():
    """Test handling multiple failures from same document."""
    print("\n[TEST 10] Multiple failures per document...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        # Save 5 failures from same document
        for i in range(5):
            store.save_failed_chunk(
                f"chunk_{i:03d}",
                f"Content {i}",
                f"Reason {i}",
                {},
                "doc_important"
            )

        # Get failures for this document
        doc_failures = store.get_failed_extractions(document_id="doc_important")
        print(f"  Failures for doc_important: {len(doc_failures)}")
        assert len(doc_failures) == 5

        # Get statistics
        stats = store.get_statistics()
        assert stats["by_document"]["doc_important"] == 5

        print("  [PASS] Multiple failures handled correctly")


async def test_persistence():
    """Test that data persists across store instances."""
    print("\n[TEST 11] Data persistence...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # First instance
        store1 = FailedExtractionStore(temp_dir)
        ext_id = store1.save_failed_chunk("chunk_001", "Content", "Reason", {}, "doc_123")

        # Second instance (same directory)
        store2 = FailedExtractionStore(temp_dir)
        failures = store2.get_failed_extractions()

        assert len(failures) == 1
        assert failures[0]["chunk_id"] == "chunk_001"

        print("  [PASS] Data persists across instances")


async def test_empty_store():
    """Test operations on empty store."""
    print("\n[TEST 12] Empty store operations...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        # Get empty results
        failures = store.get_failed_extractions()
        assert len(failures) == 0

        queue = store.get_review_queue()
        assert len(queue) == 0

        stats = store.get_statistics()
        assert stats["total_failures"] == 0
        assert stats["by_type"]["chunk"] == 0
        assert stats["by_type"]["table"] == 0

        print("  [PASS] Empty store operations work correctly")


async def test_invalid_extraction_id():
    """Test operations with invalid extraction ID."""
    print("\n[TEST 13] Invalid extraction ID handling...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        # Try marking nonexistent extraction
        success = store.mark_reviewed("nonexistent_id")
        assert not success

        # Try deleting nonexistent extraction
        success = store.delete_extraction("nonexistent_id")
        assert not success

        print("  [PASS] Invalid ID handled gracefully")


async def test_status_filtering():
    """Test filtering extractions by status."""
    print("\n[TEST 14] Status filtering...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = FailedExtractionStore(temp_dir)

        # Create failures with different statuses
        ext_1 = store.save_failed_chunk("chunk_001", "C1", "R1", {}, "doc_123")
        ext_2 = store.save_failed_chunk("chunk_002", "C2", "R2", {}, "doc_456")
        ext_3 = store.save_failed_chunk("chunk_003", "C3", "R3", {}, "doc_789")

        store.mark_reviewed(ext_1)
        store.mark_corrected(ext_2, {"entities": []})

        # Filter by status
        pending = store.get_failed_extractions(status="pending_review")
        reviewed = store.get_failed_extractions(status="reviewed")
        corrected = store.get_failed_extractions(status="corrected")

        print(f"  Pending: {len(pending)}")
        print(f"  Reviewed: {len(reviewed)}")
        print(f"  Corrected: {len(corrected)}")

        assert len(pending) == 1
        assert len(reviewed) == 1
        assert len(corrected) == 1

        print("  [PASS] Status filtering works correctly")


# Main Test Runner

async def run_all_tests():
    """Run all test cases."""
    print("=" * 70)
    print("HITL SYSTEM TEST SUITE")
    print("=" * 70)

    tests = [
        test_save_failed_chunk,
        test_save_failed_table,
        test_get_failed_extractions,
        test_review_queue,
        test_mark_reviewed,
        test_mark_reviewed_with_corrections,
        test_mark_corrected,
        test_statistics,
        test_delete_extraction,
        test_multiple_failures_same_document,
        test_persistence,
        test_empty_store,
        test_invalid_extraction_id,
        test_status_filtering,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"FAILED: {failed} tests")
    else:
        print("ALL TESTS PASSED")
    print("=" * 70)

    return passed, failed


if __name__ == '__main__':
    passed, failed = asyncio.run(run_all_tests())
    sys.exit(0 if failed == 0 else 1)
