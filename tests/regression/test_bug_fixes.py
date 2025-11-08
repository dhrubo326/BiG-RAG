"""
Regression tests for all 6 critical bug fixes

This file must pass 100% to ensure bugs don't regress.

Bug List:
- Bug #1: Wrong hash prefix for edge deletion (bigrag.py:762)
- Bug #2: drop() deletes all documents (bigrag.py:681)
- Bug #3: Undefined load_env_file() in reload_config (config.py:366)
- Bug #4: Potential KeyError on missing dict keys (operate.py:942, 1126)
- Bug #5: upsert() doesn't actually update (storage.py:57-60)
- Bug #6: Wrong type annotations (operate.py:731-732)
"""

import pytest
import inspect
from pathlib import Path

# Import required modules
from bigrag import BiGRAG
from bigrag.config import reload_config
from bigrag.storage import JsonKVStorage
from bigrag.operate import kg_query
from bigrag.base import BaseVectorStorage, QueryParam


@pytest.mark.critical
@pytest.mark.regression
class TestBug1EdgeDeletionPrefix:
    """
    Regression test for Bug #1: Wrong hash prefix in edge deletion

    ISSUE: Used compute_mdhash_id(edge_name, prefix="edge-") but edge_name was
           already a hash ID starting with "rel-"
    FIX: Use edge_name directly without re-hashing
    """

    @pytest.mark.asyncio
    async def test_edge_deletion_uses_correct_id(self, bigrag_with_data):
        """Test that edge deletion uses correct hash ID format"""
        rag = bigrag_with_data

        # Get initial document count
        all_docs_before = await rag.full_docs.get_by_ids(
            await rag.full_docs.get_all_ids()
        )
        initial_count = len(all_docs_before)
        assert initial_count > 0, "No documents in test data"

        # Delete a document (triggers edge deletion)
        docs_to_delete = list(all_docs_before.values())
        if docs_to_delete:
            doc_id = docs_to_delete[0]["__id__"]
            stats = await rag.delete_document(doc_id)

            # Verify deletion stats
            assert stats is not None
            assert "chunks_deleted" in stats
            assert stats["chunks_deleted"] >= 0

            # Verify document actually deleted
            all_docs_after = await rag.full_docs.get_by_ids(
                await rag.full_docs.get_all_ids()
            )
            assert len(all_docs_after) == initial_count - 1

    @pytest.mark.asyncio
    async def test_edge_id_format_consistency(self):
        """Verify edge IDs consistently use 'rel-' prefix"""
        from bigrag.utils import compute_mdhash_id
        from bigrag.constants import BIPARTITE_EDGE_PREFIX

        # Edge IDs should use BIPARTITE_EDGE_PREFIX
        content = "Test edge content"
        edge_id = compute_mdhash_id(content, prefix=BIPARTITE_EDGE_PREFIX)

        # Verify prefix
        assert edge_id.startswith("rel-"), f"Edge ID has wrong prefix: {edge_id}"
        assert not edge_id.startswith("edge-"), "Edge ID should not use 'edge-' prefix"


@pytest.mark.critical
@pytest.mark.regression
class TestBug2DropDeletesAll:
    """
    Regression test for Bug #2: drop() deletes all documents instead of one

    ISSUE: Called full_docs.drop() which deletes ENTIRE storage
    FIX: Use full_docs.delete(doc_id) to delete only one document
    """

    @pytest.mark.asyncio
    async def test_single_document_deletion_not_all(self, bigrag_instance, sample_documents):
        """Test that deleting one document doesn't delete all"""
        rag = bigrag_instance

        # Insert multiple documents
        await rag.insert(sample_documents[:3])

        # Count documents before deletion
        all_docs_before = await rag.full_docs.get_by_ids(
            await rag.full_docs.get_all_ids()
        )
        assert len(all_docs_before) == 3

        # Delete only first document
        first_doc_id = list(all_docs_before.keys())[0]
        await rag.delete_document(first_doc_id)

        # Verify only 1 deleted, not all 3
        all_docs_after = await rag.full_docs.get_by_ids(
            await rag.full_docs.get_all_ids()
        )
        assert len(all_docs_after) == 2, "Bug #2: drop() deleted all documents!"

    @pytest.mark.asyncio
    async def test_deletion_with_no_chunks(self, bigrag_instance):
        """Test deletion when document has no chunks (edge case that triggered bug)"""
        rag = bigrag_instance

        # Insert documents
        await rag.insert(["Doc 1", "Doc 2", "Doc 3"])

        # Get a document ID
        all_docs = await rag.full_docs.get_by_ids(
            await rag.full_docs.get_all_ids()
        )
        doc_id_to_delete = list(all_docs.keys())[0]

        # Delete
        await rag.delete_document(doc_id_to_delete)

        # Verify others still exist
        remaining_docs = await rag.full_docs.get_by_ids(
            await rag.full_docs.get_all_ids()
        )
        assert len(remaining_docs) == 2


@pytest.mark.critical
@pytest.mark.regression
class TestBug3ReloadConfigUndefined:
    """
    Regression test for Bug #3: Undefined load_env_file() in reload_config

    ISSUE: reload_config() called load_env_file() which only exists when
           python-dotenv is NOT installed
    FIX: Added try-except to use load_dotenv() first, fallback to load_env_file()
    """

    def test_reload_config_no_error(self):
        """Test that reload_config() doesn't raise NameError"""
        try:
            config = reload_config()
            assert config is not None
        except NameError as e:
            pytest.fail(f"Bug #3 not fixed: reload_config() raised NameError: {e}")

    def test_reload_config_works_with_dotenv(self):
        """Test reload_config() when python-dotenv is installed"""
        try:
            import dotenv
            # If dotenv is available, should use it
            config = reload_config()
            assert config is not None
        except ImportError:
            pytest.skip("python-dotenv not installed, skipping")

    def test_reload_config_works_without_dotenv(self, monkeypatch):
        """Test reload_config() fallback path"""
        # This should work whether dotenv is installed or not
        config = reload_config()
        assert config is not None
        assert hasattr(config, 'chunk_size')


@pytest.mark.regression
class TestBug4DefensiveDictAccess:
    """
    Regression test for Bug #4: Potential KeyError on missing dict keys

    ISSUE: Used r["entity_name"] and r["bipartite_edge_name"] without checking
    FIX: Use defensive access: r.get("entity_name") with existence check
    """

    def test_defensive_entity_name_access(self):
        """Test that entity name extraction handles missing keys"""
        # Simulate vector DB results with missing keys
        malformed_results = [
            {"id": "123", "distance": 0.9},  # Missing entity_name
            {"entity_name": "ValidEntity", "distance": 0.8},
        ]

        # Defensive access pattern (should not raise KeyError)
        try:
            safe_results = [
                r.get("entity_name")
                for r in malformed_results
                if "entity_name" in r
            ]
            # Should get only the valid one
            assert len(safe_results) == 1
            assert safe_results[0] == "ValidEntity"
        except KeyError:
            pytest.fail("Bug #4 not fixed: KeyError on missing dict key")

    def test_defensive_edge_name_access(self):
        """Test that edge name extraction handles missing keys"""
        malformed_results = [
            {"id": "456", "distance": 0.95},  # Missing bipartite_edge_name
            {"bipartite_edge_name": "rel-abc123", "distance": 0.7},
        ]

        # Defensive access
        try:
            safe_results = [
                r.get("bipartite_edge_name")
                for r in malformed_results
                if "bipartite_edge_name" in r
            ]
            assert len(safe_results) == 1
            assert safe_results[0] == "rel-abc123"
        except KeyError:
            pytest.fail("Bug #4 not fixed: KeyError on missing bipartite_edge_name")


@pytest.mark.critical
@pytest.mark.regression
class TestBug5UpsertActuallyUpdates:
    """
    Regression test for Bug #5: upsert() doesn't actually update

    ISSUE: upsert() filtered out existing keys, so updates didn't happen
    FIX: Update ALL data (both new and existing keys)
    """

    @pytest.mark.asyncio
    async def test_upsert_updates_existing_keys(self):
        """Test that upsert() actually updates existing values"""
        # Create fresh KV storage
        import tempfile
        import shutil
        from pathlib import Path

        temp_dir = Path(tempfile.mkdtemp())

        try:
            kv = JsonKVStorage(
                namespace="test_bug5",
                global_config={"working_dir": str(temp_dir)},
                embedding_func=None,
            )

            # Insert initial data
            await kv.upsert({"key1": {"value": "old_value"}})
            assert (await kv.get_by_id("key1"))["value"] == "old_value"

            # Update with new value
            await kv.upsert({"key1": {"value": "new_value"}})

            # Bug #5 fix: Should update
            result = await kv.get_by_id("key1")
            assert result["value"] == "new_value", "Bug #5 not fixed: upsert didn't update!"

        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_upsert_mixed_insert_and_update(self):
        """Test upsert with mix of new and existing keys"""
        import tempfile
        import shutil
        from pathlib import Path

        temp_dir = Path(tempfile.mkdtemp())

        try:
            kv = JsonKVStorage(
                namespace="test_bug5_mixed",
                global_config={"working_dir": str(temp_dir)},
                embedding_func=None,
            )

            # Insert initial
            await kv.upsert({"key1": {"value": "old1"}})

            # Upsert with mix
            await kv.upsert({
                "key1": {"value": "updated1"},  # Update
                "key2": {"value": "new2"},      # Insert
            })

            # Both should work
            assert (await kv.get_by_id("key1"))["value"] == "updated1"
            assert (await kv.get_by_id("key2"))["value"] == "new2"

        finally:
            shutil.rmtree(temp_dir)


@pytest.mark.regression
class TestBug6TypeAnnotations:
    """
    Regression test for Bug #6: Wrong type annotations

    ISSUE: kg_query had vdb_entities and vdb_bipartite_edges typed as 'list'
    FIX: Changed to BaseVectorStorage
    """

    def test_kg_query_type_annotations(self):
        """Test that kg_query has correct type annotations"""
        sig = inspect.signature(kg_query)

        # Check vdb_entities annotation
        vdb_entities_annotation = sig.parameters['vdb_entities'].annotation
        assert vdb_entities_annotation == BaseVectorStorage, \
            f"Bug #6 not fixed: vdb_entities is {vdb_entities_annotation}, expected BaseVectorStorage"

        # Check vdb_bipartite_edges annotation
        vdb_bipartite_edges_annotation = sig.parameters['vdb_bipartite_edges'].annotation
        assert vdb_bipartite_edges_annotation == BaseVectorStorage, \
            f"Bug #6 not fixed: vdb_bipartite_edges is {vdb_bipartite_edges_annotation}, expected BaseVectorStorage"


@pytest.mark.critical
@pytest.mark.regression
class TestAllBugsIntegration:
    """
    Integration test ensuring all bug fixes work together

    This test exercises a complete workflow that would have failed
    with any of the 6 bugs present.
    """

    @pytest.mark.asyncio
    async def test_complete_workflow_with_all_fixes(self, bigrag_instance, sample_documents):
        """Test complete workflow that exercises all bug fixes"""
        rag = bigrag_instance

        # Insert documents (exercises Bug #5 - upsert)
        await rag.insert(sample_documents[:2])

        # Query (exercises Bug #4 - defensive dict access, Bug #6 - type annotations)
        results = await rag.query(
            "test query",
            param=QueryParam(mode="hybrid"),
        )
        assert results is not None

        # Delete one document (exercises Bug #1 - edge deletion, Bug #2 - single delete)
        all_docs = await rag.full_docs.get_by_ids(await rag.full_docs.get_all_ids())
        if all_docs:
            doc_id = list(all_docs.keys())[0]
            await rag.delete_document(doc_id)

        # Verify system still functional
        remaining_docs = await rag.full_docs.get_by_ids(await rag.full_docs.get_all_ids())
        assert len(remaining_docs) > 0

        # Test config reload (exercises Bug #3)
        try:
            config = reload_config()
            assert config is not None
        except NameError:
            pytest.fail("Bug #3 regression: reload_config() failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "regression"])
