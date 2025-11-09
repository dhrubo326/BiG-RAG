"""
Unit tests for bigrag.storage module

Tests storage layer implementations: JsonKVStorage, NanoVectorDBStorage, NetworkXStorage.
Validates Bug #5 fix (upsert actually updates).
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from bigrag.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from bigrag.base import TextChunkSchema
from bigrag.utils import EmbeddingFunc


class TestJsonKVStorage:
    """Test JSON key-value storage"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    async def kv_storage(self, temp_dir):
        """Create fresh KV storage instance"""
        storage = JsonKVStorage(
            namespace="test",
            global_config={"working_dir": str(temp_dir), "embedding_batch_num": 32},
            embedding_func=None,
        )
        return storage

    @pytest.mark.asyncio
    async def test_upsert_insert_new_keys(self, kv_storage):
        """Test upsert with new keys (insert behavior)"""
        data = {
            "key1": {"value": "data1"},
            "key2": {"value": "data2"},
        }

        left_data = await kv_storage.upsert(data)

        # All keys should be new
        assert len(left_data) == 2
        assert "key1" in left_data
        assert "key2" in left_data

        # Verify data is stored
        assert await kv_storage.get_by_id("key1") == {"value": "data1"}
        assert await kv_storage.get_by_id("key2") == {"value": "data2"}

    @pytest.mark.asyncio
    async def test_upsert_update_existing_keys(self, kv_storage):
        """Test Bug #5 fix: upsert actually updates existing keys"""
        # Insert initial data
        initial_data = {
            "key1": {"value": "old_value"},
        }
        await kv_storage.upsert(initial_data)

        # Verify initial value
        assert (await kv_storage.get_by_id("key1"))["value"] == "old_value"

        # Update with new value
        update_data = {
            "key1": {"value": "new_value"},
        }
        left_data = await kv_storage.upsert(update_data)

        # left_data should be empty (no new keys)
        assert len(left_data) == 0

        # Bug #5 fix: Value should be updated
        result = await kv_storage.get_by_id("key1")
        assert result["value"] == "new_value", "upsert did not update existing key!"

    @pytest.mark.asyncio
    async def test_upsert_mixed_new_and_existing(self, kv_storage):
        """Test upsert with mix of new and existing keys"""
        # Insert initial data
        await kv_storage.upsert({"key1": {"value": "old1"}})

        # Upsert with mix
        mixed_data = {
            "key1": {"value": "updated1"},  # Update
            "key2": {"value": "new2"},      # Insert
        }
        left_data = await kv_storage.upsert(mixed_data)

        # Only key2 should be in left_data (new)
        assert len(left_data) == 1
        assert "key2" in left_data

        # Both should be in storage with correct values
        assert (await kv_storage.get_by_id("key1"))["value"] == "updated1"
        assert (await kv_storage.get_by_id("key2"))["value"] == "new2"

    @pytest.mark.asyncio
    async def test_delete_single_item(self, kv_storage):
        """Test deleting single item"""
        # Insert data
        await kv_storage.upsert({"key1": {"value": "data1"}})

        # Delete
        result = await kv_storage.delete("key1")
        assert result is True

        # Verify deleted
        assert await kv_storage.get_by_id("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_item(self, kv_storage):
        """Test deleting nonexistent item"""
        result = await kv_storage.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_many(self, kv_storage):
        """Test deleting multiple items"""
        # Insert data
        await kv_storage.upsert({
            "key1": {"value": "data1"},
            "key2": {"value": "data2"},
            "key3": {"value": "data3"},
        })

        # Delete multiple
        deleted_count = await kv_storage.delete_many(["key1", "key3"])
        assert deleted_count == 2

        # Verify key2 still exists, key1 and key3 deleted
        assert await kv_storage.get_by_id("key1") is None
        assert await kv_storage.get_by_id("key2") is not None
        assert await kv_storage.get_by_id("key3") is None

    @pytest.mark.asyncio
    async def test_drop_deletes_all(self, kv_storage):
        """Test drop clears all data"""
        # Insert data
        await kv_storage.upsert({
            "key1": {"value": "data1"},
            "key2": {"value": "data2"},
        })

        # Drop
        await kv_storage.drop()

        # Verify all deleted
        assert await kv_storage.get_by_id("key1") is None
        assert await kv_storage.get_by_id("key2") is None

    @pytest.mark.asyncio
    async def test_get_by_ids(self, kv_storage):
        """Test batch retrieval by IDs"""
        # Insert data
        await kv_storage.upsert({
            "key1": {"value": "data1"},
            "key2": {"value": "data2"},
            "key3": {"value": "data3"},
        })

        # Get by IDs
        results = await kv_storage.get_by_ids(["key1", "key3", "nonexistent"])

        # Should return dict with existing keys
        assert len(results) == 2
        assert "key1" in results
        assert "key3" in results
        assert "nonexistent" not in results

    @pytest.mark.asyncio
    async def test_filter_keys(self, kv_storage):
        """Test filtering keys"""
        # Insert data
        await kv_storage.upsert({
            "key1": {"value": "data1"},
            "key2": {"value": "data2"},
        })

        # Filter: which keys are NOT in storage?
        not_in_storage = await kv_storage.filter_keys(["key1", "key3", "key4"])

        # key1 exists, key3 and key4 don't
        assert "key1" not in not_in_storage
        assert "key3" in not_in_storage
        assert "key4" in not_in_storage


class TestNanoVectorDBStorage:
    """Test vector database storage"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    async def mock_embedding_func(self):
        """Mock embedding function for testing"""
        import numpy as np

        async def embed(texts):
            # Return simple embeddings (just for testing)
            return [np.random.rand(128).tolist() for _ in texts]

        # Wrap in EmbeddingFunc dataclass (required by NanoVectorDBStorage)
        return EmbeddingFunc(
            embedding_dim=128,
            max_token_size=8192,
            func=embed,
            concurrent_limit=16
        )

    @pytest.fixture
    async def vector_storage(self, temp_dir, mock_embedding_func):
        """Create fresh vector storage instance"""
        storage = NanoVectorDBStorage(
            namespace="test",
            global_config={"working_dir": str(temp_dir), "embedding_batch_num": 32},
            embedding_func=mock_embedding_func,
            meta_fields=["entity_name"],
        )
        return storage

    @pytest.mark.asyncio
    async def test_upsert_vectors(self, vector_storage):
        """Test upserting vectors"""
        data = {
            "id1": {"content": "Text one", "entity_name": "Entity1"},
            "id2": {"content": "Text two", "entity_name": "Entity2"},
        }

        await vector_storage.upsert(data)

        # Query to verify data is indexed
        results = await vector_storage.query("Text one", top_k=1)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_vectors(self, vector_storage):
        """Test querying vectors"""
        # Insert test data
        data = {
            "id1": {"content": "Apple is a fruit", "entity_name": "Apple"},
            "id2": {"content": "Car is a vehicle", "entity_name": "Car"},
        }
        await vector_storage.upsert(data)

        # Query
        results = await vector_storage.query("fruit", top_k=2)

        # Should return results
        assert len(results) > 0
        assert isinstance(results, list)
        assert all("id" in r for r in results)

    @pytest.mark.asyncio
    async def test_delete_vectors(self, vector_storage):
        """Test deleting vectors"""
        # Insert data
        data = {
            "id1": {"content": "Text one", "entity_name": "Entity1"},
            "id2": {"content": "Text two", "entity_name": "Entity2"},
        }
        await vector_storage.upsert(data)

        # Delete one
        deleted = await vector_storage.delete(["id1"])
        assert deleted > 0

        # Query should not return deleted item
        results = await vector_storage.query("Text one", top_k=5)
        # id1 should not be in results
        assert all(r["id"] != "id1" for r in results)


class TestNetworkXStorage:
    """Test NetworkX graph storage"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    async def graph_storage(self, temp_dir):
        """Create fresh graph storage instance"""
        storage = NetworkXStorage(
            namespace="test",
            global_config={"working_dir": str(temp_dir), "embedding_batch_num": 32},
            embedding_func=None,
        )
        return storage

    @pytest.mark.asyncio
    async def test_upsert_node(self, graph_storage):
        """Test inserting nodes"""
        await graph_storage.upsert_node(
            "node1",
            node_data={"name": "Node1", "type": "test"},
        )

        # Retrieve node
        node = await graph_storage.get_node("node1")
        assert node is not None
        assert node["name"] == "Node1"
        assert node["type"] == "test"

    @pytest.mark.asyncio
    async def test_upsert_edge(self, graph_storage):
        """Test inserting edges"""
        # First insert nodes
        await graph_storage.upsert_node("node1", node_data={"name": "Node1"})
        await graph_storage.upsert_node("node2", node_data={"name": "Node2"})

        # Insert edge
        await graph_storage.upsert_edge(
            "node1",
            "node2",
            edge_data={"relation": "connects"},
        )

        # Get edges
        edges = await graph_storage.get_node_edges("node1")
        assert len(edges) > 0

    @pytest.mark.asyncio
    async def test_delete_node(self, graph_storage):
        """Test deleting nodes"""
        # Insert node
        await graph_storage.upsert_node("node1", node_data={"name": "Node1"})

        # Verify exists
        assert await graph_storage.has_node("node1") is True

        # Delete
        await graph_storage.delete_node("node1")

        # Verify deleted
        assert await graph_storage.has_node("node1") is False

    @pytest.mark.asyncio
    async def test_get_node_edges(self, graph_storage):
        """Test getting node edges"""
        # Create small graph
        await graph_storage.upsert_node("A", node_data={})
        await graph_storage.upsert_node("B", node_data={})
        await graph_storage.upsert_node("C", node_data={})

        await graph_storage.upsert_edge("A", "B", edge_data={})
        await graph_storage.upsert_edge("A", "C", edge_data={})

        # Get edges for A
        edges = await graph_storage.get_node_edges("A")
        assert len(edges) == 2

    @pytest.mark.asyncio
    async def test_has_node(self, graph_storage):
        """Test checking node existence"""
        # Initially doesn't exist
        assert await graph_storage.has_node("test_node") is False

        # Insert node
        await graph_storage.upsert_node("test_node", node_data={})

        # Now exists
        assert await graph_storage.has_node("test_node") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
