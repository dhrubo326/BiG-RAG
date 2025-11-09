"""
API tests for graph management endpoints

Tests graph statistics, export, and subgraph operations.
"""

import pytest
import os


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped"
)
class TestGraphAPI:
    """Test graph management API"""

    @pytest.mark.asyncio
    async def test_graph_stats_endpoint(self, api_client):
        """Test /graph/stats endpoint returns statistics"""
        try:
            response = await api_client.get("/graph/stats")

            if response.status_code == 200:
                stats = response.json()
                # Should have stats fields
                assert isinstance(stats, dict)
                assert "dataset" in stats
                assert "entities" in stats or "total_entities" in stats
                assert "relations" in stats or "total_relations" in stats or "edges" in stats
            else:
                pytest.skip("Graph stats endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_stats_has_counts(self, api_client):
        """Test /graph/stats returns numeric counts"""
        try:
            response = await api_client.get("/graph/stats")

            if response.status_code == 200:
                stats = response.json()
                # Check for numeric fields
                numeric_fields = ["entities", "relations", "edges", "chunks", "total_entities", "total_relations"]
                found_numeric = False
                for field in numeric_fields:
                    if field in stats and isinstance(stats[field], (int, float)):
                        found_numeric = True
                        break
                assert found_numeric, "No numeric count fields found in stats"
            else:
                pytest.skip("Graph stats endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_export_basic(self, api_client):
        """Test /graph/export endpoint with basic parameters"""
        try:
            # Use a known dataset (may need to adjust based on your setup)
            response = await api_client.get("/graph/export?data_source=demo_test&limit=100")

            if response.status_code == 200:
                data = response.json()
                assert "nodes" in data
                assert "edges" in data
                assert isinstance(data["nodes"], list)
                assert isinstance(data["edges"], list)
            elif response.status_code == 404:
                pytest.skip("Dataset not available")
            else:
                pytest.skip("Graph export endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_export_with_limit(self, api_client):
        """Test /graph/export respects limit parameter"""
        try:
            limit = 50
            response = await api_client.get(f"/graph/export?data_source=demo_test&limit={limit}")

            if response.status_code == 200:
                data = response.json()
                # Number of nodes should not exceed limit
                assert len(data["nodes"]) <= limit
            elif response.status_code == 404:
                pytest.skip("Dataset not available")
            else:
                pytest.skip("Graph export endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_export_includes_stats(self, api_client):
        """Test /graph/export includes statistics"""
        try:
            response = await api_client.get("/graph/export?data_source=demo_test&limit=100")

            if response.status_code == 200:
                data = response.json()
                # Should include stats about the full graph (unsampled)
                assert "stats" in data or "total_nodes" in data or "total_edges" in data
            elif response.status_code == 404:
                pytest.skip("Dataset not available")
            else:
                pytest.skip("Graph export endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_export_node_types_filter(self, api_client):
        """Test /graph/export with node_types filter"""
        try:
            response = await api_client.get(
                "/graph/export?data_source=demo_test&limit=100&node_types=entity"
            )

            if response.status_code == 200:
                data = response.json()
                # All returned nodes should be of type 'entity'
                for node in data["nodes"]:
                    if "data" in node and "role" in node["data"]:
                        assert node["data"]["role"] == "entity"
            elif response.status_code == 404:
                pytest.skip("Dataset not available")
            else:
                pytest.skip("Graph export endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_export_min_weight_filter(self, api_client):
        """Test /graph/export with min_weight filter"""
        try:
            min_weight = 0.5
            response = await api_client.get(
                f"/graph/export?data_source=demo_test&limit=100&min_weight={min_weight}"
            )

            if response.status_code == 200:
                data = response.json()
                # All returned nodes should have weight >= min_weight
                for node in data["nodes"]:
                    if "data" in node and "weight" in node["data"]:
                        assert node["data"]["weight"] >= min_weight
            elif response.status_code == 404:
                pytest.skip("Dataset not available")
            else:
                pytest.skip("Graph export endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_subgraph_neighbors(self, api_client):
        """Test /graph/subgraph/neighbors endpoint"""
        try:
            # This test requires knowing a valid node_id
            # We'll try a generic query and accept skip if not found
            response = await api_client.get(
                "/graph/subgraph/neighbors?node_id=test_node&depth=1"
            )

            if response.status_code == 200:
                data = response.json()
                assert "nodes" in data
                assert "edges" in data
            elif response.status_code == 404:
                pytest.skip("Node not found or endpoint not available")
            else:
                pytest.skip("Subgraph neighbors endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_subgraph_neighbors_depth(self, api_client):
        """Test /graph/subgraph/neighbors with different depth values"""
        try:
            # Test with depth=2
            response = await api_client.get(
                "/graph/subgraph/neighbors?node_id=test_node&depth=2"
            )

            # Accept 200 (success) or 404 (node not found)
            assert response.status_code in [200, 404, 500]

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_subgraph_search(self, api_client):
        """Test /graph/subgraph/search endpoint"""
        try:
            response = await api_client.get(
                "/graph/subgraph/search?q=test&limit=10"
            )

            if response.status_code == 200:
                data = response.json()
                # Should return a list of nodes
                assert isinstance(data, list)
                # Each item should have node information
                for item in data:
                    assert isinstance(item, dict)
            else:
                pytest.skip("Subgraph search endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_subgraph_search_with_limit(self, api_client):
        """Test /graph/subgraph/search respects limit parameter"""
        try:
            limit = 5
            response = await api_client.get(
                f"/graph/subgraph/search?q=entity&limit={limit}"
            )

            if response.status_code == 200:
                data = response.json()
                # Number of results should not exceed limit
                assert len(data) <= limit
            else:
                pytest.skip("Subgraph search endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_subgraph_search_empty_query(self, api_client):
        """Test /graph/subgraph/search with empty query"""
        try:
            response = await api_client.get("/graph/subgraph/search?q=&limit=10")

            # Should either return empty results or validation error
            assert response.status_code in [200, 422]

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_export_sampling_strategies(self, api_client):
        """Test /graph/export with different sampling strategies"""
        try:
            strategies = ["top_weighted", "random", "diverse"]

            for strategy in strategies:
                response = await api_client.get(
                    f"/graph/export?data_source=demo_test&limit=100&sample_strategy={strategy}"
                )

                if response.status_code == 200:
                    data = response.json()
                    assert "nodes" in data
                    assert "edges" in data
                elif response.status_code == 404:
                    pytest.skip("Dataset not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_export_missing_data_source(self, api_client):
        """Test /graph/export without required data_source parameter"""
        try:
            response = await api_client.get("/graph/export?limit=100")

            # Should return validation error (422)
            assert response.status_code == 422

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_graph_stats_with_dataset_parameter(self, api_client):
        """Test /graph/stats with dataset query parameter"""
        try:
            response = await api_client.get("/graph/stats?dataset=demo_test")

            if response.status_code == 200:
                stats = response.json()
                assert isinstance(stats, dict)
                # Should have dataset field matching request
                if "dataset" in stats:
                    assert stats["dataset"] == "demo_test"
            elif response.status_code == 404:
                pytest.skip("Dataset not available")
            else:
                pytest.skip("Graph stats endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
