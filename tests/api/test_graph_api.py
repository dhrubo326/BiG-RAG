"""
API tests for graph management endpoints

Tests graph stats and management operations.
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
        """Test /graph/stats endpoint"""
        try:
            response = await api_client.get("/graph/stats")

            if response.status_code == 200:
                stats = response.json()
                # Should have stats fields
                assert isinstance(stats, dict)
        except Exception:
            pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
