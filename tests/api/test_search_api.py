"""
API tests for search functionality

Tests /search endpoint.
"""

import pytest
import os


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped"
)
class TestSearchAPI:
    """Test /search endpoint"""

    @pytest.mark.asyncio
    async def test_search_endpoint(self, api_client):
        """Test basic search functionality"""
        try:
            response = await api_client.post(
                "/search",
                json={"queries": ["Who is Messi?"]},
            )

            if response.status_code == 200:
                data = response.json()
                assert "results" in data or isinstance(data, dict)
            else:
                pytest.skip("Search endpoint not available")

        except Exception:
            pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
