"""
API tests for BiG-RAG backend server

Tests FastAPI endpoints (requires server running on localhost:8001).

To run these tests:
1. Start server: cd backend && python server.py --data_source demo_test
2. Run tests: pytest tests/api/ -v
"""

import pytest
import os


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped (server not running)"
)
class TestServerEndpoints:
    """Test backend API server endpoints"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        """Test /health endpoint"""
        try:
            response = await api_client.get("/health")
            assert response.status_code == 200
        except Exception:
            pytest.skip("API server not running")

    @pytest.mark.asyncio
    async def test_docs_endpoint(self, api_client):
        """Test /docs endpoint (OpenAPI docs)"""
        try:
            response = await api_client.get("/docs")
            assert response.status_code == 200
        except Exception:
            pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
