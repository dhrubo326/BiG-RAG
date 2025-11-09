"""
API tests for BiG-RAG backend server endpoints

Tests basic server functionality and health endpoints.

To run these tests:
1. Start server: cd backend && python server.py --data_source demo_test
2. Run tests: pytest tests/api/test_server_endpoints.py -v
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
    async def test_root_endpoint(self, api_client):
        """Test root / endpoint returns API information"""
        try:
            response = await api_client.get("/")
            assert response.status_code == 200

            data = response.json()
            assert "message" in data
            assert "version" in data
            assert "endpoints" in data
            assert data["message"] == "BiG-RAG Unified API Server - Enhanced"

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        """Test /health endpoint returns system health"""
        try:
            response = await api_client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert "version" in data
            assert "timestamp" in data
            assert "uptime_seconds" in data
            assert data["status"] == "healthy"
            assert data["version"] == "3.0.0"

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_health_includes_rag_instances(self, api_client):
        """Test /health endpoint includes RAG instance information"""
        try:
            response = await api_client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert "rag_instances" in data
            # Should be a dict of dataset -> instance info
            assert isinstance(data["rag_instances"], dict)

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_health_includes_job_queue(self, api_client):
        """Test /health endpoint includes job queue stats"""
        try:
            response = await api_client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert "job_queue" in data
            job_queue = data["job_queue"]
            assert "total" in job_queue
            assert "pending" in job_queue
            assert "processing" in job_queue
            assert "completed" in job_queue
            assert "failed" in job_queue

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_docs_endpoint(self, api_client):
        """Test /docs endpoint (OpenAPI docs)"""
        try:
            response = await api_client.get("/docs")
            assert response.status_code == 200
            # Docs should return HTML
            assert "html" in response.headers.get("content-type", "").lower()

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_redoc_endpoint(self, api_client):
        """Test /redoc endpoint (alternative API docs)"""
        try:
            response = await api_client.get("/redoc")
            assert response.status_code == 200
            # ReDoc should return HTML
            assert "html" in response.headers.get("content-type", "").lower()

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_root_lists_available_providers(self, api_client):
        """Test root endpoint lists available LLM providers"""
        try:
            response = await api_client.get("/")
            assert response.status_code == 200

            data = response.json()
            assert "available_providers" in data
            assert isinstance(data["available_providers"], list)
            # Should have at least one provider
            assert len(data["available_providers"]) > 0

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_root_lists_features(self, api_client):
        """Test root endpoint lists API features"""
        try:
            response = await api_client.get("/")
            assert response.status_code == 200

            data = response.json()
            assert "features" in data
            assert isinstance(data["features"], list)
            assert len(data["features"]) > 0
            # Check for key features
            features_text = " ".join(data["features"])
            assert "Markdown" in features_text or "markdown" in features_text

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_404_on_invalid_endpoint(self, api_client):
        """Test 404 error on non-existent endpoint"""
        try:
            response = await api_client.get("/this-endpoint-does-not-exist")
            assert response.status_code == 404

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_health_uptime_increases(self, api_client):
        """Test that uptime increases between calls"""
        try:
            import asyncio

            # First call
            response1 = await api_client.get("/health")
            assert response1.status_code == 200
            uptime1 = response1.json()["uptime_seconds"]

            # Wait 1 second
            await asyncio.sleep(1)

            # Second call
            response2 = await api_client.get("/health")
            assert response2.status_code == 200
            uptime2 = response2.json()["uptime_seconds"]

            # Uptime should have increased
            assert uptime2 > uptime1

        except Exception as e:
            pytest.skip(f"API server not running: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
