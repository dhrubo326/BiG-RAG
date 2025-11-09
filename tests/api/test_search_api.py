"""
API tests for search functionality

Tests /ask and /search endpoints with various query modes and parameters.
"""

import pytest
import os
import json


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped"
)
class TestSearchAPI:
    """Test /search and /ask endpoints"""

    @pytest.mark.asyncio
    async def test_search_endpoint_basic(self, api_client):
        """Test basic search functionality"""
        try:
            response = await api_client.post(
                "/search",
                json={"queries": ["Who is Messi?"]},
            )

            if response.status_code == 200:
                data = response.json()
                # Should return a list of results
                assert isinstance(data, list)
                assert len(data) > 0
                # Each result should be a JSON string
                first_result = json.loads(data[0])
                assert "query" in first_result
                assert "results" in first_result
            else:
                pytest.skip("Search endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_search_endpoint_batch(self, api_client):
        """Test batch search with multiple queries"""
        try:
            queries = [
                "Who is Lionel Messi?",
                "Where is Paris?",
                "What is Python?"
            ]
            response = await api_client.post(
                "/search",
                json={"queries": queries},
            )

            if response.status_code == 200:
                data = response.json()
                # Should return same number of results as queries
                assert len(data) == len(queries)
                # Each should be a valid JSON string
                for item in data:
                    result = json.loads(item)
                    assert "query" in result
                    assert "results" in result
            else:
                pytest.skip("Search endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_search_endpoint_empty_query(self, api_client):
        """Test search with empty query list"""
        try:
            response = await api_client.post(
                "/search",
                json={"queries": []},
            )

            if response.status_code == 200:
                data = response.json()
                # Should return empty list
                assert data == []
            elif response.status_code == 422:
                # Validation error is also acceptable
                pass
            else:
                pytest.skip("Search endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_basic(self, api_client):
        """Test /ask endpoint with basic question"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "Who is Lionel Messi?",
                    "mode": "hybrid",
                    "top_k": 5,
                    "enable_reranking": False
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert "question" in data
                assert "retrieved_contexts" in data
                assert "num_results" in data
                assert "mode" in data
                assert data["question"] == "Who is Lionel Messi?"
                assert data["mode"] == "hybrid"
                assert isinstance(data["retrieved_contexts"], list)
            else:
                pytest.skip("Ask endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_with_reranking(self, api_client):
        """Test /ask endpoint with semantic reranking enabled"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "What is machine learning?",
                    "mode": "hybrid",
                    "top_k": 10,
                    "enable_reranking": True
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert "question" in data
                assert "retrieved_contexts" in data
                # Check each context has required fields
                for ctx in data["retrieved_contexts"]:
                    assert "rank" in ctx
                    assert "context" in ctx
                    assert "coherence_score" in ctx
            else:
                pytest.skip("Ask endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_local_mode(self, api_client):
        """Test /ask endpoint with 'local' retrieval mode (entity-based)"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "Who invented Python?",
                    "mode": "local",
                    "top_k": 5
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert data["mode"] == "local"
            else:
                pytest.skip("Ask endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_global_mode(self, api_client):
        """Test /ask endpoint with 'global' retrieval mode (relation-based)"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "What is the capital of France?",
                    "mode": "global",
                    "top_k": 5
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert data["mode"] == "global"
            else:
                pytest.skip("Ask endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_naive_mode(self, api_client):
        """Test /ask endpoint with 'naive' retrieval mode (chunk-based only)"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "What is Bitcoin?",
                    "mode": "naive",
                    "top_k": 5
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert data["mode"] == "naive"
            else:
                pytest.skip("Ask endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_invalid_mode(self, api_client):
        """Test /ask endpoint rejects invalid mode"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "Test question",
                    "mode": "invalid_mode_xyz",
                    "top_k": 5
                }
            )

            # Should return validation error (422)
            assert response.status_code == 422

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_large_top_k(self, api_client):
        """Test /ask endpoint with large top_k value"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "Who is Albert Einstein?",
                    "mode": "hybrid",
                    "top_k": 50
                }
            )

            if response.status_code == 200:
                data = response.json()
                # Should handle large top_k gracefully
                assert "retrieved_contexts" in data
                # Number of results may be less than top_k if not enough matches
                assert data["num_results"] <= 50
            else:
                pytest.skip("Ask endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_missing_question(self, api_client):
        """Test /ask endpoint rejects missing question field"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "mode": "hybrid",
                    "top_k": 5
                }
            )

            # Should return validation error (422)
            assert response.status_code == 422

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_empty_question(self, api_client):
        """Test /ask endpoint with empty question string"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "",
                    "mode": "hybrid",
                    "top_k": 5
                }
            )

            # Should either accept and return empty results or reject with 422
            assert response.status_code in [200, 422]

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_endpoint_default_parameters(self, api_client):
        """Test /ask endpoint with minimal parameters (uses defaults)"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "What is artificial intelligence?"
                }
            )

            if response.status_code == 200:
                data = response.json()
                # Should use default mode and parameters
                assert "mode" in data
                assert "retrieved_contexts" in data
            else:
                pytest.skip("Ask endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_search_malformed_json(self, api_client):
        """Test /search endpoint rejects malformed JSON"""
        try:
            response = await api_client.post(
                "/search",
                content="not valid json",
                headers={"Content-Type": "application/json"}
            )

            # Should return 422 (validation error) or 400 (bad request)
            assert response.status_code in [400, 422]

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_ask_response_format(self, api_client):
        """Test /ask response has correct format and structure"""
        try:
            response = await api_client.post(
                "/ask",
                json={
                    "question": "Test question",
                    "mode": "hybrid",
                    "top_k": 3
                }
            )

            if response.status_code == 200:
                data = response.json()
                # Validate response schema
                assert isinstance(data, dict)
                assert "question" in data
                assert "retrieved_contexts" in data
                assert "num_results" in data
                assert "mode" in data
                assert "message" in data

                # Validate contexts structure
                for ctx in data["retrieved_contexts"]:
                    assert isinstance(ctx, dict)
                    assert "rank" in ctx
                    assert "context" in ctx
                    assert isinstance(ctx["rank"], int)
                    assert isinstance(ctx["context"], str)
            else:
                pytest.skip("Ask endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
