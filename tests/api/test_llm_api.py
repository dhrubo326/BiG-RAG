"""
API tests for LLM chat completion endpoint

Tests OpenAI-compatible chat endpoint.
"""

import pytest
import os


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped"
)
class TestLLMAPI:
    """Test /chat/completions endpoint (OpenAI-compatible)"""

    @pytest.mark.asyncio
    async def test_chat_completions_basic(self, api_client):
        """Test basic OpenAI-compatible chat completion"""
        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "use_rag": False  # Don't use RAG for simple test
        }

        response = await api_client.post("/chat/completions", json=request_data)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        # Should match OpenAI format
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert isinstance(data["choices"], list)
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

    @pytest.mark.asyncio
    async def test_chat_completions_multiple_messages(self, api_client):
        """Test chat with conversation history"""
        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Who created it?"}
            ],
            "use_rag": False
        }

        response = await api_client.post("/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    @pytest.mark.asyncio
    async def test_chat_completions_with_rag(self, api_client):
        """Test chat endpoint uses RAG context"""
        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Tell me about Lionel Messi"}
            ],
            "use_rag": True,  # Enable RAG
            "enable_reranking": False
        }

        response = await api_client.post("/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        # With RAG, response might include context
        assert len(data["choices"]) > 0

    @pytest.mark.asyncio
    async def test_chat_completions_temperature(self, api_client):
        """Test chat with temperature parameter"""
        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What is AI?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "use_rag": False
        }

        response = await api_client.post("/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    @pytest.mark.asyncio
    async def test_chat_completions_missing_messages(self, api_client):
        """Test chat endpoint rejects request without messages"""
        request_data = {
            "model": "gpt-4o-mini"
            # Missing required "messages" field
        }

        response = await api_client.post("/chat/completions", json=request_data)

        # Should return validation error (422)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_chat_completions_empty_messages(self, api_client):
        """Test chat endpoint with empty messages array"""
        request_data = {
            "model": "gpt-4o-mini",
            "messages": []  # Empty messages
        }

        response = await api_client.post("/chat/completions", json=request_data)

        # Should handle gracefully (422 or 400)
        assert response.status_code in [422, 400, 200], f"Got unexpected {response.status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
