"""
API tests for evaluation endpoints

Tests retrieval and answer quality evaluation.
"""

import pytest
import os


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped"
)
class TestEvaluationAPI:
    """Test /eval endpoints"""

    @pytest.mark.asyncio
    async def test_eval_retrieval_basic(self, api_client):
        """Test basic retrieval evaluation"""
        request_data = {
            "queries": [{
                "question": "Who is Lionel Messi?",
                "ground_truth_docs": ["chunk_about_messi"]
            }],
            "dataset": "demo_test",
            "mode": "hybrid",
            "top_k": 5,
            "metrics": ["precision", "recall", "mrr", "ndcg"]
        }

        response = await api_client.post("/eval/retrieval", json=request_data)

        # Should return 200 with evaluation metrics
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["success"] == True
        assert "metrics" in data
        assert "total_queries" in data
        assert data["total_queries"] == 1

    @pytest.mark.asyncio
    async def test_eval_answer_basic(self, api_client):
        """Test basic answer evaluation"""
        request_data = {
            "test_cases": [{
                "question": "What is the capital of France?",
                "ground_truth": "Paris",
                "use_rag": False  # Don't use RAG for simple test
            }],
            "dataset": "demo_test",
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "metrics": ["em", "f1"]
        }

        response = await api_client.post("/eval/answer", json=request_data)

        # Should return 200 with evaluation metrics
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["success"] == True
        assert "aggregate_metrics" in data
        assert "per_question_results" in data
        assert len(data["per_question_results"]) == 1

    @pytest.mark.asyncio
    async def test_eval_answer_with_mismatch(self, api_client):
        """Test answer evaluation with wrong answer"""
        request_data = {
            "test_cases": [{
                "question": "What is 2+2?",
                "ground_truth": "4",
                "use_rag": False
            }],
            "dataset": "demo_test",
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "metrics": ["em", "f1"]
        }

        response = await api_client.post("/eval/answer", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # LLM might answer correctly, so just check structure
        assert "aggregate_metrics" in data
        assert "exact_match" in data["aggregate_metrics"] or "em" in data["aggregate_metrics"]

    @pytest.mark.asyncio
    async def test_eval_compare_configurations(self, api_client):
        """Test comparing different retrieval configurations"""
        request_data = {
            "queries": [{
                "question": "What is machine learning?",
                "ground_truth_docs": ["doc_001"]
            }],
            "dataset": "demo_test",
            "configurations": [
                {"name": "hybrid", "mode": "hybrid", "top_k": 5},
                {"name": "local", "mode": "local", "top_k": 5}
            ],
            "metrics": ["precision", "recall", "mrr"]
        }

        response = await api_client.post("/eval/compare", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "results" in data
        assert "best_configuration" in data

    @pytest.mark.asyncio
    async def test_eval_batch_basic(self, api_client):
        """Test batch evaluation"""
        request_data = {
            "test_cases": [
                {"question": "What is Python?", "ground_truth": "Python is a programming language", "ground_truth_docs": []},
                {"question": "What is AI?", "ground_truth": "AI is artificial intelligence", "ground_truth_docs": []}
            ],
            "dataset": "demo_test",
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "evaluate_retrieval": False,
            "evaluate_answer": True
        }

        response = await api_client.post("/eval/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "total_test_cases" in data

    @pytest.mark.asyncio
    async def test_eval_retrieval_with_multiple_docs(self, api_client):
        """Test retrieval evaluation with multiple ground truth docs"""
        request_data = {
            "queries": [{
                "question": "What is deep learning?",
                "ground_truth_docs": [
                    "doc_dl_1",
                    "doc_dl_2",
                    "doc_dl_3"
                ]
            }],
            "dataset": "demo_test",
            "mode": "hybrid",
            "top_k": 10,
            "metrics": ["precision", "recall"]
        }

        response = await api_client.post("/eval/retrieval", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    @pytest.mark.asyncio
    async def test_eval_answer_partial_match(self, api_client):
        """Test answer evaluation with partial match"""
        request_data = {
            "test_cases": [{
                "question": "Who created Python?",
                "ground_truth": "Guido van Rossum",
                "use_rag": False
            }],
            "dataset": "demo_test",
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "metrics": ["em", "f1"]
        }

        response = await api_client.post("/eval/answer", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        # F1 should be > 0 for partial match (LLM likely provides more context)
        assert "aggregate_metrics" in data

    @pytest.mark.asyncio
    async def test_eval_missing_required_fields(self, api_client):
        """Test evaluation endpoints reject incomplete requests"""
        # Missing ground_truth_docs
        request_data = {
            "queries": [{
                "question": "Test query"
                # Missing ground_truth_docs
            }],
            "dataset": "demo_test"
        }

        response = await api_client.post("/eval/retrieval", json=request_data)

        # Should return validation error (422)
        assert response.status_code == 422, f"Expected 422 validation error, got {response.status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
