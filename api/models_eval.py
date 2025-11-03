"""
Pydantic Models for Evaluation Endpoints

Request and response models for:
- /eval/retrieval
- /eval/answer
- /eval/compare
- /eval/batch
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ==============================================================================
# Request Models
# ==============================================================================

class QueryWithGroundTruth(BaseModel):
    """Single query with ground truth for evaluation"""
    question: str = Field(..., description="Query text")
    ground_truth_docs: List[str] = Field(..., description="List of relevant document IDs")
    ground_truth_answer: Optional[str] = Field(None, description="Ground truth answer (optional)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is BiG-RAG?",
                "ground_truth_docs": ["doc_001", "doc_005"],
                "ground_truth_answer": "BiG-RAG is a bipartite graph RAG framework"
            }
        }


class RetrievalEvalRequest(BaseModel):
    """Request for retrieval evaluation"""
    queries: List[QueryWithGroundTruth] = Field(..., description="List of queries to evaluate")
    dataset: str = Field("demo_test", description="Dataset to query")
    mode: str = Field("hybrid", description="Retrieval mode (local, global, hybrid, naive)")
    top_k: int = Field(5, ge=1, le=100, description="Number of documents to retrieve")
    metrics: List[str] = Field(
        ["precision", "recall", "f1", "hit", "mrr", "ndcg"],
        description="Metrics to calculate"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "queries": [
                    {
                        "question": "What is artificial intelligence?",
                        "ground_truth_docs": ["doc_001", "doc_003"]
                    }
                ],
                "dataset": "demo_test",
                "mode": "hybrid",
                "top_k": 5,
                "metrics": ["precision", "recall", "hit", "mrr", "ndcg"]
            }
        }


class AnswerEvalTestCase(BaseModel):
    """Single test case for answer evaluation"""
    question: str = Field(..., description="Question to ask")
    ground_truth: str = Field(..., description="Expected answer")
    use_rag: bool = Field(True, description="Whether to use RAG for retrieval")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the capital of France?",
                "ground_truth": "Paris",
                "use_rag": True
            }
        }


class AnswerEvalRequest(BaseModel):
    """Request for answer quality evaluation"""
    test_cases: List[AnswerEvalTestCase] = Field(..., description="List of test cases")
    dataset: str = Field("demo_test", description="Dataset for retrieval")
    llm_provider: str = Field("openai", description="LLM provider (openai, anthropic, etc.)")
    model: str = Field("gpt-4o-mini", description="Model name")
    mode: str = Field("hybrid", description="Retrieval mode")
    top_k: int = Field(5, description="Number of contexts to retrieve")
    metrics: List[str] = Field(
        ["em", "f1", "rouge_l"],
        description="Answer quality metrics to calculate"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "test_cases": [
                    {
                        "question": "What is AI?",
                        "ground_truth": "Artificial Intelligence",
                        "use_rag": True
                    }
                ],
                "dataset": "demo_test",
                "llm_provider": "openai",
                "metrics": ["em", "f1", "rouge_l"]
            }
        }


class CompareConfig(BaseModel):
    """Configuration for comparison"""
    name: str = Field(..., description="Configuration name")
    mode: str = Field(..., description="Retrieval mode")
    top_k: int = Field(5, description="Number of documents to retrieve")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "hybrid_k5",
                "mode": "hybrid",
                "top_k": 5
            }
        }


class CompareEvalRequest(BaseModel):
    """Request for comparative evaluation"""
    queries: List[QueryWithGroundTruth] = Field(..., description="List of queries to evaluate")
    dataset: str = Field("demo_test", description="Dataset to query")
    configurations: List[CompareConfig] = Field(..., description="Configurations to compare")
    metrics: List[str] = Field(
        ["precision", "recall", "hit", "mrr"],
        description="Metrics to calculate"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "queries": [
                    {
                        "question": "What is machine learning?",
                        "ground_truth_docs": ["doc_001"]
                    }
                ],
                "dataset": "demo_test",
                "configurations": [
                    {"name": "hybrid", "mode": "hybrid", "top_k": 5},
                    {"name": "local", "mode": "local", "top_k": 5}
                ],
                "metrics": ["precision", "recall", "mrr", "latency"]
            }
        }


class BatchEvalRequest(BaseModel):
    """Request for batch evaluation from file"""
    dataset_file: str = Field(..., description="Path to QA dataset file (JSON)")
    data_source: str = Field("demo_test", description="BiG-RAG dataset to query")
    mode: str = Field("hybrid", description="Retrieval mode")
    top_k: int = Field(5, description="Number of documents to retrieve")
    metrics: List[str] = Field(
        ["em", "f1", "precision", "recall"],
        description="Metrics to calculate"
    )
    use_llm: bool = Field(True, description="Whether to use LLM for answer generation")
    llm_provider: Optional[str] = Field("openai", description="LLM provider")
    save_results: bool = Field(False, description="Save detailed results to file")
    output_file: Optional[str] = Field(None, description="Output file path")
    limit: Optional[int] = Field(None, description="Limit number of questions to evaluate")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_file": "test_datasets/eval_qa.json",
                "data_source": "demo_test",
                "mode": "hybrid",
                "top_k": 5,
                "metrics": ["em", "f1", "precision", "recall"],
                "use_llm": True,
                "save_results": True
            }
        }


# ==============================================================================
# Response Models
# ==============================================================================

class PerQueryRetrievalResult(BaseModel):
    """Retrieval evaluation result for single query"""
    question: str = Field(..., description="Query text")
    retrieved_docs: List[str] = Field(..., description="Retrieved document IDs")
    relevant_retrieved: List[str] = Field(..., description="Relevant docs that were retrieved")
    metrics: Dict[str, float] = Field(..., description="Metric scores for this query")
    latency_ms: Optional[float] = Field(None, description="Retrieval latency in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is BiG-RAG?",
                "retrieved_docs": ["doc_001", "doc_003", "doc_005"],
                "relevant_retrieved": ["doc_001", "doc_005"],
                "metrics": {
                    "precision@5": 0.667,
                    "recall@5": 1.0,
                    "mrr": 1.0
                }
            }
        }


class RetrievalEvalResponse(BaseModel):
    """Response for retrieval evaluation"""
    success: bool = Field(..., description="Whether evaluation succeeded")
    total_queries: int = Field(..., description="Total number of queries evaluated")
    metrics: Dict[str, float] = Field(..., description="Aggregate metrics across all queries")
    per_query_results: List[PerQueryRetrievalResult] = Field(..., description="Per-query breakdown")
    evaluation_time: float = Field(..., description="Total evaluation time in seconds")
    latency_stats: Optional[Dict[str, float]] = Field(
        None,
        description="Latency statistics (mean_ms, median_ms, p95_ms, p99_ms, min_ms, max_ms)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_queries": 2,
                "metrics": {
                    "precision@5": 0.7,
                    "recall@5": 0.85,
                    "f1@5": 0.77,
                    "mrr": 0.75
                },
                "per_query_results": [],
                "evaluation_time": 3.45
            }
        }


class PerQuestionAnswerResult(BaseModel):
    """Answer evaluation result for single question"""
    question: str = Field(..., description="Question text")
    ground_truth: str = Field(..., description="Ground truth answer")
    predicted_answer: str = Field(..., description="Generated answer")
    metrics: Dict[str, float] = Field(..., description="Metric scores")
    retrieval_used: bool = Field(..., description="Whether retrieval was used")
    num_contexts_used: int = Field(0, description="Number of context documents used")
    generation_time: float = Field(..., description="Time to generate answer (seconds)")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is AI?",
                "ground_truth": "Artificial Intelligence",
                "predicted_answer": "AI stands for Artificial Intelligence",
                "metrics": {
                    "exact_match": 0.0,
                    "f1": 0.667,
                    "rouge_l": 0.5
                },
                "retrieval_used": True,
                "num_contexts_used": 3,
                "generation_time": 1.23
            }
        }


class AnswerEvalResponse(BaseModel):
    """Response for answer quality evaluation"""
    success: bool = Field(..., description="Whether evaluation succeeded")
    total_questions: int = Field(..., description="Total questions evaluated")
    aggregate_metrics: Dict[str, float] = Field(..., description="Aggregate metrics")
    per_question_results: List[PerQuestionAnswerResult] = Field(..., description="Per-question results")
    total_time: float = Field(..., description="Total evaluation time (seconds)")
    with_rag_performance: Optional[Dict[str, float]] = Field(None, description="Performance with RAG")
    without_rag_performance: Optional[Dict[str, float]] = Field(None, description="Performance without RAG")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_questions": 5,
                "aggregate_metrics": {
                    "exact_match": 0.6,
                    "f1": 0.78,
                    "rouge_l": 0.72
                },
                "per_question_results": [],
                "total_time": 12.5
            }
        }


class CompareEvalResponse(BaseModel):
    """Response for comparative evaluation"""
    success: bool = Field(..., description="Whether comparison succeeded")
    comparison_results: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Results per configuration"
    )
    best_configuration: str = Field(..., description="Name of best performing configuration")
    ranking: List[str] = Field(..., description="Configurations ranked by performance")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "comparison_results": {
                    "hybrid": {
                        "precision@5": 0.8,
                        "recall@5": 1.0,
                        "avg_latency_ms": 234
                    },
                    "local": {
                        "precision@5": 0.6,
                        "recall@5": 0.8,
                        "avg_latency_ms": 156
                    }
                },
                "best_configuration": "hybrid",
                "ranking": ["hybrid", "local"]
            }
        }


class BatchEvalPerformance(BaseModel):
    """Performance statistics for batch evaluation"""
    total_time: float = Field(..., description="Total time (seconds)")
    avg_time_per_query: float = Field(..., description="Average time per query (seconds)")
    total_llm_calls: int = Field(0, description="Total LLM API calls")
    total_embedding_calls: int = Field(0, description="Total embedding API calls")


class BatchEvalResponse(BaseModel):
    """Response for batch evaluation"""
    success: bool = Field(..., description="Whether batch evaluation succeeded")
    dataset: str = Field(..., description="Dataset name")
    total_questions: int = Field(..., description="Total questions in dataset")
    processed: int = Field(..., description="Questions successfully processed")
    failed: int = Field(..., description="Questions that failed")
    metrics: Dict[str, Dict[str, float]] = Field(..., description="Retrieval and answer metrics")
    performance: BatchEvalPerformance = Field(..., description="Performance statistics")
    results_saved_to: Optional[str] = Field(None, description="Path to saved results file")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "dataset": "demo_test",
                "total_questions": 50,
                "processed": 50,
                "failed": 0,
                "metrics": {
                    "retrieval": {
                        "precision@5": 0.68,
                        "recall@5": 0.82
                    },
                    "answer": {
                        "exact_match": 0.44,
                        "f1": 0.62
                    }
                },
                "performance": {
                    "total_time": 125.5,
                    "avg_time_per_query": 2.51,
                    "total_llm_calls": 50,
                    "total_embedding_calls": 50
                },
                "results_saved_to": "evaluation_results/eval_2025_10_30.json"
            }
        }
