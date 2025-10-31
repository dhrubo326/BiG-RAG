# BiG-RAG API Testing & Evaluation Endpoints Plan

**Version:** 2.0
**Date:** 2025-10-31
**Purpose:** Comprehensive testing, error validation, and accuracy measurement endpoints
**Status:** 🚀 Phase 3 Complete - Evaluation Endpoints Live!

---

## Implementation Status

| Phase | Status | Endpoints | Description |
|-------|--------|-----------|-------------|
| **Phase 1-2** | ✅ Complete | Document Management | Upload, insert, delete, list documents |
| **Phase 3** | ✅ Complete | Evaluation Core | `/eval/retrieval`, `/eval/answer`, `/eval/compare`, `/eval/batch` |
| **Phase 4** | 📋 Planned | Debug & Testing | Pipeline inspection, query analysis, edge case testing |
| **Phase 5** | 📋 Planned | Analytics | Usage stats, performance monitoring, exports |
| **Phase 6** | 📋 Planned | Error Testing | Error injection, failure analysis |

---

## Quick Start Guide

**Already implemented and ready to use:**

1. **Test Retrieval Quality**: Use `/eval/retrieval` to measure Precision, Recall, MRR, NDCG
2. **Test Answer Quality**: Use `/eval/answer` to measure Exact Match, F1, ROUGE-L
3. **Compare Configurations**: Use `/eval/compare` to compare retrieval modes side-by-side
4. **Batch Evaluation**: Use `/eval/batch` to evaluate entire test datasets

**How to test:**
- See [TESTING_GUIDE.md](TESTING_GUIDE.md) for step-by-step instructions
- Sample test dataset: [test_datasets/eval_qa.json](test_datasets/eval_qa.json)
- Implementation details: [EVALUATION_IMPLEMENTATION_SUMMARY.md](EVALUATION_IMPLEMENTATION_SUMMARY.md)

**Server requirement:**
```bash
# Ensure server is running
python script_api.py --data_source demo_test
```

---

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Endpoints (✅ LIVE)](#evaluation-endpoints)
3. [Testing & Debug Endpoints (📋 Planned)](#testing--debug-endpoints)
4. [Error Simulation & Validation (📋 Planned)](#error-simulation--validation)
5. [Analytics & Monitoring (📋 Planned)](#analytics--monitoring)
6. [Implementation Phases](#implementation-phases)
7. [File Structure](#file-structure)
8. [Upload Endpoint - Auto KG Building](#upload-endpoint---auto-kg-building)

---

## 1. Overview

### Objectives

This plan extends the BiG-RAG API with endpoints for:

- ✅ **Accuracy Measurement**: Evaluate retrieval and generation quality with standard metrics
- ✅ **Systematic Testing**: Automated testing with various query types and edge cases
- ✅ **Error Validation**: Test error handling and edge case behavior
- ✅ **Performance Benchmarking**: Measure latency, throughput, and resource usage
- ✅ **Debug & Inspection**: Deep dive into retrieval pipeline internals
- ✅ **Comparative Analysis**: Compare retrieval modes and model configurations

### Key Metrics

**Retrieval Metrics:**
- Precision@K, Recall@K, F1@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Average Precision (MAP)

**Answer Quality Metrics:**
- Exact Match (EM)
- Token-level F1
- ROUGE-L
- BERTScore
- Semantic Similarity (SimCSE)

**Performance Metrics:**
- Query latency (p50, p95, p99)
- Tokens per second
- API call counts (LLM, embedding)
- Memory usage

---

## 2. Evaluation Endpoints (✅ LIVE)

**Status:** All 4 endpoints implemented and functional
**Documentation:** See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed testing instructions

### 2.1 POST /eval/retrieval ✅

**Purpose:** Evaluate retrieval quality with ground truth documents

**Request:**
```json
{
  "queries": [
    {
      "question": "What is BiG-RAG?",
      "ground_truth_docs": ["doc_001", "doc_005"],
      "ground_truth_answer": "BiG-RAG is a bipartite graph RAG framework"
    }
  ],
  "dataset": "demo_test",
  "mode": "hybrid",
  "top_k": 5,
  "metrics": ["precision", "recall", "mrr", "ndcg"]
}
```

**Response:**
```json
{
  "success": true,
  "total_queries": 1,
  "metrics": {
    "precision@5": 0.4,
    "recall@5": 1.0,
    "f1@5": 0.571,
    "mrr": 1.0,
    "ndcg@5": 0.934
  },
  "per_query_results": [
    {
      "question": "What is BiG-RAG?",
      "retrieved_docs": ["doc_001", "doc_003", "doc_005", "doc_010", "doc_012"],
      "relevant_retrieved": ["doc_001", "doc_005"],
      "precision": 0.4,
      "recall": 1.0,
      "reciprocal_rank": 1.0,
      "ndcg": 0.934
    }
  ],
  "evaluation_time": 2.34
}
```

**Features:**
- Batch evaluation with multiple queries
- Multiple metric calculation in single request
- Per-query breakdown for detailed analysis
- Support for all retrieval modes (local, global, hybrid, naive)

---

### 2.2 POST /eval/answer ✅

**Purpose:** Evaluate answer quality against ground truth

**Request:**
```json
{
  "test_cases": [
    {
      "question": "What is the capital of France?",
      "ground_truth": "Paris",
      "use_rag": true
    },
    {
      "question": "Who invented the telephone?",
      "ground_truth": "Alexander Graham Bell",
      "use_rag": true
    }
  ],
  "dataset": "demo_test",
  "llm_provider": "openai",
  "model": "gpt-4o-mini",
  "metrics": ["em", "f1", "rouge_l", "bertscore", "semantic_similarity"]
}
```

**Response:**
```json
{
  "success": true,
  "total_questions": 2,
  "aggregate_metrics": {
    "exact_match": 0.5,
    "f1_score": 0.75,
    "rouge_l": 0.68,
    "bertscore": 0.82,
    "semantic_similarity": 0.88
  },
  "per_question_results": [
    {
      "question": "What is the capital of France?",
      "ground_truth": "Paris",
      "predicted_answer": "The capital of France is Paris.",
      "metrics": {
        "exact_match": 0,
        "f1_score": 1.0,
        "rouge_l": 0.5,
        "bertscore": 0.95,
        "semantic_similarity": 0.98
      },
      "retrieval_used": true,
      "num_contexts_used": 3,
      "generation_time": 1.2
    }
  ],
  "total_time": 5.6,
  "with_rag_performance": {
    "avg_em": 0.5,
    "avg_f1": 0.75
  }
}
```

**Features:**
- End-to-end answer evaluation (retrieval + generation)
- Multiple metrics in one call
- Compare RAG vs non-RAG performance
- Track retrieval usage per question

---

### 2.3 POST /eval/compare ✅

**Purpose:** Compare different retrieval modes or configurations

**Request:**
```json
{
  "queries": [
    {
      "question": "What is machine learning?",
      "ground_truth_docs": ["doc_001"]
    }
  ],
  "dataset": "demo_test",
  "configurations": [
    {
      "name": "hybrid",
      "mode": "hybrid",
      "top_k": 5
    },
    {
      "name": "local_only",
      "mode": "local",
      "top_k": 5
    },
    {
      "name": "global_only",
      "mode": "global",
      "top_k": 5
    },
    {
      "name": "naive",
      "mode": "naive",
      "top_k": 5
    }
  ],
  "metrics": ["precision", "recall", "mrr", "latency"]
}
```

**Response:**
```json
{
  "success": true,
  "comparison_results": {
    "hybrid": {
      "precision@5": 0.6,
      "recall@5": 1.0,
      "mrr": 1.0,
      "avg_latency_ms": 234
    },
    "local_only": {
      "precision@5": 0.4,
      "recall@5": 0.8,
      "mrr": 0.5,
      "avg_latency_ms": 156
    },
    "global_only": {
      "precision@5": 0.2,
      "recall@5": 0.4,
      "mrr": 0.33,
      "avg_latency_ms": 178
    },
    "naive": {
      "precision@5": 0.2,
      "recall@5": 0.6,
      "mrr": 0.5,
      "avg_latency_ms": 98
    }
  },
  "best_configuration": "hybrid",
  "ranking": ["hybrid", "local_only", "global_only", "naive"]
}
```

**Features:**
- Side-by-side comparison of retrieval modes
- Performance vs accuracy tradeoff analysis
- Automatic ranking by specified metric

---

### 2.4 POST /eval/batch ✅

**Purpose:** Batch evaluation from QA dataset file

**Request:**
```json
{
  "dataset_file": "datasets/demo_test/raw/qa_test.json",
  "data_source": "demo_test",
  "mode": "hybrid",
  "top_k": 5,
  "metrics": ["em", "f1", "precision", "recall"],
  "use_llm": true,
  "llm_provider": "openai",
  "save_results": true,
  "output_file": "evaluation_results/eval_2025_10_30.json"
}
```

**Response:**
```json
{
  "success": true,
  "dataset": "demo_test",
  "total_questions": 50,
  "processed": 50,
  "failed": 0,
  "metrics": {
    "retrieval": {
      "precision@5": 0.68,
      "recall@5": 0.82,
      "f1@5": 0.74,
      "mrr": 0.76
    },
    "answer": {
      "exact_match": 0.44,
      "f1_score": 0.62,
      "rouge_l": 0.58
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
```

**Features:**
- Load test set from file (supports QA datasets)
- Automatic metric calculation
- Save detailed results for later analysis
- Progress tracking for long evaluations

---

## 3. Testing & Debug Endpoints (📋 Planned)

**Status:** Not yet implemented - Phase 4
**Priority:** Medium

### 3.1 POST /debug/retrieval 📋

**Purpose:** Inspect retrieval pipeline step-by-step

**Request:**
```json
{
  "query": "What is artificial intelligence?",
  "dataset": "demo_test",
  "mode": "hybrid",
  "top_k": 5,
  "include_embeddings": false,
  "include_scores": true
}
```

**Response:**
```json
{
  "success": true,
  "query": "What is artificial intelligence?",
  "query_embedding_shape": [1536],
  "retrieval_steps": {
    "1_entity_matching": {
      "matched_entities": [
        {
          "name": "Artificial Intelligence",
          "type": "TECHNOLOGY",
          "score": 0.95,
          "source_docs": ["doc_001", "doc_003"]
        },
        {
          "name": "Machine Learning",
          "type": "TECHNOLOGY",
          "score": 0.82,
          "source_docs": ["doc_001", "doc_005"]
        }
      ],
      "top_k": 10,
      "search_time_ms": 23
    },
    "2_edge_matching": {
      "matched_edges": [
        {
          "relation": "is_a_field_of",
          "entities": ["Artificial Intelligence", "Computer Science"],
          "score": 0.88,
          "source_docs": ["doc_001"]
        }
      ],
      "top_k": 10,
      "search_time_ms": 18
    },
    "3_graph_traversal": {
      "traversed_paths": [
        "Query -> Entity(AI) -> Edge(is_field_of) -> Doc(doc_001)",
        "Query -> Entity(AI) -> Doc(doc_001)",
        "Query -> Entity(ML) -> Doc(doc_005)"
      ],
      "unique_docs": ["doc_001", "doc_003", "doc_005", "doc_008", "doc_012"]
    },
    "4_chunk_retrieval": {
      "retrieved_chunks": [
        {
          "chunk_id": "doc_001_chunk_0",
          "score": 0.92,
          "content_preview": "Artificial Intelligence (AI) is the simulation of human intelligence...",
          "tokens": 150,
          "document_id": "doc_001"
        }
      ],
      "total_chunks": 15
    },
    "5_ranking": {
      "final_results": [
        {
          "document_id": "doc_001",
          "title": "Introduction to AI",
          "relevance_score": 0.92,
          "rank": 1,
          "matched_via": ["entity", "edge"]
        }
      ]
    }
  },
  "total_time_ms": 234,
  "cache_hits": 0
}
```

**Features:**
- Step-by-step pipeline visibility
- Entity and edge matching details
- Graph traversal paths
- Chunk-level retrieval scores
- Performance breakdown per stage

---

### 3.2 GET /debug/document/{document_id} 📋

**Purpose:** Inspect how a document was processed and indexed

**Request:**
```
GET /debug/document/upload-abc123?dataset=demo_test
```

**Response:**
```json
{
  "success": true,
  "document_id": "upload-abc123",
  "title": "BiG-RAG Research Paper",
  "processing_details": {
    "upload_date": "2025-10-30T10:30:00Z",
    "indexed_date": "2025-10-30T10:35:00Z",
    "processing_time": 298.5,
    "status": "indexed"
  },
  "chunking": {
    "total_chunks": 12,
    "chunk_size": 1200,
    "overlap": 100,
    "chunks": [
      {
        "chunk_id": "upload-abc123_chunk_0",
        "tokens": 1200,
        "start_offset": 0,
        "end_offset": 1200,
        "content_preview": "BiG-RAG: A Bipartite Graph Retrieval-Augmented..."
      }
    ]
  },
  "entity_extraction": {
    "total_entities": 35,
    "extraction_method": "gpt-4o-mini",
    "gleaning_rounds": 2,
    "entities_by_type": {
      "TECHNOLOGY": 12,
      "PERSON": 5,
      "ORGANIZATION": 8,
      "CONCEPT": 10
    },
    "top_entities": [
      {
        "name": "BiG-RAG",
        "type": "TECHNOLOGY",
        "description": "Bipartite Graph Retrieval-Augmented Generation framework",
        "mentions": 15,
        "chunks": ["chunk_0", "chunk_1", "chunk_5"]
      }
    ]
  },
  "relation_extraction": {
    "total_edges": 28,
    "edges_by_type": {
      "is_a": 5,
      "uses": 8,
      "developed_by": 3,
      "related_to": 12
    },
    "sample_edges": [
      {
        "source_entity": "BiG-RAG",
        "relation": "uses",
        "target_entity": "Bipartite Graph",
        "source_chunk": "chunk_0"
      }
    ]
  },
  "embedding": {
    "embedding_model": "text-embedding-3-large",
    "embedding_dim": 1536,
    "total_embeddings": 47,
    "breakdown": {
      "entity_embeddings": 35,
      "edge_embeddings": 28,
      "chunk_embeddings": 12
    }
  },
  "graph_statistics": {
    "node_degree": {
      "avg": 2.3,
      "max": 8,
      "min": 1
    },
    "connected_documents": 5,
    "entity_overlap_with_corpus": 0.23
  }
}
```

**Features:**
- Complete processing pipeline inspection
- Chunking details with previews
- Entity extraction results with types
- Relation extraction details
- Graph connectivity analysis

---

### 3.3 POST /debug/query-analysis 📋

**Purpose:** Analyze query characteristics and predict retrieval difficulty

**Request:**
```json
{
  "queries": [
    "What is AI?",
    "Explain the relationship between deep learning, machine learning, and artificial intelligence in the context of modern neural networks",
    "Who developed BiG-RAG and when was it published?"
  ],
  "dataset": "demo_test"
}
```

**Response:**
```json
{
  "success": true,
  "analyses": [
    {
      "query": "What is AI?",
      "characteristics": {
        "length": 3,
        "tokens": 4,
        "type": "definition",
        "complexity": "simple",
        "entities_detected": ["AI"],
        "has_temporal_reference": false,
        "has_numerical_reference": false
      },
      "predicted_difficulty": "easy",
      "suggested_mode": "hybrid",
      "suggested_top_k": 3,
      "corpus_coverage": {
        "matching_documents": 12,
        "matching_entities": 5,
        "estimated_precision": 0.8
      }
    },
    {
      "query": "Explain the relationship between deep learning, machine learning, and artificial intelligence...",
      "characteristics": {
        "length": 18,
        "tokens": 23,
        "type": "explanation",
        "complexity": "complex",
        "entities_detected": ["deep learning", "machine learning", "artificial intelligence", "neural networks"],
        "has_temporal_reference": false,
        "has_numerical_reference": false,
        "multi_hop": true
      },
      "predicted_difficulty": "hard",
      "suggested_mode": "hybrid",
      "suggested_top_k": 10,
      "corpus_coverage": {
        "matching_documents": 8,
        "matching_entities": 4,
        "estimated_precision": 0.6
      }
    }
  ]
}
```

**Features:**
- Query complexity analysis
- Entity detection in query
- Difficulty prediction
- Optimal retrieval configuration suggestion
- Corpus coverage estimation

---

### 3.4 POST /test/edge-cases 📋

**Purpose:** Test system behavior with edge cases

**Request:**
```json
{
  "test_scenarios": [
    "empty_query",
    "very_long_query",
    "special_characters",
    "non_english",
    "numeric_only",
    "repeated_words",
    "malformed_request"
  ],
  "dataset": "demo_test"
}
```

**Response:**
```json
{
  "success": true,
  "test_results": {
    "empty_query": {
      "test": "",
      "status": "error",
      "error_code": 400,
      "error_message": "Query cannot be empty",
      "handled_correctly": true
    },
    "very_long_query": {
      "test": "What is... [5000 words]",
      "status": "success",
      "truncated": true,
      "original_length": 5000,
      "processed_length": 4096,
      "results_count": 5,
      "handled_correctly": true
    },
    "special_characters": {
      "test": "What is AI? @#$%^&*()",
      "status": "success",
      "sanitized_query": "What is AI",
      "results_count": 5,
      "handled_correctly": true
    }
  },
  "total_tests": 7,
  "passed": 7,
  "failed": 0,
  "edge_cases_handled": true
}
```

**Features:**
- Automated edge case testing
- Error handling validation
- Input sanitization verification
- Comprehensive test coverage

---

## 4. Error Simulation & Validation (📋 Planned)

**Status:** Not yet implemented - Phase 6
**Priority:** Low

### 4.1 POST /test/error-injection 📋

**Purpose:** Simulate various error conditions for testing error handling

**Request:**
```json
{
  "error_type": "api_timeout",
  "endpoint": "/ask",
  "parameters": {
    "question": "What is AI?",
    "timeout_after_ms": 100
  },
  "expected_behavior": "graceful_fallback"
}
```

**Error Types to Test:**
- `api_timeout`: LLM/embedding API timeout
- `api_rate_limit`: Rate limit exceeded
- `api_auth_failure`: Invalid API key
- `corpus_not_found`: Dataset doesn't exist
- `index_corrupted`: FAISS index corrupted
- `out_of_memory`: Simulated OOM
- `network_failure`: Network connectivity issue

**Response:**
```json
{
  "success": true,
  "error_injected": "api_timeout",
  "system_response": {
    "status_code": 503,
    "error_message": "Service temporarily unavailable - LLM API timeout",
    "fallback_used": true,
    "recovery_attempted": true
  },
  "validation": {
    "error_handled_gracefully": true,
    "user_friendly_message": true,
    "proper_status_code": true,
    "logged_correctly": true
  }
}
```

---

### 4.2 GET /test/failed-jobs 📋

**Purpose:** Retrieve and analyze failed processing jobs

**Request:**
```
GET /test/failed-jobs?limit=50&include_details=true
```

**Response:**
```json
{
  "success": true,
  "total_failed_jobs": 3,
  "failed_jobs": [
    {
      "job_id": "job-xyz789",
      "document_id": "upload-failed123",
      "dataset": "demo_test",
      "failed_at": "2025-10-30T11:00:00Z",
      "error_stage": "extracting_entities",
      "error_message": "OpenAI API rate limit exceeded",
      "error_type": "rate_limit_error",
      "retry_count": 3,
      "can_retry": true,
      "stack_trace": "..."
    }
  ],
  "error_statistics": {
    "by_stage": {
      "extracting_entities": 2,
      "embedding": 1
    },
    "by_type": {
      "rate_limit_error": 2,
      "timeout_error": 1
    }
  },
  "recommendations": [
    "Reduce batch size for entity extraction",
    "Implement exponential backoff for API calls",
    "Check OpenAI API quota"
  ]
}
```

---

## 5. Analytics & Monitoring (📋 Planned)

**Status:** Not yet implemented - Phase 5
**Priority:** Medium-Low

### 5.1 GET /analytics/query-stats 📋

**Purpose:** Query usage analytics

**Request:**
```
GET /analytics/query-stats?time_range=7d&group_by=day
```

**Response:**
```json
{
  "success": true,
  "time_range": "2025-10-23 to 2025-10-30",
  "total_queries": 1234,
  "unique_queries": 567,
  "query_statistics": {
    "avg_queries_per_day": 176,
    "peak_day": "2025-10-28",
    "peak_queries": 245,
    "avg_query_length": 12.5,
    "avg_response_time_ms": 234
  },
  "top_queries": [
    {
      "query": "What is artificial intelligence?",
      "count": 45,
      "avg_results": 5,
      "avg_latency_ms": 198
    }
  ],
  "query_types": {
    "definition": 234,
    "explanation": 456,
    "comparison": 123,
    "factual": 421
  },
  "retrieval_modes_used": {
    "hybrid": 890,
    "local": 200,
    "global": 100,
    "naive": 44
  },
  "failed_queries": {
    "count": 12,
    "percentage": 0.97,
    "top_failures": [
      {
        "query": "Empty query test",
        "error": "Query cannot be empty",
        "count": 5
      }
    ]
  }
}
```

---

### 5.2 GET /analytics/document-usage 📋

**Purpose:** Document retrieval statistics

**Request:**
```
GET /analytics/document-usage?dataset=demo_test&sort_by=retrieval_count
```

**Response:**
```json
{
  "success": true,
  "dataset": "demo_test",
  "total_documents": 50,
  "total_retrievals": 5678,
  "document_usage": [
    {
      "document_id": "doc_001",
      "title": "Introduction to AI",
      "retrieval_count": 234,
      "avg_rank": 1.2,
      "avg_score": 0.89,
      "unique_queries": 89,
      "last_retrieved": "2025-10-30T09:30:00Z"
    }
  ],
  "usage_distribution": {
    "highly_used": 5,
    "moderately_used": 15,
    "rarely_used": 20,
    "never_used": 10
  },
  "recommendations": [
    "Consider removing 10 documents that have never been retrieved",
    "Documents doc_001, doc_003, doc_005 are frequently retrieved - ensure they're up to date"
  ]
}
```

---

### 5.3 GET /analytics/performance 📋

**Purpose:** System performance metrics

**Request:**
```
GET /analytics/performance?time_range=24h&metrics=all
```

**Response:**
```json
{
  "success": true,
  "time_range": "last_24h",
  "query_performance": {
    "total_queries": 456,
    "latency": {
      "p50": 189,
      "p95": 456,
      "p99": 789,
      "max": 1234,
      "avg": 234
    },
    "throughput": {
      "queries_per_second": 5.3,
      "peak_qps": 12.5
    }
  },
  "api_usage": {
    "llm_calls": {
      "total": 456,
      "tokens_used": 234567,
      "cost_usd": 2.34
    },
    "embedding_calls": {
      "total": 456,
      "tokens_used": 123456,
      "cost_usd": 0.12
    }
  },
  "resource_usage": {
    "avg_memory_mb": 1234,
    "peak_memory_mb": 2345,
    "cpu_percent": 45.6
  },
  "cache_statistics": {
    "hit_rate": 0.23,
    "total_hits": 105,
    "total_misses": 351
  }
}
```

---

### 5.4 POST /analytics/export 📋

**Purpose:** Export analytics data for external analysis

**Request:**
```json
{
  "export_type": "query_logs",
  "time_range": "7d",
  "format": "csv",
  "include_fields": [
    "timestamp",
    "query",
    "dataset",
    "mode",
    "top_k",
    "latency_ms",
    "results_count",
    "success"
  ],
  "filters": {
    "status": "success",
    "min_latency_ms": 100
  }
}
```

**Response:**
```json
{
  "success": true,
  "export_file": "exports/query_logs_2025_10_30.csv",
  "records_exported": 1234,
  "file_size_mb": 2.3,
  "download_url": "/downloads/query_logs_2025_10_30.csv",
  "expires_at": "2025-10-31T00:00:00Z"
}
```

**Export Types:**
- `query_logs`: All query logs with metadata
- `evaluation_results`: Evaluation results over time
- `document_stats`: Document usage statistics
- `performance_metrics`: Performance metrics time series

---

## 6. Implementation Phases

### Phase 3: Evaluation Core ✅ COMPLETE

**Status:** ✅ Completed on 2025-10-31
**Time Taken:** 1 day

**Completed Tasks:**
1. ✅ Created `api/metrics.py` (485 lines):
   - Metric calculation functions (EM, F1, Precision, Recall, MRR, NDCG, MAP, ROUGE-L)
   - Text normalization (SQuAD/HotpotQA style)
   - Ground truth comparison logic
   - Batch metric aggregation

2. ✅ Created `api/models_eval.py` (370 lines):
   - Request models: RetrievalEvalRequest, AnswerEvalRequest, CompareEvalRequest, BatchEvalRequest
   - Response models: RetrievalEvalResponse, AnswerEvalResponse, CompareEvalResponse, BatchEvalResponse
   - Validation with Pydantic Field validators
   - Example schemas in docstrings

3. ✅ Created `api/evaluation.py` (541 lines):
   - evaluate_single_retrieval() - Single query evaluation
   - evaluate_retrieval() - Multiple query evaluation
   - evaluate_single_answer() - Answer quality evaluation
   - compare_configurations() - Side-by-side comparison
   - batch_evaluate() - File-based batch evaluation
   - load_qa_dataset() - Dataset loading
   - save_evaluation_results() - Results export

4. ✅ Implemented endpoints in `script_api.py`:
   - ✅ POST /eval/retrieval - Retrieval quality evaluation
   - ✅ POST /eval/answer - Answer quality evaluation
   - ✅ POST /eval/compare - Configuration comparison
   - ✅ POST /eval/batch - Batch evaluation from file

5. ✅ Created test dataset:
   - test_datasets/eval_qa.json (10 questions with ground truth)

6. ✅ Created documentation:
   - TESTING_GUIDE.md (443 lines) - Step-by-step testing instructions
   - EVALUATION_IMPLEMENTATION_SUMMARY.md (550 lines) - Implementation details

**Deliverables:**
- ✅ All `/eval/*` endpoints functional and tested
- ✅ Batch evaluation from QA files working
- ✅ Export results to JSON
- ✅ Comprehensive documentation
- ✅ Sample test dataset ready

**Files Created:**
- api/metrics.py
- api/models_eval.py
- api/evaluation.py
- test_datasets/eval_qa.json
- TESTING_GUIDE.md
- EVALUATION_IMPLEMENTATION_SUMMARY.md

---

### Phase 4: Debug & Testing Tools (Priority: Medium)

**Estimated Time:** 2-3 days

**Tasks:**
1. Create `api/debug.py`:
   - Pipeline inspection functions
   - Query analysis utilities
   - Edge case test generators

2. Implement endpoints:
   - ✅ POST /debug/retrieval
   - ✅ GET /debug/document/{id}
   - ✅ POST /debug/query-analysis
   - ✅ POST /test/edge-cases

3. Create test data generators:
   - Edge case query templates
   - Synthetic error scenarios

**Deliverables:**
- Deep pipeline visibility
- Automated edge case testing
- Query difficulty prediction

---

### Phase 5: Analytics & Monitoring (Priority: Medium-Low)

**Estimated Time:** 3-4 days

**Tasks:**
1. Create `api/analytics.py`:
   - Query logging middleware
   - Usage statistics aggregation
   - Performance metric collection

2. Implement storage:
   - Query log database (SQLite or JSON)
   - Performance metrics time series
   - Document usage tracking

3. Implement endpoints:
   - ✅ GET /analytics/query-stats
   - ✅ GET /analytics/document-usage
   - ✅ GET /analytics/performance
   - ✅ POST /analytics/export

**Deliverables:**
- Real-time analytics dashboard data
- Exportable reports
- Performance monitoring

---

### Phase 6: Error Testing (Priority: Low)

**Estimated Time:** 1-2 days

**Tasks:**
1. Create `api/error_testing.py`:
   - Error injection framework
   - Validation utilities
   - Recovery testing

2. Implement endpoints:
   - ✅ POST /test/error-injection
   - ✅ GET /test/failed-jobs

**Deliverables:**
- Comprehensive error testing
- Failure analysis tools

---

## 7. File Structure

```
BiG-RAG/
├── api/
│   ├── __init__.py
│   ├── jobs.py                    # Existing
│   ├── registry.py                # Existing
│   ├── utils.py                   # Existing
│   ├── kg_utils.py                # Existing
│   ├── models.py                  # Existing
│   │
│   ├── evaluation.py              # NEW - Phase 3
│   ├── models_eval.py             # NEW - Phase 3
│   ├── metrics.py                 # NEW - Phase 3
│   │
│   ├── debug.py                   # NEW - Phase 4
│   ├── testing.py                 # NEW - Phase 4
│   ├── query_analysis.py          # NEW - Phase 4
│   │
│   ├── analytics.py               # NEW - Phase 5
│   ├── monitoring.py              # NEW - Phase 5
│   ├── query_logger.py            # NEW - Phase 5
│   │
│   └── error_testing.py           # NEW - Phase 6
│
├── evaluation_results/            # NEW - Store evaluation outputs
│   ├── eval_2025_10_30.json
│   └── comparison_results.json
│
├── analytics_data/                # NEW - Analytics storage
│   ├── query_logs.db              # SQLite for query logs
│   └── performance_metrics.json
│
├── exports/                       # NEW - Export downloads
│   └── query_logs_2025_10_30.csv
│
└── test_datasets/                 # NEW - Test QA sets
    ├── eval_qa.json
    └── edge_cases.json
```

---

## 8. Sample Test Datasets

### Format for QA Test Files

**test_datasets/eval_qa.json:**
```json
{
  "name": "BiG-RAG Evaluation Set",
  "version": "1.0",
  "dataset": "demo_test",
  "total_questions": 50,
  "questions": [
    {
      "id": "q001",
      "question": "What is artificial intelligence?",
      "ground_truth_answer": "Artificial intelligence is the simulation of human intelligence processes by machines.",
      "ground_truth_docs": ["doc_001", "doc_003"],
      "difficulty": "easy",
      "type": "definition",
      "category": "AI_basics"
    },
    {
      "id": "q002",
      "question": "How does deep learning differ from traditional machine learning?",
      "ground_truth_answer": "Deep learning uses neural networks with multiple layers to learn hierarchical representations, while traditional ML relies on hand-crafted features.",
      "ground_truth_docs": ["doc_005", "doc_008", "doc_012"],
      "difficulty": "medium",
      "type": "comparison",
      "category": "ML_concepts"
    }
  ]
}
```

---

## 9. Usage Examples

### Example 1: Evaluate Retrieval Quality

```bash
# Evaluate retrieval on test set
curl -X POST "http://localhost:8001/eval/retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "question": "What is BiG-RAG?",
        "ground_truth_docs": ["doc_001"]
      }
    ],
    "dataset": "demo_test",
    "mode": "hybrid",
    "top_k": 5,
    "metrics": ["precision", "recall", "mrr"]
  }'
```

### Example 2: Compare Retrieval Modes

```bash
# Compare all retrieval modes
curl -X POST "http://localhost:8001/eval/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "question": "Explain machine learning",
        "ground_truth_docs": ["doc_005"]
      }
    ],
    "configurations": [
      {"name": "hybrid", "mode": "hybrid", "top_k": 5},
      {"name": "local", "mode": "local", "top_k": 5},
      {"name": "global", "mode": "global", "top_k": 5}
    ]
  }'
```

### Example 3: Debug Retrieval Pipeline

```bash
# Inspect what's happening inside retrieval
curl -X POST "http://localhost:8001/debug/retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "dataset": "demo_test",
    "mode": "hybrid",
    "include_scores": true
  }'
```

### Example 4: Batch Evaluation from File

```bash
# Run evaluation on entire test set
curl -X POST "http://localhost:8001/eval/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_file": "test_datasets/eval_qa.json",
    "data_source": "demo_test",
    "mode": "hybrid",
    "metrics": ["em", "f1", "precision", "recall"],
    "save_results": true
  }'
```

### Example 5: Test Edge Cases

```bash
# Run automated edge case tests
curl -X POST "http://localhost:8001/test/edge-cases" \
  -H "Content-Type: application/json" \
  -d '{
    "test_scenarios": [
      "empty_query",
      "very_long_query",
      "special_characters"
    ],
    "dataset": "demo_test"
  }'
```

### Example 6: View Analytics

```bash
# Get query statistics
curl "http://localhost:8001/analytics/query-stats?time_range=7d"

# Get document usage
curl "http://localhost:8001/analytics/document-usage?dataset=demo_test"

# Get performance metrics
curl "http://localhost:8001/analytics/performance?time_range=24h"
```

---

## 10. Benefits

### For Developers
- ✅ **Automated Testing**: Comprehensive test coverage without manual work
- ✅ **Error Visibility**: Clear understanding of failure modes
- ✅ **Performance Monitoring**: Track and optimize system performance
- ✅ **Debug Tools**: Deep insight into pipeline internals

### For Researchers
- ✅ **Metric Calculation**: Standard IR and NLP metrics automatically computed
- ✅ **Comparative Analysis**: Easy comparison of different configurations
- ✅ **Reproducibility**: Consistent evaluation methodology
- ✅ **Export Capabilities**: Analysis in external tools (Excel, Python, R)

### For Production
- ✅ **Quality Assurance**: Continuous monitoring of retrieval quality
- ✅ **Usage Analytics**: Understand user behavior and popular queries
- ✅ **Cost Tracking**: Monitor API usage and costs
- ✅ **Early Warning**: Detect degradation before users complain

---

## 11. Next Steps

1. **Review this plan** - Ensure it covers your testing needs
2. **Prioritize phases** - Which phase to implement first?
3. **Prepare test data** - Create QA test sets for evaluation
4. **Start with Phase 3** - Core evaluation endpoints
5. **Iterate based on feedback** - Add/modify endpoints as needed

---

## 8. Upload Endpoint - Auto KG Building

### How Does /upload Work?

**Short Answer:** YES - When you upload a .txt or .md file via `/upload`, the system automatically builds the Knowledge Graph and creates all necessary indexing. You don't need to do anything else!

### Automatic Processing Pipeline

When you POST to `/upload` with a markdown or text file:

```bash
curl -X POST "http://localhost:8001/upload" \
  -F "file=@my_document.md" \
  -F "dataset=demo_test"
```

**The system automatically:**

#### Step 1: File Validation & Storage
- ✅ Validates file type (.txt or .md)
- ✅ For .md files: Converts markdown to plain text (removes formatting)
- ✅ Generates unique document ID (MD5 hash)
- ✅ Saves to `datasets/demo_test/raw/corpus.jsonl`

#### Step 2: Background Job Creation
- ✅ Creates a processing job with unique job_id
- ✅ Returns immediately with job_id for tracking
- ✅ Processing happens asynchronously in background

#### Step 3: Automatic Knowledge Graph Building
**No user action required - these all happen automatically:**

1. **Chunking:**
   - Splits document into chunks (1200 tokens with 100 overlap)
   - Preserves semantic boundaries

2. **Entity Extraction:**
   - Uses GPT-4o-mini for entity extraction
   - Applies gleaning (multiple rounds for completeness)
   - Extracts entities with types (PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, etc.)

3. **Relation Extraction:**
   - Extracts relationships between entities (n-ary relations)
   - Creates bipartite graph edges (Document ↔ Entity ↔ Relation)

4. **Embedding Generation:**
   - Generates embeddings for:
     - Text chunks (OpenAI text-embedding-3-large, 1536 dims)
     - Entities
     - Relations

5. **FAISS Indexing:**
   - Updates FAISS indices:
     - `expr/demo_test/index.bin` (chunk embeddings)
     - `expr/demo_test/index_entity.bin` (entity embeddings)
     - `expr/demo_test/index_bipartite_edge.bin` (relation embeddings)

6. **Metadata Storage:**
   - Updates JSON metadata stores:
     - `expr/demo_test/kv_store_text_chunks.json`
     - `expr/demo_test/kv_store_entities.json`
     - `expr/demo_test/kv_store_bipartite_edges.json`

#### Step 4: Ready for Retrieval
- ✅ Document is immediately available for `/search` and `/ask` queries
- ✅ Integrated into existing Knowledge Graph (incremental update)
- ✅ No restart required - BiGRAG hot-reloads indices

### How to Track Upload Progress

```bash
# Step 1: Upload file
RESPONSE=$(curl -X POST "http://localhost:8001/upload" \
  -F "file=@my_document.md" \
  -F "dataset=demo_test")

# Extract job_id from response
JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

# Step 2: Check status
curl "http://localhost:8001/status/$JOB_ID"
```

**Status Response:**
```json
{
  "job_id": "job-abc123",
  "status": "processing",
  "stage": "extracting_entities",
  "progress": 60,
  "created_at": "2025-10-31T10:00:00Z",
  "updated_at": "2025-10-31T10:02:30Z",
  "result": null
}
```

**Possible statuses:**
- `pending`: Job queued but not started
- `processing`: Currently building KG (stages: chunking → extracting_entities → extracting_relations → embedding → indexing)
- `completed`: Ready for retrieval
- `failed`: Error occurred (check error message)

### What You DON'T Need to Do

❌ **You don't need to:**
- Run `script_build.py` manually
- Regenerate FAISS indices
- Restart the server
- Manually add to corpus.jsonl
- Configure entity extraction
- Set up embeddings

✅ **Everything is automatic!**

### Dataset Structure

When you upload to `dataset=demo_test`:

**Before upload:**
```
datasets/demo_test/
├── raw/
│   └── corpus.jsonl          # 12 existing documents
expr/demo_test/
├── kv_store_*.json           # Existing metadata
└── index*.bin                # Existing FAISS indices
```

**After upload:**
```
datasets/demo_test/
├── raw/
│   └── corpus.jsonl          # 13 documents (your new doc appended)
expr/demo_test/
├── kv_store_*.json           # Updated metadata (includes new entities/relations)
└── index*.bin                # Updated FAISS indices (includes new embeddings)
```

### Key Points

1. **One Dataset = One Unified KG**
   - Each upload to `demo_test` extends the same Knowledge Graph
   - Entities from new documents can connect to existing entities
   - Cross-document retrieval works immediately

2. **Incremental Updates**
   - New documents don't rebuild the entire graph
   - Only new entities/relations are extracted
   - FAISS indices updated incrementally

3. **No Downtime**
   - Server keeps running while processing
   - Existing queries work normally
   - New document available as soon as status = "completed"

4. **Requirements**
   - OpenAI API key must be set (for GPT-4o-mini entity extraction)
   - Server must be running (`python script_api.py --data_source demo_test`)
   - Sufficient API quota (entity extraction makes multiple LLM calls)

### Example: Full Upload Workflow

```bash
# 1. Upload a new document
curl -X POST "http://localhost:8001/upload" \
  -F "file=@research_paper.md" \
  -F "dataset=demo_test"

# Response:
# {
#   "message": "File uploaded successfully",
#   "document_id": "upload-xyz789",
#   "job_id": "job-abc123",
#   "filename": "research_paper.md"
# }

# 2. Monitor processing (poll every 10 seconds)
watch -n 10 'curl -s "http://localhost:8001/status/job-abc123" | jq'

# 3. When status="completed", test retrieval
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is the main contribution of this research paper?"],
    "mode": "hybrid",
    "top_k": 5
  }'

# 4. Document is now part of the Knowledge Graph!
```

### Troubleshooting

**Q: Upload successful but status stuck at "pending"?**
- Check server logs for errors
- Ensure OpenAI API key is valid
- Check API quota limits

**Q: Status shows "failed"?**
- Get error details: `curl "http://localhost:8001/status/job-abc123"`
- Common causes:
  - OpenAI API rate limit
  - Invalid API key
  - Document too large (>100MB)
  - Markdown parsing error

**Q: How long does processing take?**
- Small document (1-5 pages): 30-90 seconds
- Medium document (10-20 pages): 2-5 minutes
- Large document (50+ pages): 5-15 minutes
- Depends on: document size, number of entities, API latency

**Q: Can I upload multiple files simultaneously?**
- Yes! Each gets a separate job_id
- Jobs process in parallel (up to system limits)

**Q: What file formats are supported?**
- Currently: `.txt` and `.md` (markdown)
- Planned: `.pdf`, `.docx`, `.html`

---

## Conclusion

This comprehensive plan provides everything needed for:
- ✅ Measuring retrieval and answer quality
- ✅ Systematic testing and debugging
- ✅ Error validation and handling
- ✅ Performance monitoring and optimization
- ✅ Usage analytics and insights
- ✅ **Automatic Knowledge Graph building from uploaded documents**

**Current Status:**
- ✅ **Phase 1-2 Complete:** Document Management (upload, insert, delete, list)
- ✅ **Phase 3 Complete:** Evaluation Endpoints (retrieval, answer, compare, batch evaluation)
- 📋 **Phase 4 Planned:** Debug & Testing Tools
- 📋 **Phase 5 Planned:** Analytics & Monitoring
- 📋 **Phase 6 Planned:** Error Testing

**Total Estimated Time:** 9-13 days for full implementation
**Completed So Far:** 4 days (Phases 1-3)
**Remaining:** 5-9 days (Phases 4-6)

**Recommended Priority:**
1. ✅ Phase 3 (Evaluation Core) - **COMPLETE**
2. Phase 4 (Debug Tools) - **Next**
3. Phase 5 (Analytics)
4. Phase 6 (Error Testing)

Ready to proceed with Phase 4? 🚀
