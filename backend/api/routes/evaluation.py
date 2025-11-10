"""
Evaluation routes
"""

from fastapi import APIRouter, HTTPException
from bigrag.utils import logger

from ..core.dependencies import RAGDep, LLMDep, EmbeddingDep
from ..models.models_eval import (
    RetrievalEvalRequest, RetrievalEvalResponse,
    AnswerEvalRequest, AnswerEvalResponse,
    CompareEvalRequest, CompareEvalResponse,
    BatchEvalRequest, BatchEvalResponse,
    BatchGenerateRequest, BatchGenerateResponse,
    EvaluateResultsRequest, EvaluateResultsResponse,
    PerQueryRetrievalResult, PerQuestionAnswerResult
)
from ..services.evaluation import (
    evaluate_retrieval, evaluate_single_answer,
    compare_configurations, batch_evaluate
)
from ..services.csv_evaluation import batch_generate_endpoint, evaluate_results_endpoint
from ..services.metrics import aggregate_metrics


router = APIRouter(prefix="/eval", tags=["Evaluation"])


@router.post("/retrieval", response_model=RetrievalEvalResponse)
async def evaluate_retrieval_endpoint(
    request: RetrievalEvalRequest,
    rag: RAGDep,
    embedding_manager: EmbeddingDep
):
    """
    Evaluate retrieval quality with ground truth documents.

    **Metrics Calculated:**
    - Precision@K: Fraction of retrieved docs that are relevant
    - Recall@K: Fraction of relevant docs that were retrieved
    - F1@K: Harmonic mean of precision and recall
    - MRR (Mean Reciprocal Rank): 1 / rank of first relevant doc
    - NDCG@K: Normalized Discounted Cumulative Gain

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/retrieval" \\
      -H "Content-Type: application/json" \\
      -d '{
        "queries": [{
          "question": "What is BiG-RAG?",
          "ground_truth_docs": ["doc_001"]
        }],
        "dataset": "demo_test",
        "mode": "hybrid",
        "top_k": 5,
        "metrics": ["precision", "recall", "mrr"]
      }'
    ```
    """
    try:
        # Convert queries to dict format
        queries_list = [
            {
                "question": q.question,
                "ground_truth_docs": q.ground_truth_docs
            }
            for q in request.queries
        ]

        # Run evaluation
        eval_results = await evaluate_retrieval(
            queries=queries_list,
            rag_instance=rag,
            mode=request.mode,
            top_k=request.top_k,
            metrics=request.metrics,
            embedding_manager=embedding_manager
        )

        # Convert to response model
        per_query_models = [
            PerQueryRetrievalResult(
                question=result["question"],
                retrieved_docs=result["retrieved_docs"],
                relevant_retrieved=result["relevant_retrieved"],
                metrics=result["metrics"],
                latency_ms=result.get("latency_ms")
            )
            for result in eval_results["per_query_results"]
        ]

        return RetrievalEvalResponse(
            success=True,
            total_queries=eval_results["total_queries"],
            metrics=eval_results["metrics"],
            per_query_results=per_query_models,
            evaluation_time=eval_results["evaluation_time"],
            latency_stats=eval_results.get("latency_stats")
        )

    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/answer", response_model=AnswerEvalResponse)
async def evaluate_answer_endpoint(
    request: AnswerEvalRequest,
    rag: RAGDep,
    llm_manager: LLMDep,
    embedding_manager: EmbeddingDep
):
    """
    Evaluate answer quality against ground truth.

    **Metrics Calculated:**
    - Exact Match (EM): Binary match after normalization
    - Token F1: Token-level F1 score
    - ROUGE-L: Longest Common Subsequence F1

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/answer" \\
      -H "Content-Type: application/json" \\
      -d '{
        "test_cases": [{
          "question": "What is AI?",
          "ground_truth": "Artificial Intelligence",
          "use_rag": true
        }],
        "dataset": "demo_test",
        "metrics": ["em", "f1", "rouge_l"]
      }'
    ```
    """
    try:
        # Get LLM function
        llm_func = llm_manager.get_model_func(request.llm_provider, request.model)

        per_question_results = []
        all_metrics = []

        for test_case in request.test_cases:
            # Pre-compute entity/relation matches if FlagEmbedding
            entity_match = None
            relation_match = None
            if embedding_manager.mode == "flagembedding" and test_case.use_rag:
                entity_match = await embedding_manager.search_entities(test_case.question, request.top_k)
                relation_match = await embedding_manager.search_relations(test_case.question, request.top_k)

            # Evaluate single answer
            result = await evaluate_single_answer(
                question=test_case.question,
                ground_truth=test_case.ground_truth,
                rag_instance=rag,
                llm_func=llm_func,
                use_rag=test_case.use_rag,
                mode=request.mode,
                top_k=request.top_k,
                metrics=request.metrics,
                entity_match=entity_match,
                relation_match=relation_match
            )

            per_question_results.append(PerQuestionAnswerResult(
                question=test_case.question,
                ground_truth=test_case.ground_truth,
                predicted_answer=result["predicted_answer"],
                metrics=result["metrics"],
                retrieval_used=result["retrieval_used"],
                num_contexts_used=result["num_contexts_used"],
                generation_time=result["generation_time"]
            ))

            all_metrics.append(result["metrics"])

        # Aggregate metrics
        aggregate = aggregate_metrics(all_metrics)

        return AnswerEvalResponse(
            success=True,
            total_questions=len(request.test_cases),
            aggregate_metrics=aggregate,
            per_question_results=per_question_results,
            total_time=sum(r.generation_time for r in per_question_results)
        )

    except Exception as e:
        logger.error(f"Answer evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/compare", response_model=CompareEvalResponse)
async def compare_configurations_endpoint(
    request: CompareEvalRequest,
    rag: RAGDep,
    embedding_manager: EmbeddingDep
):
    """
    Compare different retrieval configurations side-by-side.

    **Use Cases:**
    - Compare retrieval modes (hybrid vs local vs global vs naive)
    - Compare different top_k values
    - Find optimal configuration

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/compare" \\
      -H "Content-Type: application/json" \\
      -d '{
        "queries": [{
          "question": "What is ML?",
          "ground_truth_docs": ["doc_001"]
        }],
        "configurations": [
          {"name": "hybrid", "mode": "hybrid", "top_k": 5},
          {"name": "local", "mode": "local", "top_k": 5}
        ],
        "metrics": ["precision", "recall", "mrr"]
      }'
    ```
    """
    try:
        # Convert queries and configurations to dict format
        queries_list = [
            {
                "question": q.question,
                "ground_truth_docs": q.ground_truth_docs
            }
            for q in request.queries
        ]

        configs_list = [
            {
                "name": c.name,
                "mode": c.mode,
                "top_k": c.top_k
            }
            for c in request.configurations
        ]

        # Run comparison
        comparison_results = await compare_configurations(
            queries=queries_list,
            configurations=configs_list,
            rag_instance=rag,
            metrics=request.metrics,
            embedding_manager=embedding_manager
        )

        return CompareEvalResponse(
            success=True,
            total_queries=len(queries_list),
            configurations_tested=len(configs_list),
            results=comparison_results["results"],
            best_configuration=comparison_results["best_configuration"],
            comparison_time=comparison_results["comparison_time"]
        )

    except Exception as e:
        logger.error(f"Configuration comparison failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/batch", response_model=BatchEvalResponse)
async def batch_evaluate_endpoint(
    request: BatchEvalRequest,
    rag: RAGDep,
    llm_manager: LLMDep,
    embedding_manager: EmbeddingDep
):
    """
    Run batch evaluation with multiple questions.

    Combines retrieval and answer evaluation for comprehensive testing.

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/batch" \\
      -H "Content-Type: application/json" \\
      -d '{
        "test_cases": [
          {"question": "What is AI?", "ground_truth": "Artificial Intelligence"}
        ],
        "dataset": "demo_test",
        "evaluate_retrieval": true,
        "evaluate_answer": true
      }'
    ```
    """
    try:
        llm_func = llm_manager.get_model_func(request.llm_provider, request.model)

        eval_results = await batch_evaluate(
            test_cases=[
                {
                    "question": tc.question,
                    "ground_truth": tc.ground_truth,
                    "ground_truth_docs": tc.ground_truth_docs
                }
                for tc in request.test_cases
            ],
            rag_instance=rag,
            llm_func=llm_func,
            mode=request.mode,
            top_k=request.top_k,
            evaluate_retrieval=request.evaluate_retrieval,
            evaluate_answer=request.evaluate_answer,
            embedding_manager=embedding_manager
        )

        return BatchEvalResponse(
            success=True,
            total_test_cases=len(request.test_cases),
            retrieval_metrics=eval_results.get("retrieval_metrics"),
            answer_metrics=eval_results.get("answer_metrics"),
            per_case_results=eval_results.get("per_case_results", []),
            total_time=eval_results.get("total_time", 0.0)
        )

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@router.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate_from_csv(
    request: BatchGenerateRequest,
    rag: RAGDep,
    llm_manager: LLMDep,
    embedding_manager: EmbeddingDep
):
    """
    Batch process questions from CSV and generate answers.

    **Use Case:** Systematic evaluation on standard datasets.
    Separates generation from evaluation to avoid redundant processing.

    **Workflow:**
    1. Load questions from CSV
    2. For each question: retrieve context + generate answer
    3. Save results to CSV for later evaluation

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/batch_generate" \\
      -H "Content-Type: application/json" \\
      -d '{
        "questions_csv_path": "datasets/test_questions.csv",
        "output_csv_path": "results/answers.csv",
        "llm_provider": "openai",
        "model": "gpt-4o-mini"
      }'
    ```
    """
    try:
        results = await batch_generate_endpoint(
            questions_csv_path=request.questions_csv_path,
            output_csv_path=request.output_csv_path,
            rag_instance=rag,
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            llm_provider=request.llm_provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            enable_reranking=request.enable_reranking
        )

        return BatchGenerateResponse(**results)

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@router.post("/evaluate_results", response_model=EvaluateResultsResponse)
async def evaluate_csv_results(request: EvaluateResultsRequest):
    """
    Evaluate saved results CSV for accuracy.

    **Use Case:** Calculate metrics on previously generated answers.

    **Metrics:**
    - EM (Exact Match)
    - F1 (Token F1)
    - ROUGE-L (optional)
    - Answer Rate (for no-answer questions)

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/evaluate_results" \\
      -H "Content-Type: application/json" \\
      -d '{
        "results_csv_path": "results/answers.csv",
        "metrics": ["em", "f1", "rouge_l"],
        "export_formats": ["json", "csv", "markdown"]
      }'
    ```
    """
    try:
        results = await evaluate_results_endpoint(
            results_csv_path=request.results_csv_path,
            metrics=request.metrics,
            export_formats=request.export_formats,
            output_dir=request.output_dir
        )

        return EvaluateResultsResponse(**results)

    except Exception as e:
        logger.error(f"Results evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Results evaluation failed: {str(e)}")


