"""
Evaluation Logic for BiG-RAG

Helper functions for running evaluations:
- Retrieval evaluation
- Answer quality evaluation
- Comparative evaluation
- Batch processing
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from bigrag import BiGRAG, QueryParam

from api.metrics import (
    calculate_retrieval_metrics,
    calculate_answer_metrics,
    aggregate_metrics
)


# ==============================================================================
# Retrieval Evaluation
# ==============================================================================

async def evaluate_single_retrieval(
    query: str,
    ground_truth_docs: List[str],
    rag_instance: BiGRAG,
    mode: str = "hybrid",
    top_k: int = 5,
    metrics: Optional[List[str]] = None,
    entity_match: Optional[Any] = None,
    bipartite_edge_match: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluate retrieval for a single query

    Args:
        query: Query text
        ground_truth_docs: List of relevant document IDs
        rag_instance: BiGRAG instance
        mode: Retrieval mode
        top_k: Number of documents to retrieve
        metrics: Metrics to calculate
        entity_match: Pre-computed entity matches (for FlagEmbedding mode)
        bipartite_edge_match: Pre-computed edge matches (for FlagEmbedding mode)

    Returns:
        Dict with retrieved docs and metrics
    """
    # Query the RAG system
    result = await rag_instance.aquery(
        query,
        param=QueryParam(
            mode=mode,
            only_need_context=True,
            top_k=top_k
        ),
        entity_match=entity_match,
        bipartite_edge_match=bipartite_edge_match
    )

    # Extract retrieved document IDs
    # BiGRAG now returns: {"<knowledge>": text, "<coherence>": score, "<source_ids>": [chunk_ids]}
    retrieved_docs = []
    if result:
        for context in result:
            # Get source IDs (chunk IDs) from the knowledge context
            source_ids = context.get('<source_ids>', [])
            if source_ids:
                for chunk_id in source_ids:
                    if not chunk_id:
                        continue

                    # Look up the chunk to get its full_doc_id
                    try:
                        chunk_data = await rag_instance.text_chunks.get_by_id(chunk_id)
                        if chunk_data and 'full_doc_id' in chunk_data:
                            doc_id = chunk_data['full_doc_id']
                            if doc_id and doc_id not in retrieved_docs:
                                retrieved_docs.append(doc_id)
                        else:
                            # Fallback: use chunk_id if no full_doc_id found
                            if chunk_id not in retrieved_docs:
                                retrieved_docs.append(chunk_id)
                    except Exception as e:
                        # If lookup fails, use chunk_id as fallback
                        if chunk_id not in retrieved_docs:
                            retrieved_docs.append(chunk_id)

    # Calculate metrics
    metric_scores = calculate_retrieval_metrics(
        retrieved=retrieved_docs,
        relevant=ground_truth_docs,
        k=top_k,
        metrics=metrics
    )

    # Find which relevant docs were retrieved
    retrieved_set = set(retrieved_docs)
    relevant_set = set(ground_truth_docs)
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_set]

    return {
        "retrieved_docs": retrieved_docs,
        "relevant_retrieved": relevant_retrieved,
        "metrics": metric_scores
    }


async def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    rag_instance: BiGRAG,
    mode: str = "hybrid",
    top_k: int = 5,
    metrics: Optional[List[str]] = None,
    embedding_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluate retrieval for multiple queries

    Args:
        queries: List of query dicts with 'question' and 'ground_truth_docs'
        rag_instance: BiGRAG instance
        mode: Retrieval mode
        top_k: Number of documents to retrieve
        metrics: Metrics to calculate
        embedding_manager: EmbeddingManager for FlagEmbedding mode

    Returns:
        Dict with aggregate metrics and per-query results
    """
    start_time = time.time()
    per_query_results = []
    all_metrics = []

    for query_data in queries:
        question = query_data['question']
        ground_truth = query_data['ground_truth_docs']

        # For FlagEmbedding mode, pre-compute matches
        entity_match = None
        edge_match = None
        if embedding_manager and hasattr(embedding_manager, 'mode'):
            if embedding_manager.mode == "flagembedding":
                entity_match = await embedding_manager.search_entities(question, top_k)
                edge_match = await embedding_manager.search_edges(question, top_k)

        # Evaluate single query
        result = await evaluate_single_retrieval(
            query=question,
            ground_truth_docs=ground_truth,
            rag_instance=rag_instance,
            mode=mode,
            top_k=top_k,
            metrics=metrics,
            entity_match=entity_match,
            bipartite_edge_match=edge_match
        )

        per_query_results.append({
            "question": question,
            "retrieved_docs": result["retrieved_docs"],
            "relevant_retrieved": result["relevant_retrieved"],
            "metrics": result["metrics"]
        })

        all_metrics.append(result["metrics"])

    # Aggregate metrics
    aggregate = aggregate_metrics(all_metrics)

    evaluation_time = time.time() - start_time

    return {
        "total_queries": len(queries),
        "metrics": aggregate,
        "per_query_results": per_query_results,
        "evaluation_time": round(evaluation_time, 2)
    }


# ==============================================================================
# Answer Quality Evaluation
# ==============================================================================

async def evaluate_single_answer(
    question: str,
    ground_truth: str,
    rag_instance: BiGRAG,
    llm_func: Any,
    use_rag: bool = True,
    mode: str = "hybrid",
    top_k: int = 5,
    metrics: Optional[List[str]] = None,
    entity_match: Optional[Any] = None,
    bipartite_edge_match: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluate answer generation for a single question

    Args:
        question: Question to ask
        ground_truth: Expected answer
        rag_instance: BiGRAG instance
        llm_func: LLM function for generation
        use_rag: Whether to use retrieval
        mode: Retrieval mode
        top_k: Number of contexts to retrieve
        metrics: Answer metrics to calculate
        entity_match: Pre-computed entity matches
        bipartite_edge_match: Pre-computed edge matches

    Returns:
        Dict with generated answer and metrics
    """
    start_time = time.time()

    # Generate answer
    if use_rag:
        # Query with RAG
        result = await rag_instance.aquery(
            question,
            param=QueryParam(
                mode=mode,
                only_need_context=False,  # Generate answer
                top_k=top_k
            ),
            entity_match=entity_match,
            bipartite_edge_match=bipartite_edge_match
        )

        # Extract answer
        if isinstance(result, str):
            predicted_answer = result
            num_contexts = 0
        elif isinstance(result, dict):
            predicted_answer = result.get('response', result.get('answer', ''))
            num_contexts = len(result.get('context', []))
        else:
            predicted_answer = str(result)
            num_contexts = 0
    else:
        # Generate without RAG (direct LLM call)
        prompt = f"Answer this question concisely: {question}"
        predicted_answer = await llm_func(prompt, max_tokens=200)
        num_contexts = 0

    generation_time = time.time() - start_time

    # Calculate metrics
    metric_scores = calculate_answer_metrics(
        prediction=predicted_answer,
        ground_truth=ground_truth,
        metrics=metrics
    )

    return {
        "predicted_answer": predicted_answer,
        "metrics": metric_scores,
        "retrieval_used": use_rag,
        "num_contexts_used": num_contexts,
        "generation_time": round(generation_time, 2)
    }


# ==============================================================================
# Comparative Evaluation
# ==============================================================================

async def compare_configurations(
    queries: List[Dict[str, Any]],
    configurations: List[Dict[str, Any]],
    rag_instance: BiGRAG,
    metrics: Optional[List[str]] = None,
    embedding_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Compare different retrieval configurations

    Args:
        queries: List of query dicts with 'question' and 'ground_truth_docs'
        configurations: List of config dicts with 'name', 'mode', 'top_k'
        rag_instance: BiGRAG instance
        metrics: Metrics to calculate
        embedding_manager: EmbeddingManager for FlagEmbedding mode

    Returns:
        Dict with comparison results and ranking
    """
    comparison_results = {}

    for config in configurations:
        config_name = config['name']
        mode = config['mode']
        top_k = config.get('top_k', 5)

        # Run evaluation for this configuration
        eval_results = await evaluate_retrieval(
            queries=queries,
            rag_instance=rag_instance,
            mode=mode,
            top_k=top_k,
            metrics=metrics,
            embedding_manager=embedding_manager
        )

        # Store aggregate metrics
        comparison_results[config_name] = eval_results['metrics']

    # Rank configurations by first metric (or default to precision)
    if metrics and comparison_results:
        # Use first metric for ranking
        ranking_metric = f"{metrics[0]}@{configurations[0].get('top_k', 5)}" if metrics[0] in ['precision', 'recall', 'f1', 'ndcg'] else metrics[0]

        # Find actual metric name in results
        for result in comparison_results.values():
            if result:
                ranking_metric = list(result.keys())[0]
                break

        # Rank by metric score (descending)
        ranked = sorted(
            comparison_results.items(),
            key=lambda x: x[1].get(ranking_metric, 0),
            reverse=True
        )

        best_configuration = ranked[0][0] if ranked else None
        ranking = [name for name, _ in ranked]
    else:
        best_configuration = None
        ranking = []

    return {
        "comparison_results": comparison_results,
        "best_configuration": best_configuration,
        "ranking": ranking
    }


# ==============================================================================
# Batch Evaluation from File
# ==============================================================================

async def load_qa_dataset(file_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load QA dataset from JSON file

    Args:
        file_path: Path to dataset file
        limit: Maximum number of questions to load

    Returns:
        List of question dicts

    Expected format:
        {
            "questions": [
                {
                    "question": "...",
                    "ground_truth_answer": "...",
                    "ground_truth_docs": ["doc_001"]
                }
            ]
        }
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support multiple formats
    if isinstance(data, list):
        questions = data
    elif 'questions' in data:
        questions = data['questions']
    elif 'test_cases' in data:
        questions = data['test_cases']
    else:
        raise ValueError("Invalid dataset format. Expected 'questions' or 'test_cases' field")

    # Apply limit
    if limit:
        questions = questions[:limit]

    return questions


async def save_evaluation_results(
    results: Dict[str, Any],
    output_file: str
) -> str:
    """
    Save evaluation results to file

    Args:
        results: Evaluation results dict
        output_file: Output file path

    Returns:
        Path to saved file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return str(output_path)


async def batch_evaluate(
    dataset_file: str,
    rag_instance: BiGRAG,
    llm_func: Any,
    mode: str = "hybrid",
    top_k: int = 5,
    metrics: Optional[List[str]] = None,
    use_llm: bool = True,
    save_results: bool = False,
    output_file: Optional[str] = None,
    limit: Optional[int] = None,
    embedding_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run batch evaluation from QA dataset file

    Args:
        dataset_file: Path to QA dataset file
        rag_instance: BiGRAG instance
        llm_func: LLM function for generation
        mode: Retrieval mode
        top_k: Number of documents to retrieve
        metrics: Metrics to calculate
        use_llm: Whether to generate answers (vs just retrieval)
        save_results: Whether to save detailed results
        output_file: Output file path
        limit: Limit number of questions
        embedding_manager: EmbeddingManager for FlagEmbedding mode

    Returns:
        Dict with evaluation results
    """
    start_time = time.time()

    # Load dataset
    questions = await load_qa_dataset(dataset_file, limit=limit)

    total_questions = len(questions)
    processed = 0
    failed = 0

    retrieval_metrics_list = []
    answer_metrics_list = []

    detailed_results = []

    # Process each question
    for q_data in questions:
        try:
            question = q_data.get('question', '')
            ground_truth_answer = q_data.get('ground_truth_answer', q_data.get('ground_truth', ''))
            ground_truth_docs = q_data.get('ground_truth_docs', [])

            # Evaluate retrieval if ground truth docs provided
            if ground_truth_docs:
                entity_match = None
                edge_match = None
                if embedding_manager and hasattr(embedding_manager, 'mode'):
                    if embedding_manager.mode == "flagembedding":
                        entity_match = await embedding_manager.search_entities(question, top_k)
                        edge_match = await embedding_manager.search_edges(question, top_k)

                retrieval_result = await evaluate_single_retrieval(
                    query=question,
                    ground_truth_docs=ground_truth_docs,
                    rag_instance=rag_instance,
                    mode=mode,
                    top_k=top_k,
                    metrics=metrics,
                    entity_match=entity_match,
                    bipartite_edge_match=edge_match
                )
                retrieval_metrics_list.append(retrieval_result['metrics'])
            else:
                retrieval_result = None

            # Evaluate answer if LLM enabled and ground truth answer provided
            if use_llm and ground_truth_answer:
                answer_result = await evaluate_single_answer(
                    question=question,
                    ground_truth=ground_truth_answer,
                    rag_instance=rag_instance,
                    llm_func=llm_func,
                    use_rag=True,
                    mode=mode,
                    top_k=top_k,
                    metrics=metrics,
                    entity_match=entity_match if ground_truth_docs else None,
                    bipartite_edge_match=edge_match if ground_truth_docs else None
                )
                answer_metrics_list.append(answer_result['metrics'])
            else:
                answer_result = None

            # Store detailed result
            detailed_results.append({
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "ground_truth_docs": ground_truth_docs,
                "retrieval_result": retrieval_result,
                "answer_result": answer_result
            })

            processed += 1

        except Exception as e:
            failed += 1
            print(f"Failed to process question: {question[:50]}... Error: {e}")

    # Aggregate metrics
    aggregate_retrieval = aggregate_metrics(retrieval_metrics_list) if retrieval_metrics_list else {}
    aggregate_answer = aggregate_metrics(answer_metrics_list) if answer_metrics_list else {}

    total_time = time.time() - start_time

    # Prepare results
    results = {
        "dataset": dataset_file,
        "total_questions": total_questions,
        "processed": processed,
        "failed": failed,
        "metrics": {
            "retrieval": aggregate_retrieval,
            "answer": aggregate_answer
        },
        "performance": {
            "total_time": round(total_time, 2),
            "avg_time_per_query": round(total_time / processed, 2) if processed > 0 else 0,
            "total_llm_calls": processed if use_llm else 0,
            "total_embedding_calls": processed
        },
        "detailed_results": detailed_results
    }

    # Save results if requested
    results_saved_to = None
    if save_results and output_file:
        results_saved_to = await save_evaluation_results(results, output_file)

    results["results_saved_to"] = results_saved_to

    return results
