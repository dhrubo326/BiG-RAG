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

from .metrics import (
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
    relation_match: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluate retrieval for a single query

    Args:
        query: Query text
        ground_truth_docs: List of relevant document IDs (full_doc_id format)
        rag_instance: BiGRAG instance
        mode: Retrieval mode
        top_k: Number of documents to retrieve
        metrics: Metrics to calculate
        entity_match: Pre-computed entity matches (for FlagEmbedding mode)
        relation_match: Pre-computed relation matches (for FlagEmbedding mode)

    Returns:
        Dict with retrieved docs and metrics

    Note:
        Ground truth doc IDs must be full_doc_id values (e.g., "doc-abc123...")
        from corpus.jsonl, NOT chunk IDs.
    """
    # Query the RAG system (track latency)
    start_time = time.time()

    result = await rag_instance.aquery(
        query,
        param=QueryParam(
            mode=mode,
            only_need_context=True,
            top_k=top_k
        ),
        entity_match=entity_match,
        relation_match=relation_match
    )

    latency_ms = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Extract retrieved document IDs
    # BiGRAG returns: {"<knowledge>": text, "<coherence>": score, "<source_ids>": [chunk_ids]}
    retrieved_docs = []  # List of unique document IDs (maintains order)
    retrieved_doc_set = set()  # Set for deduplication

    if result:
        for context in result:
            # Get source IDs (chunk IDs) from the knowledge context
            source_ids = context.get('<source_ids>', [])
            if not source_ids:
                continue

            for chunk_id in source_ids:
                if not chunk_id:
                    continue

                # Look up the chunk to get its full_doc_id
                try:
                    chunk_data = await rag_instance.text_chunks.get_by_id(chunk_id)

                    if chunk_data and 'full_doc_id' in chunk_data:
                        doc_id = chunk_data['full_doc_id']
                        if doc_id and doc_id not in retrieved_doc_set:
                            retrieved_docs.append(doc_id)
                            retrieved_doc_set.add(doc_id)
                    else:
                        # ERROR: Chunk missing full_doc_id - this should not happen
                        # Log warning but continue (don't crash evaluation)
                        from bigrag.utils import logger
                        logger.warning(
                            f"[Evaluation] Chunk '{chunk_id}' missing 'full_doc_id' field. "
                            f"This chunk will be ignored in retrieval metrics. "
                            f"Check your data indexing pipeline."
                        )

                except Exception as e:
                    # ERROR: Failed to lookup chunk
                    from bigrag.utils import logger
                    logger.error(
                        f"[Evaluation] Failed to lookup chunk '{chunk_id}': {e}. "
                        f"This chunk will be ignored in retrieval metrics."
                    )

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
        "metrics": metric_scores,
        "latency_ms": round(latency_ms, 2)
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
    # Check corpus size to warn about trivial evaluation
    from bigrag.utils import logger
    try:
        all_chunk_ids = await rag_instance.text_chunks.all_keys()

        # Get unique document count from chunks
        unique_docs = set()
        for chunk_id in all_chunk_ids[:100]:  # Sample first 100 chunks
            chunk = await rag_instance.text_chunks.get_by_id(chunk_id)
            if chunk and 'full_doc_id' in chunk:
                unique_docs.add(chunk['full_doc_id'])

        doc_count = len(unique_docs)

        if doc_count <= 5:
            logger.warning(
                f"\n{'='*70}\n"
                f"⚠️  EVALUATION WARNING: Trivial Dataset Detected\n"
                f"{'='*70}\n"
                f"  Corpus contains only ~{doc_count} documents.\n"
                f"  With so few documents, retrieval will likely return ground truth\n"
                f"  documents for ANY query (even nonsense queries), resulting in\n"
                f"  artificially high metrics (near 1.0).\n"
                f"\n"
                f"  For meaningful evaluation, use a dataset with:\n"
                f"  - At least 100 documents (minimum)\n"
                f"  - Recommended: 1000+ documents\n"
                f"\n"
                f"  Current dataset appears to be 'demo_test', which is only for\n"
                f"  testing API functionality, NOT for evaluation.\n"
                f"{'='*70}\n"
            )
    except Exception as e:
        logger.debug(f"Could not check corpus size: {e}")

    start_time = time.time()
    per_query_results = []
    all_metrics = []
    all_latencies = []

    for query_data in queries:
        question = query_data['question']
        ground_truth = query_data['ground_truth_docs']

        # For FlagEmbedding mode, pre-compute matches
        entity_match = None
        relation_match = None
        if embedding_manager and hasattr(embedding_manager, 'mode'):
            if embedding_manager.mode == "flagembedding":
                entity_match = await embedding_manager.search_entities(question, top_k)
                relation_match = await embedding_manager.search_relations(question, top_k)

        # Evaluate single query
        result = await evaluate_single_retrieval(
            query=question,
            ground_truth_docs=ground_truth,
            rag_instance=rag_instance,
            mode=mode,
            top_k=top_k,
            metrics=metrics,
            entity_match=entity_match,
            relation_match=relation_match
        )

        per_query_results.append({
            "question": question,
            "retrieved_docs": result["retrieved_docs"],
            "relevant_retrieved": result["relevant_retrieved"],
            "metrics": result["metrics"],
            "latency_ms": result["latency_ms"]
        })

        all_metrics.append(result["metrics"])
        all_latencies.append(result["latency_ms"])

    # Aggregate metrics
    aggregate = aggregate_metrics(all_metrics)

    # Calculate latency statistics
    latency_stats = None
    if all_latencies:
        import numpy as np
        latencies_array = np.array(all_latencies)
        latency_stats = {
            "mean_ms": round(float(np.mean(latencies_array)), 2),
            "median_ms": round(float(np.median(latencies_array)), 2),
            "min_ms": round(float(np.min(latencies_array)), 2),
            "max_ms": round(float(np.max(latencies_array)), 2),
            "p95_ms": round(float(np.percentile(latencies_array, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies_array, 99)), 2),
            "std_ms": round(float(np.std(latencies_array)), 2)
        }

    evaluation_time = time.time() - start_time

    # Check if results look suspiciously perfect (indicates trivial evaluation)
    perfect_scores = 0
    for result in per_query_results:
        # Check if all metrics are close to 1.0
        metrics_values = list(result["metrics"].values())
        if metrics_values and all(v >= 0.95 for v in metrics_values):
            perfect_scores += 1

    if perfect_scores == len(queries) and len(queries) > 0:
        from bigrag.utils import logger
        logger.warning(
            f"\n⚠️  SUSPICIOUS RESULTS: All {len(queries)} queries achieved near-perfect scores.\n"
            f"   This likely indicates:\n"
            f"   1. Corpus is too small (retrieval always finds ground truth)\n"
            f"   2. Ground truth IDs are incorrect/duplicated\n"
            f"   3. Dataset is trivial (e.g., demo_test with 1 document)\n"
            f"\n"
            f"   For reliable evaluation, use a real dataset with 100+ documents.\n"
        )

    return {
        "total_queries": len(queries),
        "metrics": aggregate,
        "per_query_results": per_query_results,
        "evaluation_time": round(evaluation_time, 2),
        "latency_stats": latency_stats
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
    relation_match: Optional[Any] = None
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
        relation_match: Pre-computed relation matches

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
            relation_match=relation_match
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
    start_time = time.time()  # Track comparison time
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

    comparison_time = time.time() - start_time

    return {
        "results": comparison_results,  # Changed from "comparison_results"
        "best_configuration": best_configuration,
        "ranking": ranking,
        "comparison_time": comparison_time  # Added timing metadata
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
                relation_match = None
                if embedding_manager and hasattr(embedding_manager, 'mode'):
                    if embedding_manager.mode == "flagembedding":
                        entity_match = await embedding_manager.search_entities(question, top_k)
                        relation_match = await embedding_manager.search_relations(question, top_k)

                retrieval_result = await evaluate_single_retrieval(
                    query=question,
                    ground_truth_docs=ground_truth_docs,
                    rag_instance=rag_instance,
                    mode=mode,
                    top_k=top_k,
                    metrics=metrics,
                    entity_match=entity_match,
                    relation_match=relation_match
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
                    relation_match=relation_match if ground_truth_docs else None
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
