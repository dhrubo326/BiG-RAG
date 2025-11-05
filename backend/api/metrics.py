"""
Evaluation Metrics for BiG-RAG

Implements standard Information Retrieval and NLP metrics:
- Retrieval: Precision@K, Recall@K, F1@K, MRR, NDCG@K, MAP
- Answer Quality: Exact Match (EM), Token F1, ROUGE-L
"""

import re
import string
from typing import List, Set, Dict, Any, Optional
from collections import Counter
import numpy as np


# ==============================================================================
# Text Normalization
# ==============================================================================

def normalize_answer(text: str) -> str:
    """
    Normalize text for evaluation (following SQuAD/HotpotQA style)

    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Remove extra whitespace
    """
    # Handle None
    if text is None:
        return ""

    # Handle NaN (float type where value != value)
    if isinstance(text, float):
        # NaN is the only float where x != x
        if text != text:  # NaN check
            return ""
        # Convert other floats to string
        text = str(text)

    # Convert non-string types to string
    if not isinstance(text, str):
        text = str(text)

    # Handle empty strings
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text.strip()


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization after normalization"""
    normalized = normalize_answer(text)
    return normalized.split()


# ==============================================================================
# Retrieval Metrics
# ==============================================================================

def precision_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Precision@K: Fraction of retrieved documents that are relevant

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs
        k: Cutoff rank (if None, uses len(retrieved))

    Returns:
        Precision score (0.0 to 1.0)

    Example:
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = ['doc1', 'doc3', 'doc6']
        precision_at_k(retrieved, relevant, k=5) = 2/5 = 0.4
    """
    if not retrieved:
        return 0.0

    k = k or len(retrieved)
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)

    num_relevant = sum(1 for doc in retrieved_at_k if doc in relevant_set)
    return num_relevant / len(retrieved_at_k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Recall@K: Fraction of relevant documents that were retrieved

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs
        k: Cutoff rank (if None, uses len(retrieved))

    Returns:
        Recall score (0.0 to 1.0)

    Example:
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = ['doc1', 'doc3', 'doc6']
        recall_at_k(retrieved, relevant, k=5) = 2/3 = 0.667
    """
    if not relevant:
        return 0.0

    k = k or len(retrieved)
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)

    num_found = sum(1 for doc in retrieved_at_k if doc in relevant_set)
    return num_found / len(relevant)


def f1_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    F1@K: Harmonic mean of Precision@K and Recall@K

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs
        k: Cutoff rank (if None, uses len(retrieved))

    Returns:
        F1 score (0.0 to 1.0)
    """
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): 1 / rank of first relevant document

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs

    Returns:
        MRR score (0.0 to 1.0)

    Example:
        retrieved = ['doc2', 'doc1', 'doc3']
        relevant = ['doc1', 'doc3']
        MRR = 1/2 = 0.5 (first relevant doc at rank 2)
    """
    if not retrieved or not relevant:
        return 0.0

    relevant_set = set(relevant)

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def hit_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Hit@K (Success@K): Binary metric indicating if at least one relevant doc was retrieved

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs
        k: Cutoff rank (if None, uses len(retrieved))

    Returns:
        1.0 if at least one relevant doc in top-K, else 0.0

    Example:
        retrieved = ['doc2', 'doc4', 'doc1', 'doc5']
        relevant = ['doc1', 'doc3']
        hit_at_k(retrieved, relevant, k=3) = 1.0 (doc1 at rank 3)
        hit_at_k(retrieved, relevant, k=2) = 0.0 (no relevant docs in top 2)

    Note:
        This is a binary metric useful for evaluating whether the system
        retrieved ANY relevant information. It's less granular than Recall@K
        but easier to interpret for tasks where finding at least one relevant
        document is sufficient.
    """
    if not retrieved or not relevant:
        return 0.0

    k = k or len(retrieved)
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)

    # Check if any retrieved doc is relevant
    for doc_id in retrieved_at_k:
        if doc_id in relevant_set:
            return 1.0

    return 0.0


def dcg_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Discounted Cumulative Gain at K

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs
        k: Cutoff rank (if None, uses len(retrieved))

    Returns:
        DCG score
    """
    if not retrieved:
        return 0.0

    k = k or len(retrieved)
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_at_k, start=1):
        if doc_id in relevant_set:
            # Binary relevance: 1 if relevant, 0 otherwise
            relevance = 1.0
            dcg += relevance / np.log2(rank + 1)

    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Normalized Discounted Cumulative Gain at K

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs
        k: Cutoff rank (if None, uses len(retrieved))

    Returns:
        NDCG score (0.0 to 1.0)

    Example:
        retrieved = ['doc1', 'doc3', 'doc2', 'doc5']
        relevant = ['doc1', 'doc2']
        NDCG@4 ≈ 0.86
    """
    if not relevant:
        return 0.0

    k = k or len(retrieved)

    # Actual DCG
    dcg = dcg_at_k(retrieved, relevant, k)

    # Ideal DCG (all relevant docs at top)
    ideal_retrieved = list(relevant)[:k]
    idcg = dcg_at_k(ideal_retrieved, relevant, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """
    Average Precision: Mean of precision values at each relevant document

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs

    Returns:
        AP score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    precision_sum = 0.0
    num_relevant_found = 0

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            num_relevant_found += 1
            precision_at_rank = num_relevant_found / rank
            precision_sum += precision_at_rank

    if num_relevant_found == 0:
        return 0.0

    return precision_sum / len(relevant)


def mean_average_precision(queries_retrieved: List[List[str]],
                          queries_relevant: List[List[str]]) -> float:
    """
    Mean Average Precision (MAP): Mean of AP across multiple queries

    Args:
        queries_retrieved: List of retrieved document lists for each query
        queries_relevant: List of relevant document lists for each query

    Returns:
        MAP score (0.0 to 1.0)
    """
    if not queries_retrieved or not queries_relevant:
        return 0.0

    ap_scores = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(queries_retrieved, queries_relevant)
    ]

    return np.mean(ap_scores)


# ==============================================================================
# Answer Quality Metrics
# ==============================================================================

def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact Match (EM): Binary score - 1 if exact match after normalization, 0 otherwise

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        EM score (0.0 or 1.0)

    Example:
        prediction = "The capital is Paris"
        ground_truth = "Paris"
        EM = 0.0 (not exact match)

        prediction = "paris"
        ground_truth = "Paris"
        EM = 1.0 (match after normalization)
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 Score (following SQuAD metric)

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score (0.0 to 1.0)

    Example:
        prediction = "The capital of France is Paris"
        ground_truth = "Paris"
        Tokens pred: ['capital', 'of', 'france', 'is', 'paris']
        Tokens truth: ['paris']
        Common: ['paris']
        Precision: 1/5 = 0.2
        Recall: 1/1 = 1.0
        F1: 2 * 0.2 * 1.0 / (0.2 + 1.0) = 0.333
    """
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    # Count token occurrences
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    # Calculate overlap
    common_tokens = pred_counter & truth_counter
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)

    return 2 * (precision * recall) / (precision + recall)


def rouge_l(prediction: str, ground_truth: str) -> float:
    """
    ROUGE-L: Longest Common Subsequence based F1

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        ROUGE-L score (0.0 to 1.0)
    """
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    # Calculate LCS length
    lcs_length = _lcs_length(pred_tokens, truth_tokens)

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(truth_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Calculate length of Longest Common Subsequence using dynamic programming
    """
    m, n = len(seq1), len(seq2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


# ==============================================================================
# Batch Metric Calculation
# ==============================================================================

def calculate_retrieval_metrics(
    retrieved: List[str],
    relevant: List[str],
    k: int = 5,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate multiple retrieval metrics at once

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant/ground truth document IDs
        k: Cutoff rank for @K metrics
        metrics: List of metric names to calculate (if None, calculates all)

    Returns:
        Dictionary of metric_name -> score

    Example:
        calculate_retrieval_metrics(
            retrieved=['doc1', 'doc2', 'doc3'],
            relevant=['doc1', 'doc3'],
            k=5,
            metrics=['precision', 'recall', 'f1', 'mrr']
        )
        # Returns: {
        #     'precision@5': 0.667,
        #     'recall@5': 1.0,
        #     'f1@5': 0.8,
        #     'mrr': 1.0
        # }
    """
    available_metrics = {
        'precision': lambda: precision_at_k(retrieved, relevant, k),
        'recall': lambda: recall_at_k(retrieved, relevant, k),
        'f1': lambda: f1_at_k(retrieved, relevant, k),
        'mrr': lambda: mean_reciprocal_rank(retrieved, relevant),
        'hit': lambda: hit_at_k(retrieved, relevant, k),
        'ndcg': lambda: ndcg_at_k(retrieved, relevant, k),
        'map': lambda: average_precision(retrieved, relevant)
    }

    # If no metrics specified, calculate all
    if metrics is None:
        metrics = list(available_metrics.keys())

    results = {}
    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower in available_metrics:
            score = available_metrics[metric_lower]()
            # Format metric name with @K for rank-based metrics
            if metric_lower in ['precision', 'recall', 'f1', 'hit', 'ndcg']:
                metric_name = f"{metric_lower}@{k}"
            else:
                metric_name = metric_lower
            results[metric_name] = round(score, 4)

    return results


def calculate_answer_metrics(
    prediction: str,
    ground_truth: str,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate multiple answer quality metrics at once

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        metrics: List of metric names to calculate (if None, calculates all)

    Returns:
        Dictionary of metric_name -> score

    Example:
        calculate_answer_metrics(
            prediction="The capital of France is Paris",
            ground_truth="Paris",
            metrics=['em', 'f1', 'rouge_l']
        )
        # Returns: {
        #     'exact_match': 0.0,
        #     'f1': 0.333,
        #     'rouge_l': 0.5
        # }
    """
    available_metrics = {
        'em': lambda: exact_match(prediction, ground_truth),
        'exact_match': lambda: exact_match(prediction, ground_truth),
        'f1': lambda: token_f1(prediction, ground_truth),
        'token_f1': lambda: token_f1(prediction, ground_truth),
        'rouge_l': lambda: rouge_l(prediction, ground_truth)
    }

    # If no metrics specified, calculate all unique metrics
    if metrics is None:
        metrics = ['em', 'f1', 'rouge_l']

    results = {}
    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower in available_metrics:
            score = available_metrics[metric_lower]()
            # Normalize metric name
            if metric_lower in ['em', 'exact_match']:
                metric_name = 'exact_match'
            elif metric_lower in ['f1', 'token_f1']:
                metric_name = 'f1'
            else:
                metric_name = metric_lower

            if metric_name not in results:  # Avoid duplicates
                results[metric_name] = round(score, 4)

    return results


def aggregate_metrics(metric_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries/examples

    Args:
        metric_results: List of metric dictionaries

    Returns:
        Dictionary of metric_name -> average_score

    Example:
        metric_results = [
            {'precision@5': 0.8, 'recall@5': 1.0},
            {'precision@5': 0.6, 'recall@5': 0.8}
        ]
        aggregate_metrics(metric_results)
        # Returns: {'precision@5': 0.7, 'recall@5': 0.9}
    """
    if not metric_results:
        return {}

    # Get all metric names
    all_metrics = set()
    for result in metric_results:
        all_metrics.update(result.keys())

    # Calculate averages
    aggregated = {}
    for metric in all_metrics:
        values = [r[metric] for r in metric_results if metric in r]
        if values:
            aggregated[metric] = round(np.mean(values), 4)

    return aggregated
