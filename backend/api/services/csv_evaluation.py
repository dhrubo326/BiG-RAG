"""
CSV-based evaluation endpoints for BiG-RAG.

This module provides endpoints for batch processing questions and evaluating results:
    - POST /eval/batch_generate: Process questions CSV, generate answers, save to CSV
    - POST /eval/evaluate_results: Evaluate saved results CSV for accuracy

These endpoints enable systematic evaluation of BiG-RAG on standard datasets
by separating generation and evaluation steps to avoid redundant processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
import logging

from .answer_generation import (
    batch_generate_answers,
    is_no_answer_response
)
from .metrics import exact_match, token_f1
from .export import export_all_formats

logger = logging.getLogger(__name__)


async def batch_generate_endpoint(
    questions_csv_path: str,
    output_csv_path: str,
    rag_instance,
    llm_manager,
    embedding_manager,
    llm_provider: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
    top_k: int = 5,
    enable_reranking: bool = True
) -> Dict[str, Any]:
    """
    Batch process questions CSV and generate answers with retrieval context.

    This endpoint:
    1. Loads questions from CSV
    2. For each question:
       - Retrieves relevant context from knowledge graph
       - Generates answer using LLM
       - Tracks latency
    3. Saves results to CSV with all fields for later evaluation

    Args:
        questions_csv_path: Path to input CSV with questions
        output_csv_path: Path to save results CSV
        rag_instance: BiGRAG instance
        llm_manager: LLM manager
        embedding_manager: Embedding manager
        llm_provider: LLM provider (OpenAI, etc.)
        model: Model name
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Max answer length
        top_k: Number of context items to retrieve
        enable_reranking: Use semantic reranking

    Returns:
        Response dict with:
            - success: bool
            - total_questions: int
            - results_saved_to: str
            - processing_time_seconds: float
            - average_latency_ms: float
            - questions_by_type: dict

    Raises:
        FileNotFoundError: If input CSV doesn't exist
        RuntimeError: If processing fails
    """
    start_time = time.time()

    # Validate input file
    questions_path = Path(questions_csv_path)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions CSV not found: {questions_csv_path}")

    # Load questions
    logger.info(f"Loading questions from: {questions_csv_path}")
    try:
        df = pd.read_csv(questions_csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read questions CSV: {e}")

    # Validate required columns
    required_cols = ['question', 'golden_answer', 'document_index', 'question_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV: {missing_cols}. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Fill NaN values with empty strings to avoid type errors
    df = df.fillna('')

    total_questions = len(df)
    logger.info(f"Processing {total_questions} questions...")

    # Convert dataframe to list of dicts
    questions_list = df.to_dict('records')

    # Progress callback for logging
    def progress_callback(current, total):
        if current % 10 == 0 or current == total:
            logger.info(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

    # Batch process questions
    results = await batch_generate_answers(
        questions=questions_list,
        rag_instance=rag_instance,
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
        enable_reranking=enable_reranking,
        progress_callback=progress_callback
    )

    # Convert results to dataframe
    results_df = pd.DataFrame(results)

    # Create output directory if needed
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    logger.info(f"Results saved to: {output_csv_path}")

    # Calculate statistics
    processing_time = time.time() - start_time
    latencies = [r['latency_ms'] for r in results if r.get('latency_ms')]
    average_latency = np.mean(latencies) if latencies else 0.0

    # Count questions by type
    questions_by_type = df['question_type'].value_counts().to_dict()

    return {
        "success": True,
        "total_questions": total_questions,
        "results_saved_to": str(output_csv_path),
        "processing_time_seconds": round(processing_time, 2),
        "average_latency_ms": round(average_latency, 2),
        "questions_by_type": questions_by_type,
        "errors": sum(1 for r in results if r.get('error') is not None)
    }


async def evaluate_results_endpoint(
    results_csv_path: str,
    metrics: List[str] = None,
    export_formats: List[str] = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Evaluate saved results CSV for accuracy.

    This endpoint:
    1. Loads results CSV (output from batch_generate)
    2. Calculates metrics per question:
       - EM (Exact Match)
       - F1 (Token F1)
       - ROUGE-L (optional)
       - Answer Rate (for no-answer questions)
    3. Aggregates metrics overall and by question type
    4. Exports results in multiple formats (JSON, CSV, Markdown, LaTeX)

    Args:
        results_csv_path: Path to results CSV from batch_generate
        metrics: List of metrics to calculate (default: ["em", "f1", "answer_rate"])
        export_formats: Export formats (default: ["json", "csv", "markdown", "latex"])
        output_dir: Directory for exported files (default: same as results CSV)

    Returns:
        Response dict with:
            - success: bool
            - total_questions: int
            - overall_metrics: dict
            - metrics_by_type: dict
            - per_question_results: list
            - exported_files: list

    Raises:
        FileNotFoundError: If results CSV doesn't exist
        RuntimeError: If evaluation fails
    """
    # Default values
    if metrics is None:
        metrics = ["em", "f1", "answer_rate"]
    if export_formats is None:
        export_formats = ["json", "csv", "markdown", "latex"]

    # Validate input file
    results_path = Path(results_csv_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv_path}")

    # Set output directory
    if output_dir is None:
        output_dir = str(results_path.parent)

    # Load results
    logger.info(f"Loading results from: {results_csv_path}")
    try:
        df = pd.read_csv(results_csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read results CSV: {e}")

    # Validate required columns
    required_cols = ['question', 'golden_answer', 'generated_answer', 'question_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV: {missing_cols}. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Fill NaN values with empty strings to avoid type errors
    df = df.fillna('')

    total_questions = len(df)
    logger.info(f"Evaluating {total_questions} questions...")

    # Calculate metrics per question
    per_question_scores = []

    for idx, row in df.iterrows():
        question = row['question']
        golden = row['golden_answer']
        generated = row['generated_answer']
        q_type = row['question_type']

        # Calculate metrics
        scores = {
            'question': question,
            'question_type': q_type,
            'golden_answer': golden,
            'generated_answer': generated
        }

        # EM and F1 (always calculated)
        if 'em' in metrics:
            scores['em'] = exact_match(generated, golden)

        if 'f1' in metrics:
            scores['f1'] = token_f1(generated, golden)

        # ROUGE-L (optional - requires rouge_score package)
        if 'rouge_l' in metrics:
            try:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_scores = scorer.score(golden, generated)
                scores['rouge_l'] = rouge_scores['rougeL'].fmeasure
            except ImportError:
                logger.warning("rouge_score package not installed, skipping ROUGE-L")
                scores['rouge_l'] = None

        # Answer Rate (for no-answer questions)
        if 'answer_rate' in metrics and q_type == 'no_answer':
            scores['refused_to_answer'] = is_no_answer_response(generated)

        per_question_scores.append(scores)

    # Aggregate overall metrics
    overall_metrics = {}

    if 'em' in metrics:
        overall_metrics['em'] = np.mean([s['em'] for s in per_question_scores])

    if 'f1' in metrics:
        overall_metrics['f1'] = np.mean([s['f1'] for s in per_question_scores])

    if 'rouge_l' in metrics:
        rouge_vals = [s.get('rouge_l') for s in per_question_scores if s.get('rouge_l') is not None]
        overall_metrics['rouge_l'] = np.mean(rouge_vals) if rouge_vals else None

    # Aggregate by question type
    metrics_by_type = {}

    for q_type in df['question_type'].unique():
        type_scores = [s for s in per_question_scores if s['question_type'] == q_type]

        type_metrics = {'count': len(type_scores)}

        if 'em' in metrics:
            type_metrics['em'] = np.mean([s['em'] for s in type_scores])

        if 'f1' in metrics:
            type_metrics['f1'] = np.mean([s['f1'] for s in type_scores])

        if 'rouge_l' in metrics:
            rouge_vals = [s.get('rouge_l') for s in type_scores if s.get('rouge_l') is not None]
            type_metrics['rouge_l'] = np.mean(rouge_vals) if rouge_vals else None

        # Special handling for no-answer questions
        if q_type == 'no_answer' and 'answer_rate' in metrics:
            refused_count = sum(1 for s in type_scores if s.get('refused_to_answer', False))
            type_metrics['answer_refusal_rate'] = refused_count / len(type_scores) * 100
            type_metrics['hallucination_rate'] = 100 - type_metrics['answer_refusal_rate']

        metrics_by_type[q_type] = type_metrics

    # Prepare export data
    export_data = {
        'overall': overall_metrics,
        'by_type': metrics_by_type,
        'per_question': per_question_scores
    }

    # Export results
    exported_files = []
    if export_formats:
        try:
            exported = export_all_formats(
                results=export_data,
                output_dir=output_dir,
                base_name=results_path.stem + "_evaluation",
                caption="BiG-RAG Evaluation Results",
                formats=export_formats
            )
            exported_files = exported.get('files', [])
            logger.info(f"Exported {len(exported_files)} result files")
        except Exception as e:
            logger.warning(f"Export failed: {e}")

    return {
        "success": True,
        "total_questions": total_questions,
        "overall_metrics": {k: round(float(v), 4) if v is not None else None for k, v in overall_metrics.items()},
        "metrics_by_type": {
            k: {mk: round(float(mv), 4) if isinstance(mv, (int, float)) and mk != 'count' else mv
                for mk, mv in type_metrics.items()}
            for k, type_metrics in metrics_by_type.items()
        },
        "per_question_results": per_question_scores[:10],  # Return first 10 for preview
        "exported_files": exported_files
    }
