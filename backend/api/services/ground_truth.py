"""
Ground Truth Validation and Management for BiG-RAG Evaluation

This module provides utilities for:
- Validating ground truth document IDs against corpus
- Loading and validating evaluation datasets
- Auto-generating ground truth from QA datasets
- Format conversion and data quality checks
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from bigrag.utils import logger


# ==============================================================================
# Ground Truth Validation
# ==============================================================================

async def validate_ground_truth_ids(
    dataset: str,
    doc_ids: List[str],
    working_dir: str = "expr"
) -> Tuple[List[str], List[str]]:
    """
    Validate that ground truth document IDs exist in corpus

    Args:
        dataset: Dataset name (e.g., "2WikiMultiHopQA")
        doc_ids: List of document IDs to validate
        working_dir: Working directory (default: "expr")

    Returns:
        Tuple of (valid_ids, invalid_ids)

    Example:
        valid, invalid = await validate_ground_truth_ids(
            "2WikiMultiHopQA",
            ["doc-abc123", "doc-xyz789"]
        )
        if invalid:
            print(f"Warning: {len(invalid)} IDs not found in corpus")
    """
    # Load corpus to get all valid document IDs
    corpus_file = Path(f"datasets/{dataset}/raw/corpus.jsonl")

    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_file}. "
            f"Cannot validate ground truth without corpus."
        )

    # Extract all document IDs from corpus
    valid_doc_ids = set()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get('id', '')
            if doc_id:
                valid_doc_ids.add(doc_id)

    # Validate each ground truth ID
    valid_ids = []
    invalid_ids = []

    for doc_id in doc_ids:
        if doc_id in valid_doc_ids:
            valid_ids.append(doc_id)
        else:
            invalid_ids.append(doc_id)

    if invalid_ids:
        logger.warning(
            f"[Ground Truth Validation] {len(invalid_ids)}/{len(doc_ids)} IDs not found in corpus. "
            f"First few invalid IDs: {invalid_ids[:5]}"
        )

    return valid_ids, invalid_ids


async def validate_eval_dataset(
    dataset_file: str,
    dataset_name: str,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load and validate evaluation dataset

    Args:
        dataset_file: Path to evaluation dataset JSON file
        dataset_name: Dataset name for validation (e.g., "2WikiMultiHopQA")
        strict: If True, raise error on invalid IDs. If False, just warn.

    Returns:
        Dictionary with validation results:
        {
            "valid": True/False,
            "total_questions": int,
            "valid_questions": int,
            "invalid_questions": int,
            "errors": [list of error messages],
            "warnings": [list of warnings]
        }

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset format is invalid
        RuntimeError: If strict=True and validation fails
    """
    dataset_path = Path(dataset_file)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_file}")

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support multiple formats
    if isinstance(data, list):
        questions = data
    elif 'questions' in data:
        questions = data['questions']
    elif 'test_cases' in data:
        questions = data['test_cases']
    else:
        raise ValueError(
            "Invalid dataset format. Expected 'questions' or 'test_cases' field, "
            f"or a list of questions. Got keys: {list(data.keys())}"
        )

    # Validation results
    errors = []
    warnings = []
    valid_questions = 0
    invalid_questions = 0

    # Collect all document IDs for batch validation
    all_doc_ids = set()
    for q in questions:
        if 'ground_truth_docs' in q:
            all_doc_ids.update(q['ground_truth_docs'])

    # Validate all doc IDs at once
    if all_doc_ids:
        valid_ids, invalid_ids = await validate_ground_truth_ids(
            dataset_name,
            list(all_doc_ids)
        )
        invalid_ids_set = set(invalid_ids)

        if invalid_ids:
            warnings.append(
                f"{len(invalid_ids)} document IDs not found in corpus: {invalid_ids[:10]}"
            )
    else:
        invalid_ids_set = set()

    # Validate each question
    for i, q in enumerate(questions):
        q_errors = []

        # Check required fields
        if 'question' not in q:
            q_errors.append(f"Question {i}: Missing 'question' field")

        # Check ground truth fields (at least one required)
        has_ground_truth_answer = 'ground_truth_answer' in q or 'ground_truth' in q
        has_ground_truth_docs = 'ground_truth_docs' in q

        if not has_ground_truth_answer and not has_ground_truth_docs:
            q_errors.append(
                f"Question {i}: Missing ground truth. "
                f"Need 'ground_truth_answer' or 'ground_truth_docs'"
            )

        # Check ground truth docs validity
        if has_ground_truth_docs:
            q_doc_ids = q['ground_truth_docs']
            if not isinstance(q_doc_ids, list):
                q_errors.append(
                    f"Question {i}: 'ground_truth_docs' must be a list, "
                    f"got {type(q_doc_ids)}"
                )
            elif any(doc_id in invalid_ids_set for doc_id in q_doc_ids):
                invalid_in_q = [d for d in q_doc_ids if d in invalid_ids_set]
                q_errors.append(
                    f"Question {i}: Invalid doc IDs: {invalid_in_q}"
                )

        if q_errors:
            errors.extend(q_errors)
            invalid_questions += 1
        else:
            valid_questions += 1

    # Determine if validation passed
    is_valid = len(errors) == 0

    if not is_valid and strict:
        raise RuntimeError(
            f"Evaluation dataset validation failed with {len(errors)} errors:\n"
            + "\n".join(errors[:10])
        )

    return {
        "valid": is_valid,
        "total_questions": len(questions),
        "valid_questions": valid_questions,
        "invalid_questions": invalid_questions,
        "errors": errors,
        "warnings": warnings
    }


# ==============================================================================
# Ground Truth Generation
# ==============================================================================

def create_ground_truth_from_qa(
    qa_file: str,
    corpus_file: str,
    output_file: str,
    match_strategy: str = "exact"
) -> Dict[str, Any]:
    """
    Auto-generate ground truth document IDs from QA dataset

    This function matches questions/answers to corpus documents to create
    ground truth document IDs for evaluation.

    Args:
        qa_file: Path to QA dataset (JSON with questions and answers)
        corpus_file: Path to corpus.jsonl
        output_file: Path to save generated ground truth
        match_strategy: How to match answers to docs:
            - "exact": Exact string match
            - "fuzzy": Fuzzy string matching (not implemented yet)
            - "embedding": Semantic similarity (not implemented yet)

    Returns:
        Dictionary with generation statistics

    Example:
        stats = create_ground_truth_from_qa(
            "datasets/2WikiMultiHopQA/raw/qa_test.json",
            "datasets/2WikiMultiHopQA/raw/corpus.jsonl",
            "evaluation_datasets/2wiki_ground_truth.json"
        )
        print(f"Matched {stats['matched_questions']}/{stats['total_questions']}")
    """
    # Load QA dataset
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # Load corpus
    corpus_docs = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus_docs.append(doc)

    # Match questions to documents
    matched_questions = 0
    unmatched_questions = 0
    ground_truth_dataset = []

    for qa in qa_data:
        question = qa.get('question', '')
        answer = qa.get('answer', qa.get('golden_answers', [''])[0])

        # Find documents containing the answer (simple exact match for now)
        matching_doc_ids = []

        for doc in corpus_docs:
            content = doc.get('contents', '').lower()
            if answer.lower() in content:
                matching_doc_ids.append(doc['id'])

        if matching_doc_ids:
            matched_questions += 1
            ground_truth_dataset.append({
                "question": question,
                "ground_truth_answer": answer,
                "ground_truth_docs": matching_doc_ids[:5]  # Limit to top 5
            })
        else:
            unmatched_questions += 1
            logger.warning(
                f"[Ground Truth Generation] Could not match answer to any document: {answer[:50]}..."
            )

    # Save ground truth dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "questions": ground_truth_dataset,
            "metadata": {
                "source_qa_file": qa_file,
                "source_corpus_file": corpus_file,
                "match_strategy": match_strategy,
                "total_questions": len(qa_data),
                "matched_questions": matched_questions,
                "unmatched_questions": unmatched_questions
            }
        }, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[Ground Truth Generation] Saved to {output_file}. "
        f"Matched: {matched_questions}/{len(qa_data)}"
    )

    return {
        "total_questions": len(qa_data),
        "matched_questions": matched_questions,
        "unmatched_questions": unmatched_questions,
        "output_file": str(output_path)
    }


# ==============================================================================
# Format Conversion
# ==============================================================================

def convert_hotpotqa_to_eval_format(
    hotpot_file: str,
    output_file: str,
    limit: Optional[int] = None
) -> None:
    """
    Convert HotpotQA dataset to BiG-RAG evaluation format

    Args:
        hotpot_file: Path to HotpotQA JSON file
        output_file: Path to save converted dataset
        limit: Maximum number of questions to convert

    HotpotQA Format:
        {
            "_id": "...",
            "question": "...",
            "answer": "...",
            "supporting_facts": [["doc_title", 0], ["doc_title", 2], ...]
        }

    BiG-RAG Eval Format:
        {
            "question": "...",
            "ground_truth_answer": "...",
            "ground_truth_docs": ["doc-hash1", "doc-hash2", ...]
        }
    """
    # Implementation note: This requires mapping doc titles to doc IDs
    # Left as TODO since it's dataset-specific
    raise NotImplementedError(
        "HotpotQA conversion requires title-to-ID mapping. "
        "Use create_ground_truth_from_qa() instead."
    )


def convert_squad_to_eval_format(
    squad_file: str,
    output_file: str,
    limit: Optional[int] = None
) -> None:
    """
    Convert SQuAD dataset to BiG-RAG evaluation format

    Similar to convert_hotpotqa_to_eval_format but for SQuAD
    """
    raise NotImplementedError("SQuAD conversion not yet implemented")


# ==============================================================================
# Dataset Statistics
# ==============================================================================

def get_dataset_stats(dataset_file: str) -> Dict[str, Any]:
    """
    Get statistics about evaluation dataset

    Args:
        dataset_file: Path to evaluation dataset JSON

    Returns:
        Dictionary with stats:
        - total_questions
        - avg_ground_truth_docs_per_question
        - questions_with_answer
        - questions_with_docs
        - avg_question_length
        - avg_answer_length
    """
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        questions = data
    elif 'questions' in data:
        questions = data['questions']
    else:
        questions = data.get('test_cases', [])

    # Calculate statistics
    total = len(questions)
    with_answer = sum(1 for q in questions if 'ground_truth_answer' in q or 'ground_truth' in q)
    with_docs = sum(1 for q in questions if 'ground_truth_docs' in q)

    doc_counts = [len(q.get('ground_truth_docs', [])) for q in questions]
    avg_docs = sum(doc_counts) / len(doc_counts) if doc_counts else 0

    question_lengths = [len(q.get('question', '').split()) for q in questions]
    avg_q_len = sum(question_lengths) / len(question_lengths) if question_lengths else 0

    answer_lengths = [
        len(str(q.get('ground_truth_answer', q.get('ground_truth', ''))).split())
        for q in questions
    ]
    avg_a_len = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0

    return {
        "total_questions": total,
        "questions_with_answer": with_answer,
        "questions_with_docs": with_docs,
        "avg_ground_truth_docs_per_question": round(avg_docs, 2),
        "avg_question_length_words": round(avg_q_len, 2),
        "avg_answer_length_words": round(avg_a_len, 2),
        "min_docs_per_question": min(doc_counts) if doc_counts else 0,
        "max_docs_per_question": max(doc_counts) if doc_counts else 0
    }
