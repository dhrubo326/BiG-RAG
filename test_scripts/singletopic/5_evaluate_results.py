"""
SingleTopic Evaluation Script

This script evaluates BiG-RAG's performance with FLEXIBLE matching to fairly assess accuracy:
1. Strict Exact Match (EM)
2. Lenient F1 Score (token overlap)
3. Partial Match (contains golden answer)
4. Fuzzy Match (edit distance < threshold)
5. Special handling for no-answer questions

Focus: Retrieval quality + answer synthesis quality, not just string matching.
"""

import pandas as pd
import json
import re
import string
from collections import Counter
from pathlib import Path
from datetime import datetime
import difflib

# ============================================================================
# Configuration
# ============================================================================

DATASET = "SingleTopic"
RESULTS_CSV = f"datasets/{DATASET}/results/generation_results.csv"
OUTPUT_DIR = Path(f"datasets/{DATASET}/results")

# Flexible matching thresholds
FUZZY_MATCH_THRESHOLD = 0.8  # 80% similarity for fuzzy match
PARTIAL_MATCH_MIN_WORDS = 2  # Minimum words to consider partial match

# ============================================================================
# Text Normalization
# ============================================================================

def normalize_answer(text):
    """
    Normalize text for evaluation (following SQuAD/HotpotQA style)

    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Remove extra whitespace
    """
    if pd.isna(text) or text == "":
        return ""

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text.strip()

def tokenize(text):
    """Simple whitespace tokenization after normalization"""
    normalized = normalize_answer(text)
    return normalized.split()

# ============================================================================
# Evaluation Metrics
# ============================================================================

def exact_match(prediction, ground_truth):
    """
    Strict Exact Match (EM): Binary score - 1 if exact match after normalization

    This is the strictest metric.
    """
    if not prediction or not ground_truth:
        return 0.0

    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def token_f1(prediction, ground_truth):
    """
    Token-level F1 Score (following SQuAD metric)

    Measures overlap between predicted and ground truth tokens.
    More lenient than EM - allows partial matches.
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

def partial_match(prediction, ground_truth):
    """
    Partial Match: Check if normalized golden answer is contained in prediction

    Example:
        Prediction: "The Tanker enemy wields an AK-47 rifle"
        Golden: "Tanker wields AK-47"
        Result: 1.0 (golden answer is contained in prediction)

    This is very lenient - good for cases where model gives more detail than needed.
    """
    if not prediction or not ground_truth:
        return 0.0

    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(ground_truth)

    # Check if golden answer words are all in prediction
    truth_words = truth_norm.split()

    if len(truth_words) < PARTIAL_MATCH_MIN_WORDS:
        # Too short to be meaningful
        return 0.0

    # Check if all golden answer words appear in prediction
    pred_words_set = set(pred_norm.split())
    truth_words_set = set(truth_words)

    if truth_words_set.issubset(pred_words_set):
        return 1.0

    return 0.0

def fuzzy_match(prediction, ground_truth):
    """
    Fuzzy Match: Use sequence similarity to allow small differences

    Uses difflib.SequenceMatcher to compute similarity ratio.
    Threshold: 0.8 (80% similar)

    This handles typos, minor wording differences, etc.
    """
    if not prediction or not ground_truth:
        return 0.0

    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(ground_truth)

    similarity = difflib.SequenceMatcher(None, pred_norm, truth_norm).ratio()

    return 1.0 if similarity >= FUZZY_MATCH_THRESHOLD else 0.0

def is_no_answer_response(text):
    """
    Check if response correctly indicates "no answer" or "unanswerable"

    For no-answer questions, we want the model to refuse or say it doesn't know.
    """
    if not text:
        return False

    text_lower = normalize_answer(text)

    no_answer_phrases = [
        'no answer', 'unanswerable', 'cannot be answered', 'cannot answer',
        'insufficient information', 'not enough information',
        'does not provide', 'not mentioned', 'unknown', 'unclear',
        'no information', 'dont know', 'do not know', 'not specified',
        'not stated', 'not given', 'not available', 'no data'
    ]

    return any(phrase in text_lower for phrase in no_answer_phrases)

def has_retrieval_context(retrieval_context):
    """Check if retrieval successfully returned context"""
    if pd.isna(retrieval_context) or not retrieval_context:
        return False

    # Check if context is not empty string
    return len(str(retrieval_context).strip()) > 10

# ============================================================================
# Evaluation Runner
# ============================================================================

def evaluate_results(results_df):
    """
    Run comprehensive evaluation on generation results

    Returns:
        dict with overall_metrics, metrics_by_type, per_question_results
    """
    print()
    print("[1/3] Calculating metrics for answerable questions...")

    # Filter successful questions (no errors)
    # Note: Empty error cells are read as NaN by pandas, not empty string
    successful_df = results_df[results_df['error'].isna()].copy()

    # Separate answerable vs no-answer questions
    # For no-answer questions, golden_answer is NaN
    answerable_df = successful_df[successful_df['golden_answer'].notna()].copy()
    no_answer_df = successful_df[successful_df['question_type'] == 'no_answer'].copy()

    # Evaluate answerable questions
    em_scores = []
    f1_scores = []
    partial_scores = []
    fuzzy_scores = []
    has_context_list = []

    for idx, row in answerable_df.iterrows():
        prediction = row['generated_answer']
        golden = row['golden_answer']

        em_scores.append(exact_match(prediction, golden))
        f1_scores.append(token_f1(prediction, golden))
        partial_scores.append(partial_match(prediction, golden))
        fuzzy_scores.append(fuzzy_match(prediction, golden))
        has_context_list.append(has_retrieval_context(row['retrieval_context']))

    # Evaluate no-answer questions
    print()
    print("[2/3] Evaluating no-answer questions...")

    refusal_count = 0
    hallucination_count = 0

    for idx, row in no_answer_df.iterrows():
        prediction = row['generated_answer']

        if is_no_answer_response(prediction):
            refusal_count += 1
        elif prediction and len(str(prediction).strip()) > 10:
            # Model gave a substantive answer when it shouldn't
            hallucination_count += 1

    # Calculate overall metrics
    print()
    print("[3/3] Computing aggregate metrics...")

    overall_metrics = {
        'total_questions': len(results_df),
        'successful_questions': len(successful_df),
        'failed_questions': len(results_df) - len(successful_df),
        'success_rate': len(successful_df) / len(results_df) if len(results_df) > 0 else 0.0,

        # Answerable questions metrics
        'answerable_count': len(answerable_df),
        'exact_match': sum(em_scores) / len(em_scores) if em_scores else 0.0,
        'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'partial_match': sum(partial_scores) / len(partial_scores) if partial_scores else 0.0,
        'fuzzy_match': sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0,

        # Retrieval quality
        'retrieval_success_rate': sum(has_context_list) / len(has_context_list) if has_context_list else 0.0,

        # No-answer questions metrics
        'no_answer_count': len(no_answer_df),
        'no_answer_refusal_rate': (refusal_count / len(no_answer_df)) if len(no_answer_df) > 0 else 0.0,
        'no_answer_hallucination_rate': (hallucination_count / len(no_answer_df)) if len(no_answer_df) > 0 else 0.0,

        # Latency
        'avg_latency_ms': results_df[results_df['latency_ms'] > 0]['latency_ms'].mean() if len(results_df) > 0 else 0.0,
    }

    # Metrics by question type
    metrics_by_type = {}

    for qtype in ['single_passage', 'multi_passage', 'no_answer']:
        subset = successful_df[successful_df['question_type'] == qtype].copy()

        if qtype == 'no_answer':
            refusals = sum(1 for idx, row in subset.iterrows() if is_no_answer_response(row['generated_answer']))
            hallucinations = len(subset) - refusals

            metrics_by_type[qtype] = {
                'count': len(subset),
                'refusal_rate': (refusals / len(subset)) if len(subset) > 0 else 0.0,
                'hallucination_rate': (hallucinations / len(subset)) if len(subset) > 0 else 0.0
            }
        else:
            em_subset = []
            f1_subset = []
            partial_subset = []
            fuzzy_subset = []

            for idx, row in subset.iterrows():
                if row['generated_answer'] and row['golden_answer']:
                    em_subset.append(exact_match(row['generated_answer'], row['golden_answer']))
                    f1_subset.append(token_f1(row['generated_answer'], row['golden_answer']))
                    partial_subset.append(partial_match(row['generated_answer'], row['golden_answer']))
                    fuzzy_subset.append(fuzzy_match(row['generated_answer'], row['golden_answer']))

            metrics_by_type[qtype] = {
                'count': len(subset),
                'exact_match': sum(em_subset) / len(em_subset) if em_subset else 0.0,
                'f1_score': sum(f1_subset) / len(f1_subset) if f1_subset else 0.0,
                'partial_match': sum(partial_subset) / len(partial_subset) if partial_subset else 0.0,
                'fuzzy_match': sum(fuzzy_subset) / len(fuzzy_subset) if fuzzy_subset else 0.0
            }

    # Per-question results (for detailed analysis)
    per_question_results = []

    for idx, row in answerable_df.iterrows():
        per_question_results.append({
            'question': row['question'],
            'golden_answer': row['golden_answer'],
            'generated_answer': row['generated_answer'],
            'question_type': row['question_type'],
            'exact_match': exact_match(row['generated_answer'], row['golden_answer']),
            'f1_score': token_f1(row['generated_answer'], row['golden_answer']),
            'partial_match': partial_match(row['generated_answer'], row['golden_answer']),
            'fuzzy_match': fuzzy_match(row['generated_answer'], row['golden_answer']),
            'has_context': has_retrieval_context(row['retrieval_context'])
        })

    return {
        'overall_metrics': overall_metrics,
        'metrics_by_type': metrics_by_type,
        'per_question_results': per_question_results
    }

# ============================================================================
# Report Generators
# ============================================================================

def save_json_report(evaluation_results, output_path):
    """Save evaluation results as JSON"""
    report = {
        'dataset': DATASET,
        'timestamp': datetime.now().isoformat(),
        'evaluation_results': evaluation_results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def save_csv_report(evaluation_results, output_path):
    """Save metrics by type as CSV"""
    rows = []

    for qtype, metrics in evaluation_results['metrics_by_type'].items():
        row = {'question_type': qtype}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

def save_markdown_report(evaluation_results, output_path):
    """Save human-readable Markdown report"""
    overall = evaluation_results['overall_metrics']
    by_type = evaluation_results['metrics_by_type']

    lines = [
        "# SingleTopic Evaluation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Dataset**: {DATASET}",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        f"- **Total Questions**: {overall['total_questions']}",
        f"- **Successful**: {overall['successful_questions']} ({overall['success_rate']*100:.1f}%)",
        f"- **Failed**: {overall['failed_questions']}",
        f"- **Average Latency**: {overall['avg_latency_ms']:.2f} ms",
        "",
        "### Answer Quality (Answerable Questions)",
        "",
        f"- **Exact Match (EM)**: {overall['exact_match']:.4f} ({overall['exact_match']*100:.2f}%)",
        f"- **F1 Score**: {overall['f1_score']:.4f} ({overall['f1_score']*100:.2f}%)",
        f"- **Partial Match**: {overall['partial_match']:.4f} ({overall['partial_match']*100:.2f}%)",
        f"- **Fuzzy Match (80%)**: {overall['fuzzy_match']:.4f} ({overall['fuzzy_match']*100:.2f}%)",
        "",
        "> **Note**: Partial and Fuzzy matches are more lenient metrics that reward correct answers with different wording.",
        "",
        "### Retrieval Quality",
        "",
        f"- **Retrieval Success Rate**: {overall['retrieval_success_rate']:.4f} ({overall['retrieval_success_rate']*100:.2f}%)",
        "",
        "### No-Answer Handling",
        "",
        f"- **Total No-Answer Questions**: {overall['no_answer_count']}",
        f"- **Refusal Rate** (correctly said 'no answer'): {overall['no_answer_refusal_rate']:.4f} ({overall['no_answer_refusal_rate']*100:.2f}%)",
        f"- **Hallucination Rate** (gave answer when shouldn't): {overall['no_answer_hallucination_rate']:.4f} ({overall['no_answer_hallucination_rate']*100:.2f}%)",
        "",
        "---",
        "",
        "## Metrics by Question Type",
        "",
        "| Question Type | Count | EM | F1 | Partial | Fuzzy | Refusal Rate | Hallucination Rate |",
        "|---------------|-------|----|----|---------|-------|--------------|---------------------|"
    ]

    for qtype in ['single_passage', 'multi_passage', 'no_answer']:
        if qtype in by_type:
            metrics = by_type[qtype]
            if qtype == 'no_answer':
                lines.append(
                    f"| {qtype} | {metrics['count']} | - | - | - | - | "
                    f"{metrics['refusal_rate']*100:.1f}% | {metrics['hallucination_rate']*100:.1f}% |"
                )
            else:
                lines.append(
                    f"| {qtype} | {metrics['count']} | "
                    f"{metrics['exact_match']:.4f} | {metrics['f1_score']:.4f} | "
                    f"{metrics['partial_match']:.4f} | {metrics['fuzzy_match']:.4f} | - | - |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "### Answer Quality Metrics",
        "",
        "1. **Exact Match (EM)**: Strictest metric - 1.0 if normalized answer exactly matches golden answer",
        "2. **F1 Score**: Token-level F1 - measures overlap between predicted and golden tokens",
        "3. **Partial Match**: 1.0 if all golden answer words appear in prediction (any order)",
        "4. **Fuzzy Match**: 1.0 if prediction is ≥80% similar to golden answer (edit distance)",
        "",
        "### Interpretation Guide",
        "",
        "- **EM** is very strict - use for benchmarking against papers",
        "- **F1** is standard QA metric - balances precision and recall",
        "- **Partial Match** is lenient - good for assessing if core facts are present",
        "- **Fuzzy Match** handles typos and minor wording differences",
        "",
        "**Recommendation**: Report F1 as primary metric, use Partial/Fuzzy to understand system behavior.",
        "",
        "---",
        "",
        "Generated by `test_scripts/singletopic/5_evaluate_results.py`"
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  SingleTopic Evaluation")
    print("=" * 70)

    # Check if results file exists
    if not Path(RESULTS_CSV).exists():
        print(f"\n[FAIL] Results file not found: {RESULTS_CSV}")
        print(f"\nPlease run answer generation first:")
        print(f"  python test_scripts/singletopic/4_generate_answers.py")
        print()
        return 1

    # Load results
    print(f"\n[1/4] Loading results from {RESULTS_CSV}...")
    results_df = pd.read_csv(RESULTS_CSV)
    print(f"[OK] Loaded {len(results_df)} question results")

    # Run evaluation
    print(f"\n[2/4] Running evaluation with flexible matching...")
    evaluation_results = evaluate_results(results_df)

    # Save reports
    print(f"\n[3/4] Saving evaluation reports...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "evaluation_report.json"
    csv_path = OUTPUT_DIR / "evaluation_report.csv"
    md_path = OUTPUT_DIR / "evaluation_report.md"

    save_json_report(evaluation_results, json_path)
    save_csv_report(evaluation_results, csv_path)
    save_markdown_report(evaluation_results, md_path)

    print(f"[OK] Saved reports:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")

    # Print summary
    print(f"\n[4/4] Evaluation Summary")
    print("-" * 70)

    overall = evaluation_results['overall_metrics']

    print(f"\nOverall Performance:")
    print(f"  Total Questions:     {overall['total_questions']}")
    print(f"  Success Rate:        {overall['success_rate']*100:.1f}%")
    print(f"\nAnswer Quality (Answerable Questions):")
    print(f"  Exact Match (EM):    {overall['exact_match']:.4f} ({overall['exact_match']*100:.2f}%)")
    print(f"  F1 Score:            {overall['f1_score']:.4f} ({overall['f1_score']*100:.2f}%)")
    print(f"  Partial Match:       {overall['partial_match']:.4f} ({overall['partial_match']*100:.2f}%)")
    print(f"  Fuzzy Match:         {overall['fuzzy_match']:.4f} ({overall['fuzzy_match']*100:.2f}%)")
    print(f"\nRetrieval Quality:")
    print(f"  Success Rate:        {overall['retrieval_success_rate']*100:.1f}%")
    print(f"\nNo-Answer Handling:")
    print(f"  Refusal Rate:        {overall['no_answer_refusal_rate']*100:.1f}%")
    print(f"  Hallucination Rate:  {overall['no_answer_hallucination_rate']*100:.1f}%")

    print("\n" + "=" * 70)
    print("  [OK] Evaluation Complete!")
    print("=" * 70)
    print(f"\nView detailed report: {md_path}")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
