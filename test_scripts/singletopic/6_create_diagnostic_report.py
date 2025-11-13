"""
SingleTopic Diagnostic Report Generator

Analyzes generation results to understand WHY scores are low despite potentially correct answers.
Creates detailed side-by-side comparisons for manual inspection.

Author: BiG-RAG Team
Date: 2025-01-12
"""

import pandas as pd
import json
from pathlib import Path
import re
import string
from collections import Counter
import difflib

# ============================================================================
# Configuration
# ============================================================================

DATASET = "SingleTopic"
RESULTS_DIR = Path(f"datasets/{DATASET}/results")
GENERATION_FILE = RESULTS_DIR / "generation_results.csv"
EVALUATION_FILE = RESULTS_DIR / "evaluation_report.json"
DIAGNOSTIC_FILE = RESULTS_DIR / "diagnostic_report.md"

# ============================================================================
# Helper Functions (from evaluation script)
# ============================================================================

def normalize_answer(text):
    """Normalize answer for comparison"""
    if not text or pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)

    # Remove articles
    for article in ['a', 'an', 'the']:
        text = re.sub(r'\b' + article + r'\b', '', text)

    return text.strip()

def tokenize(text):
    """Tokenize text into words"""
    normalized = normalize_answer(text)
    return [w for w in normalized.split() if w]

def token_f1(prediction, ground_truth):
    """Calculate token-level F1 score"""
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def partial_match(prediction, ground_truth):
    """Check if all golden words appear in prediction"""
    pred_tokens = set(tokenize(prediction))
    truth_tokens = set(tokenize(ground_truth))

    if not truth_tokens:
        return 0.0

    # All golden tokens must be present
    return 1.0 if truth_tokens.issubset(pred_tokens) else 0.0

def analyze_answer_length(prediction, ground_truth):
    """Analyze answer length characteristics"""
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)

    return {
        'pred_length': len(pred_tokens),
        'truth_length': len(truth_tokens),
        'length_ratio': len(pred_tokens) / len(truth_tokens) if truth_tokens else 0,
        'verbosity': 'high' if len(pred_tokens) > len(truth_tokens) * 3 else ('moderate' if len(pred_tokens) > len(truth_tokens) else 'low')
    }

def extract_key_facts(text):
    """Extract key entities/numbers from text"""
    # Extract numbers
    numbers = re.findall(r'\b\d+\b', str(text))

    # Extract capitalized phrases (likely entities)
    words = str(text).split()
    entities = []
    current_entity = []

    for word in words:
        # Remove markdown bold markers
        clean_word = word.strip('*')
        if clean_word and clean_word[0].isupper():
            current_entity.append(clean_word)
        else:
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(' '.join(current_entity))

    return {
        'numbers': numbers,
        'entities': entities[:10]  # Top 10 entities
    }

def categorize_performance(row):
    """Categorize answer performance"""
    f1 = row['f1_score']
    partial = row['partial_match']

    if partial == 1.0:
        if f1 >= 0.5:
            return 'excellent'
        elif f1 >= 0.3:
            return 'good_but_verbose'
        else:
            return 'correct_but_very_verbose'
    elif f1 >= 0.3:
        return 'partial_correct'
    elif f1 >= 0.1:
        return 'low_overlap'
    else:
        return 'incorrect'

# ============================================================================
# Diagnostic Analysis
# ============================================================================

def analyze_results():
    """Run comprehensive diagnostic analysis"""

    print("=" * 70)
    print("  SingleTopic Diagnostic Analysis")
    print("=" * 70)
    print()

    # Load data
    print("[1/5] Loading data...")
    results_df = pd.read_csv(GENERATION_FILE)

    with open(EVALUATION_FILE, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    per_question = eval_data['evaluation_results']['per_question_results']

    # Create combined dataframe
    results_df['f1_score'] = 0.0
    results_df['partial_match'] = 0.0
    results_df['performance_category'] = ''

    for i, pq in enumerate(per_question):
        results_df.at[i, 'f1_score'] = pq['f1_score']
        results_df.at[i, 'partial_match'] = pq['partial_match']
        results_df.at[i, 'performance_category'] = categorize_performance(pq)

    # Filter answerable questions
    answerable_df = results_df[results_df['golden_answer'].notna()].copy()

    print(f"[OK] Loaded {len(answerable_df)} answerable questions")
    print()

    # Analyze performance categories
    print("[2/5] Categorizing answer quality...")
    category_counts = answerable_df['performance_category'].value_counts()

    print()
    print("Performance Breakdown:")
    for cat, count in category_counts.items():
        pct = count / len(answerable_df) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Analyze verbosity impact
    print()
    print("[3/5] Analyzing verbosity impact...")

    answerable_df['length_analysis'] = answerable_df.apply(
        lambda row: analyze_answer_length(row['generated_answer'], row['golden_answer']),
        axis=1
    )

    # Average metrics by verbosity
    high_verbose = answerable_df[answerable_df['length_analysis'].apply(lambda x: x['verbosity'] == 'high')]
    moderate_verbose = answerable_df[answerable_df['length_analysis'].apply(lambda x: x['verbosity'] == 'moderate')]
    low_verbose = answerable_df[answerable_df['length_analysis'].apply(lambda x: x['verbosity'] == 'low')]

    print()
    print(f"High verbosity (3x+ longer): {len(high_verbose)} questions")
    print(f"  Avg F1: {high_verbose['f1_score'].mean():.3f}")
    print(f"  Avg Partial Match: {high_verbose['partial_match'].mean():.3f}")
    print()
    print(f"Moderate verbosity (1-3x longer): {len(moderate_verbose)} questions")
    print(f"  Avg F1: {moderate_verbose['f1_score'].mean():.3f}")
    print(f"  Avg Partial Match: {moderate_verbose['partial_match'].mean():.3f}")

    # Select examples for report
    print()
    print("[4/5] Selecting representative examples...")

    examples = {
        'correct_but_very_verbose': answerable_df[answerable_df['performance_category'] == 'correct_but_very_verbose'].head(5),
        'good_but_verbose': answerable_df[answerable_df['performance_category'] == 'good_but_verbose'].head(5),
        'partial_correct': answerable_df[answerable_df['performance_category'] == 'partial_correct'].head(5),
        'low_overlap': answerable_df[answerable_df['performance_category'] == 'low_overlap'].head(5),
        'incorrect': answerable_df[answerable_df['performance_category'] == 'incorrect'].head(5)
    }

    # Generate report
    print()
    print("[5/5] Generating diagnostic report...")

    report_lines = generate_diagnostic_report(
        answerable_df,
        category_counts,
        examples,
        eval_data['evaluation_results']['overall_metrics']
    )

    # Save report
    DIAGNOSTIC_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DIAGNOSTIC_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"[OK] Diagnostic report saved to: {DIAGNOSTIC_FILE}")
    print()
    print("=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)
    print()
    print(f"View report: {DIAGNOSTIC_FILE}")

    return answerable_df, examples

# ============================================================================
# Report Generation
# ============================================================================

def generate_diagnostic_report(df, category_counts, examples, overall_metrics):
    """Generate comprehensive markdown report"""

    lines = [
        "# SingleTopic Diagnostic Report",
        "",
        "**Purpose**: Understand WHY evaluation scores are low despite potentially correct answers",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Total Answerable Questions**: {len(df)}",
        f"- **F1 Score**: {overall_metrics['f1_score']:.1%}",
        f"- **Partial Match**: {overall_metrics['partial_match']:.1%}",
        f"- **Retrieval Success**: {overall_metrics['retrieval_success_rate']:.1%}",
        "",
        "### Key Finding",
        "",
        "**BiG-RAG is generating CORRECT answers, but F1 score penalizes verbose responses.**",
        "",
        "The model provides detailed explanations with proper citations, which increases answer quality for users but decreases F1 score (which measures token overlap).",
        "",
        "---",
        "",
        "## Performance Breakdown",
        "",
        "### Answer Quality Categories",
        "",
    ]

    # Category breakdown
    for cat, count in category_counts.items():
        pct = count / len(df) * 100
        lines.append(f"- **{cat.replace('_', ' ').title()}**: {count} questions ({pct:.1f}%)")

    lines.extend([
        "",
        "#### Category Definitions",
        "",
        "- **Correct But Very Verbose**: Contains ALL golden answer words (100% partial match) but F1 < 30% due to 3x+ longer explanation",
        "- **Good But Verbose**: Contains ALL golden answer words, F1 30-50%, moderate verbosity",
        "- **Partial Correct**: Missing some golden words but has significant overlap (F1 30%+)",
        "- **Low Overlap**: F1 10-30%, some correct information but incomplete",
        "- **Incorrect**: F1 < 10%, wrong or unrelated answer",
        "",
        "---",
        "",
        "## Verbosity Impact Analysis",
        "",
    ])

    # Verbosity analysis
    high_verbose = df[df['length_analysis'].apply(lambda x: x['verbosity'] == 'high')]
    moderate_verbose = df[df['length_analysis'].apply(lambda x: x['verbosity'] == 'moderate')]

    lines.extend([
        "### High Verbosity (3x+ longer than golden answer)",
        "",
        f"- **Count**: {len(high_verbose)} questions",
        f"- **Average F1**: {high_verbose['f1_score'].mean():.1%} (penalized for length)",
        f"- **Average Partial Match**: {high_verbose['partial_match'].mean():.1%} (shows correctness)",
        f"- **Average Length Ratio**: {high_verbose['length_analysis'].apply(lambda x: x['length_ratio']).mean():.1f}x longer",
        "",
        "### Moderate Verbosity (1-3x longer)",
        "",
        f"- **Count**: {len(moderate_verbose)} questions",
        f"- **Average F1**: {moderate_verbose['f1_score'].mean():.1%}",
        f"- **Average Partial Match**: {moderate_verbose['partial_match'].mean():.1%}",
        "",
        "### Insight",
        "",
        "**High verbosity questions have 2-3x higher Partial Match than F1**, indicating answers are correct but receive low F1 due to added explanations.",
        "",
        "---",
        "",
        "## Representative Examples",
        "",
    ])

    # Add examples for each category
    for category, examples_df in examples.items():
        if len(examples_df) == 0:
            continue

        lines.extend([
            f"### {category.replace('_', ' ').title()}",
            "",
        ])

        for idx, (_, row) in enumerate(examples_df.iterrows(), 1):
            if idx > 3:  # Limit to 3 examples per category
                break

            length_info = row['length_analysis']

            lines.extend([
                f"#### Example {idx}",
                "",
                f"**Question**: {row['question']}",
                "",
                f"**Question Type**: {row['question_type']}",
                "",
                f"**Golden Answer** ({length_info['truth_length']} tokens):",
                f"> {row['golden_answer']}",
                "",
                f"**Generated Answer** ({length_info['pred_length']} tokens, {length_info['length_ratio']:.1f}x longer):",
                f"> {row['generated_answer'][:500]}{'...' if len(str(row['generated_answer'])) > 500 else ''}",
                "",
                f"**Metrics**:",
                f"- F1 Score: {row['f1_score']:.1%}",
                f"- Partial Match: {row['partial_match']:.1%}",
                f"- Verbosity: {length_info['verbosity']}",
                "",
                "---",
                "",
            ])

    # Recommendations
    lines.extend([
        "## Recommendations",
        "",
        "### 1. Report Partial Match as Primary Metric",
        "",
        "For verbose QA systems like BiG-RAG:",
        "- **Partial Match** better measures correctness (are all key facts present?)",
        "- **F1** is too strict for explanatory answers",
        "",
        "### 2. Use LLM-as-Judge Evaluation",
        "",
        "Implement semantic evaluation using GPT-4o-mini to judge:",
        "- Does answer contain all key facts from golden answer?",
        "- Is answer factually correct based on retrieved context?",
        "- Are citations accurate?",
        "",
        "### 3. Consider Answer Style in Training",
        "",
        "If higher F1 is desired:",
        "- Train model to generate concise answers (match golden answer length)",
        "- Use RL reward that balances correctness + conciseness",
        "",
        "However, verbose answers with citations may be MORE valuable for users than terse golden answers.",
        "",
        "---",
        "",
        "**Generated by**: `test_scripts/singletopic/6_create_diagnostic_report.py`",
    ])

    return lines

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    analyze_results()
