"""
LLM-as-Judge Evaluator for SingleTopic

Uses GPT-4o-mini to semantically evaluate answer correctness.
More accurate than F1/token matching for verbose, explanatory answers.

Usage:
    python 7_llm_evaluator.py                    # Evaluate 10 samples (default)
    python 7_llm_evaluator.py --num_samples 20   # Evaluate 20 samples
    python 7_llm_evaluator.py --num_samples -1   # Evaluate ALL 80 answerable questions

Cost Estimate:
    - 10 samples: ~$0.01-0.02
    - 20 samples: ~$0.02-0.05
    - 80 samples: ~$0.10-0.50

Author: BiG-RAG Team
Date: 2025-01-12
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import time
from openai import OpenAI

# ============================================================================
# Configuration
# ============================================================================

DATASET = "SingleTopic"
RESULTS_DIR = Path(f"datasets/{DATASET}/results")
GENERATION_FILE = RESULTS_DIR / "generation_results.csv"
LLM_EVAL_FILE = RESULTS_DIR / "llm_evaluation_report.json"
LLM_EVAL_MD_FILE = RESULTS_DIR / "llm_evaluation_report.md"

# OpenAI API Configuration
API_KEY_FILE = "openai_api_key.txt"

# LLM Evaluation Prompt Template
EVAL_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of AI-generated answers for a question-answering system.

**Question**: {question}

**Golden Answer** (reference truth):
{golden_answer}

**Generated Answer** (system output):
{generated_answer}

**Retrieved Context** (what the system retrieved from knowledge base):
{retrieval_context}

---

Please evaluate the generated answer on the following criteria:

1. **Factual Correctness** (0-5 points):
   - Does the answer contain the key facts from the golden answer?
   - Is the information factually accurate?
   - Score: 5 = all key facts correct, 3 = mostly correct, 1 = mostly wrong, 0 = completely wrong

2. **Completeness** (0-5 points):
   - Does the answer cover all important points from the golden answer?
   - Score: 5 = fully complete, 3 = missing minor details, 1 = missing major details, 0 = missing most information

3. **Relevance** (0-5 points):
   - Is the answer directly relevant to the question?
   - Does it avoid unnecessary tangents?
   - Score: 5 = perfectly relevant, 3 = mostly relevant, 1 = partially off-topic, 0 = off-topic

4. **Citation Quality** (0-5 points):
   - Are sources properly cited?
   - Do citations match the retrieved context?
   - Score: 5 = excellent citations, 3 = adequate, 1 = poor, 0 = no citations or wrong citations

5. **User Helpfulness** (0-5 points):
   - Would this answer help a user understand the topic?
   - Does additional context add value?
   - Score: 5 = very helpful, 3 = somewhat helpful, 1 = barely helpful, 0 = not helpful

**Output Format** (respond ONLY with valid JSON):
```json
{{
  "factual_correctness": <0-5>,
  "completeness": <0-5>,
  "relevance": <0-5>,
  "citation_quality": <0-5>,
  "user_helpfulness": <0-5>,
  "overall_score": <0-25>,
  "reasoning": "<1-2 sentence explanation>",
  "verdict": "<CORRECT|MOSTLY_CORRECT|PARTIALLY_CORRECT|INCORRECT>"
}}
```

Respond with JSON only, no additional text.
"""

# ============================================================================
# LLM Client Setup
# ============================================================================

def load_api_key():
    """Load OpenAI API key from file"""
    key_path = Path(API_KEY_FILE)

    if not key_path.exists():
        print(f"[FAIL] OpenAI API key file not found: {API_KEY_FILE}")
        print("Please create the file with your API key.")
        return None

    with open(key_path, 'r') as f:
        api_key = f.read().strip()

    return api_key

def create_openai_client():
    """Create OpenAI client"""
    api_key = load_api_key()

    if not api_key:
        return None

    return OpenAI(api_key=api_key)

# ============================================================================
# LLM Evaluation
# ============================================================================

def evaluate_single_answer(client, question, golden_answer, generated_answer, retrieval_context):
    """
    Use GPT-4o-mini to evaluate a single answer

    Returns:
        dict with scores and verdict, or None if evaluation fails
    """

    # Format prompt
    prompt = EVAL_PROMPT_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer[:1000],  # Limit length to save cost
        retrieval_context=retrieval_context[:1000] if retrieval_context else "No context retrieved"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        # Parse JSON response
        response_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        result = json.loads(response_text.strip())

        return result

    except json.JSONDecodeError as e:
        print(f"[WARN] Failed to parse JSON response: {e}")
        print(f"       Response: {response_text[:200]}")
        return None
    except Exception as e:
        print(f"[ERROR] LLM evaluation failed: {e}")
        return None

def run_llm_evaluation(num_samples=10):
    """
    Run LLM-based evaluation on answerable questions

    Args:
        num_samples: Number of samples to evaluate (-1 for all)

    Returns:
        dict with evaluation results
    """

    print("=" * 70)
    print("  LLM-as-Judge Evaluation (GPT-4o-mini)")
    print("=" * 70)
    print()

    # Load data
    print("[1/5] Loading generation results...")
    results_df = pd.read_csv(GENERATION_FILE)

    # Filter answerable questions (has golden answer)
    answerable_df = results_df[results_df['golden_answer'].notna()].copy()

    total_answerable = len(answerable_df)
    print(f"[OK] Found {total_answerable} answerable questions")

    # Determine sample size
    if num_samples == -1:
        num_samples = total_answerable
        print(f"[INFO] Evaluating ALL {num_samples} questions")
    else:
        num_samples = min(num_samples, total_answerable)
        print(f"[INFO] Evaluating {num_samples} samples")

    # Sample questions
    sample_df = answerable_df.head(num_samples).copy()

    # Estimate cost
    estimated_cost_low = num_samples * 0.001
    estimated_cost_high = num_samples * 0.006
    print()
    print(f"[COST] Estimated cost: ${estimated_cost_low:.3f} - ${estimated_cost_high:.3f}")
    print()

    # Initialize OpenAI client
    print("[2/5] Initializing OpenAI client...")
    client = create_openai_client()

    if not client:
        print("[FAIL] Cannot proceed without API key")
        return None

    print("[OK] Client ready")
    print()

    # Run evaluation
    print(f"[3/5] Evaluating {num_samples} answers with GPT-4o-mini...")
    print()

    evaluations = []
    success_count = 0

    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        print(f"[{idx}/{num_samples}] Evaluating question: {row['question'][:60]}...")

        eval_result = evaluate_single_answer(
            client,
            row['question'],
            row['golden_answer'],
            row['generated_answer'],
            row['retrieval_context']
        )

        if eval_result:
            eval_result['question'] = row['question']
            eval_result['question_type'] = row['question_type']
            evaluations.append(eval_result)
            success_count += 1

            print(f"       Verdict: {eval_result['verdict']} (Score: {eval_result['overall_score']}/25)")
        else:
            print(f"       [FAIL] Evaluation failed")

        # Rate limiting (avoid hitting API limits)
        if idx < num_samples:
            time.sleep(0.5)  # 0.5 second delay between requests

    print()
    print(f"[OK] Completed {success_count}/{num_samples} evaluations")

    # Compute aggregate metrics
    print()
    print("[4/5] Computing aggregate metrics...")

    if not evaluations:
        print("[FAIL] No successful evaluations")
        return None

    # Calculate averages
    avg_factual = sum(e['factual_correctness'] for e in evaluations) / len(evaluations)
    avg_completeness = sum(e['completeness'] for e in evaluations) / len(evaluations)
    avg_relevance = sum(e['relevance'] for e in evaluations) / len(evaluations)
    avg_citation = sum(e['citation_quality'] for e in evaluations) / len(evaluations)
    avg_helpfulness = sum(e['user_helpfulness'] for e in evaluations) / len(evaluations)
    avg_overall = sum(e['overall_score'] for e in evaluations) / len(evaluations)

    # Verdict distribution
    verdict_counts = {}
    for e in evaluations:
        verdict = e['verdict']
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    # Metrics by question type
    by_type = {}
    for e in evaluations:
        qtype = e['question_type']
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(e['overall_score'])

    type_averages = {
        qtype: sum(scores) / len(scores)
        for qtype, scores in by_type.items()
    }

    # Compile results
    results = {
        'dataset': DATASET,
        'num_evaluated': len(evaluations),
        'average_scores': {
            'factual_correctness': round(avg_factual, 2),
            'completeness': round(avg_completeness, 2),
            'relevance': round(avg_relevance, 2),
            'citation_quality': round(avg_citation, 2),
            'user_helpfulness': round(avg_helpfulness, 2),
            'overall': round(avg_overall, 2)
        },
        'verdict_distribution': verdict_counts,
        'accuracy_estimate': round(verdict_counts.get('CORRECT', 0) / len(evaluations) * 100, 1),
        'mostly_correct_rate': round((verdict_counts.get('CORRECT', 0) + verdict_counts.get('MOSTLY_CORRECT', 0)) / len(evaluations) * 100, 1),
        'by_question_type': type_averages,
        'per_question_evaluations': evaluations
    }

    # Save reports
    print()
    print("[5/5] Saving evaluation reports...")

    # JSON report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LLM_EVAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Markdown report
    md_report = generate_markdown_report(results)
    with open(LLM_EVAL_MD_FILE, 'w', encoding='utf-8') as f:
        f.write(md_report)

    print(f"[OK] Saved reports:")
    print(f"     - {LLM_EVAL_FILE}")
    print(f"     - {LLM_EVAL_MD_FILE}")

    return results

# ============================================================================
# Report Generation
# ============================================================================

def generate_markdown_report(results):
    """Generate human-readable markdown report"""

    lines = [
        "# LLM-as-Judge Evaluation Report",
        "",
        f"**Dataset**: {results['dataset']}",
        f"**Evaluator**: GPT-4o-mini",
        f"**Questions Evaluated**: {results['num_evaluated']}",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "### Average Scores (out of 5)",
        "",
        f"- **Factual Correctness**: {results['average_scores']['factual_correctness']:.2f}/5",
        f"- **Completeness**: {results['average_scores']['completeness']:.2f}/5",
        f"- **Relevance**: {results['average_scores']['relevance']:.2f}/5",
        f"- **Citation Quality**: {results['average_scores']['citation_quality']:.2f}/5",
        f"- **User Helpfulness**: {results['average_scores']['user_helpfulness']:.2f}/5",
        "",
        f"**Overall Score**: {results['average_scores']['overall']:.2f}/25 ({results['average_scores']['overall']/25*100:.1f}%)",
        "",
        "---",
        "",
        "## Accuracy Estimates",
        "",
        f"- **Fully Correct**: {results['accuracy_estimate']:.1f}%",
        f"- **Correct or Mostly Correct**: {results['mostly_correct_rate']:.1f}%",
        "",
        "### Verdict Distribution",
        "",
    ]

    for verdict, count in sorted(results['verdict_distribution'].items(), key=lambda x: -x[1]):
        pct = count / results['num_evaluated'] * 100
        lines.append(f"- **{verdict}**: {count} questions ({pct:.1f}%)")

    lines.extend([
        "",
        "---",
        "",
        "## Performance by Question Type",
        "",
    ])

    for qtype, avg_score in sorted(results['by_question_type'].items(), key=lambda x: -x[1]):
        lines.append(f"- **{qtype}**: {avg_score:.2f}/25 ({avg_score/25*100:.1f}%)")

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "### Score Ranges",
        "",
        "- **20-25 points (80-100%)**: Excellent answer, fully correct and helpful",
        "- **15-19 points (60-79%)**: Good answer, mostly correct with minor issues",
        "- **10-14 points (40-59%)**: Fair answer, partially correct but incomplete",
        "- **5-9 points (20-39%)**: Poor answer, mostly incorrect",
        "- **0-4 points (0-19%)**: Very poor answer, wrong or irrelevant",
        "",
        "---",
        "",
        "## Comparison with Token-Based Metrics",
        "",
        "**Why LLM-as-Judge is Better**:",
        "",
        "1. **Semantic Understanding**: Recognizes paraphrasing and equivalent facts",
        "2. **Contextual Evaluation**: Considers whether additional context adds value",
        "3. **Citation Assessment**: Evaluates proper source attribution",
        "4. **User-Centric**: Measures helpfulness, not just token overlap",
        "",
        "**Token-based F1** penalizes verbose but correct answers.",
        "**LLM-based evaluation** rewards factual correctness and user value.",
        "",
        "---",
        "",
        f"**Generated by**: `test_scripts/singletopic/7_llm_evaluator.py`",
    ])

    return '\n'.join(lines)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluator for SingleTopic")
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help="Number of samples to evaluate (-1 for all, default: 10)"
    )

    args = parser.parse_args()

    # Run evaluation
    results = run_llm_evaluation(num_samples=args.num_samples)

    if results:
        print()
        print("=" * 70)
        print("  Evaluation Complete!")
        print("=" * 70)
        print()
        print(f"Overall Score: {results['average_scores']['overall']:.2f}/25 ({results['average_scores']['overall']/25*100:.1f}%)")
        print(f"Fully Correct: {results['accuracy_estimate']:.1f}%")
        print(f"Mostly Correct: {results['mostly_correct_rate']:.1f}%")
        print()
        print(f"View detailed report: {LLM_EVAL_MD_FILE}")
    else:
        print()
        print("[FAIL] Evaluation failed")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
