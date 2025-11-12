"""
SingleTopic Answer Generation Script

This script:
1. Loads questions from all_questions_unified.csv
2. Queries BiG-RAG for each question (with retrieval context)
3. Saves results to generation_results.csv with STRICT row alignment
4. Handles errors gracefully without breaking row order
"""

import pandas as pd
import requests
import json
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

API_URL = "http://localhost:8001"
DATASET = "SingleTopic"

# Input/output paths
QUESTIONS_CSV = f"datasets/{DATASET}/processed/all_questions_unified.csv"
OUTPUT_CSV = f"datasets/{DATASET}/results/generation_results.csv"

# API settings
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0  # Deterministic answers
MAX_TOKENS = 500
TOP_K = 5
ENABLE_RERANKING = True
TIMEOUT_SECONDS = 60

# ============================================================================
# Helper Functions
# ============================================================================

def check_server():
    """Check if BiG-RAG server is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data.get('dataset', 'unknown')
        return False, None
    except Exception as e:
        return False, str(e)

def query_bigrag(question: str):
    """
    Query BiG-RAG for answer + retrieval context

    Returns:
        dict with keys: generated_answer, retrieval_context, retrieval_metadata, latency_ms, error
    """
    result = {
        'generated_answer': '',
        'retrieval_context': '',
        'retrieval_metadata': '',
        'latency_ms': 0.0,
        'error': ''
    }

    try:
        # Step 1: Get answer from chat/completions
        start_time = time.time()

        chat_response = requests.post(
            f"{API_URL}/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "model": LLM_MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "use_rag": True,
                "enable_reranking": ENABLE_RERANKING
            },
            timeout=TIMEOUT_SECONDS
        )

        latency_ms = (time.time() - start_time) * 1000

        if chat_response.status_code != 200:
            result['error'] = f"Chat API error: HTTP {chat_response.status_code}"
            result['latency_ms'] = latency_ms
            return result

        chat_data = chat_response.json()
        generated_answer = chat_data['choices'][0]['message']['content']
        result['generated_answer'] = generated_answer

        # Step 2: Get retrieval context from /ask endpoint
        # (since /chat/completions doesn't return retrieval details)
        try:
            ask_response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "mode": "hybrid",
                    "top_k": TOP_K,
                    "enable_reranking": ENABLE_RERANKING
                },
                timeout=30
            )

            if ask_response.status_code == 200:
                ask_data = ask_response.json()
                contexts = ask_data.get('retrieved_contexts', [])

                # Format retrieval context (clean presentation)
                formatted_contexts = []
                metadata_list = []

                for i, ctx in enumerate(contexts[:TOP_K], 1):
                    context_text = ctx.get('context', '')
                    if context_text:
                        formatted_contexts.append(f"[Source {i}] {context_text}")
                        metadata_list.append({
                            'rank': ctx.get('rank', i),
                            'coherence_score': ctx.get('coherence_score', 0.0)
                        })

                result['retrieval_context'] = "\n---\n".join(formatted_contexts)
                result['retrieval_metadata'] = json.dumps(metadata_list)

        except Exception as e:
            # Non-fatal: answer is more important than retrieval context
            result['error'] = f"Retrieval context unavailable: {str(e)[:100]}"

        result['latency_ms'] = round(latency_ms, 2)

    except requests.Timeout:
        result['error'] = f"Timeout after {TIMEOUT_SECONDS}s"
    except requests.RequestException as e:
        result['error'] = f"Request error: {str(e)[:100]}"
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)[:100]}"

    return result

# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("=" * 70)
    print("  SingleTopic Answer Generation")
    print("=" * 70)
    print()

    # Check if server is running
    print("[1/5] Checking BiG-RAG server...")
    is_running, dataset_name = check_server()

    if not is_running:
        print(f"[FAIL] BiG-RAG server is not running at {API_URL}")
        print()
        print("Please start the server first:")
        print(f"  cd backend && python server.py --data_source {DATASET}")
        print()
        return 1

    print(f"[OK] Server is running (dataset: {dataset_name})")

    if dataset_name != DATASET:
        print(f"[WARNING] Server is using dataset '{dataset_name}' but we expect '{DATASET}'")
        print(f"          Results may be incorrect!")

    # Load questions
    print()
    print("[2/5] Loading questions...")

    if not Path(QUESTIONS_CSV).exists():
        print(f"[FAIL] Questions file not found: {QUESTIONS_CSV}")
        return 1

    questions_df = pd.read_csv(QUESTIONS_CSV)
    print(f"[OK] Loaded {len(questions_df)} questions")

    # Verify columns
    required_cols = ['question', 'golden_answer', 'document_index', 'question_type']
    missing_cols = [col for col in required_cols if col not in questions_df.columns]
    if missing_cols:
        print(f"[FAIL] Missing columns in CSV: {missing_cols}")
        return 1

    # Create output directory
    output_dir = Path(OUTPUT_CSV).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process questions
    print()
    print("[3/5] Generating answers...")
    print(f"       This will take approximately {len(questions_df) * 3 / 60:.1f} minutes")
    print()

    results = []
    success_count = 0
    error_count = 0

    # Use tqdm for progress bar
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Progress"):
        question = row['question']
        golden_answer = row['golden_answer']
        document_index = row['document_index']
        question_type = row['question_type']

        # Query BiG-RAG
        query_result = query_bigrag(question)

        # Build result row (MUST maintain exact order)
        result_row = {
            'question': question,
            'golden_answer': golden_answer,
            'generated_answer': query_result['generated_answer'],
            'retrieval_context': query_result['retrieval_context'],
            'retrieval_context_metadata': query_result['retrieval_metadata'],
            'document_index': document_index,
            'question_type': question_type,
            'latency_ms': query_result['latency_ms'],
            'error': query_result['error']
        }

        results.append(result_row)

        if not query_result['error']:
            success_count += 1
        else:
            error_count += 1

    # Save results
    print()
    print("[4/5] Saving results...")

    results_df = pd.DataFrame(results)

    # CRITICAL: Verify row count matches input
    if len(results_df) != len(questions_df):
        print(f"[FAIL] Row count mismatch!")
        print(f"       Input: {len(questions_df)} rows")
        print(f"       Output: {len(results_df)} rows")
        print(f"       Results NOT saved to prevent data corruption.")
        return 1

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Results saved to: {OUTPUT_CSV}")

    # Summary stats
    print()
    print("[5/5] Summary")
    print("-" * 70)
    print(f"  Total questions:    {len(results_df)}")
    print(f"  Successful:         {success_count} ({success_count/len(results_df)*100:.1f}%)")
    print(f"  Errors:             {error_count} ({error_count/len(results_df)*100:.1f}%)")

    if results_df['latency_ms'].sum() > 0:
        avg_latency = results_df[results_df['latency_ms'] > 0]['latency_ms'].mean()
        print(f"  Avg latency:        {avg_latency:.2f} ms")

    print()
    print("  By question type:")
    type_counts = results_df.groupby('question_type').size()
    for qtype, count in type_counts.items():
        success_for_type = len(results_df[(results_df['question_type'] == qtype) & (results_df['error'] == '')])
        print(f"    {qtype:20s} {count:3d} questions ({success_for_type:3d} successful)")

    print()

    if error_count > 0:
        print("  [WARNING] Some questions failed. Check 'error' column in results CSV.")
        print()
        # Show first few errors
        error_samples = results_df[results_df['error'] != ''].head(3)
        print("  Sample errors:")
        for idx, row in error_samples.iterrows():
            print(f"    - Q{idx+1}: {row['error']}")
        print()

    print("=" * 70)
    print("  [OK] Answer Generation Complete")
    print("=" * 70)
    print()
    print(f"Next step: Run evaluation")
    print(f"  python test_scripts/singletopic/5_evaluate_results.py")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
