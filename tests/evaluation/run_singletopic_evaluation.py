"""
Complete evaluation pipeline for SingleTopic dataset

This script automates the entire workflow:
1. Validates dataset structure
2. Builds knowledge graph
3. Starts API server
4. Generates answers for all questions
5. Evaluates results
6. Exports metrics in multiple formats
"""

import os
import sys
import time
import subprocess
import json
import requests
from pathlib import Path


def run_command(cmd, description, check=True, timeout=None):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("[STDERR]", result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command timed out after {timeout} seconds")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with return code {e.returncode}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print("[STDERR]", e.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


def wait_for_api(port=8001, max_retries=30):
    """Wait for API server to be ready"""
    print(f"\nWaiting for API server on port {port}...")

    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"[OK] API server is ready!")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        time.sleep(2)
        print(f"  Retry {i+1}/{max_retries}...")

    print("[ERROR] API server did not start in time")
    return False


def check_api_health(port=8001):
    """Check API server health"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n[API Health Check]")
            print(f"  Status: {data.get('status')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Dataset: {data.get('rag_instances', {})}")
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return False


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("SINGLETOPIC EVALUATION PIPELINE")
    print("="*80)

    base_dir = Path(__file__).parent
    dataset = "SingleTopic"

    # Step 1: Validate dataset
    print("\n[Step 1] Validating dataset structure...")
    if not run_command(
        "python validate_singletopic_dataset.py",
        "Dataset validation",
        timeout=30
    ):
        print("\n[FAIL] Dataset validation failed!")
        print("Please fix dataset errors before proceeding.")
        return 1

    # Step 2: Build knowledge graph
    print("\n[Step 2] Building knowledge graph...")
    print("This will take 10-30 minutes depending on API rate limits...")
    print("Progress will be shown in build_graph.log")
    print()

    build_cmd = f"python script_build.py --data_source {dataset} --batch_size 5"
    if not run_command(build_cmd, "Knowledge graph construction", timeout=7200):
        print("\n[FAIL] Graph construction failed!")
        print("Check build_graph.log for details.")
        return 1

    # Step 3: Start API server (in background)
    print("\n[Step 3] Starting API server...")

    # Kill any existing server on port 8001
    if sys.platform == 'win32':
        os.system("FOR /F \"tokens=5\" %p IN ('netstat -aon ^| findstr :8001') DO taskkill /F /PID %p >nul 2>&1")
    else:
        os.system("fuser -k 8001/tcp 2>/dev/null")

    time.sleep(2)

    # Start server in background
    api_log = base_dir / "api_singletopic.log"
    if sys.platform == 'win32':
        # Windows: use START to run in background
        subprocess.Popen(
            f"python script_api.py --data_source {dataset} > {api_log} 2>&1",
            shell=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Linux/Mac: use nohup
        subprocess.Popen(
            f"nohup python script_api.py --data_source {dataset} > {api_log} 2>&1 &",
            shell=True
        )

    # Wait for server to be ready
    if not wait_for_api():
        print("\n[FAIL] API server did not start!")
        print(f"Check {api_log} for details.")
        return 1

    # Check health
    if not check_api_health():
        print("\n[WARN] API health check returned unexpected status")

    # Step 4: Generate answers
    print("\n[Step 4] Generating answers for all 120 questions...")
    print("This will take 5-15 minutes depending on API speed...")
    print()

    questions_csv = base_dir / "datasets" / dataset / "processed" / "all_questions_unified.csv"
    results_dir = base_dir / "datasets" / dataset / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "generation_results.csv"

    batch_generate_payload = {
        "questions_csv_path": str(questions_csv),
        "output_csv_path": str(results_csv),
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "top_k": 5,
        "enable_reranking": True
    }

    try:
        print(f"Calling /eval/batch_generate endpoint...")
        response = requests.post(
            "http://localhost:8001/eval/batch_generate",
            json=batch_generate_payload,
            timeout=1800  # 30 minutes
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n[OK] Generation completed!")
            print(f"  Total questions: {result.get('total_questions')}")
            print(f"  Successful: {result.get('successful')}")
            print(f"  Failed: {result.get('failed')}")
            print(f"  Average latency: {result.get('avg_latency_ms', 0):.2f} ms")
            print(f"  Results saved to: {results_csv}")
        else:
            print(f"\n[ERROR] Generation failed with status {response.status_code}")
            print(response.text)
            return 1
    except requests.Timeout:
        print("\n[ERROR] Generation request timed out")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        return 1

    # Step 5: Evaluate results
    print("\n[Step 5] Evaluating results...")

    evaluate_payload = {
        "results_csv_path": str(results_csv),
        "metrics": ["em", "f1", "rouge_l", "answer_rate"],
        "export_formats": ["json", "csv", "markdown", "latex"],
        "output_dir": str(results_dir)
    }

    try:
        print(f"Calling /eval/evaluate_results endpoint...")
        response = requests.post(
            "http://localhost:8001/eval/evaluate_results",
            json=evaluate_payload,
            timeout=300  # 5 minutes
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n[OK] Evaluation completed!")
            print(f"\n{'='*80}")
            print("EVALUATION RESULTS")
            print(f"{'='*80}")

            metrics = result.get('metrics', {}).get('overall', {})
            print(f"\n[Overall Metrics]")
            print(f"  EM (Exact Match):  {metrics.get('em', 0)*100:.2f}%")
            print(f"  F1 Score:          {metrics.get('f1', 0)*100:.2f}%")
            if 'rouge_l' in metrics:
                print(f"  ROUGE-L:           {metrics.get('rouge_l', 0)*100:.2f}%")
            if 'answer_rate' in metrics:
                print(f"  Answer Rate:       {metrics.get('answer_rate', 0)*100:.2f}%")

            print(f"\n[By Question Type]")
            by_type = result.get('metrics', {}).get('by_type', {})
            for qtype, qmetrics in by_type.items():
                print(f"  {qtype}:")
                print(f"    EM: {qmetrics.get('em', 0)*100:.2f}%, F1: {qmetrics.get('f1', 0)*100:.2f}%")

            print(f"\n[Exported Files]")
            for fmt, path in result.get('exported_files', {}).items():
                print(f"  - {fmt}: {path}")

            return 0
        else:
            print(f"\n[ERROR] Evaluation failed with status {response.status_code}")
            print(response.text)
            return 1
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
