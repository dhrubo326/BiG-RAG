"""
Simple command-line script to test BiG-RAG knowledge graph by asking questions

Usage:
    python test_ask_question.py "Your question here"
    python test_ask_question.py "What is AI?" --top_k 3 --mode hybrid
"""

import requests
import json
import sys
import argparse
from typing import Optional


def ask_question(
    question: str,
    top_k: int = 5,
    mode: str = "hybrid",
    host: str = "localhost",
    port: int = 8001
) -> Optional[dict]:
    """
    Ask a question to BiG-RAG API

    Args:
        question: The question to ask
        top_k: Number of results to retrieve
        mode: Retrieval mode (hybrid, local, global, naive)
        host: API server host
        port: API server port

    Returns:
        API response as dict, or None if error
    """
    url = f"http://{host}:{port}/ask"

    payload = {
        "question": question,
        "top_k": top_k,
        "mode": mode
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to server at {host}:{port}")
        print(f"   Make sure the API server is running:")
        print(f"   python script_api.py --data_source demo_test")
        return None
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed: {e}")
        return None


def print_results(response: dict):
    """Pretty print the API response"""
    print("\n" + "="*80)
    print("📝 QUESTION")
    print("="*80)
    print(f"{response['question']}\n")

    print("="*80)
    print(f"📚 RETRIEVED CONTEXTS ({response['num_results']} results, mode: {response['mode']})")
    print("="*80)

    if response['num_results'] == 0:
        print(f"⚠️  {response.get('message', 'No results found')}\n")
        return

    for ctx in response['retrieved_contexts']:
        rank = ctx['rank']
        context = ctx['context']
        score = ctx['coherence_score']

        print(f"\n[Result {rank}] (Coherence: {score:.4f})")
        print("-" * 80)

        # Truncate very long contexts
        if len(context) > 500:
            context = context[:500] + "... (truncated)"

        print(context)

    print("\n" + "="*80)
    print(f"✅ {response.get('message', 'Query completed successfully')}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question to BiG-RAG knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_ask_question.py "What is Artificial Intelligence?"
  python test_ask_question.py "What is machine learning?" --top_k 3
  python test_ask_question.py "Explain neural networks" --mode local --top_k 5

Retrieval Modes:
  hybrid  - Combines entity and relation retrieval (default, recommended)
  local   - Entity-based retrieval only
  global  - Relation-based retrieval only
  naive   - Direct text chunk retrieval
        """
    )

    parser.add_argument(
        'question',
        type=str,
        help='The question to ask (use quotes for multi-word questions)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of results to retrieve (default: 5)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='hybrid',
        choices=['hybrid', 'local', 'global', 'naive'],
        help='Retrieval mode (default: hybrid)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='API server host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='API server port (default: 8001)'
    )

    args = parser.parse_args()

    print("\n🔍 Testing BiG-RAG Knowledge Graph")
    print(f"📡 Server: {args.host}:{args.port}")
    print(f"⚙️  Mode: {args.mode} | Top-K: {args.top_k}")

    # Ask question
    response = ask_question(
        question=args.question,
        top_k=args.top_k,
        mode=args.mode,
        host=args.host,
        port=args.port
    )

    if response:
        print_results(response)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
