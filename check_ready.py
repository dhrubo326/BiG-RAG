"""
Quick readiness check for BiG-RAG testing
Run this with venv activated to verify everything is ready
"""
import sys
from pathlib import Path

print("\n" + "="*80)
print("BiG-RAG Readiness Check")
print("="*80 + "\n")

# Check 1: Python packages
print("1. Checking Python packages...")
missing = []
packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'openai': 'OpenAI SDK',
    'faiss': 'FAISS',
    'tiktoken': 'Tiktoken',
    'tenacity': 'Tenacity',
    'networkx': 'NetworkX',
}

for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f"   OK - {name}")
    except ImportError:
        print(f"   MISSING - {name}")
        missing.append(pkg)

if missing:
    print(f"\n   Install missing packages: pip install {' '.join(missing)}")
else:
    print("   All packages installed!")

print()

# Check 2: BiGRAG module
print("2. Checking BiG-RAG module...")
try:
    from bigrag import BiGRAG, QueryParam
    from bigrag.llm import gpt_4o_mini_complete, openai_embedding
    print("   OK - BiG-RAG can be imported")
except ImportError as e:
    print(f"   ERROR - {e}")

print()

# Check 3: API Key
print("3. Checking OpenAI API key...")
api_key_file = Path("openai_api_key.txt")
if api_key_file.exists():
    key = api_key_file.read_text().strip()
    if key.startswith("sk-"):
        masked = key[:10] + "..." + key[-4:]
        print(f"   OK - API key found: {masked}")
    else:
        print("   WARNING - Key doesn't start with 'sk-'")
else:
    print("   MISSING - Create openai_api_key.txt")

print()

# Check 4: Dataset
print("4. Checking demo dataset...")
corpus = Path("datasets/demo_test/raw/corpus.jsonl")
qa = Path("datasets/demo_test/raw/qa_test.json")

if corpus.exists():
    lines = sum(1 for _ in open(corpus, encoding='utf-8'))
    print(f"   OK - Corpus: {lines} documents")
else:
    print("   MISSING - corpus.jsonl")

if qa.exists():
    import json
    qa_data = json.loads(qa.read_text(encoding='utf-8'))
    print(f"   OK - Test questions: {len(qa_data)} questions")
else:
    print("   MISSING - qa_test.json")

print()

# Check 5: OpenAI connection (optional)
print("5. Testing OpenAI API connection...")
try:
    import os
    from openai import OpenAI

    if api_key_file.exists():
        os.environ["OPENAI_API_KEY"] = api_key_file.read_text().strip()

    client = OpenAI()
    response = client.embeddings.create(
        input=["test connection"],
        model="text-embedding-3-small"
    )
    print(f"   OK - API connection successful")
    print(f"   Embedding dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"   ERROR - {e}")

print()
print("="*80)
print("READY TO RUN TESTS!")
print("="*80)
print("\nNext steps:")
print("  1. python test_build_graph.py")
print("  2. python test_retrieval.py")
print("  3. python test_end_to_end.py")
print()
