"""
Quick Setup Test
Verifies all components are ready for BiG-RAG testing
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("="*80)
    print("Testing Module Imports")
    print("="*80)

    modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("openai", "OpenAI SDK"),
        ("tiktoken", "Tiktoken (OpenAI tokenizer)"),
        ("tenacity", "Tenacity (retry logic)"),
        ("networkx", "NetworkX (graph storage)"),
        ("faiss", "FAISS (vector search)"),
        ("spacy", "spaCy (NLP)"),
        ("nltk", "NLTK (NLP)"),
        ("aiohttp", "aiohttp (async HTTP)"),
        ("pydantic", "Pydantic (data validation)"),
    ]

    all_ok = True
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description:30s} - OK")
        except ImportError as e:
            print(f"✗ {description:30s} - MISSING")
            all_ok = False

    print("")
    return all_ok


def test_bigrag_import():
    """Test if BiGRAG can be imported"""
    print("="*80)
    print("Testing BiG-RAG Import")
    print("="*80)

    try:
        from bigrag import BiGRAG, QueryParam
        from bigrag.llm import gpt_4o_mini_complete, openai_embedding
        print("✓ BiG-RAG modules imported successfully")
        print("")
        return True
    except ImportError as e:
        print(f"✗ BiG-RAG import failed: {e}")
        print("")
        return False


def test_api_key():
    """Test if OpenAI API key is available"""
    print("="*80)
    print("Testing OpenAI API Key")
    print("="*80)

    api_key_file = Path("openai_api_key.txt")

    if not api_key_file.exists():
        print("✗ openai_api_key.txt not found")
        print("  Please create this file with your OpenAI API key")
        print("")
        return False

    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()

    if not api_key:
        print("✗ openai_api_key.txt is empty")
        print("")
        return False

    if not api_key.startswith("sk-"):
        print("⚠ API key doesn't start with 'sk-' (might be invalid)")
        print("")
        return False

    # Mask the key for security
    masked = api_key[:10] + "..." + api_key[-4:]
    print(f"✓ API key found: {masked}")
    print("")
    return True


def test_dataset():
    """Test if demo dataset exists"""
    print("="*80)
    print("Testing Demo Dataset")
    print("="*80)

    corpus_path = Path("datasets/demo_test/raw/corpus.jsonl")
    qa_path = Path("datasets/demo_test/raw/qa_test.json")

    all_ok = True

    if corpus_path.exists():
        import json
        count = sum(1 for _ in open(corpus_path, encoding='utf-8'))
        print(f"✓ Corpus found: {count} documents")
    else:
        print("✗ Corpus not found: datasets/demo_test/raw/corpus.jsonl")
        all_ok = False

    if qa_path.exists():
        import json
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"✓ Test questions found: {len(qa_data)} questions")
    else:
        print("✗ Test questions not found: datasets/demo_test/raw/qa_test.json")
        all_ok = False

    print("")
    return all_ok


def test_openai_connection():
    """Test OpenAI API connection"""
    print("="*80)
    print("Testing OpenAI API Connection")
    print("="*80)

    try:
        import os
        from openai import OpenAI

        # Load API key
        api_key_file = Path("openai_api_key.txt")
        if api_key_file.exists():
            with open(api_key_file, 'r') as f:
                api_key = f.read().strip()
            os.environ["OPENAI_API_KEY"] = api_key

        # Test connection with a minimal API call
        client = OpenAI()
        response = client.embeddings.create(
            input=["test"],
            model="text-embedding-3-small"
        )

        print("✓ OpenAI API connection successful")
        print(f"  Embedding dimension: {len(response.data[0].embedding)}")
        print("")
        return True

    except Exception as e:
        print(f"✗ OpenAI API connection failed: {e}")
        print("")
        return False


def main():
    """Run all setup tests"""
    print("\n")
    print("="*80)
    print("BiG-RAG Setup Verification")
    print("="*80)
    print("\n")

    results = {}

    # Test 1: Module imports
    results['imports'] = test_imports()

    # Test 2: BiGRAG import
    if results['imports']:
        results['bigrag'] = test_bigrag_import()
    else:
        print("⚠ Skipping BiG-RAG import test (dependencies missing)")
        print("")
        results['bigrag'] = False

    # Test 3: API key
    results['api_key'] = test_api_key()

    # Test 4: Dataset
    results['dataset'] = test_dataset()

    # Test 5: OpenAI connection (only if API key exists)
    if results['api_key']:
        results['openai'] = test_openai_connection()
    else:
        print("⚠ Skipping OpenAI connection test (API key not found)")
        print("")
        results['openai'] = False

    # Summary
    print("="*80)
    print("SETUP VERIFICATION SUMMARY")
    print("="*80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.capitalize():20s}: {status}")

    print("="*80)
    print("")

    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("")
        print("You're ready to run BiG-RAG tests:")
        print("  1. python test_build_graph.py")
        print("  2. python test_retrieval.py")
        print("  3. python test_end_to_end.py")
        print("")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("")
        print("Next steps:")

        if not results['imports']:
            print("  1. Install dependencies:")
            print("     pip install -r requirements_test.txt")
            print("     OR run: install_test_dependencies.bat")
            print("")

        if not results['api_key']:
            print("  2. Create openai_api_key.txt with your OpenAI API key:")
            print("     echo sk-your-key-here > openai_api_key.txt")
            print("")

        if not results['openai']:
            print("  3. Verify your OpenAI API key is valid and has credits")
            print("")

        return 1


if __name__ == "__main__":
    sys.exit(main())
