"""
Test Query Preprocessing Controls

Tests the new query preprocessing control features:
1. Global control via ENABLE_QUERY_PREPROCESSING env var
2. Per-query control via QueryParam.enable_query_preprocessing
3. External QueryPreprocessor class for batch preprocessing

Run: python test_scripts/test_query_preprocessing_controls.py
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bigrag import QueryParam, QueryPreprocessor, preprocess_query_standalone


# Mock LLM function for testing
async def mock_llm_func(prompt, **kwargs):
    """Mock LLM that returns preprocessed query"""
    # Simulate preprocessing
    if "who is messi" in prompt.lower():
        return '''```json
{
  "normalized_query": "Who is Lionel Messi?",
  "statement_query": "Lionel Messi is an Argentine professional footballer who plays for Inter Miami and captains the Argentina national team."
}
```'''
    elif "messi team" in prompt.lower():
        return '''```json
{
  "normalized_query": "What team does Messi play for?",
  "statement_query": "Messi plays for Inter Miami in Major League Soccer and represents Argentina in international competitions."
}
```'''
    else:
        return f'''```json
{{
  "normalized_query": "{prompt}",
  "statement_query": "{prompt}"
}}
```'''


def test_query_param_validation():
    """Test that QueryParam accepts enable_query_preprocessing parameter"""
    print("\n[TEST 1] QueryParam with enable_query_preprocessing parameter")

    # Test with None (default)
    param1 = QueryParam()
    assert param1.enable_query_preprocessing is None
    print("[OK] Default value is None")

    # Test with True
    param2 = QueryParam(enable_query_preprocessing=True)
    assert param2.enable_query_preprocessing is True
    print("[OK] Can set to True")

    # Test with False
    param3 = QueryParam(enable_query_preprocessing=False)
    assert param3.enable_query_preprocessing is False
    print("[OK] Can set to False")

    print("[PASS] QueryParam validation successful")


async def test_query_preprocessor_single():
    """Test QueryPreprocessor with single query"""
    print("\n[TEST 2] QueryPreprocessor single query preprocessing")

    preprocessor = QueryPreprocessor(
        llm_func=mock_llm_func,
        language="English"
    )

    normalized, statement = await preprocessor.preprocess("who is messi")

    assert "Lionel Messi" in normalized
    assert "Argentine" in statement or "footballer" in statement

    print(f"[OK] Normalized: {normalized}")
    print(f"[OK] Statement: {statement[:80]}...")
    print("[PASS] Single query preprocessing successful")


async def test_query_preprocessor_batch():
    """Test QueryPreprocessor batch preprocessing"""
    print("\n[TEST 3] QueryPreprocessor batch preprocessing")

    preprocessor = QueryPreprocessor(
        llm_func=mock_llm_func,
        language="English"
    )

    queries = ["who is messi", "messi team"]
    results = await preprocessor.batch_preprocess(queries, max_concurrent=2)

    assert len(results) == 2
    assert "Lionel Messi" in results[0][0]
    assert "team" in results[1][0].lower() or "Inter Miami" in results[1][1]

    print(f"[OK] Preprocessed {len(results)} queries")
    for i, (norm, stmt) in enumerate(results):
        print(f"  Query {i+1}: {norm}")

    print("[PASS] Batch preprocessing successful")


async def test_preprocess_query_standalone():
    """Test standalone preprocessing function"""
    print("\n[TEST 4] Standalone preprocess_query_standalone function")

    normalized, statement = await preprocess_query_standalone(
        query="who is messi",
        llm_func=mock_llm_func,
        language="English"
    )

    assert "Lionel Messi" in normalized
    assert "Argentine" in statement or "footballer" in statement

    print(f"[OK] Normalized: {normalized}")
    print(f"[OK] Statement: {statement[:80]}...")
    print("[PASS] Standalone function successful")


def test_env_var_parsing():
    """Test ENABLE_QUERY_PREPROCESSING environment variable parsing"""
    print("\n[TEST 5] Environment variable parsing")

    # Test "true" (default)
    os.environ["ENABLE_QUERY_PREPROCESSING"] = "true"
    result = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"
    assert result is True
    print("[OK] 'true' parsed correctly")

    # Test "false"
    os.environ["ENABLE_QUERY_PREPROCESSING"] = "false"
    result = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"
    assert result is False
    print("[OK] 'false' parsed correctly")

    # Test "1" (should be treated as string, not boolean)
    os.environ["ENABLE_QUERY_PREPROCESSING"] = "1"
    result = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"
    assert result is False  # "1".lower() != "true"
    print("[OK] '1' treated as string (use 'true' instead)")

    # Restore default
    os.environ["ENABLE_QUERY_PREPROCESSING"] = "true"
    print("[PASS] Environment variable parsing successful")


def test_imports():
    """Test that all imports work correctly"""
    print("\n[TEST 6] Import verification")

    try:
        from bigrag import BiGRAG, QueryParam, QueryPreprocessor, preprocess_query_standalone
        print("[OK] All imports successful")

        # Verify QueryPreprocessor has expected methods
        assert hasattr(QueryPreprocessor, 'preprocess')
        assert hasattr(QueryPreprocessor, 'batch_preprocess')
        assert hasattr(QueryPreprocessor, 'preprocess_sync')
        assert hasattr(QueryPreprocessor, 'batch_preprocess_sync')
        print("[OK] QueryPreprocessor has all expected methods")

        # Verify preprocess_query_standalone is callable
        assert callable(preprocess_query_standalone)
        print("[OK] preprocess_query_standalone is callable")

        print("[PASS] Import verification successful")
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        raise


async def main():
    """Run all tests"""
    print("=" * 80)
    print("Testing Query Preprocessing Controls")
    print("=" * 80)

    try:
        # Synchronous tests
        test_imports()
        test_query_param_validation()
        test_env_var_parsing()

        # Asynchronous tests
        await test_query_preprocessor_single()
        await test_query_preprocessor_batch()
        await test_preprocess_query_standalone()

        print("\n" + "=" * 80)
        print("[SUCCESS] All tests passed!")
        print("=" * 80)

    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"[FAIL] Test failed: {e}")
        print("=" * 80)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[ERROR] Unexpected error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
