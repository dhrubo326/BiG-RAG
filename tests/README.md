# BiG-RAG Test Suite

Comprehensive testing framework for BiG-RAG to ensure production-ready quality.

---

## Quick Start

### 1. Install Test Dependencies

```cmd
cd tests
pip install -r requirements-test.txt
```

### 2. Set Environment Variables (Optional)

```cmd
REM For LLM extraction tests (optional - tests will be skipped if not set)
set OPENAI_API_KEY=your-api-key-here

REM To skip frontend tests
set SKIP_FRONTEND=true

REM To enable verbose logging
set BIGRAG_LOG_LEVEL=DEBUG
```

### 3. Run Tests

```cmd
REM Run all tests
pytest

REM Run with coverage report
pytest --cov=bigrag --cov-report=html

REM Run specific phase (recommended for first-time)
pytest tests/e2e/test_full_pipeline.py -v
```

---

## Test Structure

```
tests/
|
|-- conftest.py                    # Shared fixtures and pytest configuration
|-- pytest.ini                     # Pytest settings
|-- requirements-test.txt          # Test dependencies
|-- TEST_PLAN.md                   # Detailed test plan (READ THIS FIRST)
|-- README.md                      # This file
|
|-- fixtures/                      # Test data and fixtures
|   |-- test_documents.py          # Complex test data generators
|   `-- sample_data.json           # Pre-built test data
|
|-- unit/                          # Unit tests (60% coverage)
|   |-- test_base.py               # Base classes and schemas
|   |-- test_config.py             # Configuration management
|   |-- test_operate.py            # Graph operations
|   |-- test_reranker.py           # Semantic reranking
|   |-- test_storage.py            # Storage layer
|   `-- test_utils.py              # Utility functions
|
|-- integration/                   # Integration tests (30% coverage)
|   |-- test_entity_extraction.py  # LLM entity extraction
|   |-- test_graph_vector_sync.py  # Graph-vector consistency
|   |-- test_retrieval_pipeline.py # Three-path retrieval
|   `-- test_storage_consistency.py # Storage synchronization
|
|-- e2e/                           # End-to-end tests (10% coverage)
|   |-- test_document_lifecycle.py # Full document lifecycle
|   |-- test_full_pipeline.py      # Complete pipeline test
|   `-- test_three_path_retrieval.py # Path A, B, C validation
|
|-- regression/                    # Bug regression tests
|   `-- test_bug_fixes.py          # Tests for 6 fixed bugs
|
|-- api/                           # API endpoint tests
|   |-- test_graph_api.py          # Graph management endpoints
|   |-- test_search_api.py         # Search functionality
|   `-- test_server_endpoints.py   # Server health and status
|
|-- frontend/                      # Frontend integration tests
|   `-- test_ui_integration.py     # UI integration (optional)
|
|-- performance/                   # Performance and stress tests
|   |-- test_concurrency.py        # Concurrent operations
|   `-- test_large_scale.py        # 1000+ documents
|
|-- edge_cases/                    # Edge case and error handling
|   `-- test_edge_cases.py         # Unusual inputs and errors
|
`-- test_output/                   # Test artifacts (auto-generated)
    |-- logs/                      # Test execution logs
    `-- coverage/                  # Coverage reports
```

---

## Test Execution Order

**IMPORTANT:** Follow this order for systematic testing.

### Phase 1: Critical Path (MUST PASS)

Run these first. If any fail, STOP and fix before proceeding.

```cmd
REM Test 1: Full pipeline (insert -> query -> delete)
pytest tests/e2e/test_full_pipeline.py::test_complete_pipeline -v

REM Test 2: Regression tests (6 bug fixes)
pytest tests/regression/test_bug_fixes.py -v
```

**Exit Criteria:** 100% pass rate (all tests green)

---

### Phase 2: Core Functionality

Test all major components.

```cmd
REM Unit tests
pytest tests/unit/ -v

REM Integration tests
pytest tests/integration/ -v
```

**Exit Criteria:** >=95% pass rate

---

### Phase 3: Advanced Features

Test advanced retrieval and lifecycle.

```cmd
pytest tests/e2e/ -v
```

**Exit Criteria:** >=90% pass rate

---

### Phase 4: API Validation

Test server endpoints (requires server running).

```cmd
REM Start server in separate terminal
cd backend
python server.py --data_source demo_test

REM Run API tests
pytest tests/api/ -v
```

**Exit Criteria:** All API tests pass

---

### Phase 5: Robustness

Stress testing and edge cases.

```cmd
pytest tests/performance/ tests/edge_cases/ -v
```

**Exit Criteria:** No crashes, acceptable performance

---

## Test Categories (Markers)

Run tests by category using pytest markers:

```cmd
REM Critical tests only
pytest -m critical

REM Unit tests only
pytest -m unit

REM Integration tests only
pytest -m integration

REM End-to-end tests only
pytest -m e2e

REM Regression tests only
pytest -m regression

REM API tests only
pytest -m api

REM Performance tests only
pytest -m performance

REM Skip slow tests (>5 seconds)
pytest -m "not slow"

REM Windows-specific tests
pytest -m windows
```

---

## Advanced Usage

### Parallel Execution (Faster)

```cmd
REM Auto-detect CPU cores
pytest -n auto

REM Use specific number of workers
pytest -n 4
```

### Coverage Report

```cmd
REM Generate HTML coverage report
pytest --cov=bigrag --cov-report=html

REM View report
htmlcov\index.html
```

### Run Specific Tests

```cmd
REM Run specific file
pytest tests/unit/test_utils.py

REM Run specific test function
pytest tests/unit/test_utils.py::test_compute_mdhash_id

REM Run tests matching pattern
pytest -k "hash" -v
```

### Debugging Failed Tests

```cmd
REM Stop on first failure
pytest -x

REM Show print statements
pytest -s

REM Run last failed tests
pytest --lf

REM Show local variables on failure
pytest -l
```

### Performance Analysis

```cmd
REM Show slowest 10 tests
pytest --durations=10

REM Benchmark mode (for performance tests)
pytest tests/performance/ --benchmark-only
```

---

## Test Data

### Pre-Built Test Data

Tests use the **demo_test** dataset (already built KG) for realistic testing:

- **Location:** `expr/demo_test/`
- **Content:** Complex football/sports data with 50+ entities, 100+ relations
- **Advantages:**
  - Real-world complexity (multi-hop queries)
  - Tests entity extraction accuracy
  - Validates relation normalization
  - Stress tests retrieval system

### Synthetic Test Data

Some tests generate synthetic data using Faker library:

- 1000+ documents for stress testing
- Random queries for robustness
- Edge cases (empty strings, special characters, etc.)

---

## Environment Configuration

### Required Environment Variables

```cmd
REM None required - tests will use defaults
```

### Optional Environment Variables

```cmd
REM Enable LLM extraction tests (requires API key)
set OPENAI_API_KEY=sk-...

REM Skip frontend tests (if UI not implemented)
set SKIP_FRONTEND=true

REM Custom API base URL
set API_BASE_URL=http://localhost:8001

REM Test log level
set BIGRAG_LOG_LEVEL=WARNING

REM Custom working directory for tests
set TEST_WORKING_DIR=d:/test_output
```

---

## Interpreting Test Results

### Test Output Format

```
tests/unit/test_utils.py::test_compute_mdhash_id PASSED         [10%]
tests/unit/test_utils.py::test_normalize_entity_type PASSED     [20%]
tests/unit/test_storage.py::test_upsert_updates FAILED          [30%]
```

- **PASSED:** Test succeeded
- **FAILED:** Test failed (check assertion error)
- **SKIPPED:** Test skipped (usually due to missing dependencies)
- **ERROR:** Test error (usually import or setup issue)

### Common Failure Reasons

1. **ImportError:** Missing dependencies or bigrag not installed
   - **Solution:** `pip install -e ..`

2. **FileNotFoundError:** Missing test data or KG files
   - **Solution:** Build demo_test KG first or use fixtures

3. **AssertionError:** Test assertion failed
   - **Solution:** Check error message, may indicate actual bug

4. **Timeout:** Test took too long (>300 seconds)
   - **Solution:** Check for infinite loops or network issues

---

## Coverage Goals

| Component | Target Coverage | Priority |
|-----------|-----------------|----------|
| bigrag/bigrag.py | 85% | Critical |
| bigrag/operate.py | 85% | Critical |
| bigrag/storage.py | 80% | High |
| bigrag/base.py | 90% | High |
| bigrag/utils.py | 90% | High |
| bigrag/config.py | 80% | Medium |
| bigrag/reranker.py | 75% | Medium |

**Overall Target:** 80% code coverage

---

## Continuous Integration (CI)

To set up automated testing (optional):

```cmd
REM Create .github/workflows/tests.yml
REM Add pytest configuration
REM Configure coverage reporting
```

---

## Troubleshooting

### Problem: Tests fail with "No module named 'bigrag'"

**Solution:**
```cmd
cd tests
pip install -e ..
```

### Problem: API tests timeout or fail

**Solution:**
```cmd
REM Ensure backend server is running
cd backend
python server.py --data_source demo_test

REM In another terminal, run API tests
pytest tests/api/ -v
```

### Problem: Out of memory during performance tests

**Solution:**
```cmd
REM Run tests sequentially (disable parallel)
pytest tests/performance/ -n 0

REM Or skip performance tests
pytest -m "not performance"
```

### Problem: Windows encoding errors

**Solution:**
```cmd
set PYTHONIOENCODING=utf-8
pytest
```

### Problem: Permission errors creating test files

**Solution:**
```cmd
REM Run as administrator or use different working dir
set TEST_WORKING_DIR=C:\temp\bigrag_tests
pytest
```

---

## Best Practices

1. **Run critical tests first** - Catch major issues early
2. **Use markers** - Run specific test categories
3. **Check coverage** - Aim for 80%+ coverage
4. **Fix failures immediately** - Don't accumulate technical debt
5. **Update regression tests** - Add test when fixing a bug
6. **Use realistic data** - demo_test dataset is preferred
7. **Test on Windows** - Validate platform compatibility

---

## Contributing New Tests

When adding new tests:

1. **Choose correct directory** - unit/integration/e2e/etc.
2. **Follow naming convention** - `test_*.py` and `test_*()` functions
3. **Add markers** - Use `@pytest.mark.unit` etc.
4. **Use fixtures** - Reuse shared setup from conftest.py
5. **Document complex tests** - Add docstrings
6. **Test both success and failure** - Positive and negative cases
7. **Add to TEST_PLAN.md** - Update documentation

---

## Getting Help

- **Test Plan:** See `TEST_PLAN.md` for detailed strategy
- **Pytest Docs:** https://docs.pytest.org/
- **Coverage Docs:** https://coverage.readthedocs.io/
- **Issues:** Check GitHub issues for known problems

---

## Test Metrics Dashboard

After running tests, check:

```cmd
REM View coverage report
htmlcov\index.html

REM Check test summary
pytest --co -q

REM Show test statistics
pytest --collect-only
```

---

## Quick Reference

```cmd
REM Most common commands

REM 1. Run all tests with coverage
pytest --cov=bigrag --cov-report=html

REM 2. Run critical tests only
pytest -m critical -v

REM 3. Run fast tests (skip slow ones)
pytest -m "not slow" -n auto

REM 4. Debug single test
pytest tests/unit/test_utils.py::test_compute_mdhash_id -s -v

REM 5. Re-run failed tests
pytest --lf -v

REM 6. Generate HTML report
pytest --html=report.html --self-contained-html
```

---

## Test Suite Statistics

- **Total Test Files:** 21
- **Estimated Test Count:** 150+
- **Execution Time:** ~15-30 minutes (full suite)
- **Code Coverage Target:** 80%+
- **Platform:** Windows (primary), Linux (untested)
- **Python Version:** 3.11+

---

## License

Same as BiG-RAG project.

---

**Last Updated:** 2025-01-09
**Version:** 1.0
