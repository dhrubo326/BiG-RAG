# BiG-RAG Test Plan

**Version:** 1.0
**Last Updated:** 2025-01-09
**Status:** Active

---

## 1. Test Objectives

### 1.1 Primary Goals
- Ensure BiG-RAG is production-ready with zero critical bugs
- Validate all core functionality works as designed
- Verify recent bug fixes do not regress
- Achieve minimum 80% code coverage
- Validate system performance under load
- Ensure API stability and correctness

### 1.2 Success Criteria
- [ ] All critical path tests pass (100%)
- [ ] All regression tests pass (100%)
- [ ] Code coverage >= 80%
- [ ] No critical or high-severity bugs
- [ ] API response time < 2 seconds (95th percentile)
- [ ] System handles 1000+ documents without crashes

---

## 2. Test Scope

### 2.1 In Scope
- Core BiG-RAG functionality (insert, query, delete)
- Three-path retrieval system (Entity, Relation, Chunk)
- Storage layer (Graph, Vector, KV)
- Entity extraction and normalization
- Document metadata preservation
- Hash-based node IDs
- Weighted RRF scoring
- Semantic reranking
- API endpoints (backend server)
- Document deletion cascade
- Configuration management
- Error handling and edge cases

### 2.2 Out of Scope
- RL training pipeline (verl framework)
- Frontend UI (if not yet implemented)
- Multi-node distributed testing
- Security penetration testing
- Performance optimization (only validation)

---

## 3. Test Types and Coverage

| Test Type | Files | Coverage Target | Priority |
|-----------|-------|-----------------|----------|
| Unit Tests | 6 files | 60% of tests | HIGH |
| Integration Tests | 4 files | 30% of tests | HIGH |
| End-to-End Tests | 3 files | 10% of tests | CRITICAL |
| Regression Tests | 1 file | 6 bug fixes | CRITICAL |
| API Tests | 3 files | All endpoints | HIGH |
| Frontend Tests | 1 file | Basic UI flow | MEDIUM |
| Performance Tests | 2 files | Stress testing | MEDIUM |
| Edge Cases | 1 file | Error handling | HIGH |

**Total Test Files:** 21+

---

## 4. Test Execution Order

### Phase 1: Critical Path Validation (Run First)
**Purpose:** Ensure basic functionality works before detailed testing

1. `tests/e2e/test_full_pipeline.py` - Complete insert -> query -> delete flow
2. `tests/regression/test_bug_fixes.py` - Verify 6 recent bug fixes

**Exit Criteria:** All tests pass. If any fail, stop and fix before proceeding.

---

### Phase 2: Core Functionality Testing
**Purpose:** Validate all major components work correctly

3. `tests/unit/test_utils.py` - Utility functions
4. `tests/unit/test_storage.py` - Storage layer
5. `tests/unit/test_operate.py` - Graph operations
6. `tests/unit/test_config.py` - Configuration management
7. `tests/integration/test_storage_consistency.py` - Storage sync
8. `tests/integration/test_retrieval_pipeline.py` - Three-path retrieval

**Exit Criteria:** >=95% pass rate. Document failures for investigation.

---

### Phase 3: Advanced Features Testing
**Purpose:** Validate advanced features and integrations

9. `tests/e2e/test_three_path_retrieval.py` - Path A, B, C validation
10. `tests/e2e/test_document_lifecycle.py` - Complete document lifecycle
11. `tests/integration/test_entity_extraction.py` - LLM extraction
12. `tests/integration/test_graph_vector_sync.py` - Graph-vector consistency

**Exit Criteria:** >=90% pass rate.

---

### Phase 4: API and Interface Testing
**Purpose:** Validate external interfaces

13. `tests/api/test_server_endpoints.py` - API server endpoints
14. `tests/api/test_search_api.py` - Search functionality
15. `tests/api/test_graph_api.py` - Graph management API
16. `tests/frontend/test_ui_integration.py` - UI integration (if applicable)

**Exit Criteria:** All API tests pass. Frontend tests optional.

---

### Phase 5: Robustness and Performance
**Purpose:** Validate system handles stress and edge cases

17. `tests/performance/test_large_scale.py` - 1000+ documents
18. `tests/performance/test_concurrency.py` - Concurrent operations
19. `tests/edge_cases/test_edge_cases.py` - Unusual inputs

**Exit Criteria:** No crashes. Performance within acceptable limits.

---

### Phase 6: Specialized Unit Tests
**Purpose:** Deep validation of specific components

20. `tests/unit/test_base.py` - Base classes and schemas
21. `tests/unit/test_reranker.py` - Semantic reranking

**Exit Criteria:** All pass.

---

## 5. Test Environment

### 5.1 Hardware Requirements
- **OS:** Windows 10/11
- **RAM:** Minimum 8GB (16GB recommended)
- **Disk:** 5GB free space for test data
- **CPU:** Multi-core recommended for parallel testing

### 5.2 Software Requirements
- Python 3.11+
- Virtual environment (venv)
- All dependencies from `requirements-test.txt`
- Optional: OpenAI API key (for LLM extraction tests)

### 5.3 Environment Setup
```cmd
cd tests
python -m venv test_venv
test_venv\Scripts\activate
pip install -r requirements-test.txt
pip install -e ..
```

---

## 6. Test Data

### 6.1 Fixed Test Data
- Located in `tests/fixtures/`
- `sample_corpus.json` - 100 sample documents
- `sample_qa.json` - 50 question-answer pairs
- `test_documents.py` - Programmatic test data generators

### 6.2 Generated Test Data
- Faker library generates synthetic data
- 1000+ documents for stress testing
- Random queries for robustness testing

---

## 7. Test Execution Commands

### 7.1 Run All Tests
```cmd
pytest
```

### 7.2 Run by Phase (Recommended)
```cmd
REM Phase 1: Critical
pytest tests/e2e/test_full_pipeline.py tests/regression/test_bug_fixes.py -v

REM Phase 2: Core
pytest tests/unit/ tests/integration/ -v

REM Phase 3: Advanced
pytest tests/e2e/ -v

REM Phase 4: API
pytest tests/api/ -v

REM Phase 5: Performance
pytest tests/performance/ tests/edge_cases/ -v
```

### 7.3 Run by Marker
```cmd
REM Critical tests only
pytest -m critical

REM Unit tests only
pytest -m unit

REM Skip slow tests
pytest -m "not slow"
```

### 7.4 Run with Coverage
```cmd
pytest --cov=bigrag --cov-report=html
```

### 7.5 Parallel Execution (Faster)
```cmd
pytest -n auto
```

---

## 8. Defect Management

### 8.1 Severity Levels
- **CRITICAL:** System crash, data loss, security breach
- **HIGH:** Major feature broken, incorrect results
- **MEDIUM:** Minor feature issue, performance degradation
- **LOW:** Cosmetic issues, documentation errors

### 8.2 Bug Reporting
When a test fails:
1. Note the test name and file
2. Capture the assertion error message
3. Check if issue is reproducible
4. Classify severity
5. Document in issue tracker (GitHub Issues)

---

## 9. Test Metrics

Track these metrics during testing:

| Metric | Formula | Target |
|--------|---------|--------|
| **Pass Rate** | (Passed / Total) * 100 | >= 95% |
| **Code Coverage** | Lines covered / Total lines | >= 80% |
| **Defect Density** | Bugs found / Total tests | <= 5% |
| **Execution Time** | Total time for all tests | <= 30 min |

---

## 10. Known Limitations

### 10.1 Test Limitations
- LLM extraction tests require OpenAI API key (can be skipped)
- Frontend tests require UI to be implemented
- Performance tests may take 5-10 minutes
- Some tests require internet connection (for embedding models)

### 10.2 Platform Limitations
- Windows-only testing (Linux/Mac not validated in this plan)
- Single-machine testing (no distributed/multi-node tests)

---

## 11. Regression Test Coverage

Tests for previously fixed bugs:

| Bug ID | Description | Test Function | File |
|--------|-------------|---------------|------|
| Bug #1 | Wrong hash prefix (edge deletion) | `test_bug1_edge_deletion_prefix` | `test_bug_fixes.py` |
| Bug #2 | drop() deletes all documents | `test_bug2_single_document_deletion` | `test_bug_fixes.py` |
| Bug #3 | Undefined load_env_file() | `test_bug3_reload_config` | `test_bug_fixes.py` |
| Bug #4 | KeyError on missing dict keys | `test_bug4_defensive_dict_access` | `test_bug_fixes.py` |
| Bug #5 | upsert() doesn't update | `test_bug5_upsert_updates` | `test_bug_fixes.py` |
| Bug #6 | Wrong type annotations | `test_bug6_type_annotations` | `test_bug_fixes.py` |

---

## 12. Continuous Testing

### 12.1 Pre-Commit Checklist
Before committing code:
- [ ] Run critical tests: `pytest -m critical`
- [ ] Run affected unit tests
- [ ] Verify no new failures

### 12.2 Pre-Release Checklist
Before release:
- [ ] Run full test suite: `pytest`
- [ ] Verify coverage >= 80%
- [ ] All critical/high bugs resolved
- [ ] Performance tests pass
- [ ] API tests pass
- [ ] Regression tests pass

---

## 13. Test Maintenance

### 13.1 When to Update Tests
- After fixing a bug (add regression test)
- After adding a feature (add corresponding tests)
- When test becomes flaky (fix or remove)
- When requirements change (update assertions)

### 13.2 Test Review Schedule
- **Weekly:** Review failed tests
- **Monthly:** Review test coverage
- **Quarterly:** Refactor outdated tests

---

## 14. Appendix

### 14.1 Useful Commands

```cmd
REM Install test dependencies
pip install -r requirements-test.txt

REM Run tests with verbose output
pytest -v

REM Run specific test file
pytest tests/unit/test_utils.py

REM Run specific test function
pytest tests/unit/test_utils.py::test_compute_mdhash_id

REM Run and show print statements
pytest -s

REM Run and stop on first failure
pytest -x

REM Generate HTML report
pytest --html=report.html --self-contained-html

REM Run only failed tests from last run
pytest --lf

REM Show slowest tests
pytest --durations=10
```

### 14.2 Troubleshooting

**Issue:** Tests fail with import errors
**Solution:** Ensure bigrag is installed: `pip install -e ..`

**Issue:** API tests timeout
**Solution:** Start backend server first: `cd backend && python server.py`

**Issue:** Out of memory errors
**Solution:** Run tests in smaller batches or disable parallel execution

**Issue:** Windows encoding errors
**Solution:** Set `PYTHONIOENCODING=utf-8` before running tests

---

## 15. Sign-Off

Test plan approved by: _________________
Date: _________________

Test execution completed by: _________________
Date: _________________

All critical tests passed: [ ] Yes [ ] No
Code coverage achieved: _____%
Total bugs found: _____
Release approved: [ ] Yes [ ] No
