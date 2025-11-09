# BiG-RAG Test Plan

**Version:** 1.1
**Last Updated:** 2025-01-09
**Status:** Active

## Update Log

- **2025-01-09:** Phase 3 Integration Tests implemented (30 tests across 4 files)
  - Added test_entity_extraction.py (6 tests)
  - Added test_graph_vector_sync.py (6 tests)
  - Added test_retrieval_pipeline.py (10 tests)
  - Added test_storage_consistency.py (8 tests)
  - Discovered and fixed Bug #7 (dict unpacking in operate.py)
  - Fixed Bug #8 (test-level attribute name error)

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
| Unit Tests | 10 files (+4 NEW) | 60% of tests | HIGH |
| Integration Tests | 4 files (30 tests) | 30% of tests | HIGH |
| End-to-End Tests | 3 files | 10% of tests | CRITICAL |
| Regression Tests | 1 file | 6 bug fixes | CRITICAL |
| API Tests | 3 files | All endpoints | HIGH |
| Frontend Tests | 1 file | Basic UI flow | MEDIUM |
| Performance Tests | 2 files | Stress testing | MEDIUM |
| Edge Cases | 1 file | Error handling | HIGH |

**Total Test Files:** 25+ (4 new unit tests + 4 integration tests added 2025-01-09)

### NEW Unit Tests (2025-01-09)
1. **test_chunking.py** - Comprehensive chunking tests (40+ tests)
   - Basic chunking, metadata preservation, edge cases
2. **test_graph_building.py** - Graph construction tests (30+ tests)
   - Node creation, weight aggregation, graph structure
3. **test_embedding.py** - Embedding preparation tests (35+ tests)
   - Function wrapping, batch processing, dimensions
4. **test_retrieval.py** - Retrieval logic tests (40+ tests)
   - RRF scoring, paths A/B/C, weighted retrieval

### NEW Integration Tests (2025-01-09)
1. **test_entity_extraction.py** - LLM entity extraction (6 tests)
   - Entity normalization, type mapping, relation quality, caching, metadata context, multi-entity handling
2. **test_graph_vector_sync.py** - Graph-vector synchronization (6 tests)
   - Entity/relation roundtrip, edge deletion, embedding consistency, vector search integration
3. **test_retrieval_pipeline.py** - Complete retrieval pipeline (10 tests)
   - Entity/relation/chunk paths, hybrid mode, RRF scoring, reranking, metadata preservation
4. **test_storage_consistency.py** - Storage layer sync (8 tests)
   - Insert/delete/upsert sync, concurrent operations, entity counts, chunk-entity mapping

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

**Unit Tests:**
3. `tests/unit/test_utils.py` - Utility functions
4. `tests/unit/test_storage.py` - Storage layer
5. `tests/unit/test_operate.py` - Graph operations
6. `tests/unit/test_config.py` - Configuration management
7. `tests/unit/test_base.py` - Base classes and schemas
8. `tests/unit/test_reranker.py` - Semantic reranking
9. `tests/unit/test_chunking.py` - Text chunking (NEW 2025-01-09)
10. `tests/unit/test_graph_building.py` - Graph construction (NEW 2025-01-09)
11. `tests/unit/test_embedding.py` - Embedding preparation (NEW 2025-01-09)
12. `tests/unit/test_retrieval.py` - Retrieval logic (NEW 2025-01-09)

**Integration Tests (NEW 2025-01-09):**
13. `tests/integration/test_storage_consistency.py` - Storage sync (8 tests)
14. `tests/integration/test_retrieval_pipeline.py` - Three-path retrieval (10 tests)
15. `tests/integration/test_entity_extraction.py` - LLM extraction (6 tests)
16. `tests/integration/test_graph_vector_sync.py` - Graph-vector consistency (6 tests)

**Exit Criteria:** >=95% pass rate. Document failures for investigation.

---

### Phase 3: End-to-End Workflows
**Purpose:** Validate complete user workflows

17. `tests/e2e/test_three_path_retrieval.py` - Path A, B, C validation
18. `tests/e2e/test_document_lifecycle.py` - Complete document lifecycle

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

### 7.6 Manual One-by-One Testing Guide

**Purpose:** Execute each test file individually to isolate failures and understand component behavior.

**Prerequisites:**
```cmd
cd tests
..\venv\Scripts\activate  # Use your venv path
```

#### Phase 1: Critical Path (MUST PASS - Stop if ANY fail)

```cmd
# Test 1/25: Full Pipeline E2E
pytest e2e/test_full_pipeline.py -v --tb=short
# Expected: 5-7 tests PASSED
# Tests: Complete insert -> query -> delete workflow
# Time: ~30 seconds

# Test 2/25: Regression Tests (6 Bug Fixes)
pytest regression/test_bug_fixes.py -v --tb=short
# Expected: 6 tests PASSED
# Tests: Bug #1-6 validation (edge deletion, document deletion, etc.)
# Time: ~20 seconds

# CHECKPOINT: If ANY test fails, STOP and fix before proceeding
```

#### Phase 2: Core Functionality - Unit Tests

```cmd
# Test 3/25: Base Classes and Schemas
pytest unit/test_base.py -v --tb=short
# Expected: 10-15 tests PASSED
# Tests: BaseGraphStorage, BaseVectorStorage, BaseKVStorage, TextChunkSchema
# Time: ~10 seconds

# Test 4/25: Utility Functions
pytest unit/test_utils.py -v --tb=short
# Expected: 15-20 tests PASSED
# Tests: compute_mdhash_id, normalize_entity_type, string utilities
# Time: ~15 seconds

# Test 5/25: Storage Layer
pytest unit/test_storage.py -v --tb=short
# Expected: 20-25 tests PASSED
# Tests: JsonKVStorage, NanoVectorDBStorage, NetworkXStorage CRUD operations
# Time: ~20 seconds

# Test 6/25: Configuration Management
pytest unit/test_config.py -v --tb=short
# Expected: 8-12 tests PASSED
# Tests: Config loading, validation, default values
# Time: ~10 seconds

# Test 7/25: Graph Operations
pytest unit/test_operate.py -v --tb=short
# Expected: 25-30 tests PASSED
# Tests: Entity extraction, chunking, graph building, normalization
# Time: ~30 seconds

# Test 8/25: Semantic Reranking
pytest unit/test_reranker.py -v --tb=short
# Expected: 10-15 tests PASSED
# Tests: Cross-encoder reranking, score computation
# Time: ~15 seconds
# Note: May download cross-encoder model on first run (~330MB)

# Test 9/25: Document Chunking (NEW)
pytest unit/test_chunking.py -v --tb=short
# Expected: 40+ tests PASSED
# Tests: Text chunking, metadata preservation, edge cases
# Time: ~40 seconds

# Test 10/25: Graph Building Logic (NEW)
pytest unit/test_graph_building.py -v --tb=short
# Expected: 30+ tests PASSED
# Tests: Node creation, weight aggregation, graph structure validation
# Time: ~30 seconds

# Test 11/25: Embedding Preparation (NEW)
pytest unit/test_embedding.py -v --tb=short
# Expected: 35+ tests PASSED
# Tests: Embedding function wrapping, batch processing, dimensions
# Time: ~35 seconds

# Test 12/25: Retrieval Logic (NEW)
pytest unit/test_retrieval.py -v --tb=short
# Expected: 40+ tests PASSED
# Tests: RRF scoring, Path A/B/C retrieval, weighted ranking
# Time: ~40 seconds

# CHECKPOINT: Target >=95% pass rate for Phase 2
```

#### Phase 2: Core Functionality - Integration Tests

```cmd
# Test 13/25: Storage Consistency (NEW - 2025-01-09)
pytest integration/test_storage_consistency.py -v --tb=short
# Expected: 8 tests PASSED
# Tests: Insert/delete/upsert sync, concurrent operations, entity counts, chunk-entity mapping
# Time: ~30 seconds

# Test 14/25: Retrieval Pipeline (NEW - 2025-01-09)
pytest integration/test_retrieval_pipeline.py -v --tb=short
# Expected: 10 tests PASSED
# Tests: Entity/relation/chunk paths, hybrid mode, RRF, reranking, metadata preservation
# Time: ~40 seconds

# Test 15/25: Entity Extraction (NEW - 2025-01-09)
pytest integration/test_entity_extraction.py -v --tb=short
# Expected: 6 tests PASSED (requires OPENAI_API_KEY)
# Tests: LLM extraction, entity normalization, type mapping, relation quality, caching
# Time: ~60-90 seconds (with API calls)
# Note: Tests will FAIL if API key not set (not skipped)

# Test 16/25: Graph-Vector Sync (NEW - 2025-01-09)
pytest integration/test_graph_vector_sync.py -v --tb=short
# Expected: 6 tests PASSED
# Tests: Entity/relation roundtrip, edge deletion, embedding consistency, vector search
# Time: ~25 seconds

# CHECKPOINT: All 30 integration tests should pass (100%)
```

#### Phase 3: End-to-End Workflows

```cmd
# Test 17/25: Document Lifecycle
pytest e2e/test_document_lifecycle.py -v --tb=short
# Expected: 8-10 tests PASSED
# Tests: Complete document CRUD operations
# Time: ~30 seconds

# Test 18/25: Three-Path Retrieval Validation
pytest e2e/test_three_path_retrieval.py -v --tb=short
# Expected: 10-12 tests PASSED
# Tests: Path A (entity), Path B (relation), Path C (chunk) validation
# Time: ~35 seconds

# CHECKPOINT: Target >=90% pass rate for Phase 3 (E2E tests)
```

#### Phase 4: API Tests (Requires Backend Server)

**PREREQUISITE:** Start backend server in separate terminal
```cmd
# Terminal 1:
cd backend
python server.py --data_source demo_test

# Wait for "Server started" message before running tests
```

```cmd
# Terminal 2: Run API tests

# Test 19/25: Server Endpoints
pytest api/test_server_endpoints.py -v --tb=short
# Expected: 8-10 tests PASSED
# Tests: /health, /stats, /status endpoints
# Time: ~15 seconds

# Test 20/25: Search API
pytest api/test_search_api.py -v --tb=short
# Expected: 12-15 tests PASSED
# Tests: /search endpoint, query modes, response format
# Time: ~25 seconds

# Test 21/25: Graph API
pytest api/test_graph_api.py -v --tb=short
# Expected: 10-12 tests PASSED
# Tests: /documents, /entities, /rebuild endpoints
# Time: ~20 seconds

# CHECKPOINT: All API tests must pass (100%)
```

#### Phase 5: Performance and Edge Cases

```cmd
# Test 22/25: Large Scale Performance
pytest performance/test_large_scale.py -v --tb=short
# Expected: 5-8 tests PASSED
# Tests: 1000+ document handling, memory usage, scalability
# Time: ~5-10 MINUTES (slow)
# Note: May take significant time and memory

# Test 23/25: Concurrency
pytest performance/test_concurrency.py -v --tb=short
# Expected: 6-8 tests PASSED
# Tests: Concurrent inserts, queries, deletes
# Time: ~30 seconds

# Test 24/25: Edge Cases
pytest edge_cases/test_edge_cases.py -v --tb=short
# Expected: 15-20 tests PASSED
# Tests: Empty inputs, special characters, malformed data
# Time: ~25 seconds

# CHECKPOINT: No crashes, acceptable performance
```

#### Phase 6: Frontend (Optional - Skip if UI not ready)

```cmd
# Test 25/25: UI Integration
pytest frontend/test_ui_integration.py -v --tb=short
# Expected: 5-8 tests PASSED (or SKIPPED)
# Tests: UI integration with backend
# Time: ~20 seconds
# Note: May be skipped if SKIP_FRONTEND=true
```

#### Summary Command (All Tests)

```cmd
# Run all 25+ test files (use after individual testing)
pytest -v --tb=short
```

---

## 8. Test Output Interpretation

### 8.1 Understanding Pytest Output

#### Success Example
```
tests/unit/test_utils.py::test_compute_mdhash_id PASSED                    [ 10%]
tests/unit/test_utils.py::test_normalize_entity_type PASSED                [ 20%]
tests/unit/test_utils.py::test_string_normalization PASSED                 [ 30%]

======================== 10 passed in 2.35s ========================
```

**Indicators:**
- ✅ All tests show `PASSED`
- ✅ Final line shows `X passed in Y.Ys`
- ✅ No `FAILED`, `ERROR`, or traceback messages

**Action:** Continue to next test file.

---

#### Failure Example
```
tests/unit/test_storage.py::test_upsert_updates FAILED                     [ 30%]
_________________________ test_upsert_updates __________________________
    def test_upsert_updates():
        storage = JsonKVStorage()
>       await storage.upsert("key1", {"value": "new"})
E       AssertionError: Expected 'new' but got 'old'

tests/unit/test_storage.py:45: AssertionError
==================== 1 failed, 9 passed in 2.51s ====================
```

**Indicators:**
- ❌ Test shows `FAILED` status
- ❌ Traceback shows assertion error
- ❌ Final line shows `X failed, Y passed`

**Action:** Follow Bug Triage Workflow (Section 8.3 below).

---

#### Skipped Test Example
```
tests/integration/test_entity_extraction.py::test_llm_extraction SKIPPED   [ 40%]

Reason: No OPENAI_API_KEY environment variable set
==================== 8 passed, 2 skipped in 1.85s ====================
```

**Indicators:**
- ⚠️ Test shows `SKIPPED` status
- ℹ️ Reason provided (usually missing dependency)
- ✅ Final line shows `X passed, Y skipped`

**Action:** OK to skip if dependency is optional (e.g., OpenAI API key).

---

#### Error Example
```
tests/api/test_search_api.py::test_search_endpoint ERROR                   [ 50%]
_________________________ test_search_endpoint _________________________
E   requests.exceptions.ConnectionError: Connection refused (localhost:8001)

tests/api/test_search_api.py:15: ConnectionError
==================== 1 error in 0.35s ====================
```

**Indicators:**
- ❌ Test shows `ERROR` status
- ❌ Exception raised before test logic runs
- ❌ Usually infrastructure issue (server not running)

**Action:** Fix environment issue (e.g., start backend server).

---

### 8.2 Test Result Status Types

| Status | Meaning | Severity | Action |
|--------|---------|----------|--------|
| **PASSED** | Test succeeded | ✅ None | Continue testing |
| **FAILED** | Assertion failed | ❌ High | Investigate bug (see Section 8.3) |
| **ERROR** | Exception raised | ❌ High | Fix environment or code error |
| **SKIPPED** | Test skipped | ⚠️ Low | OK if dependency optional |
| **XFAIL** | Expected to fail | ℹ️ None | Known issue, documented |
| **XPASS** | Unexpectedly passed | ⚠️ Medium | Remove XFAIL marker |

---

### 8.3 Bug Triage Workflow

When a test **FAILS** or has **ERROR**, follow this decision tree:

```
Test Failed/Error
    │
    ├─ Is it a Critical Path test (Phase 1)?
    │   ├─ YES → STOP ALL TESTING
    │   │         Document as CRITICAL bug
    │   │         Fix immediately before proceeding
    │   │
    │   └─ NO → Continue to next step
    │
    ├─ Can you reproduce it?
    │   ├─ Run test again: pytest <file>::<function> -v
    │   ├─ YES → Real bug, proceed to classification
    │   └─ NO → Flaky test, document and re-run
    │
    ├─ Is it an ERROR (exception) or FAILED (assertion)?
    │   ├─ ERROR → Likely environment issue
    │   │          Check: Server running? Dependencies installed?
    │   │          Fix environment and re-run
    │   │
    │   └─ FAILED → Real bug in code, proceed to classification
    │
    ├─ Classify Severity:
    │   ├─ CRITICAL: System crash, data loss, security issue
    │   │   └─ Action: STOP testing, fix immediately
    │   │
    │   ├─ HIGH: Major feature broken, incorrect results
    │   │   └─ Action: Document, fix before release
    │   │
    │   ├─ MEDIUM: Minor feature issue, performance degradation
    │   │   └─ Action: Document, schedule fix
    │   │
    │   └─ LOW: Cosmetic issue, edge case
    │       └─ Action: Document, optional fix
    │
    └─ Create Bug Report (see Section 8.4 below)
```

---

### 8.4 Bug Reporting Template

When a test fails, document it using this template:

```markdown
## Bug Report #<number>

**Severity:** [CRITICAL / HIGH / MEDIUM / LOW]
**Test File:** tests/<category>/<file>.py
**Test Function:** test_<function_name>
**Discovered By:** [Your name or "Automated Testing"]
**Date:** YYYY-MM-DD

### Description
[Brief 1-2 sentence description of the bug]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Error Message
```
[Paste exact error message and traceback]
```

### Steps to Reproduce
1. Navigate to tests directory: `cd tests`
2. Activate venv: `..\venv\Scripts\activate`
3. Run test: `pytest <file>::<function> -v`
4. Observe failure at line <number>

### Environment
- OS: Windows 10/11
- Python Version: 3.11.x
- BiGRAG Version: [commit hash or version]
- Pytest Version: [from `pip show pytest`]

### Root Cause Analysis
[Your analysis of why this happens - fill after investigation]

### Suggested Fix
[Proposed solution - fill after analysis]

### Related Files
- Source file: bigrag/<module>.py:<line>
- Test file: tests/<category>/<file>.py:<line>

### Priority
[URGENT / HIGH / NORMAL / LOW]

### Assigned To
[Team member name or "Unassigned"]
```

**Example:**

```markdown
## Bug Report #7

**Severity:** HIGH
**Test File:** tests/unit/test_storage.py
**Test Function:** test_upsert_updates
**Discovered By:** Manual Testing - Phase 2
**Date:** 2025-01-09

### Description
JsonKVStorage.upsert() does not update existing keys.

### Expected Behavior
Calling `upsert("key1", {"value": "new"})` on existing key should update the value to "new".

### Actual Behavior
Value remains "old" after upsert() call.

### Error Message
```
E       AssertionError: Expected 'new' but got 'old'
tests/unit/test_storage.py:45: AssertionError
```

### Steps to Reproduce
1. cd tests
2. ..\venv\Scripts\activate
3. pytest unit/test_storage.py::test_upsert_updates -v
4. See assertion failure at line 45

### Environment
- OS: Windows 11
- Python Version: 3.11.11
- BiGRAG Version: commit cf777e9
- Pytest Version: 8.3.4

### Root Cause Analysis
JsonKVStorage.upsert() always calls insert() instead of checking if key exists.

### Suggested Fix
Update bigrag/storage.py line 28-32:
```python
async def upsert(self, id, data):
    if id in self._data:
        self._data[id].update(data)  # Update existing
    else:
        self._data[id] = data  # Insert new
```

### Related Files
- Source file: bigrag/storage.py:28
- Test file: tests/unit/test_storage.py:45

### Priority
HIGH (affects data integrity)

### Assigned To
Unassigned
```

---

## 9. Quick Reference Card

### 9.1 Common Test Commands Cheat Sheet

```cmd
# Run single test file
pytest tests/unit/test_utils.py -v --tb=short

# Run specific test function
pytest tests/unit/test_utils.py::test_compute_mdhash_id -v

# Run all tests in a category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run with coverage
pytest --cov=bigrag --cov-report=html

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run only failed tests from last run
pytest --lf

# Show 10 slowest tests
pytest --durations=10
```

### 9.2 Troubleshooting Quick Guide

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -e ..` from tests directory |
| API tests timeout | Start backend: `cd backend && python server.py` |
| Out of memory | Run tests in smaller batches, disable `-n auto` |
| Encoding errors (Windows) | Set `PYTHONIOENCODING=utf-8` |
| Tests hang | Check for background processes, restart venv |
| Slow tests | Use `pytest -m "not slow"` to skip performance tests |

---

## 10. Test Metrics

Track these metrics during testing:

| Metric | Formula | Target |
|--------|---------|--------|
| **Pass Rate** | (Passed / Total) * 100 | >= 95% |
| **Code Coverage** | Lines covered / Total lines | >= 80% |
| **Defect Density** | Bugs found / Total tests | <= 5% |
| **Execution Time** | Total time for all tests | <= 30 min |

---

## 11. Known Limitations

### 11.1 Test Limitations
- LLM extraction tests require OpenAI API key (can be skipped)
- Frontend tests require UI to be implemented
- Performance tests may take 5-10 minutes
- Some tests require internet connection (for embedding models)

### 11.2 Platform Limitations
- Windows-only testing (Linux/Mac not validated in this plan)
- Single-machine testing (no distributed/multi-node tests)

---

## 12. Regression Test Coverage

Tests for previously fixed bugs:

| Bug ID | Description | Test Function | File | Date Fixed |
|--------|-------------|---------------|------|------------|
| Bug #1 | Wrong hash prefix (edge deletion) | `test_bug1_edge_deletion_prefix` | `test_bug_fixes.py` | Pre-2025 |
| Bug #2 | drop() deletes all documents | `test_bug2_single_document_deletion` | `test_bug_fixes.py` | Pre-2025 |
| Bug #3 | Undefined load_env_file() | `test_bug3_reload_config` | `test_bug_fixes.py` | Pre-2025 |
| Bug #4 | KeyError on missing dict keys | `test_bug4_defensive_dict_access` | `test_bug_fixes.py` | Pre-2025 |
| Bug #5 | upsert() doesn't update | `test_bug5_upsert_updates` | `test_bug_fixes.py` | Pre-2025 |
| Bug #6 | Wrong type annotations | `test_bug6_type_annotations` | `test_bug_fixes.py` | Pre-2025 |
| Bug #7 | Dict unpacking in reranker | Covered by all integration tests | `operate.py:971-979` | 2025-01-09 |
| Bug #8 | Test used wrong attribute name | Test-level fix (not core bug) | 4 integration test files | 2025-01-09 |

### Bug #7 Details (CRITICAL - Fixed 2025-01-09)
**Location:** [bigrag/operate.py:971-979](bigrag/operate.py#L971-L979)
**Impact:** 24/30 integration tests failing
**Root Cause:** Reranker returns list of dicts, but code unpacked as tuple
**Fix:** Changed from `for content, sources, score in reranked` to `for item in reranked` with dict access

### Bug #8 Details (Test-Level - Fixed 2025-01-09)
**Location:** 4 integration test files
**Impact:** 4/30 integration tests failing after Bug #7 fix
**Root Cause:** Tests referenced non-existent `rag.knowledge_graph_inst` attribute
**Fix:** Updated to use correct attribute `rag.chunk_entity_relation_graph`

---

## 13. Continuous Testing

### 13.1 Pre-Commit Checklist
Before committing code:
- [ ] Run critical tests: `pytest -m critical`
- [ ] Run affected unit tests
- [ ] Verify no new failures

### 13.2 Pre-Release Checklist
Before release:
- [ ] Run full test suite: `pytest`
- [ ] Verify coverage >= 80%
- [ ] All critical/high bugs resolved
- [ ] Performance tests pass
- [ ] API tests pass
- [ ] Regression tests pass

---

## 14. Test Maintenance

### 14.1 When to Update Tests
- After fixing a bug (add regression test)
- After adding a feature (add corresponding tests)
- When test becomes flaky (fix or remove)
- When requirements change (update assertions)

### 14.2 Test Review Schedule
- **Weekly:** Review failed tests
- **Monthly:** Review test coverage
- **Quarterly:** Refactor outdated tests

---

## 15. Appendix

### 15.1 Useful Commands

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

### 15.2 Troubleshooting

**Issue:** Tests fail with import errors
**Solution:** Ensure bigrag is installed: `pip install -e ..`

**Issue:** API tests timeout
**Solution:** Start backend server first: `cd backend && python server.py`

**Issue:** Out of memory errors
**Solution:** Run tests in smaller batches or disable parallel execution

**Issue:** Windows encoding errors
**Solution:** Set `PYTHONIOENCODING=utf-8` before running tests

### 15.3 Phase 3 Integration Test Implementation (2025-01-09)

**Overview:** Expanded 7 skeleton integration tests to 30 comprehensive tests across 4 files.

**Implementation Details:**

1. **test_entity_extraction.py** (6 tests)
   - `test_entity_normalization_after_extraction` - Validates TYPE_NORMALIZATION_MAP usage
   - `test_entity_type_normalized_correctly` - Checks entity type consistency
   - `test_relation_extraction_quality` - Validates relation completeness
   - `test_llm_cache_reduces_api_calls` - Verifies LLM response caching
   - `test_extraction_with_metadata_context` - Tests metadata-enhanced extraction
   - `test_multiple_entities_in_single_doc` - Multi-entity handling

2. **test_graph_vector_sync.py** (6 tests)
   - `test_entity_in_both_graph_and_vector` - Entity exists in both storage layers
   - `test_edge_deletion_removes_from_vector` - Cascade deletion validation
   - `test_chunk_embedding_matches_content` - Embedding consistency check
   - `test_entity_embedding_consistency` - Entity vector validation
   - `test_relation_embedding_consistency` - Relation vector validation
   - `test_vector_search_returns_graph_nodes` - Search integration test

3. **test_retrieval_pipeline.py** (10 tests)
   - `test_entity_extraction_to_retrieval` - Path A validation
   - `test_relation_extraction_to_retrieval` - Path B validation
   - `test_chunk_based_retrieval_path_c` - Path C validation
   - `test_hybrid_mode_combines_all_paths` - Three-path combination
   - `test_rrf_scoring_across_paths` - RRF algorithm validation
   - `test_reranking_improves_results` - Semantic reranking test
   - `test_metadata_preserved_in_results` - Metadata flow validation
   - `test_query_with_no_results_graceful` - Edge case handling
   - `test_multi_document_retrieval` - Multi-doc retrieval
   - `test_weighted_retrieval_ranking` - Weight-based ranking

4. **test_storage_consistency.py** (8 tests)
   - `test_insert_syncs_all_layers` - Insert synchronization
   - `test_delete_syncs_all_layers` - Delete synchronization
   - `test_upsert_maintains_consistency` - Upsert validation
   - `test_concurrent_operations_stay_synced` - Concurrency test
   - `test_entity_count_matches_across_layers` - Count consistency
   - `test_chunk_to_entity_mapping_consistency` - Mapping validation
   - `test_metadata_sync_across_storage` - Metadata sync
   - `test_vector_count_matches_graph_nodes` - Vector-graph alignment

**Bugs Discovered:**
- **Bug #7 (CRITICAL):** Dict unpacking error in operate.py:971-979 (affected 24/30 tests)
- **Bug #8 (Test-Level):** Wrong attribute name in 4 test files (affected 4/30 tests)

**Result:** After fixes, all 30 integration tests pass (100%)

---

## 16. Sign-Off

Test plan approved by: _________________
Date: _________________

Test execution completed by: _________________
Date: _________________

All critical tests passed: [ ] Yes [ ] No
Code coverage achieved: _____%
Total bugs found: _____
Release approved: [ ] Yes [ ] No
