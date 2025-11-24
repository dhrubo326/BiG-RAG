# BiG-RAG Test Execution Guide

**Complete guide for running all test phases in a new session**

---

## Quick Start (TL;DR)

**Note**: Phases 1, 2, and 3 are already complete ✅. Focus on Phase 4 (API Tests).

```bash
# 0. Install REQUIRED dependencies (NEW SESSION)
pip install pytest-asyncio httpx

# 1. Create demo dataset (first time only)
mkdir -p datasets/demo_test/raw
# Add 5 sample documents to datasets/demo_test/raw/corpus.jsonl (see Phase 4 below)
python script_build.py --data_source demo_test

# 2. Start backend server (Terminal 1)
cd backend && python server.py --data_source demo_test

# 3. Run Phase 4 API tests (Terminal 2)
cd tests && pytest api/ -v --tb=short
```

---

## Complete Test Execution Workflow

### Phase 1: Critical Path Tests (19 tests)

**Purpose**: Validate core BiGRAG functionality
**Duration**: ~30 seconds
**Required**: Must pass 100%

```bash
# Run critical path tests
pytest tests/critical/ -v --tb=short

# Expected: 19/19 PASSED
```

**Tests Included:**
- Basic initialization
- Document insertion (sync/async)
- Simple queries
- Document deletion
- Config loading

---

### Phase 2: Unit Tests (157 tests)

**Purpose**: Test individual components in isolation
**Duration**: ~2-3 minutes
**Required**: ≥95% pass rate

```bash
# Run all unit tests
pytest tests/unit/ -v --tb=short

# Expected: 157/157 PASSED
```

**Test Categories:**
- **test_chunking.py** (40 tests): Text chunking, metadata preservation
- **test_graph_building.py** (30 tests): Node creation, weight aggregation
- **test_embedding.py** (35 tests): Embedding preparation, batch processing
- **test_retrieval.py** (40 tests): RRF scoring, three-path retrieval
- **test_utils.py, test_storage.py, test_config.py, etc.** (12 tests)

---

### Phase 3: Integration Tests (30 tests)

**Purpose**: Test component interactions
**Duration**: ~5-10 minutes (includes LLM API calls)
**Required**: ≥90% pass rate

#### Prerequisites
```bash
# Set OpenAI API key (required for entity extraction tests)
export OPENAI_API_KEY=your-key-here
# OR
echo "your-key-here" > ../openai_api_key.txt
```

#### Run Tests
```bash
# Run all integration tests
pytest tests/integration/ -v --tb=short

# Expected: 30/30 PASSED
```

**Test Files:**
1. **test_entity_extraction.py** (6 tests) - Requires OpenAI API key
   - Entity normalization, type mapping, relation quality

2. **test_graph_vector_sync.py** (6 tests)
   - Graph-vector synchronization, embedding consistency

3. **test_retrieval_pipeline.py** (10 tests)
   - Three-path retrieval, hybrid mode, RRF, reranking

4. **test_storage_consistency.py** (8 tests)
   - Insert/delete/upsert sync, concurrent operations

---

### Phase 4: API Tests (40 tests) - **CURRENT PHASE**

**Status**: Phases 1, 2, and 3 are already complete ✅
**Purpose**: Test backend API endpoints
**Duration**: ~2-3 minutes
**Required**: ≥90% pass rate

---

#### Step 1: Create Demo Test Dataset (First Time Only)

**Option A: Quick Demo with Sample Files**
```bash
# Create demo dataset directory
mkdir -p datasets/demo_test/raw

# Create sample corpus file
cat > datasets/demo_test/raw/corpus.jsonl << 'EOF'
{"id": "doc_001", "contents": "Lionel Messi is an Argentine professional footballer who plays as a forward for Inter Miami in Major League Soccer. He won the FIFA World Cup 2022 with Argentina.", "title": "Messi Career"}
{"id": "doc_002", "contents": "Cristiano Ronaldo is a Portuguese footballer who plays for Al Nassr. He has won five Ballon d'Or awards and is considered one of the greatest players of all time.", "title": "Ronaldo Career"}
{"id": "doc_003", "contents": "Virat Kohli is an Indian cricketer and former captain of the Indian national team. He is regarded as one of the best batsmen in cricket history.", "title": "Kohli Career"}
{"id": "doc_004", "contents": "Sachin Tendulkar is a retired Indian cricketer widely regarded as one of the greatest batsmen ever. He scored 100 international centuries during his career.", "title": "Tendulkar Legacy"}
{"id": "doc_005", "contents": "The FIFA World Cup 2022 was held in Qatar. Argentina won the tournament, defeating France in the final on penalties.", "title": "World Cup 2022"}
EOF

# Build knowledge graph
cd ..
python script_build.py --data_source demo_test
```

**Option B: Use Existing Sample Files**
```bash
# If you have football_news.txt and cricket_news.txt
# Create corpus from those files (convert to JSONL format manually)
# Then run: python script_build.py --data_source demo_test
```

**Expected Output:**
```
expr/demo_test/
├── kv_store_full_docs.json
├── kv_store_text_chunks.json
├── vdb_entities.json
├── vdb_relations.json
├── vdb_chunks.json
└── graph_chunk_entity_relation.graphml
```

---

#### Step 2: Install Required Packages (NEW SESSION)
```bash
# CRITICAL: pytest-asyncio is REQUIRED for all async tests
pip install pytest-asyncio

# httpx is required for API client
pip install httpx
```

**Why pytest-asyncio?**
- All API tests use async fixtures (`api_client`)
- pytest.ini has `asyncio_mode = auto` which requires pytest-asyncio
- Without it, you'll get: "PytestRemovedIn9Warning: requested async fixture with no plugin"

---

#### Step 3: Start Backend Server (REQUIRED)
```bash
# Terminal 1: Start server
cd backend
python server.py --data_source demo_test

# Wait for: "Server started on http://0.0.0.0:8001"
# Server will load the demo_test knowledge graph
```

---

#### Step 4: Run API Tests (in new terminal)
```bash
# Terminal 2: Run tests
cd tests
pytest api/ -v --tb=short

# Expected: 36-40/40 PASSED (some may skip if endpoints not implemented)
```

---

#### Test Coverage Details

**test_server_endpoints.py** (10 tests):
- ✅ Root endpoint (/) - API info, features, providers
- ✅ Health endpoint (/health) - Status, uptime, RAG instances, job queue
- ✅ API documentation (/docs, /redoc)
- ✅ Error handling (404 on invalid endpoints)
- ✅ Uptime monitoring (increases over time)

**test_search_api.py** (15 tests):
- ✅ Basic search (/search) - Single and batch queries
- ✅ Ask endpoint (/ask) - Basic Q&A
- ✅ All retrieval modes (hybrid, local, global, naive)
- ✅ Semantic reranking (enable/disable)
- ✅ Parameter validation (top_k, mode, question)
- ✅ Error cases (invalid mode, missing fields, malformed JSON)
- ✅ Response format validation
- ✅ Edge cases (empty queries, large top_k)

**test_graph_api.py** (15 tests):
- ✅ Graph statistics (/graph/stats) - Counts and metrics
- ✅ Graph export (/graph/export) - Cytoscape format
- ✅ Export parameters (limit, node_types, min_weight, sampling strategies)
- ✅ Subgraph operations (neighbors, search)
- ✅ Parameter validation (missing data_source, depth, limit)
- ✅ Filter validation (node types, weight thresholds)

---

#### Step 5: Stop Server After Tests
```bash
# Linux/macOS
fuser -k 8001/tcp

# Windows
netstat -ano | findstr :8001
taskkill /PID <pid> /F
```

---

#### Troubleshooting

**Issue**: All 40 API tests failing with "PytestRemovedIn9Warning: requested async fixture"
**Cause**: pytest-asyncio package not installed ⚠️ **MOST COMMON**
**Solution**:
```bash
pip install pytest-asyncio
```

**Issue**: All tests skipped
**Cause**: Backend server not running
**Solution**: Start server in Terminal 1

**Issue**: Connection refused
**Cause**: Server on different port
**Solution**:
```bash
export API_BASE_URL=http://localhost:YOUR_PORT
pytest api/ -v
```

**Issue**: Some graph tests failing
**Cause**: demo_test dataset too small
**Solution**: Add more documents to corpus.jsonl or use a larger dataset

**Issue**: httpx import error
**Cause**: httpx package not installed
**Solution**: `pip install httpx`

---

### Phase 5: End-to-End Tests (Coming Soon)

**Purpose**: Test complete workflows
**Files**: `tests/e2e/test_*.py`

```bash
pytest tests/e2e/ -v --tb=short
```

---

### Phase 6: Regression Tests

**Purpose**: Validate previously fixed bugs
**Duration**: ~30 seconds

```bash
pytest tests/regression/ -v --tb=short
```

---

## Running Specific Test Subsets

### By Marker
```bash
# Critical tests only
pytest -m critical -v

# Unit tests only
pytest -m unit -v

# Integration tests only
pytest -m integration -v

# API tests only
pytest -m api -v

# LLM tests only (requires OpenAI key)
pytest -m llm -v
```

### By File
```bash
# Single file
pytest tests/unit/test_chunking.py -v

# Single test function
pytest tests/unit/test_chunking.py::TestChunking::test_basic_chunking -v
```

### By Directory
```bash
# All tests in a directory
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/api/ -v
```

---

## Environment Setup

### Required Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# CRITICAL for all async tests (Phases 3 & 4)
pip install pytest-asyncio

# Required for API tests (Phase 4)
pip install httpx

# Optional: For reranking tests
pip install sentence-transformers
```

### Environment Variables
```bash
# OpenAI API key (for LLM tests)
export OPENAI_API_KEY=your-key-here

# API base URL (for API tests)
export API_BASE_URL=http://localhost:8001

# Skip tests
export SKIP_API_TESTS=true  # Skip API tests if server not running
export SKIP_FRONTEND=true   # Skip frontend tests
```

---

## Complete Test Suite (All Phases)

### Run Everything
```bash
# Without API tests (no server needed)
pytest tests/ -v --tb=short -m "not api"

# With API tests (server required)
# Terminal 1:
cd backend && python server.py --data_source SingleTopic

# Terminal 2:
pytest tests/ -v --tb=short
```

### Expected Results
| Phase | Tests | Duration | Pass Rate |
|-------|-------|----------|-----------|
| Phase 1: Critical | 19 | 30s | 100% |
| Phase 2: Unit | 157 | 2-3min | ≥95% |
| Phase 3: Integration | 30 | 5-10min | ≥90% |
| Phase 4: API | 40 | 2-3min | ≥90% |
| **Total** | **246** | **~15min** | **≥95%** |

---

## Troubleshooting

### Issue: "No module named 'bigrag'"
**Solution**: Install package in development mode
```bash
pip install -e ..
```

### Issue: "OpenAI API key not set"
**Solution**: Set environment variable or skip LLM tests
```bash
export OPENAI_API_KEY=your-key-here
# OR
pytest -m "not llm" -v
```

### Issue: "Connection refused" (API tests)
**Solution**: Start backend server first
```bash
cd backend && python server.py --data_source demo_test
```

### Issue: "Test timeout"
**Solution**: Increase timeout in pytest.ini or skip slow tests
```bash
pytest -m "not slow" -v
```

### Issue: Tests failing on Windows
**Solution**: Check environment variables and paths
```bash
# Use forward slashes in paths
# Set PYTHONIOENCODING=utf-8
set PYTHONIOENCODING=utf-8
pytest tests/ -v
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: BiG-RAG Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt

    - name: Run Phase 1 (Critical)
      run: pytest tests/critical/ -v

    - name: Run Phase 2 (Unit)
      run: pytest tests/unit/ -v

    - name: Run Phase 3 (Integration)
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: pytest tests/integration/ -v

    - name: Create Demo Dataset
      run: |
        mkdir -p datasets/demo_test/raw
        # Add corpus.jsonl with sample data
        python script_build.py --data_source demo_test

    - name: Start Backend Server
      run: |
        cd backend
        python server.py --data_source demo_test &
        sleep 10

    - name: Run Phase 4 (API)
      run: pytest tests/api/ -v
```

---

## Coverage Report

### Generate Coverage
```bash
# Run with coverage
pytest tests/ --cov=bigrag --cov-report=html --cov-report=term

# View HTML report
# Open htmlcov/index.html in browser
```

### Target Coverage
- **Overall**: ≥80%
- **Core modules** (bigrag.py, operate.py): ≥90%
- **Utility modules**: ≥70%

---

## Test Maintenance

### Adding New Tests
1. Create test file in appropriate directory (`unit/`, `integration/`, `api/`)
2. Follow naming convention: `test_*.py`
3. Use appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
4. Add docstrings to test functions
5. Run locally before committing

### Updating Tests After Code Changes
1. Run full test suite: `pytest tests/ -v`
2. Fix failing tests or update assertions
3. Add regression tests for bug fixes
4. Update TEST_PLAN.md if test structure changes

---

## Quick Reference Card

**Note**: Phases 1-3 already complete ✅. Focus on Phase 4.

```bash
# ==============================================================================
# PHASE 4: API TESTS (CURRENT PHASE)
# ==============================================================================

# Step 0: Install dependencies (NEW SESSION - CRITICAL!)
pip install pytest-asyncio httpx

# Step 1: Create demo dataset (first time only)
mkdir -p datasets/demo_test/raw
# Add sample corpus.jsonl (see Phase 4 section for content)
python script_build.py --data_source demo_test

# Step 2: Start backend server (Terminal 1)
cd backend && python server.py --data_source demo_test

# Step 3: Run API tests (Terminal 2)
cd tests && pytest api/ -v --tb=short

# Step 4: Stop server
# Windows: netstat -ano | findstr :8001 then taskkill /PID <pid> /F
# Linux/macOS: fuser -k 8001/tcp

# ==============================================================================
# PREVIOUS PHASES (Already Complete ✅)
# ==============================================================================

# PHASE 1: Critical (19/19 PASSED)
# pytest tests/critical/ -v

# PHASE 2: Unit (157/157 PASSED)
# pytest tests/unit/ -v

# PHASE 3: Integration (30/30 PASSED)
# pytest tests/integration/ -v
```

---

## Support

- **Test Plan**: See [TEST_PLAN.md](TEST_PLAN.md)
- **Issues**: Check GitHub issues or create new one
- **Documentation**: See [../docs/](../docs/)
