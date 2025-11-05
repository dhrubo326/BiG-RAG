# BiG-RAG Test Suite

This directory contains test files for verifying BiG-RAG functionality.

## Test Files

### 1. `test_setup.py`
**Purpose**: Verify environment setup and dependencies

**Tests**:
- Python version compatibility
- Required packages installation
- CUDA availability (if applicable)
- OpenAI API key configuration

**Usage**:
```bash
python tests/test_setup.py
```

### 2. `test_build_graph.py`
**Purpose**: Test bipartite graph construction from corpus

**Tests**:
- Document chunking
- Entity extraction (GPT-4o-mini)
- N-ary relation extraction
- Bipartite edge creation
- Storage file generation
- Vector database creation (NanoVectorDB)

**Prerequisites**:
- OpenAI API key in `openai_api_key.txt`
- Test corpus in `datasets/test_wiki/raw/corpus.jsonl`

**Usage**:
```bash
python tests/test_build_graph.py
```

**Expected Output**:
- `expr/test_wiki/kv_store_full_docs.json`
- `expr/test_wiki/kv_store_text_chunks.json`
- `expr/test_wiki/vdb_entities.json`
- `expr/test_wiki/vdb_bipartite_edges.json`
- `expr/test_wiki/vdb_chunks.json`
- `expr/test_wiki/graph_chunk_entity_relation.graphml`

### 3. `test_retrieval.py`
**Purpose**: Test vector search and knowledge retrieval

**Tests**:
- Entity-based retrieval
- Relation-based retrieval
- Hybrid retrieval (entity + relation)
- Top-k ranking
- Context formatting

**Prerequisites**:
- Pre-built bipartite graph (from `test_build_graph.py`)

**Usage**:
```bash
python tests/test_retrieval.py
```

**Test Queries**:
1. "What is the capital of France?"
2. "Who directed Nosferatu?"
3. "What is the relationship between Paris and France?"
... (10 total queries)

### 4. `test_end_to_end.py`
**Purpose**: Test complete BiG-RAG pipeline with LLM generation

**Tests**:
- Query → Retrieval → LLM Answer generation
- Tool-augmented generation cycle
- Answer quality verification
- Multi-hop reasoning

**Prerequisites**:
- Pre-built bipartite graph
- OpenAI API key

**Usage**:
```bash
python tests/test_end_to_end.py
```

---

## Running All Tests

```bash
# Run sequentially
python tests/test_setup.py
python tests/test_build_graph.py
python tests/test_retrieval.py
python tests/test_end_to_end.py
```

---

## Test Data

Test data is located in:
- `datasets/test_wiki/raw/corpus.jsonl` - Small Wikipedia corpus (5 documents)
- `datasets/test_wiki/raw/qa_test.json` - Test questions

To create custom test data, see [docs/DATASET_AND_CORPUS_GUIDE.md](../docs/DATASET_AND_CORPUS_GUIDE.md).

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'bigrag'`
- **Fix**: Run tests from project root: `python tests/test_*.py`

**Issue**: `FileNotFoundError: openai_api_key.txt`
- **Fix**: Create file with your OpenAI API key in project root

**Issue**: `AssertionError` in retrieval tests
- **Fix**: Ensure `test_build_graph.py` completed successfully first

**Issue**: Low success rate in end-to-end tests
- **Fix**: Check OpenAI API quota, verify graph quality, review test queries
# BiG-RAG Test Scripts

Test and validation scripts for BiG-RAG functionality.

---

## 🧪 Test Categories

### Retrieval Tests

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_all_retrieval_modes.py` | Test all 4 retrieval modes (local, global, hybrid, naive) | `python test_all_retrieval_modes.py` |
| `test_chunk_retrieval_debug.py` | Debug chunk-based retrieval (Path C) | `python test_chunk_retrieval_debug.py` |
| `test_multi_doc_query.py` | Test queries across multiple documents | `python test_multi_doc_query.py` |
| `test_retrieval_demo.py` | Demo retrieval with example queries | `python test_retrieval_demo.py` |

### API Tests

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_api_simple.py` | Simple API endpoint tests | `python test_api_simple.py` |
| `test_api_detailed.py` | Detailed API functionality tests | `python test_api_detailed.py` |

### Feature Tests

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_improvements.py` | Test Phase 2-4 improvements (metadata, deletion, reranking) | `python test_improvements.py` |

### Dataset Validation

| Script | Purpose | Usage |
|--------|---------|-------|
| `validate_singletopic_dataset.py` | Validate SingleTopic dataset structure | `python validate_singletopic_dataset.py` |

### Evaluation Runners

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_singletopic_evaluation.py` | Complete SingleTopic evaluation pipeline | `python run_singletopic_evaluation.py` |

---

## 🔧 Legacy/Utility Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `build_kg_from_corpus.py` | Build KG from corpus (old method) | ⚠️ Use `script_build.py` instead |
| `convert_text_to_corpus.py` | Convert text to corpus format | ⚠️ Legacy |
| `check_ready.py` | Check system readiness | ⚠️ Legacy |
| `eval.py` | Old evaluation script | ⚠️ Use evaluation API instead |

**Note**: Legacy scripts kept for reference. Use main framework scripts (`script_build.py`, `script_process.py`) instead.

---

## 🚀 Running Tests

### Prerequisites

```bash
# Ensure BiG-RAG is installed
pip install -e .

# Ensure backend is running (for API tests)
cd backend
python server.py --data_source SingleTopic
```

### Run All Retrieval Tests

```bash
cd test_scripts

# Test all modes
python test_all_retrieval_modes.py

# Test chunk retrieval specifically
python test_chunk_retrieval_debug.py

# Demo retrieval
python test_retrieval_demo.py
```

### Run API Tests

```bash
# Simple API test
python test_api_simple.py

# Detailed API test
python test_api_detailed.py
```

### Run Improvements Test

```bash
# Test all Phase 2-4 features
python test_improvements.py
```

### Validate Dataset

```bash
# Validate SingleTopic dataset
python validate_singletopic_dataset.py
```

### Run Complete Evaluation

```bash
# Full evaluation pipeline
python run_singletopic_evaluation.py
```

---

## 📊 Test Reports

Test results are documented in [`../docs/reports/`](../docs/reports/):

- [GRAPH_CONSTRUCTION_TEST_REPORT.md](../docs/reports/GRAPH_CONSTRUCTION_TEST_REPORT.md)
- [CHUNK_RETRIEVAL_ANALYSIS.md](../docs/reports/CHUNK_RETRIEVAL_ANALYSIS.md)
- [RETRIEVAL_VALIDATION_REPORT.md](../docs/reports/RETRIEVAL_VALIDATION_REPORT.md)
- [COMPREHENSIVE_QA_REPORT.md](../docs/reports/COMPREHENSIVE_QA_REPORT.md)
- [SINGLETOPIC_EVALUATION_DIAGNOSIS.md](../docs/reports/SINGLETOPIC_EVALUATION_DIAGNOSIS.md)

---

## 🐛 Debugging

### Common Issues

**Issue: ModuleNotFoundError: No module named 'bigrag'**
```bash
# Install BiG-RAG in development mode
pip install -e .
```

**Issue: API connection refused**
```bash
# Start backend server first
cd backend
python server.py --data_source SingleTopic
```

**Issue: Dataset not found**
```bash
# Check dataset exists
ls datasets/SingleTopic/

# If not, process dataset first
python script_process.py --data_source SingleTopic
python script_build.py --data_source SingleTopic
```

---

## 📝 Adding New Tests

1. Create test file: `test_your_feature.py`
2. Follow naming convention: `test_*.py`
3. Add docstring with purpose and usage
4. Update this README with entry
5. Run test and document results in `../docs/reports/`

Example template:
```python
"""
Test: [Feature Name]
Purpose: [What this tests]
Usage: python test_your_feature.py
"""

def test_your_feature():
    # Your test code
    pass

if __name__ == '__main__':
    test_your_feature()
```

---

Last Updated: November 5, 2025

---
