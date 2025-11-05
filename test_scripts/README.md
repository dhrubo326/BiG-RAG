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
