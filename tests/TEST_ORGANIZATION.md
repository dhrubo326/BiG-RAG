# BiG-RAG Test Organization Guide

## Overview

All test scripts have been consolidated into a single `tests/` directory with clear organization by purpose.

## Directory Structure

```
tests/
├── __init__.py                    # Python package marker
├── README.md                      # Test documentation
├── TEST_ORGANIZATION.md          # This file
│
├── data_preparation/              # Data conversion and preparation scripts
│   ├── build_kg_from_corpus.py   # Build knowledge graph from corpus
│   ├── convert_singletopic_docs.py # Convert SingleTopic documents
│   ├── convert_text_to_corpus.py # Convert text files to corpus format
│   └── prepare_singletopic_questions.py # Prepare questions for SingleTopic
│
├── evaluation/                    # Evaluation and validation scripts
│   ├── eval.py                   # General evaluation script
│   ├── run_singletopic_evaluation.py # SingleTopic evaluation runner
│   └── validate_singletopic_dataset.py # Dataset validation
│
├── integration/                   # Integration and API tests
│   ├── test_all_retrieval_modes.py # Test all retrieval modes
│   ├── test_api_detailed.py      # Detailed API tests
│   ├── test_api_simple.py        # Simple API tests
│   ├── test_chunk_retrieval_debug.py # Debug chunk retrieval
│   ├── test_multi_doc_query.py   # Test multi-document queries
│   └── test_retrieval_demo.py    # Retrieval demonstration
│
└── (root level)                   # Core unit tests
    ├── check_ready.py             # System readiness check
    ├── test_ask_question.py       # Test question answering
    ├── test_build_graph.py        # Test graph building
    ├── test_end_to_end.py         # End-to-end tests
    ├── test_improvements.py       # Test recent improvements
    ├── test_retrieval.py          # Basic retrieval tests
    └── test_setup.py              # Setup verification
```

## Running Tests

### Quick System Check
```bash
cd tests
python check_ready.py
python test_setup.py
```

### Test Recent Features
```bash
cd tests
python test_improvements.py
```

### Run Integration Tests
```bash
cd tests/integration
python test_api_simple.py
python test_all_retrieval_modes.py
```

### Data Preparation
```bash
cd tests/data_preparation
python convert_text_to_corpus.py --input data.txt --output corpus.jsonl
python build_kg_from_corpus.py --data_source my_dataset
```

### Evaluation
```bash
cd tests/evaluation
python run_singletopic_evaluation.py
python validate_singletopic_dataset.py
```

## Migration Notes

The following directories have been removed and their contents consolidated:
- `scripts/` → Moved to `tests/data_preparation/`
- `test_scripts/` → Moved to appropriate subdirectories in `tests/`
- Duplicate and outdated scripts have been removed

## Test Categories

### Unit Tests (root level)
Basic functionality tests for core BiG-RAG features:
- Graph building
- Retrieval operations
- Question answering
- Setup verification

### Integration Tests
Tests that verify API endpoints and multi-component interactions:
- API endpoint testing
- Retrieval mode testing
- Multi-document queries

### Data Preparation
Utility scripts for preparing data:
- Converting documents to corpus format
- Building knowledge graphs
- Preparing evaluation questions

### Evaluation
Scripts for evaluating model performance:
- Dataset validation
- Evaluation metrics computation
- Performance benchmarking

## Best Practices

1. **Run check_ready.py first** to verify system setup
2. **Use data_preparation scripts** to prepare your data
3. **Run integration tests** after starting the API server
4. **Use evaluation scripts** to measure performance

## Dependencies

Most tests require:
- BiG-RAG installed (`pip install -r requirements.txt`)
- OpenAI API key configured
- Some tests require the API server running (`cd backend && python server.py`)

## Adding New Tests

- Place unit tests in `tests/` root
- Place API/integration tests in `tests/integration/`
- Place data scripts in `tests/data_preparation/`
- Place evaluation scripts in `tests/evaluation/`