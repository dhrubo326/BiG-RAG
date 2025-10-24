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
- FAISS index creation

**Prerequisites**:
- OpenAI API key in `openai_api_key.txt`
- Test corpus in `datasets/test_wiki/raw/corpus.jsonl`

**Usage**:
```bash
python tests/test_build_graph.py
```

**Expected Output**:
- `expr/test_wiki/kv_store_entities.json`
- `expr/test_wiki/kv_store_bipartite_edges.json`
- `expr/test_wiki/kv_store_text_chunks.json`
- `expr/test_wiki/index_entity.bin`
- `expr/test_wiki/index_bipartite_edge.bin`
- `expr/test_wiki/index.bin`

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
