# BiG-RAG Testing Guide (OpenAI Models)

This guide provides instructions for testing BiG-RAG with OpenAI models (gpt-4o-mini and text-embedding-3-large).

## Overview

The test suite consists of three scripts:

1. **test_build_graph.py** - Build knowledge graph from demo dataset
2. **test_retrieval.py** - Test retrieval functionality
3. **test_end_to_end.py** - Test complete RAG pipeline (retrieval + LLM generation)

## Prerequisites

### 1. OpenAI API Key

Create a file named `openai_api_key.txt` in the project root with your OpenAI API key:

```bash
echo "sk-your-api-key-here" > openai_api_key.txt
```

### 2. Install Dependencies

#### Option A: Virtual Environment (Lightweight - No RL Training)

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install PyTorch
pip install torch torchvision torchaudio

# Install BiG-RAG dependencies
pip install -r requirements_graphrag_only.txt

# Install additional test dependencies
pip install openai tenacity
```

#### Option B: Full Conda Environment (Complete Setup)

```bash
# Create conda environment
conda create -n bigrag python==3.11.11
conda activate bigrag

# Install PyTorch with CUDA
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install BiG-RAG
pip3 install -e .
pip3 install -r requirements.txt
```

## Test Dataset

The demo dataset is already created in `datasets/demo_test/raw/`:

- **corpus.jsonl** - 10 documents about AI/ML topics
- **qa_test.json** - 10 test questions with expected answers

Topics covered:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Python for AI
- Computer Vision
- Neural Networks
- TensorFlow
- PyTorch
- Reinforcement Learning

## Running Tests

### Step 1: Build Knowledge Graph

Build the bipartite knowledge graph from the demo corpus:

```bash
python test_build_graph.py
```

**What it does:**
- Loads 10 AI/ML documents from `datasets/demo_test/raw/corpus.jsonl`
- Uses gpt-4o-mini to extract entities and relations
- Creates bipartite graph structure
- Generates embeddings with text-embedding-3-large (3072 dimensions)
- Saves graph to `expr/demo_test/`

**Expected output:**
```
expr/demo_test/
├── kv_store_text_chunks.json       # Text chunk metadata
├── kv_store_entities.json          # Extracted entities
└── kv_store_bipartite_edges.json   # Relations (bipartite edges)
```

**Time:** 3-8 minutes
**Cost:** ~$0.10-0.30 USD

**Logs:** Check `build_graph.log` for detailed execution logs

### Step 2: Test Retrieval

Test the knowledge graph query functionality:

```bash
python test_retrieval.py
```

**What it does:**
- Tests single query retrieval
- Compares all retrieval modes (hybrid, local, global, naive)
- Runs all 10 QA test questions
- Measures retrieval success rate and coherence scores

**Expected output:**
- Successful retrieval for all queries
- Coherence scores (0-1, higher is better)
- Comparison of retrieval modes

**Time:** 1-2 minutes
**Cost:** ~$0.05 USD

**Logs:** Check `test_retrieval.log` for results

### Step 3: End-to-End RAG Test

Test complete RAG pipeline with LLM answer generation:

```bash
python test_end_to_end.py
```

**What it does:**
- Retrieves context from BiGRAG for each question
- Uses gpt-4o-mini to synthesize answers
- Compares generated answers with expected answers
- Runs interactive demo with example questions

**Expected output:**
- Answer generation for all 10 questions
- Match rate against expected answers
- Interactive demo responses

**Time:** 2-4 minutes
**Cost:** ~$0.10-0.20 USD

**Logs:** Check `test_end_to_end.log` for detailed results

## Configuration

### Model Selection

All test scripts use OpenAI models by default:

```python
from bigrag import BiGRAG
from bigrag.llm import gpt_4o_mini_complete, openai_embedding

rag = BiGRAG(
    working_dir="expr/demo_test",
    llm_model_func=gpt_4o_mini_complete,      # Entity extraction
    embedding_func=openai_embedding,           # Embeddings (text-embedding-3-large)
    enable_llm_cache=True,                     # Cache responses to reduce costs
)
```

### Switching to gpt-4o (Higher Quality)

To use gpt-4o instead of gpt-4o-mini:

```python
from bigrag.llm import gpt_4o_complete

rag = BiGRAG(
    llm_model_func=gpt_4o_complete,  # Use gpt-4o
    # ... other parameters
)
```

**Note:** gpt-4o is ~15x more expensive but may provide better entity extraction quality.

### Embedding Model Options

#### text-embedding-3-small (Default in bigrag.llm)

```python
from bigrag.llm import openai_embedding

# Uses text-embedding-3-small (1536 dimensions)
rag = BiGRAG(
    embedding_func=openai_embedding,
    # ...
)
```

#### text-embedding-3-large (Used in Tests)

```python
from bigrag.openai_embedding import openai_embedding_large

# Uses text-embedding-3-large (3072 dimensions)
rag = BiGRAG(
    embedding_func=openai_embedding_large(),
    # ...
)
```

**Recommendation:** text-embedding-3-large provides better accuracy for complex domains.

### Chunking Parameters

Adjust chunking behavior:

```python
rag = BiGRAG(
    chunk_token_size=1200,           # Tokens per chunk
    chunk_overlap_token_size=100,    # Overlap between chunks
    # ...
)
```

### Entity Extraction Parameters

Control entity extraction quality vs. speed:

```python
rag = BiGRAG(
    entity_extract_max_gleaning=1,   # Number of extraction passes (1-3)
    entity_summary_to_max_tokens=500, # Max tokens per entity summary
    # ...
)
```

**Note:** Higher `max_gleaning` = better quality but slower and more expensive.

## Retrieval Modes

BiGRAG supports 4 retrieval modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **hybrid** | Entity + Relation paths (default) | Best overall performance |
| **local** | Entity-based retrieval only | Entity-focused queries |
| **global** | Relation-based retrieval only | Relationship queries |
| **naive** | Direct text chunk search | Baseline comparison |

Specify mode in query parameters:

```python
from bigrag import QueryParam

param = QueryParam(
    mode="hybrid",
    top_k=5,
    max_token_for_text_unit=4000,
)

results = rag.query("What is AI?", param=param)
```

## Output Files

After running tests, you'll have:

```
BiG-RAG/
├── expr/demo_test/                  # Knowledge graph
│   ├── kv_store_text_chunks.json
│   ├── kv_store_entities.json
│   └── kv_store_bipartite_edges.json
├── build_graph.log                  # Build logs
├── test_retrieval.log               # Retrieval test logs
└── test_end_to_end.log              # End-to-end test logs
```

## Troubleshooting

### Issue: OpenAI API Key Not Found

**Error:** `ValueError: OpenAI API key not found`

**Solution:** Ensure `openai_api_key.txt` exists in project root with valid API key.

### Issue: Rate Limit Errors

**Error:** `RateLimitError: Rate limit exceeded`

**Solution:** Scripts include automatic retry logic. If persistent:
- Add delays between batches
- Reduce `batch_size` in build script
- Upgrade OpenAI API tier

### Issue: No Results from Retrieval

**Problem:** Queries return empty results

**Solutions:**
1. Check that graph was built successfully (`expr/demo_test/` exists)
2. Verify embeddings were created (files should be non-empty)
3. Try different retrieval modes (hybrid, local, global)
4. Increase `top_k` parameter

### Issue: Import Errors

**Error:** `ModuleNotFoundError: No module named 'bigrag'`

**Solution:** Install BiG-RAG package:
```bash
pip install -e .
```

### Issue: Graph Build Fails

**Problem:** Entity extraction errors

**Solutions:**
1. Check API key is valid and has credits
2. Review `build_graph.log` for detailed error messages
3. Reduce batch size (edit `test_build_graph.py` line 60: `batch_size = 2`)
4. Enable verbose logging

## Cost Estimation

Based on demo dataset (10 documents):

| Operation | Model | Tokens | Cost |
|-----------|-------|--------|------|
| Entity Extraction | gpt-4o-mini | ~50,000 | $0.01 |
| Embeddings | text-embedding-3-large | ~10,000 | $0.001 |
| Retrieval Tests | text-embedding-3-large | ~5,000 | $0.0005 |
| Answer Generation | gpt-4o-mini | ~20,000 | $0.01 |
| **Total** | | ~85,000 | **~$0.02** |

**Note:** Actual costs may vary. LLM caching significantly reduces costs during development.

## Next Steps

After successful testing:

1. **Create API Server**: Use FastAPI to expose BiGRAG as REST API
2. **Scale to Larger Dataset**: Test with your own corpus
3. **Integrate with Applications**: Connect to chatbots, apps, etc.
4. **Optimize Performance**: Tune chunking, retrieval parameters
5. **Switch LLM Providers**: Try Gemini, Claude, or local models

## Architecture Benefits

### Modular Design

The test setup demonstrates BiG-RAG's modular architecture:

```python
# Easy to swap LLM providers
from bigrag.llm import gpt_4o_mini_complete  # OpenAI
# from bigrag.llm import ollama_model_complete  # Local
# from bigrag.llm import bedrock_complete  # AWS

# Easy to swap embedding models
from bigrag.llm import openai_embedding  # OpenAI
# from bigrag.openai_embedding import openai_embedding_large  # OpenAI Large
# ... or implement custom embedding function

rag = BiGRAG(
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
)
```

### Async-First

All operations use async/await for better performance:

```python
# Async query (preferred)
results = await rag.aquery(query, param)

# Sync wrapper (for compatibility)
results = rag.query(query, param)
```

### Storage Abstraction

Easy to switch storage backends without code changes:

```python
rag = BiGRAG(
    graph_storage="NetworkXStorage",     # Default: in-memory
    vector_storage="NanoVectorDBStorage", # Default: SQLite
    kv_storage="JsonKVStorage",          # Default: JSON files
)

# For production, switch to:
# graph_storage="Neo4JStorage"
# vector_storage="MilvusVectorDBStorage"
# kv_storage="MongoDBKVStorage"
```

## Support

For issues or questions:
- Check logs: `build_graph.log`, `test_retrieval.log`, `test_end_to_end.log`
- Review CLAUDE.md for detailed architecture documentation
- Open issue on GitHub with error logs

## References

- **CLAUDE.md**: Comprehensive project documentation
- **docs/DATASET_AND_CORPUS_GUIDE.md**: Creating custom datasets
- **bigrag/llm.py**: All available LLM and embedding functions
- **bigrag/base.py**: Storage abstraction interfaces
