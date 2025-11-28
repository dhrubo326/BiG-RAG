# BiG-RAG: Bipartite Graph Retrieval-Augmented Generation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/dhrubo326/BiG-RAG)](https://github.com/dhrubo326/BiG-RAG/issues)
[![GitHub Stars](https://img.shields.io/github/stars/dhrubo326/BiG-RAG)](https://github.com/dhrubo326/BiG-RAG/stargazers)

**BiG-RAG** is an advanced RAG framework that uses bipartite graph structures to enhance knowledge retrieval and reasoning capabilities for large language models.

## 📑 Table of Contents

- [What is BiG-RAG?](#what-is-bigrag)
- [Quick Start](#quick-start)
  - [Step 0: Clone the Repository](#step-0-clone-the-repository)
  - [Step 1: Installation](#step-1-installation)
  - [Step 2: Quick Test with Demo Dataset](#step-2-quick-test-with-demo-dataset)
- [Building Your Own Knowledge Graph](#building-your-own-knowledge-graph)
- [Using BiG-RAG in Your Code](#using-bigrag-in-your-code)
- [Project Structure](#-project-structure)
- [Testing BiG-RAG](#testing-bigrag)
- [Retrieval Modes](#retrieval-modes)
- [Storage Backends](#storage-backends)
- [Recent Improvements](#-recent-improvements-january-2025)
- [Advanced Features](#advanced-features)
- [System Requirements](#system-requirements)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Documentation](#-documentation)
- [Contributing](#contributing)
- [Support & Community](#support--community)
- [Acknowledgments](#acknowledgments)

---

## What is BiG-RAG?

BiG-RAG constructs a **bipartite knowledge graph** using **n-ary relation extraction** from your documents. This graph-based approach enables more sophisticated multi-hop reasoning compared to traditional vector-only RAG systems.

**Key Features:**
- **Bipartite Graph Structure**: Documents ↔ Entities ↔ Relations for enhanced knowledge representation
- **Three-Path Retrieval** ⭐: Entity-based (Path A) + Relation-based (Path B) + Chunk-based (Path C) for +15-25% recall improvement
- **Semantic Reranking** ⭐: Cross-encoder reranking for +10-20% precision improvement
- **Cascade Document Deletion** ⭐: Smart deletion with shared entity preservation (~1-2s, no rebuild needed)
- **Metadata Preservation** ⭐: Document metadata flows through extraction for +2-3 F1 improvement
- **React Web UI**: Modern React 19 + TypeScript interface with graph visualization
- **Multiple Storage Backends**: Support for Milvus, ChromaDB, Neo4J, MongoDB, Oracle, TiDB
- **Flexible Retrieval Modes**: Hybrid, local (entity-based), global (relation-based), naive (text-only)
- **OpenAI Integration**: Ready-to-use with GPT models for testing and development
- **RL Training Framework**: GRPO, PPO, REINFORCE++ for training LLMs with graph-based retrieval
- **Async-First Design**: Efficient concurrent processing for large-scale applications

---

## Quick Start

### Step 0: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/dhrubo326/BiG-RAG.git
cd BiG-RAG
```

### Step 1: Installation

#### Using Python venv (Recommended)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install PyTorch (CPU version)
pip install torch torchvision torchaudio
# For GPU support, see: https://pytorch.org/get-started/locally/

# 5. Install BiG-RAG dependencies
pip install -r requirements.txt

# 6. Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

> **Note:** For advanced setup (RL training with GPUs), see [CLAUDE.md](CLAUDE.md) for conda environment setup.

### Step 2: Quick Test with Demo Dataset

BiG-RAG includes a pre-built demo dataset (`demo_test`) for immediate testing:

```bash
# Start the backend API server (unified mode)
cd backend
python server.py --unified

# The server will run on http://localhost:8001
# Visit http://localhost:8001/docs to see the API documentation
```

**Test the API:**
```bash
# In another terminal, test retrieval
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about the demo dataset", "mode": "hybrid"}'
```

**Or use the Web UI:**
```bash
# Terminal 1: Backend (already running from above)
cd backend && python server.py --unified

# Terminal 2: Frontend
cd frontend
npm install  # First time only
npm run dev

# Open http://localhost:5173 in your browser
```

---

## Building Your Own Knowledge Graph

After testing with the demo dataset, you can build a knowledge graph from your own documents:

### Step 3: Prepare Your Data

Create a corpus file (`corpus.jsonl`) with your documents:

```json
{"id": "doc_001", "contents": "Your document text here...", "title": "Document Title"}
{"id": "doc_002", "contents": "Another document...", "title": "Another Title"}
```

Place it in: `datasets/your_dataset/raw/corpus.jsonl`

### Step 4: Build Knowledge Graph

Set your OpenAI API key:
```bash
echo "your-api-key-here" > openai_api_key.txt
```

Build the bipartite graph:

**Standard Pipeline (Fast, General-Purpose):**
```bash
python script_build.py --data_source your_dataset
```

**Production Pipeline (Higher Accuracy for Educational Content):**
```bash
python script_build.py --data_source your_dataset --production
```

**Pipeline Comparison:**

| Feature | Standard Pipeline | Production Pipeline |
|---------|-------------------|---------------------|
| **Chunking** | Token-based (1200 tokens) | Table-aware (extracts tables intact) |
| **Table Handling** | May split across chunks | Preserved as structured data |
| **Entity Extraction** | Single LLM pass | Dual mode (tables + paragraphs) |
| **Validation** | Basic orphan detection | Numeric targets (95-99% validation rate) |
| **Best For** | General documents, speed | Educational content with tables/lists |
| **Cost** | ~$0.01/doc | ~$0.16-0.40/doc |
| **Speed** | ~60s/doc | ~120-180s/doc |
| **F1 Improvement** | Baseline | +2-3 points |

**When to Use Production Pipeline:**
- Documents with tables, lists, or structured content (KUET admission info, course catalogs, etc.)
- Educational datasets where accuracy > speed
- When validation quality is critical

See [Graph_indexing_plan.md](Graph_indexing_plan.md) for detailed technical explanation.

Both pipelines will:
- Extract entities and relations from your documents using GPT-4o-mini
- Create bipartite graph structure
- Generate embeddings with FlagEmbedding
- Save to `expr/your_dataset/`

**Time estimate:** 2-4 hours for ~10K documents with standard pipeline, 4-8 hours with production pipeline (depends on corpus size and OpenAI API rate limits)

### Step 5: Start Server in Unified Mode

```bash
cd backend
python server.py --unified
```

The API server runs on `http://localhost:8001/docs`

**Note**: Unified mode enables multi-subgraph support and all advanced indexing features.

---

## Using BiG-RAG in Your Code

```python
from bigrag import BiGRAG, QueryParam

# Initialize with your dataset
rag = BiGRAG(working_dir="expr/your_dataset")

# Query the knowledge graph
result = rag.query(
    "Your question here",
    param=QueryParam(mode="hybrid", top_k=10)
)

print(result)
```

---

## 📁 Project Structure

```
BiG-RAG/
├── README.md                    # This file - Getting started guide
├── CLAUDE.md                    # Comprehensive system reference
├── DEVELOPMENT.md               # Development status and guides
│
├── backend/                     # FastAPI REST API server
│   ├── api/                    # API route modules
│   ├── server.py               # Main server entry point
│   └── README.md               # Backend API documentation
│
├── frontend/                    # React Web UI (Nov 2025)
│   ├── src/                    # React 19 + TypeScript + Tailwind v4
│   └── README.md               # Frontend setup and development
│
├── bigrag/                      # Core BiG-RAG Python library
│   ├── bigrag.py               # Main BiGRAG class
│   ├── operate.py              # Graph operations (build, query)
│   ├── reranker.py             # Semantic reranking
│   ├── storage.py              # Default storage implementations
│   └── kg/                     # Optional storage backends (Milvus, Neo4J, etc.)
│
├── verl/                        # RL training framework (Volcano Engine RL)
│   └── trainer/                # GRPO, PPO, REINFORCE++ implementations
│
├── agent/                       # Tool-based agent system
│   ├── llm_agent/              # Tool generation manager
│   └── tool/                   # Tool environment and implementations
│
├── evaluation/                  # Evaluation metrics and benchmarks
│   └── README.md               # Evaluation guide
│
├── inference/                   # Model inference and deployment
│   └── README.md               # Inference guide
│
├── tests/                       # Comprehensive test suite
│   ├── README.md               # Test documentation
│   ├── api/                    # API endpoint tests
│   ├── integration/            # Integration tests
│   ├── e2e/                    # End-to-end pipeline tests
│   └── performance/            # Performance benchmarks
│
├── docs/                        # Technical documentation
│   └── technical/              # Design specs, logging guides
│
├── datasets/                    # QA datasets and corpora
│   ├── demo_test/              # Pre-built demo dataset
│   ├── SingleTopic/            # Sample dataset
│   └── README.md               # Dataset format guide
│
├── expr/                        # Built knowledge graphs
│   └── [dataset_name]/         # Generated graph files per dataset
│
├── script_build.py             # Build knowledge graph from corpus
├── script_process.py           # Process raw datasets to parquet
└── setup.py                    # Python package setup
```

See [CLAUDE.md](CLAUDE.md) for comprehensive system reference and [DEVELOPMENT.md](DEVELOPMENT.md) for development guides.

---

## Testing BiG-RAG

We provide a comprehensive test suite organized by category:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/api/              # API endpoint tests
pytest tests/integration/      # Integration tests
pytest tests/unit/             # Unit tests
pytest tests/e2e/              # End-to-end tests

# Run with coverage
pytest tests/ --cov=bigrag --cov-report=html
```

See [`tests/README.md`](tests/README.md) for detailed test documentation and guides.

---

## Retrieval Modes

BiG-RAG supports multiple retrieval strategies with **Three-Path Architecture** ⭐:

- **`hybrid`** (default): Combines **Path A (entities) + Path B (relations) + Path C (chunks)** for best multi-hop reasoning
  - Returns 10 total context items: 5 structured (entities + relations) + 5 semantic chunks
  - Supports optional semantic reranking for improved precision
- **`local`**: Entity-focused retrieval (Path A only), faster but less comprehensive
- **`global`**: Relation-focused retrieval (Path B only), good for factual queries
- **`naive`**: Direct text chunk retrieval, baseline comparison

Example:
```python
from bigrag.base import QueryParam

# Hybrid mode with reranking (best for complex queries)
result = rag.query(query, param=QueryParam(mode="hybrid", top_k=10, enable_reranking=True))

# Hybrid mode without reranking (faster)
result = rag.query(query, param=QueryParam(mode="hybrid", top_k=10, enable_reranking=False))

# Local mode (fastest)
result = rag.query(query, param=QueryParam(mode="local", top_k=10))
```

---

## Storage Backends

BiG-RAG supports multiple storage backends:

**Default (In-Memory):**
- NetworkX for graph
- NanoVectorDB for vectors
- JSON files for metadata

**Enterprise (Optional):**
- **Vector DBs**: Milvus, ChromaDB, TiDB, Oracle
- **Graph DBs**: Neo4J, MongoDB, Oracle

To use external backends:
```python
from bigrag.kg.milvus_impl import MilvusVectorDBStorage

rag = BiGRAG(
    vector_db_storage_cls=MilvusVectorDBStorage,
    working_dir="expr/your_dataset"
)
```

---

## Dataset Structure

```
datasets/your_dataset/
├── raw/
│   └── corpus.jsonl          # Your documents (required)
└── processed/
    └── [auto-generated]       # Processed data
```

**Corpus Format:**
```json
{"id": "unique_id", "contents": "text content", "title": "optional title"}
```

For more details, see `datasets/README.md`

---

## ⭐ Recent Improvements (January 2025)

BiG-RAG has been significantly enhanced with major improvements and bug fixes:

### Phase 2: Critical Fixes
- **Metadata Preservation**: Document metadata (title, tags, category) now flows through chunking → entity extraction (+2-3 F1 improvement)
- **Cascade Document Deletion**: Full cascade cleanup across all storage layers with smart shared entity preservation

### Phase 3: Three-Path Retrieval + Reranking
- **Three-Path Architecture**: Path A (entities) + Path B (relations) + Path C (chunks) → 10 total context items (+15-25% recall, +10-20% precision)
- **Semantic Reranking**: Cross-encoder based reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` (+10-20% precision at ~50-100ms latency)

### Phase 4: Bug Fixes
Fixed 5 critical bugs:
1. Missing chunks_vdb indexing (Path C was broken)
2. API reading non-existent files (document stats showed 0 entities/edges)
3. Entity weights missing (all weights were 0)
4. Incomplete rebuild cleanup
5. Incomplete document deletion (only removed from corpus, not KG)

**All bugs are now fixed. System is production-ready.**

### November 2025: Infrastructure Enhancements

**Centralized Logging System** (November 10, 2025):
- **Component-separated logs**: Core (logs/bigrag-core/), Backend (logs/backend/), Frontend (browser console)
- **Log rotation**: Size-based (10MB) and time-based (daily) with automatic cleanup
- **Structured logging**: Optional JSON format for log aggregation tools (ELK, Splunk)
- **Multiple handlers**: Console, file, error-only streams
- **Production-ready**: ~350 lines across 9 files for comprehensive log management

**Frontend Logger** (TypeScript):
- Browser console logger with module-specific loggers (apiLogger, graphLogger, chatLogger)
- Environment-based log level configuration
- Structured format with timestamps and context

For detailed configuration and usage, see [docs/technical/LOGGING_GUIDE.md](docs/technical/LOGGING_GUIDE.md).

For complete technical details and development guides, see [CLAUDE.md](CLAUDE.md) and [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Advanced Features

BiG-RAG includes advanced components for research and production use:

### RL Training Framework
Train LLMs to actively query the knowledge graph during generation:
- **GRPO** (Group Relative Policy Optimization) - Recommended for starting
- **PPO** (Proximal Policy Optimization) - Standard RL algorithm
- **REINFORCE++** - Variance-reduced policy gradients

See [CLAUDE.md](CLAUDE.md) for RL training setup and configuration.

### Tool-Based Agent System
Agents that iteratively query the knowledge graph during reasoning:
- Multi-turn retrieval loops
- Tool call generation and execution
- Answer synthesis with retrieved context

### Evaluation & Benchmarks
Comprehensive metrics for RAG systems:
- Exact Match (EM)
- Token-level F1 score
- Semantic similarity (SimCSE)
- Multi-hop QA benchmarks (2WikiMultiHopQA, HotpotQA, Musique)

See [evaluation/README.md](evaluation/README.md) for details.

---

## System Requirements

### For Knowledge Graph Building & Retrieval

**Minimum:**
- Python 3.11+
- 4GB RAM
- CPU only (works for small datasets <1K documents)
- 5GB free disk space

**Recommended:**
- Python 3.11+
- 16GB+ RAM
- GPU with 8GB+ VRAM (faster embedding generation)
- SSD storage
- OpenAI API key (for entity extraction)

### For RL Training (Advanced)

**Minimum:**
- 4 x GPUs with 48GB VRAM each (for 3B parameter models)
- 32GB+ system RAM
- 100GB+ free disk space

**Recommended:**
- 8 x GPUs with 80GB VRAM each (for 7B+ parameter models)
- 64GB+ system RAM
- NVMe SSD storage
- Multi-node cluster (for large-scale training)

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'bigrag'**
```bash
# Solution: Install in development mode
pip install -e .
```

**2. Server fails to start: "Failed to load graph"**
```bash
# Solution: Check if knowledge graph files exist
ls expr/demo_test/

# If missing, rebuild the graph
python script_build.py --data_source demo_test
```

**3. Frontend shows "Network Error"**
```bash
# Solution: Ensure backend is running
curl http://localhost:8001/
# Should return: {"message": "BiG-RAG Unified API Server..."}

# If not running:
cd backend && python server.py --unified
```

**4. OpenAI API rate limit errors**
```bash
# Solution: Set your API key
echo "sk-your-api-key-here" > openai_api_key.txt

# Or use environment variable
export OPENAI_API_KEY="sk-your-api-key-here"
```

**5. Out of memory during graph building**
```bash
# Solution: Process smaller batches
# Edit script_build.py and reduce chunk_size or batch_size parameters
```

For more troubleshooting help, see [CLAUDE.md](CLAUDE.md) or open an issue on GitHub.

---

## FAQ

### Understanding Weight Values

BiG-RAG assigns weight values to entities and relations in the knowledge graph:

**Entity Weights:**
- Calculation: Sum of importance scores (0-100) across all occurrences
- Interpretation:
  - 400+: Very central entity (mentioned 4+ times with high scores)
  - 200-399: Important entity (2-3 mentions)
  - 100-199: Mentioned entity (1-2 mentions)
  - <100: Peripheral entity

**Relation Weights:**
- Calculation: Sum of completeness scores (0-10) across all occurrences
- Higher weight = more frequently mentioned + more complete information

**Why not normalized?** Un-normalized weights preserve frequency information, which is valuable for ranking and understanding graph centrality.

For detailed weight semantics and usage examples, see [CLAUDE.md - Weight Semantics](CLAUDE.md#weight-semantics).

---

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - AI assistant guidance and comprehensive system reference
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development status, implementation notes, and technical guides
- **[BIGRAG_UI_PLAN.md](BIGRAG_UI_PLAN.md)** - Frontend UI implementation plan
- **[backend/README.md](backend/README.md)** - Backend API documentation
- **[frontend/README.md](frontend/README.md)** - Frontend setup and development
- **[tests/README.md](tests/README.md)** - Testing documentation and guides

---

## License

See [LICENSE](LICENSE) file for details.

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository** on GitHub
2. **Clone your fork**: `git clone https://github.com/your-username/BiG-RAG.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** and add tests
5. **Run tests**: `pytest tests/`
6. **Commit**: `git commit -m "Add your feature"`
7. **Push**: `git push origin feature/your-feature-name`
8. **Open a Pull Request** on GitHub

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=bigrag --cov-report=html
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development guides.

---

## Support & Community

### Get Help

- **Issues**: [Report bugs or request features](https://github.com/dhrubo326/BiG-RAG/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/dhrubo326/BiG-RAG/discussions)
- **Documentation**: [CLAUDE.md](CLAUDE.md) for comprehensive reference

### Before Opening an Issue

1. Check [existing issues](https://github.com/dhrubo326/BiG-RAG/issues) to avoid duplicates
2. Review the [Troubleshooting](#troubleshooting) section
3. Provide:
   - Python version (`python --version`)
   - OS and version
   - Error messages or logs
   - Steps to reproduce the issue

---

## Acknowledgments

BiG-RAG builds upon excellent research and open-source projects:
- **[Agent-R1](https://github.com/0russwest0/Agent-R1)**: Tool-augmented RL training
- **[LightRAG](https://github.com/HKUDS/LightRAG)**: Lightweight graph RAG
- **[HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)**: Hippocampus-inspired RAG
- **[VERL](https://github.com/volcengine/verl)**: Volcano Engine RL Framework (Bytedance)

Thanks to the open-source community for foundational tools and frameworks!
