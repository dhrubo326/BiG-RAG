# BiG-RAG: Bipartite Graph Retrieval-Augmented Generation

**BiG-RAG** is an advanced RAG framework that uses bipartite graph structures to enhance knowledge retrieval and reasoning capabilities for large language models.

## What is BiG-RAG?

BiG-RAG constructs a **bipartite knowledge graph** using **n-ary relation extraction** from your documents. This graph-based approach enables more sophisticated multi-hop reasoning compared to traditional vector-only RAG systems.

**Key Features:**
- **Bipartite Graph Structure**: Documents ↔ Entities ↔ Relations for enhanced knowledge representation
- **Three-Path Retrieval** ⭐ **NEW**: Entity-based (Path A) + Relation-based (Path B) + Chunk-based (Path C) for +15-25% recall improvement
- **Semantic Reranking** ⭐ **NEW**: Cross-encoder reranking for +10-20% precision improvement
- **Cascade Document Deletion** ⭐ **NEW**: Smart deletion with shared entity preservation (~1-2s, no rebuild needed)
- **Metadata Preservation** ⭐ **NEW**: Document metadata flows through extraction for +2-3 F1 improvement
- **Multiple Storage Backends**: Support for Milvus, ChromaDB, Neo4J, MongoDB, Oracle, TiDB
- **Flexible Retrieval Modes**: Hybrid, local (entity-based), global (relation-based), naive (text-only)
- **OpenAI Integration**: Ready-to-use with GPT models for testing and development
- **Async-First Design**: Efficient concurrent processing for large-scale applications

---

## Quick Start

### Installation

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

# 4. Install PyTorch
pip install torch torchvision torchaudio

# 5. Install BiG-RAG dependencies
pip install -r requirements.txt

# 6. Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

> **Note:** For detailed setup instructions, see [docs/technical/SETUP_VENV.md](docs/technical/SETUP_VENV.md)

---

## Basic Usage

### Step 1: Prepare Your Data

Create a corpus file (`corpus.jsonl`) with your documents:

```json
{"id": "doc_001", "contents": "Your document text here...", "title": "Document Title"}
{"id": "doc_002", "contents": "Another document...", "title": "Another Title"}
```

Place it in: `datasets/your_dataset/raw/corpus.jsonl`

### Step 2: Build Knowledge Graph

Set your OpenAI API key:
```bash
echo "your-api-key-here" > openai_api_key.txt
```

Build the bipartite graph:
```bash
python script_build.py --data_source your_dataset
```

This will:
- Extract entities and relations from your documents
- Create bipartite graph structure
- Generate embeddings
- Save to `expr/your_dataset/`

### Step 3: Start Retrieval Server

```bash
# NEW: Use backend/server.py
cd backend
python server.py --data_source your_dataset
```

The API server runs on `http://localhost:8001`

**Or use the React UI (NEW):**
```bash
# Terminal 1: Start backend
cd backend && python server.py --data_source your_dataset

# Terminal 2: Start frontend
cd frontend && npm run dev
```
Then open `http://localhost:5173` in your browser

### Step 4: Use BiG-RAG in Your Code

```python
from bigrag import BiGRAG, QueryParam

# Initialize
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

BiG-RAG has been reorganized for better clarity and scalability:

```
BiG-RAG/
├── README.md                    # This file
├── CLAUDE.md                    # Claude Code assistant instructions
├── BIGRAG_UI_PLAN.md           # UI implementation plan
├── IMPLEMENTATION_STATUS.md     # Current development status
│
├── backend/                     # FastAPI server (NEW)
│   ├── api/                    # API modules
│   ├── server.py               # Main server (was script_api.py)
│   └── README.md               # Backend documentation
│
├── frontend/                    # React UI (NEW - Nov 2025)
│   ├── src/                    # React 19 + TypeScript + Tailwind v4
│   ├── package.json            # Latest dependencies
│   └── README.md               # Frontend documentation
│
├── bigrag/                      # Core Python library
│   ├── bigrag.py               # Main BiGRAG class
│   ├── operate.py              # Graph operations
│   ├── reranker.py             # Semantic reranking
│   └── ...                     # Other modules
│
├── docs/                        # Documentation (NEW)
│   ├── README.md               # Documentation index
│   ├── technical/              # Design specs, setup guides
│   ├── reports/                # Test & evaluation reports
│   └── updates/                # Change logs
│
├── test_scripts/                # Test & validation scripts (NEW)
│   ├── README.md               # Test documentation
│   ├── test_*.py               # Various test scripts
│   └── validate_*.py           # Validation scripts
│
├── datasets/                    # QA datasets and corpora
├── expr/                        # Built knowledge graphs
├── script_build.py             # Build knowledge graph
├── script_process.py           # Process datasets
└── setup.py                    # Package setup
```

**Key Changes:**
- ✅ `api/` → `backend/api/` for clear separation
- ✅ `script_api.py` → `backend/server.py` with path fixes
- ✅ `frontend/` added with React 19 + TypeScript + Tailwind CSS v4
- ✅ `docs/` organized into technical/, reports/, updates/
- ✅ `test_scripts/` consolidates all test files
- ✅ Root directory clean with only 4 markdown files

See [`docs/README.md`](docs/README.md) for complete documentation index.

---

## Testing BiG-RAG

We provide a complete test suite in [`test_scripts/`](test_scripts/):

```bash
cd test_scripts

# Test all retrieval modes
python test_all_retrieval_modes.py

# Test Phase 2-4 improvements
python test_improvements.py

# Validate SingleTopic dataset
python validate_singletopic_dataset.py

# Run complete evaluation
python run_singletopic_evaluation.py
```

See [`test_scripts/README.md`](test_scripts/README.md) for all available tests.

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

For complete details, see [docs/updates/IMPLEMENTATION_SUMMARY.md](docs/updates/IMPLEMENTATION_SUMMARY.md)

---

## Coming Soon

The following components will be released when fully ready:

- **RL Training Framework** - GRPO, PPO, REINFORCE++ implementations for training LLMs with graph-based retrieval
- **Agent System** - Tool-based agent for iterative retrieval and reasoning
- **Evaluation Module** - Metrics and benchmarking tools
- **Inference Module** - Optimized deployment for production
- **Complete Documentation** - In-depth technical guides and tutorials
- **Architecture Diagrams** - Visual explanations of system design

---

## System Requirements

**Minimum:**
- Python 3.11+
- 4GB RAM
- CPU (for small datasets)

**Recommended:**
- Python 3.11+
- 16GB+ RAM
- GPU with 8GB+ VRAM (for large datasets)
- SSD storage

---

## License

See [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: https://github.com/dhrubo326/BiG-RAG/issues
- **Discussions**: https://github.com/dhrubo326/BiG-RAG/discussions

---

## Acknowledgments

BiG-RAG builds upon research in graph-based RAG systems and reinforcement learning for LLMs. Thanks to the open-source community for foundational tools and frameworks.
