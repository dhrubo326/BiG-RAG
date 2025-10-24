# BiG-RAG: Bipartite Graph Retrieval-Augmented Generation

**BiG-RAG** is an advanced RAG framework that uses bipartite graph structures to enhance knowledge retrieval and reasoning capabilities for large language models.

## What is BiG-RAG?

BiG-RAG constructs a **bipartite knowledge graph** using **n-ary relation extraction** from your documents. This graph-based approach enables more sophisticated multi-hop reasoning compared to traditional vector-only RAG systems.

**Key Features:**
- **Bipartite Graph Structure**: Documents ↔ Entities ↔ Relations for enhanced knowledge representation
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
pip install -r requirements_graphrag_only.txt

# 6. Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

> **Note:** For detailed setup instructions, see [SETUP_VENV.md](SETUP_VENV.md)

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
python script_api.py --data_source your_dataset
```

The API server runs on `http://localhost:8001`

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

## Testing BiG-RAG

We provide a complete test suite to verify your installation:

```bash
# Build a demo knowledge graph
python test_build_graph.py

# Test retrieval functionality
python test_retrieval.py

# Test end-to-end RAG pipeline
python test_end_to_end.py
```

---

## Retrieval Modes

BiG-RAG supports multiple retrieval strategies:

- **`hybrid`** (default): Combines entity + relation retrieval for best multi-hop reasoning
- **`local`**: Entity-focused retrieval, faster but less comprehensive
- **`global`**: Relation-focused retrieval, good for factual queries
- **`naive`**: Direct text chunk retrieval, baseline comparison

Example:
```python
# Hybrid mode (best for complex queries)
result = rag.query(query, param=QueryParam(mode="hybrid", top_k=10))

# Local mode (faster)
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
