# BiG-RAG Development Notes

**Purpose**: Technical implementation details, architecture patterns, and developer guidance
**Audience**: Developers working on or extending BiG-RAG
**Last Updated**: 2025-10-24

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Storage Plugin System](#storage-plugin-system)
3. [Bipartite Graph Structure](#bipartite-graph-structure)
4. [Testing Framework](#testing-framework)
5. [Known Issues and Solutions](#known-issues-and-solutions)
6. [Performance Considerations](#performance-considerations)
7. [Extension Points](#extension-points)

---

## Architecture Overview

### Core Components

BiG-RAG consists of several layered components:

```
┌─────────────────────────────────────────────────────────────┐
│                      BiGRAG Class                            │
│  (Main interface - bigrag/bigrag.py)                         │
├─────────────────────────────────────────────────────────────┤
│  - ainsert(): Async document insertion                      │
│  - aquery(): Async retrieval with tool calls                │
│  - Manages: entities_vdb, bipartite_edges_vdb, text_chunks  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  Storage Layer (Pluggable)                   │
├─────────────────────────────────────────────────────────────┤
│  Base Classes (bigrag/base.py):                             │
│    - BaseVectorStorage: Vector DB interface                 │
│    - BaseKVStorage: Key-value storage interface             │
│    - BaseGraphStorage: Graph database interface             │
│                                                              │
│  Default Implementations (bigrag/storage.py):               │
│    - NanoVectorDBStorage: In-memory vector DB               │
│    - JsonKVStorage: JSON file-based KV store                │
│    - NetworkXStorage: In-memory graph (NetworkX)            │
│                                                              │
│  Optional Backends (bigrag/kg/*.py):                        │
│    - Milvus, ChromaDB, TiDB (vector)                        │
│    - Neo4J, MongoDB, Oracle (graph)                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   Retrieval Operations                       │
│  (bigrag/operate.py)                                         │
├─────────────────────────────────────────────────────────────┤
│  - extract_entities(): LLM-based entity extraction          │
│  - kg_query(): Graph traversal and retrieval                │
│  - _get_node_data(): Entity lookups                         │
│  - _get_edge_data(): Relation lookups                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    FAISS/Vector Search                       │
│  (Embedding-based similarity)                                │
├─────────────────────────────────────────────────────────────┤
│  - Entity embeddings: 3072-dim (text-embedding-3-large)     │
│  - Edge embeddings: 3072-dim                                │
│  - Chunk embeddings: 3072-dim                               │
│  - Index type: IndexFlatIP (inner product)                  │
└─────────────────────────────────────────────────────────────┘
```

### Async-First Design

Nearly all BiG-RAG operations are async:

```python
# ✅ Correct usage (async)
await bigrag.ainsert(documents)
contexts = await bigrag.aquery(query, param)

# ⚠️ Synchronous wrappers (discouraged but available)
bigrag.insert(documents)  # Internally calls ainsert()
contexts = bigrag.query(query, param)  # Internally calls aquery()
```

**Why async?**
- Enables concurrent operations (batch processing)
- Better resource utilization (I/O-bound tasks)
- Required for distributed workers in RL training
- Allows multiple API calls without blocking

---

## Storage Plugin System

### Design Philosophy

BiG-RAG uses abstract base classes with lazy imports to support multiple backends without requiring all dependencies.

### Base Classes

**Location**: `bigrag/base.py`

```python
class BaseVectorStorage:
    """Abstract interface for vector databases"""
    async def query(query: str, top_k: int) -> List[Dict]
    async def upsert(data: Dict[str, Dict])

class BaseKVStorage:
    """Abstract interface for key-value stores"""
    async def all_keys() -> List[str]
    async def get_by_id(id: str) -> Dict
    async def upsert(data: Dict[str, Dict])

class BaseGraphStorage:
    """Abstract interface for graph databases"""
    async def upsert_node(node_id: str, node_data: Dict)
    async def upsert_edge(source: str, target: str, edge_data: Dict)
    async def has_node(node_id: str) -> bool
    async def has_edge(source: str, target: str) -> bool
```

### Default Implementations

**Location**: `bigrag/storage.py`

**NanoVectorDBStorage**:
- In-memory vector database
- Uses FAISS for similarity search
- Persistent storage via JSON files
- Metadata filtering support

**JsonKVStorage**:
- JSON file-based key-value store
- Each namespace stored separately
- Efficient for small to medium datasets

**NetworkXStorage**:
- In-memory graph using NetworkX
- Supports bipartite graph operations
- GraphML export for visualization

### Optional Backends

**Location**: `bigrag/kg/*.py`

Available implementations:
- **Vector**: Milvus, ChromaDB, TiDB, Oracle
- **Graph**: Neo4J, MongoDB, Oracle
- **KV**: MongoDB, Oracle, TiDB

### Adding a New Backend

1. **Inherit from base class**:
```python
from bigrag.base import BaseVectorStorage

class MyVectorDB(BaseVectorStorage):
    async def query(self, query: str, top_k: int):
        # Your implementation
        pass
```

2. **Add to lazy import** in `bigrag/bigrag.py`:
```python
def lazy_external_import():
    if "myvectordb" in storage_backend:
        from bigrag.kg.myvectordb_impl import MyVectorDB
        return MyVectorDB
```

3. **Configure via constructor**:
```python
rag = BiGRAG(
    vector_db_storage_cls=MyVectorDB,
    working_dir="expr/dataset"
)
```

---

## Bipartite Graph Structure

### Graph Architecture

Unlike traditional hypergraphs, BiG-RAG uses a **true bipartite graph**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Bipartite Graph Structure                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Document Chunks                    Entities & Relations     │
│  ┌──────────┐                         ┌──────────┐          │
│  │  Doc A   │◄──────────────────────► │ Entity 1 │          │
│  └──────────┘                         └──────────┘          │
│       ▲                                     ▲                │
│       │                                     │                │
│       ▼                                     ▼                │
│  ┌──────────┐     Bipartite Edge    ┌──────────┐           │
│  │  Doc B   │◄──────────────────────►│ Relation │           │
│  └──────────┘                         └──────────┘          │
│       ▲                                     ▲                │
│       │                                     │                │
│       ▼                                     ▤                │
│  ┌──────────┐                         ┌──────────┐          │
│  │  Doc C   │◄──────────────────────► │ Entity 2 │          │
│  └──────────┘                         └──────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Properties**:
- **Two node types**: Documents (chunks) and Semantic nodes (entities + relations)
- **Edges**: Connect documents to the entities/relations they contain
- **No direct edges**: Documents don't connect to documents; entities don't connect to entities
- **Queries**: Traverse from query → entities → relations → documents

### Storage Organization

BiG-RAG maintains three separate vector databases:

1. **Text Chunks VDB** (`text_chunks`):
   - Stores document chunks
   - Used for naive retrieval mode
   - Embeddings of raw text

2. **Entities VDB** (`entities_vdb`):
   - Stores extracted entities
   - Metadata: `entity_name`, `description`, `source_id`
   - Used for local retrieval mode

3. **Bipartite Edges VDB** (`bipartite_edges_vdb`):
   - Stores relations (n-ary facts)
   - Metadata: `bipartite_edge_name`, `source_id`, `description`
   - Used for global retrieval mode

### Retrieval Modes

**Hybrid Mode** (default, most effective):
```python
param = QueryParam(mode="hybrid", top_k=10)
result = await rag.aquery(query, param)
```
- Combines entity + relation retrieval
- Traverses bipartite graph structure
- Best for multi-hop reasoning

**Local Mode** (entity-focused):
```python
param = QueryParam(mode="local", top_k=10)
result = await rag.aquery(query, param)
```
- Entity-based retrieval only
- Faster but less comprehensive

**Global Mode** (relation-focused):
```python
param = QueryParam(mode="global", top_k=10)
result = await rag.aquery(query, param)
```
- Relation-based retrieval only
- Good for factual queries

**Naive Mode** (baseline):
```python
param = QueryParam(mode="naive", top_k=10)
result = await rag.aquery(query, param)
```
- Direct text chunk retrieval
- No graph traversal
- Baseline comparison

---

## Testing Framework

### Test Suite Overview

**Created**: 2025-10-24
**Purpose**: Validate BiG-RAG functionality with OpenAI models
**Location**: Root directory

### Test Scripts

#### 1. test_build_graph.py

**Purpose**: Build knowledge graph from demo corpus

**What it does**:
- Loads 10 documents from `datasets/demo_test/raw/corpus.jsonl`
- Extracts entities and relations using gpt-4o-mini
- Creates bipartite graph structure
- Generates embeddings with text-embedding-3-large (3072-dim)
- Saves to `expr/demo_test/`

**Output**:
```
expr/demo_test/
├── kv_store_text_chunks.json          # Text chunk metadata
├── vdb_entities.json                  # Entity VDB (with embeddings)
├── vdb_bipartite_edges.json           # Edge VDB (with embeddings)
└── graph_chunk_entity_relation.graphml # Graph visualization
```

**Runtime**: 3-8 minutes
**Cost**: ~$0.01-0.02 USD (OpenAI API)

#### 2. test_retrieval.py

**Purpose**: Test retrieval functionality

**What it does**:
- Loads pre-built knowledge graph
- Runs 10 test queries
- Tests all retrieval modes: hybrid, local, global, naive
- Measures coherence scores

**Output**:
```
Total questions: 10
Successful retrievals: 10/10
Success rate: 100.0%
Average coherence: 1.76
```

#### 3. test_end_to_end.py

**Purpose**: Test complete RAG pipeline

**What it does**:
- Retrieves context for each question
- Generates answers using gpt-4o-mini
- Compares with ground truth
- Calculates accuracy

**Output**:
```
Total questions: 10
Correct answers: 9/10
Success rate: 90.0%
```

### Demo Dataset

**Location**: `datasets/demo_test/`

**Corpus** (`raw/corpus.jsonl`):
- 10 documents on AI/ML topics
- Topics: AI, ML, Deep Learning, NLP, Python, Computer Vision, Neural Networks, TensorFlow, PyTorch, RL

**QA Pairs** (`raw/qa_test.json`):
- 10 questions with ground truth answers
- Designed to test single-hop and multi-hop reasoning

### Running Tests

```bash
# Step 1: Set OpenAI API key
echo "sk-your-api-key" > openai_api_key.txt

# Step 2: Activate environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Step 3: Run tests
python test_build_graph.py    # Build graph
python test_retrieval.py       # Test retrieval
python test_end_to_end.py      # Test RAG pipeline
```

---

## Known Issues and Solutions

### Issue 1: Unicode Logging on Windows

**Problem**: Windows console (cp1252) can't display Unicode emojis (✓, ⚠, ❌)

**Error**:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```

**Solution**: Use UTF-8 encoding handlers
```python
import sys
import io

# File handler with UTF-8
logging.FileHandler('app.log', encoding='utf-8')

# Console handler with UTF-8 wrapper
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

**Status**: Fixed in test scripts

---

### Issue 2: FAISS Index Compatibility

**Problem**: FAISS indices created on GPU may not load on CPU

**Solution**: Use CPU-compatible index types
```python
import faiss

# Use IndexFlatIP for CPU/GPU compatibility
index = faiss.IndexFlatIP(dimension)

# Avoid: IndexIVFFlat (GPU-specific)
```

**Status**: BiG-RAG uses IndexFlatIP by default

---

### Issue 3: Memory Usage with Large Datasets

**Problem**: In-memory storage (NanoVectorDBStorage) may consume excessive RAM

**Solution**: Use external vector databases
```python
from bigrag.kg.milvus_impl import MilvusVectorDBStorage

rag = BiGRAG(
    vector_db_storage_cls=MilvusVectorDBStorage,
    working_dir="expr/dataset"
)
```

**Recommendation**:
- **< 10K documents**: Use default (NanoVectorDBStorage)
- **10K - 100K documents**: Use Milvus or ChromaDB
- **> 100K documents**: Use enterprise solution (Oracle, TiDB)

---

### Issue 4: Rate Limits with OpenAI API

**Problem**: Hitting rate limits during entity extraction

**Solution**: Implement exponential backoff (already done in `bigrag/openai_embedding.py`)
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def embed_batch(texts):
    # API call with automatic retry
    pass
```

**Status**: Implemented in OpenAI integration

---

## Performance Considerations

### Embedding Generation

**Bottleneck**: Embedding API calls are slow for large corpora

**Optimization strategies**:
1. **Batch processing**: Process chunks in batches
   ```python
   batch_size = 100
   for i in range(0, len(texts), batch_size):
       batch = texts[i:i+batch_size]
       embeddings = await embed_batch(batch)
   ```

2. **Caching**: Cache embeddings to avoid recomputation
   ```python
   # BiGRAG automatically caches in working_dir
   rag = BiGRAG(working_dir="expr/dataset")  # Reuses cached embeddings
   ```

3. **Use faster models**: Trade quality for speed
   ```python
   # Fast: text-embedding-3-small (1536-dim)
   # Slow: text-embedding-3-large (3072-dim)
   ```

### Entity Extraction

**Bottleneck**: LLM API calls for entity extraction

**Optimization strategies**:
1. **Parallel processing**: Use `asyncio.gather()`
   ```python
   tasks = [extract_entities(chunk) for chunk in chunks]
   results = await asyncio.gather(*tasks)
   ```

2. **Chunk size tuning**: Balance extraction quality vs. speed
   ```python
   # Larger chunks = fewer API calls but may miss entities
   # Smaller chunks = more API calls but better coverage
   chunk_size = 1200  # Default, works well
   ```

3. **Use cheaper models**: gpt-4o-mini instead of gpt-4
   ```python
   llm_model_func = gpt_4o_mini_complete  # Faster, cheaper
   ```

### Query Performance

**Bottleneck**: Graph traversal can be slow for large graphs

**Optimization strategies**:
1. **Limit top_k**: Reduce number of candidates
   ```python
   param = QueryParam(top_k=5)  # Instead of 10 or 20
   ```

2. **Use local mode**: Skip relation traversal
   ```python
   param = QueryParam(mode="local")  # Faster than hybrid
   ```

3. **FAISS optimization**: Use IVF index for large datasets
   ```python
   # For > 100K vectors, use IVF index
   nlist = 100
   quantizer = faiss.IndexFlatIP(dimension)
   index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
   ```

---

## Extension Points

### Custom Entity Extraction

Override `extract_entities()` in `bigrag/operate.py`:

```python
async def custom_extract_entities(
    chunks: List[str],
    llm_model_func: callable,
) -> List[Dict]:
    # Your custom extraction logic
    # Return: [{"entity_name": ..., "description": ...}, ...]
    pass
```

### Custom Retrieval Logic

Extend `kg_query()` in `bigrag/operate.py`:

```python
async def custom_kg_query(
    query: str,
    graph: NetworkXStorage,
    entities_vdb: BaseVectorStorage,
    bipartite_edges_vdb: BaseVectorStorage,
    param: QueryParam,
) -> Dict:
    # Your custom retrieval logic
    pass
```

### Custom Storage Backend

Implement base classes:

```python
from bigrag.base import BaseVectorStorage

class MyVectorDB(BaseVectorStorage):
    async def query(self, query: str, top_k: int):
        # Connect to your vector DB
        # Return similar vectors
        pass

    async def upsert(self, data: Dict[str, Dict]):
        # Insert/update vectors
        pass
```

### Custom Embedding Function

Provide custom embedding function:

```python
async def my_embedding_func(texts: List[str]) -> List[List[float]]:
    # Your embedding model
    # Return: [[emb1], [emb2], ...]
    pass

rag = BiGRAG(
    embedding_func=my_embedding_func,
    working_dir="expr/dataset"
)
```

---

## Best Practices

### 1. Always Use Async/Await

```python
# ✅ Good
async def process():
    result = await rag.aquery(query, param)

# ❌ Bad
def process():
    result = rag.query(query, param)  # Blocks event loop
```

### 2. Reuse BiGRAG Instances

```python
# ✅ Good - Create once, reuse
rag = BiGRAG(working_dir="expr/dataset")
for query in queries:
    result = await rag.aquery(query, param)

# ❌ Bad - Creates multiple instances
for query in queries:
    rag = BiGRAG(working_dir="expr/dataset")  # Loads from disk every time
    result = await rag.aquery(query, param)
```

### 3. Use Appropriate Retrieval Mode

```python
# Multi-hop reasoning → hybrid
param = QueryParam(mode="hybrid")

# Single-hop factual → local
param = QueryParam(mode="local")

# Baseline comparison → naive
param = QueryParam(mode="naive")
```

### 4. Cache Expensive Operations

```python
# BiGRAG automatically caches:
# - Embeddings (in working_dir)
# - LLM responses (optional, via hashing_kv)

rag = BiGRAG(
    working_dir="expr/dataset",  # Enables caching
    llm_response_cache=JsonKVStorage(...)  # Cache LLM calls
)
```

### 5. Monitor API Costs

```python
# Use cheaper models for development
llm_model_func = gpt_4o_mini_complete  # $0.15/1M tokens
embedding_func = text_embedding_3_small  # $0.02/1M tokens

# Use expensive models for production
llm_model_func = gpt_4_complete  # $5/1M tokens
embedding_func = text_embedding_3_large  # $0.13/1M tokens
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("bigrag")
logger.setLevel(logging.DEBUG)
```

### Inspect Graph Structure

```python
# Export graph for visualization
rag.chunk_entity_relation_graph.write_graphml("graph.graphml")

# Open in Gephi, Cytoscape, or NetworkX
import networkx as nx
G = nx.read_graphml("graph.graphml")
```

### Check Vector DB Contents

```python
# List all entities
entity_keys = await rag.entities_vdb.all_keys()
print(f"Total entities: {len(entity_keys)}")

# Inspect specific entity
entity = await rag.entities_vdb.get_by_id("entity_id")
print(entity)
```

### Profile Performance

```python
import time

start = time.time()
result = await rag.aquery(query, param)
elapsed = time.time() - start
print(f"Query took {elapsed:.2f}s")
```

---

## Future Development Ideas

### Potential Enhancements

1. **Incremental Updates**: Support adding documents without full rebuild
2. **Multi-modal Support**: Images, videos, audio in knowledge graph
3. **Federated Learning**: Distributed graph construction
4. **Graph Visualization UI**: Interactive exploration of knowledge graph
5. **Auto-tuning**: Optimize hyperparameters automatically
6. **Batch Querying**: Process multiple queries in parallel
7. **Streaming Retrieval**: Stream results as they're found

### Research Directions

1. **Better Entity Linking**: Improve entity resolution across documents
2. **Relation Extraction**: More sophisticated n-ary relation extraction
3. **Graph Compression**: Reduce storage requirements for large graphs
4. **Dynamic Chunking**: Adaptive chunk sizes based on content
5. **Cross-lingual Support**: Multi-language knowledge graphs

---

**For production deployment guidance, see [README.md](README.md)**
**For complete change history, see [CHANGELOG.md](CHANGELOG.md)**
**For AI agent development reference, see [CLAUDE.md](CLAUDE.md)**
