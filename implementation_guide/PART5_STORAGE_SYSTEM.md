# Part 5: Storage System

**Deep-Dive Documentation for BiG-RAG Framework**

---

## 1. Conceptual Overview

### What Problem Does This Solve?

**Problem:** Different deployment scenarios need different storage backends:
- **Development**: Fast in-memory storage
- **Production**: Persistent, scalable databases
- **Enterprise**: Existing database infrastructure
- **Cost optimization**: Balance between performance and cost

**BiG-RAG Solution:** Pluggable storage architecture with abstract base classes:
- **Graph Storage**: NetworkX (dev) → Neo4J (prod) → Oracle (enterprise)
- **Vector Storage**: NanoVectorDB/FAISS (dev) → Milvus (prod) → Oracle (enterprise)
- **KV Storage**: JSON (dev) → MongoDB (prod) → TiDB (enterprise)

### Architecture

```
┌───────────────────────────────────────────────────────────┐
│              STORAGE ARCHITECTURE                          │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  BiGRAG Core                                               │
│    ↓                                                       │
│  Abstract Base Classes (bigrag/base.py)                    │
│    ├─ BaseGraphStorage                                    │
│    ├─ BaseVectorStorage                                   │
│    └─ BaseKVStorage                                       │
│    ↓                                                       │
│  Implementations                                           │
│    ├─ Default (bigrag/storage.py)                         │
│    │   ├─ NetworkXStorage (in-memory graph)              │
│    │   ├─ NanoVectorDBStorage (FAISS-based)             │
│    │   └─ JsonKVStorage (file-based)                     │
│    │                                                       │
│    └─ Enterprise (bigrag/kg/)                             │
│        ├─ Neo4JStorage                                    │
│        ├─ OracleGraphStorage                              │
│        ├─ MongoKVStorage                                  │
│        ├─ MilvusVectorDBStorage                           │
│        └─ TiDBVectorStorage                               │
└───────────────────────────────────────────────────────────┘
```

### Comparison to Standard GraphRAG

BiG-RAG implements all core GraphRAG features plus enhanced capabilities:

| Feature | Standard GraphRAG | BiG-RAG | Status |
|---------|-------------------|---------|--------|
| **Text chunking** | ✅ Yes | ✅ Yes (1200 tokens, 100 overlap) | ✅ Implemented |
| **Chunk storage** | ✅ Yes | ✅ Yes (KV + vector DB) | ✅ Implemented |
| **Entity extraction** | ✅ Yes | ✅ Yes (GPT-4o-mini) | ✅ Implemented |
| **Entity embeddings** | ✅ Yes | ✅ Yes (OpenAI embeddings) | ✅ Implemented |
| **Vector search** | ✅ Yes | ✅ Yes (entities + relations) | ✅ Implemented |
| **Graph structure** | ✅ Yes | ✅ Yes (bipartite graph) | ✅ Implemented |
| **Graph traversal** | ✅ Yes | ✅ Yes (during retrieval) | ✅ Implemented |
| **Hybrid retrieval** | ✅ Yes | ✅ Yes (vector + graph) | ✅ Implemented |
| **Relation extraction** | ⚠️ Often binary | ✅ N-ary relations | ✅ Enhanced |
| **Dual-path retrieval** | ❌ Usually entity-only | ✅ Entity + relation paths | ✅ Enhanced |

**BiG-RAG's Enhanced Features:**
1. **Bipartite graph structure**: Entities and relations as separate node types (not traditional entity graphs)
2. **N-ary relations**: Preserves complex relationships in natural language
3. **✨ Three-path retrieval**: Entity + Relation + Chunk-based vector searches
4. **Three-layer storage**: KV storage + Vector DB + Graph DB working together
5. **Reciprocal rank fusion**: Combines multiple retrieval paths intelligently
6. **✨ Document deletion system**: Cascade cleanup with smart partial/full deletion

---

## 2. Implementation Details

### Complete Storage Flow

**During Graph Construction** (`script_build.py`):

```
Input: Raw Documents
   ↓
1. Chunk into 1200-token segments (100 overlap)
   │  Implementation: bigrag/operate.py → chunking_by_token_size()
   ↓
2. Store chunks in KV storage
   ├─→ kv_store_text_chunks.json (metadata)
   └─→ vdb_chunks (vector embeddings)
   │  Implementation: bigrag/bigrag.py → text_chunks.upsert()
   ↓
3. Extract entities from chunks (GPT-4o-mini)
   │  Implementation: bigrag/operate.py → extract_entities()
   ↓
4. Store entities
   ├─→ graph_chunk_entity_relation.graphml (metadata in graph nodes)
   ├─→ vdb_entities.json (vector embeddings via NanoVectorDB)
   └─→ chunk_entity_relation_graph (in-memory NetworkX graph)
   │  Implementation: bigrag/operate.py → _merge_nodes_then_upsert()
   ↓
5. Extract n-ary relations from chunks
   │  Implementation: bigrag/operate.py → extract_entities() [bipartite edges]
   ↓
6. Store relations
   ├─→ graph_chunk_entity_relation.graphml (metadata in graph nodes)
   ├─→ vdb_bipartite_edges.json (vector embeddings via NanoVectorDB)
   └─→ chunk_entity_relation_graph (graph edges)
   │  Implementation: bigrag/operate.py → _merge_edges_then_upsert()
   ↓
7. Build vector indices
   ├─→ vdb_entities.json (entity vectors in NanoVectorDB)
   ├─→ vdb_bipartite_edges.json (relation vectors in NanoVectorDB)
   └─→ vdb_chunks.json (chunk vectors in NanoVectorDB)
   │  Implementation: bigrag/storage.py → NanoVectorDBStorage
   ↓
Output: Complete Bipartite Graph
```

**During Retrieval** (`script_api.py` or `aquery()`):

```
Input: User Query
   ↓
1. Embed query
   │  Implementation: embedding_func() via OpenAI/FlagEmbedding
   ↓
2. Parallel vector searches
   ├─→ vdb_entities.query() (find similar entities)
   └─→ vdb_bipartite_edges.query() (find similar relations)
   │  Implementation: bigrag/operate.py → _build_query_context()
   ↓
3. For each result, traverse graph
   ├─→ Get entity metadata (graph.get_node())
   ├─→ Get entity degree (graph.node_degree())
   ├─→ Find connected relations (_find_most_related_edges_from_entities())
   └─→ Find connected entities
   │  Implementation: bigrag/operate.py → _get_node_data(), _get_edge_data()
   ↓
4. Reciprocal rank fusion
   ├─→ Combine entity-based results
   ├─→ Combine relation-based results
   └─→ Score = 1/(rank+1) for each result
   │  Implementation: bigrag/operate.py → _merge_and_rank()
   ↓
5. Return top-k fused results
   ↓
Output: Ranked Knowledge Contexts
```

**Key Storage Interactions:**
- **Write operations**: Always async with `await storage.upsert(data)`
- **Read operations**: Always async with `await storage.query(query, top_k)`
- **Persistence**: Triggered by `index_done_callback()` after batch operations
- **Caching**: LLM responses cached in `llm_response_cache` (optional KV storage)

---

### Base Classes

**File:** `bigrag/base.py`

```python
class BaseVectorStorage(ABC):
    @abstractmethod
    async def query(self, query: str, top_k: int) -> list[dict]: pass

    @abstractmethod
    async def upsert(self, data: dict[str, dict]): pass

    @abstractmethod
    async def index_done_callback(self): pass

class BaseKVStorage(ABC):
    @abstractmethod
    async def get_by_id(self, id: str): pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str], fields: set[str]): pass

    @abstractmethod
    async def filter_keys(self, data: list[str]) -> set[str]: pass

    @abstractmethod
    async def upsert(self, data: dict[str, dict]): pass

class BaseGraphStorage(ABC):
    @abstractmethod
    async def has_node(self, node_id: str) -> bool: pass

    @abstractmethod
    async def get_node(self, node_id: str) -> dict: pass

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict): pass

    @abstractmethod
    async def upsert_edge(self, src: str, tgt: str, edge_data: dict): pass

    @abstractmethod
    async def get_node_edges(self, node_id: str) -> list[tuple]: pass
```

### Default Implementations

**NetworkXStorage:**
```python
class NetworkXStorage(BaseGraphStorage):
    """In-memory graph with NetworkX backend"""

    def __init__(self, namespace: str, working_dir: str):
        self._graph = nx.Graph()
        self.namespace = namespace
        self.working_dir = working_dir

    async def upsert_node(self, node_id: str, node_data: dict):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict):
        self._graph.add_edge(src, tgt, **edge_data)

    async def index_done_callback(self):
        # Stabilize and save
        self._stabilize_graph()
        nx.write_graphml(self._graph, f"{self.working_dir}/{self.namespace}.graphml")
```

**NanoVectorDBStorage:**
```python
class NanoVectorDBStorage(BaseVectorStorage):
    """NanoVectorDB (FAISS-based) vector storage"""

    async def query(self, query: str, top_k: int) -> list[dict]:
        # Embed query
        embedding = await self.global_config["embedding_func"]([query])

        # FAISS search
        results = self._db.query(embedding[0], top_k)

        return results

    async def upsert(self, data: dict[str, dict]):
        # Batch embed
        texts = [self._format_text(d) for d in data.values()]
        embeddings = await self._embed_batch(texts)

        # Insert
        for (id, item), embedding in zip(data.items(), embeddings):
            item["__vector__"] = embedding
            self._db.upsert({id: item})
```

**JsonKVStorage:**
```python
class JsonKVStorage(BaseKVStorage):
    """File-based key-value storage"""

    def __init__(self, namespace: str, working_dir: str):
        self.namespace = namespace
        self.working_dir = working_dir
        self._data = {}
        self._load()

    def _load(self):
        filepath = f"{self.working_dir}/{self.namespace}.json"
        if os.path.exists(filepath):
            with open(filepath) as f:
                self._data = json.load(f)

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    async def index_done_callback(self):
        filepath = f"{self.working_dir}/{self.namespace}.json"
        with open(filepath, 'w') as f:
            json.dump(self._data, f, indent=2)
```

---

## 3. Configuration Reference

### Selecting Storage Backends

```python
# Development (default)
rag = BiGRAG(
    working_dir="./expr/dev",
    graph_storage="NetworkXStorage",
    vector_storage="NanoVectorDBStorage",
    kv_storage="JsonKVStorage"
)

# Production (scalable)
rag = BiGRAG(
    working_dir="./expr/prod",
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorage",
    kv_storage="MongoKVStorage",
    storage_config={
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "milvus_host": "localhost",
        "milvus_port": 19530,
        "mongo_uri": "mongodb://localhost:27017"
    }
)

# Enterprise (Oracle)
rag = BiGRAG(
    graph_storage="OracleGraphStorage",
    vector_storage="OracleVectorDBStorage",
    kv_storage="OracleKVStorage",
    storage_config={
        "oracle_dsn": "localhost:1521/ORCLPDB1",
        "oracle_user": "system",
        "oracle_password": "password"
    }
)
```

---

## 4. Usage Examples

### Basic Usage

```python
# Use default storage
rag = BiGRAG(working_dir="./expr/my_kg")
rag.insert(documents)  # Automatically uses JSON/NetworkX/FAISS

# Query
context = rag.query("What is Paris?")
```

### Enterprise Migration

```python
# Step 1: Build with default storage
rag_dev = BiGRAG(
    working_dir="./expr/dev",
    graph_storage="NetworkXStorage"
)
rag_dev.insert(documents)

# Step 2: Export to Neo4J
def migrate_to_neo4j(source_dir: str, neo4j_uri: str):
    # Load from NetworkX
    graph = nx.read_graphml(f"{source_dir}/graph_chunk_entity_relation.graphml")

    # Connect to Neo4J
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", "password"))

    with driver.session() as session:
        # Create nodes
        for node, data in graph.nodes(data=True):
            session.run(
                "CREATE (n:Node {id: $id, data: $data})",
                id=node, data=data
            )

        # Create edges
        for src, tgt, data in graph.edges(data=True):
            session.run(
                "MATCH (a:Node {id: $src}), (b:Node {id: $tgt}) "
                "CREATE (a)-[r:EDGE $data]->(b)",
                src=src, tgt=tgt, data=data
            )

migrate_to_neo4j("./expr/dev", "bolt://localhost:7687")

# Step 3: Use Neo4J backend
rag_prod = BiGRAG(
    working_dir="./expr/prod",
    graph_storage="Neo4JStorage",
    storage_config={"neo4j_uri": "bolt://localhost:7687"}
)
context = rag_prod.query("What is Paris?")  # Queries Neo4J
```

---

## 5. Troubleshooting

### Issue: Storage Backend Not Found

```python
# Error: ModuleNotFoundError: No module named 'neo4j'

# Solution: Install optional dependencies
pip install neo4j
pip install pymilvus
pip install pymongo

# Or install all enterprise backends
pip install -r requirements_enterprise.txt
```

### Issue: Graph Too Large for Memory

```python
# Problem: NetworkX graph exceeds RAM

# Solution 1: Use Neo4J
rag = BiGRAG(graph_storage="Neo4JStorage")

# Solution 2: Partition graph
def build_graph_partitioned(documents, partition_size=10000):
    for i in range(0, len(documents), partition_size):
        partition = documents[i:i+partition_size]
        rag = BiGRAG(working_dir=f"./expr/partition_{i//partition_size}")
        rag.insert(partition)
```

---

## 6. API Reference

### Adding Custom Backend

```python
# Step 1: Implement base class
class MyVectorStorage(BaseVectorStorage):
    async def query(self, query: str, top_k: int) -> list[dict]:
        # Your implementation
        pass

    async def upsert(self, data: dict[str, dict]):
        # Your implementation
        pass

    async def index_done_callback(self):
        # Your implementation
        pass

# Step 2: Register in lazy_external_import()
# File: bigrag/bigrag.py
def lazy_external_import(cls_name: str):
    if cls_name == "MyVectorStorage":
        from bigrag.kg.vectordb_impl.my_impl import MyVectorStorage
        return MyVectorStorage
    # ...

# Step 3: Use
rag = BiGRAG(vector_storage="MyVectorStorage")
```

### ✨ Document Deletion API (NEW)

**Purpose:** Remove documents from the knowledge graph with cascade cleanup

**Method Signature:**
```python
async def adelete_document(self, doc_id: str) -> dict:
    """
    Delete a document and all associated data.

    Args:
        doc_id: Document ID or original content

    Returns:
        Deletion statistics dictionary
    """
```

**Smart Deletion Logic:**

```
Document "doc-abc123"
   ↓
1. Find all chunks: chunk-001, chunk-042, chunk-058
   ↓
2. Find entities/edges referencing those chunks:
   • Entity "PARIS" (source_ids: chunk-001, chunk-042, chunk-100)
   • Entity "FRANCE" (source_ids: chunk-042)  ← Only from this doc
   ↓
3. Smart cleanup:
   • PARIS: Remove chunk-001, chunk-042 from source_ids (keep chunk-100)
   • FRANCE: DELETE completely (no other sources)
   ↓
4. Delete chunks from storage:
   • Delete chunk-001, chunk-042, chunk-058 from text_chunks
   • Delete chunk-001, chunk-042, chunk-058 from vdb_chunks
   ↓
5. Delete document from full_docs
```

**Usage Examples:**

```python
# Example 1: Delete by document ID
from bigrag import BiGRAG

rag = BiGRAG(working_dir="./expr/demo_test")
stats = rag.delete_document("doc-abc123...")

# Returns:
# {
#     "status": "success",
#     "doc_id": "doc-abc123",
#     "chunks_deleted": 15,
#     "entities_deleted": 3,    # Entities unique to this doc
#     "entities_updated": 8,    # Entities shared with other docs
#     "edges_deleted": 5,
#     "edges_updated": 12
# }

# Example 2: Delete by content (auto-computes ID)
original_content = "The Eiffel Tower is in Paris, France."
stats = rag.delete_document(original_content)

# Example 3: Async version
import asyncio

async def delete_multiple(doc_ids):
    rag = BiGRAG(working_dir="./expr/demo_test")
    results = []
    for doc_id in doc_ids:
        stats = await rag.adelete_document(doc_id)
        results.append(stats)
    return results

results = asyncio.run(delete_multiple(["doc-1", "doc-2", "doc-3"]))
```

**Storage Impact:**

| Storage Component | Full Delete | Partial Update |
|-------------------|-------------|----------------|
| **full_docs** | ✅ Always deleted | N/A |
| **text_chunks** | ✅ All doc chunks deleted | N/A |
| **vdb_chunks** | ✅ All doc chunks deleted | N/A |
| **vdb_entities** | ✅ If unique to doc | ⚠️ Update if shared |
| **vdb_bipartite_edges** | ✅ If unique to doc | ⚠️ Update if shared |
| **chunk_entity_relation_graph** | ✅ Nodes with 0 sources | ⚠️ Remove source_ids |

**Benefits:**
- ✅ **Data hygiene**: Remove outdated/incorrect documents
- ✅ **Storage management**: Prevent indefinite growth
- ✅ **Testing**: Easily reset test data
- ✅ **GDPR compliance**: Remove user data on request
- ✅ **Incremental updates**: Delete old version before inserting new

**Performance:**
- Complexity: O(chunks × entities) - depends on graph density
- Typical: ~100-500ms for documents with 10-50 chunks
- Async-first for non-blocking deletion
```

---

## 7. Performance Analysis

### Storage Comparison

| Backend | Read Latency | Write Throughput | Scalability | Cost |
|---------|--------------|------------------|-------------|------|
| NetworkX | 0.1ms | 10K/s | 100K nodes | Free |
| Neo4J | 2ms | 5K/s | 10M nodes | $ |
| FAISS | 1ms | 50K/s | 10M vectors | Free |
| Milvus | 5ms | 100K/s | 1B vectors | $$ |
| MongoDB | 3ms | 20K/s | Unlimited | $ |

---

## 8. Testing Guide

```python
def test_storage_backend(storage: BaseVectorStorage):
    """Test any vector storage implementation"""

    # Test upsert
    data = {
        "test_1": {"content": "Test", "__vector__": np.random.rand(1536).tolist()}
    }
    await storage.upsert(data)

    # Test query
    results = await storage.query("Test", top_k=1)
    assert len(results) == 1
    assert results[0]["content"] == "Test"

# Use with any backend
test_storage_backend(NanoVectorDBStorage())
test_storage_backend(MilvusVectorDBStorage())
```

### Verifying Storage Files

After running `script_build.py`, verify the following files are created in your working directory:

```bash
expr/2WikiMultiHopQA/
├── kv_store_full_docs.json            # ✅ Full document metadata (KV storage)
├── kv_store_text_chunks.json          # ✅ Text chunks (KV storage)
├── kv_store_llm_response_cache.json   # ✅ LLM cache (optional)
├── vdb_entities.json                  # ✅ Entity vectors (NanoVectorDB)
├── vdb_bipartite_edges.json           # ✅ Relation vectors (NanoVectorDB)
├── vdb_chunks.json                    # ✅ Chunk vectors (NanoVectorDB)
└── graph_chunk_entity_relation.graphml # ✅ Graph structure + entity/relation metadata
```

**Note**: Entity and relation **metadata** (names, descriptions, source_ids, weights) are stored in the GraphML file, not in separate JSON files.

**Expected File Sizes** (for ~10K document corpus):

| File | Typical Size | Purpose |
|------|--------------|---------|
| `kv_store_*.json` | 100KB - 10MB | Document and chunk metadata |
| `vdb_*.json` | 10MB - 100MB | NanoVectorDB indices with embeddings |
| `graph_*.graphml` | 1MB - 50MB | NetworkX graph with entity/relation metadata |

**How to Verify**:

```bash
# Check all files exist
ls -lh expr/2WikiMultiHopQA/

# Validate JSON files are readable
python -c "import json; json.load(open('expr/2WikiMultiHopQA/kv_store_text_chunks.json'))"

# Check vector DB contents
python -c "import json; vdb = json.load(open('expr/2WikiMultiHopQA/vdb_entities.json')); print(f'Entities: {len(vdb[\"data\"])}')"

# Check graph structure
python -c "import networkx as nx; G = nx.read_graphml('expr/2WikiMultiHopQA/graph_chunk_entity_relation.graphml'); print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')"
```

**Expected Output**:
```
Entities: 15234
Shape: (15234, 1536)  # or (15234, 3072) for text-embedding-3-large
```

**If Files Are Missing**:
1. Check build logs for errors: `tail -f build.log`
2. Verify OpenAI API key is set: `cat openai_api_key.txt`
3. Ensure corpus.jsonl exists: `ls datasets/2WikiMultiHopQA/raw/corpus.jsonl`
4. Re-run build: `python script_build.py --data_source 2WikiMultiHopQA`

**Storage Size Estimates**:

| Corpus Size | Total Storage | Entity Index | Relation Index | Chunks Index |
|-------------|--------------|--------------|----------------|--------------|
| 1K docs | ~100 MB | ~10 MB | ~10 MB | ~50 MB |
| 10K docs | ~1 GB | ~100 MB | ~100 MB | ~500 MB |
| 100K docs | ~10 GB | ~1 GB | ~1 GB | ~5 GB |
| 1M docs | ~100 GB | ~10 GB | ~10 GB | ~50 GB |

**Note**: Actual sizes depend on:
- Document length and complexity
- Number of entities/relations extracted
- Embedding dimensions (1536 vs 3072)
- Storage backend (JSON is larger than binary formats)

---

## Summary

**Key Takeaways:**
1. **Abstract base classes** enable pluggable backends
2. **Default implementations** for development (NetworkX, FAISS, JSON)
3. **Enterprise backends** for production (Neo4J, Milvus, MongoDB, Oracle)
4. **Lazy loading** avoids requiring all dependencies
5. **Easy migration** from dev to production backends
