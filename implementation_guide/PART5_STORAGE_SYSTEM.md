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
- **Vector Storage**: FAISS (dev) → Milvus (prod) → Oracle (enterprise)
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
│    │   ├─ NanoVectorDBStorage (FAISS)                    │
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

---

## 2. Implementation Details

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
    """FAISS-based vector storage"""

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

---

## Summary

**Key Takeaways:**
1. **Abstract base classes** enable pluggable backends
2. **Default implementations** for development (NetworkX, FAISS, JSON)
3. **Enterprise backends** for production (Neo4J, Milvus, MongoDB, Oracle)
4. **Lazy loading** avoids requiring all dependencies
5. **Easy migration** from dev to production backends
