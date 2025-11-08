# BiG-RAG Implementation and Feature Structure Guide

**Version:** 3.2
**Last Updated:** 2025-01-08 (Post-Implementation)
**Purpose:** Comprehensive A-to-Z implementation reference for BiG-RAG framework development, maintenance, testing, debugging, and optimization

**✨ Latest Updates (Jan 2025):**
- ✅ Three-path retrieval system (Entity + Relation + Chunk)
- ✅ Metadata preservation pipeline (title, category, tags)
- ✅ Document deletion system with cascade cleanup
- ✅ Semantic reranking with cross-encoder
- ✅ Reranking toggle for performance optimization
- ✅ Bipartite architecture documentation (NEW - Jan 8, 2025)
- ✅ **Hash-based node IDs** for bipartite edges (30-40% file size reduction) - Jan 8, 2025
- ✅ **Entity type normalization** with 40+ mappings (consistent typing) - Jan 8, 2025
- ✅ **Semaphore control** for LLM API calls (prevents rate limits) - Jan 8, 2025
- ✅ **Weight documentation** with comprehensive semantics guide - Jan 8, 2025
- ✅ **Improved prompts** with Role/Instructions/Examples structure - Jan 8, 2025
- ✅ **Retry wrapper** with exponential backoff for transient failures - Jan 8, 2025
- ✅ **Logging infrastructure** with rotating file handler - Jan 8, 2025
- ✅ **Constants file** for centralized code-level defaults - Jan 8, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [BiGRAG Core Library](#bigrag-core-library)
4. [Storage System](#storage-system)
5. [Entity Extraction Pipeline](#entity-extraction-pipeline)
6. [Query and Retrieval System](#query-and-retrieval-system)
7. [Tool-Augmented Generation](#tool-augmented-generation)
8. [Pipeline Scripts](#pipeline-scripts)
9. [Data Structures and Formats](#data-structures-and-formats)
10. [Extension Points](#extension-points)
11. [Testing Framework](#testing-framework)
12. [Critical Implementation Details](#critical-implementation-details)
13. [Performance Considerations](#performance-considerations)
14. [Best Practices](#best-practices)
15. [Debugging Tips](#debugging-tips)
16. [Known Issues and Solutions](#known-issues-and-solutions)
17. [Future Development Ideas](#future-development-ideas)

**📖 Related Documentation:**
- **[Bipartite Architecture Explained](BIPARTITE_ARCHITECTURE_EXPLAINED.md)** - Deep dive into graph structure and design decisions
- **[Part 1: Graph Construction](PART1_GRAPH_CONSTRUCTION.md)** - Detailed graph construction pipeline
- **[Implementation Summary](../docs/updates/IMPLEMENTATION_SUMMARY.md)** - Recent improvements and features

---

## Overview

BiG-RAG is a bipartite graph-based retrieval-augmented generation framework that combines:
- **Knowledge Graph Construction**: Extracts entities and relations from text corpora with metadata preservation
- **Three-Path Retrieval**: Entity-based (Path A), relation-based (Path B), and chunk-based (Path C) semantic search
- **Semantic Reranking**: Cross-encoder based reranking of chunk candidates for improved relevance
- **Tool-Augmented Generation**: LLMs learn to query knowledge during generation
- **RL Training Integration**: Models trained via GRPO to optimize retrieval and reasoning

**Key Design Principles:**
- **Async-first architecture**: All major operations use async/await patterns
- **Pluggable storage backends**: Abstract base classes with multiple implementations
- **Lazy dependency loading**: Optional backends loaded only when instantiated
- **Cost optimization**: LLM response caching, batch processing, incremental construction

---

## Core Architecture

### High-Level Component Structure

```
BiG-RAG Framework
├── bigrag/                    # Core knowledge graph library
│   ├── bigrag.py             # Main BiGRAG orchestration class
│   ├── operate.py            # Graph construction & three-path retrieval
│   ├── reranker.py           # ✨ NEW: Semantic reranking module
│   ├── storage.py            # Default storage implementations
│   ├── base.py               # Abstract storage interfaces (with metadata)
│   ├── constants.py          # ✨ NEW: Code-level constants (Jan 2025)
│   ├── llm.py                # LLM and embedding integrations
│   ├── prompt.py             # Extraction prompt templates
│   ├── utils.py              # Utilities (caching, encoding, retry, logging)
│   └── kg/                   # Optional storage backends
│       ├── graph_impl/       # Neo4J, Oracle, MongoDB
│       └── vectordb_impl/    # Milvus, ChromaDB, TiDB
│
├── agent/                     # Tool-augmented generation system
│   ├── llm_agent/
│   │   └── generation.py     # ToolGenerationManager (tool loop)
│   └── tool/
│       ├── tool_env.py       # ToolEnv (execution environment)
│       └── tools/
│           └── search_tool.py # SearchTool implementation
│
├── script_build.py            # Graph construction pipeline
├── script_api.py              # FastAPI retrieval server
├── script_process.py          # Dataset preprocessing
│
└── datasets/                  # Data storage
    └── {dataset_name}/
        ├── raw/              # Corpus and QA pairs
        └── processed/        # Parquet files for training
```

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Data Preprocessing                                     │
├─────────────────────────────────────────────────────────────────┤
│  Input:  datasets/{name}/raw/corpus.jsonl                       │
│          datasets/{name}/raw/qa_*.json                          │
│  Script: script_process.py                                      │
│  Output: datasets/{name}/processed/*.parquet                    │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Graph Construction                                     │
├─────────────────────────────────────────────────────────────────┤
│  Input:  datasets/{name}/raw/corpus.jsonl                       │
│  Script: script_build.py                                        │
│  Process: 1. Chunk documents (1200 tokens, 100 overlap)         │
│           2. Extract entities with multi-turn gleaning          │
│           3. Build bipartite graph                              │
│           4. Generate embeddings (text-embedding-3-large)       │
│           5. Create FAISS indices                               │
│  Output: expr/{name}/kv_store_*.json, index*.bin               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Retrieval Server                                       │
├─────────────────────────────────────────────────────────────────┤
│  Input:  expr/{name}/ (graph files)                             │
│  Script: script_api.py                                          │
│  Service: FastAPI on port 8001                                  │
│  Endpoint: POST /search → returns relevant context              │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: RL Training (uses external verl library)              │
├─────────────────────────────────────────────────────────────────┤
│  • LLM generates responses with <query> tags                    │
│  • SearchTool calls retrieval server                            │
│  • Context injected as <knowledge> tags                         │
│  • Reward based on final answer quality                         │
│  • GRPO updates policy to improve querying                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## BiGRAG Core Library

### Main Class: `BiGRAG` (bigrag/bigrag.py)

**File Location:** `bigrag/bigrag.py` (552 lines)

**Class Definition:**
```python
@dataclass
class BiGRAG:
    # Core configuration
    working_dir: str = "./expr"
    enable_llm_cache: bool = True

    # Chunking parameters
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100

    # Entity extraction parameters
    entity_extract_max_gleaning: int = 2
    entity_summary_to_max_tokens: int = 500

    # LLM and embedding functions
    embedding_func: EmbeddingFunc = None
    llm_model_func: callable = None

    # Storage backend configurations
    graph_storage: str = "NetworkXStorage"
    vector_storage: str = "NanoVectorDBStorage"
    kv_storage: str = "JsonKVStorage"
```

**Key Methods:**

#### `async ainsert(docs: str | list[str], metadata: dict | list[dict] = None)`
**Purpose:** Insert documents into the knowledge graph with metadata preservation

**Process:**
1. Generate unique document IDs (MD5 hash of content)
2. Filter out already-processed documents via `kv_storage.filter_keys()`
3. Chunk documents using `chunking_by_token_size()` **✨ with metadata preservation**
4. Extract entities with multi-turn gleaning: `extract_entities()` **✨ using document context**
5. Merge duplicate entities and build bipartite graph
6. Generate embeddings for entities, relations, and chunks
7. Upsert to vector databases and graph storage
8. Call `index_done_callback()` to persist data

**Input Format:**
```python
# Single document
rag.insert("Text content", metadata={"title": "My Doc", "category": "science"})

# Multiple documents
rag.insert(
    ["Doc 1", "Doc 2"],
    metadata=[
        {"title": "First", "tags": ["ai"]},
        {"title": "Second", "tags": ["ml"]}
    ]
)
```

**✨ Metadata Benefits:**
- Improves entity extraction accuracy (+2-3 F1 points)
- Provides document context to LLM during extraction
- Enables filtering and categorization
- Preserves traceability from chunks → documents

**Cost Optimization:**
- Uses `llm_response_cache` (HashingKV) to avoid redundant API calls
- Batches embedding generation (32 items per batch)
- Incremental construction (skips already-processed docs)

#### `async aquery(query: str, param: QueryParam) -> str`
**Purpose:** Query the knowledge graph and return formatted context

**Process:**
1. Embed query using `embedding_func`
2. Execute `kg_query()` based on `param.mode`:
   - `local`: Entity-based retrieval (Path A only)
   - `global`: Relation-based retrieval (Path B only)
   - `hybrid`: **✨ Three-path retrieval (Path A + B + C)** - default
   - `naive`: Direct chunk similarity (Path C only)
3. **✨ Optional semantic reranking** (if `param.enable_reranking=True`)
4. Format results as natural language
5. Truncate to `max_token_for_text_unit`

**Returns:** Formatted string with retrieved context

**✨ Three-Path Architecture:**
```
Query → Path A (Entities)  → top-60 entities → RRF
     → Path B (Relations) → top-60 edges   → RRF → top-5 structured
     → Path C (Chunks)    → 10 candidates  → rerank → top-5 chunks

Output: 5 structured + 5 chunks = 10 total context items
```

**Performance Impact:**
- With reranking: +10-20% precision, ~50-100ms latency
- Without reranking: Faster, still returns 10 items (5+5)

#### `async adelete_document(doc_id: str) -> dict`
**Purpose:** ✨ NEW - Delete a document and cascade cleanup of all associated data

**Process:**
1. Find all chunks belonging to the document
2. Identify entities/edges referencing those chunks
3. Smart deletion logic:
   - **Full delete**: Entities/edges unique to this document
   - **Partial update**: Remove this doc's chunks from shared entities/edges
4. Delete chunks from `text_chunks` and `vdb_chunks`
5. Delete document from `full_docs`
6. Persist changes

**Returns:** Deletion statistics
```python
{
    "status": "success",
    "doc_id": "doc-abc123",
    "chunks_deleted": 15,
    "entities_deleted": 3,     # Unique to this doc
    "entities_updated": 8,     # Shared with other docs
    "edges_deleted": 5,
    "edges_updated": 12
}
```

**Usage:**
```python
# Delete by document ID
stats = rag.delete_document("doc-abc123")

# Delete by content (computes ID automatically)
stats = rag.delete_document("The original document text...")
```

### Storage Component Initialization

**Seven Storage Instances:**
```python
# 1. LLM cache (cost optimization)
self.llm_response_cache = JsonKVStorage(
    namespace="llm_response_cache",
    global_config={"embedding_func": self.embedding_func}
)

# 2. Full documents (original texts)
self.full_docs = JsonKVStorage(namespace="full_docs")

# 3. Text chunks (after chunking)
self.text_chunks = JsonKVStorage(namespace="text_chunks")

# 4. Bipartite graph (NetworkX)
self.chunk_entity_relation_graph = NetworkXStorage(
    namespace="chunk_entity_relation",
    global_config={"embedding_func": self.embedding_func}
)

# 5. Entity embeddings (FAISS via NanoVectorDB)
self.vdb_entities = NanoVectorDBStorage(
    namespace="entities",
    global_config={
        "embedding_func": self.embedding_func,
        "embedding_batch_num": 32
    }
)

# 6. Bipartite edge embeddings
self.vdb_bipartite_edges = NanoVectorDBStorage(namespace="bipartite_edges")

# 7. Chunk embeddings (for naive mode)
self.vdb_chunks = NanoVectorDBStorage(namespace="chunks")
```

### Lazy Backend Loading

**Function:** `lazy_external_import()` (lines 64-107)

**Purpose:** Load optional storage backends only when requested

**Supported Backends:**
```python
# Graph storage
"Neo4JStorage" → bigrag.kg.graph_impl.neo4j_impl
"OracleGraphStorage" → bigrag.kg.graph_impl.oracle_impl
"MongoGraphStorage" → bigrag.kg.graph_impl.mongo_impl

# Vector storage
"MilvusVectorDBStorage" → bigrag.kg.vectordb_impl.milvus_impl
"ChromaVectorDBStorage" → bigrag.kg.vectordb_impl.chroma_impl
"OracleVectorDBStorage" → bigrag.kg.vectordb_impl.oracle_impl
"TiDBVectorStorage" → bigrag.kg.vectordb_impl.tidb_impl

# KV storage
"OracleKVStorage" → bigrag.kg.kv_impl.oracle_impl
"MongoKVStorage" → bigrag.kg.kv_impl.mongo_impl
"TiDBKVStorage" → bigrag.kg.kv_impl.tidb_impl
```

**Usage Pattern:**
```python
# Specify backend in configuration
rag = BiGRAG(
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorage",
    kv_storage="MongoKVStorage"
)
```

**Benefits:**
- No need to install all dependencies (Neo4J, Milvus, etc.)
- Graceful degradation if import fails
- Easy to add new backends without modifying core code

---

## Storage System

### Abstract Base Classes (bigrag/base.py)

**File Location:** `bigrag/base.py` (132 lines)

All storage backends must implement these abstract interfaces.

#### `BaseVectorStorage`

**Required Methods:**
```python
@abstractmethod
async def query(self, query: str, top_k: int) -> list[dict]:
    """
    Semantic similarity search

    Args:
        query: Search query string
        top_k: Number of results to return

    Returns:
        List of dicts with keys: id, distance, __vector__, metadata
    """
    pass

@abstractmethod
async def upsert(self, data: dict[str, dict]):
    """
    Insert or update vectors

    Args:
        data: {id: {metadata fields + __vector__: embedding}}
    """
    pass

@abstractmethod
async def index_done_callback(self):
    """Called after all upserts to finalize indexing"""
    pass

@property
@abstractmethod
def embedding_dim(self) -> int:
    """Return vector dimensionality"""
    pass
```

#### `BaseKVStorage`

**Required Methods:**
```python
@abstractmethod
async def get_by_id(self, id: str):
    """Retrieve single item by ID"""
    pass

@abstractmethod
async def get_by_ids(self, ids: list[str], fields: set[str] | None = None):
    """Batch retrieval with optional field filtering"""
    pass

@abstractmethod
async def filter_keys(self, data: list[str]) -> set[str]:
    """
    Return IDs that DON'T exist in storage

    Used for incremental construction to skip processed docs
    """
    pass

@abstractmethod
async def upsert(self, data: dict[str, dict]):
    """Insert or update key-value pairs"""
    pass

@abstractmethod
async def drop(self):
    """Delete all data in namespace"""
    pass
```

#### `BaseGraphStorage`

**Required Methods:**
```python
@abstractmethod
async def has_node(self, node_id: str) -> bool:
    """Check if node exists"""
    pass

@abstractmethod
async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
    """Check if edge exists"""
    pass

@abstractmethod
async def get_node(self, node_id: str) -> dict | None:
    """Retrieve node data"""
    pass

@abstractmethod
async def node_degree(self, node_id: str) -> int:
    """Count edges connected to node"""
    pass

@abstractmethod
async def edge_degree(self, src_id: str, tgt_id: str) -> int:
    """Count edges between two nodes (for multigraph)"""
    pass

@abstractmethod
async def get_edge(self, source_node_id: str, target_node_id: str) -> dict | None:
    """Retrieve edge data"""
    pass

@abstractmethod
async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]]:
    """Get all edges connected to a node"""
    pass

@abstractmethod
async def upsert_node(self, node_id: str, node_data: dict[str, str]):
    """Insert or update node"""
    pass

@abstractmethod
async def upsert_edge(
    self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
):
    """Insert or update edge"""
    pass

@abstractmethod
async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
    """
    Generate structural embeddings (Node2Vec, etc.)

    Returns: (embeddings_matrix, node_ids)

    Note: Currently unused in BiG-RAG (semantic embeddings preferred)
    """
    pass
```

### Default Implementations (bigrag/storage.py)

**File Location:** `bigrag/storage.py` (318 lines)

#### `JsonKVStorage`

**Implementation:**
- In-memory `dict` for fast access
- JSON file persistence on `index_done_callback()`
- Thread-safe with async operations

**File Format:**
```json
{
  "doc_id_1": {
    "content": "...",
    "metadata": {...}
  }
}
```

**Key Features:**
- `filter_keys()`: Returns set difference (IDs not in storage)
- `get_by_ids()`: Optionally filters fields to reduce memory
- Namespace-based file separation (`{namespace}.json`)

#### `NanoVectorDBStorage`

**Implementation:**
- Built on `nano-vectordb` library (in-memory FAISS)
- Cosine similarity search with configurable threshold (default 0.2)
- Automatic embedding dimension detection

**Storage Format:**
```json
{
  "__id__": "entity_1",
  "__vector__": [0.1, 0.2, ..., 0.5],  # Embedding
  "entity_name": "Paris",
  "entity_type": "geo",
  "description": "Capital of France"
}
```

**Batch Embedding:**
```python
async def _embed_batch(self, texts: list[str]) -> np.ndarray:
    """Generate embeddings in batches of 32"""
    return await self.global_config["embedding_func"](texts)
```

**Query Process:**
1. Embed query text
2. Cosine similarity search via NanoVectorDB
3. Filter by threshold (default 0.2)
4. Return top-k results with metadata

#### `NetworkXStorage`

**Implementation:**
- Uses NetworkX `Graph` (undirected)
- GraphML serialization for persistence
- Graph stabilization on callback

**Graph Stabilization** (`_stabilize_graph()`):
```python
def _stabilize_graph(self):
    # 1. Extract largest connected component
    largest_component = max(
        nx.connected_components(self._graph),
        key=len
    )
    self._graph = self._graph.subgraph(largest_component).copy()

    # 2. Node name normalization
    for node in self._graph.nodes():
        # Uppercase first letter
        # Remove HTML entity codes
        normalized = node.upper() if len(node) > 0 else node
        # Update in place

    # 3. Sort edges deterministically
    self._graph = nx.Graph(sorted(self._graph.edges()))
```

**Node Embedding** (unused feature):
```python
async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
    """
    Structural embedding via Node2Vec

    Parameters from global_config:
    - dimensions: 128
    - num_walks: 10
    - walk_length: 80
    - window_size: 10
    - iterations: 3
    """
    from graspologic.embed import node2vec_embed
    # ... implementation
```

**Why unused?** Semantic embeddings from transformer models (text-embedding-3-large) provide better retrieval than structural embeddings for knowledge graphs.

### Query Parameter Configuration

**Class:** `QueryParam` (bigrag/base.py, lines 19-31)

```python
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"

    # Vector search parameters
    top_k: int = 60

    # Token limits for context formatting
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000

    # ✨ Semantic reranking toggle
    enable_reranking: bool = True
```

**Mode Descriptions:**
- `local`: Entity-based retrieval (Path A only: query → entities → relations → descriptions)
- `global`: Relation-based retrieval (Path B only: query → relations → entities → descriptions)
- `hybrid`: Combined three-path retrieval with unified ranking (Path A + B + C, default, best performance)
- `naive`: Direct text chunk similarity (Path C only, baseline for comparison)

---

## Entity Extraction Pipeline

### Text Chunking (bigrag/operate.py)

**Function:** `chunking_by_token_size()` (lines 46-118)

**Algorithm:**
```python
def chunking_by_token_size(
    content: str,
    max_token_size: int = 1200,
    overlap_token_size: int = 100
) -> list[dict]:
    """
    Token-aware chunking with overlap

    Process:
    1. Encode full text with tiktoken
    2. Create sliding windows of max_token_size
    3. Add overlap_token_size at boundaries
    4. Decode back to text
    5. Track chunk order for reconstruction

    Returns:
        [
            {
                "content": chunk_text,
                "tokens": token_count,
                "chunk_order_index": position
            }
        ]
    """
```

**Why Token-Based?**
- Respects model's token limits (important for LLM context windows)
- More accurate than character-based splitting
- Ensures chunks fit within embedding model limits

**Overlap Strategy:**
- Last 100 tokens of chunk N = first 100 tokens of chunk N+1
- Preserves context at boundaries
- Improves entity extraction accuracy for split sentences

### Entity Extraction with Multi-Turn Gleaning (bigrag/operate.py)

**Function:** `extract_entities()` (lines 122-273)

**Core Implementation:**

```python
async def extract_entities(
    chunks: list[dict],
    entity_types: list[str],
    llm_model_func: callable,
    llm_response_cache: BaseKVStorage
) -> list[dict]:
    """
    Multi-turn entity extraction with gleaning

    Process for each chunk:
    1. Initial extraction with prompt template
    2. Check if LLM response in cache (MD5 hash)
    3. Parse entities from LLM response
    4. Gleaning loop (max 2 iterations):
       a. Ask "What entities did I miss?"
       b. Parse additional entities
       c. Check if completeness achieved
       d. Continue or stop
    5. Return combined entities + bipartite edges
    """
```

**Prompt Engineering:**

**Templates:** Defined in `bigrag/prompt.py`

```python
# Initial extraction
PROMPTS["entity_extraction"] = """
-Goal-
Given a text document, identify entities and relationships.

-Steps-
1. Identify entities: {entity_types}
2. Format as tuples: (entity_name, entity_type, entity_description)
3. Record as: entity_name<|>entity_type<|>entity_description
4. Multiple entities: separate with ##
5. Complete marker: <|COMPLETE|>

-Examples-
{examples}

-Real Data-
######################
Text: {input_text}
######################
Output:
"""

# Gleaning prompt
PROMPTS["entity_extraction_continue"] = """
Previously extracted:
{previous_entities}

Did you miss any entities? If yes, provide them in the same format.
If no more entities, output <|COMPLETE|>
"""
```

**Delimiter System:**
- `<|>`: Separates tuple elements
- `##`: Separates multiple records
- `<|COMPLETE|>`: Signals completion

**Entity Types** (default):
```python
["organization", "person", "geo", "event", "category"]
```

**Extracted Data Structure:**

Two types of nodes extracted:

1. **Entity Nodes:**
```python
{
    "entity_name": "Paris",
    "entity_type": "geo",
    "entity_description": "Capital city of France",
    "weight": 85,  # Importance score 0-100
    "source_id": "chunk_abc123"
}
```

2. **Bipartite Edge Nodes** (from "hyper-relation" outputs):
```python
{
    "relation": "Paris<|>capital_of<|>France",
    "weight": 90,
    "source_id": "chunk_abc123",
    "completeness": 95  # LLM-assessed completeness
}
```

**Caching Strategy:**

```python
# Cache key = MD5(prompt + system_prompt + kwargs)
cache_key = hash_func({
    "prompt": extraction_prompt,
    "system_prompt": system_prompt,
    "kwargs": json.dumps(llm_kwargs)
})

# Check cache before API call
cached_response = await llm_response_cache.get_by_id(cache_key)
if cached_response:
    return cached_response

# Call LLM and cache result
response = await llm_model_func(prompt, **kwargs)
await llm_response_cache.upsert({cache_key: response})
```

**Cost Impact:**
- Reduces API calls by ~60-70% during graph construction
- Especially effective for overlapping chunks with similar content
- Transparent to caller (no API changes)

### Bipartite Graph Construction (bigrag/operate.py)

**📖 For detailed bipartite architecture explanation, see [BIPARTITE_ARCHITECTURE_EXPLAINED.md](BIPARTITE_ARCHITECTURE_EXPLAINED.md)**

**Function:** `_merge_nodes_then_upsert()` (lines 397-518)

**Node Merging Strategy:**

```python
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict]
) -> dict:
    """
    Merge duplicate entities across chunks

    Process:
    1. Collect all entity types for this name
    2. Use most frequent type (mode)
    3. Concatenate descriptions with <SEP>
    4. If combined description > 500 tokens:
       - Call LLM to summarize
       - Cache summary for reuse
    5. Sum weights across occurrences
    6. Collect all source chunk IDs

    Returns unified node data
    """
```

**Type Resolution:**
```python
# Example: "Paris" appears as both "geo" and "location"
types = ["geo", "geo", "location", "geo"]
most_common_type = Counter(types).most_common(1)[0][0]
# Result: "geo"
```

**Description Merging:**
```python
# Concatenate with separator
descriptions = [
    "Capital of France",
    "Largest city in France",
    "Located on the Seine River"
]
combined = "<SEP>".join(descriptions)

# Summarize if too long
if token_count(combined) > 500:
    summary = await llm_summarize(combined)
    final_description = summary
else:
    final_description = combined
```

**Weight Aggregation:**
```python
# Cumulative importance score
weights = [85, 90, 78]
total_weight = sum(weights)  # 253
```

**Source Tracking:**
```python
# De-duplicated set of chunk references
source_ids = ["chunk_1", "chunk_1", "chunk_2"]
unique_sources = list(set(source_ids))  # ["chunk_1", "chunk_2"]
```

**Function:** `_merge_bipartite_edges_then_upsert()` (lines 520-612)

**Bipartite Edge Creation:**

```python
async def _merge_bipartite_edges_then_upsert(
    relation_content: str,
    edges_data: list[dict]
) -> dict:
    """
    Create relation nodes from LLM-extracted "hyper-relations"

    Process:
    1. Assign unique ID to relation content
    2. Sum weights (importance scores)
    3. Collect source chunk IDs
    4. Create node in graph with role="bipartite_edge"

    Note: These are NODES in the bipartite graph, not edges!
    """
```

**Bipartite Edge Node Structure:**
```python
{
    "id": "edge_xyz789",
    "content": "Paris is the capital of France",
    "weight": 180,  # Cumulative
    "source_id": ["chunk_1", "chunk_3"],
    "role": "bipartite_edge"  # Distinguishes from entity nodes
}
```

**Function:** `_merge_edges_then_upsert()` (lines 614-697)

**Edge Creation Between Nodes:**

```python
async def _merge_edges_then_upsert(
    entity_name: str,
    edges_data: list[dict]
) -> None:
    """
    Create edges connecting bipartite_edge nodes to entity nodes

    Process:
    1. For each relation involving entity_name:
       a. Create edge: bipartite_edge_node ↔ entity_node
       b. Store metadata: weight, source_id
       c. Undirected edges (NetworkX Graph)

    Result: True bipartite graph structure
    """
```

**Final Graph Structure:**

```
┌────────────────────────────────────────────────────────────────┐
│                    Bipartite Graph                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Entity Nodes                 Bipartite Edge Nodes            │
│  (role="entity")              (role="bipartite_edge")         │
│                                                                │
│  ┌──────────────┐             ┌──────────────────────┐        │
│  │  Paris       │◄───────────►│  "Paris is capital   │        │
│  │  type: geo   │             │   of France"         │        │
│  │  weight: 253 │             │  weight: 180         │        │
│  └──────────────┘             └──────────────────────┘        │
│        ▲                               ▲                       │
│        │                               │                       │
│        │                               │                       │
│        ▼                               ▼                       │
│  ┌──────────────┐             ┌──────────────────────┐        │
│  │  France      │◄───────────►│  "France is a        │        │
│  │  type: geo   │             │   European country"  │        │
│  │  weight: 198 │             │  weight: 145         │        │
│  └──────────────┘             └──────────────────────┘        │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Key Properties:
- Two node types: Entity nodes and Bipartite edge nodes
- Edges only connect entity ↔ bipartite_edge (no entity-entity edges)
- Undirected edges (NetworkX Graph)
- Metadata: weight, source_id on both nodes and edges
```

---

## Query and Retrieval System

### Query Execution Flow (bigrag/bigrag.py)

**Function:** `kg_query()` (lines 333-425)

**High-Level Process:**

```python
async def kg_query(
    query: str,
    param: QueryParam
) -> list[dict]:
    """
    Execute knowledge graph query

    Modes:
    - local: Entity-based retrieval
    - global: Relation-based retrieval
    - hybrid: Combined three-path (default)
    - naive: Direct chunk similarity

    Returns: List of context dicts with content and relevance scores
    """
```

**Hybrid Mode Implementation** (lines 372-401):

```python
# 1. Embed query three times (for entity, edge, and chunk indices)
query_embedding = await embedding_func([query, query, query])

# 2. Entity-based retrieval (Path A)
entity_results = await _get_node_data(
    query_embedding[0],
    top_k=param.top_k
)

# 3. Relation-based retrieval (Path B)
relation_results = await _get_edge_data(
    query_embedding[1],
    top_k=param.top_k
)

# 4. Chunk-based retrieval (Path C)
chunk_results = await vdb_chunks.query(
    query_embedding[2],
    top_k=10  # Get candidates for reranking
)

# 5. Optional semantic reranking
if param.enable_reranking:
    chunk_results = await rerank_chunks(chunk_results, query)

# 6. Combine with reciprocal rank scoring
combined = _merge_and_rank(entity_results, relation_results, chunk_results)

# 7. Format and return top-k
return combined[:param.top_k]
```

### Entity-Based Retrieval (bigrag/operate.py)

**Function:** `_get_node_data()` (lines 816-891)

**Process:**

```python
async def _get_node_data(
    query_embedding: np.ndarray,
    top_k: int
) -> list[dict]:
    """
    Entity-based retrieval path

    Steps:
    1. Vector search in vdb_entities
       → Get top-k entity nodes by semantic similarity

    2. For each entity node:
       a. Retrieve node data (type, description, weight)
       b. Get connected edges via graph.get_node_edges()
       c. Retrieve edge node data (relation content)

    3. Rank edges by:
       - Node degree (how many entities connected)
       - Weight (importance score)

    4. Return formatted results with:
       - Entity descriptions
       - Top-N related relations
    """
```

**Ranking Function:**

```python
def _find_most_related_edges_from_entities(
    entity_nodes: list[str]
) -> list[str]:
    """
    Find and rank edges connected to entities

    Ranking criteria:
    1. Node degree (descending) - more connections = more important
    2. Weight (descending) - higher importance score

    Returns: Sorted list of edge node IDs
    """
    edge_scores = {}
    for entity in entity_nodes:
        edges = await graph.get_node_edges(entity)
        for edge in edges:
            degree = await graph.node_degree(edge)
            weight = edge_data["weight"]
            edge_scores[edge] = (degree, weight)

    # Sort by (degree, weight) tuple - lexicographic ordering
    sorted_edges = sorted(
        edge_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [edge_id for edge_id, _ in sorted_edges]
```

### Relation-Based Retrieval (bigrag/operate.py)

**Function:** `_get_edge_data()` (lines 893-968)

**Process:**

```python
async def _get_edge_data(
    query_embedding: np.ndarray,
    top_k: int
) -> list[dict]:
    """
    Relation-based retrieval path

    Steps:
    1. Vector search in vdb_bipartite_edges
       → Get top-k relation nodes by semantic similarity

    2. For each relation node:
       a. Retrieve node data (content, weight)
       b. Calculate node degree (connectivity)

    3. Sort relations by:
       - Weight (descending)
       - Node degree (descending)

    4. Get connected entity nodes

    5. Return formatted results with:
       - Relation descriptions
       - Connected entity names
    """
```

**Why This Works:**
- Relations capture multi-entity contexts
- High-weight relations are informationally dense
- High-degree relations connect many relevant entities
- Provides complementary information to entity-based retrieval

### Hybrid Ranking (bigrag/operate.py)

**Function:** `_merge_and_rank()` (lines 970-1045)

**Reciprocal Rank Fusion:**

```python
def _merge_and_rank(
    entity_results: list[dict],
    relation_results: list[dict],
    chunk_results: list[dict]
) -> list[dict]:
    """
    Combine three-path results with unified scoring

    Reciprocal Rank Formula:
        score = sum(1 / (rank + 1) for all occurrences)

    Example:
        Item appears at rank 3 in entity results: 1/4 = 0.25
        Item appears at rank 7 in relation results: 1/8 = 0.125
        Item appears at rank 1 in chunk results: 1/2 = 0.5
        Combined score: 0.25 + 0.125 + 0.5 = 0.875

    Process:
    1. Assign ranks to entity results (0, 1, 2, ...)
    2. Assign ranks to relation results (0, 1, 2, ...)
    3. Assign ranks to chunk results (0, 1, 2, ...)
    4. Calculate reciprocal rank score for each item
    5. Merge and deduplicate by content hash
    6. Sort by combined score (descending)

    Returns: Unified ranked list
    """
```

**Why Reciprocal Rank?**
- Gives more weight to top-ranked items (1/1=1.0 vs 1/100=0.01)
- Naturally combines rankings from different sources
- No hyperparameter tuning needed
- Robust to different scale distributions

**Deduplication:**
```python
# Use MD5 hash of content for deduplication
seen = set()
deduplicated = []
for item in combined:
    content_hash = hashlib.md5(item["content"].encode()).hexdigest()
    if content_hash not in seen:
        seen.add(content_hash)
        deduplicated.append(item)
```

### Naive Mode (Baseline) (bigrag/bigrag.py)

**Implementation:**

```python
async def _naive_query(
    query: str,
    top_k: int
) -> list[dict]:
    """
    Direct chunk similarity search

    Process:
    1. Query vdb_chunks with query embedding
    2. Return top-k chunks by cosine similarity
    3. No graph traversal

    Purpose: Baseline for comparison
    Performance: Usually worse than hybrid mode
    """
    chunk_results = await self.vdb_chunks.query(query, top_k)
    return [
        {
            "content": chunk["content"],
            "score": chunk["distance"]
        }
        for chunk in chunk_results
    ]
```

**Why Hybrid Outperforms Naive:**
- Naive: Single-step similarity (query → chunks)
- Hybrid: Multi-hop reasoning (query → entities/relations → descriptions)
- Graph structure provides semantic context
- Entity/relation abstraction reduces noise

---

## Tool-Augmented Generation

### ToolEnv: Execution Environment (agent/tool/tool_env.py)

**File Location:** `agent/tool/tool_env.py` (446 lines)

**Purpose:** Manages tool execution state and orchestrates tool calls during LLM generation

**Key Data Structures:**

```python
@dataclass
class ToolEnv:
    # Configuration
    config: ToolEnvConfig
    tools: dict[str, Tool]  # {tool_name: Tool instance}

    # State tracking
    reward: float = 0.0
    tool_history: list[dict] = []  # [{tool, args, result}]
    steps_taken: int = 0

    # Action tracking for analysis
    _actions: list[str] = []  # All LLM responses
    _actions_valid: list[bool] = []  # Correctly formatted?
    _actions_effective: list[bool] = []  # Successfully executed?

    # Question and ground truth
    question: str = ""
    answers: list[str] = []
```

**Configuration:**

```python
@dataclass
class ToolEnvConfig:
    max_turns: int = 5
    max_prompt_length: int = 4096
    max_response_length: int = 4096
    max_tool_response_length: int = 1000

    # Tool call markers
    tool_call_start: str = "<query>"
    tool_call_end: str = "</query>"
    tool_response_start: str = "<knowledge>"
    tool_response_end: str = "</knowledge>"

    # Answer markers
    answer_start: str = "<answer>"
    answer_end: str = "</answer>"
```

**Core Methods:**

#### `step(env: ToolEnv, action_text: str) -> tuple`

**Purpose:** Execute one environment step (single sequence)

**Process:**

```python
def step(env: ToolEnv, action_text: str) -> tuple:
    """
    Execute one tool interaction step

    Args:
        env: ToolEnv instance
        action_text: LLM-generated text (may contain <query> tags)

    Returns:
        (observation, reward, done, info)

    observation: Tool response text or error message
    reward: Reward for this step (usually 0.0 for intermediate)
    done: True if answer generated or max_turns reached
    info: Metadata dict
    """

    # 1. Track action
    env._actions.append(action_text)

    # 2. Extract tool call
    tool_call = extract_tool_call(
        action_text,
        env.config.tool_call_start,
        env.config.tool_call_end
    )

    # 3. Validate format
    if not tool_call or "tool" not in tool_call or "args" not in tool_call:
        env._actions_valid.append(False)
        env._actions_effective.append(False)
        return ("Invalid tool call format", 0.0, False, {})

    env._actions_valid.append(True)

    # 4. Check if tool exists
    tool_name = tool_call["tool"]
    if tool_name not in env.tools:
        env._actions_effective.append(False)
        return (f"Tool {tool_name} not found", 0.0, False, {})

    # 5. Execute tool
    tool = env.tools[tool_name]
    try:
        result = tool.execute(tool_call["args"])
        env._actions_effective.append(True)
    except Exception as e:
        env._actions_effective.append(False)
        return (f"Tool execution failed: {e}", 0.0, False, {})

    # 6. Calculate reward (tool-specific)
    step_reward = tool.calculate_reward(tool_call["args"], result)
    env.reward += step_reward

    # 7. Update tracking
    env.steps_taken += 1
    env.tool_history.append({
        "tool": tool_name,
        "args": tool_call["args"],
        "result": result
    })

    # 8. Check termination
    done = (
        env.steps_taken >= env.config.max_turns or
        env.config.answer_end in action_text
    )

    # 9. Format observation
    observation = f"{env.config.tool_response_start}{result}{env.config.tool_response_end}"

    return (observation, step_reward, done, {"tool": tool_name})
```

#### `step_batch(envs: list[ToolEnv], actions: list[str]) -> tuple`

**Purpose:** Batch execution for parallel environments

**Process:**

```python
def step_batch(
    envs: list[ToolEnv],
    actions: list[str]
) -> tuple[list[str], list[float], list[bool], list[dict]]:
    """
    Batch tool execution with grouping optimization

    Process:
    1. Extract tool calls from all actions
    2. Group by tool name (search tool, calculator, etc.)
    3. Execute each tool's batch_execute() once per group
    4. Map results back to original indices
    5. Return parallel lists of observations, rewards, dones, infos

    Optimization: Batch execution is faster than sequential
    """

    # 1. Parse all tool calls
    tool_calls = [
        extract_tool_call(action, envs[i].config.tool_call_start, envs[i].config.tool_call_end)
        for i, action in enumerate(actions)
    ]

    # 2. Group by tool name
    tool_groups = defaultdict(list)
    for idx, call in enumerate(tool_calls):
        if call and "tool" in call:
            tool_name = call["tool"]
            tool_groups[tool_name].append((idx, call["args"]))

    # 3. Batch execute each tool
    results = {}
    for tool_name, calls in tool_groups.items():
        indices = [idx for idx, _ in calls]
        args_list = [args for _, args in calls]

        tool = envs[0].tools[tool_name]  # Same tools for all envs
        batch_results = tool.batch_execute(args_list)

        for idx, result in zip(indices, batch_results):
            results[idx] = result

    # 4. Reconstruct parallel lists
    observations = []
    rewards = []
    dones = []
    infos = []

    for idx, env in enumerate(envs):
        if idx in results:
            obs = f"{env.config.tool_response_start}{results[idx]}{env.config.tool_response_end}"
            reward = 0.0  # Tool-specific reward calculation
            done = env.steps_taken >= env.config.max_turns
            info = {"tool": tool_calls[idx]["tool"]}
        else:
            # Invalid or failed call
            obs = "Invalid tool call"
            reward = 0.0
            done = False
            info = {}

        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)

    return observations, rewards, dones, infos
```

**Tool Call Extraction:**

```python
def extract_tool_call(
    text: str,
    start_tag: str,
    end_tag: str
) -> dict | None:
    """
    Parse tool call from LLM output

    Expected format:
        <query>{"tool": "search", "args": {"query": "..."}}</query>

    Returns:
        {"tool": "search", "args": {...}} or None if invalid
    """
    pattern = f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    content = match.group(1).strip()

    try:
        parsed = json.loads(content)
        if "tool" in parsed and "args" in parsed:
            return parsed
    except json.JSONDecodeError:
        return None

    return None
```

### ToolGenerationManager: Generation Loop (agent/llm_agent/generation.py)

**File Location:** `agent/llm_agent/generation.py` (350+ lines)

**Purpose:** Orchestrates tool-augmented generation with iterative retrieval

**Configuration:**

```python
@dataclass
class ToolGenerationConfig:
    max_turns: int
    max_prompt_length: int
    max_response_length: int
    max_tool_response_length: int

    tool_call_start: str = "<query>"
    tool_call_end: str = "</query>"
    tool_response_start: str = "<knowledge>"
    tool_response_end: str = "</knowledge>"
```

**Main Loop:** `run_llm_loop()` (lines 266-330)

**Process:**

```python
class ToolGenerationManager:
    def run_llm_loop(
        self,
        gen_batch,
        envs: list[ToolEnv],
        initial_input_ids: torch.Tensor
    ) -> dict:
        """
        Iterative tool-augmented generation loop

        Process:
        1. Initialize rolling state (input_ids)
        2. Loop up to max_turns:
           a. Generate sequences with vLLM
           b. Postprocess: extract first <query>...</query>
           c. Execute tool calls (batch or sequential)
           d. Update rolling state with tool responses
           e. Track active sequences (mask out invalid calls)
           f. Check termination (all done or max_turns)
        3. Return final sequences and metadata

        Key Feature: Active masking
        - Sequences with invalid tool calls stop generating
        - Only "active" sequences continue to next turn
        - Saves computation and improves training stability
        """

        # Initialize state
        rollings = [initial_input_ids.clone() for _ in range(len(envs))]
        active_mask = torch.ones(len(envs), dtype=torch.bool)

        for turn in range(self.config.max_turns):
            # 1. Filter active sequences
            rollings_active = [r for i, r in enumerate(rollings) if active_mask[i]]
            envs_active = [e for i, e in enumerate(envs) if active_mask[i]]

            # 2. Generate with GPU padding
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # 3. Postprocess responses (extract only first tool call)
            responses_ids, responses_str, new_active_masks = self._postprocess_responses(
                gen_output,
                envs_active
            )

            # 4. Execute tool calls
            if self.use_batch_tool_calls:
                tool_responses = self._execute_tool_calls_batch(
                    responses_str,
                    envs_active,
                    new_active_masks
                )
            else:
                tool_responses = self._execute_tool_calls(
                    responses_str,
                    envs_active,
                    new_active_masks
                )

            # 5. Update rolling state
            rollings_active = self._update_rolling_state(
                rollings_active,
                responses_ids,
                tool_responses
            )

            # 6. Update active mask
            active_mask = self._merge_active_masks(
                active_mask,
                new_active_masks
            )

            # 7. Check termination
            if not active_mask.any():
                break

        # 8. Return final sequences
        return {
            "input_ids": rollings,
            "active_mask": active_mask,
            "turns": [env.steps_taken for env in envs]
        }
```

**GPU Padding:** `_generate_with_gpu_padding()` (lines 219-264)

**Purpose:** Handle batch sizes not divisible by GPU count

```python
def _generate_with_gpu_padding(
    self,
    input_ids: list[torch.Tensor]
) -> dict:
    """
    Add padding sequences if needed for multi-GPU generation

    Process:
    1. Check if batch_size % n_gpus == 0
    2. If not, add dummy sequences (copies of last item)
    3. Generate with vLLM
    4. Remove padding from output

    Returns: Generation output for original batch only
    """
    batch_size = len(input_ids)
    n_gpus = self.n_gpus

    # Calculate padding needed
    if batch_size % n_gpus != 0:
        padding_size = n_gpus - (batch_size % n_gpus)
        # Add copies of last sequence
        padded_input_ids = input_ids + [input_ids[-1].clone()] * padding_size
    else:
        padded_input_ids = input_ids
        padding_size = 0

    # Generate
    gen_output = self.rollout_worker.generate(padded_input_ids)

    # Remove padding
    if padding_size > 0:
        gen_output["sequences"] = gen_output["sequences"][:-padding_size]

    return gen_output
```

**Postprocessing:** `_postprocess_responses()` (lines 156-195)

**Purpose:** Extract first tool call and validate format

```python
def _postprocess_responses(
    self,
    gen_output: dict,
    envs: list[ToolEnv]
) -> tuple:
    """
    Extract tool calls and determine which sequences remain active

    Process:
    1. Decode generated token IDs to text
    2. Extract first <query>...</query> tag
    3. Validate tool call format (JSON parsing)
    4. Mark sequences as active or inactive
    5. Truncate response if needed

    Returns:
        response_ids: Token IDs for responses
        response_str: Decoded text
        active_masks: Boolean mask (True = continue, False = stop)
    """
    responses_ids = []
    responses_str = []
    active_masks = []

    for idx, seq in enumerate(gen_output["sequences"]):
        # Decode
        text = self.tokenizer.decode(seq)

        # Extract first tool call
        pattern = f"{re.escape(self.config.tool_call_start)}(.*?){re.escape(self.config.tool_call_end)}"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            # Valid tool call found
            call_text = match.group(0)  # Include tags
            responses_str.append(call_text)
            responses_ids.append(self.tokenizer.encode(call_text))
            active_masks.append(True)
        else:
            # No valid tool call - stop this sequence
            responses_str.append(text)
            responses_ids.append(seq)
            active_masks.append(False)

    return responses_ids, responses_str, torch.tensor(active_masks)
```

**Active Masking Example:**

```
Turn 1:
  Seq 0: <query>{"tool": "search", "args": {"query": "Paris"}}</query>  → Active
  Seq 1: <query>{"tool": "search", "args": {"query": "London"}}</query> → Active
  Seq 2: I don't know the answer.  → Inactive (no tool call)

Turn 2:
  Seq 0: <query>{"tool": "search", "args": {"query": "France"}}</query> → Active
  Seq 1: Invalid JSON: <query>{broken</query> → Inactive (invalid format)
  Seq 2: (not processed - already inactive)

Turn 3:
  Seq 0: <answer>Paris is the capital of France</answer> → Done
  Seq 1: (not processed - already inactive)
  Seq 2: (not processed - already inactive)

Final result: Only Seq 0 completed successfully
```

### SearchTool Implementation (agent/tool/tools/search_tool.py)

**File Location:** `agent/tool/tools/search_tool.py` (83 lines)

**Class Definition:**

```python
class SearchTool(Tool):
    """
    Knowledge graph retrieval tool

    Makes HTTP requests to retrieval server (port 8001)
    """

    name: str = "search"
    description: str = "Search knowledge graph for relevant information"

    # Configuration
    api_url: str = "http://localhost:8001/search"
    timeout: int = 30
    max_retries: int = 3
```

**Methods:**

#### `execute(args: dict) -> str`

**Implementation:**

```python
def execute(self, args: dict) -> str:
    """
    Single execution (NOT IMPLEMENTED)

    Note: Only batch_execute() is functional
    Sequential execution not supported
    """
    pass  # Stub only
```

**Reason:** Batch execution is more efficient for distributed training. Sequential calls would require maintaining connection state per worker.

#### `batch_execute(args_list: list[dict]) -> list[str]`

**Implementation:**

```python
def batch_execute(self, args_list: list[dict]) -> list[str]:
    """
    Batch retrieval from knowledge graph

    Args:
        args_list: [{"query": "search text"}, ...]

    Returns:
        List of formatted search results (one per query)

    HTTP Request:
        POST http://localhost:8001/search
        Body: {"queries": ["query1", "query2", ...]}

    Response Format:
        [
            [{"<knowledge>": "...", "<coherence>": 0.95}, ...],
            [{"<knowledge>": "...", "<coherence>": 0.88}, ...],
            ...
        ]
    """

    # Extract queries
    queries = [args["query"] for args in args_list]

    # HTTP request
    response = requests.post(
        self.api_url,
        json={"queries": queries},
        timeout=self.timeout
    )
    response.raise_for_status()

    # Parse response
    results = response.json()

    # Format for LLM
    formatted_results = []
    for result_list in results:
        # Concatenate top results
        context_pieces = [
            item["<knowledge>"]
            for item in result_list[:5]  # Top 5 results
        ]
        formatted = "\n\n".join(context_pieces)
        formatted_results.append(formatted)

    return formatted_results
```

**Error Handling:**

```python
# Retry logic
for attempt in range(self.max_retries):
    try:
        response = requests.post(...)
        return results
    except requests.RequestException as e:
        if attempt < self.max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
            continue
        else:
            # Return error message
            return [f"Search failed: {e}"] * len(args_list)
```

#### `calculate_reward(args: dict, result: str) -> float`

**Implementation:**

```python
def calculate_reward(self, args: dict, result: str) -> float:
    """
    Calculate intermediate reward for tool call

    Current implementation: Always return 0.0

    Reason: Reward assigned at final answer only
    Intermediate rewards could be added for:
    - Query diversity
    - Result relevance
    - Coverage improvement
    """
    return 0.0
```

**Note:** All reward comes from final answer evaluation (EM/F1 scores). Tool-specific rewards are not currently used but could be added for finer-grained credit assignment.

---

## Pipeline Scripts

### script_process.py: Dataset Preprocessing

**File Location:** `script_process.py` (lines vary)

**Purpose:** Convert raw QA data to parquet format for training

**Usage:**
```bash
python script_process.py --data_source 2WikiMultiHopQA
```

**Process:**

```python
def main(data_source: str):
    """
    Preprocess dataset to parquet format

    Steps:
    1. Load raw QA files from datasets/{data_source}/raw/
    2. Apply instruction templates
    3. Format as standardized dicts
    4. Save as parquet files
    5. Generate train/dev/test splits
    """

    # 1. Load raw data
    raw_dir = f"datasets/{data_source}/raw"
    train_data = json.load(open(f"{raw_dir}/qa_train.json"))
    dev_data = json.load(open(f"{raw_dir}/qa_dev.json"))
    test_data = json.load(open(f"{raw_dir}/qa_test.json"))

    # 2. Apply templates
    processed_train = [
        format_qa_pair(item, data_source)
        for item in train_data
    ]

    # 3. Convert to DataFrame
    df_train = pd.DataFrame(processed_train)

    # 4. Save as parquet
    output_dir = f"datasets/{data_source}/processed"
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_parquet(f"{output_dir}/train.parquet")
```

**Template Application:**

```python
def format_qa_pair(item: dict, data_source: str) -> dict:
    """
    Apply instruction template to QA pair

    Template format:
        <|im_start|>system
        You are a helpful assistant. Use tools to answer questions.
        <|im_end|>
        <|im_start|>user
        {question}
        <|im_end|>
        <|im_start|>assistant

    Returns:
        {
            "prompt": formatted_template,
            "question": original_question,
            "answers": list_of_answers
        }
    """
    template = get_template(data_source)
    prompt = template.format(question=item["question"])

    return {
        "prompt": prompt,
        "question": item["question"],
        "answers": item["golden_answers"]
    }
```

**Output Format:**

```
datasets/{data_source}/processed/
├── train.parquet     # Training set
├── dev.parquet       # Validation set
└── test.parquet      # Test set

Parquet columns:
- prompt: str (formatted instruction)
- question: str (original question)
- answers: list[str] (ground truth answers)
```

### script_build.py: Graph Construction Pipeline

**File Location:** `script_build.py` (291 lines)

**Purpose:** Build bipartite knowledge graph from corpus

**Usage:**
```bash
python script_build.py --data_source 2WikiMultiHopQA
```

**Configuration:**

```python
# API key loading
def load_api_key():
    """Load OpenAI API key from file or environment"""
    if os.path.exists("openai_api_key.txt"):
        with open("openai_api_key.txt") as f:
            return f.read().strip()
    return os.environ.get("OPENAI_API_KEY")

# Model configuration
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# Chunking parameters
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

# Batch processing
BATCH_SIZE = 5  # Documents per batch
```

**Main Process:**

```python
def main(data_source: str):
    """
    Build knowledge graph from corpus

    Steps:
    1. Initialize BiGRAG instance
    2. Load corpus from datasets/{data_source}/raw/corpus.jsonl
    3. Batch processing with retry logic
    4. Save to expr/{data_source}/
    5. Verify output files
    """

    # 1. Initialize BiGRAG
    working_dir = f"./expr/{data_source}"
    os.makedirs(working_dir, exist_ok=True)

    rag = BiGRAG(
        working_dir=working_dir,
        enable_llm_cache=True,
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
        entity_extract_max_gleaning=2,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding(
            model=EMBEDDING_MODEL,
            api_key=api_key
        )
    )

    # 2. Load corpus
    corpus_path = f"datasets/{data_source}/raw/corpus.jsonl"
    documents = []
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            documents.append({
                "content": doc["contents"],
                "title": doc.get("title", ""),
                "metadata": doc.get("metadata", {})
            })

    # 3. Batch processing
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{len(documents)//BATCH_SIZE + 1}")

        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # This calls ainsert() internally
                rag.insert(batch)
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print(f"Failed to process batch {i}")

    # 4. Verify output
    verify_output_files(working_dir)
```

**Output Files:**

```
expr/{data_source}/
├── kv_store_text_chunks.json          # Chunk metadata
├── kv_store_full_docs.json            # Original documents
├── vdb_entities.json                  # Entity vector DB
├── vdb_bipartite_edges.json           # Relation vector DB
├── graph_chunk_entity_relation.graphml # Bipartite graph
└── llm_response_cache.json            # LLM cache
```

**File Verification:**

```python
def verify_output_files(working_dir: str):
    """Check that all expected files were created"""
    required_files = [
        "kv_store_text_chunks.json",
        "vdb_entities.json",
        "vdb_bipartite_edges.json",
        "graph_chunk_entity_relation.graphml"
    ]

    for filename in required_files:
        filepath = os.path.join(working_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing output file: {filepath}")

        # Check file size
        size = os.path.getsize(filepath)
        print(f"{filename}: {size / 1024 / 1024:.2f} MB")
```

### script_api.py: Retrieval Server

**File Location:** `script_api.py` (300+ lines)

**Purpose:** Expose BiG-RAG retrieval via FastAPI

**Usage:**
```bash
python script_api.py --data_source 2WikiMultiHopQA
```

**Server Configuration:**

```python
# FastAPI app
app = FastAPI(
    title="BiG-RAG Retrieval API",
    version="1.0.0",
    description="Bipartite graph-based retrieval server"
)

# Port and host
PORT = 8001
HOST = "0.0.0.0"  # Allow external connections

# CORS middleware (for web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
```

**Main Components:**

#### EmbeddingManager

**Purpose:** Load FAISS indices and handle retrieval

```python
class EmbeddingManager:
    """Manages embeddings and FAISS indices"""

    def __init__(self, working_dir: str):
        self.working_dir = working_dir

        # Load vector indices (NanoVectorDB)
        self.entity_index = self._load_index("vdb_entities.json")
        self.edge_index = self._load_index("vdb_bipartite_edges.json")

        # Load metadata from GraphML
        # Entity and relation metadata (names, descriptions, weights) are stored in the graph
        self.graph = self._load_graph("graph_chunk_entity_relation.graphml")
        self.entity_metadata = self._extract_entity_metadata(self.graph)
        self.edge_metadata = self._extract_edge_metadata(self.graph)

        # Auto-detect embedding format
        self.embedding_format = self._detect_embedding_format()

    def _detect_embedding_format(self) -> str:
        """
        Detect if embeddings are OpenAI or FlagEmbedding format

        OpenAI: Direct list of floats
        FlagEmbedding: Nested structure
        """
        sample = self.entity_metadata[0]
        if isinstance(sample["__vector__"], list):
            return "openai"
        else:
            return "flag_embedding"

    async def search_entities(self, query: str, top_k: int) -> list[dict]:
        """Entity-based retrieval"""
        # Embed query
        query_vec = await self.embed_query(query)

        # FAISS search
        distances, indices = self.entity_index.search(query_vec, top_k)

        # Retrieve metadata
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            entity_data = self.entity_metadata[idx]
            results.append({
                "id": entity_data["id"],
                "name": entity_data["entity_name"],
                "type": entity_data["entity_type"],
                "description": entity_data["description"],
                "score": float(dist)
            })

        return results

    async def search_edges(self, query: str, top_k: int) -> list[dict]:
        """Relation-based retrieval"""
        query_vec = await self.embed_query(query)
        distances, indices = self.edge_index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            edge_data = self.edge_metadata[idx]
            results.append({
                "id": edge_data["id"],
                "content": edge_data["content"],
                "weight": edge_data["weight"],
                "score": float(dist)
            })

        return results
```

#### LLMProviderManager

**Purpose:** Manage multiple LLM providers with fallback

```python
class LLMProviderManager:
    """Manages LLM providers with automatic fallback"""

    def __init__(self):
        self.providers = []

        # Add providers in priority order
        if os.getenv("OPENAI_API_KEY"):
            self.providers.append(OpenAIProvider())

        if os.getenv("ANTHROPIC_API_KEY"):
            self.providers.append(AnthropicProvider())

        if os.getenv("GOOGLE_API_KEY"):
            self.providers.append(GoogleProvider())

        # Default fallback
        self.providers.append(OpenAIProvider(model="gpt-4o-mini"))

    async def complete(self, prompt: str, **kwargs) -> str:
        """Try providers in order until success"""
        last_error = None

        for provider in self.providers:
            try:
                return await provider.complete(prompt, **kwargs)
            except Exception as e:
                last_error = e
                continue

        # All failed
        raise RuntimeError(f"All LLM providers failed: {last_error}")
```

**API Endpoints:**

#### POST /search

**Purpose:** Main retrieval endpoint

```python
@app.post("/search")
async def search(request: SearchRequest) -> list[list[dict]]:
    """
    Batch retrieval from knowledge graph

    Request:
        {
            "queries": ["query1", "query2", ...],
            "top_k": 60,
            "mode": "hybrid"
        }

    Response:
        [
            [  # Results for query1
                {
                    "<knowledge>": "Context text...",
                    "<coherence>": 0.95
                },
                ...
            ],
            [  # Results for query2
                ...
            ]
        ]
    """
    results = []

    for query in request.queries:
        # Query BiGRAG (hybrid mode by default)
        context = await bigrag_instance.aquery(
            query,
            QueryParam(
                mode=request.mode,
                top_k=request.top_k
            )
        )

        # Format response
        formatted = [
            {
                "<knowledge>": item["content"],
                "<coherence>": item.get("score", 0.0)
            }
            for item in context
        ]

        results.append(formatted)

    return results
```

#### POST /ask

**Purpose:** Full RAG pipeline (retrieve + generate)

```python
@app.post("/ask")
async def ask(request: AskRequest) -> dict:
    """
    Full RAG pipeline

    Request:
        {
            "question": "What is the capital of France?",
            "top_k": 60,
            "mode": "hybrid"
        }

    Response:
        {
            "answer": "Paris is the capital of France.",
            "context": ["retrieved context 1", "retrieved context 2", ...],
            "sources": ["source_id_1", "source_id_2", ...]
        }
    """
    # 1. Retrieve context
    context = await bigrag_instance.aquery(
        request.question,
        QueryParam(mode=request.mode, top_k=request.top_k)
    )

    # 2. Format prompt
    context_text = "\n\n".join([item["content"] for item in context[:5]])
    prompt = f"""
Context:
{context_text}

Question: {request.question}

Answer the question based on the context above.
"""

    # 3. Generate answer
    answer = await llm_manager.complete(prompt)

    # 4. Return response
    return {
        "answer": answer,
        "context": [item["content"] for item in context],
        "sources": [item.get("source_id", "") for item in context]
    }
```

#### GET /health

**Purpose:** Health check and statistics

```python
@app.get("/health")
async def health() -> dict:
    """
    Server health and statistics

    Response:
        {
            "status": "healthy",
            "data_source": "2WikiMultiHopQA",
            "entity_count": 12543,
            "edge_count": 8921,
            "embedding_dim": 3072
        }
    """
    return {
        "status": "healthy",
        "data_source": embedding_manager.data_source,
        "entity_count": len(embedding_manager.entity_metadata),
        "edge_count": len(embedding_manager.edge_metadata),
        "embedding_dim": embedding_manager.embedding_dim
    }
```

#### POST /chat/completions

**Purpose:** OpenAI-compatible chat endpoint (GPT-4o-mini)

```python
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict:
    """
    OpenAI-compatible chat completions endpoint

    Request:
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }

    Response:
        {
            "id": "chatcmpl-xyz123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "2 + 2 equals 4."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
    """
    # Call LLM provider
    response = await llm_manager.complete(
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    # Format OpenAI-compatible response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": estimate_tokens(request.messages),
            "completion_tokens": estimate_tokens(response),
            "total_tokens": estimate_tokens(request.messages) + estimate_tokens(response)
        }
    }
```

**Usage Example:**

```bash
# Curl
curl -X POST http://localhost:8001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Explain quantum physics in simple terms"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'

# Python
import requests

response = requests.post(
    "http://localhost:8001/chat/completions",
    json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    }
)

answer = response.json()["choices"][0]["message"]["content"]
print(answer)
```

#### GET /

**Purpose:** Root endpoint showing available services

```python
@app.get("/")
async def root() -> dict:
    """
    API information and available endpoints

    Response:
        {
            "message": "BiG-RAG API Server",
            "version": "1.0.0",
            "dataset": "2WikiMultiHopQA",
            "endpoints": {
                "retrieval": "/search",
                "chat": "/chat/completions",
                "rag": "/ask",
                "health": "/health",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    """
    return {
        "message": "BiG-RAG API Server",
        "version": "1.0.0",
        "dataset": embedding_manager.data_source,
        "endpoints": {
            "retrieval": "/search",
            "chat": "/chat/completions",
            "rag": "/ask",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }
```

#### API Documentation

**Interactive Documentation:**

- **Swagger UI**: `http://localhost:8001/docs`
  - Interactive API testing interface
  - Try out endpoints directly in browser
  - View request/response schemas

- **ReDoc**: `http://localhost:8001/redoc`
  - Clean, searchable documentation
  - Better for reading and reference
  - Mobile-friendly

**Server Startup:**

```python
def main(data_source: str):
    """Start retrieval server"""

    # 1. Load BiGRAG graph
    working_dir = f"./expr/{data_source}"
    global bigrag_instance
    bigrag_instance = BiGRAG(working_dir=working_dir)

    # 2. Initialize managers
    global embedding_manager
    embedding_manager = EmbeddingManager(working_dir)

    global llm_manager
    llm_manager = LLMProviderManager()

    # 3. Start server
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, required=True)
    args = parser.parse_args()
    main(args.data_source)
```

**Important Notes:**

1. **Server must run during training**: RL training makes HTTP requests to port 8001
2. **Blocking operation**: Server runs indefinitely until terminated
3. **Resource usage**: Loads all indices into memory (RAM requirement scales with corpus size)
4. **Concurrent requests**: FastAPI handles async requests with automatic worker pooling

---

## Data Structures and Formats

### Bipartite Graph Structure

**Conceptual Model:**

```
Two node types:
1. Entity Nodes (role="entity")
   - Properties: entity_name, entity_type, description, weight, source_id

2. Bipartite Edge Nodes (role="bipartite_edge")
   - Properties: content, weight, source_id

Edges (undirected):
- Connect entity nodes to bipartite edge nodes
- No entity-entity edges
- No edge-edge connections
```

**NetworkX Representation:**

```python
# Add entity node
graph.add_node(
    "Paris",
    entity_name="Paris",
    entity_type="geo",
    description="Capital of France",
    weight=253,
    source_id=["chunk_1", "chunk_2"],
    role="entity"
)

# Add bipartite edge node
graph.add_node(
    "edge_123",
    content="Paris is the capital of France",
    weight=180,
    source_id=["chunk_1"],
    role="bipartite_edge"
)

# Add edge connecting them
graph.add_edge(
    "Paris",
    "edge_123",
    weight=180,
    source_id="chunk_1"
)
```

**Serialization Format (GraphML):**

```xml
<graphml>
  <graph edgedefault="undirected">
    <node id="Paris">
      <data key="entity_name">Paris</data>
      <data key="entity_type">geo</data>
      <data key="description">Capital of France</data>
      <data key="weight">253</data>
      <data key="role">entity</data>
    </node>
    <node id="edge_123">
      <data key="content">Paris is the capital of France</data>
      <data key="weight">180</data>
      <data key="role">bipartite_edge</data>
    </node>
    <edge source="Paris" target="edge_123">
      <data key="weight">180</data>
    </edge>
  </graph>
</graphml>
```

### Vector Storage Format

**NanoVectorDB JSON Structure:**

```json
{
  "__id__": "entity_paris",
  "__vector__": [0.1, 0.2, ..., 0.5],
  "entity_name": "Paris",
  "entity_type": "geo",
  "description": "Capital city of France, located on the Seine River",
  "weight": 253,
  "source_id": ["chunk_abc", "chunk_def"]
}
```

**Vector Database Files (NanoVectorDB):**

Files: `vdb_entities.json`, `vdb_bipartite_edges.json`, `vdb_chunks.json`

Format: JSON with embedded vectors and metadata

**Structure:**
```json
{
  "data": [
    {
      "id": "entity_123",
      "__vector__": [0.1, 0.2, ...],  // Embedding vector
      "metadata": {
        "name": "Paris",
        "description": "Capital of France",
        "entity_type": "LOCATION"
      }
    }
  ]
}
```

**Loading:**
```python
from bigrag.storage import NanoVectorDBStorage
vdb = NanoVectorDBStorage(embedding_dim=1024, storage_file="vdb_entities.json")
```

### KV Storage Format

**JSON Files:** `kv_store_*.json`

```json
{
  "chunk_id_1": {
    "content": "Text content of chunk...",
    "tokens": 1180,
    "chunk_order_index": 0,
    "doc_id": "doc_123",
    "metadata": {...}
  },
  "chunk_id_2": {
    "content": "Next chunk...",
    "tokens": 1195,
    "chunk_order_index": 1,
    "doc_id": "doc_123"
  }
}
```

### Training Data Format

**Parquet Schema:**

```
Column: prompt (string)
  - Formatted instruction template with question
  - Includes system prompt and chat template markers

Column: question (string)
  - Original question text

Column: answers (list[string])
  - Ground truth answers (multiple acceptable answers)
```

**Example Row:**

```python
{
    "prompt": "<|im_start|>system\nYou are a helpful assistant...<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "question": "What is the capital of France?",
    "answers": ["Paris", "paris"]
}
```

---

## Extension Points

### Adding New Storage Backends

**Step 1:** Implement abstract base class

```python
# bigrag/kg/vectordb_impl/my_vectordb.py

from bigrag.base import BaseVectorStorage

class MyVectorDBStorage(BaseVectorStorage):
    def __init__(self, namespace: str, global_config: dict):
        self.namespace = namespace
        self.config = global_config
        # Initialize your vector DB client
        self.client = MyVectorDBClient(...)

    async def query(self, query: str, top_k: int) -> list[dict]:
        # Implement similarity search
        embedding = await self.config["embedding_func"]([query])
        results = self.client.search(embedding[0], top_k)
        return results

    async def upsert(self, data: dict[str, dict]):
        # Implement insert/update
        for id, item in data.items():
            self.client.insert(id, item)

    async def index_done_callback(self):
        # Finalize indexing
        self.client.build_index()

    @property
    def embedding_dim(self) -> int:
        return self.config.get("embedding_dim", 1536)
```

**Step 2:** Register in lazy loading

```python
# bigrag/bigrag.py, in lazy_external_import()

elif cls_name == "MyVectorDBStorage":
    from bigrag.kg.vectordb_impl.my_vectordb import MyVectorDBStorage
    return MyVectorDBStorage
```

**Step 3:** Use in configuration

```python
rag = BiGRAG(
    vector_storage="MyVectorDBStorage",
    graph_storage="NetworkXStorage",
    kv_storage="JsonKVStorage"
)
```

### Adding New Tools

**Step 1:** Implement Tool base class

```python
# agent/tool/tools/my_tool.py

from agent.tool.base_tool import Tool

class MyTool(Tool):
    """Custom tool implementation"""

    name: str = "my_tool"
    description: str = "Description of what this tool does"

    def execute(self, args: dict) -> str:
        """
        Single execution

        Args:
            args: {"param1": value1, "param2": value2}

        Returns:
            Result string
        """
        # Implement tool logic
        result = my_function(args["param1"], args["param2"])
        return str(result)

    def batch_execute(self, args_list: list[dict]) -> list[str]:
        """
        Batch execution (more efficient)

        Args:
            args_list: [{"param1": v1}, {"param1": v2}, ...]

        Returns:
            [result1, result2, ...]
        """
        # Batch implementation
        results = [self.execute(args) for args in args_list]
        return results

    def calculate_reward(self, args: dict, result: str) -> float:
        """
        Calculate reward for this tool call

        Returns:
            Reward value (0.0 for no reward)
        """
        # Implement reward logic if needed
        return 0.0
```

**Step 2:** Register tool in ToolEnv

```python
# Training script or configuration

from agent.tool.tools.my_tool import MyTool

tools = {
    "search": SearchTool(),
    "my_tool": MyTool()
}

envs = [
    ToolEnv(config=config, tools=tools)
    for _ in range(batch_size)
]
```

**Step 3:** Update prompt template

```python
# Prompt template should mention available tools

system_prompt = """
You are a helpful assistant with access to the following tools:
1. search: Query knowledge graph for information
2. my_tool: [Description of your tool]

To use a tool, generate:
<query>{"tool": "tool_name", "args": {...}}</query>
"""
```

### Customizing Entity Extraction

**Step 1:** Modify prompt template

```python
# bigrag/prompt.py

PROMPTS["entity_extraction"] = """
-Goal-
Extract entities of the following types: {entity_types}
[Add your custom instructions here]

-Format-
entity_name<|>entity_type<|>entity_description<|>custom_field

[Add examples]
"""
```

**Step 2:** Update extraction logic

```python
# bigrag/operate.py, in extract_entities()

# Parse custom fields
parts = line.split("<|>")
entity = {
    "entity_name": parts[0],
    "entity_type": parts[1],
    "entity_description": parts[2],
    "custom_field": parts[3] if len(parts) > 3 else None
}
```

**Step 3:** Update graph construction

```python
# bigrag/operate.py, in _merge_nodes_then_upsert()

# Include custom field in node data
node_data = {
    "entity_name": entity_name,
    "entity_type": merged_type,
    "description": merged_description,
    "weight": total_weight,
    "custom_field": merged_custom_field,  # Add custom handling
    "source_id": source_ids
}
```

### Adding New Embedding Models

**Step 1:** Implement embedding function

```python
# bigrag/llm.py

def my_embedding_func(api_key: str, model: str = "my-model"):
    """Custom embedding function"""

    async def embed(texts: list[str]) -> np.ndarray:
        # Call your embedding API
        response = await my_api_client.embed(
            texts=texts,
            model=model,
            api_key=api_key
        )

        # Convert to numpy array
        embeddings = np.array(response.embeddings)
        return embeddings

    return embed
```

**Step 2:** Use in BiGRAG initialization

```python
from bigrag.llm import my_embedding_func

rag = BiGRAG(
    embedding_func=my_embedding_func(
        api_key="your_api_key",
        model="my-embedding-model"
    )
)
```

**Step 3:** Update dimension configuration

```python
# If dimension is non-standard, specify in vector storage config
rag = BiGRAG(
    embedding_func=my_embedding_func(...),
    vector_storage="NanoVectorDBStorage",
    global_config={
        "embedding_dim": 2048  # Custom dimension
    }
)
```

### Switching LLM Providers

**BiG-RAG supports 10+ LLM providers** through the `llm_model_func` parameter. All providers are already implemented in `bigrag/llm.py`.

#### Built-in Providers

**OpenAI (Default):**
```python
from bigrag.llm import gpt_4o_mini_complete, gpt_4o_complete

# GPT-4o-mini (recommended for cost efficiency)
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=gpt_4o_mini_complete
)

# GPT-4o (higher quality)
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=gpt_4o_complete
)
```

**Azure OpenAI:**
```python
from bigrag.llm import azure_openai_complete

# Set environment variables:
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_VERSION

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=azure_openai_complete,
    llm_model_name="gpt-4o-mini"  # Your deployment name
)
```

**AWS Bedrock (Claude):**
```python
from bigrag.llm import bedrock_complete

# Requires AWS credentials configured
# Uses boto3 for authentication

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=bedrock_complete,
    llm_model_name="anthropic.claude-3-sonnet-20240229-v1:0"
)
```

**Local Ollama:**
```python
from bigrag.llm import ollama_model_complete

# Requires Ollama running on localhost:11434
# Download models: ollama pull llama3

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3"  # or mistral, qwen2.5, etc.
)
```

**Hugging Face (Local):**
```python
from bigrag.llm import hf_model_complete

# Loads model locally with transformers
# Requires GPU for reasonable performance

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=hf_model_complete,
    llm_model_name="Qwen/Qwen2.5-7B-Instruct"
)
```

**Zhipu AI (ChatGLM):**
```python
from bigrag.llm import zhipu_complete

# Chinese LLM provider
# Set ZHIPUAI_API_KEY environment variable

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-plus"  # or glm-4-flash
)
```

**NVIDIA NIM:**
```python
from bigrag.llm import openai_complete_if_cache

async def nvidia_complete(prompt, system_prompt=None, **kwargs):
    return await openai_complete_if_cache(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        prompt=prompt,
        system_prompt=system_prompt,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="your-nvidia-api-key",
        **kwargs
    )

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=nvidia_complete
)
```

#### OpenAI-Compatible Providers

Many providers offer OpenAI-compatible APIs. Use `openai_complete_if_cache` with custom `base_url`:

**DeepSeek:**
```python
from bigrag.llm import openai_complete_if_cache

async def deepseek_complete(prompt, system_prompt=None, **kwargs):
    return await openai_complete_if_cache(
        model="deepseek-chat",
        prompt=prompt,
        system_prompt=system_prompt,
        base_url="https://api.deepseek.com/v1",
        api_key="your-deepseek-api-key",
        **kwargs
    )

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=deepseek_complete
)
```

**Google Gemini:**
```python
from bigrag.llm import openai_complete_if_cache

async def gemini_complete(prompt, system_prompt=None, **kwargs):
    return await openai_complete_if_cache(
        model="gemini-1.5-flash",
        prompt=prompt,
        system_prompt=system_prompt,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="your-google-api-key",
        **kwargs
    )

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=gemini_complete
)
```

**Together AI:**
```python
from bigrag.llm import openai_complete_if_cache

async def together_complete(prompt, system_prompt=None, **kwargs):
    return await openai_complete_if_cache(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        prompt=prompt,
        system_prompt=system_prompt,
        base_url="https://api.together.xyz/v1",
        api_key="your-together-api-key",
        **kwargs
    )

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=together_complete
)
```

**Groq:**
```python
from bigrag.llm import openai_complete_if_cache

async def groq_complete(prompt, system_prompt=None, **kwargs):
    return await openai_complete_if_cache(
        model="llama-3.1-70b-versatile",
        prompt=prompt,
        system_prompt=system_prompt,
        base_url="https://api.groq.com/openai/v1",
        api_key="your-groq-api-key",
        **kwargs
    )

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=groq_complete
)
```

#### Multi-Model Load Balancing

**Distribute requests across multiple models for reliability:**

```python
from bigrag.llm import MultiModel, Model

# Define multiple models
models = [
    Model(
        func=gpt_4o_mini_complete,
        name="gpt-4o-mini"
    ),
    Model(
        func=deepseek_complete,
        name="deepseek-chat"
    ),
    Model(
        func=gemini_complete,
        name="gemini-1.5-flash"
    )
]

# Create multi-model manager (round-robin load balancing)
multi_model = MultiModel(models)

# Use with BiGRAG
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=multi_model.llm_model_func
)
```

**Benefits:**
- Automatic failover if one provider is down
- Cost distribution across providers
- Rate limit mitigation
- Higher throughput

#### Cost Optimization Tips

**1. Use cheaper models for extraction:**
```python
# GPT-4o-mini is 15x cheaper than GPT-4o
rag = BiGRAG(
    llm_model_func=gpt_4o_mini_complete  # $0.15/1M input tokens
)
```

**2. Use local models for large-scale processing:**
```python
# Ollama is free but requires local GPU
rag = BiGRAG(
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:7b"  # Free, runs locally
)
```

**3. LLM response caching is automatic:**
```python
# All providers automatically use caching
# Identical prompts return cached results
# No code changes needed
```

#### Provider Comparison

| Provider | Cost | Speed | Quality | Local | Best For |
|----------|------|-------|---------|-------|----------|
| **GPT-4o-mini** | $ | Fast | High | ❌ | Production (recommended) |
| **GPT-4o** | $$$ | Medium | Highest | ❌ | Complex reasoning |
| **DeepSeek** | $ | Fast | High | ❌ | Cost-efficient alternative |
| **Gemini 1.5 Flash** | $ | Very Fast | High | ❌ | High throughput |
| **Claude (Bedrock)** | $$ | Medium | Very High | ❌ | Instruction following |
| **Ollama (Llama 3)** | Free | Medium | Good | ✅ | Development/testing |
| **Ollama (Qwen2.5)** | Free | Medium | Good | ✅ | Multilingual |
| **Together AI** | $$ | Fast | High | ❌ | GPU inference |
| **Groq** | $$ | Very Fast | Good | ❌ | Low latency |

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

## Critical Implementation Details

### 1. Multi-Turn Entity Extraction with Gleaning

**Location:** `bigrag/operate.py`, `extract_entities()` function

**Process:**

```python
# Initial extraction
response_1 = await llm_model_func(extraction_prompt)
entities_1 = parse_entities(response_1)

# Gleaning loop (max 2 iterations)
for iteration in range(entity_extract_max_gleaning):
    gleaning_prompt = f"""
Previously extracted:
{format_entities(entities_1)}

Did you miss any entities? If yes, provide them.
If no more entities, output <|COMPLETE|>
"""

    response_n = await llm_model_func(gleaning_prompt)

    if "<|COMPLETE|>" in response_n:
        break

    entities_n = parse_entities(response_n)
    entities_1.extend(entities_n)

return entities_1
```

**Impact:**
- Improves entity coverage by 15-25%
- Minimal cost increase (only 2 additional API calls per chunk)
- LLM decides when to stop (completeness assessment)

### 2. LLM Response Caching

**Location:** `bigrag/utils.py`, `bigrag/bigrag.py`

**Implementation:**

```python
# Cache key generation
def generate_cache_key(prompt: str, system_prompt: str, kwargs: dict) -> str:
    combined = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "kwargs": json.dumps(kwargs, sort_keys=True)
    }
    return hashlib.md5(json.dumps(combined).encode()).hexdigest()

# Cache check
cache_key = generate_cache_key(prompt, system_prompt, kwargs)
cached = await llm_response_cache.get_by_id(cache_key)

if cached:
    return cached["response"]

# Call LLM
response = await llm_model_func(prompt, **kwargs)

# Cache result
await llm_response_cache.upsert({
    cache_key: {
        "response": response,
        "timestamp": time.time()
    }
})
```

**Cost Savings:**
- Reduces API calls by 60-70% during graph construction
- Especially effective for overlapping chunks
- Zero impact on user-facing API (transparent caching)

### 3. Active Masking in Tool Generation

**Location:** `agent/llm_agent/generation.py`, `run_llm_loop()` method

**Logic:**

```python
# Initialize all sequences as active
active_mask = torch.ones(batch_size, dtype=torch.bool)

for turn in range(max_turns):
    # Filter to active sequences only
    rollings_active = [r for i, r in enumerate(rollings) if active_mask[i]]

    # Generate and extract tool calls
    responses = generate_and_extract(rollings_active)

    # Update mask: inactive if no valid tool call
    new_active_masks = [has_valid_tool_call(r) for r in responses]

    # Merge with previous mask (once inactive, stays inactive)
    active_mask = active_mask & new_active_masks

    # Early termination if all inactive
    if not active_mask.any():
        break
```

**Benefits:**
- Saves computation on sequences that won't succeed
- Improves training stability (fewer invalid trajectories)
- Reduces memory usage (fewer active sequences each turn)

### 4. Graph Stabilization

**Location:** `bigrag/storage.py`, `NetworkXStorage._stabilize_graph()` method

**Process:**

```python
def _stabilize_graph(self):
    """Stabilize graph for reproducibility and quality"""

    # 1. Extract largest connected component
    components = list(nx.connected_components(self._graph))
    largest = max(components, key=len)
    self._graph = self._graph.subgraph(largest).copy()

    # 2. Normalize node names
    mapping = {}
    for node in self._graph.nodes():
        # Uppercase first letter
        normalized = node[0].upper() + node[1:] if len(node) > 0 else node
        # Remove HTML entities
        normalized = re.sub(r'&#\d+;', '', normalized)
        mapping[node] = normalized

    self._graph = nx.relabel_nodes(self._graph, mapping)

    # 3. Sort edges for determinism
    edges = sorted(self._graph.edges())
    self._graph = nx.Graph(edges)
```

**Why This Matters:**
- Ensures reproducibility across builds
- Removes disconnected subgraphs (low-quality entities)
- Normalizes entity names for better matching

### 5. Node Merging with LLM Summarization

**Location:** `bigrag/operate.py`, `_merge_nodes_then_upsert()` function

**Strategy:**

```python
# Concatenate descriptions
descriptions = [node["description"] for node in nodes_data]
combined = "<SEP>".join(descriptions)

# Check token count
token_count = len(tiktoken_encoder.encode(combined))

if token_count > entity_summary_to_max_tokens:
    # Use LLM to summarize
    summary_prompt = f"""
Summarize the following descriptions into {entity_summary_to_max_tokens} tokens:

{combined}

Summary:
"""

    summary = await llm_model_func(summary_prompt)
    final_description = summary
else:
    final_description = combined
```

**Trade-offs:**
- Prevents description bloat (important for prompt limits)
- Small cost increase (summarization API calls)
- May lose some specificity (but preserves key information)

### 6. Reciprocal Rank Fusion

**Location:** `bigrag/operate.py`, `_merge_and_rank()` function

**Formula:**

```
For each item appearing in multiple result lists:
  score = sum(1 / (rank_i + 1) for all occurrences)

Where:
  rank_i = position in result list i (0-indexed)
```

**Example:**

```python
# Entity results: ["Paris", "France", "Europe"]
# Relation results: ["France", "Paris", "Germany"]

# Paris: 1/(0+1) + 1/(1+1) = 1.0 + 0.5 = 1.5
# France: 1/(1+1) + 1/(0+1) = 0.5 + 1.0 = 1.5
# Europe: 1/(2+1) = 0.33
# Germany: 1/(2+1) = 0.33

# Final ranking: [Paris, France, Europe, Germany]
```

**Why This Works:**
- No hyperparameters to tune
- Natural weighting (top ranks get more weight)
- Robust to different score scales
- Well-established in information retrieval

### 7. Event Loop Management

**Location:** `bigrag/utils.py`, `always_get_an_event_loop()` function

**Implementation:**

```python
def always_get_an_event_loop():
    """
    Get or create event loop (handles closed loops)

    Required for:
    - Windows compatibility
    - Jupyter notebooks
    - Ray workers (distributed training)
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        # Create new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
```

**Why This Matters:**
- Windows event loops can close unexpectedly
- Ray workers need new loops per worker
- Jupyter notebooks use nested loops
- Without this, async operations would fail

### 8. Batch-Only Tool Execution

**Location:** `agent/tool/tools/search_tool.py`, `batch_execute()` method

**Design Decision:**

```python
class SearchTool(Tool):
    def execute(self, args: dict) -> str:
        """Not implemented - use batch_execute()"""
        pass

    def batch_execute(self, args_list: list[dict]) -> list[str]:
        """Only batch execution is implemented"""
        # Make single HTTP request with all queries
        response = requests.post(
            self.api_url,
            json={"queries": [args["query"] for args in args_list]}
        )
        return response.json()
```

**Rationale:**
- Batch execution is 10-50x faster (single HTTP request)
- Distributed training always uses batches
- Sequential execution would require per-worker HTTP clients
- Simpler implementation with fewer edge cases

### 9. GPU Padding for Data Parallelism

**Location:** `agent/llm_agent/generation.py`, `_generate_with_gpu_padding()` method

**Why Needed:**

```python
# Problem: Batch size must be divisible by GPU count
batch_size = 13
n_gpus = 4
# 13 % 4 = 1 (not divisible)

# Solution: Add padding
padding_size = 4 - (13 % 4) = 3
padded_batch_size = 13 + 3 = 16 (divisible by 4)

# Add copies of last sequence
padded_batch = batch + [batch[-1]] * 3

# Generate
output = vllm.generate(padded_batch)

# Remove padding from output
final_output = output[:13]
```

**Why This Approach:**
- vLLM requires even distribution across GPUs
- Padding is cheaper than adjusting batch size
- Ensures consistent batch structure across workers

### 10. Query Mode Selection

**Location:** `bigrag/base.py`, `QueryParam.mode` field

**Available Modes:**

1. **local**: Entity-based retrieval
   - Best for: Specific entity queries ("Who is X?")
   - Traversal: query → entities → relations

2. **global**: Relation-based retrieval
   - Best for: Relationship queries ("What connects X and Y?")
   - Traversal: query → relations → entities

3. **hybrid**: Combined three-path (default)
   - Best for: General questions (most robust)
   - Traversal: All three paths (A+B+C) + reciprocal rank fusion

4. **naive**: Direct chunk similarity
   - Best for: Baseline comparison only
   - Traversal: query → chunks (no graph)

**When to Use Each:**

```python
# Entity-focused question
QueryParam(mode="local")
# "Who is Marie Curie?"

# Relationship-focused question
QueryParam(mode="global")
# "What is the relationship between France and Germany?"

# General question (recommended default)
QueryParam(mode="hybrid")
# "What factors led to World War I?"

# Baseline for evaluation
QueryParam(mode="naive")
# Compare against graph-based retrieval
```

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
entity_keys = await rag.vdb_entities.all_keys()
print(f"Total entities: {len(entity_keys)}")

# Inspect specific entity
entity = await rag.vdb_entities.get_by_id("entity_id")
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

## Summary

This comprehensive guide covers the complete A-to-Z implementation of the BiG-RAG framework, including:

### Core Implementation (Sections 1-10)
1. **Core Architecture**: Async-first, pluggable storage, lazy loading
2. **Graph Construction**: Multi-turn extraction, bipartite structure, merging strategies
3. **Retrieval System**: Three-path querying, reciprocal rank fusion, FAISS indices
4. **Tool Integration**: Active masking, batch execution, environment management
5. **Extension Points**: Adding backends, tools, embeddings, customizing extraction

### Development & Operations (Sections 11-17)
6. **Testing Framework**: Demo dataset, test scripts, validation procedures
7. **Critical Implementation Details**: 10 key patterns and design decisions
8. **Performance Considerations**: Embedding, extraction, and query optimization
9. **Best Practices**: Async/await, caching, API cost management
10. **Debugging Tips**: Logging, profiling, graph inspection
11. **Known Issues**: Solutions for Unicode, FAISS, memory, rate limits
12. **Future Development**: Enhancement ideas and research directions

### Key Takeaways

**Technical Highlights:**
- **Bipartite graph** (not hypergraph): True two-layer structure with entities and relations as separate node types
- **Multi-turn gleaning**: Iterative entity extraction improves coverage by 15-25%
- **LLM caching**: Transparent cost optimization with 60-70% API call reduction
- **Active masking**: Sequences stop generating when tool calls fail, saving computation
- **Hybrid retrieval**: Combines entity and relation paths for robust performance
- **Async patterns**: Critical for distributed training and event loop management

**Development Guidance:**
- **Testing**: Demo dataset with 10 test cases validates all major functionality
- **Performance**: Batch processing, caching, and local models reduce costs by 60-70%
- **Debugging**: GraphML export, verbose logging, and profiling tools included
- **Extensibility**: All extension points designed for minimal core code changes

**Production Readiness:**
- Storage backends support enterprise databases (Neo4J, Milvus, Oracle, TiDB)
- 10+ LLM providers supported with automatic failover
- Rate limiting and exponential backoff implemented
- Memory usage scales with external vector databases

This guide enables LLMs, developers, and DevOps teams to understand, deploy, debug, optimize, and extend the BiG-RAG framework effectively in both development and production environments.
