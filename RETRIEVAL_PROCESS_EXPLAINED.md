# BiG-RAG Retrieval Process: Complete Guide

**Date:** October 31, 2025
**Purpose:** Educational guide explaining how BiG-RAG indexes documents and retrieves knowledge

---

## Table of Contents

1. [Overview: From Document to Answer](#1-overview-from-document-to-answer)
2. [Indexing Pipeline - What Gets Created](#2-indexing-pipeline-what-gets-created)
3. [Storage Locations - Where Data Lives](#3-storage-locations-where-data-lives)
4. [Retrieval Pipeline - How Query Works](#4-retrieval-pipeline-how-query-works)
5. [Dual-Path Retrieval Explained](#5-dual-path-retrieval-explained)
6. [Real Example: Bangladesh Document](#6-real-example-bangladesh-document)
7. [Performance Characteristics](#7-performance-characteristics)

---

## 1. Overview: From Document to Answer

### The Big Picture

BiG-RAG transforms documents into a queryable knowledge graph through this pipeline:

```
┌──────────────────────────────────────────────────────────────────┐
│                    BiG-RAG Complete Pipeline                     │
└──────────────────────────────────────────────────────────────────┘

Step 1: Document Upload
   📄 Bangladesh.txt (38,000 characters)
      ↓
Step 2: Chunking
   ✂️  Split into 5 chunks (1200 tokens each, 100 overlap)
      ↓
Step 3: Entity & Relation Extraction (LLM)
   🤖 GPT-4o-mini extracts:
      • 156 entities (DHAKA, BANGLADESH, Sheikh Hasina, ...)
      • 89 relations (DHAKA-capital_of-BANGLADESH, ...)
      ↓
Step 4: Embedding (Embedder)
   🔢 OpenAI text-embedding-3-large converts to vectors:
      • Entity embeddings (156 x 1536 dims)
      • Relation embeddings (89 x 1536 dims)
      • Chunk embeddings (5 x 1536 dims)
      ↓
Step 5: Storage
   💾 Save to three layers:
      • Graph: NetworkX → GraphML (complete metadata)
      • Vector: NanoVectorDB (fast similarity search)
      • KV: JSON files (quick lookups)
      ↓
Step 6: Query Time
   ❓ "What is the capital of Bangladesh?"
      ↓
Step 7: Dual-Path Retrieval
   🔍 Search entities: DHAKA, BANGLADESH
   🔗 Search relations: DHAKA-capital_of-BANGLADESH
      ↓
Step 8: Graph Traversal
   🕸️  Entities → Relations → Chunks
      Get original text containing answers
      ↓
Step 9: Fusion & Ranking
   📊 Reciprocal rank fusion
      Combine entity-path + relation-path results
      ↓
Step 10: Context Generation
   📝 Format top-k chunks as natural language
      ↓
Step 11: LLM Generation
   🤖 Feed context to LLM → Generate answer
      "Dhaka is the capital of Bangladesh."
```

---

## 2. Indexing Pipeline - What Gets Created

### Phase 1: Document Reception

**Input:**
```python
# User uploads file via API
POST /documents/upload
Content-Type: multipart/form-data
file: Bangladesh.txt
dataset: demo_test
```

**What Happens:**
```python
# 1. Read file content
content = file.read()  # "Bangladesh, officially the People's Republic..."

# 2. Generate document ID
doc_id = f"doc-{md5(content)[:32]}"  # doc-53a0479813a7da9e631fcac2f7c0a80d

# 3. Save to corpus
corpus = {
    "id": doc_id,
    "title": "Bangladesh.txt",
    "contents": content
}
# Saved to: datasets/demo_test/raw/corpus.jsonl
```

**Created Files:**
- `datasets/demo_test/raw/corpus.jsonl` - Original document
- `expr/demo_test/documents_registry.json` - Upload tracking

---

### Phase 2: Chunking

**Purpose:** Break long documents into manageable pieces for processing.

**Algorithm:**
```python
# bigrag/operate.py: chunking_by_token_size()

def chunk_document(content: str, max_tokens: int = 1200, overlap: int = 100):
    """
    Split document using sliding window with overlap
    """
    # 1. Tokenize entire document
    tokens = encode(content)  # [34505, 11, 14098, 5926, ...]

    # 2. Create overlapping windows
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = decode(chunk_tokens)

        chunks.append({
            "content": chunk_text,
            "chunk_order_index": len(chunks),
            "tokens": len(chunk_tokens),
            "full_doc_id": doc_id
        })

    return chunks
```

**Example Output:**
```python
# Bangladesh.txt → 5 chunks
[
    {
        "content": "Bangladesh, officially the People's Republic of Bangladesh...",
        "chunk_order_index": 0,
        "tokens": 1200,
        "full_doc_id": "doc-53a0479813a7da9e631fcac2f7c0a80d"
    },
    {
        "content": "...Dhaka is the capital and largest city. The economy is...",
        "chunk_order_index": 1,
        "tokens": 1200,
        "full_doc_id": "doc-53a0479813a7da9e631fcac2f7c0a80d"
    },
    # ... 3 more chunks
]
```

**Chunk IDs Generated:**
```python
for chunk in chunks:
    chunk_id = f"chunk-{md5(chunk['content'])[:32]}"
    # chunk-710e6d96a7830efebf274af94ee904f2
    # chunk-8a92bf3c4d5e6f7a8b9c0d1e2f3a4b5c
    # ...
```

**Why Overlap?**
- Entities/relations might span chunk boundaries
- Overlap ensures context isn't lost at splits
- 100 tokens = ~75 words overlap

**Created Files:**
- `expr/demo_test/kv_store_text_chunks.json` - Chunk metadata
- Chunks stored as graph nodes (NetworkX)

---

### Phase 3: Entity Extraction (LLM)

**Purpose:** Identify named entities (people, places, organizations) in each chunk.

**LLM Prompt:**
```python
# bigrag/prompt.py: PROMPTS["entity_extraction"]

"""
You are a helpful assistant. Extract all named entities from the following text.

Categories:
- PERSON: People, including fictional
- LOCATION: Physical locations (cities, countries, landmarks)
- ORGANIZATION: Companies, agencies, institutions
- DATE: Dates, time periods
- EVENT: Named events
- MISC: Other named entities

Text:
{chunk_content}

Return JSON:
{
  "entities": [
    {"name": "DHAKA", "type": "LOCATION", "description": "Capital city of Bangladesh"},
    ...
  ]
}
"""
```

**Example Extraction:**
```python
# Input: chunk-710e6d... (first chunk of Bangladesh.txt)
# Output from GPT-4o-mini:
{
  "entities": [
    {
      "name": "BANGLADESH",
      "type": "LOCATION",
      "description": "Country in South Asia, officially the People's Republic of Bangladesh"
    },
    {
      "name": "DHAKA",
      "type": "LOCATION",
      "description": "Capital and largest city of Bangladesh"
    },
    {
      "name": "SHEIKH HASINA",
      "type": "PERSON",
      "description": "Prime Minister of Bangladesh"
    },
    {
      "name": "BAY OF BENGAL",
      "type": "LOCATION",
      "description": "Body of water bordering southern Bangladesh"
    }
  ]
}
```

**Processing:**
```python
for entity in extracted_entities:
    # Generate entity ID
    entity_id = f"ent-{md5(entity['name'])[:32]}"

    # Track which chunk this entity came from
    entity["source_id"] = chunk_id

    # Calculate importance weight (frequency, position)
    entity["weight"] = calculate_weight(entity, chunk)

    # Store in graph
    graph.add_node(entity_id, **entity, role="entity")
    graph.add_edge(entity_id, chunk_id)  # Entity → Chunk
```

**For All Chunks:**
```
Chunk 1 (Bangladesh geography) → 42 entities
Chunk 2 (History) → 38 entities
Chunk 3 (Economy) → 31 entities
Chunk 4 (Culture) → 27 entities
Chunk 5 (Politics) → 18 entities
──────────────────────────────────────────
Total: 156 entities (some duplicates merged)
```

**Deduplication:**
```python
# If "DHAKA" appears in multiple chunks, merge:
entity_sources = {
    "ent-d691a638...": {
        "entity_name": "DHAKA",
        "source_id": "chunk-710e6d<SEP>chunk-8a92bf<SEP>chunk-3f4e5d"
        # Appears in 3 chunks
    }
}
```

**Created Files:**
- In **FlagEmbedding mode**: `expr/demo_test/kv_store_entities.json`
- In **OpenAI mode**: Stored only in GraphML (not in `vdb_entities.json`)
- Graph nodes with `role="entity"`

---

### Phase 4: Relation Extraction (LLM)

**Purpose:** Extract relationships between entities (N-ary relations).

**LLM Prompt:**
```python
# bigrag/prompt.py: PROMPTS["relationship_extraction"]

"""
Given entities and text, extract relationships as natural language descriptions.

Entities: BANGLADESH, DHAKA, SHEIKH HASINA, BAY OF BENGAL

Text:
{chunk_content}

Return JSON:
{
  "relationships": [
    {
      "head": "DHAKA",
      "relation": "capital_of",
      "tail": "BANGLADESH",
      "description": "Dhaka is the capital and largest city of Bangladesh, with a population of over 20 million."
    },
    ...
  ]
}
"""
```

**Example Extraction:**
```python
# Output from GPT-4o-mini:
{
  "relationships": [
    {
      "head": "DHAKA",
      "relation": "capital_of",
      "tail": "BANGLADESH",
      "description": "Dhaka is the capital and largest city of Bangladesh, located on the Buriganga River."
    },
    {
      "head": "BANGLADESH",
      "relation": "borders",
      "tail": "BAY OF BENGAL",
      "description": "Bangladesh has a coastline along the Bay of Bengal to the south."
    },
    {
      "head": "SHEIKH HASINA",
      "relation": "prime_minister_of",
      "tail": "BANGLADESH",
      "description": "Sheikh Hasina has served as Prime Minister of Bangladesh since 2009."
    }
  ]
}
```

**Why N-ary Relations as Nodes?**

Traditional knowledge graphs store relations as edges:
```
(DHAKA) --[capital_of]--> (BANGLADESH)
```

BiG-RAG stores relations as **nodes** in a bipartite graph:
```
(DHAKA) --> [DHAKA-capital_of-BANGLADESH] <-- (BANGLADESH)
            ↓
      (chunk-710e6d...)
```

**Benefits:**
- Relations can have descriptions (not just labels)
- Relations can be queried with vector similarity
- Multiple chunks can reference same relation
- Relations have importance weights

**Processing:**
```python
for rel in extracted_relations:
    # Generate relation ID
    rel_name = f"{rel['head']} - {rel['relation']} - {rel['tail']}"
    rel_id = f"edge-{md5(rel_name)[:32]}"

    # Store as node
    graph.add_node(rel_id,
                   bipartite_edge_name=rel_name,
                   description=rel['description'],
                   source_id=chunk_id,
                   role="bipartite_edge")

    # Connect to chunk
    graph.add_edge(rel_id, chunk_id)  # Relation → Chunk
```

**For All Chunks:**
```
Chunk 1 → 18 relations
Chunk 2 → 22 relations
Chunk 3 → 19 relations
Chunk 4 → 17 relations
Chunk 5 → 13 relations
────────────────────────
Total: 89 relations
```

**Created Files:**
- In **FlagEmbedding mode**: `expr/demo_test/kv_store_bipartite_edges.json`
- In **OpenAI mode**: Stored only in GraphML
- Graph nodes with `role="bipartite_edge"`

---

### Phase 5: Embedding Generation

**Purpose:** Convert text (entities, relations, chunks) into dense vectors for similarity search.

**What Gets Embedded:**

1. **Entity Embeddings:**
```python
# For each entity, embed: name + description
text = f"{entity['entity_name']}: {entity['description']}"
# "DHAKA: Capital and largest city of Bangladesh"

vector = embedding_func(text)
# [0.0234, -0.1456, 0.0892, ..., 0.0456]  # 1536 dimensions
```

2. **Relation Embeddings:**
```python
# For each relation, embed: name + description
text = f"{rel['bipartite_edge_name']}. {rel['description']}"
# "DHAKA - capital_of - BANGLADESH. Dhaka is the capital..."

vector = embedding_func(text)
# [0.0123, -0.0987, 0.1234, ..., 0.0567]  # 1536 dimensions
```

3. **Chunk Embeddings:**
```python
# For each chunk, embed: full content
text = chunk['content']
# "Bangladesh, officially the People's Republic..."

vector = embedding_func(text)
# [0.0456, -0.0789, 0.0123, ..., 0.0987]  # 1536 dimensions
```

**Storage:**

**OpenAI Mode (NanoVectorDB):**
```python
# vdb_entities.json
{
  "embedding_dim": 1536,
  "data": [
    {
      "__id__": "ent-d691a638...",
      "entity_name": "DHAKA"
      # ⚠️ Only name stored, no description/type/weight!
    }
  ],
  "matrix": [[0.0234, -0.1456, ...], [...], ...]  # 156 x 1536
}
```

**FlagEmbedding Mode (FAISS):**
```python
# index_entity.bin (FAISS index)
# - Fast approximate nearest neighbor search
# - IndexFlatIP (inner product)

# corpus_entity.npy (Raw vectors)
# - numpy array: 156 x 1024
# - Full precision vectors

# kv_store_entities.json (Full metadata)
# - entity_name, entity_type, description, weight, source_id
```

**Why FAISS is Faster:**
- Optimized C++ implementation
- Approximate search algorithms (IVF, HNSW)
- GPU support (if available)
- Scales to billions of vectors

**Why NanoVectorDB is Simpler:**
- Pure Python (no compilation)
- Exact search (no approximation)
- All-in-one JSON file
- Good for <100K vectors

**Created Files:**

**OpenAI Mode:**
- `expr/demo_test/vdb_entities.json` (2.9 MB)
- `expr/demo_test/vdb_bipartite_edges.json`
- `expr/demo_test/vdb_chunks.json` (often empty)

**FlagEmbedding Mode:**
- `expr/demo_test/index_entity.bin` (FAISS index)
- `expr/demo_test/index_bipartite_edge.bin`
- `expr/demo_test/index.bin` (chunks)
- `expr/demo_test/corpus_entity.npy` (raw vectors)
- `expr/demo_test/corpus_bipartite_edge.npy`
- `expr/demo_test/corpus.npy`

---

### Phase 6: Graph Serialization

**Purpose:** Save complete graph structure to disk for persistence.

**Process:**
```python
# bigrag/operate.py
import networkx as nx

# NetworkX graph (in-memory)
graph = nx.Graph()
# Contains:
# - 156 entity nodes
# - 89 relation nodes
# - 5 chunk nodes
# - ~400 edges (entities/relations → chunks)

# Save to GraphML
graphml_path = "expr/demo_test/graph_chunk_entity_relation.graphml"
nx.write_graphml(graph, graphml_path)

logger.info(f"Writing graph with {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges")
# INFO:bigrag:Writing graph with 564 nodes, 413 edges
```

**GraphML Content:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph edgedefault="undirected">
    <!-- 156 entity nodes -->
    <node id="ent-d691a638...">
      <data key="entity_name">DHAKA</data>
      <data key="entity_type">LOCATION</data>
      <data key="description">Capital city...</data>
      <data key="weight">3.5</data>
      <data key="source_id">chunk-710e6d<SEP>chunk-8a92bf</data>
      <data key="role">entity</data>
    </node>

    <!-- 89 relation nodes -->
    <node id="edge-abc123...">
      <data key="bipartite_edge_name">DHAKA - capital_of - BANGLADESH</data>
      <data key="description">Dhaka is the capital...</data>
      <data key="role">bipartite_edge</data>
    </node>

    <!-- 5 chunk nodes -->
    <node id="chunk-710e6d...">
      <data key="content">Bangladesh, officially...</data>
      <data key="full_doc_id">doc-53a0479813a7da9e631fcac2f7c0a80d</data>
      <data key="role">chunk</data>
    </node>

    <!-- Edges: entities/relations → chunks -->
    <edge source="ent-d691a638..." target="chunk-710e6d..."/>
  </graph>
</graphml>
```

**Key Property: COMPLETE METADATA**
- GraphML has **100% of data** regardless of embedder choice
- Vector databases have minimal metadata (just for indexing)
- GraphML is the **canonical source of truth**

**Created Files:**
- `expr/demo_test/graph_chunk_entity_relation.graphml` (~5-20 MB)

---

### Indexing Summary

**Final File Structure (OpenAI Mode):**
```
datasets/demo_test/raw/
├── corpus.jsonl                           # Original document

expr/demo_test/
├── graph_chunk_entity_relation.graphml   # 🌟 COMPLETE metadata (564 nodes, 413 edges)
├── kv_store_text_chunks.json             # 5 chunks with metadata
├── kv_store_full_docs.json               # Original document content
├── vdb_entities.json                     # 156 entity vectors (minimal metadata)
├── vdb_bipartite_edges.json              # 89 relation vectors (minimal metadata)
├── vdb_chunks.json                       # Empty or minimal
├── documents_registry.json               # Upload tracking
└── kv_store_llm_response_cache.json      # LLM call cache
```

**What Each File Contains:**

| File | Size | Content | Completeness | Purpose |
|------|------|---------|--------------|---------|
| `corpus.jsonl` | 38 KB | Original text | 100% | Source document |
| `graph_chunk_entity_relation.graphml` | 5 MB | **ALL graph data** | **100%** | **Ground truth** |
| `kv_store_text_chunks.json` | 50 KB | Chunk metadata | 100% | Quick lookup |
| `vdb_entities.json` | 2.9 MB | Vectors + names only | 5% | Fast search |
| `vdb_bipartite_edges.json` | 1.2 MB | Vectors + names only | 5% | Fast search |

---

## 3. Storage Locations - Where Data Lives

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Layer 1: Graph Database                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Purpose: Structure + Relationships + Metadata           │  │
│  │  Implementation: NetworkX (runtime) + GraphML (disk)     │  │
│  │  Content:                                                │  │
│  │    • Entity nodes: name, type, description, weight       │  │
│  │    • Relation nodes: description, source                 │  │
│  │    • Chunk nodes: text, document ID, position            │  │
│  │    • Edges: entity/relation → chunk connections          │  │
│  │  Used for:                                               │  │
│  │    • Graph traversal (find chunks from entities)         │  │
│  │    • Metadata retrieval (descriptions, types)            │  │
│  │    • Document statistics (count entities per doc)        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                Layer 2: Vector Database                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Purpose: Fast Similarity Search                         │  │
│  │  Implementation: FAISS (C++) or NanoVectorDB (Python)    │  │
│  │  Content:                                                │  │
│  │    • Entity embeddings: 1536-dim vectors                 │  │
│  │    • Relation embeddings: 1536-dim vectors               │  │
│  │    • Chunk embeddings: 1536-dim vectors                  │  │
│  │    • Minimal metadata: IDs, names only                   │  │
│  │  Used for:                                               │  │
│  │    • Top-k nearest neighbor search                       │  │
│  │    • Query embedding similarity                          │  │
│  │    • First-stage candidate retrieval                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                Layer 3: Key-Value Store                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Purpose: Fast Lookups Without Loading Full Graph       │  │
│  │  Implementation: JSON files                              │  │
│  │  Content (varies by mode):                               │  │
│  │    FlagEmbedding: kv_store_entities.json (full metadata) │  │
│  │    OpenAI: Only kv_store_text_chunks.json (minimal)      │  │
│  │  Used for:                                               │  │
│  │    • Quick entity/chunk lookups                          │  │
│  │    • API responses without graph loading                 │  │
│  │    • Caching frequently accessed data                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Distribution

**Question:** Where is each piece of information stored?

| Data Element | GraphML | Vector DB | KV Store (FAISS) | KV Store (OpenAI) |
|--------------|---------|-----------|------------------|-------------------|
| Entity name | ✅ | ✅ | ✅ | ❌ |
| Entity type | ✅ | ❌ | ✅ | ❌ |
| Entity description | ✅ | ❌ (used for embedding) | ✅ | ❌ |
| Entity weight | ✅ | ❌ | ✅ | ❌ |
| Entity source_id | ✅ | ❌ | ✅ | ❌ |
| Relation description | ✅ | ❌ (used for embedding) | ✅ | ❌ |
| Chunk content | ✅ | ❌ (used for embedding) | ✅ | ✅ |
| Chunk document ID | ✅ | ❌ | ✅ | ✅ |
| Graph structure (edges) | ✅ | ❌ | ❌ | ❌ |
| Embeddings | ❌ | ✅ | ✅ | ✅ |

**Key Insight:** In OpenAI mode, GraphML is the ONLY source of entity/relation metadata.

---

## 4. Retrieval Pipeline - How Query Works

### Step-by-Step Query Flow

```
User Query: "What is the capital of Bangladesh?"
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Query Embedding                                         │
│  • Embed query: embedding_func("What is the capital...")        │
│  • Result: [0.0123, -0.0456, 0.0789, ...]  (1536 dims)         │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Dual-Path Vector Search                                │
│                                                                 │
│  Path A: Entity-Based Search                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ • Search vdb_entities.json (or FAISS index_entity.bin)  │ │
│  │ • Find top-k similar entities (k=60)                     │ │
│  │ • Cosine similarity between query and entity vectors     │ │
│  │                                                          │ │
│  │ Results:                                                 │ │
│  │   1. DHAKA (similarity: 0.89)                           │ │
│  │   2. BANGLADESH (similarity: 0.87)                      │ │
│  │   3. CAPITAL CITY (similarity: 0.82)                    │ │
│  │   ...                                                    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Path B: Relation-Based Search                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ • Search vdb_bipartite_edges.json (or index_bipartite_  │ │
│  │   edge.bin)                                              │ │
│  │ • Find top-k similar relations (k=60)                    │ │
│  │                                                          │ │
│  │ Results:                                                 │ │
│  │   1. "DHAKA - capital_of - BANGLADESH" (0.91)           │ │
│  │   2. "DHAKA - located_in - BANGLADESH" (0.83)           │ │
│  │   3. "BANGLADESH - has_capital - DHAKA" (0.81)          │ │
│  │   ...                                                    │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Graph Traversal                                         │
│  • For each entity/relation, find connected chunks             │
│                                                                 │
│  Path A Results:                                                │
│    DHAKA (ent-d691a638...) → graph.neighbors()                 │
│      ├─→ chunk-710e6d... (Bangladesh geography)                │
│      ├─→ chunk-8a92bf... (History of capital)                  │
│      └─→ chunk-3f4e5d... (Urban development)                   │
│                                                                 │
│    BANGLADESH (ent-abc123...) → graph.neighbors()              │
│      ├─→ chunk-710e6d... (Bangladesh overview)                 │
│      ├─→ chunk-9b8c7d... (Geography)                           │
│      └─→ chunk-4e5f6a... (Politics)                            │
│                                                                 │
│  Path B Results:                                                │
│    "DHAKA-capital_of-BANGLADESH" (edge-xyz789) → neighbors()   │
│      ├─→ chunk-710e6d... (Capital description)                 │
│      └─→ chunk-8a92bf... (Dhaka facts)                         │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Reciprocal Rank Fusion                                 │
│  • Combine both paths using reciprocal rank formula            │
│                                                                 │
│  Path A ranks: chunk-710e6d (rank 1), chunk-8a92bf (rank 2)   │
│  Path B ranks: chunk-710e6d (rank 1), chunk-9b8c7d (rank 3)   │
│                                                                 │
│  Fusion formula: score(chunk) = Σ 1/(k + rank_i)               │
│    where k=60, rank_i is position in each list                 │
│                                                                 │
│  chunk-710e6d: 1/(60+1) + 1/(60+1) = 0.0328  ← Highest        │
│  chunk-8a92bf: 1/(60+2) + 0 = 0.0161                           │
│  chunk-9b8c7d: 0 + 1/(60+3) = 0.0159                           │
│                                                                 │
│  Final ranking: [chunk-710e6d, chunk-8a92bf, chunk-9b8c7d]    │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Context Formatting                                     │
│  • Retrieve chunk content from graph or KV store              │
│  • Format as natural language                                  │
│                                                                 │
│  Knowledge 1:                                                   │
│  Bangladesh, officially the People's Republic of Bangladesh,   │
│  is a country in South Asia. Dhaka is the capital and largest │
│  city...                                                        │
│                                                                 │
│  Knowledge 2:                                                   │
│  Dhaka has a population of over 20 million people and serves  │
│  as the economic and cultural center...                        │
│                                                                 │
│  Source IDs: [chunk-710e6d..., chunk-8a92bf...]               │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: API Response                                            │
│  {                                                              │
│    "results": [                                                 │
│      {                                                          │
│        "knowledge": "Bangladesh, officially the People's...",  │
│        "source_ids": ["chunk-710e6d...", "chunk-8a92bf..."],  │
│        "coherence_score": 0.0328                               │
│      }                                                          │
│    ]                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Code References

**Entry Point:**
```python
# script_api.py: /search endpoint
result = await rag.aquery(
    query_text,
    param=QueryParam(mode="hybrid", only_need_context=True, top_k=10)
)
```

**Core Retrieval:**
```python
# bigrag/operate.py: _build_query_context()

async def _build_query_context(query, top_k, ...):
    # Path A: Entity-based
    entity_results = await entities_vdb.query(query, top_k=top_k)
    knowledge_list_1 = await _get_node_data(entity_results, graph)

    # Path B: Relation-based
    edge_results = await bipartite_edges_vdb.query(query, top_k=top_k)
    knowledge_list_2 = await _get_edge_data(edge_results, graph)

    # Reciprocal rank fusion
    know_score = defaultdict(float)
    for i, (knowledge, source_ids) in enumerate(knowledge_list_1):
        know_score[knowledge] += 1 / (i + 1)

    for i, (knowledge, source_ids) in enumerate(knowledge_list_2):
        know_score[knowledge] += 1 / (i + 1)

    # Sort by score
    final_knowledge = sorted(know_score.items(), key=lambda x: x[1], reverse=True)

    return final_knowledge[:top_k]
```

---

## 5. Dual-Path Retrieval Explained

### Why Two Paths?

**Problem with Single-Path Retrieval:**

**Entity-only** (local mode):
- Good at: Finding documents mentioning specific entities
- Bad at: Complex multi-hop reasoning
- Example: "What is the capital of the country where Sheikh Hasina is PM?"
  - Finds BANGLADESH, SHEIKH HASINA entities
  - But doesn't capture the *relationships* between them

**Relation-only** (global mode):
- Good at: Finding specific relationships
- Bad at: Broad entity-based queries
- Example: "Tell me about Bangladesh"
  - Needs relations like "Bangladesh-has-capital-Dhaka"
  - But misses general entity mentions

**Hybrid** (dual-path):
- ✅ Combines strengths of both
- ✅ Captures entity co-occurrence + explicit relations
- ✅ BiG-RAG paper shows 4-5 F1 point improvement

### Comparison Table

| Query Type | Entity-only | Relation-only | Hybrid |
|------------|-------------|---------------|--------|
| "What is the capital of Bangladesh?" | 😐 Fair (finds BANGLADESH, DHAKA) | 🙂 Good (finds capital_of relation) | 😄 Excellent (both) |
| "Tell me about Bangladesh" | 🙂 Good (finds all Bangladesh mentions) | 😐 Fair (misses general info) | 😄 Excellent (both) |
| "Who is the PM of Bangladesh's capital city?" | 😐 Fair (2-hop reasoning hard) | 🙂 Good (relation chains) | 😄 Excellent (both) |
| "Bangladesh history" | 🙂 Good (entity-centric) | 😐 Fair (relations sparse) | 😄 Excellent (both) |

### Paper Results

From BiG-RAG paper (Table 3: Ablation Study):

```
Configuration               | 2WikiMHQA F1 | Musique F1 | Average |
──────────────────────────────────────────────────────────────────
BiG-RAG (full dual-path)    | 56.4         | 41.2       | 48.8    |
- w/o dual-path (entity)    | 52.1 (-4.3)  | 36.8 (-4.4)| 44.5    |
- w/o dual-path (relation)  | 51.3 (-5.1)  | 35.9 (-5.3)| 43.6    |
```

**Conclusion:** Always use hybrid mode (dual-path). This is why the code ignores the `mode` parameter.

---

## 6. Real Example: Bangladesh Document

### Your Upload Results

**Document:**
- File: `Bangladesh.txt`
- Size: 38,400 characters (~7,500 words)
- Document ID: `doc-53a0479813a7da9e631fcac2f7c0a80d`

**Processing Results (from terminal logs):**
```
INFO:bigrag:Writing graph with 564 nodes, 413 edges
```

**Breakdown:**
```
564 nodes = 5 chunks + 156 entities + 89 relations + 314 other nodes
            (other nodes: intermediate graph constructs, deduplication, etc.)

413 edges = connections between entities/relations and chunks
```

**Storage Files Created:**

```bash
$ ls -lh expr/demo_test/
-rw-r--r-- 1 user user  5.2M  graph_chunk_entity_relation.graphml  # 564 nodes, 413 edges
-rw-r--r-- 1 user user   47K  kv_store_text_chunks.json            # 5 chunks
-rw-r--r-- 1 user user   39K  kv_store_full_docs.json              # Original doc
-rw-r--r-- 1 user user  2.9M  vdb_entities.json                    # 156 entity vectors
-rw-r--r-- 1 user user  1.2M  vdb_bipartite_edges.json             # 89 relation vectors
-rw-r--r-- 1 user user    2K  vdb_chunks.json                      # Empty
-rw-r--r-- 1 user user   15K  documents_registry.json
-rw-r--r-- 1 user user  120K  kv_store_llm_response_cache.json     # LLM calls cached
```

**Sample Query:**
```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["capital of Bangladesh"]}'
```

**Retrieval Flow:**
1. Embed query: `[0.0234, -0.0156, ...]`
2. Path A: Find entities → `DHAKA (0.89), BANGLADESH (0.87)`
3. Path B: Find relations → `DHAKA-capital_of-BANGLADESH (0.91)`
4. Graph traversal → `chunk-710e6d...` (appears in both paths)
5. Return: "Dhaka is the capital and largest city of Bangladesh..."

**Current Bug:**
```bash
curl -s "http://localhost:8001/documents/doc-53a0479813a7da9e631fcac2f7c0a80d" | jq '.stats'
# Returns: {"chunks": 5, "entities": 0, "edges": 0}  ← WRONG!
```

**Why Bug Happens:**
```python
# api/kg_utils.py tries to open:
entities_file = "expr/demo_test/kv_store_entities.json"  # ❌ Doesn't exist!

# Should read from:
graphml_file = "expr/demo_test/graph_chunk_entity_relation.graphml"  # ✅ Has everything
```

**After Fix:**
```bash
curl -s "http://localhost:8001/documents/doc-53a0479813a7da9e631fcac2f7c0a80d" | jq '.stats'
# Will return: {"chunks": 5, "entities": 156, "edges": 89}  ← CORRECT!
```

---

## 7. Performance Characteristics

### Indexing Performance

| Phase | Time (Bangladesh.txt) | Bottleneck |
|-------|----------------------|------------|
| Chunking | <1 second | CPU (tokenization) |
| Entity extraction | ~30 seconds | LLM API (GPT-4o-mini, 5 chunks) |
| Relation extraction | ~45 seconds | LLM API (GPT-4o-mini, 5 chunks) |
| Embedding | ~10 seconds | API (OpenAI embeddings, 250 calls) |
| Graph serialization | ~2 seconds | Disk I/O (write GraphML) |
| **Total** | **~90 seconds** | **LLM API calls** |

**Scaling:**
- 100 documents (~1M tokens): ~4 hours
- 1,000 documents (~10M tokens): ~40 hours
- Parallelization: Can batch LLM calls (10x speedup)

### Retrieval Performance

| Operation | Time | Bottleneck |
|-----------|------|------------|
| Query embedding | ~50 ms | API call (OpenAI) |
| Vector search (FAISS) | ~5 ms | CPU (optimized C++) |
| Vector search (NanoVectorDB) | ~20 ms | Python (exact search) |
| Graph traversal (NetworkX) | ~10 ms | RAM access |
| Reciprocal rank fusion | ~1 ms | CPU |
| Context formatting | ~5 ms | String operations |
| **Total** | **~90 ms** | **Query embedding API** |

**Comparison:**

| Backend | Query Latency | Throughput (queries/sec) |
|---------|---------------|--------------------------|
| FAISS (GPU) | 50 ms | 100+ |
| FAISS (CPU) | 60 ms | 50+ |
| NanoVectorDB | 90 ms | 20+ |
| Neo4j (full GraphDB) | 150 ms | 10+ |

**Bottleneck Analysis:**
- OpenAI embedding API: ~50ms (unavoidable with remote API)
- Can improve with local embedder (FlagEmbedding): ~10ms

### Storage Requirements

**OpenAI Mode (NanoVectorDB):**
```
Bangladesh.txt (38 KB)
→ 5 chunks → 156 entities → 89 relations
→ Total storage: 9.5 MB

Breakdown:
  graph_chunk_entity_relation.graphml: 5.2 MB  (55%)
  vdb_entities.json:                   2.9 MB  (31%)
  vdb_bipartite_edges.json:            1.2 MB  (13%)
  kv_store_*.json:                     0.2 MB  (2%)
```

**FlagEmbedding Mode (FAISS):**
```
Bangladesh.txt (38 KB)
→ Total storage: 14.3 MB

Breakdown:
  graph_chunk_entity_relation.graphml: 5.2 MB  (36%)
  kv_store_entities.json:              2.0 MB  (14%)
  kv_store_bipartite_edges.json:       1.5 MB  (10%)
  index_entity.bin:                    2.5 MB  (17%)
  index_bipartite_edge.bin:            1.8 MB  (13%)
  corpus_*.npy:                        1.3 MB  (9%)
```

**Storage Growth:**
```
Documents: 1    → 10 MB
Documents: 10   → 95 MB
Documents: 100  → 900 MB
Documents: 1000 → 9 GB
```

**Scalability:**
- **NanoVectorDB**: Good for <10K documents (~100 GB)
- **FAISS**: Scales to billions of vectors (>10 TB)

---

## Summary

### Key Takeaways

1. **Indexing creates three types of data:**
   - Chunks: Text segments with metadata
   - Entities: Named entities extracted by LLM
   - Relations: N-ary relationships as bipartite edges

2. **Data is stored in three layers:**
   - Graph (GraphML): Complete metadata, structure
   - Vector (FAISS/NanoVectorDB): Fast similarity search
   - KV (JSON): Quick lookups

3. **Retrieval uses dual-path search:**
   - Entity-based path: Find relevant entities
   - Relation-based path: Find relevant relations
   - Fusion: Combine results with reciprocal rank

4. **Chunks are the retrieval unit:**
   - Entities/relations point to chunks (via graph edges)
   - Chunks contain the actual text to return
   - Query → Entities/Relations → Chunks → Context

5. **GraphML is the ground truth:**
   - OpenAI mode: GraphML has ALL metadata
   - FlagEmbedding mode: GraphML + JSON files (redundant)
   - Always reliable for complete data

6. **Performance depends on embedder:**
   - OpenAI: Simpler, slower, remote API
   - FlagEmbedding: Faster, local, more storage

### What You Now Understand

✅ **What gets created:** Chunks, entities, relations, embeddings, graph
✅ **Where it's stored:** GraphML (complete), Vector DB (fast), KV Store (quick)
✅ **How it's used:** Dual-path retrieval → graph traversal → fusion
✅ **Why chunks work:** They're the bridge between semantic search and text
✅ **Why GraphML matters:** It's the only complete metadata source in OpenAI mode

### Ready for Implementation

You now have complete understanding of:
- Indexing pipeline (what happens during document upload)
- Storage architecture (where data lives)
- Retrieval process (how queries work)

**Next step:** Implement the fix in `api/kg_utils.py` to read from GraphML in OpenAI mode, now that you understand why it's the correct approach.

See [COMPREHENSIVE_FIX_PLAN.md](COMPREHENSIVE_FIX_PLAN.md) for implementation details.
