# GraphML in BiG-RAG: Complete Guide

**Date:** October 31, 2025
**Purpose:** Educational guide explaining GraphML's role in BiG-RAG storage architecture

---

## Table of Contents

1. [What is GraphML?](#1-what-is-graphml)
2. [Why BiG-RAG Uses GraphML](#2-why-big-rag-uses-graphml)
3. [What Data is Stored in GraphML](#3-what-data-is-stored-in-graphml)
4. [GraphML vs Other Storage Formats](#4-graphml-vs-other-storage-formats)
5. [Graph Visualization](#5-graph-visualization)
6. [LLM and Embedder Switching](#6-llm-and-embedder-switching)
7. [Working with GraphML Files](#7-working-with-graphml-files)

---

## 1. What is GraphML?

### Definition

**GraphML** is an XML-based file format for representing graph data structures. It's a standard format developed by the graph drawing community for storing:
- Nodes (vertices) with attributes
- Edges (connections) with attributes
- Hierarchical graph structures
- Metadata and custom properties

### File Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="entity_name" for="node" attr.name="entity_name" attr.type="string"/>
  <key id="description" for="node" attr.name="description" attr.type="string"/>
  <key id="source_id" for="node" attr.name="source_id" attr.type="string"/>
  <key id="role" for="node" attr.name="role" attr.type="string"/>

  <graph edgedefault="undirected">
    <!-- Nodes -->
    <node id="ent-abc123">
      <data key="entity_name">Bangladesh</data>
      <data key="description">Country in South Asia</data>
      <data key="source_id">chunk-xyz<SEP>chunk-uvw</data>
      <data key="role">entity</data>
    </node>

    <node id="chunk-xyz">
      <data key="content">Bangladesh is a country...</data>
      <data key="role">chunk</data>
    </node>

    <!-- Edges -->
    <edge source="ent-abc123" target="chunk-xyz"/>
  </graph>
</graphml>
```

### Key Properties

✅ **Human-readable**: XML format, can inspect with text editor
✅ **Standard format**: Supported by many graph tools and libraries
✅ **Rich metadata**: Store arbitrary attributes on nodes/edges
✅ **Tool-agnostic**: Works with NetworkX, Neo4j, Gephi, yEd, etc.
✅ **Self-describing**: Schema embedded in file (key definitions)

---

## 2. Why BiG-RAG Uses GraphML

### Universal Storage Format

BiG-RAG uses GraphML as the **canonical source of truth** for graph structure and metadata. Regardless of which vector database you use (FAISS, NanoVectorDB, Milvus), the complete graph is always saved to GraphML.

### Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   BiG-RAG Storage Layers                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: Graph Database                                       │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  NetworkX (in-memory) → GraphML (persistent)           │   │
│  │  • Complete graph structure                            │   │
│  │  • ALL node attributes (entity_name, type, weight)     │   │
│  │  • ALL edge attributes (relation descriptions)         │   │
│  │  • Source tracking (chunk_id → entity/edge mapping)    │   │
│  └────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  Layer 2: Vector Database                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  FAISS / NanoVectorDB / Milvus                         │   │
│  │  • Entity embeddings (for similarity search)           │   │
│  │  • Edge embeddings (for relation search)               │   │
│  │  • Chunk embeddings (for text search)                  │   │
│  │  ⚠️  Minimal metadata (just names + vectors)           │   │
│  └────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  Layer 3: Key-Value Store                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  JSON files (kv_store_*.json or vdb_*.json)            │   │
│  │  • Quick lookup without loading full graph             │   │
│  │  • Varies by backend (FAISS has more, NanoVectorDB less)│  │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Key Insight: GraphML has EVERYTHING. Other layers optimize for speed.
```

### Why GraphML is Essential

1. **Completeness**: Always contains full metadata regardless of embedder choice
2. **Portability**: Can export/import to other graph databases (Neo4j, etc.)
3. **Debugging**: Human-readable format for inspecting graph structure
4. **Recovery**: Can rebuild vector indices from GraphML if needed
5. **Tool Integration**: Compatible with graph analysis/visualization tools

---

## 3. What Data is Stored in GraphML

### Node Types

BiG-RAG's GraphML contains **three types of nodes**:

#### 1. Entity Nodes (role="entity")

```xml
<node id="ent-d691a638159289c95fd27b91ed6ff9b2">
  <data key="entity_name">DHAKA</data>
  <data key="entity_type">LOCATION</data>
  <data key="description">Capital city of Bangladesh</data>
  <data key="weight">3.5</data>
  <data key="source_id">chunk-710e6d<SEP>chunk-8a92bf</data>
  <data key="role">entity</data>
</node>
```

**Fields:**
- `entity_name`: Extracted entity (e.g., "DHAKA", "Bangladesh")
- `entity_type`: NER type (PERSON, LOCATION, ORGANIZATION, etc.)
- `description`: LLM-generated description
- `weight`: Importance score (based on frequency, context)
- `source_id`: Chunk IDs where this entity appears (separated by `<SEP>`)
- `role`: Always "entity"

#### 2. Relation Nodes (role="bipartite_edge")

```xml
<node id="edge-abc123">
  <data key="bipartite_edge_name">DHAKA - capital_of - BANGLADESH</data>
  <data key="description">Dhaka is the capital and largest city of Bangladesh</data>
  <data key="weight">4.2</data>
  <data key="source_id">chunk-710e6d<SEP>chunk-8a92bf</data>
  <data key="role">bipartite_edge</data>
</node>
```

**Fields:**
- `bipartite_edge_name`: Relation triple (head - relation - tail)
- `description`: Natural language description of the relation
- `weight`: Importance/confidence score
- `source_id`: Chunk IDs where this relation appears
- `role`: Always "bipartite_edge"

#### 3. Chunk Nodes (role="chunk")

```xml
<node id="chunk-710e6d96a7830efebf274af94ee904f2">
  <data key="content">Bangladesh, officially the People's Republic of Bangladesh...</data>
  <data key="full_doc_id">doc-53a0479813a7da9e631fcac2f7c0a80d</data>
  <data key="chunk_order_index">0</data>
  <data key="tokens">1200</data>
  <data key="role">chunk</data>
</node>
```

**Fields:**
- `content`: Actual text content
- `full_doc_id`: Parent document ID
- `chunk_order_index`: Position in document (0, 1, 2, ...)
- `tokens`: Token count
- `role`: Always "chunk"

### Edge Structure

**Edges connect entities/relations to the chunks they appear in:**

```xml
<!-- Entity → Chunk edges -->
<edge source="ent-d691a638159289c95fd27b91ed6ff9b2" target="chunk-710e6d96a7830efebf274af94ee904f2"/>

<!-- Relation → Chunk edges -->
<edge source="edge-abc123" target="chunk-710e6d96a7830efebf274af94ee904f2"/>
```

**Bipartite Graph Property:**
- Entities/Relations connect to Chunks
- Entities do NOT connect directly to other Entities
- Relations do NOT connect directly to other Relations
- Chunks do NOT connect directly to other Chunks

---

## 4. GraphML vs Other Storage Formats

### Comparison Table

| Format | Purpose | Completeness | Speed | When Used |
|--------|---------|--------------|-------|-----------|
| **GraphML** | Persistent graph storage | ✅ 100% complete | 🐌 Slow (XML parsing) | On-disk storage, export, debugging |
| **NetworkX** | In-memory graph operations | ✅ 100% complete | ⚡ Fast (in-memory) | During runtime, graph traversal |
| **FAISS (.bin)** | Fast vector similarity | ❌ Vectors only | ⚡⚡ Very fast | Entity/edge/chunk similarity search |
| **NanoVectorDB (.json)** | All-in-one vector DB | ❌ Minimal metadata | ⚡ Fast | OpenAI embedding mode |
| **JSON (kv_store_*.json)** | Quick metadata lookup | ✅ Full metadata (FAISS mode) | ⚡ Fast | FlagEmbedding mode metadata |

### Storage Mode Differences

#### FlagEmbedding Mode (FAISS)

```
expr/demo_test/
├── graph_chunk_entity_relation.graphml  ✅ 100% complete
├── kv_store_entities.json               ✅ Full metadata (redundant with GraphML)
├── kv_store_bipartite_edges.json        ✅ Full metadata (redundant with GraphML)
├── kv_store_text_chunks.json            ✅ Chunk metadata
├── index_entity.bin                     ⚡ FAISS index (vectors only)
├── index_bipartite_edge.bin             ⚡ FAISS index (vectors only)
└── corpus_entity.npy                    ⚡ Raw embeddings
```

**Pros:**
- Duplicate storage (GraphML + JSON) for faster API queries
- FAISS indices extremely fast for large-scale retrieval

**Cons:**
- More disk space (data stored twice)
- More complex to maintain consistency

#### OpenAI Mode (NanoVectorDB)

```
expr/demo_test/
├── graph_chunk_entity_relation.graphml  ✅ 100% complete
├── kv_store_text_chunks.json            ✅ Chunk metadata
├── kv_store_full_docs.json              ✅ Original documents
├── vdb_entities.json                    ⚠️  Names + vectors only
├── vdb_bipartite_edges.json             ⚠️  Names + vectors only
└── vdb_chunks.json                      ⚠️  Empty or minimal
```

**Pros:**
- Less disk space (GraphML is the only complete copy)
- Simpler architecture (fewer files)
- Pure Python (no C++ dependencies)

**Cons:**
- Must read GraphML for full metadata (slightly slower)
- NanoVectorDB slower than FAISS for large datasets

**Key Insight:** In OpenAI mode, `vdb_*.json` files are NOT metadata stores - they're just vector indices with minimal fields. GraphML is the ONLY source of complete metadata.

---

## 5. Graph Visualization

### Why Visualize?

Graph visualization helps you:
- Understand entity relationships
- Debug extraction quality
- Identify dense/sparse regions
- Present results to stakeholders

### Tools for GraphML Visualization

#### 1. yEd Graph Editor (Recommended for Quick Views)

**Installation:** Download from [yWorks](https://www.yworks.com/products/yed)

**Usage:**
```bash
# Open GraphML file
File → Open → graph_chunk_entity_relation.graphml

# Apply automatic layout
Layout → Hierarchical
Layout → Organic (for entity networks)

# Filter by node type
Edit → Properties Mapper → role == "entity"
```

**Pros:**
- Free, works on Windows/Mac/Linux
- Fast rendering
- Many layout algorithms
- Export to PNG/PDF/SVG

**Cons:**
- Desktop app (not web-based)
- Limited to ~50K nodes

#### 2. Gephi (Best for Large Graphs)

**Installation:** Download from [gephi.org](https://gephi.org/)

**Usage:**
```bash
# Import GraphML
File → Open → graph_chunk_entity_relation.graphml

# Apply layout
Layout → Force Atlas 2 (for entity clustering)
Layout → OpenORD (for large graphs)

# Color by node type
Appearance → Nodes → Partition → role
```

**Pros:**
- Handles millions of nodes
- Beautiful visualizations
- Network analysis tools (centrality, clustering)
- Export high-res images

**Cons:**
- Java-based (can be slow)
- Steeper learning curve

#### 3. NetworkX + Matplotlib (Python)

**Code:**
```python
import networkx as nx
import matplotlib.pyplot as plt

# Load GraphML
graph = nx.read_graphml("expr/demo_test/graph_chunk_entity_relation.graphml")

# Filter to entities only (for cleaner visualization)
entity_nodes = [n for n, d in graph.nodes(data=True) if d.get("role") == "entity"]
entity_graph = graph.subgraph(entity_nodes)

# Draw
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(entity_graph, k=0.5, iterations=50)
nx.draw(entity_graph, pos,
        node_size=50,
        node_color='lightblue',
        with_labels=True,
        font_size=8)
plt.savefig("entity_graph.png", dpi=300)
plt.show()
```

**Pros:**
- Scriptable, reproducible
- Easy to filter/transform before visualizing
- Integrate with data analysis pipeline

**Cons:**
- Slower than specialized tools
- Not interactive

#### 4. Pyvis (Interactive Web Visualization)

**Installation:**
```bash
pip install pyvis
```

**Code:**
```python
from pyvis.network import Network
import networkx as nx

# Load GraphML
graph = nx.read_graphml("expr/demo_test/graph_chunk_entity_relation.graphml")

# Convert to Pyvis (filter to first 100 nodes for performance)
nodes = list(graph.nodes(data=True))[:100]
subgraph = graph.subgraph([n[0] for n in nodes])

net = Network(height="750px", width="100%", notebook=True)
net.from_nx(subgraph)

# Color by role
for node in net.nodes:
    role = graph.nodes[node['id']].get('role', 'unknown')
    if role == 'entity':
        node['color'] = 'lightblue'
    elif role == 'bipartite_edge':
        node['color'] = 'lightgreen'
    elif role == 'chunk':
        node['color'] = 'lightcoral'

net.show("graph.html")
```

**Pros:**
- Interactive (zoom, pan, click)
- Works in Jupyter notebooks
- Shareable HTML files

**Cons:**
- Slow for large graphs (>1000 nodes)

### Example Visualizations

**Entity Co-occurrence Network:**
```python
import networkx as nx

# Load graph
graph = nx.read_graphml("expr/demo_test/graph_chunk_entity_relation.graphml")

# Build entity co-occurrence graph
entity_cooccurrence = nx.Graph()
chunks = [n for n, d in graph.nodes(data=True) if d.get("role") == "chunk"]

for chunk in chunks:
    # Get entities in this chunk
    entities = [n for n in graph.neighbors(chunk)
                if graph.nodes[n].get("role") == "entity"]

    # Connect co-occurring entities
    for i, e1 in enumerate(entities):
        for e2 in entities[i+1:]:
            if entity_cooccurrence.has_edge(e1, e2):
                entity_cooccurrence[e1][e2]['weight'] += 1
            else:
                entity_cooccurrence.add_edge(e1, e2, weight=1)

# Visualize
nx.write_graphml(entity_cooccurrence, "entity_cooccurrence.graphml")
```

---

## 6. LLM and Embedder Switching

### Key Question: Can I Switch LLMs/Embedders Without Rebuilding?

**Short Answer:**
- ✅ **Switching LLMs (GPT-4 → Claude → Llama)**: NO rebuild needed for retrieval (but affects quality of NEW extractions)
- ❌ **Switching Embedders (OpenAI → FlagEmbedding)**: MUST rebuild vector indices (but GraphML stays same)

### Detailed Explanation

#### Switching LLMs (for Entity/Relation Extraction)

**What LLMs do in BiG-RAG:**
```
Document → LLM extracts entities/relations → Store in graph
```

**When you switch LLMs (e.g., GPT-4o-mini → Claude Sonnet):**

✅ **Existing data preserved:**
- GraphML file unchanged
- All previously extracted entities/relations remain
- Retrieval still works perfectly

⚠️ **New extractions differ:**
- New documents processed with new LLM
- May extract different entities/relations (better or worse)
- Graph quality depends on LLM capabilities

**Example:**
```bash
# Build graph with GPT-4o-mini
python script_build.py --llm openai --data_source demo_test

# Later, switch to Claude for new documents
# (modify script_api.py to use Claude)
# Upload new document → extracts with Claude
# Old entities (GPT) + new entities (Claude) coexist in same GraphML
```

**Recommendation:** Stick with one LLM per dataset for consistency. If switching, consider rebuilding entire graph.

#### Switching Embedders (for Vector Similarity)

**What Embedders do in BiG-RAG:**
```
Entity/Relation/Chunk text → Embedder → Vector → Store in FAISS/NanoVectorDB
```

**When you switch embedders (e.g., OpenAI → FlagEmbedding):**

❌ **Must rebuild vector indices:**
- Different models produce different vector dimensions
- OpenAI text-embedding-3-large: 1536 dims
- FlagEmbedding bge-large-en-v1.5: 1024 dims
- Incompatible - cannot mix

✅ **GraphML stays same:**
- Graph structure unchanged
- Entity/relation metadata unchanged
- Only vectors need regeneration

**How to switch:**

```bash
# Current: OpenAI embeddings (NanoVectorDB)
expr/demo_test/
├── graph_chunk_entity_relation.graphml  ← Keep this
├── vdb_entities.json                    ← Delete
├── vdb_bipartite_edges.json             ← Delete
└── vdb_chunks.json                      ← Delete

# Rebuild with FlagEmbedding:
# 1. Modify bigrag/bigrag.py to use FlagEmbedding
# 2. Load GraphML and re-generate embeddings
# 3. New files created:
expr/demo_test/
├── graph_chunk_entity_relation.graphml  ← Same as before
├── kv_store_entities.json               ← New
├── kv_store_bipartite_edges.json        ← New
├── index_entity.bin                     ← New (FAISS)
├── index_bipartite_edge.bin             ← New (FAISS)
└── corpus_entity.npy                    ← New (embeddings)
```

**Code to rebuild vectors from GraphML:**

```python
import networkx as nx
from bigrag import BiGRAG

# Load existing GraphML
graph = nx.read_graphml("expr/demo_test/graph_chunk_entity_relation.graphml")

# Initialize BiGRAG with new embedder
rag = BiGRAG(
    working_dir="expr/demo_test_flagembedding",
    embedding_func="flag",  # Switch to FlagEmbedding
)

# Extract entities from GraphML
entities = []
for node_id, node_data in graph.nodes(data=True):
    if node_data.get("role") == "entity":
        entities.append({
            "entity_name": node_data.get("entity_name"),
            "description": node_data.get("description"),
            "entity_type": node_data.get("entity_type"),
        })

# Re-embed and index
await rag._insert_entities(entities)

# Similarly for relations and chunks...
```

### Migration Summary

| Change | GraphML | Vector DB | Rebuild Required? | Time |
|--------|---------|-----------|-------------------|------|
| Switch LLM (same embedder) | ✅ Keep | ✅ Keep | ❌ No (for old data) | Instant |
| Switch Embedder (same LLM) | ✅ Keep | ❌ Delete | ✅ Yes (re-embed) | Minutes to hours |
| Both LLM + Embedder | ❌ Rebuild | ❌ Rebuild | ✅ Yes (full rebuild) | Hours |

---

## 7. Working with GraphML Files

### Loading GraphML in Python

```python
import networkx as nx

# Load graph
graph = nx.read_graphml("expr/demo_test/graph_chunk_entity_relation.graphml")

# Basic stats
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# Count by role
from collections import Counter
roles = Counter(d.get("role") for n, d in graph.nodes(data=True))
print(f"Entities: {roles['entity']}")
print(f"Relations: {roles['bipartite_edge']}")
print(f"Chunks: {roles['chunk']}")
```

### Querying GraphML

```python
# Find all entities in a specific document
document_id = "doc-53a0479813a7da9e631fcac2f7c0a80d"

# Step 1: Get chunk IDs for this document
chunks = [
    node_id for node_id, data in graph.nodes(data=True)
    if data.get("role") == "chunk" and data.get("full_doc_id") == document_id
]

# Step 2: Get entities connected to these chunks
entities = []
for chunk_id in chunks:
    for neighbor in graph.neighbors(chunk_id):
        node_data = graph.nodes[neighbor]
        if node_data.get("role") == "entity":
            entities.append({
                "name": node_data.get("entity_name"),
                "type": node_data.get("entity_type"),
                "description": node_data.get("description"),
            })

# Remove duplicates
entities = {e['name']: e for e in entities}.values()
print(f"Found {len(entities)} entities")
```

### Exporting GraphML to Other Formats

```python
# Export to JSON (for web visualization)
import json
from networkx.readwrite import json_graph

graph_json = json_graph.node_link_data(graph)
with open("graph.json", "w") as f:
    json.dump(graph_json, f)

# Export to CSV (nodes and edges separately)
import pandas as pd

# Nodes
nodes = [
    {"id": n, **d}
    for n, d in graph.nodes(data=True)
]
pd.DataFrame(nodes).to_csv("nodes.csv", index=False)

# Edges
edges = [
    {"source": u, "target": v, **d}
    for u, v, d in graph.edges(data=True)
]
pd.DataFrame(edges).to_csv("edges.csv", index=False)

# Export to Neo4j Cypher script
def generate_neo4j_cypher(graph):
    cypher = []

    # Create nodes
    for node_id, data in graph.nodes(data=True):
        props = ", ".join(f"{k}: '{v}'" for k, v in data.items())
        cypher.append(f"CREATE (n:{data['role']} {{id: '{node_id}', {props}}})")

    # Create relationships
    for u, v in graph.edges():
        cypher.append(f"MATCH (a {{id: '{u}'}}), (b {{id: '{v}'}}) CREATE (a)-[:MENTIONS]->(b)")

    return "\n".join(cypher)

with open("import.cypher", "w") as f:
    f.write(generate_neo4j_cypher(graph))
```

---

## Summary

### Key Takeaways

1. **GraphML is the canonical storage format** - always contains 100% of graph data
2. **Vector databases optimize for speed** - but sacrifice completeness
3. **Storage mode differences are minimal** - GraphML unifies both
4. **Visualization tools exist** - yEd, Gephi, NetworkX, Pyvis
5. **LLM switching affects quality** - but doesn't break existing data
6. **Embedder switching requires rebuild** - but GraphML stays same

### When to Use GraphML

✅ **Use GraphML when:**
- API queries need full metadata (entities, edges, descriptions)
- Debugging graph construction issues
- Exporting data to other systems
- Switching embedders (re-embed from GraphML)
- Visualizing graph structure

❌ **Don't use GraphML for:**
- Fast similarity search (use vector DB instead)
- Real-time retrieval (use FAISS/NanoVectorDB)
- Large-scale production queries (too slow)

### Implementation Recommendation

For your current issue (document stats showing 0), the solution is clear:

**Read from GraphML when other formats don't have complete metadata.**

This is exactly what we'll implement in `api/kg_utils.py` - detect storage mode, and use GraphML as fallback for OpenAI mode.

---

**Next:** Read [RETRIEVAL_PROCESS_EXPLAINED.md](RETRIEVAL_PROCESS_EXPLAINED.md) to understand how all storage layers work together during retrieval.
