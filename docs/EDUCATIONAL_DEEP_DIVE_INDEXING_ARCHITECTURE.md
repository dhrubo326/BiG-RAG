# Educational Deep Dive: BiG-RAG Indexing Processes and Architecture

**Author:** Technical Analysis for Understanding BiG-RAG Knowledge Graph Construction
**Date:** January 2025
**Purpose:** Comprehensive educational guide to BiG-RAG's indexing and graph building
**Level:** From fundamentals to advanced implementation details

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Foundational Concepts](#foundational-concepts)
3. [Complete Indexing Pipeline](#complete-indexing-pipeline)
4. [Document Processing and Chunking](#document-processing-and-chunking)
5. [Entity Extraction](#entity-extraction)
6. [Relation Extraction](#relation-extraction)
7. [Graph Construction](#graph-construction)
8. [Vector Index Building](#vector-index-building)
9. [Storage Architecture](#storage-architecture)
10. [Performance and Scalability](#performance-and-scalability)
11. [Implementation Guide](#implementation-guide)

---

## Executive Summary

### What is Indexing in BiG-RAG?

**Indexing** is the process of transforming raw documents into a queryable bipartite knowledge graph with vector indices.

```
Raw Documents (corpus.jsonl)
         ↓
[INDEXING PIPELINE]
         ↓
Bipartite Knowledge Graph + Vector Indices
```

### The Complete Pipeline

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **1. Chunking** | Raw documents | Split into 1200-token chunks (100 overlap) | Text chunks |
| **2. Entity Extraction** | Text chunks | spaCy NER + LLM extraction | Entity list |
| **3. Entity Resolution** | Entities | Cosine similarity merging (>0.90) | Deduplicated entities |
| **4. Relation Extraction** | Text chunks | LLM n-ary relation extraction | Relation list |
| **5. Graph Construction** | Entities + Relations | Build bipartite graph (E ↔ R) | NetworkX graph |
| **6. Embedding** | Entities + Relations + Chunks | bge-large-en-v1.5 (1536-dim) | Vector embeddings |
| **7. Vector Indexing** | Embeddings | FAISS index construction | 3 FAISS indices |
| **8. KV Storage** | Metadata | JSON serialization | 2 KV stores |

### Key Statistics

| Metric | Value (per 10K documents) |
|--------|---------------------------|
| **Processing Time** | 50-70 minutes |
| **Storage Size** | ~2.2 GB total |
| **Entity Count** | ~8,000-12,000 entities |
| **Relation Count** | ~15,000-20,000 relations |
| **Graph Edges** | ~60,000-80,000 edges |
| **LLM API Calls** | ~10,000-15,000 (GPT-4o-mini) |
| **Cost** | ~$0.50-$1.50 USD |

### Core Architecture

**Storage Components:**
1. **Graph Storage** - `bigrag_graph.graphml` (NetworkX bipartite graph)
2. **Vector Storage** - 3 FAISS indices (entities, relations, chunks)
3. **KV Storage** - 2 JSON files (entity metadata, relation metadata)

---

## Foundational Concepts

### What is a Knowledge Graph?

**Traditional Knowledge Graph (Binary Relations):**
```
(Subject, Predicate, Object)

Example:
(Einstein, born_in, Germany)
(Einstein, developed, Relativity)
```

**Problem:** Real facts involve >2 entities.

**BiG-RAG Knowledge Graph (N-ary Relations):**
```
Relation Node:
  "Male hypertensive patients with creatinine 115-133 µmol/L
   indicate mild serum creatinine elevation"

Connected to entities:
  - "MALE PATIENTS"
  - "HYPERTENSION"
  - "SERUM CREATININE 115-133 ΜMOL/L"
  - "MILD SERUM CREATININE ELEVATION"
```

**Benefit:** Complete semantic context preserved!

### Why Bipartite Graph Encoding?

**Theoretical Foundation:**

A bipartite graph `G_B = (V₁ ∪ V₂, E)` has two disjoint node partitions:
- `V₁` = Entity nodes
- `V₂` = Relation nodes
- `E` = Edges connecting `V₁` ↔ `V₂` (never within same partition)

**Advantages:**
1. ✅ **Lossless transformation** from hypergraph
2. ✅ **Efficient storage** - Standard graph algorithms work
3. ✅ **Flexible retrieval** - Can search entities OR relations
4. ✅ **Natural traversal** - BFS alternates between partitions

**Mathematical Proof:**
```
Every hypergraph H = (V, E_H) where E_H contains hyperedges
can be bijectively mapped to a bipartite graph B = (V ∪ E_H, E_B)

Transformation Φ:
  - V → V (entity nodes preserved)
  - E_H → E_H (hyperedges become relation nodes)
  - For each hyperedge e ∈ E_H connecting {v₁, v₂, ..., vₙ}:
    Create edges: (v₁, e), (v₂, e), ..., (vₙ, e)
```

### Storage Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ BiG-RAG Storage Architecture                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. GRAPH STORAGE (Topology)                                │
│    File: bigrag_graph.graphml                              │
│    Format: NetworkX GraphML (XML)                          │
│    Purpose: Who connects to what                           │
│    Size: ~50-100 MB per 10K docs                           │
│                                                             │
│ 2. VECTOR STORAGE (Similarity Search)                      │
│    Files:                                                   │
│      - index_entity.bin          (FAISS)                   │
│      - index_bipartite_edge.bin  (FAISS)                   │
│      - index_text_chunks.bin     (FAISS)                   │
│    Format: FAISS flat inner product index                  │
│    Purpose: Fast vector similarity search                  │
│    Size: ~1.5-1.8 GB per 10K docs                          │
│                                                             │
│ 3. KEY-VALUE STORAGE (Metadata)                            │
│    Files:                                                   │
│      - kv_store_entities.json                              │
│      - kv_store_bipartite_edges.json                       │
│    Format: JSON dictionary                                 │
│    Purpose: Rich metadata (descriptions, types, sources)   │
│    Size: ~200-400 MB per 10K docs                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete Indexing Pipeline

### Overview

```
┌────────────────────────────────────────────────────────────────┐
│ BiG-RAG End-to-End Indexing Pipeline                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ INPUT: corpus.jsonl                                           │
│   {"id": "doc_001", "contents": "Albert Einstein...", ...}   │
│   {"id": "doc_002", "contents": "In 1905, Einstein...", ...} │
│   ... (10,000 documents)                                      │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ STAGE 1: DOCUMENT PROCESSING (5-10 min)                 │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ • Load documents from JSONL                              │ │
│ │ • Deduplicate by content hash                            │ │
│ │ • Store full documents in kv_store_full_docs.json        │ │
│ │                                                            │ │
│ │ Output: 10,000 unique documents                          │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ STAGE 2: CHUNKING (10-15 min)                           │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ • Split each document into chunks                        │ │
│ │   - Chunk size: 1200 tokens                              │ │
│ │   - Overlap: 100 tokens                                  │ │
│ │   - Tokenizer: tiktoken (GPT-4 tokenizer)                │ │
│ │ • Assign chunk IDs: "{doc_id}_chunk_{i}"                 │ │
│ │ • Store chunks in kv_store_text_chunks.json              │ │
│ │                                                            │ │
│ │ Output: ~50,000 text chunks (avg 5 chunks/doc)           │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ STAGE 3: ENTITY EXTRACTION (20-30 min, LLM-heavy)       │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ Pass 1: spaCy NER                                        │ │
│ │   • Extract named entities (PERSON, ORG, GPE, DATE...)  │ │
│ │   • Fast but incomplete                                  │ │
│ │                                                            │ │
│ │ Pass 2: LLM Extraction (GPT-4o-mini)                     │ │
│ │   • Prompt: "Extract all important entities from: {text}"│ │
│ │   • Returns: JSON list of entities                       │ │
│ │   • Confidence scores assigned                           │ │
│ │                                                            │ │
│ │ Pass 3: Entity Resolution                                │ │
│ │   • Embed all entities with bge-large-en-v1.5            │ │
│ │   • Compute pairwise cosine similarity                   │ │
│ │   • Merge entities with similarity > 0.90                │ │
│ │   • Example: "Einstein" + "Albert Einstein" → "EINSTEIN"│ │
│ │                                                            │ │
│ │ Output: ~10,000 unique entities                          │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ STAGE 4: RELATION EXTRACTION (15-25 min, LLM-heavy)     │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ • LLM Prompt (GPT-4o-mini):                              │ │
│ │   "Extract all n-ary relations from: {text}"             │ │
│ │   "Return JSON: {entities: [...], description: '...'}"   │ │
│ │                                                            │ │
│ │ • Example extraction:                                     │ │
│ │   Input: "Einstein published relativity theory in 1905" │ │
│ │   Output:                                                 │ │
│ │     {                                                     │ │
│ │       "entities": ["EINSTEIN", "RELATIVITY THEORY", "1905"],│ │
│ │       "description": "Einstein published relativity theory in 1905",│ │
│ │       "keywords": ["published", "developed"],            │ │
│ │       "confidence": 0.95                                  │ │
│ │     }                                                     │ │
│ │                                                            │ │
│ │ Output: ~18,000 n-ary relations                          │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ STAGE 5: GRAPH CONSTRUCTION (2-5 min)                   │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ • Create NetworkX bipartite graph                        │ │
│ │ • Add entity nodes (partition 0)                         │ │
│ │ • Add relation nodes (partition 1)                       │ │
│ │ • Add edges: entity ↔ relation                           │ │
│ │ • Save as bigrag_graph.graphml                           │ │
│ │                                                            │ │
│ │ Output: Graph with ~28,000 nodes, ~70,000 edges          │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ STAGE 6: EMBEDDING (10-15 min)                          │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ • Embed entities with bge-large-en-v1.5                  │ │
│ │   - Input: Entity name + description                     │ │
│ │   - Output: 1536-dim vector                              │ │
│ │                                                            │ │
│ │ • Embed relations with bge-large-en-v1.5                 │ │
│ │   - Input: Relation description                          │ │
│ │   - Output: 1536-dim vector                              │ │
│ │                                                            │ │
│ │ • Embed text chunks with bge-large-en-v1.5               │ │
│ │   - Input: Chunk content                                 │ │
│ │   - Output: 1536-dim vector                              │ │
│ │                                                            │ │
│ │ Output: 3 embedding matrices (numpy arrays)              │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ STAGE 7: VECTOR INDEXING (1-3 min)                      │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ • Build FAISS index for entities                         │ │
│ │   - Index type: Flat (exact search)                      │ │
│ │   - Metric: Inner product (cosine similarity)            │ │
│ │   - Save: index_entity.bin                               │ │
│ │                                                            │ │
│ │ • Build FAISS index for relations                        │ │
│ │   - Same configuration                                   │ │
│ │   - Save: index_bipartite_edge.bin                       │ │
│ │                                                            │ │
│ │ • Build FAISS index for chunks                           │ │
│ │   - Same configuration                                   │ │
│ │   - Save: index_text_chunks.bin                          │ │
│ │                                                            │ │
│ │ Output: 3 FAISS indices ready for retrieval              │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ FINAL OUTPUT:                                                 │
│   ✅ bigrag_graph.graphml (bipartite graph topology)         │
│   ✅ index_entity.bin (entity vector index)                  │
│   ✅ index_bipartite_edge.bin (relation vector index)        │
│   ✅ index_text_chunks.bin (chunk vector index)              │
│   ✅ kv_store_entities.json (entity metadata)                │
│   ✅ kv_store_bipartite_edges.json (relation metadata)       │
│   ✅ kv_store_text_chunks.json (chunk metadata)              │
│   ✅ kv_store_full_docs.json (original documents)            │
│                                                                │
│ TOTAL TIME: 50-70 minutes for 10,000 documents               │
│ TOTAL STORAGE: ~2.2 GB                                        │
└────────────────────────────────────────────────────────────────┘
```

---

## Document Processing and Chunking

### Why Chunking?

**Problem:** LLMs have context limits (e.g., GPT-4: 128K tokens, but expensive).

**Solution:** Split documents into smaller, overlapping chunks.

**Benefits:**
1. ✅ **Manageable size** - Each chunk fits in LLM context
2. ✅ **Precision** - Retrieve only relevant sections
3. ✅ **Overlap** - Preserve context across boundaries

### Chunking Strategy

**Parameters:**
```python
CHUNK_SIZE = 1200 tokens      # Main chunk content
OVERLAP_SIZE = 100 tokens     # Overlap with next chunk
TOKENIZER = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
```

**Example:**
```
Original Document (5000 tokens):
[===========================================]

After Chunking:
Chunk 1: [       1200 tokens       ]
Chunk 2:    [ overlap ][   1200 tokens    ]
Chunk 3:               [ overlap ][  1200 tokens  ]
Chunk 4:                          [ overlap ][ 1100 tokens ]

Total: 4 chunks with 100-token overlaps
```

### Implementation

```python
def chunk_document(document: str, chunk_size=1200, overlap=100):
    """
    Split document into overlapping chunks
    """
    # Tokenize
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(document)

    chunks = []
    start = 0

    while start < len(tokens):
        # Extract chunk
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append({
            "id": f"{doc_id}_chunk_{len(chunks)}",
            "content": chunk_text,
            "tokens": len(chunk_tokens),
            "start_token": start,
            "end_token": end
        })

        # Move to next chunk (with overlap)
        start += (chunk_size - overlap)

    return chunks
```

### Storage Format

**kv_store_text_chunks.json:**
```json
{
  "doc_001_chunk_0": {
    "id": "doc_001_chunk_0",
    "content": "Albert Einstein (1879-1955) was a German-born...",
    "tokens": 1200,
    "chunk_order_index": 0,
    "full_doc_id": "doc_001"
  },
  "doc_001_chunk_1": {
    "id": "doc_001_chunk_1",
    "content": "...physicist who developed the theory of relativity...",
    "tokens": 1200,
    "chunk_order_index": 1,
    "full_doc_id": "doc_001"
  }
}
```

---

## Entity Extraction

### Multi-Pass Extraction Strategy

**Why Multiple Passes?**

1. **spaCy NER** - Fast but misses domain-specific entities
2. **LLM Extraction** - Accurate but expensive
3. **Hybrid Approach** - Best of both worlds

### Pass 1: spaCy Named Entity Recognition

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities_spacy(text: str):
    """
    Fast named entity recognition with spaCy
    """
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "type": ent.label_,  # PERSON, ORG, GPE, DATE, etc.
            "confidence": 0.8    # spaCy default confidence
        })

    return entities
```

**Example:**
```
Input: "Albert Einstein published the theory of relativity in 1905."

spaCy Output:
[
  {"text": "Albert Einstein", "type": "PERSON", "confidence": 0.8},
  {"text": "1905", "type": "DATE", "confidence": 0.8}
]

Missing: "theory of relativity" (not a named entity in spaCy taxonomy)
```

### Pass 2: LLM Extraction

```python
async def extract_entities_llm(text: str, llm_model):
    """
    Comprehensive entity extraction with GPT-4o-mini
    """
    prompt = f"""Extract ALL important entities from the following text.
Include people, organizations, concepts, theories, dates, locations, etc.

Text: {text}

Return JSON format:
{{
  "entities": [
    {{"name": "entity name", "type": "entity type", "confidence": 0.0-1.0}}
  ]
}}
"""

    response = await llm_model(prompt)
    entities = json.loads(response)["entities"]

    return entities
```

**Example:**
```
Input: "Albert Einstein published the theory of relativity in 1905."

LLM Output:
[
  {"name": "Albert Einstein", "type": "PERSON", "confidence": 0.95},
  {"name": "theory of relativity", "type": "SCIENTIFIC_THEORY", "confidence": 0.98},
  {"name": "1905", "type": "DATE", "confidence": 0.90},
  {"name": "publication", "type": "EVENT", "confidence": 0.85}
]
```

### Pass 3: Entity Resolution (Deduplication)

**Problem:** Same entity appears with different names.

**Examples:**
- "Einstein" vs. "Albert Einstein" vs. "A. Einstein"
- "USA" vs. "United States" vs. "United States of America"

**Solution:** Cosine similarity-based merging.

```python
def resolve_entities(entities: list, threshold=0.90):
    """
    Merge similar entities using embedding similarity
    """
    # Embed all entity names
    entity_names = [e["name"] for e in entities]
    embeddings = embedding_model.encode(entity_names)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Find merge groups
    merged = []
    visited = set()

    for i in range(len(entities)):
        if i in visited:
            continue

        # Find all similar entities
        similar_indices = np.where(similarity_matrix[i] > threshold)[0]

        # Merge group
        group = [entities[j] for j in similar_indices]
        canonical_entity = select_canonical(group)  # Longest name, highest confidence

        merged.append(canonical_entity)
        visited.update(similar_indices)

    return merged
```

**Example:**
```
Before Resolution:
[
  {"name": "Einstein", "type": "PERSON", "confidence": 0.85},
  {"name": "Albert Einstein", "type": "PERSON", "confidence": 0.95},
  {"name": "A. Einstein", "type": "PERSON", "confidence": 0.80}
]

Similarity Matrix:
              Einstein  Albert Einstein  A. Einstein
Einstein         1.00         0.94           0.91
Albert Einstein  0.94         1.00           0.89
A. Einstein      0.91         0.89           1.00

After Resolution (threshold=0.90):
[
  {"name": "ALBERT EINSTEIN", "type": "PERSON", "confidence": 0.95}
]
```

### Storage Format

**kv_store_entities.json:**
```json
{
  "ALBERT_EINSTEIN": {
    "entity_name": "ALBERT EINSTEIN",
    "entity_type": "PERSON",
    "description": "German-born theoretical physicist (1879-1955)",
    "source_id": "doc_001_chunk_0",
    "confidence": 0.95,
    "aliases": ["Einstein", "A. Einstein"],
    "content": "ALBERT EINSTEIN - German-born theoretical physicist who developed the theory of relativity"
  }
}
```

---

## Relation Extraction

### N-ary Relation Extraction with LLM

**Goal:** Extract complete multi-entity facts, not just binary triples.

**Prompt Design:**
```python
async def extract_relations_llm(text: str, entities: list, llm_model):
    """
    Extract n-ary relations from text
    """
    prompt = f"""Extract all meaningful relations from the following text.
A relation should connect multiple entities and describe their interaction.

Known entities: {', '.join([e['name'] for e in entities])}

Text: {text}

Return JSON format:
{{
  "relations": [
    {{
      "entities": ["entity1", "entity2", ...],
      "description": "natural language description of the relation",
      "keywords": ["keyword1", "keyword2"],
      "confidence": 0.0-1.0
    }}
  ]
}}

Requirements:
- Each relation should involve 2 or more entities
- Description should be a complete, grammatical sentence
- Keywords should be action verbs or key predicates
- Preserve the complete semantic context
"""

    response = await llm_model(prompt)
    relations = json.loads(response)["relations"]

    return relations
```

**Example Extraction:**

```
Input Text:
"Male hypertensive patients with serum creatinine levels between 115-133 µmol/L
 are diagnosed with mild serum creatinine elevation."

Entities (from previous extraction):
- "MALE PATIENTS"
- "HYPERTENSION"
- "SERUM CREATININE 115-133 ΜMOL/L"
- "MILD SERUM CREATININE ELEVATION"

LLM Output:
{
  "relations": [
    {
      "entities": [
        "MALE PATIENTS",
        "HYPERTENSION",
        "SERUM CREATININE 115-133 ΜMOL/L",
        "MILD SERUM CREATININE ELEVATION"
      ],
      "description": "Male hypertensive patients with serum creatinine levels between 115-133 µmol/L are diagnosed with mild serum creatinine elevation",
      "keywords": ["diagnosed", "indicate", "elevation"],
      "confidence": 0.95
    }
  ]
}
```

### Why N-ary Relations Matter

**Binary Triple Decomposition (BAD):**
```
(Male Patients, has_gender, Male)
(Male Patients, has_condition, Hypertension)
(Male Patients, has_lab_value, 115-133 µmol/L)
(Male Patients, diagnosed_with, Mild Elevation)
```

**Problem:** Lost constraint that ALL conditions must co-occur!

**N-ary Relation (GOOD):**
```
Relation: "Male hypertensive patients with serum creatinine 115-133 µmol/L
           indicate mild serum creatinine elevation"

Connected to: ["MALE PATIENTS", "HYPERTENSION",
               "SERUM CREATININE 115-133 ΜMOL/L",
               "MILD SERUM CREATININE ELEVATION"]
```

**Benefit:** Complete semantic context preserved in natural language!

### Storage Format

**kv_store_bipartite_edges.json:**
```json
{
  "relation_001": {
    "id": "relation_001",
    "src_id": "MALE_PATIENTS",  // Primary entity (first in list)
    "tgt_id": "MILD_SERUM_CREATININE_ELEVATION",  // Secondary entity
    "keywords": "diagnosed, indicate, elevation",
    "weight": 0.95,
    "description": "Male hypertensive patients with serum creatinine levels between 115-133 µmol/L are diagnosed with mild serum creatinine elevation",
    "source_id": "doc_medical_001_chunk_5",
    "confidence": 0.95,
    "connected_entities": [
      "MALE_PATIENTS",
      "HYPERTENSION",
      "SERUM_CREATININE_115_133_ΜMOL_L",
      "MILD_SERUM_CREATININE_ELEVATION"
    ],
    "content": "Male hypertensive patients with serum creatinine levels between 115-133 µmol/L are diagnosed with mild serum creatinine elevation"
  }
}
```

---

## Graph Construction

### Building the Bipartite Graph

**Algorithm:**
```
1. Create empty NetworkX graph
2. For each entity:
   - Add entity node to partition 0
   - Set node attributes: type="entity", name=entity_name
3. For each relation:
   - Add relation node to partition 1
   - Set node attributes: type="relation", description=relation_desc
4. For each relation:
   - For each connected entity:
     - Add edge: (entity_node, relation_node)
     - Set edge weight = relation confidence
5. Validate bipartite property (optional but recommended)
6. Save as GraphML
```

**Implementation:**
```python
import networkx as nx

def build_bipartite_graph(entities: list, relations: list):
    """
    Construct bipartite knowledge graph
    """
    G = nx.Graph()

    # Add entity nodes (partition 0)
    for entity in entities:
        G.add_node(
            entity["name"],
            bipartite=0,  # Partition 0
            node_type="entity",
            entity_type=entity["type"],
            description=entity.get("description", ""),
            confidence=entity["confidence"]
        )

    # Add relation nodes (partition 1)
    for i, relation in enumerate(relations):
        relation_id = f"relation_{i}"
        G.add_node(
            relation_id,
            bipartite=1,  # Partition 1
            node_type="relation",
            description=relation["description"],
            keywords=",".join(relation["keywords"]),
            confidence=relation["confidence"]
        )

        # Add edges to connected entities
        for entity_name in relation["entities"]:
            if entity_name in G.nodes:
                G.add_edge(
                    entity_name,
                    relation_id,
                    weight=relation["confidence"]
                )

    # Validate bipartite property
    assert nx.is_bipartite(G), "Graph is not bipartite!"

    return G
```

**Validation:**
```python
def validate_bipartite(G):
    """
    Ensure graph satisfies bipartite constraints
    """
    # Check 1: All nodes have bipartite attribute
    for node in G.nodes:
        assert "bipartite" in G.nodes[node], f"Node {node} missing bipartite attribute"

    # Check 2: Only edges between different partitions
    for u, v in G.edges:
        assert G.nodes[u]["bipartite"] != G.nodes[v]["bipartite"], \
               f"Edge ({u}, {v}) connects same partition!"

    # Check 3: Partitions are non-empty
    partition_0 = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    partition_1 = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}

    assert len(partition_0) > 0, "Partition 0 (entities) is empty!"
    assert len(partition_1) > 0, "Partition 1 (relations) is empty!"

    print(f"✅ Valid bipartite graph:")
    print(f"   Entities: {len(partition_0)}")
    print(f"   Relations: {len(partition_1)}")
    print(f"   Edges: {G.number_of_edges()}")
```

### Graph Statistics

**For 10K documents:**
```
Nodes:
  - Entities (partition 0): ~10,000
  - Relations (partition 1): ~18,000
  - Total nodes: ~28,000

Edges:
  - Average degree (entity): ~7 relations per entity
  - Average degree (relation): ~4 entities per relation
  - Total edges: ~70,000

Graph Properties:
  - Bipartite: ✅ Yes
  - Connected: ✅ Usually yes (largest component >95% of nodes)
  - Average path length: 4-6 hops
  - Clustering coefficient: N/A (bipartite graphs have no triangles)
```

---

## Vector Index Building

### Why FAISS?

**FAISS** (Facebook AI Similarity Search) is optimized for:
1. ✅ **Speed** - Billion-scale vector search in milliseconds
2. ✅ **Memory efficiency** - Compression and quantization
3. ✅ **Flexibility** - Multiple index types (flat, HNSW, IVF)

### Index Types

**1. Flat Index (Exact Search)**
```python
import faiss

# Best for <100K vectors
dimension = 1536  # bge-large-en-v1.5
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

# Add vectors
index.add(embeddings)  # Shape: (num_vectors, 1536)

# Save
faiss.write_index(index, "index_entity.bin")
```

**2. HNSW Index (Approximate Search)**
```python
# Best for >100K vectors
dimension = 1536
M = 32  # Number of connections per node (trade-off: accuracy vs. speed)

index = faiss.IndexHNSWFlat(dimension, M)
index.add(embeddings)

faiss.write_index(index, "index_entity_large.bin")
```

**Performance Comparison:**

| Index Type | Build Time | Search Time | Accuracy | Memory |
|------------|------------|-------------|----------|--------|
| Flat | Fast | O(n) | 100% | High |
| HNSW | Slower | O(log n) | ~99% | Medium |
| IVF | Medium | O(log n) | ~95% | Low |

**Recommendation for BiG-RAG:**
- **<100K nodes**: Use Flat (exact search)
- **>100K nodes**: Use HNSW with M=32

### Building the Three Indices

```python
def build_vector_indices(
    entity_embeddings,     # Shape: (num_entities, 1536)
    relation_embeddings,   # Shape: (num_relations, 1536)
    chunk_embeddings       # Shape: (num_chunks, 1536)
):
    """
    Build three FAISS indices for entities, relations, and chunks
    """
    dimension = 1536

    # Index 1: Entities
    index_entity = faiss.IndexFlatIP(dimension)
    index_entity.add(entity_embeddings.astype(np.float32))
    faiss.write_index(index_entity, "index_entity.bin")
    print(f"✅ Entity index: {len(entity_embeddings)} vectors")

    # Index 2: Relations
    index_relation = faiss.IndexFlatIP(dimension)
    index_relation.add(relation_embeddings.astype(np.float32))
    faiss.write_index(index_relation, "index_bipartite_edge.bin")
    print(f"✅ Relation index: {len(relation_embeddings)} vectors")

    # Index 3: Text Chunks
    index_chunks = faiss.IndexFlatIP(dimension)
    index_chunks.add(chunk_embeddings.astype(np.float32))
    faiss.write_index(index_chunks, "index_text_chunks.bin")
    print(f"✅ Chunk index: {len(chunk_embeddings)} vectors")

    return index_entity, index_relation, index_chunks
```

### Query Usage

```python
# Load indices
index_entity = faiss.read_index("index_entity.bin")
index_relation = faiss.read_index("index_bipartite_edge.bin")

# Query
query = "What did Einstein discover?"
query_embedding = embedding_model.encode([query])

# Search entities
k = 10  # Top-10
distances, indices = index_entity.search(query_embedding, k)

# distances: cosine similarities (higher = more similar)
# indices: row numbers in the entity embedding matrix

print(f"Top-10 entities:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    entity_name = entity_id_list[idx]  # Map index back to entity name
    print(f"{i+1}. {entity_name} (similarity: {dist:.3f})")
```

---

## Storage Architecture

### Complete Storage Layout

```
bigrag_working_dir/
├── bigrag_graph.graphml              # NetworkX bipartite graph
├── index_entity.bin                  # FAISS entity index
├── index_bipartite_edge.bin          # FAISS relation index
├── index_text_chunks.bin             # FAISS chunk index
├── kv_store_entities.json            # Entity metadata
├── kv_store_bipartite_edges.json     # Relation metadata
├── kv_store_text_chunks.json         # Chunk metadata
├── kv_store_full_docs.json           # Original documents
└── bigrag.log                        # Processing logs
```

### Size Breakdown (for 10K documents)

| File | Size | Purpose |
|------|------|---------|
| `bigrag_graph.graphml` | 80 MB | Graph topology (XML format) |
| `index_entity.bin` | 60 MB | Entity vectors (10K × 1536 × 4 bytes) |
| `index_bipartite_edge.bin` | 110 MB | Relation vectors (18K × 1536 × 4 bytes) |
| `index_text_chunks.bin` | 300 MB | Chunk vectors (50K × 1536 × 4 bytes) |
| `kv_store_entities.json` | 50 MB | Entity metadata (descriptions, types) |
| `kv_store_bipartite_edges.json` | 120 MB | Relation metadata (descriptions, keywords) |
| `kv_store_text_chunks.json` | 200 MB | Chunk metadata (content, doc IDs) |
| `kv_store_full_docs.json` | 100 MB | Original documents |
| **TOTAL** | **~2.2 GB** | **Complete knowledge graph** |

---

## Performance and Scalability

### Indexing Performance

**Time Breakdown (10K documents):**

| Stage | Time | % of Total | Bottleneck |
|-------|------|------------|------------|
| Chunking | 5 min | 8% | Text processing |
| Entity Extraction | 25 min | 40% | **LLM API calls** |
| Relation Extraction | 20 min | 32% | **LLM API calls** |
| Graph Construction | 3 min | 5% | NetworkX operations |
| Embedding | 12 min | 19% | GPU/CPU encoding |
| FAISS Indexing | 2 min | 3% | Index construction |
| **Total** | **~62 min** | **100%** | **LLM latency** |

**Optimization Strategies:**

**1. Parallel LLM Calls**
```python
# Process chunks in batches of 50
import asyncio

async def process_batch(chunks, llm_model):
    tasks = [extract_entities_llm(chunk, llm_model) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return results

# Process all chunks
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    results = await process_batch(batch, llm_model)
```

**Speedup:** 10x faster with parallelization (25 min → 2.5 min for entity extraction)

**2. GPU Acceleration for Embeddings**
```python
# Use GPU for embedding model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

# Batch encoding
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
```

**Speedup:** 5x faster with GPU (12 min → 2.4 min)

**3. Caching**
```python
# Cache LLM responses to avoid re-extraction
import hashlib
import json

cache_file = "llm_cache.json"
cache = {}

def cached_llm_call(text, llm_model):
    # Hash text for cache key
    key = hashlib.md5(text.encode()).hexdigest()

    if key in cache:
        return cache[key]

    # Call LLM
    response = llm_model(text)
    cache[key] = response

    # Save cache
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    return response
```

**Savings:** Avoid duplicate extractions (especially useful during development/debugging)

### Scalability

**Storage Scaling:**

| Documents | Entities | Relations | Edges | Storage | Indexing Time |
|-----------|----------|-----------|-------|---------|---------------|
| 1K | ~1,000 | ~1,800 | ~7,000 | 220 MB | 6 min |
| 10K | ~10,000 | ~18,000 | ~70,000 | 2.2 GB | 62 min |
| 100K | ~100,000 | ~180,000 | ~700,000 | 22 GB | 10 hours |
| 1M | ~1,000,000 | ~1,800,000 | ~7,000,000 | 220 GB | 4 days |

**Recommendation:**
- **<10K docs**: Single machine (16GB RAM, CPU)
- **10K-100K docs**: Single machine (64GB RAM, GPU recommended)
- **>100K docs**: Distributed system (Milvus vector DB, Neo4j graph DB)

---

## Implementation Guide

### Complete Indexing Script

```python
import asyncio
from bigrag import BiGRAG

async def build_knowledge_graph(corpus_path: str, output_dir: str):
    """
    Build complete BiG-RAG knowledge graph from corpus
    """
    # Step 1: Initialize BiGRAG
    rag = BiGRAG(
        working_dir=output_dir,
        llm_model_name="gpt-4o-mini",
        embedding_dim=1536
    )

    # Step 2: Load documents
    documents = []
    with open(corpus_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc["contents"])

    print(f"Loaded {len(documents)} documents")

    # Step 3: Insert documents (automatic indexing)
    await rag.ainsert(documents)

    # Step 4: Verify indices
    print("\n✅ Indexing complete!")
    print(f"   Entities: {len(rag.entities_vdb)}")
    print(f"   Relations: {len(rag.bipartite_edges_vdb)}")
    print(f"   Graph nodes: {rag.graph.number_of_nodes()}")
    print(f"   Graph edges: {rag.graph.number_of_edges()}")

    # Step 5: Save (automatic during insert)
    print(f"\n💾 Saved to: {output_dir}")

# Run
asyncio.run(build_knowledge_graph(
    corpus_path="datasets/my_dataset/corpus.jsonl",
    output_dir="expr/my_dataset"
))
```

### Production Deployment

**Docker Container:**
```dockerfile
FROM python:3.11

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy code
COPY bigrag/ bigrag/
COPY datasets/ datasets/

# Build indices
RUN python build_indices.py

# Run API server
CMD ["python", "script_api.py", "--data_source", "my_dataset", "--port", "8001"]
```

---

## Conclusion

BiG-RAG's indexing architecture transforms raw documents into a queryable bipartite knowledge graph through:

1. **Document Chunking** - 1200-token chunks with 100-token overlap
2. **Multi-Pass Entity Extraction** - spaCy + LLM + resolution
3. **N-ary Relation Extraction** - Complete semantic context preservation
4. **Bipartite Graph Construction** - Efficient topology representation
5. **Vector Indexing** - FAISS for fast similarity search
6. **Structured Storage** - Graph + Vectors + KV stores

**Key Benefits:**
✅ **No training required** - Uses pretrained LLMs and embedding models
✅ **Scalable** - Handles 10K-100K documents on single machine
✅ **Accurate** - Preserves complete n-ary relation semantics
✅ **Fast retrieval** - FAISS enables millisecond vector search

### Next Steps

1. Read [EDUCATIONAL_DEEP_DIVE_RETRIEVAL_ARCHITECTURE.md](EDUCATIONAL_DEEP_DIVE_RETRIEVAL_ARCHITECTURE.md) to understand query processing
2. Review [DATASET_AND_CORPUS_GUIDE.md](DATASET_AND_CORPUS_GUIDE.md) for corpus preparation
3. Check [CLAUDE.md](../CLAUDE.md) for complete developer reference

---

**Questions? Issues?** See [DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md) for the complete documentation suite.
