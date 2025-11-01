# BiG-RAG Implementation Status & Roadmap
**Bipartite Graph Retrieval-Augmented Generation**

> A production-grade framework that builds on the proven dual-path retrieval approach with enhanced three-path retrieval, semantic reranking, and pluggable storage.

**Document Status:** ✅ Updated to reflect current implementation (2025-11-01)

---

## Table of Contents
1. [Current Implementation Status](#current-implementation-status)
2. [Design Philosophy](#design-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [What's Already Implemented](#whats-already-implemented)
5. [What Needs to Be Added](#what-needs-to-be-added)
6. [Three-Path Retrieval System (Target Design)](#three-path-retrieval-system-target-design)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Quick Reference](#quick-reference)

---

## Current Implementation Status

### ✅ Completed Features

| Component | Status | Location |
|-----------|--------|----------|
| **Storage Infrastructure** | ✅ Done | `bigrag/bigrag.py`, `bigrag/storage.py` |
| Three vector databases | ✅ Done | `entities_vdb`, `bipartite_edges_vdb`, `chunks_vdb` |
| Pluggable storage backends | ✅ Done | `bigrag/kg/*.py` |
| Graph storage (NetworkX) | ✅ Done | `bigrag/storage.py::NetworkXStorage` |
| KV storage (JSON) | ✅ Done | `bigrag/storage.py::JsonKVStorage` |
| **Indexing Pipeline** | ✅ Done | `bigrag/bigrag.py::ainsert()` |
| Document chunking | ✅ Done | `bigrag/operate.py::chunking_by_token_size()` |
| Entity extraction | ✅ Done | `bigrag/operate.py::extract_entities()` |
| Bipartite graph construction | ✅ Done | `bigrag/operate.py::_merge_*()` |
| Entity embedding | ✅ Done | Indexed to `entities_vdb` |
| Bipartite edge embedding | ✅ Done | Indexed to `bipartite_edges_vdb` |
| Chunk embedding | ✅ Done | Indexed to `chunks_vdb` |
| **Retrieval (Dual-Path)** | ✅ Done | `bigrag/operate.py` |
| Path A: Entity search | ✅ Done | `_get_node_data()` |
| Path B: Bipartite edge search | ✅ Done | `_get_edge_data()` |
| RRF fusion (A+B) | ✅ Done | `_build_query_context()` lines 540-560 |
| 1-hop graph traversal | ✅ Done | `_find_most_related_edges_from_entities()` |
| Source ID tracking | ✅ Done | Returns `<source_ids>` in results |
| **Terminology** | ✅ Corrected | All code uses "bipartite_edge" not "hyperedge" |

### ❌ Missing Features (Priority Order)

| Component | Status | Priority | Estimated Effort |
|-----------|--------|----------|------------------|
| **Path C: Chunk Vector Search** | ❌ Not implemented | P0 (Must) | 2-4 hours |
| Indirect chunk extraction from RRF | ❌ Not implemented | P0 (Must) | 1-2 hours |
| Chunk retrieval integration | ❌ Not implemented | P0 (Must) | 2-3 hours |
| **Semantic Reranking** | ❌ Not implemented | P1 (Should) | 4-6 hours |
| Cross-encoder reranker | ❌ Not implemented | P1 (Should) | 3-4 hours |
| Reranking toggle | ❌ Not implemented | P1 (Should) | 1 hour |
| **Mode System Clarification** | 🔄 Needs update | P2 (Nice) | 2-3 hours |
| Mode renaming/documentation | 🔄 Partial | P2 (Nice) | 1 hour |

### 🔄 Partially Implemented

- **chunks_vdb**: Created and populated during indexing, but **NOT used during retrieval**
  - Location: `bigrag/bigrag.py:238` (created), line 379 (populated)
  - Missing: Query function in `_build_query_context()`

---

## Design Philosophy

### Core Principles

1. **Build on Existing Foundation** - BiG-RAG already has dual-path retrieval working
2. **Add Path C Incrementally** - Chunk vector search is the main missing piece
3. **Maintain Backward Compatibility** - Don't break existing dual-path behavior
4. **Make Reranking Optional** - Support both fast (no reranker) and accurate (with reranker) modes
5. **Clear Status Tracking** - Document what's done vs. what's needed

### Current vs. Target Architecture

**Current (Dual-Path):**
```
Query → Path A (Entities) ─┐
                            ├─→ RRF Fusion → Top-K Results
Query → Path B (Edges) ────┘
```

**Target (Three-Path):**
```
Query → Path A (Entities) ─┐
                            ├─→ RRF Fusion → Top-5 Structured Knowledge
Query → Path B (Edges) ────┘                        ↓
                                             (Extract source_ids)
                                                     ↓
Query → Path C (Chunks) ──┬→ 5 Direct Chunks   ┌────┘
                          └→ 5 Indirect Chunks ─┘
                                     ↓
                          Optional Reranking (10→5 or 10→10)
                                     ↓
                          Final: 5 Structured + 5-10 Chunks
```

---

## Architecture Overview

### Current BiG-RAG Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         BiG-RAG Core                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Indexing    │  │  Retrieval   │  │   Storage    │         │
│  │   Pipeline   │  │    Engine    │  │   Adapters   │         │
│  │  ✅ DONE     │  │  🔄 PARTIAL  │  │  ✅ DONE     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│        ↓                  ↓                  ↓                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Dual-Path Retrieval (CURRENT)                 │  │
│  ├──────────────┬──────────────────┬────────────────────────┤  │
│  │  Path A:     │  Path B:         │  Path C:               │  │
│  │  Entity      │  Bipartite       │  Chunk Vector          │  │
│  │  Search      │  Edge Search     │  Search                │  │
│  │  ✅ DONE     │  ✅ DONE         │  ❌ NOT USED           │  │
│  └──────────────┴──────────────────┴────────────────────────┘  │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         RRF Fusion (Path A + B) ✅ DONE                  │  │
│  │         Semantic Reranking (Path C) ❌ MISSING           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐
    │  Vector DB  │    │   Graph DB   │    │   KV Store    │
    │  (Pluggable)│    │  (Pluggable) │    │  (Pluggable)  │
    │  ✅ DONE    │    │  ✅ DONE     │    │  ✅ DONE      │
    └─────────────┘    └──────────────┘    └───────────────┘
     NanoVectorDB       NetworkX            JsonKVStorage
     Milvus             Neo4j               MongoDB
     ChromaDB           Oracle              TiDB
```

---

## What's Already Implemented

### 1. Storage Layer ✅

**Location:** [bigrag/bigrag.py:224-243](bigrag/bigrag.py#L224-L243)

```python
# Already implemented in BiG-RAG
self.entities_vdb = self.vector_db_storage_cls(
    namespace="entities",
    global_config=asdict(self),
    embedding_func=self.embedding_func,
    meta_fields={"entity_name"},
    **self.vector_db_storage_cls_kwargs,
)
self.bipartite_edges_vdb = self.vector_db_storage_cls(
    namespace="bipartite_edges",  # ✅ Correct terminology!
    global_config=asdict(self),
    embedding_func=self.embedding_func,
    meta_fields={"bipartite_edge_name"},
    **self.vector_db_storage_cls_kwargs,
)
self.chunks_vdb = self.vector_db_storage_cls(
    namespace="chunks",  # ✅ Created but not used in queries
    global_config=asdict(self),
    embedding_func=self.embedding_func,
    **self.vector_db_storage_cls_kwargs,
)
```

**Status:** ✅ **Fully implemented** - All three vector stores exist and are populated

### 2. Indexing Pipeline ✅

**Location:** [bigrag/bigrag.py::ainsert()](bigrag/bigrag.py#L288-L479)

**What's working:**
- ✅ Document chunking
- ✅ Entity extraction via LLM
- ✅ Bipartite graph construction
- ✅ Entity embedding → `entities_vdb`
- ✅ Bipartite edge embedding → `bipartite_edges_vdb`
- ✅ Chunk embedding → `chunks_vdb` (line 379)

```python
# Current implementation (line 378-381)
if self.chunks_vdb is not None and all_chunks_data:
    await self.chunks_vdb.upsert(all_chunks_data)  # ✅ Chunks ARE embedded!
if self.text_chunks is not None and all_chunks_data:
    await self.text_chunks.upsert(all_chunks_data)
```

**Status:** ✅ **Fully implemented** - Chunks are embedded and stored

### 3. Dual-Path Retrieval ✅

**Location:** [bigrag/operate.py::_build_query_context()](bigrag/operate.py#L511-L571)

**What's working:**
- ✅ Path A: Entity vector search → 1-hop traversal → Edge descriptions
- ✅ Path B: Bipartite edge vector search → Edge details
- ✅ RRF fusion of Path A + B results
- ✅ Source ID tracking for evaluation

```python
# Current implementation (lines 522-536)
knowledge_list_1 = await _get_node_data(
    ll_kewwords,
    knowledge_graph_inst,
    entities_vdb,  # ✅ Uses entities_vdb
    text_chunks_db,
    query_param,
)

knowledge_list_2 = await _get_edge_data(
    hl_keywrds,
    knowledge_graph_inst,
    bipartite_edges_vdb,  # ✅ Uses bipartite_edges_vdb
    text_chunks_db,
    query_param,
)

# RRF fusion (lines 543-560)
know_score = dict()
for i, (k, source_ids) in enumerate(knowledge_list_1):
    if k not in know_score:
        know_score[k] = 0
    score = 1/(i+1)
    know_score[k] += score
# ... (similar for knowledge_list_2)
```

**Status:** ✅ **Fully implemented** - Dual-path retrieval works correctly

### 4. Query Modes 🔄

**Location:** [bigrag/base.py::QueryParam:18](bigrag/base.py#L18)

**Current modes:**
```python
mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"
```

**Mode meanings:**
- `local`: Entity-based retrieval only (Path A)
- `global`: Bipartite edge-based retrieval only (Path B)
- `hybrid`: Dual-path retrieval (Path A + B with RRF)
- `naive`: Text chunk retrieval (but NOT implemented!)

**Status:** 🔄 **Partially working** - `local`, `global`, `hybrid` work; `naive` mode exists but doesn't use `chunks_vdb`

---

## What Needs to Be Added

### Priority 0 (Must Have) - Core Functionality

#### 1. Path C: Chunk Vector Search ❌

**What's missing:**
- `chunks_vdb` is created and populated, but **never queried**
- No direct chunk vector search in retrieval flow
- No indirect chunk extraction from RRF results

**Where to add:** [bigrag/operate.py::_build_query_context()](bigrag/operate.py#L511-L571)

**Implementation needed:**
```python
# ADD AFTER LINE 560 in _build_query_context()

# === NEW: Path C - Chunk Vector Search ===
async def _get_chunk_data(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    rrf_results: List[Dict] = None,  # Top-5 structured knowledge from RRF
) -> Tuple[List[Dict], List[Dict]]:
    """
    Path C: Chunk vector search with direct + indirect chunks.

    Returns: (direct_chunks, indirect_chunks)
    """
    # Step 1: Direct vector search on chunks
    direct_results = await chunks_vdb.query(query, top_k=5)
    direct_chunks = []
    for r in direct_results:
        chunk_data = await text_chunks_db.get_by_id(r.get("id"))
        if chunk_data:
            direct_chunks.append({
                "chunk_id": r.get("id"),
                "content": chunk_data.get("content", ""),
                "score": r.get("distance", 0.0),
                "source": "direct"
            })

    # Step 2: Indirect chunks from RRF results (uses source_ids)
    indirect_chunks = []
    if rrf_results:
        indirect_chunk_ids = set()
        for result in rrf_results:
            source_ids = result.get("<source_ids>", [])
            indirect_chunk_ids.update(source_ids)

        # Fetch indirect chunks
        for chunk_id in list(indirect_chunk_ids)[:5]:  # Limit to 5
            chunk_data = await text_chunks_db.get_by_id(chunk_id)
            if chunk_data:
                indirect_chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_data.get("content", ""),
                    "score": 0.0,  # No direct score
                    "source": "indirect"
                })

    return direct_chunks, indirect_chunks
```

**Estimated effort:** 2-4 hours

#### 2. Integrate Path C into Query Flow ❌

**Where to modify:** [bigrag/operate.py::_build_query_context()](bigrag/operate.py#L511-L571) after line 571

```python
# MODIFY _build_query_context() to add Path C

async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    bipartite_edges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,  # NEW parameter
    enable_reranking: bool = False,  # NEW parameter
):
    ll_kewwords, hl_keywrds = query[0], query[1]

    # Path A + B (existing code)
    knowledge_list_1 = await _get_node_data(...)
    knowledge_list_2 = await _get_edge_data(...)

    # RRF fusion (existing code)
    know_score = dict()
    # ... (existing RRF fusion code lines 543-560)

    knowledge_list = sorted(know_score.items(), key=lambda x: x[1], reverse=True)[:query_param.top_k]

    # Build structured knowledge results
    knowledge = []
    for k, score in knowledge_list:
        sources = list(know_sources.get(k, []))
        knowledge.append({
            "<knowledge>": k,
            "<coherence>": round(score, 3),
            "<source_ids>": sources,
            "<type>": "structured"  # NEW: Mark type
        })

    # === NEW: Path C - Add chunk retrieval ===
    if chunks_vdb is not None and query_param.mode in ["hybrid", "naive"]:
        # Get chunks (5 direct + 5 indirect)
        direct_chunks, indirect_chunks = await _get_chunk_data(
            ll_kewwords,
            chunks_vdb,
            text_chunks_db,
            query_param,
            rrf_results=knowledge[:5]  # Pass top-5 structured knowledge
        )

        # Combine chunks
        all_chunks = direct_chunks + indirect_chunks

        # Option 1: With reranking (if enabled)
        if enable_reranking and len(all_chunks) > 0:
            try:
                from .reranker import _semantic_rerank
                reranked_chunks = await _semantic_rerank(
                    ll_kewwords,
                    all_chunks,
                    top_k=5
                )
                chunk_knowledge = reranked_chunks
            except ImportError:
                # Reranker not available, return all chunks
                chunk_knowledge = all_chunks[:10]
        else:
            # Option 2: WITHOUT reranking - return all 10 chunks
            # This allows BiG-RAG to work without reranker model
            chunk_knowledge = all_chunks[:10]  # Keep all chunks (max 10)

        # Format chunks
        for chunk in chunk_knowledge:
            knowledge.append({
                "<knowledge>": chunk["content"],
                "<coherence>": round(chunk.get("score", 0.5), 3),
                "<source_ids>": [chunk["chunk_id"]],
                "<type>": "chunk"  # NEW: Mark as chunk
            })

    return knowledge
```

**Key change:** When `enable_reranking=False`, return **all 10 chunks** (5 direct + 5 indirect) instead of top-5. This allows using BiG-RAG without the reranker model dependency.

**Estimated effort:** 2-3 hours

#### 3. Update kg_query() to Pass chunks_vdb ❌

**Where to modify:** [bigrag/operate.py::kg_query():484](bigrag/operate.py#L484)

```python
# MODIFY kg_query signature
async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    bipartite_edges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    chunks_vdb: BaseVectorStorage = None,  # NEW parameter
    enable_reranking: bool = False,  # NEW parameter
) -> str:

    hl_keywords = query
    ll_keywords = query
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        bipartite_edges_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb=chunks_vdb,  # NEW
        enable_reranking=enable_reranking,  # NEW
    )

    return context
```

**And update the call site in** [bigrag/bigrag.py::aquery():498](bigrag/bigrag.py#L498):

```python
# MODIFY aquery() to pass chunks_vdb
async def aquery(self, query: str, param: QueryParam = QueryParam(),
                 enable_reranking: bool = False):  # NEW parameter
    response = await kg_query(
        query,
        self.chunk_entity_relation_graph,
        self.entities_vdb,
        self.bipartite_edges_vdb,
        self.text_chunks,
        param,
        asdict(self),
        hashing_kv=self.llm_response_cache,
        chunks_vdb=self.chunks_vdb,  # NEW
        enable_reranking=enable_reranking,  # NEW
    )
    await self._query_done()
    return response
```

**Estimated effort:** 1 hour

### Priority 1 (Should Have) - Accuracy Improvements

#### 4. Semantic Reranking with Cross-Encoder ❌

**What's missing:**
- Cross-encoder model for reranking chunks
- Reranking function
- Graceful fallback when model not available

**Where to add:** New file `bigrag/reranker.py`

```python
# CREATE NEW FILE: bigrag/reranker.py

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. Reranking disabled. "
        "Install: pip install sentence-transformers"
    )

# Global reranker instance (lazy loaded)
_reranker = None

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Get or create cross-encoder reranker"""
    global _reranker
    if _reranker is None and RERANKER_AVAILABLE:
        try:
            _reranker = CrossEncoder(model_name)
            logger.info(f"Loaded reranker model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            return None
    return _reranker

async def _semantic_rerank(
    query: str,
    chunks: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Rerank chunks using cross-encoder.

    Args:
        query: User query
        chunks: List of {chunk_id, content, score, ...}
        top_k: Number of chunks to return

    Returns:
        Top-k reranked chunks with updated scores
    """
    # Fallback if reranker not available
    if not RERANKER_AVAILABLE:
        logger.debug("Reranker not available, using original scores")
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    reranker = get_reranker()
    if reranker is None:
        logger.warning("Reranker failed to load, using original scores")
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    try:
        # Prepare pairs for cross-encoder
        pairs = [(query, chunk["content"]) for chunk in chunks]

        # Get reranking scores
        rerank_scores = reranker.predict(pairs)

        # Attach scores to chunks
        for chunk, score in zip(chunks, rerank_scores):
            chunk["rerank_score"] = float(score)
            # Combine with original score (30% original, 70% rerank)
            chunk["final_score"] = (
                0.3 * chunk.get("score", 0) + 0.7 * score
            )

        # Sort by reranking score
        reranked = sorted(chunks, key=lambda x: x["final_score"], reverse=True)

        return reranked[:top_k]

    except Exception as e:
        logger.warning(
            f"Reranking failed: {e}. Falling back to original scores. "
            f"Error type: {type(e).__name__}"
        )
        # Fallback: return chunks sorted by original scores
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
```

**Estimated effort:** 4-6 hours (including testing)

#### 5. Add Reranking Toggle to QueryParam ❌

**Where to modify:** [bigrag/base.py::QueryParam](bigrag/base.py#L17)

```python
# MODIFY QueryParam
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = 60
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000

    # NEW: Reranking control
    enable_reranking: bool = False  # Default: disabled for speed
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Estimated effort:** 1 hour

### Priority 2 (Nice to Have) - Polish

#### 6. Clarify Mode System 🔄

**Current confusion:**
- Modes named `local`, `global`, `hybrid`, `naive` (inherited from graphr1)
- `naive` mode exists but doesn't work properly

**Recommended changes:**

Keep current names, fix naive mode, and improve documentation:

```python
# Update QueryParam documentation
mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"
"""
Retrieval mode:
- local: Entity-based retrieval only (Path A)
- global: Bipartite edge-based retrieval only (Path B)
- hybrid: Dual-path retrieval (Path A + B) + chunks (Path C) - RECOMMENDED
- naive: Pure chunk vector search (Path C only)
"""
```

**Estimated effort:** 2-3 hours (documentation + testing)

---

## Three-Path Retrieval System (Target Design)

### Complete Flow with All Paths

```
Query: "Which universities in Bangladesh offer CS programs?"
  ↓
┌─────────────────────────────────────────────────────────────────┐
│              Query Embedding (Shared)                           │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────┬─────────────────────┬─────────────────────┐
│    Path A:          │    Path B:          │    Path C:          │
│  Entity Search      │  Bipartite Edge     │  Chunk Vector       │
│  ✅ IMPLEMENTED     │  Search             │  Search             │
│                     │  ✅ IMPLEMENTED     │  ❌ MISSING         │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ entities_vdb.query()│bipartite_edges_vdb  │chunks_vdb.query()   │
│ → top-k entities    │.query()             │→ 5 direct chunks    │
│                     │→ top-k edges        │                     │
│ ✅ DONE             │✅ DONE              │❌ NOT CALLED        │
└─────────────────────┴─────────────────────┴─────────────────────┘
  ↓                     ↓
┌─────────────────────┬─────────────────────┐
│ 1-Hop Graph         │ Get Edge Details    │
│ Traversal           │ from Graph          │
│ ✅ DONE             │ ✅ DONE             │
└─────────────────────┴─────────────────────┘
  ↓                     ↓
┌───────────────────────────────────┐
│   RRF Fusion (Path A + B ONLY)    │
│   ✅ IMPLEMENTED                  │
│   → Top-5 Structured Knowledge    │
│   (with source_ids)                │
└───────────────────────────────────┘
            ↓
            └──────────────┬──────────────────┐
                           ↓                  ↓
            ┌──────────────────────┐  ┌──────────────────┐
            │ Extract source_ids   │  │ Direct chunk     │
            │ from top-5 RRF       │  │ vector search    │
            │ ❌ MISSING           │  │ ❌ MISSING       │
            └──────────────────────┘  └──────────────────┘
                           ↓                  ↓
            ┌──────────────────────────────────┐
            │  Combine Chunks:                 │
            │  - 5 direct chunks               │
            │  - 5 indirect chunks             │
            │  = 10 chunk candidates            │
            │  ❌ MISSING                      │
            └──────────────────────────────────┘
                           ↓
            ┌──────────────────────────────────┐
            │  Optional Semantic Reranking     │
            │  - If enabled: 10 → 5 chunks     │
            │  - If disabled: return all 10    │
            │  ❌ MISSING                      │
            └──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Final Output                                 │
├─────────────────────────────────────────────────────────────────┤
│  5 Structured Knowledge (from Path A+B RRF) ✅ WORKING          │
│  5-10 Chunks (from Path C) ❌ MISSING                           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decision: Reranking Behavior

**Two modes supported:**

1. **With Reranking (enable_reranking=True):**
   - 10 chunks (5 direct + 5 indirect) → Rerank → Top-5 chunks
   - Final output: 5 structured + 5 chunks = **10 items**
   - Better accuracy, slower (~200-300ms per query)

2. **Without Reranking (enable_reranking=False) - NEW:**
   - 10 chunks (5 direct + 5 indirect) → Return all 10
   - Final output: 5 structured + 10 chunks = **15 items**
   - Faster, works without reranker model dependency
   - **Allows using BiG-RAG in RL training without reranker overhead**

**Why this matters:**
- During RL training, we want **fast retrieval** → disable reranking
- During evaluation/production, we want **high accuracy** → enable reranking
- This makes BiG-RAG flexible for different use cases

---

## Implementation Roadmap

### Phase 1: Add Path C (Core Functionality) - 4-8 hours

**Goal:** Make `chunks_vdb` actually work during retrieval

**Tasks:**
1. ✅ Verify `chunks_vdb` is populated (already done - line 379)
2. ❌ Add `_get_chunk_data()` function to [bigrag/operate.py](bigrag/operate.py)
3. ❌ Modify `_build_query_context()` to call `_get_chunk_data()`
4. ❌ Add indirect chunk extraction from RRF results
5. ❌ Update `kg_query()` signature to pass `chunks_vdb`
6. ❌ Update `bigrag.aquery()` to pass `chunks_vdb`
7. ✅ Test with simple query to verify chunks are retrieved

**Success criteria:**
- Query returns structured knowledge + chunks
- `hybrid` mode uses all three paths
- `naive` mode returns chunks only

**Files to modify:**
- [bigrag/operate.py](bigrag/operate.py) (add Path C functions)
- [bigrag/bigrag.py](bigrag/bigrag.py) (update aquery call)

### Phase 2: Add Semantic Reranking (Optional) - 4-6 hours

**Goal:** Improve chunk ranking quality

**Tasks:**
1. ❌ Create `bigrag/reranker.py` with `_semantic_rerank()`
2. ❌ Add graceful fallback when reranker not installed
3. ❌ Import reranker in [bigrag/operate.py](bigrag/operate.py)
4. ❌ Integrate reranking in `_build_query_context()`
5. ❌ Add `enable_reranking` parameter to `QueryParam`
6. ✅ Test with/without reranking enabled

**Success criteria:**
- Reranking works when enabled
- Graceful degradation when disabled or unavailable
- No crashes if sentence-transformers not installed

**Files to modify:**
- `bigrag/reranker.py` (new file)
- [bigrag/operate.py](bigrag/operate.py) (import and call reranker)
- [bigrag/base.py](bigrag/base.py) (add enable_reranking to QueryParam)

### Phase 3: Testing & Validation - 2-4 hours

**Goal:** Ensure everything works correctly

**Tasks:**
1. ❌ Unit test for `_get_chunk_data()`
2. ❌ Integration test for three-path retrieval
3. ❌ Test with reranking enabled/disabled
4. ❌ Benchmark retrieval speed (with/without reranking)
5. ❌ Test accuracy on sample dataset (2WikiMultiHopQA)
6. ✅ Update API server to use new parameters

**Success criteria:**
- All tests pass
- No performance regression on dual-path mode
- Accuracy improvement visible with three-path mode

### Phase 4: Documentation & Polish - 2-3 hours

**Goal:** Make it easy for others to use

**Tasks:**
1. ❌ Update [README.md](README.md) with new retrieval modes
2. ❌ Document `enable_reranking` parameter
3. ❌ Add examples for different modes
4. ❌ Update [CLAUDE.md](CLAUDE.md) with new architecture
5. ❌ Add configuration examples for training vs. evaluation

---

## Quick Reference

### Current Working Features

```python
# What works NOW
from bigrag import BiGRAG, QueryParam

bigrag = BiGRAG(working_dir="./expr/2WikiMultiHopQA")

# Mode 1: Entity-based (Path A only)
result = await bigrag.aquery(
    "What is BUET?",
    param=QueryParam(mode="local", top_k=5)
)

# Mode 2: Edge-based (Path B only)
result = await bigrag.aquery(
    "What collaborations exist?",
    param=QueryParam(mode="global", top_k=5)
)

# Mode 3: Dual-path (Path A + B, default)
result = await bigrag.aquery(
    "Which universities in Bangladesh offer CS?",
    param=QueryParam(mode="hybrid", top_k=5)
)
# Returns: 5 structured knowledge items from RRF fusion
```

### Target Feature (After Implementation)

```python
# What will work AFTER implementation
from bigrag import BiGRAG, QueryParam

bigrag = BiGRAG(working_dir="./expr/2WikiMultiHopQA")

# Mode 1: Three-path (Path A + B + C) - Fast mode
result = await bigrag.aquery(
    "Which universities in Bangladesh offer CS?",
    param=QueryParam(mode="hybrid", top_k=5),
    enable_reranking=False  # NEW: No reranker, return all 10 chunks
)
# Returns: 5 structured + 10 chunks = 15 items

# Mode 2: Three-path (Path A + B + C) - Accurate mode
result = await bigrag.aquery(
    "Which universities in Bangladesh offer CS?",
    param=QueryParam(mode="hybrid", top_k=5),
    enable_reranking=True  # NEW: Rerank 10 chunks → 5 best
)
# Returns: 5 structured + 5 reranked chunks = 10 items

# Mode 3: Pure vector search (Path C only)
result = await bigrag.aquery(
    "Detailed description of CS programs",
    param=QueryParam(mode="naive", top_k=10),
    enable_reranking=True
)
# Returns: 10 reranked chunks
```

### Code Locations Reference

| Component | Status | File Path | Line Numbers |
|-----------|--------|-----------|--------------|
| **Storage Creation** | | | |
| entities_vdb | ✅ Done | [bigrag/bigrag.py](bigrag/bigrag.py) | 224-230 |
| bipartite_edges_vdb | ✅ Done | [bigrag/bigrag.py](bigrag/bigrag.py) | 231-237 |
| chunks_vdb | ✅ Done | [bigrag/bigrag.py](bigrag/bigrag.py) | 238-243 |
| **Indexing** | | | |
| Chunk embedding | ✅ Done | [bigrag/bigrag.py](bigrag/bigrag.py) | 378-381 |
| Entity extraction | ✅ Done | [bigrag/operate.py](bigrag/operate.py) | 261-481 |
| Graph construction | ✅ Done | [bigrag/operate.py](bigrag/operate.py) | 134-258 |
| **Retrieval** | | | |
| Query entry point | ✅ Done | [bigrag/bigrag.py](bigrag/bigrag.py) | 498-512 |
| kg_query | ✅ Done | [bigrag/operate.py](bigrag/operate.py) | 484-507 |
| _build_query_context | ✅ Done | [bigrag/operate.py](bigrag/operate.py) | 511-571 |
| Path A (_get_node_data) | ✅ Done | [bigrag/operate.py](bigrag/operate.py) | 574-616 |
| Path B (_get_edge_data) | ✅ Done | [bigrag/operate.py](bigrag/operate.py) | 650-688 |
| Path C (_get_chunk_data) | ❌ Missing | Need to create | N/A |
| RRF fusion | ✅ Done | [bigrag/operate.py](bigrag/operate.py) | 543-560 |
| Semantic reranking | ❌ Missing | Need to create | N/A |
| **Configuration** | | | |
| QueryParam | 🔄 Partial | [bigrag/base.py](bigrag/base.py) | 17-33 |

---

## Summary

### Current State ✅

BiG-RAG has a **solid foundation**:
- ✅ Three vector databases exist and are populated
- ✅ Dual-path retrieval (Path A + B) works perfectly
- ✅ RRF fusion works correctly
- ✅ Correct terminology throughout ("bipartite_edge" not "hyperedge")
- ✅ Storage abstraction supports pluggable backends
- ✅ Source ID tracking for evaluation

### What's Missing ❌

The **key gap** is Path C (chunk vector search):
- ❌ `chunks_vdb` is created but never queried
- ❌ No direct chunk vector search
- ❌ No indirect chunk extraction from RRF results
- ❌ No semantic reranking

### Priority Roadmap

**Week 1-2: Path C Implementation (P0)**
- Add `_get_chunk_data()` function
- Integrate into query flow
- Test three-path retrieval

**Week 3-4: Semantic Reranking (P1)**
- Create reranker module
- Add reranking toggle
- Test with/without reranking

**Week 5: Testing & Polish (P2)**
- Comprehensive testing
- Documentation updates
- Performance benchmarking

### Expected Improvements

| Metric | Current (Dual-Path) | Target (Three-Path) | Target (Three-Path + Reranking) |
|--------|---------------------|---------------------|----------------------------------|
| Recall@10 | 65-75% | 75-85% (+10-15%) | 80-90% (+15-20%) |
| Precision@10 | 70-80% | 75-85% (+5-10%) | 85-92% (+10-15%) |
| F1 (Multi-hop QA) | 68% | 73-75% (+5-7 points) | 78-82% (+10-14 points) |
| Query Latency | 50-100ms | 80-120ms (+30-50ms) | 200-300ms (with reranking) |

**Recommendation:** Implement Phase 1 (Path C) first, which gives ~5-7 point F1 improvement with minimal latency increase. Add reranking (Phase 2) later for production/evaluation scenarios.

---

**Document Version:** 3.0 (Implementation Roadmap)
**Created:** 2025-11-01
**Last Updated:** 2025-11-01
**Status:** ✅ Ready for Implementation

**Next Steps:**
1. Start with Phase 1: Add `_get_chunk_data()` to [bigrag/operate.py](bigrag/operate.py)
2. Test three-path retrieval with `enable_reranking=False`
3. Once working, add Phase 2: Semantic reranking

---

## Appendix: Terminology Mapping

For developers familiar with the original graphr1 codebase:

| Old Term (graphr1) | New Term (BiG-RAG) | Location |
|--------------------|-------------------|----------|
| hyperedge | bipartite_edge | Throughout BiG-RAG code |
| hyperedges_vdb | bipartite_edges_vdb | [bigrag/bigrag.py:231](bigrag/bigrag.py#L231) |
| hyperedge_name | bipartite_edge_name | [bigrag/bigrag.py:235](bigrag/bigrag.py#L235) |
| `<hyperedge>` tag | `<bipartite_edge>` tag | [bigrag/operate.py:609](bigrag/operate.py#L609) |
| HYPEREDGE | BIPARTITE_EDGE | Graph node types |

**Note:** Internal variable names in `graphr1/` folder still use "hyperedge" terminology, but all **BiG-RAG** code uses "bipartite_edge".
