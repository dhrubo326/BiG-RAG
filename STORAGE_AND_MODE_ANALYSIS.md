# Storage and Mode Analysis - Expert Recommendation

**Date:** October 31, 2025
**Analysis By:** Claude (Deep Code Investigation)

---

## Executive Summary

After thorough investigation of:
1. BiG-RAG paper (`BiG-RAG_Full_Paper.md`)
2. Implementation code (`bigrag/bigrag.py`, `bigrag/operate.py`, `bigrag/base.py`)
3. Storage backends (`bigrag/storage.py`)
4. Your current setup (OpenAI embeddings with NanoVectorDB)

**Key Findings:**

✅ **Question 1 (Storage):** YES - storage should be consistent regardless of embedder choice
❌ **Question 2 (Modes):** NO - BiGRAG does NOT currently implement multiple retrieval modes (only hybrid works)

**Recommendation:** Standardize storage format + Document the mode limitation

---

## Question 1: Storage Consistency Across Embedders

### Current Situation

**Problem:** Different embedders create different file structures:

| Embedder | Files Created | Metadata Storage |
|----------|--------------|------------------|
| **OpenAI** | `vdb_entities.json`, `vdb_bipartite_edges.json`, `vdb_chunks.json` | Minimal (names only) |
| **FlagEmbedding** | `kv_store_entities.json`, `kv_store_bipartite_edges.json`, `index_entity.bin` | Full metadata |

**Both create:** `graph_chunk_entity_relation.graphml` (NetworkX graph with ALL metadata)

### What the Paper Says

From **Section 4.1: System Architecture** of `BiG-RAG_Full_Paper.md`:

> **Vector Database Layer** maintains two dense retrieval indices:
> - **Entity Index:** $\{\psi(e) : e \in V_E\}$ with dimension $d=3072$ (text-embedding-3-large)
> - **Relation Index:** $\{\psi(r) : r \in V_R\}$ with dimension $d=3072$
>
> Uses **FAISS IndexFlatIP** for L2-normalized vectors, enabling approximate nearest neighbor search in $O(\log |V|)$ expected time.
>
> **Key-Value Store Layer** provides persistent storage for:
> - Full entity metadata (names, types, descriptions)
> - Complete relation metadata (descriptions, confidence scores, provenance)
> - Document chunks and source mappings

**Paper's Design:**
- **Vector Layer:** FAISS indices for fast similarity search
- **KV Layer:** Full metadata storage (separate from vectors)
- **Graph Layer:** NetworkX/Neo4j for structural queries

### Technical Analysis: FAISS vs NanoVectorDB

#### FAISS (FlagEmbedding Mode)

**Pros:**
- ✅ **Paper-recommended** - explicitly mentioned in Section 4.1
- ✅ **Production-grade** - used by Meta, handles billions of vectors
- ✅ **Fast** - $O(\log |V|)$ approximate nearest neighbor
- ✅ **Scalable** - can use IVF indices for large datasets
- ✅ **Separation of concerns** - vectors in `.bin`, metadata in `.json`
- ✅ **Storage efficient** - binary format, smaller than JSON
- ✅ **GPU support** - can offload to GPU for massive datasets

**Cons:**
- ❌ Requires C++ compilation (more complex deployment)
- ❌ Slightly more complex code
- ❌ Separate metadata files needed

**Files Created:**
```
index_entity.bin              # FAISS index (vectors only)
corpus_entity.npy             # Numpy array backup
kv_store_entities.json        # Full metadata (name, type, description, source_id, weight)
```

#### NanoVectorDB (OpenAI Mode - Current)

**Pros:**
- ✅ **Pure Python** - no C++ dependencies, easier deployment
- ✅ **Simple** - all-in-one JSON file
- ✅ **Good for small datasets** - works fine for <100K vectors
- ✅ **OpenAI-compatible format** - easy to inspect

**Cons:**
- ❌ **NOT in paper** - not the intended design
- ❌ **Inefficient** - stores vectors + metadata in JSON (huge files)
- ❌ **Slow for large datasets** - linear search, no indexing
- ❌ **Partial metadata** - only stores `meta_fields` (entity_name), not full metadata
- ❌ **Memory inefficient** - loads entire JSON into RAM

**Files Created:**
```
vdb_entities.json             # All-in-one (vectors + minimal metadata)
{
  "embedding_dim": 1536,
  "data": [
    {"__id__": "ent-xxx", "entity_name": "DHAKA"},  # Only name!
    {"__id__": "ent-yyy", "entity_name": "BANGLADESH"}
  ],
  "matrix": [[0.1, 0.2, ...], [...]]  # Huge!
}
```

### Why Metadata is Missing in Your Setup

**Root Cause:** When using NanoVectorDB (`vector_storage="NanoVectorDBStorage"`):

```python
# bigrag/bigrag.py lines 224-230
self.entities_vdb = self.vector_db_storage_cls(
    namespace="entities",
    global_config=asdict(self),
    embedding_func=self.embedding_func,
    meta_fields={"entity_name"},  # ← ONLY stores entity_name!
    **self.vector_db_storage_cls_kwargs,
)
```

**NanoVectorDB only saves fields in `meta_fields`:**
- Entity: `entity_name` (missing: `entity_type`, `description`, `weight`, `source_id`)
- Edge: `bipartite_edge_name` (missing: `description`, `source_id`, `weight`)

**Full metadata IS saved to GraphML:**
- `graph_chunk_entity_relation.graphml` has ALL fields
- But `api/kg_utils.py` doesn't read GraphML (it expects JSON files)

### Expert Recommendation

**Option A: Switch to FAISS (Paper-Aligned, Production-Ready)** ⭐ **RECOMMENDED**

**Rationale:**
1. ✅ **Paper-recommended** - matches BiG-RAG design intent
2. ✅ **Scalable** - handles growth from 1K to 1M+ documents
3. ✅ **Performant** - faster retrieval as dataset grows
4. ✅ **Complete metadata** - separate KV storage for all fields
5. ✅ **Industry standard** - used by major production systems

**Implementation:**
```python
# In script_api.py or script_build.py
rag = BiGRAG(
    working_dir="./expr/demo_test",
    vector_storage="NanoVectorDBStorage",  # BEFORE
    # ↓ Change to:
    vector_storage="NanoVectorDBStorage",  # KEEP for small datasets
    # Or migrate to FAISS for production:
    # vector_storage="FAISSStorage",  # Production-ready
)
```

**BUT WAIT** - There's a problem: Current code doesn't have a `FAISSStorage` class that saves metadata separately!

**The Issue:**
- `NanoVectorDBStorage` saves vectors + minimal metadata to single JSON
- There's no `FAISSStorage` that properly separates vectors (FAISS) and metadata (JSON)

**The Fix Needed:**
1. Keep NanoVectorDB for simplicity (acceptable for datasets <100K vectors)
2. **But fix it to save full metadata**, not just `entity_name`

**Option B: Fix NanoVectorDB to Save Full Metadata** ⭐ **PRACTICAL CHOICE**

**Rationale:**
1. ✅ **Minimal code changes** - extend NanoVectorDBStorage
2. ✅ **Keeps simplicity** - no C++ dependencies
3. ✅ **Good enough** - works for your scale (564 entities, 413 edges)
4. ✅ **Backward compatible** - doesn't break existing code

**Implementation Strategy:**

Either:
1. **Don't try to fix NanoVectorDB** - it's designed for minimal storage
2. **Use GraphML as source of truth** - fix `api/kg_utils.py` to read from GraphML

**Option C: Use GraphML as Metadata Source** ⭐⭐⭐ **BEST SOLUTION**

**Rationale:**
1. ✅ **Already works** - GraphML has all metadata
2. ✅ **No storage changes** - use existing system
3. ✅ **LLM-independent** - storage format doesn't change with embedder
4. ✅ **Consistent** - single source of truth for metadata

**This is what my original fix plan proposed!**

### Final Recommendation for Question 1

**Answer:** Storage process IS the same regardless of embedder. The problem is:

1. **BiGRAG core** correctly stores metadata in GraphML (consistent across embedders)
2. **NanoVectorDB** only stores minimal metadata in JSON (by design)
3. **API layer** (`api/kg_utils.py`) expects metadata in JSON files (wrong assumption)

**Solution:** Fix API layer to read from GraphML (single source of truth)

**No need to standardize filenames** - just fix the API layer to use GraphML!

---

## Question 2: Does BiGRAG Support Multiple Retrieval Modes?

### Official API Definition

From `bigrag/base.py` lines 17-32:

```python
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    # Number of top-k items to retrieve; corresponds to entities in "local" mode
    # and relationships in "global" mode.
    top_k: int = 60
    # ...
```

**API promises 4 modes:**
- `local` - Entity-centric retrieval
- `global` - Relation-centric retrieval
- `hybrid` - Combined (dual-path)
- `naive` - Direct chunk retrieval

### Actual Implementation

**Investigation of `bigrag/operate.py`:**

```python
async def kg_query(query, knowledge_graph_inst, entities_vdb, bipartite_edges_vdb,
                   text_chunks_db, query_param, global_config):
    # Line 498-507
    context = await _build_query_context(
        keywords, knowledge_graph_inst, entities_vdb, bipartite_edges_vdb,
        text_chunks_db, query_param
    )
    return context

async def _build_query_context(..., query_param):
    # Lines 520-571
    # ALWAYS calls BOTH:
    knowledge_list_1 = await _get_node_data(...)      # Entity-based
    knowledge_list_2 = await _get_edge_data(...)      # Relation-based

    # ALWAYS fuses both results (reciprocal rank fusion)
    for i, (k, source_ids) in enumerate(knowledge_list_1):
        know_score[k] += 1/(i+1)
    for i, (k, source_ids) in enumerate(knowledge_list_2):
        know_score[k] += 1/(i+1)

    # Returns fused results
    return sorted(know_score.items(), ...)[:query_param.top_k]
```

**Finding:** `query_param.mode` is **NEVER CHECKED**!

**Proof:**
```bash
$ grep -rn "param.mode\|query_param.mode" bigrag/
bigrag/bigrag.py:500:  # kg_query will handle querying based on param.mode
# ↑ Just a comment, not actual code!
```

### What Paper Says About Modes

From **Section 4.3: Dual-Path Retrieval Mechanism**:

> BiG-RAG retrieves relevant knowledge through two complementary paths that are fused using reciprocal rank aggregation.
>
> #### 4.3.1 Entity-Based Retrieval Path
> **Goal:** Find relations containing entities semantically similar to query entities.
>
> #### 4.3.2 Relation-Based Retrieval Path
> **Goal:** Find relations whose descriptions match the query semantically.
>
> #### 4.3.3 Reciprocal Rank Fusion
> **Design Rationale:** Reciprocal rank fusion:
> - Balances contributions from both paths without requiring score normalization
> - Rewards relations appearing in multiple paths (higher combined score)

**Paper Conclusion:** Paper describes **dual-path (hybrid)** as the core retrieval mechanism. It does NOT describe separate "local", "global", or "naive" modes.

### Why Mode Parameter Exists

Looking at the comment in `QueryParam`:

```python
# Number of top-k items to retrieve; corresponds to entities in "local" mode
# and relationships in "global" mode.
```

**Hypothesis:** Modes were PLANNED but never implemented. The API definition exists, but the code always does hybrid.

### Current Behavior

**Regardless of `mode` parameter value:**
1. Query entity index → get top-k entities
2. Query relation index → get top-k relations
3. Fuse results using reciprocal rank
4. Return top-k fused results

**This is ALWAYS hybrid mode.**

### Expert Recommendation for Question 2

**Answer:** NO - BiGRAG currently does NOT implement multiple modes.

**Current State:**
- ✅ API defines `mode` parameter
- ❌ Implementation ignores it
- ✅ Always uses hybrid (dual-path)
- ❌ `local`, `global`, `naive` don't work

**Should We Implement Modes?**

**Arguments FOR:**
1. ✅ API already promises them
2. ✅ Could be useful for specific scenarios:
   - `local` - When you know entities but not relations
   - `global` - When you know relations but not entities
   - `naive` - Baseline comparison

**Arguments AGAINST:**
1. ❌ **Paper recommends hybrid** - best performance
2. ❌ **Additional complexity** - more code to maintain
3. ❌ **Unclear benefit** - hybrid already combines both paths
4. ❌ **Not in paper** - not part of validated design

**My Recommendation:** **DO NOT implement separate modes**

**Rationale:**
1. **Paper's design is dual-path (hybrid)** - extensively validated
2. **Separation doesn't make sense** - entity and relation searches complement each other
3. **Hybrid gives best results** - as shown in paper's ablation study (Section 6.3)

**From paper's Table 3:**
```
| Configuration | F1 | Δ F1 |
|---------------|----|----- |
| BiG-RAG (full) | 56.4 | - |
| - w/o dual-path (entity only) | 52.1 | -4.3 |
| - w/o dual-path (relation only) | 51.3 | -5.1 |
```

**Using only one path loses 4-5 F1 points!**

### What To Do About `/search` Endpoint

**Current Issue:** `/search` endpoint ignores `mode` parameter

**Options:**

**Option A: Implement All Modes** ❌ **NOT RECOMMENDED**
- Requires significant code changes
- Contradicts paper's design
- Degrades performance

**Option B: Accept Mode But Log Warning** ⚠️ **COMPROMISE**
```python
if request.mode != "hybrid":
    logger.warning(f"Mode '{request.mode}' not implemented. Using 'hybrid' (recommended).")
param = QueryParam(mode="hybrid", ...)  # Always hybrid
```

**Option C: Remove Mode Parameter** ❌ **BREAKING CHANGE**
- Would break existing API calls
- Honest but inconvenient

**Option D: Document Limitation** ✅ **RECOMMENDED**
```python
class SearchRequest(BaseModel):
    queries: List[str]
    mode: str = "hybrid"  # Note: Only "hybrid" is currently supported
    top_k: int = 10
```

**Add to API docs:**
> **Note:** BiG-RAG uses dual-path retrieval (hybrid mode) as recommended by the paper.
> The `mode` parameter is accepted for API compatibility but currently only `"hybrid"` is implemented.
> This provides the best retrieval quality by combining entity-centric and relation-centric search.

---

## Summary of Recommendations

### Question 1: Storage Consistency ✅

**Problem:** Storage format differs between OpenAI and FlagEmbedding modes

**Root Cause:** NanoVectorDB saves minimal metadata; API layer expects full metadata in JSON

**Solution:**
1. ✅ Keep current storage backends (both are valid)
2. ✅ Fix `api/kg_utils.py` to read from GraphML (universal source of truth)
3. ✅ GraphML already has all metadata regardless of embedder
4. ❌ Don't try to change NanoVectorDB or standardize filenames

**Why This Works:**
- GraphML is created by BOTH embedder modes
- GraphML has complete metadata (entities, edges, source_ids, types, descriptions)
- No code changes to BiGRAG core needed
- LLM-independent solution

### Question 2: Multiple Modes ❌

**Problem:** API defines `mode` parameter but implementation doesn't use it

**Root Cause:** Modes were planned but never implemented; code always does hybrid

**Solution:**
1. ❌ Don't implement separate modes (contradicts paper, degrades performance)
2. ✅ Document that only "hybrid" works
3. ✅ Accept mode parameter but always use hybrid (with optional warning)
4. ✅ Update API documentation to clarify

**Why This Works:**
- Paper validates hybrid (dual-path) as best approach
- Ablation study shows single-path loses 4-5 F1 points
- Simpler codebase, fewer bugs
- Matches production-validated design

---

## Implementation Plan (Revised)

### Phase 1: Fix Document Stats (API Layer Only)

**File:** `api/kg_utils.py`

**Changes:**
1. Add `detect_storage_mode()` - check for FAISS vs NanoVectorDB
2. Add `get_document_stats_from_graphml()` - read from GraphML
3. Modify all functions to route to GraphML reader
4. Keep FlagEmbedding path for backward compat

**Impact:**
- ✅ Document stats work for both OpenAI and FlagEmbedding
- ✅ No changes to BiGRAG core
- ✅ No storage format changes
- ✅ LLM-independent

**Time:** 2-3 hours

### Phase 2: Document Mode Limitation

**File:** `script_api.py`

**Changes:**
1. Add docstring to `SearchRequest` noting hybrid-only
2. Optionally add warning log if mode != "hybrid"
3. Update API documentation
4. Add note in README

**Impact:**
- ✅ Users understand current behavior
- ✅ No breaking changes
- ✅ Honest about limitations
- ✅ Aligns with paper

**Time:** 30 minutes

### Phase 3: Testing

**Tests:**
1. Upload document with OpenAI embeddings
2. Verify stats show correct counts
3. Test with FlagEmbedding (if available)
4. Test /search with different mode values
5. Verify warning logs appear

**Time:** 1 hour

**Total Time:** 3.5-4.5 hours

---

## Final Answer to Your Questions

### 1. Storage Consistency?

**YES** - The internal KG construction process IS the same regardless of embedder:

1. ✅ Same entity extraction (GPT-4o-mini)
2. ✅ Same bipartite graph structure (NetworkX)
3. ✅ Same metadata saved to GraphML
4. ✅ Same chunking process

**What differs:**
- Vector index format (FAISS .bin vs NanoVectorDB .json)
- Metadata accessibility (FAISS has separate .json, NanoVectorDB embeds minimal)

**Solution:** Use GraphML as universal metadata source (already exists!)

### 2. Multiple Modes Support?

**NO** - BiGRAG does NOT currently support multiple modes:

- API defines them but implementation ignores the parameter
- Code always uses hybrid (dual-path retrieval)
- Paper recommends hybrid as core design
- Implementing separate modes would contradict validated design

**Solution:** Document limitation, don't implement other modes

---

**Ready to proceed with revised implementation plan?**
