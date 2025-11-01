# BiG-RAG Implementation Checklist

**Status:** Ready for Implementation
**Date:** 2025-11-01
**Primary Documents:** [BiG_RAG_DESIGN.md](BiG_RAG_DESIGN.md), [BiG_RAG_TECHNICAL_SPEC.md](BiG_RAG_TECHNICAL_SPEC.md)

---

## ✅ PHASE 1: Core Storage Infrastructure (COMPLETE)

**Status:** ✅ **ALREADY DONE** - All storage layers are implemented

**Deliverables:**
- ✅ Vector storage adapter (NanoVectorDB) - [bigrag/bigrag.py:224-243](bigrag/bigrag.py#L224-L243)
- ✅ Graph storage (NetworkX) - [bigrag/storage.py](bigrag/storage.py)
- ✅ KV storage (JsonKVStorage) - [bigrag/storage.py](bigrag/storage.py)
- ✅ Base classes and schemas - [bigrag/base.py](bigrag/base.py)
- ✅ Three vector databases created: `vdb_entities`, `vdb_bipartite_edges`, `vdb_chunks`

**Action:** ✅ Proceed to Phase 2

---

## 🚨 PHASE 2: Indexing Pipeline & Critical Fixes (START HERE)

**Priority:** CRITICAL - Must complete before Phase 3

### Task 2.1: Metadata and Title Preservation ❌ 🚨

**Problem:** Metadata is discarded during indexing, causing poor entity extraction

**Files to Modify:**
1. **[bigrag/bigrag.py::ainsert():283-286](bigrag/bigrag.py#L283-L286)**
   - Add `metadata` parameter to `ainsert()`
   - Preserve title and metadata in `new_docs`

2. **[bigrag/bigrag.py::ainsert():299-310](bigrag/bigrag.py#L299-L310)**
   - Add `doc_title` and `doc_metadata` to chunks

3. **[bigrag/operate.py::extract_entities():314-322](bigrag/operate.py#L314-L322)**
   - Prepend document title to chunk content before LLM extraction

4. **[script_build.py::load_corpus():56-76](script_build.py#L56-L76)**
   - Return both documents AND metadata from load_corpus()

**Expected Improvement:** +2-3 F1 points
**Estimated Effort:** 3-4 hours
**Details:** See [BiG_RAG_DESIGN.md:452-631](BiG_RAG_DESIGN.md#L452-L631)

---

### Task 2.2: Document Deletion System ❌ 🔧

**Problem:** No way to delete indexed documents

**File to Modify:**
- **[bigrag/bigrag.py](bigrag/bigrag.py)** - Add new method `adelete_document()`

**Implementation:**
- Add `adelete_document(doc_id: str)` method
- Handle cascade deletion (delete if unique, update if shared)
- Clean up all storage layers: chunks, entities, edges, vectors
- Add synchronous wrapper `delete_document()`

**Expected Benefit:** GDPR compliance, data hygiene, testing support
**Estimated Effort:** 3-4 hours
**Details:** See [BiG_RAG_DESIGN.md:635-868](BiG_RAG_DESIGN.md#L635-L868)

---

## 🔄 PHASE 3: Three-Path Retrieval (AFTER PHASE 2)

**Priority:** HIGH - Core BiG-RAG functionality

### Task 3.1: Path C - Chunk Vector Search ❌

**Problem:** `vdb_chunks` exists but is never queried

**File to Modify:**
- **[bigrag/operate.py::_build_query_context():511-571](bigrag/operate.py#L511-L571)**

**What to Add:**
- New function `_get_chunk_data()` for direct + indirect chunk retrieval
- Direct: Vector search on `vdb_chunks` → 5 chunks
- Indirect: Extract source_ids from RRF results → 5 chunks

**Estimated Effort:** 2-4 hours
**Details:** See [BiG_RAG_DESIGN.md:873-935](BiG_RAG_DESIGN.md#L873-L935)

---

### Task 3.2: Integrate Path C into Query Flow ❌

**File to Modify:**
- **[bigrag/operate.py::_build_query_context()](bigrag/operate.py#L511-L571)**

**What to Add:**
- Call `_get_chunk_data()` after RRF fusion
- Combine 5 direct + 5 indirect chunks
- Optional reranking (if enabled)
- Append chunks to final knowledge output

**Estimated Effort:** 2-3 hours
**Details:** See [BiG_RAG_DESIGN.md:938-1024](BiG_RAG_DESIGN.md#L938-L1024)

---

### Task 3.3: Update kg_query() to Pass vdb_chunks ❌

**Files to Modify:**
1. **[bigrag/operate.py::kg_query():484](bigrag/operate.py#L484)** - Add `vdb_chunks` parameter
2. **[bigrag/bigrag.py::aquery():498](bigrag/bigrag.py#L498)** - Pass `self.vdb_chunks`

**Estimated Effort:** 1 hour
**Details:** See [BiG_RAG_DESIGN.md:1028-1086](BiG_RAG_DESIGN.md#L1028-L1086)

---

### Task 3.4: Semantic Reranking with Cross-Encoder ❌

**New File to Create:**
- **[bigrag/reranker.py](bigrag/reranker.py)** - Complete implementation provided

**What to Add:**
- Import `CrossEncoder` from sentence-transformers
- Implement `_semantic_rerank()` function
- Graceful fallback if reranker not available
- Use model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Estimated Effort:** 4-6 hours
**Details:** See [BiG_RAG_DESIGN.md:1090-1187](BiG_RAG_DESIGN.md#L1090-L1187)

---

### Task 3.5: Add Reranking Toggle to QueryParam ❌

**File to Modify:**
- **[bigrag/base.py::QueryParam:17](bigrag/base.py#L17)**

**What to Add:**
```python
enable_reranking: bool = False  # Default: disabled for speed
rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Estimated Effort:** 1 hour
**Details:** See [BiG_RAG_DESIGN.md:1191-1211](BiG_RAG_DESIGN.md#L1191-L1211)

---

## 📊 PHASE 4: Integration & Testing (AFTER PHASE 3)

**Focus:** End-to-end testing and validation

**Deliverables:**
- [ ] End-to-end integration tests
- [ ] Performance benchmarks (indexing throughput, query latency)
- [ ] API documentation
- [ ] Example notebooks

---

## 🚀 PHASE 5: Production Features (OPTIONAL)

**Focus:** Production deployment readiness

**Deliverables:**
- [ ] Milvus backend testing
- [ ] Neo4j backend testing
- [ ] REST API server
- [ ] Monitoring and logging
- [ ] Deployment guide

---

## ✅ PHASE 6: Quality Assurance Tools (FUTURE)

**Priority:** LOW - Implement after Phases 2 and 3 are stable

**Deliverables:**
- [ ] Graph validation script - [validate_graph.py](scripts/validate_graph.py)
- [ ] Orphaned node detection and cleanup
- [ ] Source ID integrity checks
- [ ] Performance profiling tools

**Note:** Complete implementation provided in [BiG_RAG_DESIGN.md:1612-1939](BiG_RAG_DESIGN.md#L1612-L1939)

---

## Implementation Order

### (MUST DO FIRST):
1. ✅ Task 2.1: Metadata Preservation (3-4 hours)
2. ✅ Task 2.2: Document Deletion (3-4 hours)
3. ✅ Test both features thoroughly

###  (DO NEXT):
4. ✅ Task 3.1: Path C Vector Search (2-4 hours)
5. ✅ Task 3.2: Integrate Path C (2-3 hours)
6. ✅ Task 3.3: Update kg_query() (1 hour)
7. ✅ Task 3.4: Semantic Reranking (4-6 hours)
8. ✅ Task 3.5: Reranking Toggle (1 hour)

### (VALIDATION):
9. ✅ Test three-path retrieval end-to-end
10. ✅ Compare EM/F1 scores before and after
11. ✅ Verify metadata preservation improves entity extraction

---

## Success Criteria

### Phase 2 Complete When:
- ✅ Metadata preserved in all chunks
- ✅ LLM sees document title during entity extraction
- ✅ Document deletion works with cascade updates
- ✅ +2-3 F1 point improvement observed

### Phase 3 Complete When:
- ✅ Path C chunk vector search working
- ✅ 5 direct + 5 indirect chunks retrieved
- ✅ Semantic reranking improves top-5 selection
- ✅ Final output: 5 structured knowledge + 5 chunks = 10 items

---

## Key Files Summary

**Core Implementation:**
- [bigrag/bigrag.py](bigrag/bigrag.py) - Main BiGRAG class (Tasks 2.1, 2.2, 3.3)
- [bigrag/operate.py](bigrag/operate.py) - Retrieval logic (Tasks 2.1, 3.1, 3.2, 3.3)
- [bigrag/base.py](bigrag/base.py) - QueryParam (Task 3.5)
- [script_build.py](script_build.py) - Corpus loading (Task 2.1)

**New Files:**
- [bigrag/reranker.py](bigrag/reranker.py) - Semantic reranking (Task 3.4)
- [scripts/validate_graph.py](scripts/validate_graph.py) - Graph validation (Phase 6)

**Documentation:**
- [BiG_RAG_DESIGN.md](BiG_RAG_DESIGN.md) - Detailed implementation guide
- [BiG_RAG_TECHNICAL_SPEC.md](BiG_RAG_TECHNICAL_SPEC.md) - Architecture specification
- [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) - This file

---

## Next Steps

1. **Start with Phase 2** (Metadata + Document Deletion)
2. **Test thoroughly** before moving to Phase 3
3. **Then implement Phase 3** (Three-Path Retrieval)
4. **Validate improvements** by comparing EM/F1 scores

---

## Questions or Issues?

Refer to:
- **Implementation details:** [BiG_RAG_DESIGN.md](BiG_RAG_DESIGN.md)
- **Architecture overview:** [BiG_RAG_TECHNICAL_SPEC.md](BiG_RAG_TECHNICAL_SPEC.md)
- **Code examples:** Both documents contain complete code samples for all tasks

---

**Status:** ✅ **READY TO IMPLEMENT**
**Start with:** Task 2.1 (Metadata Preservation)
