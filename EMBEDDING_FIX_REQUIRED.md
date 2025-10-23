# Missing Embeddings - Need to Re-run Build

## Problem Found

The retrieval test failed because **embeddings were never created**!

The first build only did Phase 1 (entity extraction) but skipped Phase 2 (embedding generation).

### Error:
```
TypeError: object of type 'NoneType' has no len()
```

**Root cause:** The FAISS indices (index.bin, index_entity.bin, index_bipartite_edge.bin) were never created, so retrieval cannot work.

---

## Solution: Re-run Build with Embeddings

I've updated `test_build_graph.py` to include **Phase 2: Embedding Generation**.

### What to Do:

```bash
# Step 1: Clean up the incomplete build
rd /s /q expr\demo_test

# Step 2: Run the FIXED build script (now includes embeddings!)
python test_build_graph.py
```

### What the Fixed Script Does:

**Phase 1: Entity Extraction** (same as before)
- Extract entities and relations
- Create bipartite graph
- Save metadata JSON files

**Phase 2: Embedding Generation** (NEW!)
- Load all text chunks, entities, and edges
- Create embeddings using OpenAI text-embedding-3-large
- Generate FAISS indices for fast similarity search
- Save:
  - `corpus.npy` + `index.bin` (text chunks)
  - `corpus_entity.npy` + `index_entity.bin` (entities)
  - `corpus_bipartite_edge.npy` + `index_bipartite_edge.bin` (relations)

---

## Expected Output After Fix:

```
================================================================================
PHASE 1: Extracting Knowledge (Entities & Relations)
================================================================================
... (same as before)

================================================================================
PHASE 2: Creating Embeddings & FAISS Indices
================================================================================
[1/3] Loading and embedding text chunks...
  - Loaded 10 text chunks
  - Creating embeddings (batch_size=32)...
  DONE: index.bin created (10 vectors, 3072 dimensions)

[2/3] Loading and embedding entities...
  - Loaded 140 entities
  - Creating embeddings (batch_size=32)...
  DONE: index_entity.bin created (140 vectors, 3072 dimensions)

[3/3] Loading and embedding bipartite edges...
  - Loaded 73 bipartite edges
  - Creating embeddings (batch_size=32)...
  DONE: index_bipartite_edge.bin created (73 vectors, 3072 dimensions)

 All embeddings and indices created successfully!
```

---

## Time & Cost:

- **Additional time**: +2-3 minutes for embedding creation
- **Additional cost**: ~$0.005 USD (very cheap)
- **Total time**: 5-10 minutes for complete build
- **Total cost**: ~$0.01-0.02 USD

---

## After Re-running Build:

Then test retrieval again:

```bash
python test_retrieval.py
```

This should now work properly! ✓

---

**Ready?** Run:
```bash
rd /s /q expr\demo_test
python test_build_graph.py
```
