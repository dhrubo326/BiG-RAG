# BiG-RAG API Servers Explained

## Two Different API Servers

BiG-RAG has **two API server implementations** that use different embedding approaches:

---

## 📘 Option 1: `script_api_openai.py` (Recommended)

### When to Use
- ✅ You built your knowledge graph with **`tests/test_build_graph.py`**
- ✅ Your graph uses **OpenAI embeddings** (`text-embedding-3-small` or `text-embedding-3-large`)
- ✅ Your graph files include: `vdb_entities.json`, `vdb_bipartite_edges.json` (NanoVectorDB format)
- ✅ You DON'T have FAISS indices: `index_entity.bin`, `index_bipartite_edge.bin`

### How It Works
```
User Query
  → OpenAI text-embedding-3-small (encode query)
  → BiGRAG.aquery() (uses NanoVectorDB internally)
  → Returns retrieved context
```

### Requirements
```bash
pip install openai tiktoken fastapi uvicorn pydantic
# NO need for FlagEmbedding!
```

### Start Server
```bash
python script_api_openai.py --data_source demo_test
```

### Features
- ✅ Uses OpenAI embeddings (same as your graph)
- ✅ Integrated with BiGRAG class directly
- ✅ Supports all retrieval modes (hybrid, local, global, naive)
- ✅ `/ask` endpoint for single questions
- ✅ `/search` endpoint for batch queries
- ✅ `/chat/completions` for LLM generation

---

## 📗 Option 2: `script_api.py` (Legacy/Alternative)

### When to Use
- ✅ You built your knowledge graph with **`script_build.py`**
- ✅ Your graph uses **FlagEmbedding** (BAAI/bge-large-en-v1.5)
- ✅ Your graph files include: `index_entity.bin`, `index_bipartite_edge.bin` (FAISS indices)
- ✅ You have pre-computed embeddings: `corpus_entity.npy`, `corpus_bipartite_edge.npy`

### How It Works
```
User Query
  → FlagEmbedding/bge-large-en-v1.5 (encode query locally)
  → FAISS.search() (fast vector search)
  → BiGRAG.aquery() (with pre-matched entities/edges)
  → Returns retrieved context
```

### Requirements
```bash
pip install FlagEmbedding faiss-cpu fastapi uvicorn pydantic
```

### Start Server
```bash
python script_api.py --data_source demo_test
```

### Features
- ✅ Uses local embeddings (no API calls for encoding)
- ✅ FAISS for ultra-fast vector search
- ✅ Good for production with high query volume
- ✅ `/ask` endpoint for single questions (added recently)
- ✅ `/search` endpoint for batch queries
- ✅ `/chat/completions` for LLM generation

---

## Which Server Should You Use?

### Check Your Graph Files

Run this command to see what you have:

```bash
ls expr/demo_test/
```

**If you see:**
- ✅ `vdb_entities.json` and `vdb_bipartite_edges.json` → Use **`script_api_openai.py`**
- ✅ `index_entity.bin` and `index_bipartite_edge.bin` → Use **`script_api.py`**

---

## Comparison Table

| Feature | script_api_openai.py | script_api.py |
|---------|---------------------|---------------|
| **Embedding Model** | OpenAI text-embedding-3-small | FlagEmbedding (bge-large-en-v1.5) |
| **Storage Format** | NanoVectorDB (.json) | FAISS (.bin) |
| **Query Encoding** | OpenAI API call | Local model inference |
| **Speed** | Slower (API latency) | Faster (local) |
| **Cost** | API costs per query | Free (local) |
| **Dependencies** | openai, tiktoken | FlagEmbedding, faiss-cpu |
| **GPU Support** | No (API-based) | Yes (optional for encoding) |
| **Best For** | Testing, development | Production, high volume |

---

## Your Situation

Based on your error:
```
ModuleNotFoundError: No module named 'FlagEmbedding'
```

**You are trying to use `script_api.py` but:**
1. You likely built your graph with `tests/test_build_graph.py` (OpenAI-based)
2. Your graph files are in NanoVectorDB format (not FAISS)
3. You don't need FlagEmbedding!

**Solution:** Use `script_api_openai.py` instead!

```bash
python script_api_openai.py --data_source demo_test
```

---

## How to Switch Between Servers

### From OpenAI → FlagEmbedding

If you want to use FlagEmbedding for faster local queries:

1. **Install FlagEmbedding:**
   ```bash
   pip install FlagEmbedding faiss-cpu
   ```

2. **Re-build your graph with `script_build.py`:**
   ```bash
   python script_build.py --data_source demo_test
   ```

3. **This will create FAISS indices:**
   - `index_entity.bin`
   - `index_bipartite_edge.bin`
   - `corpus_entity.npy`
   - `corpus_bipartite_edge.npy`

4. **Start the server:**
   ```bash
   python script_api.py --data_source demo_test
   ```

### From FlagEmbedding → OpenAI

If you want to use OpenAI embeddings instead:

1. **Re-build your graph with `tests/test_build_graph.py`:**
   ```bash
   cd tests
   python test_build_graph.py
   ```

2. **This will create NanoVectorDB files:**
   - `vdb_entities.json`
   - `vdb_bipartite_edges.json`

3. **Start the server:**
   ```bash
   python script_api_openai.py --data_source demo_test
   ```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'FlagEmbedding'`

**Problem:** You're running `script_api.py` but don't have FlagEmbedding installed.

**Solutions:**
1. **Use OpenAI server instead:** `python script_api_openai.py --data_source demo_test`
2. **OR install FlagEmbedding:** `pip install FlagEmbedding faiss-cpu`

### Error: `FAISS index not found`

**Problem:** You're running `script_api.py` but your graph was built with OpenAI models.

**Solution:** Use `python script_api_openai.py --data_source demo_test`

### Error: `vdb_entities.json not found`

**Problem:** You're running `script_api_openai.py` but your graph was built with FlagEmbedding.

**Solution:** Use `python script_api.py --data_source demo_test`

---

## Recommended Setup

**For most users (especially beginners):**

1. ✅ Build with **OpenAI models** (easier, no local dependencies):
   ```bash
   cd tests
   python test_build_graph.py
   ```

2. ✅ Use **OpenAI API server**:
   ```bash
   python script_api_openai.py --data_source demo_test
   ```

3. ✅ Test with Swagger UI:
   ```
   http://localhost:8001/docs
   ```

**For production (high query volume):**

1. Build with **FlagEmbedding** (local, faster):
   ```bash
   python script_build.py --data_source demo_test
   ```

2. Use **FlagEmbedding API server**:
   ```bash
   python script_api.py --data_source demo_test
   ```

---

## Summary

- **Two servers for two embedding approaches**
- **Check your graph files to know which to use**
- **`script_api_openai.py`** = OpenAI embeddings + NanoVectorDB (your case!)
- **`script_api.py`** = FlagEmbedding + FAISS (alternative)
- **Both provide the same endpoints:** `/ask`, `/search`, `/chat/completions`
- **Choose based on your build method**

Happy testing! 🚀
