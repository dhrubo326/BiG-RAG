# BiG-RAG Testing Guide

Quick guide for testing your knowledge graph with questions.

---

## Prerequisites

✅ You have already built a knowledge graph (you have files in `expr/demo_test/`)

---

## Method 1: Using FastAPI Swagger UI (Recommended for Beginners)

### Step 1: Start the API Server

**IMPORTANT:** Use the correct API server based on how you built your knowledge graph:

- **If you built with `tests/test_build_graph.py` (using OpenAI models):**
  ```bash
  python script_api_openai.py --data_source demo_test
  ```

- **If you built with `script_build.py` (using FlagEmbedding/FAISS):**
  ```bash
  python script_api.py --data_source demo_test
  ```

**For most users:** Use `script_api_openai.py` (recommended)

**Output:**
```
[INFO] Loading BiG-RAG for dataset: demo_test
[INFO] Embedding model loaded
[INFO] Loaded X entities
[INFO] Loaded Y bipartite edges
[INFO] BiG-RAG initialized

================================================================================
  BiG-RAG API Server Starting
  Dataset: demo_test
  Host: 0.0.0.0:8001
  Docs: http://0.0.0.0:8001/docs
================================================================================
```

### Step 2: Open Swagger UI in Browser

Open your browser and navigate to:
```
http://localhost:8001/docs
```

### Step 3: Test the `/ask` Endpoint

1. Find the **Q&A** section in Swagger UI
2. Click on **POST /ask**
3. Click **"Try it out"**
4. Enter your question in the request body:

```json
{
  "question": "What is Artificial Intelligence?",
  "top_k": 5,
  "mode": "hybrid"
}
```

5. Click **"Execute"**
6. View the response with retrieved contexts and coherence scores

**Parameters:**
- `question`: Your question (string)
- `top_k`: Number of results (default: 5)
- `mode`: Retrieval mode - `hybrid`, `local`, `global`, or `naive` (default: `hybrid`)

**Retrieval Modes:**
- `hybrid` - Combines entity and relation retrieval (**recommended**)
- `local` - Entity-based retrieval only
- `global` - Relation-based retrieval only
- `naive` - Direct text chunk retrieval

---

## Method 2: Using Command Line Script

### Step 1: Start the API Server (if not already running)

```bash
# Use OpenAI-based server (recommended)
python script_api_openai.py --data_source demo_test

# OR use FlagEmbedding-based server (if you built with script_build.py)
# python script_api.py --data_source demo_test
```

### Step 2: Ask Questions via Command Line

Open a **new terminal** (keep the server running) and run:

```bash
# Basic usage
python test_ask_question.py "What is Artificial Intelligence?"

# Specify top_k
python test_ask_question.py "What is machine learning?" --top_k 3

# Specify retrieval mode
python test_ask_question.py "Explain neural networks" --mode hybrid --top_k 5

# Try different modes
python test_ask_question.py "What is deep learning?" --mode local
python test_ask_question.py "What is deep learning?" --mode global
python test_ask_question.py "What is deep learning?" --mode naive
```

**Example Output:**
```
🔍 Testing BiG-RAG Knowledge Graph
📡 Server: localhost:8001
⚙️  Mode: hybrid | Top-K: 5

================================================================================
📝 QUESTION
================================================================================
What is Artificial Intelligence?

================================================================================
📚 RETRIEVED CONTEXTS (3 results, mode: hybrid)
================================================================================

[Result 1] (Coherence: 0.8542)
--------------------------------------------------------------------------------
Artificial Intelligence (AI) refers to the simulation of human intelligence
in machines that are programmed to think like humans and mimic their actions...

[Result 2] (Coherence: 0.7821)
--------------------------------------------------------------------------------
AI systems can learn from experience, adjust to new inputs, and perform
human-like tasks...

================================================================================
✅ Query completed successfully
================================================================================
```

---

## Method 3: Using curl (Terminal)

```bash
# Basic request
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "top_k": 5, "mode": "hybrid"}'

# Pretty-print JSON output (requires jq)
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}' | jq .
```

---

## Method 4: Using Python Script Directly

You can also use the test files in the `tests/` folder:

```bash
# Test retrieval only
cd tests
python test_retrieval.py

# Test full end-to-end pipeline (retrieval + LLM generation)
python test_end_to_end.py
```

**Note:** These scripts require:
- OpenAI API key in `openai_api_key.txt`
- Test questions in `datasets/demo_test/raw/qa_test.json`

---

## Method 5: Using Python Requests (Interactive)

Create a simple Python script:

```python
import requests

url = "http://localhost:8001/ask"
payload = {
    "question": "What is Artificial Intelligence?",
    "top_k": 5,
    "mode": "hybrid"
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Question: {result['question']}")
print(f"Found {result['num_results']} results\n")

for ctx in result['retrieved_contexts']:
    print(f"[Result {ctx['rank']}] (Score: {ctx['coherence_score']:.4f})")
    print(ctx['context'][:200] + "...\n")
```

---

## Comparing Retrieval Modes

Test the same question with different modes to see which works best:

```bash
# Hybrid mode (entity + relation)
python test_ask_question.py "What is deep learning?" --mode hybrid

# Local mode (entity only)
python test_ask_question.py "What is deep learning?" --mode local

# Global mode (relation only)
python test_ask_question.py "What is deep learning?" --mode global

# Naive mode (text chunks)
python test_ask_question.py "What is deep learning?" --mode naive
```

**Recommendation:** Start with `hybrid` mode (it combines the best of both worlds).

---

## Troubleshooting

### Server Not Starting

**Error:** `Cannot connect to server at localhost:8001`

**Solutions:**
1. Make sure the API server is running:
   ```bash
   python script_api.py --data_source demo_test
   ```
2. Check if port 8001 is already in use:
   ```bash
   # Windows
   netstat -ano | findstr :8001

   # Linux/Mac
   lsof -i :8001
   ```
3. Use a different port:
   ```bash
   python script_api.py --data_source demo_test --port 8002
   python test_ask_question.py "Your question" --port 8002
   ```

### No Results Found

If you get "No relevant context found", try:
1. Check if your knowledge graph was built successfully:
   ```bash
   ls expr/demo_test/
   ```
   Should see: `vdb_entities.json`, `vdb_bipartite_edges.json`, etc.

2. Try different retrieval modes:
   ```bash
   python test_ask_question.py "Your question" --mode naive
   ```

3. Check if your question relates to the corpus you built

### Knowledge Graph Not Found

**Error:** `Knowledge graph not found at expr/demo_test`

**Solution:** Build the knowledge graph first:
```bash
python tests/test_build_graph.py
```

---

## Example Questions to Try

Based on your `demo_test` dataset, try questions like:
- "What is Artificial Intelligence?"
- "What is machine learning?"
- "Explain neural networks"
- "What is deep learning?"
- "What are the applications of computer vision?"

---

## Next Steps

Once you've tested basic retrieval:

1. **Test end-to-end with LLM** (requires OpenAI API key):
   ```bash
   cd tests
   python test_end_to_end.py
   ```

2. **Try the `/chat/completions` endpoint** in Swagger UI:
   - Uses GPT-4o-mini to synthesize answers
   - Requires `OPENAI_API_KEY` environment variable

3. **Build larger knowledge graphs** with more documents:
   - See `docs/DATASET_AND_CORPUS_GUIDE.md`

---

## API Endpoints Summary

| Endpoint | Purpose | Usage |
|----------|---------|-------|
| `/ask` | **Interactive Q&A** (single question) | ✅ Use this for testing |
| `/search` | Batch retrieval (for training) | Used during RL training |
| `/chat/completions` | LLM completions with GPT-4o-mini | Requires OpenAI API key |
| `/health` | Server health check | Check if server is running |
| `/docs` | Interactive API documentation | Swagger UI |

---

## Tips

1. **Start with Swagger UI** - easiest way to test
2. **Use `hybrid` mode** - gives best results in most cases
3. **Adjust `top_k`** - try 3-10 results depending on your needs
4. **Keep the server running** - you can ask multiple questions without restarting
5. **Check the logs** - the server prints useful debug info

Happy testing! 🚀
