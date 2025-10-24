# BiG-RAG Testing & API Guide

Complete guide for testing your knowledge graph and using the API server.

---

## Quick Start

### 1. Start the API Server

```bash
python script_api.py --data_source demo_test
```

### 2. Test in Browser

Open: **http://localhost:8001/docs**

### 3. Ask a Question

Find **POST /ask** → Click "Try it out" → Execute:

```json
{
  "question": "What is Artificial Intelligence?",
  "top_k": 5,
  "mode": "hybrid"
}
```

---

## API Server Overview

BiG-RAG provides a unified FastAPI server that automatically detects your knowledge graph format:

- **OpenAI embeddings** (`vdb_*.json` files) → Uses OpenAI text-embedding API
- **Local embeddings** (`index_*.bin` files) → Uses FlagEmbedding (BAAI/bge-large-en-v1.5)

The server **automatically chooses** the right approach based on available files!

### Supported LLM Providers

The API server supports multiple LLM providers with minimal configuration:

| Provider | Models | Configuration |
|----------|--------|---------------|
| **OpenAI** (default) | gpt-4o-mini, gpt-4o, gpt-4 | Set `OPENAI_API_KEY` |
| **Anthropic** | claude-3-5-sonnet, claude-3-opus | Set `ANTHROPIC_API_KEY` |
| **Google** | gemini-pro, gemini-1.5-pro | Set `GOOGLE_API_KEY` |
| **Grok** | grok-beta | Set `XAI_API_KEY` |

**Default**: `gpt-4o-mini` (fast, cheap, good quality)

---

## Testing Methods

### Method 1: Swagger UI (Easiest)

1. **Start server:**
   ```bash
   python script_api.py --data_source demo_test
   ```

2. **Open browser:** http://localhost:8001/docs

3. **Test `/ask` endpoint:**
   - Find **Q&A** section
   - Click **POST /ask**
   - Click "Try it out"
   - Enter your question:
     ```json
     {
       "question": "What is Artificial Intelligence?",
       "top_k": 5,
       "mode": "hybrid"
     }
     ```
   - Click "Execute"

### Method 2: Command Line

```bash
# Start server
python script_api.py --data_source demo_test

# In another terminal, ask questions
python test_ask_question.py "What is Artificial Intelligence?"
python test_ask_question.py "What is machine learning?" --top_k 3
python test_ask_question.py "Explain neural networks" --mode hybrid
```

### Method 3: curl

```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is AI?",
    "top_k": 5,
    "mode": "hybrid"
  }'
```

### Method 4: Python Requests

```python
import requests

response = requests.post(
    "http://localhost:8001/ask",
    json={
        "question": "What is Artificial Intelligence?",
        "top_k": 5,
        "mode": "hybrid"
    }
)

result = response.json()
print(f"Found {result['num_results']} results")
for ctx in result['retrieved_contexts']:
    print(f"[{ctx['rank']}] {ctx['context'][:200]}...")
```

### Method 5: Test Scripts

```bash
# Test retrieval only
cd tests
python test_retrieval.py

# Test full pipeline (retrieval + LLM)
python test_end_to_end.py
```

---

## API Endpoints

### POST /ask - Interactive Q&A

Ask a single question and get retrieved context.

**Request:**
```json
{
  "question": "What is Artificial Intelligence?",
  "top_k": 5,
  "mode": "hybrid",
  "llm_provider": "openai"  // optional: openai, anthropic, google, grok
}
```

**Response:**
```json
{
  "question": "What is Artificial Intelligence?",
  "retrieved_contexts": [
    {
      "rank": 1,
      "context": "AI refers to...",
      "coherence_score": 0.85
    }
  ],
  "num_results": 5,
  "mode": "hybrid",
  "message": "Successfully retrieved relevant context"
}
```

**Retrieval Modes:**
- `hybrid` - Entity + relation retrieval (**recommended**)
- `local` - Entity-based only
- `global` - Relation-based only
- `naive` - Direct text chunks

### POST /search - Batch Retrieval

For training/batch processing multiple queries.

**Request:**
```json
{
  "queries": [
    "What is AI?",
    "What is machine learning?"
  ]
}
```

### POST /chat/completions - LLM Generation

OpenAI-compatible chat endpoint with RAG support.

**Request:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "What is AI?"}
  ],
  "use_rag": true,  // Use knowledge graph for context
  "temperature": 0.7,
  "llm_provider": "openai"  // optional
}
```

### GET /health - Health Check

Check server status and graph statistics.

**Response:**
```json
{
  "status": "healthy",
  "dataset": "demo_test",
  "entities_count": 150,
  "edges_count": 300,
  "chunks_count": 50,
  "embedding_mode": "openai",
  "available_providers": ["openai", "anthropic"]
}
```

---

## Switching LLM Providers

### Option 1: Environment Variables

```bash
# Use OpenAI (default)
export OPENAI_API_KEY="sk-..."
python script_api.py --data_source demo_test

# Use Claude
export ANTHROPIC_API_KEY="sk-ant-..."
python script_api.py --data_source demo_test --llm_provider anthropic

# Use Gemini
export GOOGLE_API_KEY="..."
python script_api.py --data_source demo_test --llm_provider google

# Use Grok
export XAI_API_KEY="..."
python script_api.py --data_source demo_test --llm_provider grok
```

### Option 2: Per-Request Override

```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is AI?",
    "llm_provider": "anthropic"
  }'
```

### Option 3: API Key File

Create provider-specific files:
- `openai_api_key.txt`
- `anthropic_api_key.txt`
- `google_api_key.txt`
- `grok_api_key.txt`

The server will automatically load them.

---

## Configuration

### Command Line Arguments

```bash
python script_api.py \
  --data_source demo_test \     # Dataset name
  --port 8001 \                 # Server port
  --host 0.0.0.0 \              # Server host
  --llm_provider openai \       # LLM provider (default: openai)
  --embedding_provider openai   # Embedding provider (auto-detected)
```

### Supported Configurations

| Build Method | Embedding Files | Server Mode |
|--------------|----------------|-------------|
| `tests/test_build_graph.py` | `vdb_*.json` | OpenAI embeddings |
| `script_build.py` | `index_*.bin` | FlagEmbedding (local) |

**Auto-detection**: Server checks your `expr/{dataset}/` folder and uses the appropriate mode.

---

## Troubleshooting

### Server Not Starting

**Error:** `Cannot connect to server at localhost:8001`

**Solutions:**
1. Check if server is running: `ps aux | grep script_api`
2. Check port availability:
   ```bash
   # Windows
   netstat -ano | findstr :8001

   # Linux/Mac
   lsof -i :8001
   ```
3. Use different port: `python script_api.py --port 8002`

### Error: "This event loop is already running"

**Fixed!** The current version uses proper async/await patterns.

### Error: "No module named 'FlagEmbedding'"

**Option 1**: Install FlagEmbedding:
```bash
pip install FlagEmbedding faiss-cpu
```

**Option 2**: Rebuild graph with OpenAI:
```bash
cd tests
python test_build_graph.py
```

### Error: "OPENAI_API_KEY not found"

```bash
# Set environment variable
export OPENAI_API_KEY="your-key-here"

# OR create file
echo "your-key-here" > openai_api_key.txt
```

### No Results Found

1. Check graph files exist:
   ```bash
   ls expr/demo_test/
   ```

2. Try different modes:
   ```bash
   python test_ask_question.py "Your question" --mode naive
   ```

3. Check if question relates to your corpus

### Knowledge Graph Not Found

Build the graph first:
```bash
cd tests
python test_build_graph.py
```

---

## Example Questions

Based on the `demo_test` dataset:

- "What is Artificial Intelligence?"
- "What is machine learning?"
- "Explain neural networks"
- "What is deep learning?"
- "What are the applications of computer vision?"

---

## Advanced Usage

### Comparing Retrieval Modes

```bash
# Test all modes with same question
for mode in hybrid local global naive; do
  python test_ask_question.py "What is deep learning?" --mode $mode
done
```

### Batch Testing

```python
import requests

questions = [
    "What is AI?",
    "What is ML?",
    "What is deep learning?"
]

for q in questions:
    response = requests.post(
        "http://localhost:8001/ask",
        json={"question": q, "mode": "hybrid"}
    )
    print(f"Q: {q}")
    print(f"Results: {response.json()['num_results']}\n")
```

### Using Different LLM Providers

```python
import requests

# Try multiple providers for comparison
providers = ["openai", "anthropic", "google"]

for provider in providers:
    response = requests.post(
        "http://localhost:8001/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is AI?"}],
            "llm_provider": provider,
            "use_rag": True
        }
    )
    print(f"{provider}: {response.json()['choices'][0]['message']['content']}\n")
```

---

## Performance Tips

1. **Use `hybrid` mode** for best results
2. **Adjust `top_k`**: 3-10 depending on needs
3. **Local embeddings** (FlagEmbedding) are faster for high query volume
4. **OpenAI embeddings** are easier for development/testing
5. **Keep server running** between queries to avoid reload time
6. **Use different ports** for multiple datasets simultaneously

---

## Production Deployment

### Using Local Embeddings (Faster)

1. **Install dependencies:**
   ```bash
   pip install FlagEmbedding faiss-cpu
   ```

2. **Build graph with FlagEmbedding:**
   ```bash
   python script_build.py --data_source your_dataset
   ```

3. **Start server:**
   ```bash
   python script_api.py --data_source your_dataset --host 0.0.0.0 --port 8001
   ```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements_graphrag_only.txt

EXPOSE 8001

CMD ["python", "script_api.py", "--data_source", "demo_test", "--host", "0.0.0.0"]
```

```bash
docker build -t bigrag-api .
docker run -p 8001:8001 -e OPENAI_API_KEY=$OPENAI_API_KEY bigrag-api
```

---

## Next Steps

1. **Build larger graphs** with more documents
   - See [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)

2. **Try RL training** to fine-tune models
   - See [CLAUDE.md](CLAUDE.md) for training instructions

3. **Integrate with your application**
   - Use `/chat/completions` endpoint (OpenAI-compatible)
   - Add RAG to existing chatbots

4. **Evaluate performance**
   - Run `cd evaluation && python eval.py`
   - Compare different retrieval modes
   - Test different LLM providers

---

## Summary

- ✅ **One unified API server** that auto-detects your setup
- ✅ **Multiple LLM providers** (OpenAI, Claude, Gemini, Grok)
- ✅ **Two embedding modes** (OpenAI API or local FlagEmbedding)
- ✅ **Simple switching** via environment variables or per-request
- ✅ **Production-ready** with Docker support

**Start testing**: `python script_api.py --data_source demo_test`

**View docs**: http://localhost:8001/docs

Happy testing! 🚀
