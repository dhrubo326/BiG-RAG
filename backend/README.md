# BiG-RAG Backend API

FastAPI server for BiG-RAG knowledge graph retrieval and document management.

**Status:** ✅ Production Ready (Fully Refactored - Nov 2025)

---

## 🚀 Quick Start

### Installation

```bash
# 1. Install BiG-RAG framework (from project root)
cd ..
pip install -e .

# 2. Install backend dependencies
cd backend
pip install fastapi uvicorn python-multipart
```

### Running the Server

```bash
# Start API server with default dataset
python server.py --data_source SingleTopic

# With custom configuration
python server.py --data_source SingleTopic --port 8002 --llm_provider anthropic
```

**Server will be available at:**
- **API Base:** http://localhost:8001
- **Swagger Docs:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

---

## 🔐 Environment Variables

Create a `.env` file in the project root (`BiG-RAG/.env`):

```env
# LLM Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...

# Optional Configuration
WORKING_DIR=./expr
DEFAULT_DATASET=SingleTopic
```

---

## 📡 API Endpoints

### Health & System (2 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API overview |
| GET | `/health` | System health, uptime, and statistics |

### Document Management (5 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload .txt or .md files with metadata |
| GET | `/documents` | List all documents (with filtering & pagination) |
| GET | `/documents/{id}` | Get document details, entities, and relations |
| DELETE | `/documents/{id}` | Delete document (soft or hard delete) |
| POST | `/documents/rebuild` | Rebuild entire knowledge graph |

### Job Management (1 endpoint)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/jobs/{job_id}` | Get background job status and progress |

### Graph Management (4 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/graph/stats` | Knowledge graph statistics |
| GET | `/graph/export` | Export graph in Cytoscape-compatible JSON |
| GET | `/graph/subgraph/neighbors` | Get node neighbors (N-hop traversal) |
| GET | `/graph/subgraph/search` | Search nodes by text query |

### Retrieval (2 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Ask question with RAG retrieval |
| POST | `/search` | Batch document retrieval |

### Evaluation (6 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/eval/retrieval` | Evaluate retrieval quality (Precision, Recall, F1, MRR) |
| POST | `/eval/answer` | Evaluate answer quality (EM, F1, ROUGE-L) |
| POST | `/eval/compare` | Compare retrieval configurations side-by-side |
| POST | `/eval/batch` | Batch evaluation with multiple questions |
| POST | `/eval/batch_generate` | Generate answers from CSV questions |
| POST | `/eval/evaluate_results` | Evaluate CSV results for accuracy |

### LLM (1 endpoint)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/completions` | OpenAI-compatible chat completions with RAG |

**Total: 21 endpoints**

---

## 💡 Example Usage

### Ask a Question

```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "mode": "hybrid",
    "top_k": 5,
    "enable_reranking": true
  }'
```

### Upload a Document

```bash
curl -X POST "http://localhost:8001/documents/upload" \
  -F "file=@my_document.md" \
  -F "title=My Research Paper" \
  -F 'metadata={"category":"research","tags":["AI","ML"]}'
```

### List Documents

```bash
curl "http://localhost:8001/documents?search=france&limit=10"
```

### Export Knowledge Graph

```bash
curl "http://localhost:8001/graph/export?data_source=SingleTopic&limit=100&sample_strategy=diverse"
```

### Chat with LLM (with RAG)

```bash
curl -X POST "http://localhost:8001/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "use_rag": true,
    "llm_provider": "openai",
    "model": "gpt-4o-mini"
  }'
```

### Check Job Status

```bash
curl "http://localhost:8001/jobs/job-abc123"
```

---

## 🏗️ Architecture

### Modular Design

The backend has been fully refactored into a clean, modular architecture:

```
backend/
├── server.py                     # 212 lines (was 2,713)
│
└── api/
    ├── core/                     # Dependency injection & managers
    │   ├── dependencies.py       # FastAPI DI
    │   └── managers.py           # LLM & Embedding managers
    │
    ├── routes/                   # 7 route modules (APIRouter)
    │   ├── health.py
    │   ├── documents.py
    │   ├── graph.py
    │   ├── evaluation.py
    │   ├── retrieval.py
    │   ├── jobs.py
    │   └── llm.py
    │
    ├── services/                 # Business logic (14 modules)
    │   ├── answer_generation.py
    │   ├── csv_evaluation.py
    │   ├── evaluation.py
    │   ├── graph_export.py
    │   ├── kg_utils.py
    │   └── ...
    │
    └── models/                   # Pydantic schemas
        ├── models.py
        └── models_eval.py
```

**Benefits:**
- ✅ 92% reduction in main server file
- ✅ Clean separation of concerns
- ✅ Easy to maintain and extend
- ✅ Fully tested and production-ready

---

## 🛠️ Development

### Run with Hot Reload

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8001
```

### Run Tests

```bash
# From project root
pytest tests/

# Test specific endpoints
python backend/test_endpoints.py
```

### Code Quality

```bash
# Linting
flake8 backend/

# Type checking
mypy backend/

# Format code
black backend/
```

---

## 🚢 Deployment

### Docker

```bash
# Build image
docker build -t bigrag-backend .

# Run container
docker run -p 8001:8001 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/expr:/app/expr \
  bigrag-backend
```

### Production with Gunicorn

```bash
gunicorn server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --timeout 120
```

---

## 📚 Additional Documentation

- **Refactoring Summary:** See [REFACTORING_COMPLETE.md](./REFACTORING_COMPLETE.md)
- **API Docs:** http://localhost:8001/docs (when server is running)
- **Main Project README:** [../README.md](../README.md)
- **BiG-RAG Design:** [../docs/technical/BiG_RAG_DESIGN.md](../docs/technical/BiG_RAG_DESIGN.md)

---

## ⚙️ Configuration

### Command Line Options

```bash
python server.py --help
```

Options:
- `--data_source` - Dataset name (default: from config)
- `--port` - Server port (default: 8001)
- `--host` - Server host (default: 0.0.0.0)
- `--llm_provider` - Default LLM provider (openai, anthropic, google, grok)

### Multi-LLM Support

The backend automatically detects and initializes all available LLM providers based on API keys in your `.env` file:

- **OpenAI:** GPT-4, GPT-4o, GPT-4o-mini
- **Anthropic:** Claude 3.5 Sonnet, Claude 3 Opus
- **Google:** Gemini Pro, Gemini 1.5 Pro
- **xAI:** Grok Beta

Graceful fallback: If your preferred provider fails, the system automatically falls back to other available providers.

---

## 🐛 Troubleshooting

### Server won't start

```bash
# Check if port is in use
netstat -ano | findstr :8001  # Windows
lsof -i :8001                 # Linux/Mac

# Kill process using port
taskkill /PID <pid> /F        # Windows
kill -9 <pid>                 # Linux/Mac
```

### Import errors

```bash
# Ensure BiG-RAG is installed
pip install -e ..

# Reinstall dependencies
pip install -r requirements.txt
```

### No documents found

```bash
# Check working directory
ls expr/SingleTopic/

# Rebuild graph if needed
curl -X POST http://localhost:8001/documents/rebuild \
  -F "data_source=SingleTopic"
```

---

## 📝 License

Same as BiG-RAG project (see [../LICENSE](../LICENSE))

---

## 🙏 Acknowledgments

Built with:
- **FastAPI** - Modern web framework
- **BiG-RAG** - Bipartite Graph RAG framework
- **Uvicorn** - ASGI server

---

**Last Updated:** November 8, 2025
**Backend Version:** 3.0.0
**Status:** Production Ready
