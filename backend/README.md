# BiG-RAG Backend API

FastAPI server for BiG-RAG knowledge graph retrieval and document management.

## Installation

```bash
# 1. Install backend dependencies
pip install -r requirements.txt

# 2. Install BiG-RAG framework (from parent directory)
cd ..
pip install -e .
```

## Running the Server

```bash
# Start API server
python server.py --data_source SingleTopic

# Or with custom port
python server.py --data_source SingleTopic --port 8002
```

The API will be available at:
- **API Base**: http://localhost:8001
- **Swagger Docs**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## Environment Variables

Create a `.env` file in the backend directory:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
```

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/ask` | Ask question, get answer with retrieval |
| POST | `/search` | Search knowledge graph |
| GET | `/stats` | Get graph statistics |

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/documents` | List all documents |
| GET | `/documents/{doc_id}` | Get document details |
| POST | `/documents` | Upload new document |
| DELETE | `/documents/{doc_id}` | Delete document |

### Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/eval/batch_generate` | Generate answers for evaluation |
| POST | `/eval/evaluate_results` | Evaluate generated answers |

### Graph Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/graph/export` | Export graph as JSON |

## Development

```bash
# Run with hot reload
uvicorn server:app --reload --host 0.0.0.0 --port 8001

# Run tests
pytest ../tests
```

## Deployment

See main README.md for deployment options (Docker, AWS, etc.).
