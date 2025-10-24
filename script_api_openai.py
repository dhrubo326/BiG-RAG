"""
BiG-RAG API Server (OpenAI Edition)

Designed for knowledge graphs built with OpenAI models:
- gpt-4o-mini for entity extraction
- text-embedding-3-small/large for embeddings
- NanoVectorDB for storage (not FAISS)

Usage:
    python script_api_openai.py --data_source demo_test
    python script_api_openai.py --data_source demo_test --port 8002

Prerequisites:
    - Knowledge graph built with tests/test_build_graph.py
    - OpenAI API key in openai_api_key.txt or environment variable
    - Graph files: vdb_entities.json, vdb_bipartite_edges.json, etc.
"""

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Dict, Any, Literal
import argparse

# Import BiG-RAG and OpenAI functions
from bigrag import BiGRAG, QueryParam
from bigrag.llm import gpt_4o_mini_complete, openai_embedding
from bigrag.utils import logger
import asyncio

# ============================================================================
# Configuration
# ============================================================================

# Parse command-line arguments
parser = argparse.ArgumentParser(description="BiG-RAG API Server (OpenAI Edition)")
parser.add_argument('--data_source', default='demo_test',
                    help='Dataset name (default: demo_test)')
parser.add_argument('--port', type=int, default=8001,
                    help='Server port (default: 8001)')
parser.add_argument('--host', default='0.0.0.0',
                    help='Server host (default: 0.0.0.0)')
args = parser.parse_args()
data_source = args.data_source

# Load OpenAI API key
def load_api_key():
    """Load OpenAI API key from file or environment"""
    api_key_file = "openai_api_key.txt"
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info("✓ Loaded OpenAI API key from openai_api_key.txt")
        return api_key
    elif os.getenv("OPENAI_API_KEY"):
        logger.info("✓ Using OpenAI API key from environment variable")
        return os.getenv("OPENAI_API_KEY")
    else:
        logger.warning("⚠ OPENAI_API_KEY not found!")
        logger.warning("Set it in openai_api_key.txt or as environment variable")
        return None

load_api_key()

# ============================================================================
# Initialize BiG-RAG
# ============================================================================

print(f"[INFO] Loading BiG-RAG for dataset: {data_source}")
print(f"[INFO] Working directory: expr/{data_source}")

# Initialize BiGRAG with OpenAI models
rag = BiGRAG(
    working_dir=f"expr/{data_source}",
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embedding,
    enable_llm_cache=True,
)

print(f"[INFO] BiG-RAG initialized successfully")

# Load statistics
stats = {
    "entities": 0,
    "edges": 0,
    "chunks": 0
}

try:
    # Load text chunks
    chunks_file = f"expr/{data_source}/kv_store_text_chunks.json"
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        stats["chunks"] = len(chunks)

    # Load entities from NanoVectorDB
    entities_file = f"expr/{data_source}/vdb_entities.json"
    if os.path.exists(entities_file):
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_vdb = json.load(f)
        stats["entities"] = len(entities_vdb.get('data', []))

    # Load bipartite edges from NanoVectorDB
    edges_file = f"expr/{data_source}/vdb_bipartite_edges.json"
    if os.path.exists(edges_file):
        with open(edges_file, 'r', encoding='utf-8') as f:
            edges_vdb = json.load(f)
        stats["edges"] = len(edges_vdb.get('data', []))

    print(f"[INFO] Graph statistics:")
    print(f"  - Entities: {stats['entities']}")
    print(f"  - Relations (Bipartite Edges): {stats['edges']}")
    print(f"  - Text Chunks: {stats['chunks']}")

except Exception as e:
    logger.warning(f"Could not load statistics: {e}")

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="BiG-RAG API (OpenAI Edition)",
    description="API for BiG-RAG retrieval using OpenAI models (gpt-4o-mini + text-embedding)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class AskRequest(BaseModel):
    """Request model for /ask endpoint (single question)"""
    question: str
    top_k: Optional[int] = 5
    mode: Optional[str] = "hybrid"

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is Artificial Intelligence?",
                "top_k": 5,
                "mode": "hybrid"
            }
        }


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    question: str
    retrieved_contexts: List[Dict[str, Any]]
    num_results: int
    mode: str
    message: Optional[str] = None


class SearchRequest(BaseModel):
    """Request model for /search endpoint (batch processing)"""
    queries: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "queries": [
                    "What is machine learning?",
                    "Who invented neural networks?"
                ]
            }
        }


class ChatMessage(BaseModel):
    """Single message in chat completions"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for /chat/completions endpoint (OpenAI-compatible)"""
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    dataset: str
    entities_count: int
    edges_count: int
    chunks_count: int
    api_version: str
    embedding_model: str
    llm_model: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BiG-RAG API Server (OpenAI Edition)",
        "version": "1.0.0",
        "dataset": data_source,
        "models": {
            "llm": "gpt-4o-mini",
            "embedding": "text-embedding-3-small (via OpenAI)"
        },
        "endpoints": {
            "ask": "/ask - Ask a single question (interactive Q&A)",
            "search": "/search - Batch retrieval (for training)",
            "chat": "/chat/completions - LLM completions with GPT-4o-mini",
            "health": "/health - System health check",
            "docs": "/docs - Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint with system status"""
    return HealthResponse(
        status="healthy",
        dataset=data_source,
        entities_count=stats["entities"],
        edges_count=stats["edges"],
        chunks_count=stats["chunks"],
        api_version="1.0.0",
        embedding_model="text-embedding-3-small (OpenAI)",
        llm_model="gpt-4o-mini (OpenAI)"
    )


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest):
    """
    Ask a single question and retrieve relevant context from knowledge graph

    This endpoint is designed for interactive testing and Q&A.

    **Request Body:**
    - `question`: Your question (string)
    - `top_k`: Number of results to retrieve (default: 5)
    - `mode`: Retrieval mode - "hybrid", "local", "global", or "naive" (default: "hybrid")

    **Retrieval Modes:**
    - `hybrid`: Combines entity-based and relation-based retrieval (recommended)
    - `local`: Entity-based retrieval only
    - `global`: Relation-based retrieval only
    - `naive`: Direct text chunk retrieval

    **Returns:**
    - Question asked
    - List of retrieved contexts with relevance scores
    - Number of results found
    - Mode used

    **Example:**
    ```json
    {
        "question": "What is Artificial Intelligence?",
        "top_k": 5,
        "mode": "hybrid"
    }
    ```
    """
    try:
        # Query BiGRAG (uses OpenAI embeddings internally)
        # Since this is already an async function, we can await directly
        result = await rag.aquery(
            request.question,
            param=QueryParam(
                mode=request.mode,
                only_need_context=True,
                top_k=request.top_k,
            )
        )

        # Format response
        if not result:
            return AskResponse(
                question=request.question,
                retrieved_contexts=[],
                num_results=0,
                mode=request.mode,
                message="No relevant context found for this question"
            )

        # Parse results
        contexts = []
        for i, item in enumerate(result, 1):
            contexts.append({
                "rank": i,
                "context": item if isinstance(item, str) else str(item),
                "coherence_score": 0.0  # NanoVectorDB doesn't provide scores in the same way
            })

        return AskResponse(
            question=request.question,
            retrieved_contexts=contexts,
            num_results=len(contexts),
            mode=request.mode,
            message="Successfully retrieved relevant context"
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/search", tags=["Retrieval"])
async def search(request: SearchRequest):
    """
    Knowledge graph retrieval endpoint (batch processing)

    Performs hybrid retrieval over bipartite graph for multiple queries.

    **Request Body:**
    - `queries`: List of query strings

    **Returns:**
    - List of JSON-encoded retrieval results

    **Example:**
    ```json
    {
        "queries": ["What is AI?", "What is ML?"]
    }
    ```
    """
    try:
        results = []

        for query_text in request.queries:
            result = await rag.aquery(
                query_text,
                param=QueryParam(
                    mode="hybrid",
                    only_need_context=True,
                    top_k=10,
                )
            )
            results.append(json.dumps({"query": query_text, "results": result}))

        return results

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@app.post("/chat/completions", tags=["LLM"])
async def chat_completions(request: ChatCompletionRequest):
    """
    GPT-4o-mini chat completions endpoint (OpenAI-compatible)

    Provides access to GPT-4o-mini for question answering and chat.

    **Request Body:**
    - `model`: Model name (default: gpt-4o-mini)
    - `messages`: List of chat messages with role and content
    - `temperature`: Sampling temperature (0-2, default: 1.0)
    - `max_tokens`: Maximum tokens to generate

    **Returns:**
    - OpenAI-compatible chat completion response

    **Example:**
    ```json
    {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ]
    }
    ```
    """
    try:
        # Extract system prompt and history
        system_prompt = None
        history_messages = []
        user_prompt = None

        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_prompt = msg.content
            elif msg.role == "assistant":
                history_messages.append({"role": "assistant", "content": msg.content})

        if user_prompt is None:
            raise HTTPException(status_code=400, detail="No user message found in request")

        # Call GPT-4o-mini
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        response_text = await gpt_4o_mini_complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )

        # Format OpenAI-compatible response
        import time
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # Not tracked
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"  BiG-RAG API Server Starting (OpenAI Edition)")
    print(f"  Dataset: {data_source}")
    print(f"  Host: {args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print(f"  Models: gpt-4o-mini + text-embedding-3-small")
    print("="*80 + "\n")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY not found in environment")
        print("[WARNING] API endpoints will not work properly")
        print("[WARNING] Set with: export OPENAI_API_KEY='your-key-here'\n")

    uvicorn.run(app, host=args.host, port=args.port)
