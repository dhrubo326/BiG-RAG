"""
BiG-RAG API Server

Provides FastAPI endpoints for:
1. Knowledge graph retrieval (/search)
2. LLM completions with GPT-4o-mini (/chat/completions)
3. Health checks (/health)

Usage:
    python script_api.py --data_source 2WikiMultiHopQA
    python script_api.py --data_source HotpotQA --port 8002

Environment Variables:
    OPENAI_API_KEY: Required for GPT-4o-mini endpoint
"""

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import faiss
from FlagEmbedding import FlagAutoModel
from typing import List, Optional, Dict, Any, Literal
import argparse
from bigrag import BiGRAG, QueryParam
from bigrag.llm import gpt_4o_mini_complete
import asyncio
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="BiG-RAG API Server")
parser.add_argument('--data_source', default='2WikiMultiHopQA',
                    help='Dataset name (default: 2WikiMultiHopQA)')
parser.add_argument('--port', type=int, default=8001,
                    help='Server port (default: 8001)')
parser.add_argument('--host', default='0.0.0.0',
                    help='Server host (default: 0.0.0.0)')
args = parser.parse_args()
data_source = args.data_source

# ============================================================================
# Load BiG-RAG Components
# ============================================================================

print(f"[INFO] Loading BiG-RAG for dataset: {data_source}")

# Load FlagEmbedding model for query encoding
model = FlagAutoModel.from_finetuned(
    'BAAI/bge-large-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    devices="cpu",
)
print("[INFO] Embedding model loaded")

# Load entity FAISS index and corpus
print(f"[INFO] Loading entity embeddings from expr/{data_source}/")
index_entity = faiss.read_index(f"expr/{data_source}/index_entity.bin")
corpus_entity = []
with open(f"expr/{data_source}/kv_store_entities.json") as f:
    entities = json.load(f)
    for item in entities:
        corpus_entity.append(entities[item]['entity_name'])
print(f"[INFO] Loaded {len(corpus_entity)} entities")

# Load bipartite edge FAISS index and corpus
print(f"[INFO] Loading bipartite edge embeddings from expr/{data_source}/")
index_bipartite_edge = faiss.read_index(f"expr/{data_source}/index_bipartite_edge.bin")
corpus_bipartite_edge = []
with open(f"expr/{data_source}/kv_store_bipartite_edges.json") as f:
    bipartite_edges = json.load(f)
    for item in bipartite_edges:
        corpus_bipartite_edge.append(bipartite_edges[item]['content'])
print(f"[INFO] Loaded {len(corpus_bipartite_edge)} bipartite edges")

# Initialize BiGRAG instance
rag = BiGRAG(
    working_dir=f"expr/{data_source}",
)
print(f"[INFO] BiG-RAG initialized with working_dir: expr/{data_source}")

# ============================================================================
# Helper Functions
# ============================================================================

async def process_query(query_text: str, rag_instance: BiGRAG,
                       entity_match: List[str], bipartite_edge_match: List[str]) -> Dict[str, Any]:
    """Process a single query through BiG-RAG retrieval"""
    result = await rag_instance.aquery(
        query_text,
        param=QueryParam(only_need_context=True, top_k=10),
        entity_match=entity_match,
        bipartite_edge_match=bipartite_edge_match
    )
    return {"query": query_text, "result": result}


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def _format_results(results: List[int], corpus: List[str]) -> List[str]:
    """Format FAISS search results to text list"""
    return [corpus[result] for result in results]


def queries_to_results(queries: List[str]) -> List[str]:
    """
    Batch process queries through BiG-RAG retrieval

    Args:
        queries: List of query strings

    Returns:
        List of JSON-encoded results
    """
    # Encode queries to embeddings
    embeddings = model.encode_queries(queries)

    # Search entity index
    _, ids_entity = index_entity.search(embeddings, 5)  # Top-5 entities per query
    entity_match = {
        queries[i]: _format_results(ids_entity[i], corpus_entity)
        for i in range(len(ids_entity))
    }

    # Search bipartite edge index
    _, ids_edge = index_bipartite_edge.search(embeddings, 5)  # Top-5 edges per query
    bipartite_edge_match = {
        queries[i]: _format_results(ids_edge[i], corpus_bipartite_edge)
        for i in range(len(ids_edge))
    }

    # Process queries asynchronously
    results = []
    loop = always_get_an_event_loop()
    for query_text in tqdm(queries, desc="Processing queries", unit="query"):
        result = loop.run_until_complete(
            process_query(query_text, rag,
                         entity_match[query_text],
                         bipartite_edge_match[query_text])
        )
        results.append(json.dumps({"results": result["result"]}))

    return results

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="BiG-RAG API",
    description="API for BiG-RAG retrieval and GPT-4o-mini completions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class SearchRequest(BaseModel):
    """Request model for /search endpoint"""
    queries: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "queries": [
                    "What is the capital of France?",
                    "Who directed Nosferatu in 1922?"
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
    stream: Optional[bool] = False

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


class ChatCompletionResponse(BaseModel):
    """Response model for /chat/completions endpoint"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    dataset: str
    entities_count: int
    edges_count: int
    api_version: str


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

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BiG-RAG API Server",
        "version": "1.0.0",
        "dataset": data_source,
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
        entities_count=len(corpus_entity),
        edges_count=len(corpus_bipartite_edge),
        api_version="1.0.0"
    )


@app.post("/search", tags=["Retrieval"])
async def search(request: SearchRequest):
    """
    Knowledge graph retrieval endpoint

    Performs hybrid retrieval (entity-based + relation-based) over bipartite graph.

    **Request Body:**
    - `queries`: List of query strings

    **Returns:**
    - List of JSON-encoded retrieval results

    **Example:**
    ```json
    {
        "queries": ["What is the capital of France?"]
    }
    ```
    """
    try:
        results_str = queries_to_results(request.queries)
        return results_str
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


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
        # Encode query to embedding
        embeddings = model.encode_queries([request.question])

        # Search entity and bipartite edge indices
        _, ids_entity = index_entity.search(embeddings, request.top_k)
        entity_match = _format_results(ids_entity[0], corpus_entity)

        _, ids_edge = index_bipartite_edge.search(embeddings, request.top_k)
        bipartite_edge_match = _format_results(ids_edge[0], corpus_bipartite_edge)

        # Query BiGRAG
        loop = always_get_an_event_loop()
        result = loop.run_until_complete(
            rag.aquery(
                request.question,
                param=QueryParam(
                    mode=request.mode,
                    only_need_context=True,
                    top_k=request.top_k,
                ),
                entity_match=entity_match,
                bipartite_edge_match=bipartite_edge_match
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
                "context": item.get("<knowledge>", ""),
                "coherence_score": item.get("<coherence>", 0.0)
            })

        return AskResponse(
            question=request.question,
            retrieved_contexts=contexts,
            num_results=len(contexts),
            mode=request.mode,
            message="Successfully retrieved relevant context"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


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
    - `stream`: Enable streaming responses (not yet implemented)

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
    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="Streaming is not yet implemented. Set 'stream': false"
        )

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
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"  BiG-RAG API Server Starting")
    print(f"  Dataset: {data_source}")
    print(f"  Host: {args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print("="*80 + "\n")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY not found in environment")
        print("[WARNING] /chat/completions endpoint will not work")
        print("[WARNING] Set with: export OPENAI_API_KEY='your-key-here'\n")

    uvicorn.run(app, host=args.host, port=args.port)
