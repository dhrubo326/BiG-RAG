"""
BiG-RAG Enhanced RAG API - Universal Edition
Bipartite Graph Retrieval-Augmented Generation
Supports both Local LLM (Ollama) and Paid APIs (OpenAI, Anthropic, Gemini)
Uses text-embedding-3-large for retrieval (hardcoded)
"""
import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
from bigrag import BiGRAG, QueryParam
from bigrag.openai_embedding import OpenAIEmbedding
from openai import OpenAI
import sys
import argparse
import requests

# Parse arguments
parser = argparse.ArgumentParser(description="BiG-RAG API with multi-provider LLM support")
parser.add_argument("--data_source", type=str, default="MyEducationRAG", help="Dataset name")
parser.add_argument("--port", type=int, default=8001, help="Server port")

# LLM Provider Selection
parser.add_argument("--llm-provider", type=str, default="auto",
                    choices=["auto", "ollama", "openai", "anthropic", "gemini"],
                    help="LLM provider (auto=try ollama first, then openai)")

# Provider-specific model names
parser.add_argument("--ollama-model", type=str, default="gpt-oss:120b",
                    help="Ollama model (e.g., gpt-oss:120b, deepseek-r1:70b, llama3.1:70b)")
parser.add_argument("--ollama-url", type=str, default="http://192.168.2.54:11434",
                    help="Ollama server URL")
parser.add_argument("--openai-model", type=str, default="gpt-4o-mini",
                    help="OpenAI model (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)")
parser.add_argument("--anthropic-model", type=str, default="claude-3-5-sonnet-20241022",
                    help="Anthropic model (claude-3-5-sonnet, claude-3-opus)")
parser.add_argument("--gemini-model", type=str, default="gemini-1.5-pro",
                    help="Google Gemini model")

# Generation parameters
parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
parser.add_argument("--max-tokens", type=int, default=1500, help="Max tokens for LLM response")

args = parser.parse_args()
data_source = args.data_source

# Configuration
config = {
    "embedding_model": "text-embedding-3-large",  # HARDCODED as requested
    "embedding_dimensions": 3072,
    "temperature": args.temperature,
    "max_tokens": args.max_tokens,
    "initial_retrieval": 5,   # BiG-RAG dual-path: 5 initial candidates per path
    "multi_hop_depth": 10,    # BiG-RAG multi-hop: 10-hop adaptive traversal
    "enable_synthesis": True,

    # Provider-specific configs
    "llm_provider": args.llm_provider,
    "ollama": {
        "url": args.ollama_url,
        "base_url": f"{args.ollama_url}/v1",
        "model": args.ollama_model,
        "available": False
    },
    "openai": {
        "model": args.openai_model,
        "available": False
    },
    "anthropic": {
        "model": args.anthropic_model,
        "available": False
    },
    "gemini": {
        "model": args.gemini_model,
        "available": False
    }
}

# Active provider (will be set during initialization)
active_provider = None
active_model = None

# ============================================================================
# LLM CLIENT INITIALIZATION (Lazy Loading)
# ============================================================================

# Load OpenAI API key (required for embeddings)
openai_key_path = 'openai_api_key.txt'
if os.path.exists(openai_key_path):
    with open(openai_key_path, 'r') as f:
        openai_api_key = f.read().strip()
    config["openai"]["available"] = True
    config["openai"]["api_key"] = openai_api_key
    print("✅ OpenAI API key loaded")
else:
    print("⚠️ OpenAI API key not found (embeddings and OpenAI LLM unavailable)")
    if config["llm_provider"] == "openai":
        print("❌ Cannot use OpenAI provider without API key")
        sys.exit(1)

# Load Anthropic API key (optional)
anthropic_key_path = 'anthropic_api_key.txt'
if os.path.exists(anthropic_key_path):
    with open(anthropic_key_path, 'r') as f:
        config["anthropic"]["api_key"] = f.read().strip()
    config["anthropic"]["available"] = True
    print("✅ Anthropic API key loaded")

# Load Gemini API key (optional)
gemini_key_path = 'gemini_api_key.txt'
if os.path.exists(gemini_key_path):
    with open(gemini_key_path, 'r') as f:
        config["gemini"]["api_key"] = f.read().strip()
    config["gemini"]["available"] = True
    print("✅ Gemini API key loaded")

# Check Ollama availability
def check_ollama_health():
    """Check if Ollama server is reachable"""
    try:
        response = requests.get(f"{config['ollama']['url']}/api/tags", timeout=3)
        if response.status_code == 200:
            config["ollama"]["available"] = True
            return True
    except:
        pass
    config["ollama"]["available"] = False
    return False

# Initialize LLM clients (lazy - only when needed)
llm_clients = {}

def get_ollama_client():
    """Get or create Ollama client"""
    if "ollama" not in llm_clients:
        llm_clients["ollama"] = OpenAI(
            base_url=config["ollama"]["base_url"],
            api_key="ollama"  # Ollama doesn't need real key
        )
    return llm_clients["ollama"]

def get_openai_client():
    """Get or create OpenAI client"""
    if not config["openai"]["available"]:
        raise ValueError("OpenAI API key not available")
    if "openai" not in llm_clients:
        llm_clients["openai"] = OpenAI(api_key=config["openai"]["api_key"])
    return llm_clients["openai"]

def get_anthropic_client():
    """Get or create Anthropic client"""
    if not config["anthropic"]["available"]:
        raise ValueError("Anthropic API key not available")
    if "anthropic" not in llm_clients:
        try:
            from anthropic import Anthropic
            llm_clients["anthropic"] = Anthropic(api_key=config["anthropic"]["api_key"])
        except ImportError:
            raise ValueError("anthropic package not installed. Run: pip install anthropic")
    return llm_clients["anthropic"]

def get_gemini_client():
    """Get or create Gemini client"""
    if not config["gemini"]["available"]:
        raise ValueError("Gemini API key not available")
    if "gemini" not in llm_clients:
        try:
            import google.generativeai as genai
            genai.configure(api_key=config["gemini"]["api_key"])
            llm_clients["gemini"] = genai
        except ImportError:
            raise ValueError("google-generativeai package not installed. Run: pip install google-generativeai")
    return llm_clients["gemini"]

# Determine active provider
def determine_active_provider():
    """Determine which LLM provider to use based on availability and config"""
    global active_provider, active_model

    if config["llm_provider"] == "auto":
        # Auto mode: try Ollama first, then OpenAI
        if check_ollama_health():
            active_provider = "ollama"
            active_model = config["ollama"]["model"]
            print(f"✅ Auto-selected provider: Ollama ({active_model})")
        elif config["openai"]["available"]:
            active_provider = "openai"
            active_model = config["openai"]["model"]
            print(f"✅ Auto-selected provider: OpenAI ({active_model})")
        else:
            print("❌ No LLM provider available. Need either Ollama running or OpenAI API key")
            sys.exit(1)
    else:
        # Explicit provider selection
        provider = config["llm_provider"]

        if provider == "ollama":
            if not check_ollama_health():
                print(f"❌ Ollama not reachable at {config['ollama']['url']}")
                sys.exit(1)
            active_provider = "ollama"
            active_model = config["ollama"]["model"]
        elif provider == "openai":
            if not config["openai"]["available"]:
                print("❌ OpenAI API key not found")
                sys.exit(1)
            active_provider = "openai"
            active_model = config["openai"]["model"]
        elif provider == "anthropic":
            if not config["anthropic"]["available"]:
                print("❌ Anthropic API key not found")
                sys.exit(1)
            active_provider = "anthropic"
            active_model = config["anthropic"]["model"]
        elif provider == "gemini":
            if not config["gemini"]["available"]:
                print("❌ Gemini API key not found")
                sys.exit(1)
            active_provider = "gemini"
            active_model = config["gemini"]["model"]

        print(f"✅ LLM Provider: {active_provider} ({active_model})")

determine_active_provider()

# ============================================================================
# LOAD EMBEDDING MODEL & KNOWLEDGE GRAPH
# ============================================================================

if not config["openai"]["available"]:
    print("❌ Cannot initialize embeddings without OpenAI API key")
    sys.exit(1)

print(f"[INFO] Loading OpenAI embedding model: {config['embedding_model']}")
embedding_model = OpenAIEmbedding(
    model_name=config['embedding_model'],
    api_key=config["openai"]["api_key"],
    dimensions=config["embedding_dimensions"]
)

# Load FAISS indices
print(f"[INFO] Loading FAISS indices for dataset: {data_source}")
try:
    index_entity = faiss.read_index(f"expr/{data_source}/index_entity.bin")
    corpus_entity = []
    with open(f"expr/{data_source}/kv_store_entities.json", encoding='utf-8') as f:
        entities = json.load(f)
        for item in entities:
            corpus_entity.append(entities[item]['entity_name'])

    index_hyperedge = faiss.read_index(f"expr/{data_source}/index_hyperedge.bin")
    corpus_hyperedge = []
    with open(f"expr/{data_source}/kv_store_hyperedges.json", encoding='utf-8') as f:
        hyperedges = json.load(f)
        for item in hyperedges:
            corpus_hyperedge.append(hyperedges[item]['content'])
except FileNotFoundError as e:
    print(f"❌ Dataset not found: {e}")
    print(f"   Make sure expr/{data_source}/ contains the knowledge graph files")
    sys.exit(1)

# Initialize BiGRAG
print(f"[INFO] Initializing BiGRAG with n-ary hypergraph...")
rag = BiGRAG(working_dir=f"expr/{data_source}")

print("✅ Server initialization complete!")
print(f"📊 Loaded {len(corpus_entity)} entities, {len(corpus_hyperedge)} hyperedges")

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="BiG-RAG Enhanced RAG API - Universal Edition",
    description=f"BiG-RAG with OpenAI embeddings + {active_provider.upper()} LLM",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    question: str
    use_synthesis: Optional[bool] = True
    model: Optional[str] = None  # Override model per-request
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    provider: Optional[str] = None  # Override provider per-request

class BatchQueryRequest(BaseModel):
    questions: List[str]
    use_synthesis: Optional[bool] = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

async def process_query(query_text, rag_instance, entity_match, hyperedge_match):
    result = await rag_instance.aquery(
        query_text,
        param=QueryParam(only_need_context=True, top_k=config["multi_hop_depth"]),
        entity_match=entity_match,
        hyperedge_match=hyperedge_match
    )
    return {"query": query_text, "result": result}

def _format_results(results: List, corpus) -> List[str]:
    return [corpus[result] for result in results]

def retrieve_context(question: str) -> Dict[str, Any]:
    """
    Retrieve context using OpenAI embeddings + n-ary hypergraph traversal
    """
    start_time = time.time()

    # Encode query with OpenAI
    embeddings = embedding_model.encode_queries([question])

    # Normalize for cosine similarity
    import numpy as np
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Search entities (5 initial, following BiG-RAG)
    _, entity_ids = index_entity.search(embeddings, config["initial_retrieval"])
    entity_match = {question: _format_results(entity_ids[0], corpus_entity)}

    # Search hyperedges (5 initial, following BiG-RAG)
    _, hyperedge_ids = index_hyperedge.search(embeddings, config["initial_retrieval"])
    hyperedge_match = {question: _format_results(hyperedge_ids[0], corpus_hyperedge)}

    # Get detailed context from BiGRAG (n-ary hypergraph traversal)
    loop = always_get_an_event_loop()
    result = loop.run_until_complete(
        process_query(question, rag, entity_match[question], hyperedge_match[question])
    )

    retrieval_time = (time.time() - start_time) * 1000

    return {
        "context": result["result"],
        "entities": entity_match[question],
        "relations": hyperedge_match[question],
        "retrieval_time_ms": round(retrieval_time, 2)
    }

def synthesize_answer(
    question: str,
    context: List[Dict],
    provider: str = None,
    model_name: str = None,
    temperature: float = None,
    max_tokens: int = None
) -> Dict[str, Any]:
    """
    Synthesize answer using selected LLM provider with automatic fallback
    """
    start_time = time.time()

    # Determine which provider to use
    use_provider = provider if provider else active_provider
    use_model = model_name if model_name else active_model
    fallback_used = False

    # Format context for prompt
    context_text = "\n\n".join([
        f"Context {i+1} (Coherence: {ctx.get('<coherence>', 'N/A')}):\n{ctx.get('<knowledge>', 'N/A')}"
        for i, ctx in enumerate(context[:15])  # Use top 15 contexts for comprehensive coverage
    ])

    # Create prompt
    prompt = f"""You are an expert assistant answering questions about educational institutions in Bangladesh.

Based on the following context retrieved from a knowledge hypergraph using multi-hop reasoning, provide a comprehensive answer.

CONTEXT (ordered by coherence scores from n-ary hypergraph traversal):
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Answer in the same language as the question (Bengali or English)
- Use ALL relevant information from the context, even from lower coherence scores
- The BiG-RAG system uses n-ary hypergraph traversal, so contexts may contain related but indirect information
- Connect and synthesize information across multiple contexts to form a complete answer
- If exact details aren't in the highest coherence context, look in lower-scored contexts
- Include all relevant details found across the contexts
- Be helpful and comprehensive rather than overly conservative

ANSWER:"""

    # Try primary provider
    try:
        if use_provider == "ollama":
            client = get_ollama_client()
            response = client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in educational information about Bangladesh."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or config["temperature"],
                max_tokens=max_tokens or config["max_tokens"]
            )
            answer = response.choices[0].message.content.strip()
            model_used = response.model

        elif use_provider == "openai":
            client = get_openai_client()
            response = client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in educational information about Bangladesh."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or config["temperature"],
                max_tokens=max_tokens or config["max_tokens"]
            )
            answer = response.choices[0].message.content.strip()
            model_used = response.model

        elif use_provider == "anthropic":
            client = get_anthropic_client()
            response = client.messages.create(
                model=use_model,
                max_tokens=max_tokens or config["max_tokens"],
                temperature=temperature or config["temperature"],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.content[0].text
            model_used = response.model

        elif use_provider == "gemini":
            client = get_gemini_client()
            model = client.GenerativeModel(use_model)
            response = model.generate_content(prompt)
            answer = response.text
            model_used = use_model

        else:
            raise ValueError(f"Unknown provider: {use_provider}")

    except Exception as e:
        print(f"⚠️ {use_provider} failed: {str(e)}")

        # Fallback logic
        if use_provider == "ollama" and config["openai"]["available"]:
            print("🔄 Falling back to OpenAI...")
            try:
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=config["openai"]["model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in educational information about Bangladesh."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature or config["temperature"],
                    max_tokens=max_tokens or config["max_tokens"]
                )
                answer = response.choices[0].message.content.strip()
                model_used = response.model
                use_provider = "openai"
                fallback_used = True
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {str(fallback_error)}")
                answer = f"Error: All LLM providers failed. Primary: {str(e)}, Fallback: {str(fallback_error)}"
                model_used = "none"
        else:
            answer = f"Error: {use_provider} failed - {str(e)}"
            model_used = "none"

    synthesis_time = (time.time() - start_time) * 1000

    return {
        "answer": answer,
        "model_used": model_used,
        "provider_used": use_provider,
        "fallback_used": fallback_used,
        "synthesis_time_ms": round(synthesis_time, 2)
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "BiG-RAG Enhanced RAG API - Universal Edition",
        "version": "3.0.0",
        "embedding_model": config["embedding_model"],
        "llm_provider": active_provider,
        "llm_model": active_model,
        "dataset": data_source,
        "endpoints": {
            "/query": "Main query endpoint with synthesis",
            "/demo": "Demo-friendly formatted response",
            "/search": "Raw retrieval without synthesis",
            "/health": "Health check",
            "/llm-status": "LLM provider availability status",
            "/batch_query": "Batch query processing"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "dataset": data_source,
        "entities": len(corpus_entity),
        "hyperedges": len(corpus_hyperedge),
        "embedding_model": config["embedding_model"],
        "llm_provider": active_provider,
        "llm_model": active_model
    }

@app.get("/llm-status")
def llm_status():
    """Check availability of all LLM providers"""
    status = {
        "ollama": {
            "available": check_ollama_health(),
            "url": config["ollama"]["url"],
            "model": config["ollama"]["model"]
        },
        "openai": {
            "available": config["openai"]["available"],
            "model": config["openai"]["model"]
        },
        "anthropic": {
            "available": config["anthropic"]["available"],
            "model": config["anthropic"]["model"]
        },
        "gemini": {
            "available": config["gemini"]["available"],
            "model": config["gemini"]["model"]
        },
        "active_provider": active_provider,
        "active_model": active_model
    }
    return status

@app.post("/query")
def query_endpoint(request: QueryRequest):
    """
    Main query endpoint with retrieval + synthesis
    """
    start_time = time.time()

    # Retrieve context
    retrieval_result = retrieve_context(request.question)

    # Synthesize answer if enabled
    if request.use_synthesis and config["enable_synthesis"]:
        synthesis_result = synthesize_answer(
            request.question,
            retrieval_result["context"],
            provider=request.provider,
            model_name=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    else:
        synthesis_result = {
            "answer": "Synthesis disabled",
            "model_used": "none",
            "provider_used": "none",
            "fallback_used": False,
            "synthesis_time_ms": 0
        }

    total_time = (time.time() - start_time) * 1000

    return {
        "question": request.question,
        "answer": synthesis_result["answer"],
        "model_used": synthesis_result["model_used"],
        "provider_used": synthesis_result["provider_used"],
        "fallback_used": synthesis_result["fallback_used"],
        "retrieval_time_ms": retrieval_result["retrieval_time_ms"],
        "synthesis_time_ms": synthesis_result["synthesis_time_ms"],
        "total_time_ms": round(total_time, 2),
        "context": retrieval_result["context"],
        "entities_found": retrieval_result["entities"],
        "relations_found": retrieval_result["relations"]
    }

@app.post("/demo")
def demo_endpoint(request: QueryRequest):
    """
    Demo-friendly endpoint with beautiful formatting
    """
    result = query_endpoint(request)

    return {
        "question": result["question"],
        "answer": result["answer"],
        "summary": {
            "entities_found": len(result["entities_found"]),
            "relations_found": len(result["relations_found"]),
            "contexts_used": len(result["context"]),
            "retrieval_time": f"{result['retrieval_time_ms']/1000:.2f}s",
            "synthesis_time": f"{result['synthesis_time_ms']/1000:.2f}s",
            "total_time": f"{result['total_time_ms']/1000:.2f}s",
            "provider": result["provider_used"],
            "model": result["model_used"],
            "fallback": result["fallback_used"]
        },
        "top_entities": result["entities_found"][:5],
        "sample_context": result["context"][0]["<knowledge>"][:200] + "..." if result["context"] else "No context"
    }

@app.post("/search")
def search_endpoint(request: QueryRequest):
    """
    Raw retrieval endpoint without synthesis
    """
    retrieval_result = retrieve_context(request.question)
    return retrieval_result

@app.post("/batch_query")
def batch_query_endpoint(request: BatchQueryRequest):
    """
    Process multiple questions in batch
    """
    results = []
    for question in request.questions:
        query_req = QueryRequest(question=question, use_synthesis=request.use_synthesis)
        result = query_endpoint(query_req)
        results.append(result)

    return {
        "total_questions": len(request.questions),
        "results": results
    }

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"\n{'='*80}")
    print(f"🚀 Starting BiG-RAG Enhanced RAG API - Universal Edition")
    print(f"{'='*80}")
    print(f"Dataset: {data_source}")
    print(f"Embedding Model: {config['embedding_model']} ({config['embedding_dimensions']} dimensions)")
    print(f"LLM Provider: {active_provider}")
    print(f"LLM Model: {active_model}")
    if active_provider == "ollama":
        print(f"Ollama URL: {config['ollama']['url']}")
    print(f"Entities: {len(corpus_entity)}")
    print(f"Hyperedges: {len(corpus_hyperedge)}")
    print(f"Port: {args.port}")
    print(f"Retrieval Strategy: {config['initial_retrieval']} initial → {config['multi_hop_depth']}-hop traversal")
    print(f"{'='*80}\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
