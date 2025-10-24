"""
BiG-RAG Unified API Server

A robust, production-ready API server with:
- Auto-detection of knowledge graph format (OpenAI vs FlagEmbedding)
- Multiple LLM provider support (OpenAI, Claude, Gemini, Grok)
- Graceful fallback to gpt-4o-mini
- OpenAI-compatible endpoints
- Health monitoring and statistics

Usage:
    python script_api.py --data_source demo_test
    python script_api.py --data_source demo_test --llm_provider anthropic
    python script_api.py --data_source demo_test --port 8002

Environment Variables:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY
"""

import json
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Dict, Any, Literal
import argparse

# Import BiG-RAG core
from bigrag import BiGRAG, QueryParam
from bigrag.utils import logger

# ============================================================================
# LLM Provider Manager
# ============================================================================

class LLMProviderManager:
    """Manages multiple LLM providers with automatic fallback"""

    def __init__(self, default_provider: str = "openai"):
        self.default_provider = default_provider
        self.available_providers = {}
        self._initialize_providers()

    def _load_api_key(self, key_name: str, file_name: str) -> Optional[str]:
        """Load API key from environment or file"""
        # Check environment variable
        if os.getenv(key_name):
            return os.getenv(key_name)

        # Check file
        key_file = Path(file_name)
        if key_file.exists():
            with open(key_file, 'r') as f:
                key = f.read().strip()
            os.environ[key_name] = key
            return key

        return None

    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        # OpenAI
        openai_key = self._load_api_key("OPENAI_API_KEY", "openai_api_key.txt")
        if openai_key:
            try:
                from bigrag.llm import gpt_4o_mini_complete, gpt_4o_complete
                self.available_providers["openai"] = {
                    "gpt-4o-mini": gpt_4o_mini_complete,
                    "gpt-4o": gpt_4o_complete,
                    "gpt-4": gpt_4o_complete,  # Alias
                }
                logger.info("✓ OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"⚠ OpenAI provider failed: {e}")

        # Anthropic (Claude)
        anthropic_key = self._load_api_key("ANTHROPIC_API_KEY", "anthropic_api_key.txt")
        if anthropic_key:
            try:
                # We'll implement Claude support by wrapping the API
                self.available_providers["anthropic"] = self._get_anthropic_funcs()
                logger.info("✓ Anthropic (Claude) provider initialized")
            except Exception as e:
                logger.warning(f"⚠ Anthropic provider failed: {e}")

        # Google (Gemini)
        google_key = self._load_api_key("GOOGLE_API_KEY", "google_api_key.txt")
        if google_key:
            try:
                self.available_providers["google"] = self._get_google_funcs()
                logger.info("✓ Google (Gemini) provider initialized")
            except Exception as e:
                logger.warning(f"⚠ Google provider failed: {e}")

        # xAI (Grok)
        xai_key = self._load_api_key("XAI_API_KEY", "grok_api_key.txt")
        if xai_key:
            try:
                self.available_providers["grok"] = self._get_grok_funcs()
                logger.info("✓ xAI (Grok) provider initialized")
            except Exception as e:
                logger.warning(f"⚠ xAI provider failed: {e}")

    def _get_anthropic_funcs(self):
        """Create Anthropic LLM functions"""
        async def claude_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
            try:
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

                messages = []
                if history_messages:
                    messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

                response = await client.messages.create(
                    model=kwargs.get("model", "claude-3-5-sonnet-20241022"),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    temperature=kwargs.get("temperature", 1.0),
                    system=system_prompt or "",
                    messages=messages
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Claude API error: {e}")
                raise

        return {
            "claude-3-5-sonnet": claude_complete,
            "claude-3-opus": claude_complete,
            "claude-3-sonnet": claude_complete,
        }

    def _get_google_funcs(self):
        """Create Google Gemini functions"""
        async def gemini_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

                model = genai.GenerativeModel(
                    model_name=kwargs.get("model", "gemini-pro"),
                    system_instruction=system_prompt
                )

                # Format history
                chat_history = []
                for msg in history_messages:
                    role = "user" if msg["role"] == "user" else "model"
                    chat_history.append({"role": role, "parts": [msg["content"]]})

                chat = model.start_chat(history=chat_history)
                response = await chat.send_message_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=kwargs.get("temperature", 1.0),
                        max_output_tokens=kwargs.get("max_tokens", 1024),
                    )
                )
                return response.text
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                raise

        return {
            "gemini-pro": gemini_complete,
            "gemini-1.5-pro": gemini_complete,
        }

    def _get_grok_funcs(self):
        """Create xAI Grok functions"""
        async def grok_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    api_key=os.getenv("XAI_API_KEY"),
                    base_url="https://api.x.ai/v1"
                )

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

                response = await client.chat.completions.create(
                    model="grok-beta",
                    messages=messages,
                    temperature=kwargs.get("temperature", 1.0),
                    max_tokens=kwargs.get("max_tokens", 1024),
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Grok API error: {e}")
                raise

        return {
            "grok-beta": grok_complete,
            "grok": grok_complete,
        }

    async def complete(self, prompt: str, provider: Optional[str] = None,
                      model: Optional[str] = None, **kwargs) -> str:
        """
        Complete using specified provider or fallback to default

        Args:
            prompt: User prompt
            provider: LLM provider (openai, anthropic, google, grok)
            model: Specific model name
            **kwargs: Additional parameters (system_prompt, temperature, etc.)

        Returns:
            Generated text
        """
        # Use default provider if not specified
        provider = provider or self.default_provider

        # Fallback chain: requested → default → any available
        providers_to_try = [provider]
        if provider != self.default_provider and self.default_provider in self.available_providers:
            providers_to_try.append(self.default_provider)

        # Add any other available provider as last resort
        for p in self.available_providers.keys():
            if p not in providers_to_try:
                providers_to_try.append(p)

        last_error = None
        for prov in providers_to_try:
            if prov not in self.available_providers:
                continue

            try:
                provider_models = self.available_providers[prov]

                # Select model
                if model and model in provider_models:
                    func = provider_models[model]
                else:
                    # Use first available model for this provider
                    func = list(provider_models.values())[0]

                logger.info(f"Using provider: {prov}")
                return await func(prompt, **kwargs)

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {prov} failed: {e}")
                continue

        # All providers failed
        raise Exception(f"All LLM providers failed. Last error: {last_error}")

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.available_providers.keys())

# ============================================================================
# Embedding Manager
# ============================================================================

class EmbeddingManager:
    """Auto-detects and manages embedding strategy"""

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.mode = None
        self.model = None
        self.faiss_indices = {}
        self._detect_mode()

    def _detect_mode(self):
        """Detect whether to use OpenAI or FlagEmbedding"""
        # Check for OpenAI-style files (NanoVectorDB)
        if (self.working_dir / "vdb_entities.json").exists():
            self.mode = "openai"
            logger.info("✓ Detected OpenAI embeddings (NanoVectorDB format)")
            self._init_openai()

        # Check for FlagEmbedding-style files (FAISS)
        elif (self.working_dir / "index_entity.bin").exists():
            self.mode = "flagembedding"
            logger.info("✓ Detected FlagEmbedding (FAISS format)")
            self._init_flagembedding()

        else:
            logger.warning("⚠ No embedding files detected! Server may not work properly.")
            self.mode = "none"

    def _init_openai(self):
        """Initialize OpenAI embedding mode"""
        try:
            from bigrag.llm import openai_embedding
            self.embedding_func = openai_embedding
            logger.info("✓ OpenAI embedding function loaded")
        except Exception as e:
            logger.error(f"Failed to load OpenAI embeddings: {e}")
            raise

    def _init_flagembedding(self):
        """Initialize FlagEmbedding mode"""
        try:
            import faiss
            from FlagEmbedding import FlagAutoModel

            # Load FlagEmbedding model
            self.model = FlagAutoModel.from_finetuned(
                'BAAI/bge-large-en-v1.5',
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                devices="cpu",
            )
            logger.info("✓ FlagEmbedding model loaded")

            # Load FAISS indices
            self.faiss_indices["entity"] = faiss.read_index(str(self.working_dir / "index_entity.bin"))
            self.faiss_indices["edge"] = faiss.read_index(str(self.working_dir / "index_bipartite_edge.bin"))
            logger.info("✓ FAISS indices loaded")

            # Load corpus mappings
            with open(self.working_dir / "kv_store_entities.json") as f:
                entities = json.load(f)
                self.corpus_entity = [entities[item]['entity_name'] for item in entities]

            with open(self.working_dir / "kv_store_bipartite_edges.json") as f:
                edges = json.load(f)
                self.corpus_edge = [edges[item]['content'] for item in edges]

            logger.info(f"✓ Loaded {len(self.corpus_entity)} entities, {len(self.corpus_edge)} edges")

        except ImportError:
            logger.error("FlagEmbedding not installed! Install with: pip install FlagEmbedding faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize FlagEmbedding: {e}")
            raise

    def get_embedding_func(self):
        """Get embedding function for BiGRAG"""
        if self.mode == "openai":
            return self.embedding_func
        elif self.mode == "flagembedding":
            # For FlagEmbedding, we handle embeddings manually in query
            return None
        else:
            return None

    async def search_entities(self, query: str, top_k: int = 5):
        """Search for entities matching query"""
        if self.mode == "flagembedding":
            embeddings = self.model.encode_queries([query])
            _, ids = self.faiss_indices["entity"].search(embeddings, top_k)
            return [self.corpus_entity[i] for i in ids[0]]
        return None

    async def search_edges(self, query: str, top_k: int = 5):
        """Search for bipartite edges matching query"""
        if self.mode == "flagembedding":
            embeddings = self.model.encode_queries([query])
            _, ids = self.faiss_indices["edge"].search(embeddings, top_k)
            return [self.corpus_edge[i] for i in ids[0]]
        return None

# ============================================================================
# Configuration & Initialization
# ============================================================================

parser = argparse.ArgumentParser(description="BiG-RAG Unified API Server")
parser.add_argument('--data_source', default='demo_test',
                    help='Dataset name (default: demo_test)')
parser.add_argument('--port', type=int, default=8001,
                    help='Server port (default: 8001)')
parser.add_argument('--host', default='0.0.0.0',
                    help='Server host (default: 0.0.0.0)')
parser.add_argument('--llm_provider', default='openai',
                    choices=['openai', 'anthropic', 'google', 'grok'],
                    help='Default LLM provider (default: openai)')
args = parser.parse_args()

# Initialize managers
print(f"\n[INFO] Initializing BiG-RAG for dataset: {args.data_source}")
print(f"[INFO] Working directory: expr/{args.data_source}\n")

embedding_manager = EmbeddingManager(f"expr/{args.data_source}")
llm_manager = LLMProviderManager(default_provider=args.llm_provider)

# Initialize BiGRAG
from bigrag.llm import gpt_4o_mini_complete
rag = BiGRAG(
    working_dir=f"expr/{args.data_source}",
    llm_model_func=gpt_4o_mini_complete,  # Fallback for entity extraction
    embedding_func=embedding_manager.get_embedding_func(),
    enable_llm_cache=True,
)

print(f"[INFO] BiG-RAG initialized")
print(f"[INFO] Embedding mode: {embedding_manager.mode}")
print(f"[INFO] Available LLM providers: {', '.join(llm_manager.get_available_providers())}")
print(f"[INFO] Default LLM provider: {args.llm_provider}\n")

# Load statistics
stats = {"entities": 0, "edges": 0, "chunks": 0}
try:
    chunks_file = f"expr/{args.data_source}/kv_store_text_chunks.json"
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        stats["chunks"] = len(chunks)

    if embedding_manager.mode == "openai":
        entities_file = f"expr/{args.data_source}/vdb_entities.json"
        if os.path.exists(entities_file):
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_vdb = json.load(f)
            stats["entities"] = len(entities_vdb.get('data', []))

        edges_file = f"expr/{args.data_source}/vdb_bipartite_edges.json"
        if os.path.exists(edges_file):
            with open(edges_file, 'r', encoding='utf-8') as f:
                edges_vdb = json.load(f)
            stats["edges"] = len(edges_vdb.get('data', []))

    elif embedding_manager.mode == "flagembedding":
        stats["entities"] = len(embedding_manager.corpus_entity)
        stats["edges"] = len(embedding_manager.corpus_edge)

    print(f"[INFO] Graph statistics:")
    print(f"  - Entities: {stats['entities']}")
    print(f"  - Relations: {stats['edges']}")
    print(f"  - Text Chunks: {stats['chunks']}\n")

except Exception as e:
    logger.warning(f"Could not load statistics: {e}")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="BiG-RAG Unified API",
    description="Multi-provider RAG API with auto-detection of embedding format",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# Pydantic Models
# ============================================================================

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    mode: Optional[str] = "hybrid"
    llm_provider: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is Artificial Intelligence?",
                "top_k": 5,
                "mode": "hybrid",
                "llm_provider": "openai"
            }
        }


class AskResponse(BaseModel):
    question: str
    retrieved_contexts: List[Dict[str, Any]]
    num_results: int
    mode: str
    llm_provider_used: Optional[str] = None
    message: Optional[str] = None


class SearchRequest(BaseModel):
    queries: List[str]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    llm_provider: Optional[str] = None
    use_rag: Optional[bool] = False


class HealthResponse(BaseModel):
    status: str
    dataset: str
    entities_count: int
    edges_count: int
    chunks_count: int
    embedding_mode: str
    available_providers: List[str]
    default_provider: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "BiG-RAG Unified API Server",
        "version": "2.0.0",
        "dataset": args.data_source,
        "embedding_mode": embedding_manager.mode,
        "default_llm_provider": args.llm_provider,
        "available_providers": llm_manager.get_available_providers(),
        "endpoints": {
            "ask": "/ask - Interactive Q&A",
            "search": "/search - Batch retrieval",
            "chat": "/chat/completions - LLM generation (OpenAI-compatible)",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        dataset=args.data_source,
        entities_count=stats["entities"],
        edges_count=stats["edges"],
        chunks_count=stats["chunks"],
        embedding_mode=embedding_manager.mode,
        available_providers=llm_manager.get_available_providers(),
        default_provider=args.llm_provider
    )


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest):
    """
    Ask a single question with knowledge graph retrieval

    Supports multiple retrieval modes and LLM providers.
    """
    try:
        # For FlagEmbedding mode, pre-compute entity/edge matches
        entity_match = None
        edge_match = None

        if embedding_manager.mode == "flagembedding":
            entity_match = await embedding_manager.search_entities(request.question, request.top_k)
            edge_match = await embedding_manager.search_edges(request.question, request.top_k)

        # Query BiGRAG
        result = await rag.aquery(
            request.question,
            param=QueryParam(
                mode=request.mode,
                only_need_context=True,
                top_k=request.top_k,
            ),
            entity_match=entity_match,
            bipartite_edge_match=edge_match
        )

        if not result:
            return AskResponse(
                question=request.question,
                retrieved_contexts=[],
                num_results=0,
                mode=request.mode,
                llm_provider_used=request.llm_provider or args.llm_provider,
                message="No relevant context found"
            )

        # Format results
        contexts = []
        for i, item in enumerate(result, 1):
            if isinstance(item, dict):
                contexts.append({
                    "rank": i,
                    "context": item.get("<knowledge>", str(item)),
                    "coherence_score": item.get("<coherence>", 0.0)
                })
            else:
                contexts.append({
                    "rank": i,
                    "context": str(item),
                    "coherence_score": 0.0
                })

        return AskResponse(
            question=request.question,
            retrieved_contexts=contexts,
            num_results=len(contexts),
            mode=request.mode,
            llm_provider_used=request.llm_provider or args.llm_provider,
            message="Successfully retrieved relevant context"
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/search", tags=["Retrieval"])
async def search(request: SearchRequest):
    """Batch retrieval for multiple queries"""
    try:
        results = []
        for query_text in request.queries:
            entity_match = None
            edge_match = None

            if embedding_manager.mode == "flagembedding":
                entity_match = await embedding_manager.search_entities(query_text, 5)
                edge_match = await embedding_manager.search_edges(query_text, 5)

            result = await rag.aquery(
                query_text,
                param=QueryParam(mode="hybrid", only_need_context=True, top_k=10),
                entity_match=entity_match,
                bipartite_edge_match=edge_match
            )
            results.append(json.dumps({"query": query_text, "results": result}))

        return results

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@app.post("/chat/completions", tags=["LLM"])
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint

    Supports multiple LLM providers and optional RAG integration.
    """
    try:
        # Extract prompts
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
            raise HTTPException(status_code=400, detail="No user message found")

        # Optional RAG: Retrieve context and prepend to prompt
        if request.use_rag:
            entity_match = None
            edge_match = None

            if embedding_manager.mode == "flagembedding":
                entity_match = await embedding_manager.search_entities(user_prompt, 3)
                edge_match = await embedding_manager.search_edges(user_prompt, 3)

            context_results = await rag.aquery(
                user_prompt,
                param=QueryParam(mode="hybrid", only_need_context=True, top_k=3),
                entity_match=entity_match,
                bipartite_edge_match=edge_match
            )

            if context_results:
                context_str = "\n\n".join([
                    str(item) if not isinstance(item, dict) else item.get("<knowledge>", str(item))
                    for item in context_results[:3]
                ])
                user_prompt = f"Context:\n{context_str}\n\nQuestion: {user_prompt}"

        # Call LLM
        response_text = await llm_manager.complete(
            prompt=user_prompt,
            provider=request.llm_provider,
            model=request.model,
            system_prompt=system_prompt,
            history_messages=history_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
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
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"LLM error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"  BiG-RAG Unified API Server")
    print(f"  Dataset: {args.data_source}")
    print(f"  Host: {args.host}:{args.port}")
    print(f"  Embedding: {embedding_manager.mode}")
    print(f"  LLM Providers: {', '.join(llm_manager.get_available_providers())}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print("="*80 + "\n")

    if not llm_manager.get_available_providers():
        print("[WARNING] No LLM providers available!")
        print("[WARNING] Set API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.\n")

    uvicorn.run(app, host=args.host, port=args.port)
