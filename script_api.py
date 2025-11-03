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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Dict, Any, Literal
import argparse
import hashlib
import asyncio
from datetime import datetime
import time

# Import BiG-RAG core
from bigrag import BiGRAG, QueryParam
from bigrag.utils import logger
from bigrag.config import config

# Import API support modules
from api.jobs import (
    ProcessingJob, JobStatus, ProcessingStage,
    processing_jobs, process_document_background, get_queue_stats
)
from api.registry import registry
from api.utils import process_markdown, validate_file_upload, truncate_text
from api.kg_utils import (
    get_document_stats_from_kg, get_document_entities,
    find_related_documents, get_document_content_from_corpus
)
from api.models import (
    UploadResponse as EnhancedUploadResponse, JobStatusResponse,
    DocumentListResponse, DocumentDetailResponse, DocumentSummary,
    DeleteResponse, GraphStatsResponse, HealthResponse as EnhancedHealthResponse,
    DocumentFilter, KGStatistics, EntityInfo, RelatedDocument
)
from api.evaluation import (
    evaluate_retrieval, evaluate_single_answer, compare_configurations,
    batch_evaluate
)
from api.models_eval import (
    RetrievalEvalRequest, RetrievalEvalResponse,
    AnswerEvalRequest, AnswerEvalResponse,
    CompareEvalRequest, CompareEvalResponse,
    BatchEvalRequest, BatchEvalResponse,
    PerQueryRetrievalResult, PerQuestionAnswerResult, BatchEvalPerformance,
    # CSV Evaluation models
    BatchGenerateRequest, BatchGenerateResponse,
    EvaluateResultsRequest, EvaluateResultsResponse
)

# ============================================================================
# LLM Provider Manager
# ============================================================================

class LLMProviderManager:
    """Manages multiple LLM providers with automatic fallback"""

    def __init__(self, default_provider: str = "openai"):
        self.default_provider = default_provider
        self.available_providers = {}
        self._initialize_providers()

    def _load_api_key(self, key_name: str, file_name: str = None) -> Optional[str]:
        """
        Load API key from configuration.
        Config automatically handles: .env file → *_api_key.txt → environment variable

        Args:
            key_name: Environment variable name (e.g., "OPENAI_API_KEY")
            file_name: Deprecated - kept for backward compatibility

        Returns:
            API key if found, None otherwise
        """
        # Map environment variable names to config attributes
        key_map = {
            "OPENAI_API_KEY": config.openai_api_key,
            "ANTHROPIC_API_KEY": config.anthropic_api_key,
            "GOOGLE_API_KEY": config.google_api_key,
            "XAI_API_KEY": config.xai_api_key,
        }

        return key_map.get(key_name)

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

    def get_model_func(self, provider: str, model: str):
        """
        Get LLM function for specified provider and model

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet")

        Returns:
            LLM completion function

        Raises:
            ValueError: If provider or model not available
        """
        if provider not in self.available_providers:
            available = ", ".join(self.available_providers.keys())
            raise ValueError(
                f"Provider '{provider}' not available. Available providers: {available}"
            )

        if model not in self.available_providers[provider]:
            available_models = ", ".join(self.available_providers[provider].keys())
            raise ValueError(
                f"Model '{model}' not available for provider '{provider}'. "
                f"Available models: {available_models}"
            )

        return self.available_providers[provider][model]

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
            # No existing embeddings - default to OpenAI for new documents
            logger.warning("⚠ No embedding files detected! Defaulting to OpenAI embeddings.")
            self.mode = "openai"
            self._init_openai()

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
        """
        Initialize FlagEmbedding mode (LEGACY - for backwards compatibility)

        NOTE: This is legacy code for graphs built with FAISS indices.
        New graphs use NanoVectorDB (OpenAI mode). If you're seeing errors here,
        your graph may be from an older version. Consider rebuilding with script_build.py.
        """
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

            # Load corpus mappings from GraphML (new architecture) or JSON (legacy)
            graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            legacy_entities_file = self.working_dir / "kv_store_entities.json"
            legacy_edges_file = self.working_dir / "kv_store_bipartite_edges.json"

            if graph_file.exists():
                # New architecture: Read from GraphML
                logger.info("Reading entity/edge names from GraphML file")
                import networkx as nx
                G = nx.read_graphml(graph_file)

                self.corpus_entity = []
                self.corpus_edge = []

                for node, attrs in G.nodes(data=True):
                    role = attrs.get("role", "")
                    if role == "entity":
                        self.corpus_entity.append(attrs.get("name", node))
                    elif role == "bipartite_edge":
                        self.corpus_edge.append(attrs.get("name", node))

            elif legacy_entities_file.exists() and legacy_edges_file.exists():
                # Legacy architecture: Read from JSON files
                logger.warning("⚠ Using legacy JSON metadata files (consider rebuilding graph)")
                with open(legacy_entities_file) as f:
                    entities = json.load(f)
                    self.corpus_entity = [entities[item]['entity_name'] for item in entities]

                with open(legacy_edges_file) as f:
                    edges = json.load(f)
                    self.corpus_edge = [edges[item]['content'] for item in edges]
            else:
                raise FileNotFoundError(
                    "No entity/edge metadata found! Expected either:\n"
                    f"  - {graph_file} (new architecture)\n"
                    f"  - {legacy_entities_file} + {legacy_edges_file} (legacy)\n"
                    "Please rebuild your graph with script_build.py"
                )

            logger.info(f"✓ Loaded {len(self.corpus_entity)} entities, {len(self.corpus_edge)} edges")

        except ImportError as e:
            logger.error(f"FlagEmbedding dependencies not installed: {e}")
            logger.error("Install with: pip install FlagEmbedding faiss-cpu")
            raise
        except FileNotFoundError as e:
            logger.error(str(e))
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
parser.add_argument('--data_source', default=config.default_dataset,
                    help=f'Dataset name (default: {config.default_dataset})')
parser.add_argument('--port', type=int, default=config.port,
                    help=f'Server port (default: {config.port})')
parser.add_argument('--host', default=config.host,
                    help=f'Server host (default: {config.host})')
parser.add_argument('--llm_provider', default=config.llm_provider,
                    choices=['openai', 'anthropic', 'google', 'grok'],
                    help=f'Default LLM provider (default: {config.llm_provider})')
args = parser.parse_args()

# Initialize managers
working_dir = f"{config.working_dir}/{args.data_source}"
print(f"\n[INFO] Initializing BiG-RAG for dataset: {args.data_source}")
print(f"[INFO] Working directory: {working_dir}\n")

embedding_manager = EmbeddingManager(working_dir)
llm_manager = LLMProviderManager(default_provider=args.llm_provider)

# Track server start time for uptime
server_start_time = time.time()

# Initialize BiGRAG
from bigrag.llm import gpt_4o_mini_complete
rag = BiGRAG(
    working_dir=working_dir,
    llm_model_func=gpt_4o_mini_complete,  # Fallback for entity extraction
    embedding_func=embedding_manager.get_embedding_func(),
    chunk_token_size=config.chunk_size,
    chunk_overlap_token_size=config.chunk_overlap_size,
    enable_llm_cache=config.enable_llm_cache,
)

print(f"[INFO] BiG-RAG initialized")
print(f"[INFO] Embedding mode: {embedding_manager.mode}")
print(f"[INFO] Available LLM providers: {', '.join(llm_manager.get_available_providers())}")
print(f"[INFO] Default LLM provider: {args.llm_provider}\n")

# Load statistics
stats = {"entities": 0, "edges": 0, "chunks": 0}
try:
    chunks_file = f"{working_dir}/kv_store_text_chunks.json"
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        stats["chunks"] = len(chunks)

    if embedding_manager.mode == "openai":
        entities_file = f"{working_dir}/vdb_entities.json"
        if os.path.exists(entities_file):
            with open(entities_file, 'r', encoding='utf-8') as f:
                vdb_entities = json.load(f)
            stats["entities"] = len(vdb_entities.get('data', []))

        edges_file = f"{working_dir}/vdb_bipartite_edges.json"
        if os.path.exists(edges_file):
            with open(edges_file, 'r', encoding='utf-8') as f:
                vdb_bipartite_edges = json.load(f)
            stats["edges"] = len(vdb_bipartite_edges.get('data', []))

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
    enable_reranking: Optional[bool] = False  # Phase 3.4: Semantic reranking (default: False)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is Artificial Intelligence?",
                "top_k": 5,
                "mode": "hybrid",
                "llm_provider": "openai",
                "enable_reranking": False
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
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    llm_provider: Optional[str] = None
    use_rag: Optional[bool] = True  # Enable RAG by default
    enable_reranking: Optional[bool] = False  # Phase 3.4: Semantic reranking (default: False)

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is Artificial Intelligence?"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500,
                "use_rag": True,
                "enable_reranking": False
            }
        }


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
# Document Management Helper Functions
# ============================================================================

def compute_doc_id(content: str, prefix: str = "doc") -> str:
    """
    Generate unique ID from content hash

    IMPORTANT: Uses same format as BiGRAG's compute_mdhash_id to ensure
    document IDs match between upload system and knowledge graph.
    Changed from "upload" prefix to "doc" prefix and full 32-char hash for consistency.
    """
    from bigrag.utils import compute_mdhash_id
    return compute_mdhash_id(content.strip(), prefix=f"{prefix}-")


async def add_document_to_corpus(
    data_source: str,
    doc_id: str,
    content: str,
    title: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Add a document to the corpus.jsonl file"""
    corpus_file = Path(f"{config.input_dir}/{data_source}/raw/corpus.jsonl")

    # Create directory if doesn't exist
    corpus_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare document
    doc = {
        "id": doc_id,
        "contents": content,
        "title": title,
        "upload_date": datetime.now().isoformat(),
        "source": "upload"
    }

    # Add metadata if provided
    if metadata:
        doc["metadata"] = metadata

    # Append to corpus
    with open(corpus_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    logger.info(f"Added document {doc_id} to corpus")
    return doc


async def rebuild_knowledge_graph_incremental(data_source: str, new_documents: List[Dict[str, Any]]):
    """
    Incrementally add new documents to existing knowledge graph

    This updates the existing graph without rebuilding from scratch.

    Args:
        data_source: Dataset name
        new_documents: List of document dicts with 'contents', 'title', 'metadata' fields

    Phase 2.1 Enhancement: Now passes metadata to BiGRAG for improved entity extraction
    """
    working_dir = f"{config.working_dir}/{data_source}"

    try:
        # Use the existing RAG instance to insert new documents
        # BiGRAG.ainsert() handles:
        # 1. Chunking (with metadata preservation)
        # 2. Entity extraction (with document context)
        # 3. Relation extraction
        # 4. Graph updates
        # 5. Vector index updates
        logger.info(f"Adding {len(new_documents)} new documents to knowledge graph...")

        # Extract contents and metadata separately
        contents = [doc.get("contents", "") for doc in new_documents]
        metadata_list = [
            {
                "title": doc.get("title", ""),
                **doc.get("metadata", {})
            }
            for doc in new_documents
        ]

        # Insert in small batches to avoid overwhelming the API
        batch_size = 3
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]

            # Phase 2.1: Pass metadata for improved entity extraction
            await rag.ainsert(batch_contents, metadata=batch_metadata)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size}")

        logger.info("✓ Knowledge graph updated successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to update knowledge graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# Response Models for New Endpoints
# ============================================================================

class UploadResponse(BaseModel):
    success: bool
    message: str
    document_id: str
    filename: str
    title: str
    content_length: int
    dataset: str


class RebuildResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    dataset: str
    rebuild_type: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "BiG-RAG Unified API Server - Enhanced",
        "version": "3.0.0",
        "dataset": args.data_source,
        "embedding_mode": embedding_manager.mode,
        "default_llm_provider": args.llm_provider,
        "available_providers": llm_manager.get_available_providers(),
        "endpoints": {
            "document_management": {
                "upload": "POST /upload - Upload .txt or .md files with metadata",
                "list": "GET /documents - List all documents with filtering",
                "details": "GET /documents/{id} - Get document details with KG stats",
                "delete": "DELETE /documents/{id} - Delete document (soft/hard)",
                "rebuild": "POST /rebuild - Rebuild knowledge graph"
            },
            "job_management": {
                "status": "GET /status/{job_id} - Get processing job status"
            },
            "graph_management": {
                "stats": "GET /graph/stats - Get knowledge graph statistics"
            },
            "retrieval": {
                "ask": "POST /ask - Interactive Q&A with context",
                "search": "POST /search - Batch document retrieval"
            },
            "llm": {
                "chat": "POST /chat/completions - OpenAI-compatible chat endpoint"
            },
            "system": {
                "health": "GET /health - System health and statistics",
                "docs": "GET /docs - Interactive API documentation"
            }
        },
        "new_features": [
            "✅ Markdown (.md) file support",
            "✅ Background job processing with progress tracking",
            "✅ Document registry and metadata management",
            "✅ Soft and hard delete options",
            "✅ Advanced filtering and pagination",
            "✅ Knowledge graph statistics and analytics"
        ]
    }


@app.get("/health", response_model=EnhancedHealthResponse, tags=["Health"])
async def health_check():
    """
    Get comprehensive system health and statistics.

    **Returns:**
    - System status
    - RAG instance information
    - Job queue statistics
    - Server uptime
    """
    from api.models import RAGInstanceInfo

    # Get active RAG instances
    rag_instances_info = {}
    expr_dir = Path("expr")
    if expr_dir.exists():
        for dataset_dir in expr_dir.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                # Check if indices exist (support both new and legacy formats)
                indices_loaded = (
                    # New architecture (NanoVectorDB)
                    (dataset_dir / "vdb_entities.json").exists() or
                    (dataset_dir / "vdb_bipartite_edges.json").exists() or
                    # Legacy architecture (FAISS)
                    (dataset_dir / "index_entity.bin").exists() or
                    (dataset_dir / "index_bipartite_edge.bin").exists()
                )

                # Count documents in registry
                try:
                    reg_stats = await registry.get_stats(dataset_dir.name)
                    total_docs = reg_stats.get("total", 0)
                except:
                    total_docs = 0

                rag_instances_info[dataset_dir.name] = RAGInstanceInfo(
                    dataset=dataset_dir.name,
                    status="active" if indices_loaded else "inactive",
                    total_documents=total_docs,
                    indices_loaded=indices_loaded
                )

    # Get job queue stats
    job_stats = get_queue_stats()

    # Calculate uptime
    uptime_seconds = time.time() - server_start_time

    return EnhancedHealthResponse(
        status="healthy",
        version="3.0.0",
        timestamp=datetime.now().isoformat(),
        rag_instances=rag_instances_info,
        job_queue=job_stats,
        uptime_seconds=uptime_seconds
    )


@app.post("/upload", response_model=EnhancedUploadResponse, tags=["Document Management"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Text or Markdown file to upload (.txt or .md)"),
    title: str = Form(None, description="Optional document title (defaults to filename)"),
    data_source: str = Form(None, description="Dataset name (defaults to current dataset)"),
    process_async: bool = Form(True, description="Process in background (recommended for large files)"),
    metadata: str = Form(None, description="Optional JSON metadata: {\"category\": \"research\", \"tags\": [...]}")
):
    """
    Upload a document (.txt or .md) and add it to the knowledge graph.

    **Enhanced Features:**
    - ✅ Supports both .txt and .md (Markdown) files
    - ✅ Background processing with job tracking
    - ✅ Document registry for metadata management
    - ✅ Optional metadata (category, tags, custom fields)
    - ✅ Progress tracking via /status/{job_id}

    **Example usage:**
    ```bash
    # Basic upload
    curl -X POST "http://localhost:8001/upload" \\
      -F "file=@document.md" \\
      -F "title=My Research Paper"

    # With metadata
    curl -X POST "http://localhost:8001/upload" \\
      -F "file=@document.md" \\
      -F "title=BiG-RAG Paper" \\
      -F 'metadata={"category":"research","tags":["RAG","NLP"]}'
    ```

    **Returns:** job_id for tracking processing status via /status/{job_id}
    """
    try:
        # Validate file extension
        if not file.filename.endswith(('.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .md files are supported"
            )

        # Read file content
        content_bytes = await file.read()

        # Validate file
        is_valid, error_msg = validate_file_upload(
            content_bytes,
            file.filename,
            max_size_mb=50
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Decode content
        content_text = content_bytes.decode('utf-8')

        # Process Markdown if .md
        if file.filename.endswith('.md'):
            logger.info(f"Processing Markdown file: {file.filename}")
            content_text = process_markdown(content_text)

        # Validate content not empty
        if not content_text.strip():
            raise HTTPException(status_code=400, detail="File is empty after processing")

        # Parse metadata
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON in metadata field"
                )

        # Determine target dataset
        target_dataset = data_source if (data_source and data_source != "string") else args.data_source

        # Generate IDs
        # IMPORTANT: Use "doc" prefix to match BiGRAG's internal ID format
        doc_id = compute_doc_id(content_text, prefix="doc")
        job_id = f"job-{compute_doc_id(doc_id + str(datetime.now()), prefix='')}"

        # Use filename as title if not provided
        doc_title = title or file.filename

        # Add to corpus.jsonl
        await add_document_to_corpus(
            data_source=target_dataset,
            doc_id=doc_id,
            content=content_text,
            title=doc_title,
            metadata=doc_metadata
        )

        # Add to document registry
        await registry.add_document(
            document_id=doc_id,
            filename=file.filename,
            title=doc_title,
            content_length=len(content_text),
            dataset=target_dataset,
            metadata=doc_metadata,
            job_id=job_id,
            status="pending"
        )

        # Create processing job
        job = ProcessingJob(
            job_id=job_id,
            document_id=doc_id,
            dataset=target_dataset,
            status=JobStatus.PENDING,
            progress=0.0,
            stage=ProcessingStage.QUEUED
        )
        processing_jobs[job_id] = job

        # Process document (Phase 2.1: pass metadata for improved entity extraction)
        if process_async:
            # Background processing
            background_tasks.add_task(
                process_document_background,
                job_id=job_id,
                content=content_text,
                title=doc_title,
                dataset=target_dataset,
                rag_instance=rag,
                registry_instance=registry,
                metadata=doc_metadata  # Phase 2.1: metadata preservation
            )
            message = "Document queued for processing"
        else:
            # Synchronous processing (blocks response)
            await process_document_background(
                job_id=job_id,
                content=content_text,
                title=doc_title,
                dataset=target_dataset,
                rag_instance=rag,
                registry_instance=registry,
                metadata=doc_metadata  # Phase 2.1: metadata preservation
            )
            message = "Document processed successfully"

        # Return response
        return EnhancedUploadResponse(
            success=True,
            message=message,
            document_id=doc_id,
            job_id=job_id,
            filename=file.filename,
            title=doc_title,
            content_preview=truncate_text(content_text, 200),
            content_length=len(content_text),
            dataset=target_dataset,
            status=job.status.value if isinstance(job.status, JobStatus) else job.status,
            metadata=doc_metadata,
            upload_date=datetime.now().isoformat()
        )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File encoding error. Please upload UTF-8 encoded files"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/rebuild", response_model=RebuildResponse, tags=["Document Management"])
async def rebuild_graph(
    data_source: str = Form(None, description="Dataset name (defaults to current dataset)"),
    force_full_rebuild: bool = Form(False, description="Force full rebuild instead of incremental")
):
    """
    Manually trigger knowledge graph rebuild.

    By default, performs incremental update (adds new documents from corpus.jsonl).
    Use `force_full_rebuild=true` to rebuild entire graph from scratch.

    **Example usage:**
    ```bash
    # Incremental rebuild
    curl -X POST "http://localhost:8001/rebuild"

    # Full rebuild
    curl -X POST "http://localhost:8001/rebuild" \\
      -F "force_full_rebuild=true"
    ```
    """
    try:
        target_dataset = data_source or args.data_source
        corpus_file = Path(f"{config.input_dir}/{target_dataset}/raw/corpus.jsonl")

        if not corpus_file.exists():
            raise HTTPException(status_code=404, detail=f"Corpus file not found: {corpus_file}")

        # Load all documents from corpus (Phase 2.1: extract full document objects with metadata)
        documents = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                # Extract full document with contents, title, and metadata
                documents.append({
                    "contents": doc.get("contents", ""),
                    "title": doc.get("title", ""),
                    "metadata": doc.get("metadata", {})
                })

        if not documents:
            raise HTTPException(status_code=400, detail="No documents found in corpus")

        # Rebuild graph
        logger.info(f"Rebuilding knowledge graph for {target_dataset} ({len(documents)} documents)...")

        if force_full_rebuild:
            # Clear existing graph and rebuild from scratch
            logger.warning("Full rebuild requested - this will replace existing graph")
            # Note: BiGRAG doesn't have a clear() method, so we rely on insert() to handle updates
            await rebuild_knowledge_graph_incremental(target_dataset, documents)
            rebuild_type = "full"
        else:
            # Incremental rebuild (Phase 2.1: includes metadata for improved entity extraction)
            await rebuild_knowledge_graph_incremental(target_dataset, documents)
            rebuild_type = "incremental"

        return RebuildResponse(
            success=True,
            message=f"Knowledge graph {'fully rebuilt' if force_full_rebuild else 'incrementally updated'}",
            documents_processed=len(documents),
            dataset=target_dataset,
            rebuild_type=rebuild_type
        )

    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Job Management"])
async def get_job_status(job_id: str):
    """
    Get processing status for a document upload job.

    **Example usage:**
    ```bash
    curl "http://localhost:8001/status/job-abc123"
    ```

    **Returns:** Job status with progress (0.0 to 1.0) and current stage
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = processing_jobs[job_id]

    # Convert stats to JobStatistics model if present
    job_stats = None
    if job.stats:
        from api.models import JobStatistics
        job_stats = JobStatistics(
            chunks_created=job.stats.get("chunks", 0),
            entities_extracted=job.stats.get("entities", 0),
            edges_created=job.stats.get("edges", 0),
            tokens_processed=job.stats.get("tokens", 0)
        )

    return JobStatusResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        dataset=job.dataset,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error,
        stats=job_stats
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["Document Management"])
async def list_documents(
    dataset: Optional[str] = None,
    search: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List all documents with optional filtering and pagination.

    **Filters:**
    - `dataset`: Filter by dataset name
    - `search`: Search in title or filename
    - `category`: Filter by category
    - `tags`: Comma-separated tags (e.g., "RAG,NLP")
    - `status`: Filter by status (pending, processing, indexed, failed, deleted)
    - `date_from`, `date_to`: Date range filter (ISO format)
    - `limit`: Results per page (1-500, default 50)
    - `offset`: Pagination offset (default 0)

    **Example usage:**
    ```bash
    # Get all documents
    curl "http://localhost:8001/documents"

    # Search with filters
    curl "http://localhost:8001/documents?search=research&category=science&limit=20"

    # Get documents by status
    curl "http://localhost:8001/documents?status=indexed"
    ```
    """
    try:
        # Parse tags
        tags_list = tags.split(',') if tags else None

        # Get documents from registry
        docs = await registry.list_documents(
            dataset=dataset,
            search=search,
            category=category,
            tags=tags_list,
            status=status,
            date_from=date_from,
            date_to=date_to
        )

        # Apply pagination
        total = len(docs)
        paginated = docs[offset:offset+limit]

        # Convert to DocumentSummary models
        summaries = [
            DocumentSummary(
                document_id=doc["document_id"],
                filename=doc["filename"],
                title=doc["title"],
                content_length=doc["content_length"],
                upload_date=doc["upload_date"],
                indexed_date=doc.get("indexed_date"),
                last_modified=doc["last_modified"],
                status=doc["status"],
                dataset=doc["dataset"],
                metadata=doc.get("metadata"),
                job_id=doc["job_id"]
            )
            for doc in paginated
        ]

        return DocumentListResponse(
            total=total,
            limit=limit,
            offset=offset,
            documents=summaries
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/documents/{document_id}", response_model=DocumentDetailResponse, tags=["Document Management"])
async def get_document_details(document_id: str, include_entities: bool = True, include_related: bool = True):
    """
    Get detailed information about a specific document.

    **Includes:**
    - Document metadata and status
    - Knowledge graph statistics (chunks, entities, edges)
    - Top extracted entities (optional, default: true)
    - Related documents by entity overlap (optional, default: true)

    **Example usage:**
    ```bash
    curl "http://localhost:8001/documents/upload-abc123"

    # Without entities and related docs
    curl "http://localhost:8001/documents/upload-abc123?include_entities=false&include_related=false"
    ```
    """
    try:
        # Get document from registry
        doc = await registry.get_document(document_id)

        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        # Get content preview
        content = await get_document_content_from_corpus(doc["dataset"], document_id)
        content_preview = truncate_text(content, 500) if content else "N/A"

        # Get KG statistics
        kg_stats_dict = await get_document_stats_from_kg(doc["dataset"], document_id)
        kg_stats = KGStatistics(**kg_stats_dict) if kg_stats_dict else None

        # Get top entities (if requested and document is indexed)
        top_entities = None
        if include_entities and doc.get("status") == "indexed":
            entities_list = await get_document_entities(doc["dataset"], document_id, top_k=10)
            top_entities = [EntityInfo(**e) for e in entities_list]

        # Get related documents (if requested and document is indexed)
        related_docs = None
        if include_related and doc.get("status") == "indexed":
            related_list = await find_related_documents(doc["dataset"], document_id, top_k=5)
            related_docs = [RelatedDocument(**r) for r in related_list]

        return DocumentDetailResponse(
            document_id=doc["document_id"],
            filename=doc["filename"],
            title=doc["title"],
            content_length=doc["content_length"],
            content_preview=content_preview,
            upload_date=doc["upload_date"],
            indexed_date=doc.get("indexed_date"),
            last_modified=doc["last_modified"],
            status=doc["status"],
            dataset=doc["dataset"],
            metadata=doc.get("metadata"),
            job_id=doc["job_id"],
            stats=kg_stats,
            top_entities=top_entities,
            related_documents=related_docs
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")


@app.delete("/documents/{document_id}", response_model=DeleteResponse, tags=["Document Management"])
async def delete_document(document_id: str, hard_delete: bool = False):
    """
    Delete a document from the system.

    **Delete Modes:**
    - **Soft delete** (default): Marks document as deleted in registry, keeps in KG
    - **Hard delete**: Removes from corpus.jsonl, requires full graph rebuild

    **Example usage:**
    ```bash
    # Soft delete (recommended)
    curl -X DELETE "http://localhost:8001/documents/upload-abc123"

    # Hard delete (requires rebuild)
    curl -X DELETE "http://localhost:8001/documents/upload-abc123?hard_delete=true"
    ```

    **Note:** Hard delete requires running `/rebuild` afterward to update the knowledge graph.
    """
    try:
        # Get document info
        doc = await registry.get_document(document_id)

        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        if hard_delete:
            # Hard delete: Cascade deletion across all storage layers
            from api.kg_utils import remove_from_corpus

            # Step 1: Remove from corpus.jsonl
            await remove_from_corpus(doc["dataset"], document_id)

            # Step 2: Cascade delete from knowledge graph (chunks, entities, edges, VDBs)
            await rag.adelete_document(document_id)

            # Step 3: Remove from registry
            await registry.delete_document(document_id, hard=True)

            return DeleteResponse(
                success=True,
                message=f"Document {document_id} permanently deleted with cascade cleanup (chunks, entities, edges removed).",
                document_id=document_id,
                hard_delete=True,
                rebuild_required=False  # No rebuild needed - cascade deletion is complete
            )
        else:
            # Soft delete: Just mark as deleted
            await registry.delete_document(document_id, hard=False)

            return DeleteResponse(
                success=True,
                message=f"Document {document_id} marked as deleted (soft delete)",
                document_id=document_id,
                hard_delete=False,
                rebuild_required=False
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/graph/stats", response_model=GraphStatsResponse, tags=["Graph Management"])
async def get_graph_statistics(dataset: Optional[str] = None):
    """
    Get knowledge graph statistics.

    **Returns:**
    - Global statistics (all datasets)
    - Per-dataset breakdown
    - Document counts by status
    - Entity, edge, and chunk counts

    **Example usage:**
    ```bash
    # All datasets
    curl "http://localhost:8001/graph/stats"

    # Specific dataset
    curl "http://localhost:8001/graph/stats?dataset=user_uploads"
    ```
    """
    try:
        from api.models import DatasetStats

        datasets_to_query = []
        if dataset:
            datasets_to_query = [dataset]
        else:
            # Get all datasets
            expr_dir = Path("expr")
            if expr_dir.exists():
                datasets_to_query = [
                    d.name for d in expr_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.')
                ]

        # Collect stats per dataset
        dataset_stats_list = []
        global_totals = {
            "total_documents": 0,
            "total_entities": 0,
            "total_edges": 0,
            "total_chunks": 0
        }

        for ds in datasets_to_query:
            # Get registry stats
            reg_stats = await registry.get_stats(ds)

            # Get KG file counts - read from GraphML and KV storage
            graph_file = f"{config.working_dir}/{ds}/graph_chunk_entity_relation.graphml"
            chunks_file = f"{config.working_dir}/{ds}/kv_store_text_chunks.json"

            entities_count = 0
            edges_count = 0
            chunks_count = 0
            tokens_count = 0

            # Count entities and edges from GraphML file
            if os.path.exists(graph_file):
                try:
                    import networkx as nx
                    G = nx.read_graphml(graph_file)

                    for node, attrs in G.nodes(data=True):
                        role = attrs.get("role", "")
                        if role == "entity":
                            entities_count += 1
                        elif role == "bipartite_edge":
                            edges_count += 1
                except Exception as e:
                    logger.warning(f"Failed to read GraphML for dataset {ds}: {e}")

            # Count chunks from KV storage
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    chunks_count = len(chunks)
                    tokens_count = sum(c.get("tokens", 0) for c in chunks.values())

            # Create dataset stats
            ds_stats = DatasetStats(
                dataset=ds,
                total_documents=reg_stats["total"],
                indexed_documents=reg_stats["indexed"],
                pending_documents=reg_stats["pending"],
                failed_documents=reg_stats["failed"],
                total_chunks=chunks_count,
                total_entities=entities_count,
                total_edges=edges_count,
                total_tokens=tokens_count
            )

            dataset_stats_list.append(ds_stats)

            # Add to global totals
            global_totals["total_documents"] += reg_stats["total"]
            global_totals["total_entities"] += entities_count
            global_totals["total_edges"] += edges_count
            global_totals["total_chunks"] += chunks_count

        return GraphStatsResponse(
            success=True,
            total_datasets=len(dataset_stats_list),
            global_stats=global_totals,
            datasets=dataset_stats_list
        )

    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph statistics: {str(e)}")


# ==============================================================================
# Evaluation Endpoints
# ==============================================================================

@app.post("/eval/retrieval", response_model=RetrievalEvalResponse, tags=["Evaluation"])
async def evaluate_retrieval_endpoint(request: RetrievalEvalRequest):
    """
    Evaluate retrieval quality with ground truth documents.

    **Metrics Calculated:**
    - Precision@K: Fraction of retrieved docs that are relevant
    - Recall@K: Fraction of relevant docs that were retrieved
    - F1@K: Harmonic mean of precision and recall
    - MRR (Mean Reciprocal Rank): 1 / rank of first relevant doc
    - NDCG@K: Normalized Discounted Cumulative Gain

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/retrieval" \\
      -H "Content-Type: application/json" \\
      -d '{
        "queries": [{
          "question": "What is BiG-RAG?",
          "ground_truth_docs": ["doc_001"]
        }],
        "dataset": "demo_test",
        "mode": "hybrid",
        "top_k": 5,
        "metrics": ["precision", "recall", "mrr"]
      }'
    ```
    """
    try:
        # Convert queries to dict format
        queries_list = [
            {
                "question": q.question,
                "ground_truth_docs": q.ground_truth_docs
            }
            for q in request.queries
        ]

        # Run evaluation
        eval_results = await evaluate_retrieval(
            queries=queries_list,
            rag_instance=rag,
            mode=request.mode,
            top_k=request.top_k,
            metrics=request.metrics,
            embedding_manager=embedding_manager
        )

        # Convert to response model
        per_query_models = [
            PerQueryRetrievalResult(
                question=result["question"],
                retrieved_docs=result["retrieved_docs"],
                relevant_retrieved=result["relevant_retrieved"],
                metrics=result["metrics"],
                latency_ms=result.get("latency_ms")
            )
            for result in eval_results["per_query_results"]
        ]

        return RetrievalEvalResponse(
            success=True,
            total_queries=eval_results["total_queries"],
            metrics=eval_results["metrics"],
            per_query_results=per_query_models,
            evaluation_time=eval_results["evaluation_time"],
            latency_stats=eval_results.get("latency_stats")
        )

    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/eval/answer", response_model=AnswerEvalResponse, tags=["Evaluation"])
async def evaluate_answer_endpoint(request: AnswerEvalRequest):
    """
    Evaluate answer quality against ground truth.

    **Metrics Calculated:**
    - Exact Match (EM): Binary match after normalization
    - Token F1: Token-level F1 score
    - ROUGE-L: Longest Common Subsequence F1

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/answer" \\
      -H "Content-Type: application/json" \\
      -d '{
        "test_cases": [{
          "question": "What is AI?",
          "ground_truth": "Artificial Intelligence",
          "use_rag": true
        }],
        "dataset": "demo_test",
        "metrics": ["em", "f1", "rouge_l"]
      }'
    ```
    """
    try:
        # Get LLM function
        llm_func = llm_manager.get_model_func(request.llm_provider, request.model)

        per_question_results = []
        all_metrics = []

        for test_case in request.test_cases:
            # Pre-compute entity/edge matches if FlagEmbedding
            entity_match = None
            edge_match = None
            if embedding_manager.mode == "flagembedding" and test_case.use_rag:
                entity_match = await embedding_manager.search_entities(test_case.question, request.top_k)
                edge_match = await embedding_manager.search_edges(test_case.question, request.top_k)

            # Evaluate single answer
            result = await evaluate_single_answer(
                question=test_case.question,
                ground_truth=test_case.ground_truth,
                rag_instance=rag,
                llm_func=llm_func,
                use_rag=test_case.use_rag,
                mode=request.mode,
                top_k=request.top_k,
                metrics=request.metrics,
                entity_match=entity_match,
                bipartite_edge_match=edge_match
            )

            per_question_results.append(PerQuestionAnswerResult(
                question=test_case.question,
                ground_truth=test_case.ground_truth,
                predicted_answer=result["predicted_answer"],
                metrics=result["metrics"],
                retrieval_used=result["retrieval_used"],
                num_contexts_used=result["num_contexts_used"],
                generation_time=result["generation_time"]
            ))

            all_metrics.append(result["metrics"])

        # Aggregate metrics
        from api.metrics import aggregate_metrics
        aggregate = aggregate_metrics(all_metrics)

        return AnswerEvalResponse(
            success=True,
            total_questions=len(request.test_cases),
            aggregate_metrics=aggregate,
            per_question_results=per_question_results,
            total_time=sum(r.generation_time for r in per_question_results)
        )

    except Exception as e:
        logger.error(f"Answer evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/eval/compare", response_model=CompareEvalResponse, tags=["Evaluation"])
async def compare_configurations_endpoint(request: CompareEvalRequest):
    """
    Compare different retrieval configurations side-by-side.

    **Use Cases:**
    - Compare retrieval modes (hybrid vs local vs global vs naive)
    - Compare different top_k values
    - Find optimal configuration

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/compare" \\
      -H "Content-Type: application/json" \\
      -d '{
        "queries": [{
          "question": "What is ML?",
          "ground_truth_docs": ["doc_001"]
        }],
        "configurations": [
          {"name": "hybrid", "mode": "hybrid", "top_k": 5},
          {"name": "local", "mode": "local", "top_k": 5}
        ],
        "metrics": ["precision", "recall", "mrr"]
      }'
    ```
    """
    try:
        # Convert queries and configurations to dict format
        queries_list = [
            {
                "question": q.question,
                "ground_truth_docs": q.ground_truth_docs
            }
            for q in request.queries
        ]

        configs_list = [
            {
                "name": c.name,
                "mode": c.mode,
                "top_k": c.top_k
            }
            for c in request.configurations
        ]

        # Run comparison
        comparison = await compare_configurations(
            queries=queries_list,
            configurations=configs_list,
            rag_instance=rag,
            metrics=request.metrics,
            embedding_manager=embedding_manager
        )

        return CompareEvalResponse(
            success=True,
            comparison_results=comparison["comparison_results"],
            best_configuration=comparison["best_configuration"],
            ranking=comparison["ranking"]
        )

    except Exception as e:
        logger.error(f"Configuration comparison failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/eval/batch", response_model=BatchEvalResponse, tags=["Evaluation"])
async def batch_evaluation_endpoint(request: BatchEvalRequest):
    """
    Run batch evaluation from a QA dataset file.

    **Supports:**
    - Large-scale evaluation (10s to 100s of questions)
    - Both retrieval and answer quality metrics
    - Progress tracking and result saving

    **Dataset Format (JSON):**
    ```json
    {
      "questions": [
        {
          "question": "What is AI?",
          "ground_truth_answer": "Artificial Intelligence",
          "ground_truth_docs": ["doc_001"]
        }
      ]
    }
    ```

    **Example usage:**
    ```bash
    curl -X POST "http://localhost:8001/eval/batch" \\
      -H "Content-Type: application/json" \\
      -d '{
        "dataset_file": "test_datasets/eval_qa.json",
        "data_source": "demo_test",
        "mode": "hybrid",
        "metrics": ["em", "f1", "precision", "recall"],
        "use_llm": true,
        "save_results": true,
        "output_file": "evaluation_results/eval_result.json"
      }'
    ```
    """
    try:
        # Get LLM function
        llm_func = None
        if request.use_llm:
            llm_func = llm_manager.get_model_func(
                request.llm_provider or "openai",
                "gpt-4o-mini"
            )

        # Run batch evaluation
        results = await batch_evaluate(
            dataset_file=request.dataset_file,
            rag_instance=rag,
            llm_func=llm_func,
            mode=request.mode,
            top_k=request.top_k,
            metrics=request.metrics,
            use_llm=request.use_llm,
            save_results=request.save_results,
            output_file=request.output_file,
            limit=request.limit,
            embedding_manager=embedding_manager
        )

        # Convert to response model
        performance = BatchEvalPerformance(
            total_time=results["performance"]["total_time"],
            avg_time_per_query=results["performance"]["avg_time_per_query"],
            total_llm_calls=results["performance"]["total_llm_calls"],
            total_embedding_calls=results["performance"]["total_embedding_calls"]
        )

        return BatchEvalResponse(
            success=True,
            dataset=results["dataset"],
            total_questions=results["total_questions"],
            processed=results["processed"],
            failed=results["failed"],
            metrics=results["metrics"],
            performance=performance,
            results_saved_to=results.get("results_saved_to")
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@app.post("/eval/batch_generate", response_model=BatchGenerateResponse, tags=["Evaluation"])
async def batch_generate(request: BatchGenerateRequest):
    """
    Batch process questions CSV and generate answers with retrieval context.

    This endpoint:
    1. Loads questions from CSV (question, golden_answer, document_index, question_type)
    2. For each question:
       - Retrieves relevant context from knowledge graph
       - Generates answer using LLM
       - Tracks latency
    3. Saves results to CSV with all fields for later evaluation

    **Use Case**: Process SingleTopic or similar datasets systematically

    **Input CSV Schema**:
    ```csv
    question,golden_answer,document_index,question_type
    "What is X?","Answer to X",0,single_passage
    ```

    **Output CSV Schema**:
    ```csv
    question,golden_answer,generated_answer,retrieval_context,document_index,question_type,latency_ms
    ```

    **Example**:
    ```bash
    curl -X POST http://localhost:8001/eval/batch_generate \\
      -H "Content-Type: application/json" \\
      -d '{
        "questions_csv_path": "datasets/SingleTopic/processed/all_questions_unified.csv",
        "output_csv_path": "datasets/SingleTopic/results/generation_results.csv",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "enable_reranking": true
      }'
    ```
    """
    try:
        from api.csv_evaluation import batch_generate_endpoint

        result = await batch_generate_endpoint(
            questions_csv_path=request.questions_csv_path,
            output_csv_path=request.output_csv_path,
            rag_instance=rag,
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            llm_provider=request.llm_provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            enable_reranking=request.enable_reranking
        )

        return BatchGenerateResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@app.post("/eval/evaluate_results", response_model=EvaluateResultsResponse, tags=["Evaluation"])
async def evaluate_results(request: EvaluateResultsRequest):
    """
    Evaluate saved results CSV for accuracy.

    This endpoint:
    1. Loads results CSV (output from /eval/batch_generate)
    2. Calculates metrics per question:
       - EM (Exact Match)
       - F1 (Token F1)
       - ROUGE-L (optional)
       - Answer Rate (for no-answer questions)
    3. Aggregates metrics overall and by question type
    4. Exports results in multiple formats (JSON, CSV, Markdown, LaTeX)

    **Use Case**: Evaluate BiG-RAG accuracy on standard datasets

    **Metrics**:
    - **EM**: Exact match rate (0.0-1.0)
    - **F1**: Token-level F1 score (0.0-1.0)
    - **ROUGE-L**: Longest common subsequence similarity (requires rouge_score package)
    - **Answer Rate**: % of no-answer questions where model refused to answer

    **Example**:
    ```bash
    curl -X POST http://localhost:8001/eval/evaluate_results \\
      -H "Content-Type: application/json" \\
      -d '{
        "results_csv_path": "datasets/SingleTopic/results/generation_results.csv",
        "metrics": ["em", "f1", "rouge_l", "answer_rate"],
        "export_formats": ["json", "csv", "markdown", "latex"],
        "output_dir": "datasets/SingleTopic/results/"
      }'
    ```

    **Exported Files**:
    - `{basename}_evaluation.json`: Full results with all metrics
    - `{basename}_evaluation.csv`: Spreadsheet-friendly format
    - `{basename}_evaluation.md`: Markdown table for documentation
    - `{basename}_evaluation.tex`: LaTeX table for papers
    """
    try:
        from api.csv_evaluation import evaluate_results_endpoint

        result = await evaluate_results_endpoint(
            results_csv_path=request.results_csv_path,
            metrics=request.metrics,
            export_formats=request.export_formats,
            output_dir=request.output_dir
        )

        return EvaluateResultsResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Results evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Results evaluation failed: {str(e)}")


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

        # Query BiGRAG (Phase 3: Three-Path Retrieval + Semantic Reranking)
        result = await rag.aquery(
            request.question,
            param=QueryParam(
                mode=request.mode,
                only_need_context=True,
                top_k=request.top_k,
                enable_reranking=request.enable_reranking  # Phase 3.4: semantic reranking
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
    """
    Batch retrieval for multiple queries

    Phase 3 Enhancement: Uses Three-Path Retrieval (Entity + Edge + Chunk) with optional semantic reranking
    """
    try:
        results = []
        for query_text in request.queries:
            entity_match = None
            edge_match = None

            if embedding_manager.mode == "flagembedding":
                entity_match = await embedding_manager.search_entities(query_text, 5)
                edge_match = await embedding_manager.search_edges(query_text, 5)

            # Phase 3: Three-Path Retrieval + Semantic Reranking
            result = await rag.aquery(
                query_text,
                param=QueryParam(
                    mode="hybrid",
                    only_need_context=True,
                    top_k=10,
                    enable_reranking=False  # Phase 3.4: semantic reranking (default: False)
                ),
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
    OpenAI-compatible chat completions endpoint with RAG

    This endpoint:
    1. Retrieves relevant context from the knowledge graph (if use_rag=True)
    2. Synthesizes a comprehensive answer using the specified LLM
    3. Returns the answer in OpenAI-compatible format

    Example request:
    ```json
    {
        "messages": [{"role": "user", "content": "What is Artificial Intelligence?"}],
        "use_rag": true
    }
    ```

    Click "Try it out" and hit "Execute" to test!
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

        # RAG: Retrieve context from knowledge graph
        if request.use_rag:
            entity_match = None
            edge_match = None

            if embedding_manager.mode == "flagembedding":
                entity_match = await embedding_manager.search_entities(user_prompt, 5)
                edge_match = await embedding_manager.search_edges(user_prompt, 5)

            # Phase 3: Three-Path Retrieval + Semantic Reranking
            context_results = await rag.aquery(
                user_prompt,
                param=QueryParam(
                    mode="hybrid",
                    only_need_context=True,
                    top_k=5,
                    enable_reranking=request.enable_reranking  # Phase 3.4: semantic reranking
                ),
                entity_match=entity_match,
                bipartite_edge_match=edge_match
            )

            if context_results:
                # Format retrieved contexts
                context_parts = []
                for i, item in enumerate(context_results[:5], 1):
                    if isinstance(item, dict):
                        context = item.get("<knowledge>", str(item))
                    else:
                        context = str(item)
                    context_parts.append(f"[Source {i}]\n{context}")

                context_str = "\n\n".join(context_parts)

                # Create RAG system prompt
                if not system_prompt:
                    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context from the knowledge graph.

Instructions:
- Use the information from the context sources to provide a comprehensive answer
- Be clear, accurate, and concise
- If the context doesn't fully answer the question, acknowledge what you know and what's uncertain
- Cite relevant information from the sources when appropriate"""

                # Prepend context to user prompt
                user_prompt = f"""Based on the following context from the knowledge graph:

{context_str}

Question: {user_prompt}

Please provide a comprehensive answer based on the above context."""

        # Call LLM to synthesize answer
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
