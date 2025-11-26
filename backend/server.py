"""
BiG-RAG Unified API Server

A robust, production-ready API server with modular route organization:
- Auto-detection of knowledge graph format (OpenAI vs FlagEmbedding)
- Multiple LLM provider support (OpenAI, Claude, Gemini, Grok)
- Graceful fallback to gpt-4o-mini
- OpenAI-compatible endpoints
- Health monitoring and statistics

Usage:
    python server.py --data_source demo_test
    python server.py --data_source demo_test --llm_provider anthropic
    python server.py --data_source demo_test --port 8002

Environment Variables:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add parent directory to path for bigrag imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from root .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import BiG-RAG core
from bigrag import BiGRAG
from bigrag.config import config
from bigrag.llm import gpt_4o_mini_complete, openai_embedding
from bigrag.logging_config import setup_logger

# Import core modules (managers and dependencies)
from api.core.managers import LLMProviderManager, EmbeddingManager
from api.core import dependencies

# Import route modules
from api.routes import health, documents, graph, evaluation, retrieval, jobs, llm, unified, datasets
from api import hitl_routes  # NEW (Phase 1 Step 6): HITL system routes
from api import agent


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
parser.add_argument('--unified', action='store_true',
                    help='Enable unified multi-subgraph mode')
parser.add_argument('--registry_path', default='expr/subgraph_registry.json',
                    help='Path to subgraph registry (unified mode only)')
parser.add_argument('--max_cached', type=int, default=10,
                    help='Max cached subgraphs in unified mode (default: 10, LRU eviction)')
parser.add_argument('--prewarm', nargs='*',
                    help='Subgraphs to preload at startup (unified mode)')
args = parser.parse_args()

# Setup API logger (separate from BiGRAG core logger)
api_logger = setup_logger(
    name="bigrag.api",
    log_dir=str(PROJECT_ROOT / "logs" / "backend"),
    log_file="api.log",
    level=os.getenv('LOG_LEVEL', 'INFO'),
    json_format=os.getenv('LOG_JSON_FORMAT', 'false').lower() == 'true',
    rotation='time',  # Daily rotation
    backup_count=7,
    console_output=True,
    error_separate=True
)

# Determine server mode
working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
unified_mode = args.unified

api_logger.info("="*60)
api_logger.info(f"Initializing BiG-RAG API Server")
api_logger.info(f"Mode: {'UNIFIED (multi-subgraph)' if unified_mode else 'SINGLE (single dataset)'}")

if unified_mode:
    api_logger.info(f"Registry: {args.registry_path}")
    api_logger.info(f"Max cached subgraphs: {args.max_cached}")
    if args.prewarm:
        api_logger.info(f"Prewarm subgraphs: {args.prewarm}")
else:
    working_dir = str(PROJECT_ROOT / working_dir_base / args.data_source)
    api_logger.info(f"Dataset: {args.data_source}")
    api_logger.info(f"Working directory: {working_dir}")

api_logger.info("="*60)

llm_manager = LLMProviderManager(default_provider=args.llm_provider)
server_start_time = time.time()

# Initialize based on mode
if unified_mode:
    # ========== UNIFIED MODE ==========
    from bigrag.unified import UnifiedQueryExecutor

    # Get LLM function for routing (use gpt_4o_mini_complete for routing)
    llm_func = gpt_4o_mini_complete

    # Initialize unified executor
    unified_executor = UnifiedQueryExecutor(
        registry_path=str(PROJECT_ROOT / args.registry_path),
        llm_func=llm_func,
        max_cached_subgraphs=args.max_cached,
        prewarm_subgraphs=args.prewarm,
        enable_parallel=True,
        bigrag_kwargs={
            "embedding_func": openai_embedding,  # FIX: Pass embedding function to ensure VDB loads correctly
            "llm_model_func": gpt_4o_mini_complete,
            "chunk_token_size": config.chunk_size,
            "chunk_overlap_token_size": config.chunk_overlap_size,
            "enable_llm_cache": config.enable_llm_cache,
            "addon_params": {"language": config.default_language}
        }
    )

    # Set unified executor for dependency injection
    dependencies.set_unified_executor(unified_executor)

    # Set dummy instances for single-mode routes (not used in unified mode)
    rag = None
    embedding_manager = None
    working_dir = None

    api_logger.info("Unified query executor initialized")
    api_logger.info(f"Available subgraphs: {unified_executor.get_available_subgraphs()}")

else:
    # ========== SINGLE MODE ==========
    embedding_manager = EmbeddingManager(working_dir)

    # Initialize BiGRAG
    rag = BiGRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=embedding_manager.get_embedding_func(),
        chunk_token_size=config.chunk_size,
        chunk_overlap_token_size=config.chunk_overlap_size,
        enable_llm_cache=config.enable_llm_cache,
        addon_params={"language": config.default_language},
    )

    # Set global instances for dependency injection
    dependencies.set_rag_instance(rag)
    dependencies.set_embedding_manager(embedding_manager)
    dependencies.set_server_metadata(server_start_time, args.data_source, working_dir)

# Set LLM manager for both modes
dependencies.set_llm_manager(llm_manager)

api_logger.info(f"Language configuration: {config.default_language}")
api_logger.info(f"Available LLM providers: {', '.join(llm_manager.get_available_providers())}")
api_logger.info(f"Default LLM provider: {args.llm_provider}")

# Load statistics (single mode only)
if not unified_mode:
    api_logger.info(f"Embedding mode: {embedding_manager.mode}")

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

            edges_file = f"{working_dir}/vdb_relations.json"
            if os.path.exists(edges_file):
                with open(edges_file, 'r', encoding='utf-8') as f:
                    vdb_relations = json.load(f)
                stats["edges"] = len(vdb_relations.get('data', []))

        elif embedding_manager.mode == "flagembedding":
            stats["entities"] = len(embedding_manager.corpus_entity)
            stats["edges"] = len(embedding_manager.corpus_relation)

        api_logger.info(f"Graph statistics:")
        api_logger.info(f"  - Entities: {stats['entities']}")
        api_logger.info(f"  - Relations: {stats['edges']}")
        api_logger.info(f"  - Text Chunks: {stats['chunks']}")

    except Exception as e:
        api_logger.warning(f"Could not load statistics: {e}")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="BiG-RAG Unified API",
    description="Multi-provider RAG API with modular route organization",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Mount Routers
# ============================================================================

# Health and system routes
app.include_router(health.router)

# Document management routes
app.include_router(documents.router)

# Graph management routes
app.include_router(graph.router)

# Evaluation routes
app.include_router(evaluation.router)

# Retrieval routes (Q&A and search)
app.include_router(retrieval.router)

# Job management routes
app.include_router(jobs.router)

# LLM chat completion routes
app.include_router(llm.router)

# Agent routes (multi-hop reasoning)
app.include_router(agent.router)

# HITL routes (Phase 1 Step 6): Human-in-the-Loop system
app.include_router(hitl_routes.router)

# Unified subgraph routes (only if in unified mode)
if unified_mode:
    app.include_router(unified.router)
    # Production dataset management (only in unified mode)
    app.include_router(datasets.router)


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    # Initialize agent (single mode only)
    if not unified_mode and rag:
        agent_model = os.getenv("AGENT_DEFAULT_MODEL", "gpt-4o")
        agent.initialize_agent(rag, model=agent_model)

    # Prewarm cache in unified mode
    if unified_mode:
        unified_exec = dependencies.get_unified_executor()
        if unified_exec:
            # Auto-prewarm: Load top N subgraphs by priority
            api_logger.info("[Startup] Auto-prewarming cache...")
            await unified_exec.auto_prewarm()

            # Manual prewarm: Override with explicit list if provided
            if unified_exec.cache.prewarm_list:
                api_logger.info(f"[Startup] Manual prewarm with: {unified_exec.cache.prewarm_list}")
                await unified_exec.cache.preload(unified_exec.cache.prewarm_list)

            api_logger.info("[Startup] Cache prewarming completed")

    api_logger.info("=" * 60)
    api_logger.info("BiG-RAG API Server started")
    api_logger.info(f"Documentation: http://{args.host}:{args.port}/docs")

    if unified_mode:
        api_logger.info(f"Unified query endpoint: http://{args.host}:{args.port}/api/unified/query")
        api_logger.info(f"Subgraphs endpoint: http://{args.host}:{args.port}/api/unified/subgraphs")
    else:
        api_logger.info(f"Agent endpoint: http://{args.host}:{args.port}/agent/query")

    api_logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    api_logger.info("BiG-RAG API Server shutting down")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
