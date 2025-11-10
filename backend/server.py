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
from bigrag.llm import gpt_4o_mini_complete
from bigrag.logging_config import setup_logger

# Import core modules (managers and dependencies)
from api.core.managers import LLMProviderManager, EmbeddingManager
from api.core import dependencies

# Import route modules
from api.routes import health, documents, graph, evaluation, retrieval, jobs, llm


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

# Initialize managers
working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
working_dir = str(PROJECT_ROOT / working_dir_base / args.data_source)

api_logger.info("="*60)
api_logger.info(f"Initializing BiG-RAG API Server")
api_logger.info(f"Dataset: {args.data_source}")
api_logger.info(f"Working directory: {working_dir}")
api_logger.info("="*60)

embedding_manager = EmbeddingManager(working_dir)
llm_manager = LLMProviderManager(default_provider=args.llm_provider)
server_start_time = time.time()

# Initialize BiGRAG
rag = BiGRAG(
    working_dir=working_dir,
    llm_model_func=gpt_4o_mini_complete,  # Fallback for entity extraction
    embedding_func=embedding_manager.get_embedding_func(),
    chunk_token_size=config.chunk_size,
    chunk_overlap_token_size=config.chunk_overlap_size,
    enable_llm_cache=config.enable_llm_cache,
)

# Set global instances for dependency injection
dependencies.set_rag_instance(rag)
dependencies.set_llm_manager(llm_manager)
dependencies.set_embedding_manager(embedding_manager)
dependencies.set_server_metadata(server_start_time, args.data_source, working_dir)

api_logger.info(f"BiG-RAG core initialized")
api_logger.info(f"Embedding mode: {embedding_manager.mode}")
api_logger.info(f"Available LLM providers: {', '.join(llm_manager.get_available_providers())}")
api_logger.info(f"Default LLM provider: {args.llm_provider}")

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


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    api_logger.info("=" * 60)
    api_logger.info("BiG-RAG API Server started")
    api_logger.info(f"Documentation: http://{args.host}:{args.port}/docs")
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
