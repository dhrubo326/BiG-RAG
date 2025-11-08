Current Backend Architecture Analysis
Current State
21 endpoints in a 2,713-line server.py file
Already have good separation in api/ folder:
✅ Models: Pydantic schemas (models.py, models_eval.py)
✅ Business Logic: Evaluation, jobs, registry, stats
✅ Utilities: Markdown processing, KG helpers
❌ Routes: All concentrated in server.py
Endpoint Breakdown
Root/Health - 2 endpoints
Document Management - 4 endpoints
Job Management - 1 endpoint
Graph Management - 4 endpoints
Evaluation - 6 endpoints
Q&A/Retrieval - 2 endpoints
LLM (Chat) - 1 endpoint
Problems with Current Approach
1. Maintainability Issues
2,713 lines in a single file is hard to navigate
Difficult to locate specific endpoints
High chance of merge conflicts with multiple developers
2. Scalability Concerns
As you add more endpoints (UI features, admin panels, webhooks), it will become unmanageable
Hard to test individual route groups in isolation
3. Separation of Concerns
Server initialization logic mixed with route definitions
Difficult to reuse routes across different server configurations
4. Code Organization
Already have business logic in api/ but routes are scattered in server.py
Inconsistent organization pattern
✅ Recommended Architecture: APIRouter Pattern
FastAPI's APIRouter is specifically designed for this use case. Here's the ideal structure:
Proposed Folder Structure
backend/
├── api/
│   ├── __init__.py
│   │
│   ├── routes/                    # NEW: All route definitions
│   │   ├── __init__.py
│   │   ├── documents.py           # Document management routes
│   │   ├── graph.py               # Graph management routes
│   │   ├── evaluation.py          # Evaluation routes
│   │   ├── retrieval.py           # Q&A and search routes
│   │   ├── jobs.py                # Job status routes
│   │   ├── health.py              # Health and root routes
│   │   └── llm.py                 # LLM/chat routes
│   │
│   ├── core/                      # NEW: Core dependencies and config
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── dependencies.py        # FastAPI dependencies
│   │   └── managers.py            # LLM, Embedding managers
│   │
│   ├── services/                  # RENAME: Business logic (from current api/)
│   │   ├── answer_generation.py
│   │   ├── csv_evaluation.py
│   │   ├── evaluation.py
│   │   ├── export.py
│   │   ├── ground_truth.py
│   │   ├── jobs.py
│   │   ├── kg_utils.py
│   │   ├── metrics.py
│   │   ├── registry.py
│   │   └── stats.py
│   │
│   ├── models/                    # MOVE: Pydantic models
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   ├── evaluation.py
│   │   ├── graph.py
│   │   ├── jobs.py
│   │   └── retrieval.py
│   │
│   └── utils/                     # Keep: Utility functions
│       ├── __init__.py
│       └── utils.py
│
├── server.py                      # SIMPLIFIED: Just app setup and router mounting
├── requirements.txt
└── README.md
Example Implementation
1. New File: backend/api/routes/documents.py
"""Document management routes"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Optional

from ..models.documents import (
    UploadResponse, DocumentListResponse, DocumentDetailResponse,
    DeleteResponse, RebuildResponse
)
from ..services import registry, jobs
from ..core.dependencies import get_rag_instance

router = APIRouter(prefix="/documents", tags=["Document Management"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    # ... other parameters
):
    """Upload a document (.txt or .md) and add it to the knowledge graph."""
    # Implementation here
    pass


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    dataset: Optional[str] = None,
    search: Optional[str] = None,
    # ... other parameters
):
    """List all documents with optional filtering and pagination."""
    # Implementation here
    pass


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document_details(
    document_id: str,
    include_entities: bool = True,
    include_related: bool = True
):
    """Get detailed information about a specific document."""
    # Implementation here
    pass


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str, hard_delete: bool = False):
    """Delete a document from the system."""
    # Implementation here
    pass


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_graph(
    data_source: str = Form(None),
    force_full_rebuild: bool = Form(False)
):
    """Manually trigger knowledge graph rebuild."""
    # Implementation here
    pass
2. New File: backend/api/routes/graph.py
"""Graph management routes"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from ..models.graph import GraphStatsResponse
from ..core.dependencies import get_graph_manager

router = APIRouter(prefix="/graph", tags=["Graph Management"])


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_statistics(dataset: Optional[str] = None):
    """Get knowledge graph statistics."""
    pass


@router.get("/export")
async def export_graph(
    data_source: str,
    limit: Optional[int] = 1000,
    node_types: Optional[str] = None,
    # ... other parameters
):
    """Export the knowledge graph for a dataset in Cytoscape-compatible format."""
    pass


@router.get("/subgraph/neighbors")
async def get_node_neighbors(
    node_id: str,
    depth: int = 1,
    data_source: Optional[str] = None
):
    """Get the subgraph containing a node and its neighbors."""
    pass


@router.get("/subgraph/search")
async def search_nodes(
    q: str,
    limit: int = 20,
    data_source: Optional[str] = None
):
    """Search for nodes in the graph by text query."""
    pass
3. New File: backend/api/routes/evaluation.py
"""Evaluation routes"""
from fastapi import APIRouter, HTTPException

from ..models.evaluation import (
    RetrievalEvalRequest, RetrievalEvalResponse,
    AnswerEvalRequest, AnswerEvalResponse,
    CompareEvalRequest, CompareEvalResponse,
    BatchEvalRequest, BatchEvalResponse
)
from ..services import evaluation

router = APIRouter(prefix="/eval", tags=["Evaluation"])


@router.post("/retrieval", response_model=RetrievalEvalResponse)
async def evaluate_retrieval_endpoint(request: RetrievalEvalRequest):
    """Evaluate retrieval quality with ground truth documents."""
    pass


@router.post("/answer", response_model=AnswerEvalResponse)
async def evaluate_answer_endpoint(request: AnswerEvalRequest):
    """Evaluate answer quality against ground truth."""
    pass


@router.post("/compare", response_model=CompareEvalResponse)
async def compare_configurations_endpoint(request: CompareEvalRequest):
    """Compare different retrieval configurations side-by-side."""
    pass


@router.post("/batch", response_model=BatchEvalResponse)
async def batch_evaluate_endpoint(request: BatchEvalRequest):
    """Run batch evaluation with multiple questions."""
    pass
4. New File: backend/api/core/dependencies.py
"""FastAPI dependency injection functions"""
from functools import lru_cache
from typing import Annotated
from fastapi import Depends

from bigrag import BiGRAG
from .managers import LLMProviderManager, EmbeddingManager


# Singleton instances
_rag_instance = None
_llm_manager = None
_embedding_manager = None


def get_rag_instance() -> BiGRAG:
    """Get the global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        raise RuntimeError("RAG instance not initialized")
    return _rag_instance


def get_llm_manager() -> LLMProviderManager:
    """Get the global LLM manager."""
    global _llm_manager
    if _llm_manager is None:
        raise RuntimeError("LLM manager not initialized")
    return _llm_manager


def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager."""
    global _embedding_manager
    if _embedding_manager is None:
        raise RuntimeError("Embedding manager not initialized")
    return _embedding_manager


# Type aliases for dependency injection
RAGDep = Annotated[BiGRAG, Depends(get_rag_instance)]
LLMDep = Annotated[LLMProviderManager, Depends(get_llm_manager)]
EmbeddingDep = Annotated[EmbeddingManager, Depends(get_embedding_manager)]
5. Simplified backend/server.py
"""
BiG-RAG Unified API Server

A robust, production-ready API server with modular route organization.
"""

import argparse
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bigrag import BiGRAG
from bigrag.config import config
from bigrag.utils import logger

# Import route modules
from api.routes import (
    health,
    documents,
    graph,
    evaluation,
    retrieval,
    jobs,
    llm
)

# Import core modules
from api.core.managers import LLMProviderManager, EmbeddingManager
from api.core import dependencies


# ============================================================================
# Configuration & Initialization
# ============================================================================

parser = argparse.ArgumentParser(description="BiG-RAG Unified API Server")
parser.add_argument('--data_source', default=config.default_dataset)
parser.add_argument('--port', type=int, default=config.port)
parser.add_argument('--host', default=config.host)
parser.add_argument('--llm_provider', default=config.llm_provider,
                    choices=['openai', 'anthropic', 'google', 'grok'])
args = parser.parse_args()


# Initialize managers
PROJECT_ROOT = Path(__file__).parent.parent
working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
working_dir = str(PROJECT_ROOT / working_dir_base / args.data_source)

embedding_manager = EmbeddingManager(working_dir)
llm_manager = LLMProviderManager(default_provider=args.llm_provider)
server_start_time = time.time()

# Initialize BiGRAG
from bigrag.llm import gpt_4o_mini_complete
rag = BiGRAG(
    working_dir=working_dir,
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=embedding_manager.get_embedding_func(),
    chunk_token_size=config.chunk_size,
    chunk_overlap_token_size=config.chunk_overlap_size,
    enable_llm_cache=config.enable_llm_cache,
)

# Set global instances for dependency injection
dependencies._rag_instance = rag
dependencies._llm_manager = llm_manager
dependencies._embedding_manager = embedding_manager

logger.info(f"BiG-RAG initialized for dataset: {args.data_source}")
logger.info(f"Embedding mode: {embedding_manager.mode}")
logger.info(f"Available LLM providers: {', '.join(llm_manager.get_available_providers())}")


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Mount Routers
# ============================================================================

app.include_router(health.router)           # /, /health
app.include_router(documents.router)        # /documents/*
app.include_router(graph.router)            # /graph/*
app.include_router(evaluation.router)       # /eval/*
app.include_router(retrieval.router)        # /ask, /search
app.include_router(jobs.router)             # /status/{job_id}
app.include_router(llm.router)              # /chat/completions


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("BiG-RAG API Server started")
    logger.info(f"Documentation: http://{args.host}:{args.port}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("BiG-RAG API Server shutting down")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
Benefits of This Approach
1. Maintainability ✅
Each route group in its own file (100-300 lines each)
Easy to locate and modify specific endpoints
Clear separation of concerns
2. Scalability ✅
Can add new route groups without touching existing files
Easier to test individual route modules
Better support for team collaboration (less merge conflicts)
3. Code Organization ✅
Consistent structure: routes/ → services/ → models/
Business logic separated from HTTP layer
Dependency injection for better testability
4. Performance ✅
No performance impact (routers are registered at startup)
Better code splitting for lazy loading (if needed)
5. Developer Experience ✅
Faster IDE navigation and autocomplete
Easier to onboard new developers
Better documentation organization
Migration Strategy
Phase 1: Setup Structure (Low Risk)
Create api/routes/, api/core/, api/services/, api/models/ folders
Move current api/*.py files to api/services/
Create api/core/dependencies.py and api/core/managers.py
Phase 2: Extract Routes (Module by Module)
Start with health routes (simplest, 2 endpoints)
Then job routes (1 endpoint)
Then document routes (4 endpoints)
Then graph routes (4 endpoints)
Then evaluation routes (6 endpoints)
Finally retrieval/LLM routes (3 endpoints)
Phase 3: Simplify server.py
Remove extracted route code
Add router imports and app.include_router() calls
Test each module as you go
Phase 4: Split Models (Optional)
Break models.py into domain-specific files
Keep backward compatibility with __init__.py exports
Recommendation
YES, absolutely refactor to use APIRouter pattern! Your current architecture already has good separation in api/ for services and models. The missing piece is extracting routes into their own modules using FastAPI's APIRouter. Timeline estimate:
Phase 1: 1-2 hours (setup folders, move files)
Phase 2: 4-6 hours (extract routes, one module at a time)
Phase 3: 1 hour (simplify server.py, test)
Phase 4: 2-3 hours (optional, split models)