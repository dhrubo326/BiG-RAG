"""
Health and system status routes
"""

import os
import time
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter

from ..models.models import HealthResponse as EnhancedHealthResponse, RAGInstanceInfo
from ..core.dependencies import (
    get_embedding_manager, get_llm_manager, get_server_start_time,
    get_data_source, get_working_dir
)
from ..services.jobs import get_queue_stats
from ..services.registry import registry


router = APIRouter(tags=["System"])


@router.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information and available endpoints
    """
    embedding_manager = get_embedding_manager()
    llm_manager = get_llm_manager()
    data_source = get_data_source()

    return {
        "message": "BiG-RAG Unified API Server - Enhanced",
        "version": "3.0.0",
        "dataset": data_source,
        "embedding_mode": embedding_manager.mode,
        "default_llm_provider": llm_manager.default_provider,
        "available_providers": llm_manager.get_available_providers(),
        "endpoints": {
            "system": {
                "root": "GET / - This endpoint (API information)",
                "health": "GET /health - System health and statistics",
                "docs": "GET /docs - Interactive API documentation (Swagger UI)",
                "redoc": "GET /redoc - Alternative API documentation (ReDoc)"
            },
            "document_management": {
                "upload": "POST /documents/upload - Upload .txt or .md files with metadata",
                "list": "GET /documents - List all documents with filtering and pagination",
                "details": "GET /documents/{document_id} - Get document details with KG stats",
                "delete": "DELETE /documents/{document_id} - Delete document (soft/hard)",
                "rebuild": "POST /documents/rebuild - Rebuild knowledge graph (full/incremental)"
            },
            "job_management": {
                "status": "GET /jobs/{job_id} - Get processing job status and progress"
            },
            "graph_management": {
                "stats": "GET /graph/stats - Get knowledge graph statistics",
                "export": "GET /graph/export - Export graph data (GraphML/JSON)",
                "neighbors": "GET /graph/subgraph/neighbors - Get node neighbors with filtering",
                "search": "GET /graph/subgraph/search - Search graph nodes by query"
            },
            "retrieval": {
                "ask": "POST /ask - Interactive Q&A with context and chat history",
                "search": "POST /search - Batch document retrieval with three-path algorithm"
            },
            "evaluation": {
                "retrieval": "POST /eval/retrieval - Evaluate retrieval quality (precision, recall)",
                "answer": "POST /eval/answer - Evaluate answer quality (EM, F1, similarity)",
                "compare": "POST /eval/compare - Compare different configurations",
                "batch": "POST /eval/batch - Run batch evaluation on dataset",
                "batch_generate": "POST /eval/batch_generate - Generate answers for batch evaluation",
                "evaluate_results": "POST /eval/evaluate_results - Evaluate generated results"
            },
            "llm": {
                "chat": "POST /chat/completions - OpenAI-compatible chat endpoint (multi-provider)"
            }
        },
        "features": [
            "Markdown (.md) and text (.txt) file support",
            "Background job processing with real-time progress tracking",
            "Document registry with metadata management",
            "Soft and hard delete with cascade cleanup",
            "Advanced filtering, search, and pagination",
            "Three-path retrieval (Entity + Relation + Chunk)",
            "Semantic reranking with cross-encoder",
            "Knowledge graph statistics and analytics",
            "Multi-LLM provider support (OpenAI, Anthropic, Google, Grok)",
            "Comprehensive evaluation suite (retrieval + answer quality)",
            "Graph export in multiple formats (GraphML, JSON)",
            "Subgraph extraction and neighbor search",
            "OpenAI-compatible chat endpoint",
            "Auto-detection of embedding format (OpenAI vs FlagEmbedding)"
        ],
        "quick_start": {
            "upload_document": "curl -X POST 'http://localhost:8001/documents/upload' -F 'file=@doc.txt' -F 'title=My Document'",
            "ask_question": "curl -X POST 'http://localhost:8001/ask' -H 'Content-Type: application/json' -d '{\"query\": \"Your question?\"}'",
            "list_documents": "curl 'http://localhost:8001/documents'",
            "check_health": "curl 'http://localhost:8001/health'"
        }
    }


@router.get("/health", response_model=EnhancedHealthResponse)
async def health_check():
    """
    Get comprehensive system health and statistics.

    **Returns:**
    - System status
    - RAG instance information
    - Job queue statistics
    - Server uptime
    """
    working_dir = get_working_dir()
    server_start_time = get_server_start_time()

    # Get active RAG instances
    rag_instances_info = {}
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
    expr_dir = PROJECT_ROOT / working_dir_base

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
