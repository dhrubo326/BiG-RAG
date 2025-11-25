"""
Unified Subgraph Query Routes

Provides endpoints for querying across multiple subgraphs with automatic routing.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from api.core import dependencies
from bigrag.base import QueryParam

router = APIRouter(prefix="/api/unified", tags=["Unified Subgraph"])


# ============================================================================
# Request/Response Models
# ============================================================================

class UnifiedQueryRequest(BaseModel):
    """Request model for unified query."""
    query: str = Field(..., description="User query string")
    force_subgraphs: Optional[List[str]] = Field(
        None,
        description="Force specific subgraphs (bypass routing)"
    )
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    enable_reranking: bool = Field(True, description="Enable semantic reranking")
    include_metadata: bool = Field(
        True,
        description="Include routing and execution metadata"
    )


class RoutingRequest(BaseModel):
    """Request model for routing decision only (no query execution)."""
    query: str = Field(..., description="User query string")


class AskRequest(BaseModel):
    """Simple ask request."""
    question: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=50, description="Number of results")


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/ask", summary="Simple unified ask (auto-routes to subgraph)")
async def unified_ask(request: AskRequest) -> Dict:
    """
    Simple search endpoint - automatically routes query to relevant subgraph.

    This is the easiest way to use unified mode:
    - Automatically routes your question to the right subgraph
    - Returns knowledge graph results (entities, relations, chunks)
    - No need to manually specify which subgraph to search

    Example:
        POST /api/unified/ask
        {
            "question": "Who won the 2022 World Cup?",
            "top_k": 5
        }
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    try:
        query_param = QueryParam(
            only_need_context=True,
            top_k=request.top_k
        )

        result = await executor.query(
            query=request.question,
            query_param=query_param,
            include_metadata=True
        )

        # Format response
        return {
            "question": request.question,
            "routed_to": result['routing']['subgraphs'],
            "confidence": result['routing']['confidence'],
            "results": result['results'],
            "num_results": len(result['results']),
            "execution_time": result['execution_time']
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/query", summary="Unified query across subgraphs")
async def unified_query(request: UnifiedQueryRequest) -> Dict:
    """
    Execute query with automatic routing to relevant subgraphs.

    This endpoint:
    1. Uses LLM to route query to relevant subgraph(s)
    2. Loads selected subgraphs from cache (lazy load if needed)
    3. Executes query in parallel across selected subgraphs
    4. Aggregates and returns results

    Example:
        POST /api/unified/query
        {
            "query": "Who won the 2022 World Cup?",
            "top_k": 10,
            "include_metadata": true
        }
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    try:
        query_param = QueryParam(
            only_need_context=True,
            top_k=request.top_k,
            enable_reranking=request.enable_reranking
        )

        result = await executor.query(
            query=request.query,
            query_param=query_param,
            force_subgraphs=request.force_subgraphs,
            include_metadata=request.include_metadata
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unified query failed: {str(e)}"
        )


@router.post("/route", summary="Get routing decision without executing query")
async def route_query(request: RoutingRequest) -> Dict:
    """
    Get routing decision for a query without executing it.

    Useful for debugging or showing users which subgraphs will be queried.

    Example:
        POST /api/unified/route
        {
            "query": "KUET CSE admission seats"
        }

        Response:
        {
            "subgraphs": ["kuet_test"],
            "reasoning": "Query asks about KUET admission...",
            "confidence": 0.95
        }
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    try:
        routing_decision = await executor.router.route(request.query)
        return routing_decision

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Routing failed: {str(e)}"
        )


@router.get("/subgraphs", summary="List available subgraphs")
async def list_subgraphs() -> Dict:
    """
    List all available subgraphs in the registry.

    Example response:
        {
            "subgraphs": ["demo_test", "football", "kuet_test"],
            "count": 3
        }
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    try:
        subgraphs = executor.get_available_subgraphs()
        return {
            "subgraphs": subgraphs,
            "count": len(subgraphs)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list subgraphs: {str(e)}"
        )


@router.get("/subgraphs/{subgraph_name}", summary="Get subgraph metadata")
async def get_subgraph_info(subgraph_name: str) -> Dict:
    """
    Get detailed metadata for a specific subgraph.

    Example response:
        {
            "name": "kuet_test",
            "path": "expr/kuet_test",
            "description": "KUET admission information...",
            "aliases": ["KUET", "kuet", ...],
            "topics": ["admission", "seats", ...],
            "enabled": true
        }
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    try:
        info = executor.get_subgraph_info(subgraph_name)

        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Subgraph '{subgraph_name}' not found"
            )

        # Add subgraph name to response
        response = {"name": subgraph_name, **info}
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get subgraph info: {str(e)}"
        )


@router.get("/cache/stats", summary="Get cache statistics")
async def get_cache_stats() -> Dict:
    """
    Get statistics about the subgraph cache.

    Shows:
    - Cache hits/misses/evictions
    - Currently cached subgraphs
    - Hit rate
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    try:
        stats = executor.get_cache_stats()
        return stats

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.post("/cache/clear", summary="Clear subgraph cache")
async def clear_cache() -> Dict:
    """
    Clear all cached subgraphs from memory.

    Useful for forcing reload of updated subgraphs.
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    try:
        executor.clear_cache()
        return {"status": "success", "message": "Cache cleared"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/registry/reload", summary="Reload subgraph registry")
async def reload_registry() -> Dict:
    """
    Reload subgraph registry from disk.

    Useful when registry is updated without restarting server.
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    try:
        executor.reload_registry()
        subgraphs = executor.get_available_subgraphs()

        return {
            "status": "success",
            "message": "Registry reloaded",
            "subgraphs": subgraphs,
            "count": len(subgraphs)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload registry: {str(e)}"
        )
