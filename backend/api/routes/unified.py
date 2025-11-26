"""
Unified Subgraph Query Routes

Provides endpoints for querying across multiple subgraphs with automatic routing.
"""

import os
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from api.core import dependencies
from bigrag.base import QueryParam
from bigrag.utils import logger

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


class Message(BaseModel):
    """OpenAI-compatible message format."""
    role: str = Field(..., description="Message role: system | user | assistant")
    content: str = Field(..., description="Message content")


class UnifiedChatRequest(BaseModel):
    """
    Unified chat request - extends UnifiedQueryRequest with LLM parameters.

    Inherits auto-routing behavior from /api/unified/query:
    - By default: LLM routes to relevant subgraph(s)
    - With force_subgraphs: Uses specific subgraph(s) (single-mode behavior)
    """

    # Message handling (OpenAI-compatible)
    messages: List[Message] = Field(
        ...,
        description="Chat messages in OpenAI format"
    )

    # Retrieval configuration
    use_rag: bool = Field(
        True,
        description="Enable knowledge graph retrieval (false = LLM-only mode)"
    )

    force_subgraphs: Optional[List[str]] = Field(
        None,
        description="Force specific subgraph(s) - bypasses auto-routing"
    )

    # Retrieval parameters
    mode: str = Field(
        "hybrid",
        description="Retrieval mode: hybrid | local | global | naive"
    )

    top_k: int = Field(60, ge=1, le=100, description="Items to retrieve from vector DBs")
    num_kg_in_context: int = Field(15, ge=1, le=50, description="KG relations in context")
    num_chunks_in_context: int = Field(5, ge=1, le=20, description="Text chunks in context")
    enable_reranking: bool = Field(True, description="Enable semantic reranking")

    # LLM parameters
    model: str = Field("gpt-4o-mini", description="LLM model name")
    llm_provider: Optional[str] = Field(None, description="LLM provider (auto-detect if None)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(4096, ge=1, le=16384, description="Max tokens in answer")

    # Output configuration
    output_mode: str = Field(
        "answer_with_context",
        description="Output mode: context_only | answer_only | answer_with_context"
    )

    include_metadata: bool = Field(True, description="Include routing metadata")
    include_retrieval_metrics: bool = Field(False, description="Include retrieval metrics")

    # Language
    language: Optional[str] = Field(None, description="Response language override")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "How many seats in KUET CSE?"}
                ],
                "output_mode": "answer_with_context",
                "use_rag": True,
                "top_k": 10,
                "enable_reranking": True
            }
        }


class UnifiedChatResponse(BaseModel):
    """Unified chat response."""

    answer: Optional[str] = Field(None, description="Synthesized answer from LLM")
    contexts: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved contexts")
    num_contexts: Optional[int] = Field(None, description="Number of contexts retrieved")
    routing: Optional[Dict[str, Any]] = Field(None, description="Routing decision")
    retrieval_metrics: Optional[Dict[str, Any]] = Field(None, description="Retrieval metrics")
    llm_metrics: Optional[Dict[str, Any]] = Field(None, description="LLM metrics")
    execution_time_ms: float = Field(..., description="Total execution time")
    output_mode: str = Field(..., description="Output mode used")


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/ask", summary="Simple unified ask (context-only)")
async def unified_ask(request: AskRequest) -> Dict:
    """
    Simple search endpoint - returns context only (no LLM synthesis).

    💡 **TIP:** For answers with LLM synthesis, use `/api/unified/chat` instead.

    This endpoint:
    - Automatically routes your question to the right subgraph
    - Returns knowledge graph results (entities, relations, chunks)
    - No LLM synthesis (context-only)

    **For full RAG pipeline with answer generation, use:**
    ```bash
    POST /api/unified/chat
    {
      "messages": [{"role": "user", "content": "Your question"}],
      "output_mode": "answer_with_context"
    }
    ```

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


# ============================================================================
# Helper Functions for Chat Endpoint
# ============================================================================

def extract_user_query(messages: List[Message]) -> str:
    """Extract user query from OpenAI-format messages."""
    for msg in reversed(messages):
        if msg.role == 'user':
            return msg.content
    raise HTTPException(400, "No user message found in messages array")


def extract_system_prompt(messages: List[Message]) -> Optional[str]:
    """Extract system prompt from messages."""
    for msg in messages:
        if msg.role == 'system':
            return msg.content
    return None


def extract_history(messages: List[Message]) -> List[Dict]:
    """Extract conversation history (assistant messages)."""
    history = []
    for msg in messages:
        if msg.role == 'assistant':
            history.append({'role': 'assistant', 'content': msg.content})
    return history


def format_contexts_for_llm(contexts: List[Dict], language: str = "English") -> str:
    """
    Format retrieved contexts for LLM consumption.
    Groups contexts by type (entities, relations, chunks) with metadata.
    """
    entities = []
    relations = []
    chunks = []

    for ctx in contexts:
        ctx_type = ctx.get('<type>', 'unknown')
        content = ctx.get('<knowledge>', str(ctx))
        score = ctx.get('<coherence>', 0.0)

        if ctx_type == 'entity':
            entities.append(f"- {content} (relevance: {score:.2f})")
        elif ctx_type == 'relation':
            relations.append(f"- {content} (relevance: {score:.2f})")
        elif ctx_type == 'chunk':
            metadata = ctx.get('<metadata>', {})
            meta_str = ""
            if metadata:
                meta_parts = []
                if metadata.get('title'):
                    meta_parts.append(f"Title: {metadata['title']}")
                if metadata.get('category'):
                    meta_parts.append(f"Category: {metadata['category']}")
                if metadata.get('tags'):
                    meta_parts.append(f"Tags: {', '.join(metadata['tags'])}")
                if meta_parts:
                    meta_str = f" [{', '.join(meta_parts)}]"
            chunks.append(f"- {content}{meta_str} (relevance: {score:.2f})")

    sections = []
    if entities:
        sections.append("### Knowledge Graph - Entities\n" + "\n".join(entities))
    if relations:
        sections.append("### Knowledge Graph - Relations\n" + "\n".join(relations))
    if chunks:
        sections.append("### Document Chunks\n" + "\n".join(chunks))

    if not sections:
        return "No relevant knowledge found."

    return "\n\n".join(sections)


def build_rag_system_prompt(language: str = "English") -> str:
    """Build RAG-optimized system prompt."""
    return f"""You are an expert AI assistant specializing in synthesizing information from a knowledge graph.

Instructions:
- Analyze the provided context which contains three types of information:
  1. Knowledge Graph Entities (key concepts and their descriptions)
  2. Knowledge Graph Relations (facts and relationships between concepts)
  3. Document Chunks (detailed textual evidence with metadata)
- Pay attention to metadata fields (Category, Title, Tags) which provide context about the source
- Synthesize a comprehensive, well-structured answer using information from all three sources
- Be accurate and cite relevant information when appropriate
- If the context doesn't fully answer the question, acknowledge what you know and what's uncertain
- **IMPORTANT: You MUST respond in {language} language.**"""


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken."""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return len(text) // 4  # Rough estimate


# ============================================================================
# Chat Endpoint
# ============================================================================

@router.post("/chat", summary="Unified chat with RAG + LLM synthesis", response_model=UnifiedChatResponse)
async def unified_chat(request: UnifiedChatRequest) -> UnifiedChatResponse:
    """
    Unified chat endpoint with automatic subgraph routing and LLM synthesis.

    This endpoint extends /api/unified/query with answer generation:
    - Inherits auto-routing behavior (LLM selects relevant subgraph(s))
    - Supports single-mode via force_subgraphs parameter
    - Flexible output: context-only, answer-only, or both

    **Routing Modes:**

    1. **Auto-routing (default):**
       LLM routes to relevant subgraph (e.g., "kuet_test")

    2. **Force specific subgraph (single-mode):**
       Uses only specified subgraph(s), bypasses routing

    **Output Modes:**

    - `context_only`: Returns retrieved contexts (no LLM synthesis)
    - `answer_only`: Returns synthesized answer (no context details)
    - `answer_with_context`: Returns both answer and contexts (default)

    **Example Requests:**

    ```bash
    # Full RAG pipeline (auto-routing)
    curl -X POST "http://localhost:8001/api/unified/chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "How many seats in KUET CSE?"}],
        "output_mode": "answer_with_context",
        "top_k": 10
      }'

    # Force specific subgraph
    curl -X POST "http://localhost:8001/api/unified/chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "KUET CSE seats?"}],
        "force_subgraphs": ["kuet_test"],
        "output_mode": "answer_only"
      }'

    # Context-only (no LLM synthesis)
    curl -X POST "http://localhost:8001/api/unified/chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "KUET CSE seats?"}],
        "output_mode": "context_only"
      }'
    ```
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    start_time = time.time()

    try:
        # Step 1: Extract user query from messages
        user_query = extract_user_query(request.messages)
        system_prompt = extract_system_prompt(request.messages)
        history = extract_history(request.messages)

        # Step 2: Retrieval (if use_rag=True and output_mode != answer_only)
        contexts = None
        routing_info = None
        retrieval_metrics = None

        if request.use_rag and request.output_mode != "answer_only":
            retrieval_start = time.time()

            query_param = QueryParam(
                mode=request.mode,
                only_need_context=True,
                top_k=request.top_k,
                num_kg_in_context=request.num_kg_in_context,
                num_chunks_in_context=request.num_chunks_in_context,
                enable_reranking=request.enable_reranking,
                language=request.language
            )

            result = await executor.query(
                query=user_query,
                query_param=query_param,
                force_subgraphs=request.force_subgraphs,
                include_metadata=request.include_metadata
            )

            contexts = result['results']
            routing_info = result.get('routing')

            if request.include_retrieval_metrics:
                retrieval_metrics = {
                    'retrieval_time_ms': (time.time() - retrieval_start) * 1000,
                    'num_results': len(contexts),
                    'subgraphs_queried': routing_info.get('subgraphs', []) if routing_info else []
                }

        # Step 3: Answer Generation (if output_mode includes answer)
        answer = None
        llm_metrics = None

        if request.output_mode in ["answer_only", "answer_with_context"]:
            llm_start = time.time()

            # Get LLM manager
            llm_manager = dependencies.get_llm_manager()

            # Format context for LLM
            if contexts and request.use_rag:
                context_str = format_contexts_for_llm(
                    contexts,
                    language=request.language or "English"
                )

                # Build RAG system prompt
                if not system_prompt:
                    system_prompt = build_rag_system_prompt(request.language or "English")

                # Prepend context to user query
                augmented_query = f"""Based on the following context from the knowledge graph:

{context_str}

---

Question: {user_query}

Please provide a comprehensive answer in **{request.language or 'English'}** by synthesizing information from the entities, relations, and document chunks above."""
            else:
                augmented_query = user_query
                if not system_prompt and request.language and request.language != "English":
                    system_prompt = f"You are a helpful AI assistant. Always respond in {request.language} language."

            # Call LLM
            answer = await llm_manager.complete(
                prompt=augmented_query,
                provider=request.llm_provider,
                model=request.model,
                system_prompt=system_prompt,
                history_messages=history,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            # Calculate metrics
            prompt_tokens = count_tokens(augmented_query + (system_prompt or ""), request.model)
            completion_tokens = count_tokens(answer, request.model)

            llm_metrics = {
                'model': request.model,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'latency_ms': (time.time() - llm_start) * 1000
            }

        # Step 4: Format Response
        execution_time_ms = (time.time() - start_time) * 1000

        response = UnifiedChatResponse(
            answer=answer if request.output_mode in ["answer_only", "answer_with_context"] else None,
            contexts=contexts if request.output_mode in ["context_only", "answer_with_context"] else None,
            num_contexts=len(contexts) if contexts else None,
            routing=routing_info if request.include_metadata else None,
            retrieval_metrics=retrieval_metrics,
            llm_metrics=llm_metrics,
            execution_time_ms=execution_time_ms,
            output_mode=request.output_mode
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Unified Chat] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )
