"""
Retrieval routes (Q&A and search)
"""

import json
from fastapi import APIRouter, HTTPException
from bigrag import QueryParam
from bigrag.utils import logger

from ..core.dependencies import (
    RAGDep, EmbeddingDep, get_data_source
)
from ..models.models import AskRequest, AskResponse, SearchRequest


router = APIRouter(tags=["Retrieval"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, rag: RAGDep, embedding_manager: EmbeddingDep):
    """
    Ask a single question with knowledge graph retrieval

    Supports multiple retrieval modes and LLM providers.
    """
    try:
        # For FlagEmbedding mode, pre-compute entity/edge matches
        entity_match = None
        relation_match = None

        if embedding_manager.mode == "flagembedding":
            entity_match = await embedding_manager.search_entities(request.question, request.top_k)
            relation_match = await embedding_manager.search_relations(request.question, request.top_k)

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
            relation_match=relation_match
        )

        if not result:
            return AskResponse(
                question=request.question,
                retrieved_contexts=[],
                num_results=0,
                mode=request.mode,
                llm_provider_used=request.llm_provider or get_data_source(),
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
            llm_provider_used=request.llm_provider or "default",
            message="Successfully retrieved relevant context"
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@router.post("/search")
async def search(request: SearchRequest, rag: RAGDep, embedding_manager: EmbeddingDep):
    """
    Batch retrieval for multiple queries

    Phase 3 Enhancement: Uses Three-Path Retrieval (Entity + Edge + Chunk) with optional semantic reranking
    """
    try:
        results = []
        for query_text in request.queries:
            entity_match = None
            relation_match = None

            if embedding_manager.mode == "flagembedding":
                entity_match = await embedding_manager.search_entities(query_text, 5)
                relation_match = await embedding_manager.search_relations(query_text, 5)

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
                relation_match=relation_match
            )
            results.append(json.dumps({"query": query_text, "results": result}))

        return results

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")
