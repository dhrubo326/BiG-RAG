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


# Token counting utility
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken (OpenAI tokenizer)"""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, rag: RAGDep, embedding_manager: EmbeddingDep):
    """
    Ask a single question with knowledge graph retrieval

    Supports multiple retrieval modes and LLM providers.
    """
    try:
        # For FlagEmbedding mode, pre-compute entity/relation matches
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
                message="No relevant context found",
                retrieval_tokens=0
            )

        # Format results
        contexts = []
        all_context_text = []
        for i, item in enumerate(result, 1):
            if isinstance(item, dict):
                context_text = item.get("<knowledge>", str(item))
                context_item = {
                    "rank": i,
                    "context": context_text,
                    "coherence_score": item.get("<coherence>", 0.0),
                    "type": item.get("<type>", "unknown")  # Add type (entity/relation/chunk)
                }

                # Add metadata if present (from chunk retrieval)
                if item.get("<metadata>"):
                    context_item["metadata"] = item["<metadata>"]

                # Add source IDs if present
                if item.get("<source_ids>"):
                    context_item["source_ids"] = item["<source_ids>"]

                contexts.append(context_item)
                all_context_text.append(context_text)
            else:
                context_text = str(item)
                contexts.append({
                    "rank": i,
                    "context": context_text,
                    "coherence_score": 0.0,
                    "type": "unknown"
                })
                all_context_text.append(context_text)

        # Calculate token count for retrieved context
        combined_context = "\n\n".join(all_context_text)
        retrieval_tokens = count_tokens(combined_context)

        return AskResponse(
            question=request.question,
            retrieved_contexts=contexts,
            num_results=len(contexts),
            mode=request.mode,
            llm_provider_used=request.llm_provider or "default",
            message="Successfully retrieved relevant context",
            retrieval_tokens=retrieval_tokens
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
