"""
LLM chat completion routes
"""

import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from bigrag import QueryParam
from bigrag.utils import logger

from ..core.dependencies import RAGDep, LLMDep, EmbeddingDep
from ..models.models import ChatCompletionRequest


router = APIRouter(prefix="/chat", tags=["LLM"])


@router.post("/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    rag: RAGDep,
    llm_manager: LLMDep,
    embedding_manager: EmbeddingDep
):
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
