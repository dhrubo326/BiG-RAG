"""
LLM chat completion routes
"""

import os
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from bigrag import QueryParam
from bigrag.utils import logger

from ..core.dependencies import RAGDep, LLMDep, EmbeddingDep
from ..models.models import ChatCompletionRequest

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
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is Artificial Intelligence?"}],
        "temperature": 0.7,
        "max_tokens": 4096,
        "use_rag": true,
        "top_k": 60,
        "num_kg_in_context": 15,
        "num_chunks_in_context": 5,
        "mode": "hybrid",
        "enable_reranking": false,
        "language": "English"
    }
    ```

    **Parameters:**
    - `model`: LLM model to use (default: gpt-4o-mini)
    - `temperature`: 0.0-1.0, controls randomness (default: 0.7)
    - `use_rag`: Enable knowledge graph retrieval (default: true)
    - `top_k`: Items to retrieve from vector DBs (default: 60)
    - `num_kg_in_context`: KG relations in final context (default: 15)
    - `num_chunks_in_context`: Text chunks in final context (default: 5)
    - `mode`: Retrieval mode - hybrid/local/global/naive (default: hybrid)
    - `enable_reranking`: Use semantic reranking (default: false, requires sentence-transformers)
    - `language`: Response language override (optional)

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

        # Determine response language
        # Priority: request.language > DEFAULT_LANGUAGE from env > "English"
        response_language = request.language or os.getenv('DEFAULT_LANGUAGE', 'English')

        # RAG: Retrieve context from knowledge graph
        # IMPORTANT: Uses structured formatting with metadata display
        if request.use_rag:
            entity_match = None
            relation_match = None

            if embedding_manager.mode == "flagembedding":
                entity_match = await embedding_manager.search_entities(user_prompt, request.top_k)
                relation_match = await embedding_manager.search_relations(user_prompt, request.top_k)

            # Phase 3: Three-Path Retrieval + Semantic Reranking + Structured Formatting
            # Uses only_need_context=False to get formatted string with sections and metadata
            context_str = await rag.aquery(
                user_prompt,
                param=QueryParam(
                    mode=request.mode,
                    only_need_context=False,  # Returns formatted string with sections
                    top_k=request.top_k,
                    num_kg_in_context=request.num_kg_in_context,
                    num_chunks_in_context=request.num_chunks_in_context,
                    enable_reranking=request.enable_reranking,
                    language=request.language  # Pass language from request (optional override)
                ),
                entity_match=entity_match,
                relation_match=relation_match
            )

            # context_str is now a structured string with:
            # - ### Knowledge Graph - Entities (with relevance scores)
            # - ### Knowledge Graph - Relations (with relevance scores)
            # - ### Document Chunks (with metadata: category, title, tags)
            if context_str and context_str != "No relevant knowledge found.":
                # Create RAG system prompt with language instruction
                if not system_prompt:
                    system_prompt = f"""You are an expert AI assistant specializing in synthesizing information from a knowledge graph.

Instructions:
- Analyze the provided context which contains three types of information:
  1. Knowledge Graph Entities (key concepts and their descriptions)
  2. Knowledge Graph Relations (facts and relationships between concepts)
  3. Document Chunks (detailed textual evidence with metadata)
- Pay attention to metadata fields (Category, Title, Tags) which provide context about the source
- Synthesize a comprehensive, well-structured answer using information from all three sources
- Be accurate and cite relevant information when appropriate
- If the context doesn't fully answer the question, acknowledge what you know and what's uncertain
- **IMPORTANT: You MUST respond in {response_language} language.** Match the language of the user's question."""

                # Prepend context to user prompt with language instruction
                original_question = user_prompt
                user_prompt = f"""Based on the following context from the knowledge graph:

{context_str}

---

Question: {original_question}

Please provide a comprehensive answer in **{response_language}** by synthesizing information from the entities, relations, and document chunks above."""
        else:
            # Non-RAG mode: Add language instruction to system prompt if non-English
            if not system_prompt and response_language != "English":
                system_prompt = f"You are a helpful AI assistant. Always respond in {response_language} language."

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

        # Calculate token usage
        prompt_tokens = count_tokens(user_prompt, request.model)
        if system_prompt:
            prompt_tokens += count_tokens(system_prompt, request.model)
        for msg in history_messages:
            prompt_tokens += count_tokens(msg.get("content", ""), request.model)

        completion_tokens = count_tokens(response_text, request.model)
        total_tokens = prompt_tokens + completion_tokens

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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"LLM error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
