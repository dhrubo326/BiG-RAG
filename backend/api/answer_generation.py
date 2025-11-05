"""
Answer generation module for BiG-RAG evaluation.

This module provides reusable functions for generating answers using RAG + LLM,
extracted from the /chat/completions endpoint for use in batch evaluation workflows.

Key functions:
    - generate_answer_with_rag(): Generate single answer with retrieval context
    - batch_generate_answers(): Process multiple questions in batch
    - format_context_for_display(): Format retrieval results for CSV export
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import time
import logging
from bigrag.base import QueryParam

logger = logging.getLogger(__name__)

# Evaluation-optimized system prompt (stricter than conversational prompt)
EVALUATION_SYSTEM_PROMPT = """You are an AI assistant. Answer the question based ONLY on the provided context.

Instructions:
- If the context contains the answer, provide a direct, concise answer
- If the context does NOT contain the answer, respond with: "I cannot answer this question based on the provided context."
- Do not make up information or use external knowledge
- Be precise and factual
- Cite information from the sources when appropriate"""


async def generate_answer_with_rag(
    question: str,
    rag_instance,
    llm_manager,
    embedding_manager,
    llm_provider: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,  # Deterministic for evaluation
    max_tokens: int = 500,
    top_k: int = 5,
    enable_reranking: bool = True,
    return_context: bool = False,
    system_prompt: Optional[str] = None
) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
    """
    Generate answer using RAG + LLM.

    This function performs the core RAG workflow:
    1. Retrieve relevant context from knowledge graph
    2. Format context with source citations
    3. Generate answer using LLM
    4. Optionally return retrieval context for analysis

    Args:
        question: User's question
        rag_instance: BiGRAG instance for retrieval
        llm_manager: LLM manager for answer generation
        embedding_manager: Embedding manager for entity/edge search
        llm_provider: LLM provider (OpenAI, HuggingFace, etc.)
        model: Model name (e.g., "gpt-4o-mini")
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum answer length
        top_k: Number of context items to retrieve
        enable_reranking: Use semantic reranking for chunks
        return_context: If True, return (answer, context_list)
        system_prompt: Custom system prompt (defaults to EVALUATION_SYSTEM_PROMPT)

    Returns:
        If return_context=False: Generated answer text
        If return_context=True: Tuple of (answer, retrieved_context_list)

    Example:
        >>> answer = await generate_answer_with_rag(
        ...     question="What is the capital of France?",
        ...     rag_instance=rag,
        ...     llm_manager=llm_manager,
        ...     embedding_manager=embedding_manager
        ... )
        >>> print(answer)
        "Paris"

        >>> answer, context = await generate_answer_with_rag(
        ...     question="What is the capital of France?",
        ...     rag_instance=rag,
        ...     llm_manager=llm_manager,
        ...     embedding_manager=embedding_manager,
        ...     return_context=True
        ... )
        >>> print(answer)
        "Paris"
        >>> print(len(context))
        5
    """
    try:
        # Step 1: Retrieve context from knowledge graph
        entity_match = None
        edge_match = None

        if embedding_manager.mode == "flagembedding":
            entity_match = await embedding_manager.search_entities(question, top_k)
            edge_match = await embedding_manager.search_edges(question, top_k)

        # Three-Path Retrieval + Semantic Reranking
        context_results = await rag_instance.aquery(
            question,
            param=QueryParam(
                mode="hybrid",  # Entity + Relation + Chunk retrieval
                only_need_context=True,
                top_k=top_k,
                enable_reranking=enable_reranking
            ),
            entity_match=entity_match,
            bipartite_edge_match=edge_match
        )

        # Step 2: Format retrieved context
        context_parts = []
        context_metadata = []

        if context_results:
            for i, item in enumerate(context_results[:top_k], 1):
                if isinstance(item, dict):
                    context_text = item.get("<knowledge>", str(item))
                    # Store metadata for later analysis
                    context_metadata.append({
                        "source_num": i,
                        "content": context_text,
                        "metadata": item
                    })
                else:
                    context_text = str(item)
                    context_metadata.append({
                        "source_num": i,
                        "content": context_text,
                        "metadata": {}
                    })

                context_parts.append(f"[Source {i}]\n{context_text}")

            context_str = "\n\n".join(context_parts)
        else:
            context_str = ""
            logger.warning(f"No context retrieved for question: {question}")

        # Step 3: Build prompt with context
        if system_prompt is None:
            system_prompt = EVALUATION_SYSTEM_PROMPT

        if context_str:
            user_prompt = f"""Based on the following context from the knowledge graph:

{context_str}

Question: {question}

Please provide a comprehensive answer based on the above context."""
        else:
            # No context retrieved - still ask the question but expect "cannot answer"
            user_prompt = f"""Question: {question}

Please provide an answer based on the provided context."""

        # Step 4: Generate answer using LLM
        response_text = await llm_manager.complete(
            prompt=user_prompt,
            provider=llm_provider,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Step 5: Return answer and optionally context
        if return_context:
            return response_text, context_metadata
        else:
            return response_text

    except Exception as e:
        logger.error(f"Error generating answer for question '{question}': {e}")
        error_answer = f"Error: Failed to generate answer ({str(e)})"
        if return_context:
            return error_answer, []
        else:
            return error_answer


async def batch_generate_answers(
    questions: List[Dict[str, Any]],
    rag_instance,
    llm_manager,
    embedding_manager,
    llm_provider: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
    top_k: int = 5,
    enable_reranking: bool = True,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Batch process multiple questions and generate answers.

    Args:
        questions: List of question dicts with keys: question, golden_answer, document_index, question_type
        rag_instance: BiGRAG instance
        llm_manager: LLM manager
        embedding_manager: Embedding manager
        llm_provider: LLM provider name
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max answer length
        top_k: Number of context items
        enable_reranking: Use semantic reranking
        progress_callback: Optional callback function(current, total) for progress updates

    Returns:
        List of result dicts with keys:
            - question: Original question
            - golden_answer: Ground truth answer
            - generated_answer: Model's answer
            - retrieval_context: Formatted context string
            - retrieval_context_metadata: Raw context metadata
            - document_index: Original document index
            - question_type: Question type
            - latency_ms: Generation latency
            - error: Error message (if any)

    Example:
        >>> results = await batch_generate_answers(
        ...     questions=[
        ...         {"question": "What is Paris?", "golden_answer": "Capital of France", "document_index": 0, "question_type": "single_passage"},
        ...         {"question": "What is London?", "golden_answer": "Capital of UK", "document_index": 1, "question_type": "single_passage"}
        ...     ],
        ...     rag_instance=rag,
        ...     llm_manager=llm_manager,
        ...     embedding_manager=embedding_manager
        ... )
        >>> len(results)
        2
    """
    results = []
    total = len(questions)

    for idx, q_dict in enumerate(questions):
        question = q_dict.get("question", "")
        golden_answer = q_dict.get("golden_answer", "")
        document_index = q_dict.get("document_index", "")
        question_type = q_dict.get("question_type", "")

        # Progress callback
        if progress_callback:
            progress_callback(idx + 1, total)

        # Generate answer with latency tracking
        start_time = time.time()
        error = None

        try:
            answer, context = await generate_answer_with_rag(
                question=question,
                rag_instance=rag_instance,
                llm_manager=llm_manager,
                embedding_manager=embedding_manager,
                llm_provider=llm_provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                enable_reranking=enable_reranking,
                return_context=True
            )
        except Exception as e:
            logger.error(f"Error processing question {idx+1}/{total}: {e}")
            answer = f"Error: {str(e)}"
            context = []
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000

        # Format context for display
        context_str = format_context_for_display(context)

        result = {
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": answer,
            "retrieval_context": context_str,
            "retrieval_context_metadata": context,  # For detailed analysis
            "document_index": document_index,
            "question_type": question_type,
            "latency_ms": round(latency_ms, 2),
            "error": error
        }

        results.append(result)

        logger.info(f"Processed question {idx+1}/{total} - Latency: {latency_ms:.2f}ms")

    return results


def format_context_for_display(context_metadata: List[Dict[str, Any]]) -> str:
    """
    Format retrieval context metadata for CSV export.

    Args:
        context_metadata: List of context dicts with source_num, content, metadata

    Returns:
        Formatted string representation of context

    Example:
        >>> context = [
        ...     {"source_num": 1, "content": "Paris is the capital of France.", "metadata": {}},
        ...     {"source_num": 2, "content": "Paris has a population of 2.1 million.", "metadata": {}}
        ... ]
        >>> formatted = format_context_for_display(context)
        >>> print(formatted)
        [Source 1] Paris is the capital of France.
        ---
        [Source 2] Paris has a population of 2.1 million.
    """
    if not context_metadata:
        return "[No context retrieved]"

    parts = []
    for ctx in context_metadata:
        source_num = ctx.get("source_num", "?")
        content = ctx.get("content", "")
        # Truncate very long context for CSV readability
        if len(content) > 500:
            content = content[:497] + "..."
        parts.append(f"[Source {source_num}] {content}")

    return "\n---\n".join(parts)


def is_no_answer_response(generated_answer: str) -> bool:
    """
    Detect if model refused to answer (for no-answer question evaluation).

    Args:
        generated_answer: Model's generated answer

    Returns:
        True if answer indicates refusal/uncertainty, False otherwise

    Example:
        >>> is_no_answer_response("I cannot answer this question based on the provided context.")
        True
        >>> is_no_answer_response("The capital is Paris.")
        False
    """
    no_answer_phrases = [
        "cannot answer",
        "not found in",
        "don't have information",
        "no information",
        "context does not contain",
        "not mentioned in",
        "unable to answer",
        "cannot determine",
        "not provided in",
        "insufficient information",
        "not available in",
        "do not know"
    ]

    answer_lower = generated_answer.lower()
    return any(phrase in answer_lower for phrase in no_answer_phrases)
