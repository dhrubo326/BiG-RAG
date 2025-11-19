"""
Tools available to the agent.

Provides search, variable storage, and other utilities.
"""

import time
from typing import List, Any, Optional
import asyncio

from api.agent_models import ContextItem, ExecutedAction
from agent.state import AgentState


class AgentTools:
    """
    Collection of tools available to the agent.

    Primary tool is search_bigrag which queries the BiG-RAG system.
    """

    def __init__(self, bigrag_instance):
        """
        Initialize tools with BiG-RAG instance.

        Args:
            bigrag_instance: An instance of BiGRAG for retrieval
        """
        self.bigrag = bigrag_instance

    async def search_bigrag(
        self,
        query: str,
        language: str = "English",
        top_k: int = 60,
        num_kg_in_context: int = 15,
        num_chunks_in_context: int = 5,
        enable_reranking: bool = False,
        state: Optional[AgentState] = None
    ) -> tuple[List[ContextItem], ExecutedAction]:
        """
        Query BiG-RAG retrieval system.

        Args:
            query: Search query
            language: Language for query preprocessing
            top_k: Items to retrieve from vector DBs (default: 60)
            num_kg_in_context: KG relations in final context (default: 15)
            num_chunks_in_context: Text chunks in final context (default: 5)
            enable_reranking: Enable semantic reranking (default: False)
            state: Agent state (for deduplication)

        Returns:
            Tuple of (contexts, executed_action)
        """
        start_time = time.time()

        # Check for duplicate queries
        if state and state.has_executed_similar_query(query):
            print(f"[AGENT] Skipping duplicate query: {query}")
            return [], ExecutedAction(
                action_type="search_bigrag_skipped",
                query=query,
                language=language,
                num_results=0,
                execution_time_ms=0.0
            )

        try:
            # Import QueryParam here to avoid circular imports
            from bigrag.base import QueryParam

            # Create query parameters (matching /ask endpoint behavior)
            param = QueryParam(
                mode="hybrid",
                only_need_context=True,  # Return only contexts, not full graph objects
                top_k=top_k,
                num_kg_in_context=num_kg_in_context,
                num_chunks_in_context=num_chunks_in_context,
                language=language,
                enable_reranking=enable_reranking
            )

            # Execute query
            results = await self.bigrag.aquery(query, param=param)

            # DEBUG: Log actual result count from BiG-RAG
            print(f"[AGENT DEBUG] BiG-RAG returned {len(results)} results for top_k={top_k}")

            # Convert results to ContextItem format
            contexts = []
            for idx, result in enumerate(results):
                # CRITICAL: Stop if we already have top_k contexts
                if len(contexts) >= top_k:
                    print(f"[AGENT DEBUG] Stopping at {len(contexts)} contexts (reached top_k limit)")
                    break

                # Extract text from BiG-RAG result format
                # BiG-RAG uses special keys: <knowledge>, <coherence>, <type>, <metadata>, <source_ids>
                if isinstance(result, dict):
                    # Primary content key used by BiG-RAG
                    text = result.get("<knowledge>", result.get('content', result.get('text', str(result))))

                    # Extract source IDs (BiG-RAG can return multiple sources)
                    source_ids = result.get("<source_ids>", [])
                    if source_ids:
                        source = ", ".join(source_ids) if isinstance(source_ids, list) else str(source_ids)
                    else:
                        source = result.get('source_id') or result.get('id') or f"result_{idx}"

                    # Extract metadata (BiG-RAG specific)
                    metadata = result.get("<metadata>", {})
                    if not metadata:
                        # Fallback: collect other metadata
                        metadata = {
                            k: v for k, v in result.items()
                            if k not in ['<knowledge>', '<coherence>', '<type>', '<source_ids>', '<metadata>',
                                       'content', 'text', 'source_id', 'id']
                        }

                    # Add type information (entity/relation/chunk)
                    if result.get("<type>"):
                        metadata["type"] = result["<type>"]

                    # Extract relevance score
                    score = result.get("<coherence>") or result.get('score') or result.get('relevance_score')
                else:
                    # Non-dict result (fallback)
                    text = str(result)
                    source = f"result_{idx}"
                    metadata = {}
                    score = None

                contexts.append(ContextItem(
                    text=text,
                    source=source,
                    metadata=metadata,
                    relevance_score=score
                ))

            # CRITICAL: Hard limit to top_k (safety measure)
            contexts = contexts[:top_k]

            execution_time = (time.time() - start_time) * 1000

            # Warn if we got more results than expected
            if len(results) > top_k * 2:
                print(f"[AGENT] WARNING: BiG-RAG returned {len(results)} results, limited to {top_k}")

            action = ExecutedAction(
                action_type="search_bigrag",
                query=query,
                language=language,
                num_results=len(contexts),
                execution_time_ms=execution_time
            )

            return contexts, action

        except Exception as e:
            print(f"[AGENT] Error in search_bigrag: {e}")
            import traceback
            traceback.print_exc()

            execution_time = (time.time() - start_time) * 1000

            action = ExecutedAction(
                action_type="search_bigrag_error",
                query=query,
                language=language,
                num_results=0,
                execution_time_ms=execution_time
            )

            return [], action

    async def search_bigrag_batch(
        self,
        queries: List[tuple[str, str]],  # List of (query, language) tuples
        top_k: int = 5,
        state: Optional[AgentState] = None,
        parallel: bool = True
    ) -> tuple[List[List[ContextItem]], List[ExecutedAction]]:
        """
        Execute multiple BiG-RAG searches.

        Args:
            queries: List of (query, language) tuples
            top_k: Number of contexts per query
            state: Agent state
            parallel: Execute in parallel if True, sequential if False

        Returns:
            Tuple of (list of context lists, list of actions)
        """
        if parallel:
            # Execute all queries in parallel
            tasks = [
                self.search_bigrag(query, lang, top_k, state)
                for query, lang in queries
            ]
            results = await asyncio.gather(*tasks)

            # Unzip results
            all_contexts = [contexts for contexts, _ in results]
            all_actions = [action for _, action in results]

            return all_contexts, all_actions
        else:
            # Execute sequentially
            all_contexts = []
            all_actions = []

            for query, lang in queries:
                contexts, action = await self.search_bigrag(query, lang, top_k, state)
                all_contexts.append(contexts)
                all_actions.append(action)

            return all_contexts, all_actions

    async def search_bigrag_multilingual(
        self,
        query: str,
        languages: List[str],
        top_k: int = 60,
        num_kg_in_context: int = 15,
        num_chunks_in_context: int = 5,
        enable_reranking: bool = False,
        state: Optional[AgentState] = None
    ) -> tuple[List[ContextItem], List[ExecutedAction]]:
        """
        Execute parallel BiG-RAG searches in multiple languages.

        For Bangla/Banglish queries, search in both Bangla and English
        to improve recall.

        Args:
            query: Search query
            languages: List of languages to search (e.g., ["Bangla", "English"])
            top_k: Items to retrieve from vector DBs per language (default: 60)
            num_kg_in_context: KG relations in final context per language (default: 15)
            num_chunks_in_context: Text chunks in final context per language (default: 5)
            enable_reranking: Enable semantic reranking (default: False)
            state: Agent state

        Returns:
            (merged_contexts, actions)
        """
        print(f"[AGENT] Multilingual search: {query} in languages: {languages}")

        # Execute searches in parallel for all languages
        tasks = []
        for lang in languages:
            tasks.append(self.search_bigrag(query, lang, top_k, num_kg_in_context, num_chunks_in_context, enable_reranking, state))

        results = await asyncio.gather(*tasks)

        # Merge results
        all_contexts = []
        all_actions = []

        for contexts, action in results:
            all_contexts.extend(contexts)
            all_actions.append(action)

        # Deduplicate contexts (same text might be found in multiple languages)
        seen_texts = set()
        unique_contexts = []

        for ctx in all_contexts:
            # Use first 100 chars as fingerprint
            fingerprint = ctx.text[:100].strip().lower()

            if fingerprint not in seen_texts:
                seen_texts.add(fingerprint)
                unique_contexts.append(ctx)

        # Limit to top_k
        unique_contexts = unique_contexts[:top_k]

        print(f"[AGENT] Multilingual search found {len(all_contexts)} total, {len(unique_contexts)} unique")

        return unique_contexts, all_actions

    @staticmethod
    def store_variable(key: str, value: Any, state: AgentState):
        """
        Store intermediate result in agent state.

        Args:
            key: Variable name
            value: Variable value
            state: Agent state
        """
        state.store_variable(key, value)

    @staticmethod
    def get_variable(key: str, state: AgentState) -> Optional[Any]:
        """
        Retrieve stored variable from agent state.

        Args:
            key: Variable name
            state: Agent state

        Returns:
            Variable value or None if not found
        """
        return state.get_variable(key)

    @staticmethod
    def format_contexts_for_llm(contexts: List[ContextItem], max_length: int = 5000) -> str:
        """
        Format contexts for LLM consumption.

        Args:
            contexts: List of context items
            max_length: Maximum character length

        Returns:
            Formatted string
        """
        if not contexts:
            return "No contexts retrieved."

        formatted = []
        total_length = 0

        for idx, ctx in enumerate(contexts):
            # Format with index and relevance score
            score_str = f" (score: {ctx.relevance_score:.3f})" if ctx.relevance_score else ""
            source_str = f" [source: {ctx.source}]" if ctx.source else ""

            entry = f"[{idx}]{score_str}{source_str}\n{ctx.text}\n"

            # Check length limit
            if total_length + len(entry) > max_length:
                formatted.append(f"\n[Note: Remaining {len(contexts) - idx} contexts truncated due to length]")
                break

            formatted.append(entry)
            total_length += len(entry)

        return "\n".join(formatted)

    @staticmethod
    def extract_variable_from_contexts(
        variable_name: str,
        variable_description: str,
        contexts: List[ContextItem]
    ) -> Optional[str]:
        """
        Extract specific information from contexts.

        For now, this is a simple implementation.
        TODO: Use LLM for intelligent extraction.

        Args:
            variable_name: Name of variable to extract
            variable_description: Description of what to extract
            contexts: Contexts to extract from

        Returns:
            Extracted value or None
        """
        # Simple implementation: return text from first context
        if contexts:
            return contexts[0].text[:200]  # First 200 chars
        return None
