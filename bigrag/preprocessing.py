"""
External Query Preprocessing Utility

Provides standalone query preprocessing for advanced use cases:
- Batch preprocessing (preprocess many queries in parallel)
- Testing preprocessing independently
- Custom preprocessing pipelines
- Reusing preprocessing logic outside retrieval

Usage:
    from bigrag.preprocessing import QueryPreprocessor

    # Initialize
    preprocessor = QueryPreprocessor(
        llm_func=your_llm_function,
        language="English"
    )

    # Preprocess single query
    normalized, statement = await preprocessor.preprocess("who is messi")

    # Batch preprocess (parallel)
    queries = ["who is messi", "messi team", "messi goals"]
    results = await preprocessor.batch_preprocess(queries)
"""

import asyncio
from typing import Callable, List, Tuple, Optional
from .operate import preprocess_query
from .utils import logger


class QueryPreprocessor:
    """
    Standalone query preprocessor for advanced users.

    This class provides external access to BiGRAG's query preprocessing logic,
    allowing users to preprocess queries independently of retrieval operations.

    Attributes:
        llm_func: LLM completion function (async callable)
        language: Target language for preprocessing (default: "English")
        global_config: Global configuration dict (optional, for caching)
        hashing_kv: Key-value storage for caching (optional)

    Example:
        >>> preprocessor = QueryPreprocessor(llm_func=my_llm)
        >>> normalized, statement = await preprocessor.preprocess("who is messi")
        >>> print(normalized)  # "Who is Lionel Messi?"
        >>> print(statement)   # "Lionel Messi is an Argentine footballer..."
    """

    def __init__(
        self,
        llm_func: Callable,
        language: str = "English",
        global_config: Optional[dict] = None,
        hashing_kv: Optional[object] = None,
    ):
        """
        Initialize QueryPreprocessor.

        Args:
            llm_func: Async LLM completion function
            language: Target language for preprocessing (default: "English")
            global_config: Optional global config dict (for caching)
            hashing_kv: Optional key-value storage for caching
        """
        self.llm_func = llm_func
        self.language = language
        self.global_config = global_config or {}
        self.hashing_kv = hashing_kv

        logger.info(
            f"[QueryPreprocessor] Initialized with language={language}, "
            f"caching={'enabled' if hashing_kv else 'disabled'}"
        )

    async def preprocess(self, query: str) -> Tuple[str, str]:
        """
        Preprocess a single query.

        Args:
            query: Raw user query

        Returns:
            Tuple of (normalized_query, statement_query)
            - normalized_query: Question form with typos fixed
            - statement_query: Declarative statement form

        Example:
            >>> normalized, statement = await preprocessor.preprocess("who is messi")
            >>> print(normalized)  # "Who is Lionel Messi?"
        """
        logger.debug(f"[QueryPreprocessor] Preprocessing query: {query}")

        normalized, statement = await preprocess_query(
            query=query,
            language=self.language,
            llm_func=self.llm_func,
            global_config=self.global_config,
            hashing_kv=self.hashing_kv,
        )

        logger.debug(
            f"[QueryPreprocessor] Result: normalized='{normalized}', "
            f"statement='{statement[:50]}...'"
        )

        return normalized, statement

    async def batch_preprocess(
        self,
        queries: List[str],
        max_concurrent: int = 10,
    ) -> List[Tuple[str, str]]:
        """
        Preprocess multiple queries in parallel.

        This is more efficient than calling preprocess() sequentially,
        as it parallelizes the LLM calls.

        Args:
            queries: List of raw user queries
            max_concurrent: Maximum concurrent LLM calls (default: 10)

        Returns:
            List of tuples: [(normalized_1, statement_1), (normalized_2, statement_2), ...]

        Example:
            >>> queries = ["who is messi", "messi team", "messi goals"]
            >>> results = await preprocessor.batch_preprocess(queries)
            >>> for norm, stmt in results:
            ...     print(f"Normalized: {norm}")
        """
        logger.info(
            f"[QueryPreprocessor] Batch preprocessing {len(queries)} queries "
            f"(max_concurrent={max_concurrent})"
        )

        # Create semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(max_concurrent)

        async def preprocess_with_semaphore(q: str) -> Tuple[str, str]:
            async with semaphore:
                return await self.preprocess(q)

        # Process all queries in parallel (with concurrency limit)
        results = await asyncio.gather(
            *[preprocess_with_semaphore(q) for q in queries]
        )

        logger.info(f"[QueryPreprocessor] Batch preprocessing complete: {len(results)} queries processed")

        return results

    def preprocess_sync(self, query: str) -> Tuple[str, str]:
        """
        Synchronous wrapper for preprocess().

        Use this in non-async contexts.

        Args:
            query: Raw user query

        Returns:
            Tuple of (normalized_query, statement_query)
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot use preprocess_sync() from async context. "
                "Use await preprocess() instead."
            )
        return loop.run_until_complete(self.preprocess(query))

    def batch_preprocess_sync(
        self,
        queries: List[str],
        max_concurrent: int = 10,
    ) -> List[Tuple[str, str]]:
        """
        Synchronous wrapper for batch_preprocess().

        Use this in non-async contexts.

        Args:
            queries: List of raw user queries
            max_concurrent: Maximum concurrent LLM calls (default: 10)

        Returns:
            List of tuples: [(normalized_1, statement_1), ...]
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot use batch_preprocess_sync() from async context. "
                "Use await batch_preprocess() instead."
            )
        return loop.run_until_complete(self.batch_preprocess(queries, max_concurrent))


# Convenience function for one-off preprocessing
async def preprocess_query_standalone(
    query: str,
    llm_func: Callable,
    language: str = "English",
    global_config: Optional[dict] = None,
    hashing_kv: Optional[object] = None,
) -> Tuple[str, str]:
    """
    Convenience function for preprocessing a single query without creating a preprocessor instance.

    Args:
        query: Raw user query
        llm_func: Async LLM completion function
        language: Target language (default: "English")
        global_config: Optional global config dict
        hashing_kv: Optional key-value storage for caching

    Returns:
        Tuple of (normalized_query, statement_query)

    Example:
        >>> from bigrag.preprocessing import preprocess_query_standalone
        >>> normalized, statement = await preprocess_query_standalone(
        ...     "who is messi",
        ...     llm_func=my_llm_function
        ... )
    """
    preprocessor = QueryPreprocessor(llm_func, language, global_config, hashing_kv)
    return await preprocessor.preprocess(query)
