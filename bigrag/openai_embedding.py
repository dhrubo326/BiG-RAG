"""
OpenAI Embedding Wrapper for BiG-RAG
Provides OpenAI embeddings with async support and BiGRAG-compatible interface
"""
import os
import asyncio
from typing import List, Union
import numpy as np
from openai import AsyncOpenAI, OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APIConnectionError, Timeout

from .utils import logger, wrap_embedding_func_with_attrs


class OpenAIEmbedding:
    """
    Async-first OpenAI embedding wrapper compatible with BiGRAG's embedding interface
    Supports both text-embedding-3-small and text-embedding-3-large
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str = None,
        dimensions: int = None,
    ):
        """
        Initialize OpenAI embedding model

        Args:
            model_name: OpenAI embedding model name
                - "text-embedding-3-small" (default 1536 dims, max 8191 tokens)
                - "text-embedding-3-large" (default 3072 dims, max 8191 tokens)
            api_key: OpenAI API key (if None, loads from openai_api_key.txt or env)
            dimensions: Custom embedding dimensions (optional)
        """
        self.model_name = model_name

        # Set default dimensions based on model
        if dimensions is None:
            if "large" in model_name:
                self.dimensions = 3072
            else:
                self.dimensions = 1536
        else:
            self.dimensions = dimensions

        # Load API key
        if api_key is None:
            if os.path.exists('openai_api_key.txt'):
                with open('openai_api_key.txt', 'r') as f:
                    api_key = f.read().strip()
                logger.info("Loaded OpenAI API key from openai_api_key.txt")
            else:
                api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Create openai_api_key.txt or set OPENAI_API_KEY env var"
            )

        # Initialize both sync and async clients
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

        logger.info(f"[OpenAI Embedding] Initialized model: {model_name} ({self.dimensions} dimensions)")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
    )
    async def _async_embed(self, texts: List[str]) -> np.ndarray:
        """
        Internal async method to call OpenAI API with retry logic
        """
        try:
            response = await self.async_client.embeddings.create(
                input=texts,
                model=self.model_name,
                dimensions=self.dimensions
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Synchronous encode method (for compatibility)
        Internally uses async implementation
        """
        if isinstance(texts, str):
            texts = [texts]

        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._async_embed(texts))

    async def async_encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Async encode method (preferred for BiGRAG)
        """
        if isinstance(texts, str):
            texts = [texts]

        return await self._async_embed(texts)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        Encode queries - same as encode() for OpenAI
        """
        return self.encode(queries, **kwargs)

    async def async_encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        Async encode queries
        """
        return await self.async_encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Synchronous batch encoding (for compatibility)
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.async_encode_corpus(corpus, batch_size, **kwargs)
        )

    async def async_encode_corpus(
        self,
        corpus: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Async batch encoding with progress logging
        Processes corpus in batches to avoid rate limits
        """
        all_embeddings = []
        total = len(corpus)

        logger.info(f"[OpenAI Embedding] Starting batch encoding: {total} texts, batch_size={batch_size}")

        for i in range(0, total, batch_size):
            batch = corpus[i:i+batch_size]
            embeddings = await self._async_embed(batch)
            all_embeddings.append(embeddings)

            if (i // batch_size + 1) % 5 == 0 or i + batch_size >= total:
                progress = min(i + len(batch), total)
                logger.info(f"[OpenAI Embedding] Progress: {progress}/{total} texts encoded")

        result = np.vstack(all_embeddings)
        logger.info(f"[OpenAI Embedding] Completed: {result.shape[0]} embeddings created")
        return result


def create_openai_embedding_func(
    model_name: str = "text-embedding-3-large",
    dimensions: int = None
):
    """
    Factory function to create OpenAI embedding function compatible with BiGRAG

    Args:
        model_name: OpenAI model name
        dimensions: Custom dimensions (optional)

    Returns:
        Wrapped async embedding function
    """
    embedder = OpenAIEmbedding(model_name=model_name, dimensions=dimensions)

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedder.dimensions,
        max_token_size=8191  # OpenAI embedding max tokens
    )
    async def embedding_func(texts: List[str]) -> np.ndarray:
        """
        BiGRAG-compatible async embedding function
        """
        return await embedder.async_encode(texts)

    return embedding_func


# Pre-configured embedding functions for convenience
def openai_embedding_small():
    """text-embedding-3-small (1536 dimensions)"""
    return create_openai_embedding_func(
        model_name="text-embedding-3-small",
        dimensions=1536
    )


def openai_embedding_large():
    """text-embedding-3-large (3072 dimensions)"""
    return create_openai_embedding_func(
        model_name="text-embedding-3-large",
        dimensions=3072
    )


# Default export
openai_embedding = openai_embedding_large()
