"""
Semantic Reranker Module (Phase 3.3)

Provides cross-encoder based reranking for chunk retrieval.
Uses sentence-transformers cross-encoder to rerank candidate chunks by relevance.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Lightweight (6-layer MiniLM, ~80MB)
- Fast inference (~50ms per query-doc pair)
- Trained on MS MARCO passage ranking dataset
- Returns relevance scores between query and documents
"""

import asyncio
from typing import List, Tuple, Optional
from .utils import logger


class SemanticReranker:
    """
    Cross-encoder based semantic reranker for chunk retrieval.

    Usage:
        reranker = SemanticReranker()
        ranked_chunks = await reranker.rerank(query, candidate_chunks, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ):
        """
        Initialize reranker with cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model ID
            batch_size: Batch size for inference (higher = faster but more memory)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load cross-encoder model with graceful fallback"""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info(f"[Reranker] Loaded cross-encoder: {self.model_name}")
        except ImportError:
            logger.warning(
                "[Reranker] sentence-transformers not installed. "
                "Reranking will be disabled. Install with: pip install sentence-transformers"
            )
            self.model = None
        except Exception as e:
            logger.warning(f"[Reranker] Failed to load model: {e}. Reranking disabled.")
            self.model = None

    def is_available(self) -> bool:
        """Check if reranker is available"""
        return self.model is not None

    async def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, List[str], Optional[dict]]],  # List of (content, [source_ids], metadata)
        top_k: int = 5,
    ) -> List[dict]:
        """
        Rerank candidate chunks by semantic relevance to query.

        Args:
            query: User query string
            candidates: List of (chunk_content, source_ids, metadata) tuples
            top_k: Number of top results to return

        Returns:
            List of dicts with keys: content (str), sources (list), score (float), metadata (dict, optional)
            Sorted by relevance (highest score first)
            Returns original candidates (unranked) if model unavailable
        """
        if not self.is_available():
            logger.warning("[Reranker] Model not available, returning candidates unranked")
            # Return original candidates with default scores, preserving metadata
            return [
                {
                    "content": content,
                    "sources": sources,
                    "score": 0.5,
                    **({'metadata': metadata} if metadata else {})
                }
                for content, sources, metadata in candidates[:top_k]
            ]

        if not candidates:
            return []

        # Run CPU-bound reranking in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._rerank_sync, query, candidates, top_k)
        return result

    def _rerank_sync(
        self,
        query: str,
        candidates: List[Tuple[str, List[str], Optional[dict]]],
        top_k: int,
    ) -> List[dict]:
        """Synchronous reranking (called in thread pool)"""
        # Extract just the content for scoring
        contents = [content for content, _, _ in candidates]

        # Create query-document pairs
        query_doc_pairs = [[query, doc] for doc in contents]

        # Batch inference
        try:
            scores = self.model.predict(query_doc_pairs, batch_size=self.batch_size)
        except Exception as e:
            logger.error(f"[Reranker] Prediction failed: {e}")
            # Fallback: return unranked with default scores, preserving metadata
            return [
                {
                    "content": content,
                    "sources": sources,
                    "score": 0.5,
                    **({'metadata': metadata} if metadata else {})
                }
                for content, sources, metadata in candidates[:top_k]
            ]

        # Combine candidates with scores, preserving metadata
        scored_candidates = [
            {
                "content": content,
                "sources": sources,
                "score": float(score),
                **({'metadata': metadata} if metadata else {})
            }
            for (content, sources, metadata), score in zip(candidates, scores)
        ]

        # Sort by score (descending) and take top-k
        ranked = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)[:top_k]

        logger.info(
            f"[Reranker] Reranked {len(candidates)} candidates → top-{top_k} "
            f"(scores: {ranked[0]['score']:.3f} to {ranked[-1]['score']:.3f})"
        )

        return ranked

    def rerank_sync(
        self,
        query: str,
        candidates: List[Tuple[str, List[str], Optional[dict]]],
        top_k: int = 5,
    ) -> List[dict]:
        """
        Synchronous reranking (for non-async contexts).

        Use this if you're not in an async function.
        """
        return self._rerank_sync(query, candidates, top_k)


# Global reranker instance (lazy loading)
_global_reranker: Optional[SemanticReranker] = None


def get_reranker() -> SemanticReranker:
    """Get or create global reranker instance"""
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = SemanticReranker()
    return _global_reranker


async def rerank_chunks(
    query: str,
    chunks: List[Tuple[str, List[str], Optional[dict]]],
    top_k: int = 5,
    use_reranking: bool = True,
) -> List[dict]:
    """
    Convenience function for reranking chunks.

    Args:
        query: User query
        chunks: List of (content, source_ids, metadata) tuples
        top_k: Number of results to return
        use_reranking: Whether to use reranking (if False, returns original order)

    Returns:
        List of dicts with keys: content (str), sources (list), score (float), metadata (dict, optional)
    """
    if not use_reranking:
        # Return original chunks with default scores, preserving metadata
        return [
            {
                "content": content,
                "sources": sources,
                "score": 0.5,
                **({'metadata': metadata} if metadata else {})
            }
            for content, sources, metadata in chunks[:top_k]
        ]

    reranker = get_reranker()
    return await reranker.rerank(query, chunks, top_k=top_k)
