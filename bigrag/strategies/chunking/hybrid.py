"""
HybridChunker - Hybrid chunking strategy (tables + token-based paragraphs).
"""

from bigrag.interfaces.chunker import ChunkerInterface
from typing import List, Dict, Optional


class HybridChunker(ChunkerInterface):
    """Hybrid: detect tables first, then chunk remaining text."""

    def __init__(self, api_key: str, chunk_size: int = 1200, overlap: int = 100):
        # Use semantic for tables, token for paragraphs
        from bigrag.strategies.chunking.semantic import SemanticChunker
        from bigrag.strategies.chunking.token import TokenChunker

        self.semantic_chunker = SemanticChunker(api_key, chunk_size, overlap)
        self.token_chunker = TokenChunker(chunk_size, overlap)

    async def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Use semantic chunking (includes table detection).

        Args:
            text: Document content
            metadata: Optional metadata

        Returns:
            List of chunk dicts
        """
        # Semantic chunker already handles tables + paragraphs
        return await self.semantic_chunker.chunk(text, metadata)
