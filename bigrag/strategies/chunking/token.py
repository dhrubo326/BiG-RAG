"""
TokenChunker - Token-based fixed-size chunking strategy.

COPIED FROM: bigrag/preprocessors/smart_chunker.py::split_text_by_token_size()
This implementation uses the PROVEN logic from the old production pipeline.
"""

from bigrag.interfaces.chunker import ChunkerInterface
from typing import List, Dict, Optional
import hashlib


class TokenChunker(ChunkerInterface):
    """
    Token-based fixed-size chunking (fast, simple).

    Uses character-based approximation (1 token ≈ 4 chars) with proper overlap.
    Copied from smart_chunker.py (tested in production).
    """

    def __init__(self, chunk_size: int = 1200, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk document using fixed token windows with character-based approximation.

        Algorithm (from smart_chunker.py):
        1. Convert token sizes to character sizes (1 token ≈ 4 chars)
        2. Slide window across text with overlap
        3. Each chunk = chunk_size tokens, overlap = overlap tokens

        Args:
            text: Document content
            metadata: Optional metadata

        Returns:
            List of chunk dicts with chunk_method='fixed' tracking
        """
        # COPIED FROM smart_chunker.py:13-37
        # Character-based approximation: 1 token ≈ 4 characters
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.overlap * 4

        text_chunks = []
        start = 0

        while start < len(text):
            end = start + char_chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                text_chunks.append(chunk_text)

            start = end - char_overlap

            if start >= len(text):
                break

        # Convert to chunk dicts (with metadata tracking)
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]

            chunks.append({
                'chunk_id': f'chunk-{chunk_id}',
                'type': 'paragraph',
                'content': chunk_text,
                'metadata': {
                    **(metadata or {}),
                    'chunk_method': 'fixed'  # Track chunking method (for debugging)
                }
            })

        return chunks
