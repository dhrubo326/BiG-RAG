"""
TokenChunker - Token-based fixed-size chunking strategy.
"""

from bigrag.interfaces.chunker import ChunkerInterface
from typing import List, Dict, Optional
import hashlib


class TokenChunker(ChunkerInterface):
    """Token-based fixed-size chunking (fast, simple)."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk document using fixed token windows.

        Args:
            text: Document content
            metadata: Optional metadata

        Returns:
            List of chunk dicts
        """
        # Split by sentences (simple tokenization)
        sentences = text.split('. ')

        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence.split())

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '. '.join(current_chunk) + '.'
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]

                chunks.append({
                    'chunk_id': f'chunk-{chunk_id}',
                    'type': 'paragraph',
                    'content': chunk_text,
                    'metadata': metadata or {}
                })

                # Start new chunk with overlap
                overlap_sentences = max(0, len(current_chunk) - (self.overlap // 50))
                current_chunk = current_chunk[overlap_sentences:]
                current_length = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_len

        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]

            chunks.append({
                'chunk_id': f'chunk-{chunk_id}',
                'type': 'paragraph',
                'content': chunk_text,
                'metadata': metadata or {}
            })

        return chunks
