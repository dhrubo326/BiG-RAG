"""
Token-based chunking module (from standard pipeline).

Simple, fast, reliable chunking using token counts.
This is a thin wrapper around operate.py's chunking_by_token_size() function.
"""

from typing import List, Dict
from ...utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken, compute_mdhash_id


class TokenChunker:
    """
    Token-based text chunker.

    Based on standard pipeline chunking_by_token_size() function.
    Splits text into overlapping chunks of fixed token size.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 100,
        tiktoken_model: str = "gpt-4o-mini"
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tiktoken_model = tiktoken_model

    async def chunk(
        self,
        content: str,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Chunk text into fixed-size token chunks with overlap.

        Args:
            content: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dicts with format:
            {
                'chunk_id': str,
                'content': str,
                'tokens': int,
                'chunk_order_index': int,
                'metadata': Dict
            }
        """
        metadata = metadata or {}

        # Tokenize content
        tokens = encode_string_by_tiktoken(content, model_name=self.tiktoken_model)

        # Create overlapping chunks
        chunks = []
        for index, start in enumerate(
            range(0, len(tokens), self.chunk_size - self.overlap)
        ):
            # Extract token slice
            chunk_tokens = tokens[start : start + self.chunk_size]

            # Decode back to text
            chunk_content = decode_tokens_by_tiktoken(
                chunk_tokens,
                model_name=self.tiktoken_model
            ).strip()

            # Create chunk ID
            chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")

            # Build chunk dict
            chunk = {
                'chunk_id': chunk_id,
                'content': chunk_content,
                'tokens': len(chunk_tokens),
                'chunk_order_index': index,
                'metadata': {
                    **metadata,
                    'chunking_method': 'token',
                    'chunk_size': self.chunk_size,
                    'overlap': self.overlap
                }
            }

            chunks.append(chunk)

        return chunks
