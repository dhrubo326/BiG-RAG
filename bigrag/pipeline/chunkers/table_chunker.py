"""
Table-aware chunking module (from enhanced pipeline).

Wrapper around TableAwareChunker from preprocessors.
Handles table detection and semantic chunking.
"""

from typing import List, Dict
from ...preprocessors.smart_chunker import TableAwareChunker as BaseTableAwareChunker
from ...utils import logger


class TableChunker:
    """
    Table-aware text chunker.

    Thin wrapper around preprocessors.TableAwareChunker.
    Detects and preserves table structure during chunking.
    """

    def __init__(
        self,
        api_key: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        chunk_mode: str = "semantic"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_mode = chunk_mode

        # Initialize table extractor first
        from ...preprocessors.table_extractor import GPT4TableExtractor
        table_extractor = GPT4TableExtractor(api_key=api_key)

        # Initialize underlying TableAwareChunker with table extractor
        self.base_chunker = BaseTableAwareChunker(table_extractor=table_extractor)

    async def chunk(
        self,
        content: str,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Chunk text with table-aware semantic chunking.

        Args:
            content: Text to chunk (may contain tables)
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dicts with format:
            {
                'chunk_id': str,
                'content': str,
                'tokens': int,
                'chunk_order_index': int,
                'metadata': Dict,
                'has_table': bool  # Whether chunk contains table
            }
        """
        metadata = metadata or {}

        try:
            # Use base chunker's chunk_document method
            chunks = await self.base_chunker.chunk_document(content, metadata)

            # Add metadata about chunking method
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    if 'metadata' not in chunk:
                        chunk['metadata'] = {}
                    chunk['metadata'].update({
                        'chunking_method': 'table_aware',
                        'chunk_mode': self.chunk_mode,
                        'chunk_size': self.chunk_size,
                        'overlap': self.chunk_overlap
                    })
                    # Ensure chunk_order_index exists
                    if 'chunk_order_index' not in chunk:
                        chunk['chunk_order_index'] = i

            return chunks

        except Exception as e:
            logger.error(f"[TableChunker] Table-aware chunking failed: {e}")
            # Graceful degradation: fallback to token chunking
            logger.warning("[TableChunker] Falling back to token-based chunking")

            from .token_chunker import TokenChunker
            fallback_chunker = TokenChunker(
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            return await fallback_chunker.chunk(content, metadata)
