"""
HybridChunker - Hybrid chunking strategy (tables + token-based paragraphs).
"""

from bigrag.interfaces.chunker import ChunkerInterface
from typing import List, Dict, Optional


class HybridChunker(ChunkerInterface):
    """Hybrid: detect tables first, then chunk remaining text."""

    def __init__(
        self,
        api_key: str,
        chunk_size: int = 1200,
        overlap: int = 100,
        enable_table_detection: bool = True,
        hitl_handler: Optional[any] = None  # NEW: Issue #3
    ):
        # Use semantic for tables, token for paragraphs
        from bigrag.strategies.chunking.semantic import SemanticChunker
        from bigrag.strategies.chunking.token import TokenChunker

        self.semantic_chunker = SemanticChunker(
            api_key, chunk_size, overlap, enable_table_detection,
            hitl_handler=hitl_handler  # NEW: Pass through to SemanticChunker
        )
        self.token_chunker = TokenChunker(chunk_size, overlap)

    async def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Hybrid chunking: tables get semantic processing, paragraphs get token-based chunking.

        Strategy:
        1. Use SemanticChunker to detect and extract tables (preserves table structure)
        2. Remove table regions from text
        3. Use TokenChunker for remaining paragraphs (simpler, faster, more predictable)
        4. Combine table chunks + paragraph chunks

        Args:
            text: Document content
            metadata: Optional metadata

        Returns:
            List of chunk dicts (tables + paragraphs)
        """
        import logging
        logger = logging.getLogger(__name__)

        # Step 1: Extract tables using semantic chunker
        semantic_chunks = await self.semantic_chunker.chunk(text, metadata)

        # Step 2: Separate tables from paragraphs
        table_chunks = [c for c in semantic_chunks if c.get('type') == 'table']

        logger.info(f"[HybridChunker] Detected {len(table_chunks)} tables")

        # Step 3: If tables detected, remove table regions and chunk remaining text with TokenChunker
        if self.semantic_chunker.enable_table_detection and table_chunks:
            # Remove table markdown patterns from text
            import re
            table_pattern = r'\|[^\n]+\|(?:\n\|[^\n]+\|)+'
            text_without_tables = re.sub(table_pattern, '', text)

            # Chunk table-free text using token-based chunking
            para_chunks = await self.token_chunker.chunk(text_without_tables, metadata)
            logger.info(f"[HybridChunker] Created {len(para_chunks)} paragraph chunks from non-table text")
        else:
            # No tables detected or table detection disabled - use all semantic chunks
            para_chunks = [c for c in semantic_chunks if c.get('type') != 'table']

            # If no paragraphs from semantic chunker and no table detection, fall back to token chunker
            if not para_chunks and not table_chunks:
                logger.info("[HybridChunker] No semantic chunks found, using token chunker on full text")
                para_chunks = await self.token_chunker.chunk(text, metadata)

        # Step 4: Combine tables (semantic processing) + paragraphs (token-based)
        all_chunks = table_chunks + para_chunks
        logger.info(f"[HybridChunker] Total chunks: {len(all_chunks)} ({len(table_chunks)} tables, {len(para_chunks)} paragraphs)")

        return all_chunks
