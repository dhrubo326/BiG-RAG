"""
SemanticChunker - Table-aware semantic chunking strategy.
"""

from bigrag.interfaces.chunker import ChunkerInterface
from typing import List, Dict, Optional
import hashlib


class SemanticChunker(ChunkerInterface):
    """Table-aware semantic chunking (slow, accurate)."""

    def __init__(self, api_key: str, chunk_size: int = 1200, overlap: int = 100):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Import table extractor
        try:
            from bigrag.preprocessors.table_extractor import GPT4TableExtractor
            self.table_extractor = GPT4TableExtractor(api_key=api_key)
        except ImportError:
            self.table_extractor = None

    async def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk document with table detection.

        Args:
            text: Document content
            metadata: Optional metadata

        Returns:
            List of chunk dicts
        """
        chunks = []

        # Detect tables if extractor available
        if self.table_extractor:
            try:
                # FIXED: extract_tables_from_document returns List[Dict], not Dict!
                # This was causing "'list' object has no attribute 'get'" error
                tables = await self.table_extractor.extract_tables_from_document(
                    text,
                    document_metadata=metadata
                )

                # Import utilities from TableAwareChunker (don't reinvent the wheel)
                from bigrag.preprocessors.smart_chunker import TableAwareChunker
                from bigrag.preprocessors.table_extractor import BilingualDetector
                import re

                # Add table chunks (using same approach as TableAwareChunker)
                for table in tables:
                    # Convert table to natural language (CRITICAL for embedding quality)
                    nl_content = TableAwareChunker._table_to_natural_language(table)

                    # Detect language
                    lang_info = BilingualDetector.detect_languages(nl_content)

                    chunk_id = hashlib.md5(str(table).encode()).hexdigest()[:16]
                    chunks.append({
                        'chunk_id': f'chunk-{chunk_id}',
                        'type': 'table',
                        'content': nl_content,  # Natural language, not markdown
                        'structured_data': table,  # Full table dict
                        'metadata': {
                            **(metadata or {}),
                            'table_id': table.get('table_id'),
                            'table_type': table.get('table_type'),
                            'extraction_confidence': table.get('metadata', {}).get('confidence'),
                            'validation_status': table.get('metadata', {}).get('validation_status'),
                            'language_info': lang_info
                        }
                    })

                # Remove table markdown from text for paragraph chunking
                table_pattern = r'\|[^\n]+\|(?:\n\|[^\n]+\|)+'
                text = re.sub(table_pattern, '', text)

            except Exception as e:
                print(f"[WARNING] Table extraction failed: {e}. Falling back to paragraph chunking.")

        # Chunk remaining text (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para.split())

            if current_length + para_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]

                chunks.append({
                    'chunk_id': f'chunk-{chunk_id}',
                    'type': 'paragraph',
                    'content': chunk_text,
                    'metadata': metadata or {}
                })

                # Start new chunk (no overlap for semantic chunks)
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_len

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]

            chunks.append({
                'chunk_id': f'chunk-{chunk_id}',
                'type': 'paragraph',
                'content': chunk_text,
                'metadata': metadata or {}
            })

        return chunks
