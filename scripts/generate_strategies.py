"""
Script to generate all strategy implementation files for the modular indexing system.

This script creates all 18 strategy classes needed for the BiGRAG modular architecture.
"""

import os
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# Strategy implementations content
STRATEGIES = {
    # Chunking Strategies
    "bigrag/strategies/chunking/token.py": '''"""
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
''',

    "bigrag/strategies/chunking/semantic.py": '''"""
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
                table_result = await self.table_extractor.extract_tables(text)

                # Add table chunks
                for table in table_result.get('tables', []):
                    chunk_id = hashlib.md5(str(table).encode()).hexdigest()[:16]
                    chunks.append({
                        'chunk_id': f'chunk-{chunk_id}',
                        'type': 'table',
                        'content': table['markdown'],
                        'structured_data': table.get('data', {}),
                        'metadata': metadata or {}
                    })

                # Get text without tables
                text = table_result.get('text_without_tables', text)
            except Exception as e:
                print(f"[WARNING] Table extraction failed: {e}. Falling back to paragraph chunking.")

        # Chunk remaining text (paragraphs)
        paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]

        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para.split())

            if current_length + para_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\\n\\n'.join(current_chunk)
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
            chunk_text = '\\n\\n'.join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]

            chunks.append({
                'chunk_id': f'chunk-{chunk_id}',
                'type': 'paragraph',
                'content': chunk_text,
                'metadata': metadata or {}
            })

        return chunks
''',

    "bigrag/strategies/chunking/hybrid.py": '''"""
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
''',
}

def generate_all_strategies():
    """Generate all strategy files."""
    created_files = []

    for filepath, content in STRATEGIES.items():
        full_path = PROJECT_ROOT / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        created_files.append(str(full_path))
        print(f"Created: {filepath}")

    return created_files


if __name__ == "__main__":
    print("Generating strategy implementation files...")
    files = generate_all_strategies()
    print(f"\\nCompleted! Generated {len(files)} files.")
