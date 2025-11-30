"""
ChunkerInterface - Abstract interface for document chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class ChunkerInterface(ABC):
    """Abstract interface for document chunking strategies."""

    @abstractmethod
    async def chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk document into processable segments.

        Args:
            text: Document content (markdown)
            metadata: Optional metadata (title, category, tags)

        Returns:
            List of chunks: [
                {
                    'chunk_id': 'chunk-abc123',
                    'type': 'paragraph' | 'table',
                    'content': '...',
                    'metadata': {...}
                }
            ]
        """
        pass
