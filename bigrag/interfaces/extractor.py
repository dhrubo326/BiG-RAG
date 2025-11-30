"""
ExtractorInterface - Abstract interface for entity/relation extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class ExtractorInterface(ABC):
    """Abstract interface for entity/relation extraction strategies."""

    @abstractmethod
    async def extract(self, chunks: List[Dict], language: Optional[str] = None) -> Dict:
        """
        Extract entities and relations from chunks.

        Args:
            chunks: List of chunks from chunker
            language: Language for extraction (auto-detected/from config if None)

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'failed_chunks': [...]
            }
        """
        pass
