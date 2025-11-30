"""
ExtractorInterface - Abstract interface for entity/relation extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class ExtractorInterface(ABC):
    """Abstract interface for entity/relation extraction strategies."""

    @abstractmethod
    async def extract(self, chunks: List[Dict]) -> Dict:
        """
        Extract entities and relations from chunks.

        Args:
            chunks: List of chunks from chunker

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'failed_chunks': [...]
            }
        """
        pass
