"""
MergerInterface - Abstract interface for entity merging strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class MergerInterface(ABC):
    """Abstract interface for entity merging strategies."""

    @abstractmethod
    async def merge(self, entities: List[Dict]) -> List[Dict]:
        """
        Merge duplicate entities.

        Args:
            entities: List of entities from extractor

        Returns:
            List of merged entities (duplicates consolidated)
        """
        pass
