"""
OrphanLinkerInterface - Abstract interface for orphan entity linking strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class OrphanLinkerInterface(ABC):
    """Abstract interface for orphan entity linking strategies."""

    @abstractmethod
    async def link(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Link orphan entities (entities without relation links).

        Args:
            entities: List of merged entities
            relations: List of extracted relations

        Returns:
            (linked_entities, synthetic_relations)
            - linked_entities: Entities with hyper_relation field populated
            - synthetic_relations: New relations created for orphans
        """
        pass
