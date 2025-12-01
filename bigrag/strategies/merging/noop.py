"""
No-op Merger Strategy

Provides a merger that performs no merging operations (pass-through).
Useful when merging is disabled via enable_entity_merging=False.
"""

from typing import List, Dict, Any, Tuple
from bigrag.base import MergerInterface


class NoOpMerger(MergerInterface):
    """
    Merger that performs no merging operations.

    Simply returns entities and relations unchanged (pass-through).
    Used when enable_entity_merging=False in IndexingConfig.
    """

    async def merge(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Return entities and relations unchanged.

        Args:
            entities: List of entity dicts
            relations: List of relation dicts

        Returns:
            Tuple of (entities, relations) unchanged
        """
        return entities, relations
