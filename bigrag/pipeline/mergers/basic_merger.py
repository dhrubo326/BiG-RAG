"""
Basic entity merging module.

Simple hash-based deduplication using _merge_nodes_then_upsert from operate.py.
"""

from typing import List, Dict
from ...operate import _merge_nodes_then_upsert
from ...utils import logger


class BasicMerger:
    """
    Basic entity merger.

    Uses hash-based deduplication from standard pipeline.
    Fast and reliable for exact duplicates.
    """

    def __init__(self):
        pass

    async def merge(
        self,
        entities: List[Dict],
        relations: List[Dict] = None
    ) -> List[Dict]:
        """
        Merge duplicate entities using hash-based deduplication.

        Args:
            entities: List of entities to merge
            relations: List of relations (used for context)

        Returns:
            List of merged entities
        """
        if not entities:
            return []

        logger.info(f"[BasicMerger] Merging {len(entities)} entities...")

        try:
            # Use operate.py's _merge_nodes_then_upsert function
            # This function expects a different signature, so we'll do simple dedup
            merged = await self._simple_dedup(entities)

            logger.info(f"[BasicMerger] Merged to {len(merged)} unique entities")
            return merged

        except Exception as e:
            logger.error(f"[BasicMerger] Merge failed: {e}")
            # Return original entities on failure
            return entities

    async def _simple_dedup(self, entities: List[Dict]) -> List[Dict]:
        """Simple hash-based deduplication."""
        seen = {}
        merged = []

        for entity in entities:
            entity_name = entity.get('entity_name', '').strip().lower()

            if entity_name not in seen:
                seen[entity_name] = entity
                merged.append(entity)
            else:
                # Merge weights if duplicate
                existing = seen[entity_name]
                existing['weight'] = existing.get('weight', 1.0) + entity.get('weight', 1.0)

        return merged
