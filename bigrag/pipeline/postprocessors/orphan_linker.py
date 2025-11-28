"""
Orphan entity linking postprocessor.

Links orphan entities (entities without relations) to the graph.
"""

from typing import List, Dict, Tuple
from ...utils import logger


class OrphanLinker:
    """
    Orphan entity linker.

    Post-processing step to link orphan entities to existing relations.
    Reduces the number of isolated entities in the knowledge graph.
    """

    def __init__(self, max_orphan_ratio: float = 0.1):
        self.max_orphan_ratio = max_orphan_ratio

    async def process(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Link orphan entities to relations.

        Args:
            entities: List of entities
            relations: List of relations

        Returns:
            Tuple of (entities, relations) with orphans linked
        """
        if not entities or not relations:
            return entities, relations

        logger.info(f"[OrphanLinker] Processing {len(entities)} entities, {len(relations)} relations...")

        # Identify orphan entities
        entity_names = {e.get('entity_name') for e in entities}
        entities_in_relations = set()

        for relation in relations:
            entities_in_relations.add(relation.get('head_entity'))
            entities_in_relations.add(relation.get('tail_entity'))

        orphans = entity_names - entities_in_relations
        orphan_ratio = len(orphans) / len(entities) if entities else 0

        logger.info(f"[OrphanLinker] Found {len(orphans)} orphan entities ({orphan_ratio:.1%})")

        if orphan_ratio > self.max_orphan_ratio:
            logger.warning(f"[OrphanLinker] Orphan ratio {orphan_ratio:.1%} exceeds threshold {self.max_orphan_ratio:.1%}")
            logger.warning(f"[OrphanLinker] Consider improving extraction quality")

        # For now, just log orphans (actual linking logic would go here)
        # Full implementation would use similarity matching to link orphans

        return entities, relations
