"""
Fuzzy entity merging module.

Uses ProductionEntityLinker for similarity-based deduplication.
"""

from typing import List, Dict
from ...merging.entity_linker import ProductionEntityLinker, EntityCanonicalizationMap
from ...utils import logger


class FuzzyMerger:
    """
    Fuzzy entity merger.

    Uses ProductionEntityLinker for similarity-based merging.
    More accurate but slower than BasicMerger.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        # ProductionEntityLinker requires canonicalization_map, embedding_model, llm_func
        # For basic usage, create empty canonicalization map
        canon_map = EntityCanonicalizationMap()
        self.entity_linker = ProductionEntityLinker(
            canonicalization_map=canon_map,
            embedding_model=None,  # Optional
            llm_func=None  # Optional
        )

    async def merge(
        self,
        entities: List[Dict],
        relations: List[Dict] = None
    ) -> List[Dict]:
        """
        Merge similar entities using fuzzy matching.

        Args:
            entities: List of entities to merge
            relations: List of relations (used for context)

        Returns:
            List of merged entities
        """
        if not entities:
            return []

        logger.info(f"[FuzzyMerger] Merging {len(entities)} entities with threshold={self.similarity_threshold}...")

        try:
            # Use ProductionEntityLinker for fuzzy merging
            merged = await self.entity_linker.link_and_merge(entities, relations)

            logger.info(f"[FuzzyMerger] Merged to {len(merged)} unique entities")
            return merged

        except Exception as e:
            logger.error(f"[FuzzyMerger] Fuzzy merge failed: {e}")
            logger.warning("[FuzzyMerger] Falling back to basic deduplication")

            # Fallback to basic dedup
            from .basic_merger import BasicMerger
            fallback = BasicMerger()
            return await fallback.merge(entities, relations)
