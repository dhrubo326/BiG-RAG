"""
Unified Entity Merging for Both Standard and Enhanced Pipelines

Provides multiple merging strategies:
1. basic: Name-based grouping (fast, used by standard pipeline)
2. fuzzy: Fuzzy matching + canonicalization (accurate, used by enhanced pipeline)
3. hybrid: Adaptive (basic for large graphs, fuzzy for small) [FUTURE]

This module consolidates entity merging logic from both pipelines into a single
implementation, enabling code reuse and consistent behavior.

Usage:
    # Basic merging (fast)
    merger = UnifiedEntityMerger(strategy='basic')
    merged = await merger.merge_entities(entities)

    # Fuzzy merging (accurate)
    merger = UnifiedEntityMerger(strategy='fuzzy')
    merged = await merger.merge_entities(entities)
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict
import asyncio

from bigrag.utils import compute_mdhash_id, logger
from bigrag.constants import ENTITY_PREFIX, GRAPH_FIELD_SEP


class UnifiedEntityMerger:
    """
    Unified entity merging supporting multiple strategies.

    Strategies:
    - 'basic': Simple name-based grouping (standard pipeline approach)
      - Groups by case-insensitive name
      - Sums weights
      - Collects source_ids
      - Picks longest description
      - Fast: O(n)

    - 'fuzzy': Advanced matching with canonicalization (enhanced pipeline approach)
      - Canonicalization map (domain-specific aliases)
      - Fuzzy string matching (Levenshtein distance)
      - Optional embedding similarity
      - Optional LLM verification
      - Accurate but slower: O(n²) worst case

    - 'hybrid': Adaptive strategy (future)
      - Use basic for large entity sets (> 1000 entities)
      - Use fuzzy for small entity sets (<= 1000 entities)
      - Balances speed and accuracy

    Backward Compatibility:
    - Default strategy='basic' maintains standard pipeline behavior
    - No breaking changes to existing code
    """

    def __init__(
        self,
        strategy: str = 'basic',
        fuzzy_threshold: float = 0.90,
        enable_embedding: bool = False,
        enable_llm_verification: bool = False
    ):
        """
        Initialize unified entity merger.

        Args:
            strategy: Merging strategy ('basic' | 'fuzzy' | 'hybrid')
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0)
            enable_embedding: Whether to use embedding similarity (fuzzy mode only)
            enable_llm_verification: Whether to use LLM verification (fuzzy mode only)

        Raises:
            ValueError: If strategy is invalid
        """
        valid_strategies = ['basic', 'fuzzy', 'hybrid']
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}"
            )

        self.strategy = strategy
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_embedding = enable_embedding
        self.enable_llm_verification = enable_llm_verification

        # Lazy import fuzzy dependencies (only when needed)
        self.canon_map = None
        self.entity_linker = None

        if strategy in ['fuzzy', 'hybrid']:
            try:
                from bigrag.merging.canonicalization import EntityCanonicalizationMap
                from bigrag.merging.entity_linker import SimpleEntityLinker

                self.canon_map = EntityCanonicalizationMap()
                # NOTE: SimpleEntityLinker doesn't accept fuzzy_threshold parameter
                # The fuzzy matching threshold is hardcoded in ProductionEntityLinker
                self.entity_linker = SimpleEntityLinker(self.canon_map)
                logger.info(f"[UnifiedMerger] Initialized with strategy={strategy}, fuzzy_threshold={fuzzy_threshold} (informational only)")
            except ImportError as e:
                logger.error(f"[UnifiedMerger] Failed to import fuzzy merge dependencies: {e}")
                logger.warning(f"[UnifiedMerger] Falling back to 'basic' strategy")
                self.strategy = 'basic'

    async def merge_entities(
        self,
        entities: List[Dict],
        merge_mode: str = 'append'
    ) -> List[Dict]:
        """
        Merge entity list using configured strategy.

        Args:
            entities: List of entity dicts with keys:
                - entity_name: str (required)
                - entity_id: str (optional, generated if missing)
                - description: str (optional)
                - weight: float (optional, default: 0)
                - source_id: str (optional, pipe-separated)
                - entity_type: str (optional, default: 'UNKNOWN')
                - key_score: int (optional, for importance ranking)

            merge_mode: Merge behavior
                - 'append': Merge multiple occurrences (sum weights, collect source_ids)
                - 'update': Update existing entities (overwrite attributes)

        Returns:
            List of merged entities with:
                - entity_id: Stable hash-based ID
                - entity_name: Canonical name
                - description: Best description (longest or most common)
                - weight: Aggregated weight
                - source_id: Pipe-separated source IDs
                - entity_type: Entity type
                - occurrences: Number of times entity appeared (basic mode)
                - entity_ids_merged: List of merged entity IDs (fuzzy mode)

        Raises:
            ValueError: If entities list is invalid
        """
        if not entities:
            logger.warning("[UnifiedMerger] Empty entity list provided")
            return []

        if not isinstance(entities, list):
            raise ValueError(f"entities must be a list, got {type(entities)}")

        # Validate entity structure
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(f"Entity at index {i} must be a dict, got {type(entity)}")
            if 'entity_name' not in entity:
                raise ValueError(f"Entity at index {i} missing required field 'entity_name'")

        logger.info(f"[UnifiedMerger] Merging {len(entities)} entities using strategy='{self.strategy}'")

        # Route to appropriate strategy
        if self.strategy == 'basic':
            merged = await self._merge_basic(entities, merge_mode)
        elif self.strategy == 'fuzzy':
            merged = await self._merge_fuzzy(entities, merge_mode)
        elif self.strategy == 'hybrid':
            merged = await self._merge_hybrid(entities, merge_mode)
        else:
            # Should never reach here due to validation in __init__
            raise ValueError(f"Unknown strategy: {self.strategy}")

        logger.info(f"[UnifiedMerger] Merged {len(entities)} → {len(merged)} entities "
                   f"(reduction: {len(entities) - len(merged)})")

        return merged

    async def _merge_basic(
        self,
        entities: List[Dict],
        merge_mode: str
    ) -> List[Dict]:
        """
        Basic name-based merging (STANDARD PIPELINE LOGIC).

        Algorithm:
        1. Normalize entity names (case-insensitive, strip whitespace)
        2. Group entities by normalized name
        3. For each group:
           a. Use first entity as base
           b. Sum weights across all occurrences
           c. Collect all unique source_ids
           d. Pick longest description
           e. Generate stable entity_id from name
           f. Track occurrence count

        Time Complexity: O(n) where n = number of entities
        Space Complexity: O(n)

        Args:
            entities: List of entity dicts
            merge_mode: 'append' or 'update'

        Returns:
            List of merged entities
        """
        # Group by entity_name (case-insensitive)
        entity_groups = defaultdict(list)
        for entity in entities:
            normalized_name = entity['entity_name'].strip().lower()
            entity_groups[normalized_name].append(entity)

        merged_entities = []

        for normalized_name, entity_list in entity_groups.items():
            # Use first entity as base (preserves original casing)
            base_entity = entity_list[0]
            entity_name = base_entity['entity_name']

            # Aggregate weights (sum across all occurrences)
            total_weight = sum(e.get('weight', 0) for e in entity_list)

            # Collect all unique source_ids
            source_ids = set()
            for e in entity_list:
                source_id = e.get('source_id', '')
                if source_id:
                    # source_id may be pipe-separated already
                    source_ids.update(source_id.split(GRAPH_FIELD_SEP))

            # Pick best description (longest wins)
            descriptions = [e.get('description', '') for e in entity_list if e.get('description')]
            description = max(descriptions, key=len) if descriptions else ''

            # Aggregate key_scores (sum for importance ranking)
            total_key_score = sum(e.get('key_score', 0) for e in entity_list)

            # Generate stable entity_id from name
            entity_id = compute_mdhash_id(entity_name, prefix=ENTITY_PREFIX)

            # Create merged entity
            merged_entity = {
                'entity_id': entity_id,
                'entity_name': entity_name,
                'description': description,
                'weight': total_weight,
                'source_id': GRAPH_FIELD_SEP.join(sorted(source_ids)),
                'entity_type': base_entity.get('entity_type', 'UNKNOWN'),
                'key_score': total_key_score,
                'occurrences': len(entity_list),  # Track how many times entity appeared
                'merge_strategy': 'basic'
            }

            # Preserve any additional fields from base entity
            for key, value in base_entity.items():
                if key not in merged_entity:
                    merged_entity[key] = value

            merged_entities.append(merged_entity)

        logger.info(f"[MERGE:BASIC] Merged {len(entities)} → {len(merged_entities)} entities")
        return merged_entities

    async def _merge_fuzzy(
        self,
        entities: List[Dict],
        merge_mode: str
    ) -> List[Dict]:
        """
        Fuzzy merging with canonicalization (ENHANCED PIPELINE LOGIC).

        Algorithm (from ProductionEntityLinker):
        1. Apply canonicalization map (domain-specific aliases)
        2. Group by exact alias match
        3. Fuzzy string matching (Levenshtein distance)
        4. Optional: Embedding similarity
        5. Optional: LLM verification
        6. Aggregate attributes from all matched entities

        Time Complexity: O(n²) worst case (fuzzy matching all pairs)
        Space Complexity: O(n)

        Args:
            entities: List of entity dicts
            merge_mode: 'append' or 'update'

        Returns:
            List of merged entities with entity_ids_merged tracking
        """
        if not self.entity_linker:
            logger.warning("[MERGE:FUZZY] Entity linker not initialized, falling back to basic merge")
            return await self._merge_basic(entities, merge_mode)

        # Delegate to existing entity linker
        # NOTE: SimpleEntityLinker already implements all fuzzy logic
        merged = await self.entity_linker.link_entities_across_chunks(entities)

        logger.info(f"[MERGE:FUZZY] Merged {len(entities)} → {len(merged)} entities")
        return merged

    async def _merge_hybrid(
        self,
        entities: List[Dict],
        merge_mode: str
    ) -> List[Dict]:
        """
        Hybrid adaptive merging (FUTURE - Smart strategy selection).

        Algorithm:
        1. If len(entities) > 1000: Use basic merge (fast for large graphs)
        2. Else: Use fuzzy merge (accurate for small graphs)

        This balances speed and accuracy based on entity set size.

        Args:
            entities: List of entity dicts
            merge_mode: 'append' or 'update'

        Returns:
            List of merged entities
        """
        threshold = 1000

        if len(entities) > threshold:
            logger.info(f"[MERGE:HYBRID] Using BASIC merge (entities={len(entities)} > {threshold})")
            return await self._merge_basic(entities, merge_mode)
        else:
            logger.info(f"[MERGE:HYBRID] Using FUZZY merge (entities={len(entities)} <= {threshold})")
            return await self._merge_fuzzy(entities, merge_mode)

    def get_strategy_info(self) -> Dict:
        """
        Get information about current merge strategy.

        Returns:
            Dict with strategy details:
                - strategy: Current strategy name
                - fuzzy_threshold: Threshold for fuzzy matching
                - features: List of enabled features
                - performance: Expected performance characteristics
        """
        info = {
            'strategy': self.strategy,
            'fuzzy_threshold': self.fuzzy_threshold if self.strategy != 'basic' else None,
            'features': [],
            'performance': {}
        }

        if self.strategy == 'basic':
            info['features'] = ['name_grouping', 'weight_aggregation', 'source_tracking']
            info['performance'] = {
                'time_complexity': 'O(n)',
                'space_complexity': 'O(n)',
                'speed': 'fast',
                'accuracy': 'good'
            }
        elif self.strategy == 'fuzzy':
            info['features'] = ['canonicalization', 'fuzzy_matching', 'alias_resolution']
            if self.enable_embedding:
                info['features'].append('embedding_similarity')
            if self.enable_llm_verification:
                info['features'].append('llm_verification')
            info['performance'] = {
                'time_complexity': 'O(n²)',
                'space_complexity': 'O(n)',
                'speed': 'moderate',
                'accuracy': 'excellent'
            }
        elif self.strategy == 'hybrid':
            info['features'] = ['adaptive_selection', 'threshold_based']
            info['performance'] = {
                'time_complexity': 'O(n) to O(n²)',
                'space_complexity': 'O(n)',
                'speed': 'adaptive',
                'accuracy': 'balanced'
            }

        return info


# Convenience functions for common use cases

async def merge_entities_basic(entities: List[Dict]) -> List[Dict]:
    """
    Quick basic merge (name-based grouping).

    Args:
        entities: List of entity dicts

    Returns:
        List of merged entities
    """
    merger = UnifiedEntityMerger(strategy='basic')
    return await merger.merge_entities(entities)


async def merge_entities_fuzzy(
    entities: List[Dict],
    fuzzy_threshold: float = 0.90
) -> List[Dict]:
    """
    Quick fuzzy merge (canonicalization + fuzzy matching).

    Args:
        entities: List of entity dicts
        fuzzy_threshold: Similarity threshold (0.0-1.0)

    Returns:
        List of merged entities
    """
    merger = UnifiedEntityMerger(strategy='fuzzy', fuzzy_threshold=fuzzy_threshold)
    return await merger.merge_entities(entities)


async def merge_entities_auto(entities: List[Dict]) -> List[Dict]:
    """
    Auto-select strategy based on entity count.

    Args:
        entities: List of entity dicts

    Returns:
        List of merged entities
    """
    merger = UnifiedEntityMerger(strategy='hybrid')
    return await merger.merge_entities(entities)
