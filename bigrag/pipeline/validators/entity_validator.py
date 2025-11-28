"""
Entity and relation validation module.

Validates extraction quality using numeric validation, entity quality scoring,
and relation completeness checks.
"""

from typing import List, Dict, Tuple
from ...validators.numeric_validator import NumericValidator
from ...utils import logger


# Validation thresholds (from features.py)
VALIDATION_THRESHOLDS = {
    "STRICT": {
        "numeric_coverage_min": 0.9,
        "entity_name_min_length": 2,
        "relation_description_min_length": 10,
        "max_generic_entity_ratio": 0.1
    },
    "MODERATE": {
        "numeric_coverage_min": 0.7,
        "entity_name_min_length": 1,
        "relation_description_min_length": 5,
        "max_generic_entity_ratio": 0.2
    },
    "LENIENT": {
        "numeric_coverage_min": 0.5,
        "entity_name_min_length": 1,
        "relation_description_min_length": 0,
        "max_generic_entity_ratio": 0.3
    }
}


class EntityValidator:
    """
    Entity and relation validator.

    Validates extraction results based on configurable strictness levels.
    Filters out low-quality entities and incomplete relations.
    """

    def __init__(
        self,
        enable_numeric: bool = False,
        enable_entity_quality: bool = False,
        enable_relation_quality: bool = False,
        strictness: str = "MODERATE",
        gemini_api_key: str = None
    ):
        self.enable_numeric = enable_numeric
        self.enable_entity_quality = enable_entity_quality
        self.enable_relation_quality = enable_relation_quality
        self.strictness = strictness

        # Get thresholds for strictness level
        self.thresholds = VALIDATION_THRESHOLDS.get(strictness, VALIDATION_THRESHOLDS["MODERATE"])

        # Initialize numeric validator if enabled
        self.numeric_validator = None
        if enable_numeric:
            # NumericValidator signature: __init__(api_key, use_llm_validation)
            self.numeric_validator = NumericValidator(
                api_key=gemini_api_key,
                use_llm_validation=True if gemini_api_key else False
            )

    async def validate(
        self,
        entities: List[Dict],
        relations: List[Dict],
        chunks: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Validate entities and relations.

        Args:
            entities: List of extracted entities
            relations: List of extracted relations
            chunks: Original text chunks (for numeric validation)

        Returns:
            Tuple of (validated_entities, validated_relations, validation_report)
        """
        validation_report = {
            'status': 'PASSED',
            'original_entities': len(entities),
            'original_relations': len(relations),
            'filtered_entities': 0,
            'filtered_relations': 0,
            'warnings': []
        }

        validated_entities = entities.copy()
        validated_relations = relations.copy()

        # Step 1: Numeric validation (if enabled)
        if self.enable_numeric and self.numeric_validator:
            logger.info("[EntityValidator] Running numeric validation...")
            try:
                validated_entities, validated_relations = await self._validate_numeric(
                    validated_entities,
                    validated_relations,
                    chunks
                )
                validation_report['numeric_validation'] = 'PASSED'
            except Exception as e:
                logger.warning(f"[EntityValidator] Numeric validation failed: {e}")
                validation_report['warnings'].append(f"Numeric validation error: {e}")

        # Step 2: Entity quality filtering (if enabled)
        if self.enable_entity_quality:
            logger.info("[EntityValidator] Running entity quality filtering...")
            original_count = len(validated_entities)
            validated_entities = self._filter_low_quality_entities(validated_entities)
            filtered_count = original_count - len(validated_entities)
            validation_report['filtered_entities'] = filtered_count
            if filtered_count > 0:
                logger.info(f"[EntityValidator] Filtered {filtered_count} low-quality entities")

        # Step 3: Relation quality filtering (if enabled)
        if self.enable_relation_quality:
            logger.info("[EntityValidator] Running relation quality filtering...")
            original_count = len(validated_relations)
            validated_relations = self._filter_incomplete_relations(validated_relations)
            filtered_count = original_count - len(validated_relations)
            validation_report['filtered_relations'] = filtered_count
            if filtered_count > 0:
                logger.info(f"[EntityValidator] Filtered {filtered_count} incomplete relations")

        # Update final counts
        validation_report['final_entities'] = len(validated_entities)
        validation_report['final_relations'] = len(validated_relations)

        logger.info(f"[EntityValidator] Validation complete: {len(validated_entities)} entities, {len(validated_relations)} relations")

        return validated_entities, validated_relations, validation_report

    async def _validate_numeric(
        self,
        entities: List[Dict],
        relations: List[Dict],
        chunks: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Run numeric consistency validation."""
        # Extract chunk texts
        chunk_texts = [chunk.get('content', '') if isinstance(chunk, dict) else str(chunk) for chunk in chunks]

        # Validate entities and relations
        validated_results = await self.numeric_validator.validate_batch(
            entities=entities,
            relations=relations,
            chunks=chunk_texts
        )

        return validated_results.get('entities', entities), validated_results.get('relations', relations)

    def _filter_low_quality_entities(self, entities: List[Dict]) -> List[Dict]:
        """Filter entities based on quality thresholds."""
        filtered = []
        generic_terms = ['thing', 'stuff', 'entity', 'item', 'object', 'element', 'something', 'anything']

        for entity in entities:
            entity_name = entity.get('entity_name', '').strip()

            # Check minimum length
            if len(entity_name) < self.thresholds['entity_name_min_length']:
                continue

            # Check for generic terms
            if entity_name.lower() in generic_terms:
                continue

            filtered.append(entity)

        return filtered

    def _filter_incomplete_relations(self, relations: List[Dict]) -> List[Dict]:
        """Filter relations based on completeness thresholds."""
        filtered = []

        for relation in relations:
            description = relation.get('description', '').strip()

            # Check minimum description length
            if len(description) < self.thresholds['relation_description_min_length']:
                continue

            # Ensure relation has required fields
            if not relation.get('head_entity') or not relation.get('tail_entity'):
                continue

            filtered.append(relation)

        return filtered
