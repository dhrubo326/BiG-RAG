"""
EntityValidator - Entity quality validation

Validates entities based on:
- Entity name length (filter out single-char entities)
- Description length (min quality signal)
- Generic type filtering (reject "thing", "object", etc.)
"""

from bigrag.interfaces.validator import ValidatorInterface
from typing import Dict

class EntityValidator(ValidatorInterface):
    # Generic entity types to reject (low-quality extractions)
    GENERIC_TYPES = {'thing', 'object', 'item', 'concept', 'entity', 'stuff', 'matter'}

    def __init__(self, strictness: str = "MODERATE", allow_generic_types: bool = None):
        """
        Initialize entity validator.

        Args:
            strictness: Validation level - "STRICT", "MODERATE", or "LENIENT"
            allow_generic_types: Whether to allow generic entity types (auto-set based on strictness if None)
        """
        self.strictness = strictness

        # Thresholds based on old pipeline's VALIDATION_THRESHOLDS
        if strictness == "STRICT":
            self.entity_name_min_length = 3
            self.entity_desc_min_length = 20
            self.allow_generic_types = False if allow_generic_types is None else allow_generic_types
        elif strictness == "MODERATE":
            self.entity_name_min_length = 2
            self.entity_desc_min_length = 10
            self.allow_generic_types = True if allow_generic_types is None else allow_generic_types
        else:  # LENIENT
            self.entity_name_min_length = 1
            self.entity_desc_min_length = 5
            self.allow_generic_types = True if allow_generic_types is None else allow_generic_types

    async def validate(self, extractions: Dict) -> Dict:
        """
        Validate entity quality.

        Args:
            extractions: Dict with 'entities', 'relations', 'failed_chunks'

        Returns:
            Dict with validated entities (relations passed through unchanged), and summary
        """
        # Validate entities
        valid_entities = []
        rejected_entities = 0
        for e in extractions.get('entities', []):
            entity_name = e.get('entity_name', '')
            entity_type = e.get('entity_type', '').lower()
            description = e.get('description', '')

            # Check entity name length
            if len(entity_name) < self.entity_name_min_length:
                rejected_entities += 1
                continue

            # Check generic types
            if not self.allow_generic_types and entity_type in self.GENERIC_TYPES:
                rejected_entities += 1
                continue

            # Check description length
            if len(description) < self.entity_desc_min_length:
                rejected_entities += 1
                continue

            valid_entities.append(e)

        # Compute validation status
        total_entities = len(extractions.get('entities', []))
        validity_rate = len(valid_entities) / total_entities if total_entities > 0 else 1.0

        if self.strictness == "STRICT":
            status = 'PASS' if validity_rate >= 0.90 else 'WARNING' if validity_rate >= 0.75 else 'FAIL'
        elif self.strictness == "MODERATE":
            status = 'PASS' if validity_rate >= 0.85 else 'WARNING' if validity_rate >= 0.70 else 'FAIL'
        else:  # LENIENT
            status = 'PASS' if validity_rate >= 0.75 else 'WARNING' if validity_rate >= 0.60 else 'FAIL'

        return {
            'entities': valid_entities,
            'relations': extractions.get('relations', []),  # Pass through unchanged
            'failed_chunks': extractions.get('failed_chunks', []),
            'summary': {
                'status': status,
                'entity_validity': validity_rate,
                'rejected_entities': rejected_entities,
                'validation_method': f'entity (strictness={self.strictness})',
                'note': 'Entity validation only - relations passed through unchanged'
            }
        }
