"""
SemanticValidator - Enhanced semantic quality validation

Validates entities and relations based on:
- Description length (min quality signal)
- Entity name length (filter out single-char entities)
- Generic type filtering (reject "thing", "object", etc.)
- Relation completeness (0-10 scale)
"""

from bigrag.interfaces.validator import ValidatorInterface
from typing import Dict

class SemanticValidator(ValidatorInterface):
    # Generic entity types to reject (low-quality extractions)
    GENERIC_TYPES = {'thing', 'object', 'item', 'concept', 'entity', 'stuff', 'matter'}

    def __init__(self, strictness: str = "MODERATE", allow_generic_types: bool = None):
        """
        Initialize semantic validator.

        Args:
            strictness: Validation level - "STRICT", "MODERATE", or "LENIENT"
            allow_generic_types: Whether to allow generic entity types (auto-set based on strictness if None)
        """
        self.strictness = strictness

        # Thresholds based on old pipeline's VALIDATION_THRESHOLDS
        if strictness == "STRICT":
            self.entity_name_min_length = 3
            self.entity_desc_min_length = 20
            self.relation_desc_min_length = 20
            self.relation_completeness_min = 8.0
            self.allow_generic_types = False if allow_generic_types is None else allow_generic_types
        elif strictness == "MODERATE":
            self.entity_name_min_length = 2
            self.entity_desc_min_length = 10
            self.relation_desc_min_length = 10
            self.relation_completeness_min = 6.0
            self.allow_generic_types = True if allow_generic_types is None else allow_generic_types
        else:  # LENIENT
            self.entity_name_min_length = 1
            self.entity_desc_min_length = 5
            self.relation_desc_min_length = 5
            self.relation_completeness_min = 3.0
            self.allow_generic_types = True if allow_generic_types is None else allow_generic_types

    async def validate(self, extractions: Dict) -> Dict:
        """
        Validate semantic quality of entities and relations.

        Returns:
            Dict with validated entities, relations, and summary
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

        # Validate relations
        valid_relations = []
        rejected_relations = 0
        for r in extractions.get('relations', []):
            description = r.get('description', '') or r.get('hyper_relation_content', '')
            weight = r.get('weight', 5.0)  # Completeness score

            # Check description length
            if len(description) < self.relation_desc_min_length:
                rejected_relations += 1
                continue

            # Check completeness score
            if weight < self.relation_completeness_min:
                rejected_relations += 1
                continue

            valid_relations.append(r)

        # Compute validation status
        total_items = len(extractions.get('entities', [])) + len(extractions.get('relations', []))
        valid_items = len(valid_entities) + len(valid_relations)
        validity_rate = valid_items / total_items if total_items > 0 else 1.0

        if self.strictness == "STRICT":
            status = 'PASS' if validity_rate >= 0.90 else 'WARNING' if validity_rate >= 0.75 else 'FAIL'
        elif self.strictness == "MODERATE":
            status = 'PASS' if validity_rate >= 0.85 else 'WARNING' if validity_rate >= 0.70 else 'FAIL'
        else:  # LENIENT
            status = 'PASS' if validity_rate >= 0.75 else 'WARNING' if validity_rate >= 0.60 else 'FAIL'

        return {
            'entities': valid_entities,
            'relations': valid_relations,
            'failed_chunks': extractions.get('failed_chunks', []),
            'summary': {
                'status': status,
                'semantic_validity': validity_rate,
                'rejected_entities': rejected_entities,
                'rejected_relations': rejected_relations,
                'validation_method': f'semantic (strictness={self.strictness})'
            }
        }
