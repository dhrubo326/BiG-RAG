"""
RelationValidator - Relation completeness validation

Validates relations based on:
- Description length (min quality signal)
- Completeness score (0-10 scale from LLM extraction)
"""

from bigrag.interfaces.validator import ValidatorInterface
from typing import Dict

class RelationValidator(ValidatorInterface):
    def __init__(self, strictness: str = "MODERATE"):
        """
        Initialize relation validator.

        Args:
            strictness: Validation level - "STRICT", "MODERATE", or "LENIENT"
        """
        self.strictness = strictness

        # Thresholds based on old pipeline's VALIDATION_THRESHOLDS
        if strictness == "STRICT":
            self.relation_desc_min_length = 20
            self.relation_completeness_min = 8.0
        elif strictness == "MODERATE":
            self.relation_desc_min_length = 10
            self.relation_completeness_min = 6.0
        else:  # LENIENT
            self.relation_desc_min_length = 5
            self.relation_completeness_min = 3.0

    async def validate(self, extractions: Dict) -> Dict:
        """
        Validate relation completeness.

        Args:
            extractions: Dict with 'entities', 'relations', 'failed_chunks'

        Returns:
            Dict with validated relations (entities passed through unchanged), and summary
        """
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
        total_relations = len(extractions.get('relations', []))
        validity_rate = len(valid_relations) / total_relations if total_relations > 0 else 1.0

        if self.strictness == "STRICT":
            status = 'PASS' if validity_rate >= 0.90 else 'WARNING' if validity_rate >= 0.75 else 'FAIL'
        elif self.strictness == "MODERATE":
            status = 'PASS' if validity_rate >= 0.85 else 'WARNING' if validity_rate >= 0.70 else 'FAIL'
        else:  # LENIENT
            status = 'PASS' if validity_rate >= 0.75 else 'WARNING' if validity_rate >= 0.60 else 'FAIL'

        return {
            'entities': extractions.get('entities', []),  # Pass through unchanged
            'relations': valid_relations,
            'failed_chunks': extractions.get('failed_chunks', []),
            'summary': {
                'status': status,
                'relation_validity': validity_rate,
                'rejected_relations': rejected_relations,
                'validation_method': f'relation (strictness={self.strictness})',
                'note': 'Relation validation only - entities passed through unchanged'
            }
        }
