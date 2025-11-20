"""
Cross-Chunk Consistency Validator for Production Knowledge Graph

Ensures extracted facts are consistent across document chunks.
Critical for educational domain where contradictions are unacceptable.

Validation checks:
1. Entity consistency: Same entity mentioned in different chunks should have consistent attributes
2. Numeric consistency: Same fact (e.g., "CSE has 120 seats") should have same numbers across chunks
3. Relation consistency: No contradictory relations (e.g., "CSE has 120 seats" vs "CSE has 100 seats")
4. Reference integrity: Entity references in relations must exist as entities

Strategy:
- Build entity registry (entity_name → attributes)
- Check for conflicts (same name, different attributes)
- Check numeric facts (same subject → same numbers)
- Flag contradictions for manual review
"""

import re
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from bigrag.bangla_utils import BanglaNumeralNormalizer


class ConsistencyValidator:
    """
    Validate cross-chunk consistency in extracted knowledge graph.

    Three consistency levels:
    - ENTITY_LEVEL: Check entity attribute consistency
    - NUMERIC_LEVEL: Check numeric fact consistency
    - RELATION_LEVEL: Check relation contradiction

    Returns detailed report with:
    - Conflicts found
    - Contradictory facts
    - Missing references
    - Consistency score
    """

    def __init__(self):
        """Initialize consistency validator."""
        self.normalizer = BanglaNumeralNormalizer()

    def validate_consistency(
        self,
        entities: List[Dict],
        relations: List[Dict],
        validation_level: str = "STRICT"
    ) -> Dict:
        """
        Validate consistency across all extracted data.

        Args:
            entities: All extracted entities (from all chunks)
            relations: All extracted relations (from all chunks)
            validation_level: "STRICT", "MODERATE", or "LENIENT"

        Returns:
            {
                'status': 'PASS' or 'FAIL',
                'validation_level': 'STRICT',
                'consistency_score': 0.98,
                'entity_conflicts': [...],
                'numeric_conflicts': [...],
                'relation_contradictions': [...],
                'reference_errors': [...],
                'recommendations': [...]
            }
        """

        # Step 1: Build entity registry
        entity_registry = self._build_entity_registry(entities)

        # Step 2: Check entity consistency
        entity_conflicts = self._check_entity_consistency(entity_registry)

        # Step 3: Check numeric consistency
        numeric_conflicts = self._check_numeric_consistency(
            entities,
            relations,
            entity_registry
        )

        # Step 4: Check relation contradictions
        relation_contradictions = self._check_relation_contradictions(relations)

        # Step 5: Check reference integrity
        reference_errors = self._check_reference_integrity(entities, relations)

        # Step 6: Compute consistency score
        total_checks = len(entities) + len(relations)
        total_issues = (
            len(entity_conflicts) +
            len(numeric_conflicts) +
            len(relation_contradictions) +
            len(reference_errors)
        )

        consistency_score = 1.0 - (total_issues / max(total_checks, 1))

        # Step 7: Determine status
        status = self._determine_status(
            consistency_score,
            entity_conflicts,
            numeric_conflicts,
            relation_contradictions,
            validation_level
        )

        # Step 8: Generate recommendations
        recommendations = self._generate_recommendations(
            entity_conflicts,
            numeric_conflicts,
            relation_contradictions,
            reference_errors
        )

        return {
            'status': status,
            'validation_level': validation_level,
            'consistency_score': consistency_score,
            'total_entities': len(entities),
            'total_relations': len(relations),
            'total_issues': total_issues,
            'entity_conflicts': entity_conflicts,
            'numeric_conflicts': numeric_conflicts,
            'relation_contradictions': relation_contradictions,
            'reference_errors': reference_errors,
            'recommendations': recommendations
        }

    def _build_entity_registry(self, entities: List[Dict]) -> Dict:
        """
        Build registry of entities grouped by name.

        Returns:
            {
                'CSE': [
                    {
                        'entity_name': 'CSE',
                        'entity_type': 'department_code',
                        'description': 'CSE stands for Computer Science',
                        'source_id': 'chunk_001',
                        'attributes': {'full_name': 'Computer Science'}
                    },
                    {
                        'entity_name': 'CSE',
                        'entity_type': 'department',  # CONFLICT: different type!
                        'description': 'CSE department',
                        'source_id': 'chunk_002',
                        'attributes': {}
                    }
                ]
            }
        """

        registry = defaultdict(list)

        for entity in entities:
            entity_name = entity.get('entity_name', '')

            # Normalize for comparison (convert Bangla to English)
            normalized_name = self.normalizer.bangla_to_english(entity_name)

            # Extract attributes from description
            attributes = self._extract_attributes_from_description(
                entity.get('description', '')
            )

            registry[normalized_name].append({
                'entity_name': entity_name,
                'entity_type': entity.get('entity_type', ''),
                'description': entity.get('description', ''),
                'source_id': entity.get('source_id', ''),
                'weight': entity.get('weight', 0.0),
                'attributes': attributes
            })

        return dict(registry)

    def _extract_attributes_from_description(self, description: str) -> Dict:
        """
        Extract attributes from entity description.

        Example:
        "120 হল CSE বিভাগের আসন সংখ্যা।"
        → {'value': '120', 'property': 'seat_count', 'subject': 'CSE'}

        This helps detect conflicts like:
        Chunk 1: "CSE has 120 seats"
        Chunk 2: "CSE has 100 seats"  # CONFLICT!
        """

        attributes = {}

        # Extract numbers
        numbers = re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', description)
        if numbers:
            attributes['numbers'] = numbers

        # Extract property keywords
        if 'আসন' in description or 'seat' in description.lower():
            attributes['property'] = 'seat_count'
        elif 'ফি' in description or 'fee' in description.lower():
            attributes['property'] = 'fee'
        elif 'জিপিএ' in description or 'gpa' in description.lower():
            attributes['property'] = 'gpa'
        elif 'কোড' in description or 'code' in description.lower():
            attributes['property'] = 'code'

        return attributes

    def _check_entity_consistency(self, entity_registry: Dict) -> List[Dict]:
        """
        Check for conflicting entity definitions.

        Conflicts:
        1. Same name, different types (e.g., "CSE" as both department and code)
        2. Same name, different numeric attributes (e.g., "CSE has 120" vs "CSE has 100")
        """

        conflicts = []

        for entity_name, occurrences in entity_registry.items():
            if len(occurrences) <= 1:
                continue  # No conflict possible with single occurrence

            # Check type consistency
            types = set(occ['entity_type'] for occ in occurrences)
            if len(types) > 1:
                # Type conflict
                conflicts.append({
                    'type': 'TYPE_CONFLICT',
                    'entity_name': entity_name,
                    'conflicting_types': list(types),
                    'occurrences': [
                        {
                            'type': occ['entity_type'],
                            'source': occ['source_id'],
                            'description': occ['description'][:100]
                        }
                        for occ in occurrences
                    ],
                    'severity': 'MEDIUM'  # Type conflicts might be intentional (e.g., "CSE" as both name and code)
                })

            # Check numeric attribute consistency
            numeric_attrs = [
                occ['attributes'].get('numbers', [])
                for occ in occurrences
                if occ['attributes'].get('property')
            ]

            if numeric_attrs:
                # Group by property
                property_groups = defaultdict(list)
                for occ in occurrences:
                    prop = occ['attributes'].get('property')
                    nums = occ['attributes'].get('numbers', [])
                    if prop and nums:
                        property_groups[prop].append({
                            'numbers': nums,
                            'source': occ['source_id'],
                            'description': occ['description']
                        })

                # Check for conflicting numbers within same property
                for prop, group in property_groups.items():
                    if len(group) <= 1:
                        continue

                    # Normalize numbers for comparison
                    normalized_numbers = [
                        [self.normalizer.bangla_to_english(n) for n in item['numbers']]
                        for item in group
                    ]

                    # Check if all occurrences have same numbers
                    first_nums = set(normalized_numbers[0])
                    for i, nums in enumerate(normalized_numbers[1:], 1):
                        if set(nums) != first_nums:
                            conflicts.append({
                                'type': 'NUMERIC_CONFLICT',
                                'entity_name': entity_name,
                                'property': prop,
                                'conflicting_values': [
                                    {
                                        'numbers': group[0]['numbers'],
                                        'source': group[0]['source'],
                                        'description': group[0]['description'][:100]
                                    },
                                    {
                                        'numbers': group[i]['numbers'],
                                        'source': group[i]['source'],
                                        'description': group[i]['description'][:100]
                                    }
                                ],
                                'severity': 'HIGH'  # Numeric conflicts are serious
                            })
                            break

        return conflicts

    def _check_numeric_consistency(
        self,
        entities: List[Dict],
        relations: List[Dict],
        entity_registry: Dict
    ) -> List[Dict]:
        """
        Check for contradictory numeric facts.

        Example conflicts:
        Relation 1: "CSE বিভাগে ১২০টি আসন রয়েছে।"
        Relation 2: "CSE department has 100 seats."  # CONFLICT: different number!
        """

        conflicts = []

        # Extract numeric facts from relations
        numeric_facts = defaultdict(list)

        for relation in relations:
            content = relation.get('content', '')

            # Extract subject entities (departments, programs, etc.)
            # Simple heuristic: look for known entities in relation content
            mentioned_entities = []
            for entity_name in entity_registry.keys():
                if entity_name.lower() in content.lower():
                    mentioned_entities.append(entity_name)

            # Extract numbers
            numbers = re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', content)

            # Classify fact type
            fact_type = self._classify_fact_type(content)

            if mentioned_entities and numbers and fact_type:
                for entity in mentioned_entities:
                    numeric_facts[(entity, fact_type)].append({
                        'numbers': numbers,
                        'content': content,
                        'source': relation.get('source_id', '')
                    })

        # Check for conflicts
        for (entity, fact_type), facts in numeric_facts.items():
            if len(facts) <= 1:
                continue

            # Normalize numbers for comparison
            normalized_facts = [
                {
                    'numbers': [self.normalizer.bangla_to_english(n) for n in fact['numbers']],
                    'content': fact['content'],
                    'source': fact['source']
                }
                for fact in facts
            ]

            # Check if all facts have same numbers
            first_nums = set(normalized_facts[0]['numbers'])
            for i, fact in enumerate(normalized_facts[1:], 1):
                if set(fact['numbers']) != first_nums:
                    conflicts.append({
                        'type': 'FACT_CONTRADICTION',
                        'entity': entity,
                        'fact_type': fact_type,
                        'conflicting_facts': [
                            {
                                'numbers': normalized_facts[0]['numbers'],
                                'content': normalized_facts[0]['content'][:100],
                                'source': normalized_facts[0]['source']
                            },
                            {
                                'numbers': fact['numbers'],
                                'content': fact['content'][:100],
                                'source': fact['source']
                            }
                        ],
                        'severity': 'CRITICAL'  # Fact contradictions are critical
                    })
                    break

        return conflicts

    def _classify_fact_type(self, content: str) -> Optional[str]:
        """
        Classify the type of fact from relation content.

        Returns:
            'seat_count', 'fee', 'gpa_requirement', 'deadline', etc.
        """

        content_lower = content.lower()

        if 'আসন' in content or 'seat' in content_lower:
            return 'seat_count'
        elif 'ফি' in content or 'fee' in content_lower:
            return 'fee'
        elif 'জিপিএ' in content or 'gpa' in content_lower:
            return 'gpa_requirement'
        elif 'তারিখ' in content or 'deadline' in content_lower or 'date' in content_lower:
            return 'deadline'
        else:
            return 'general'

    def _check_relation_contradictions(self, relations: List[Dict]) -> List[Dict]:
        """
        Check for logically contradictory relations.

        Example:
        Relation 1: "Admission starts on 01 December"
        Relation 2: "Admission starts on 15 December"  # CONTRADICTION
        """

        contradictions = []

        # Group relations by subject (simple heuristic: first entity mentioned)
        relation_groups = defaultdict(list)

        for relation in relations:
            content = relation.get('content', '')

            # Extract first capitalized word or Bangla phrase as subject
            subject_match = re.search(r'[A-Z][A-Za-z]+|[ক-হ]+(?:\s+[ক-হ]+){0,2}', content)
            if subject_match:
                subject = subject_match.group(0)
                relation_groups[subject].append(relation)

        # Check for contradictions within groups
        for subject, group in relation_groups.items():
            if len(group) <= 1:
                continue

            # Simple contradiction check: same keywords but different numbers
            for i, rel1 in enumerate(group):
                for rel2 in group[i + 1:]:
                    # Check if relations have similar keywords
                    similarity = self._compute_relation_similarity(
                        rel1['content'],
                        rel2['content']
                    )

                    if similarity > 0.5:  # Similar content
                        # Check if they have different numbers
                        nums1 = set(self.normalizer.bangla_to_english(n)
                                    for n in re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', rel1['content']))
                        nums2 = set(self.normalizer.bangla_to_english(n)
                                    for n in re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', rel2['content']))

                        if nums1 and nums2 and nums1 != nums2:
                            contradictions.append({
                                'type': 'RELATION_CONTRADICTION',
                                'subject': subject,
                                'similarity': similarity,
                                'relation_1': {
                                    'content': rel1['content'][:100],
                                    'source': rel1.get('source_id', ''),
                                    'numbers': list(nums1)
                                },
                                'relation_2': {
                                    'content': rel2['content'][:100],
                                    'source': rel2.get('source_id', ''),
                                    'numbers': list(nums2)
                                },
                                'severity': 'HIGH'
                            })

        return contradictions

    def _compute_relation_similarity(self, content1: str, content2: str) -> float:
        """
        Simple word-overlap similarity between two relation contents.

        Returns:
            Similarity score (0-1)
        """

        # Tokenize and normalize
        words1 = set(re.findall(r'\w+', content1.lower()))
        words2 = set(re.findall(r'\w+', content2.lower()))

        # Remove numbers (we check those separately)
        words1 = {w for w in words1 if not re.match(r'^[০-৯0-9]+$', w)}
        words2 = {w for w in words2 if not re.match(r'^[০-৯0-9]+$', w)}

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _check_reference_integrity(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> List[Dict]:
        """
        Check that all entity references in relations exist as entities.

        Example error:
        Relation: "CSE department has 120 seats"
        Entities: ["Computer Science", "120"]  # Missing "CSE" entity!
        """

        errors = []

        # Build entity name set (normalized)
        entity_names = set()
        for entity in entities:
            name = entity.get('entity_name', '')
            normalized = self.normalizer.bangla_to_english(name)
            entity_names.add(normalized.lower())

        # Check each relation for entity references
        for relation in relations:
            content = relation.get('content', '')

            # Find potential entity references (capitalized words, Bangla phrases)
            references = re.findall(r'[A-Z][A-Za-z]+|[ক-হ]+(?:\s+[ক-হ]+){0,2}', content)

            for ref in references:
                ref_normalized = self.normalizer.bangla_to_english(ref).lower()

                # Check if this reference exists as entity
                if ref_normalized not in entity_names:
                    # Allow common words (ignore short words)
                    if len(ref) >= 3:
                        errors.append({
                            'type': 'MISSING_ENTITY_REFERENCE',
                            'reference': ref,
                            'relation_content': content[:100],
                            'source': relation.get('source_id', ''),
                            'severity': 'LOW'  # This is expected (not all words are entities)
                        })

        return errors

    def _determine_status(
        self,
        consistency_score: float,
        entity_conflicts: List,
        numeric_conflicts: List,
        relation_contradictions: List,
        validation_level: str
    ) -> str:
        """Determine validation status based on level."""

        if validation_level == "STRICT":
            # No critical conflicts allowed
            critical_conflicts = [
                c for c in (entity_conflicts + numeric_conflicts + relation_contradictions)
                if c.get('severity') in ['HIGH', 'CRITICAL']
            ]
            if not critical_conflicts and consistency_score >= 0.99:
                return "PASS"
            else:
                return "FAIL"

        elif validation_level == "MODERATE":
            # Allow minor conflicts
            if consistency_score >= 0.95:
                return "PASS"
            else:
                return "FAIL"

        elif validation_level == "LENIENT":
            # Allow more conflicts
            if consistency_score >= 0.90:
                return "PASS"
            else:
                return "FAIL"

        else:
            return "FAIL"

    def _generate_recommendations(
        self,
        entity_conflicts: List,
        numeric_conflicts: List,
        relation_contradictions: List,
        reference_errors: List
    ) -> List[str]:
        """Generate actionable recommendations."""

        recommendations = []

        if entity_conflicts:
            recommendations.append(
                f"[WARN] Found {len(entity_conflicts)} entity conflicts. "
                f"Review entity definitions across chunks."
            )

        if numeric_conflicts:
            recommendations.append(
                f"[CRITICAL] Found {len(numeric_conflicts)} numeric conflicts. "
                f"Same entity has different numbers in different chunks. "
                f"Verify source document or merge entities correctly."
            )

        if relation_contradictions:
            recommendations.append(
                f"[CRITICAL] Found {len(relation_contradictions)} relation contradictions. "
                f"Same fact expressed with different values. "
                f"Check extraction quality or document consistency."
            )

        if reference_errors:
            # Only warn if there are many
            if len(reference_errors) > 10:
                recommendations.append(
                    f"[INFO] Found {len(reference_errors)} potential missing entity references. "
                    f"Consider extracting more entities or improving entity extraction."
                )

        if not entity_conflicts and not numeric_conflicts and not relation_contradictions:
            recommendations.append(
                "[SUCCESS] No critical consistency issues found. Knowledge graph is consistent."
            )

        return recommendations
