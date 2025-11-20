"""
Numeric Validator for Production Knowledge Graph

Post-extraction validation to ensure 100% numeric accuracy.
Critical for educational domain where seat counts, GPAs, fees must be exact.

Validation strategy:
1. Extract all numbers from source document
2. Extract all numbers from knowledge graph (entities + relations)
3. Check coverage: 100% of source numbers must be in KG
4. Check hallucination: 0% of KG numbers should be not in source
5. Generate detailed report with location mapping
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from bigrag.bangla_utils import BanglaNumeralNormalizer


class NumericValidator:
    """
    Comprehensive numeric validation for extracted knowledge graph.

    Three validation levels:
    - STRICT: 100% coverage, 0% hallucination (required for production)
    - MODERATE: 95%+ coverage, <5% hallucination
    - LENIENT: 90%+ coverage, <10% hallucination

    Returns detailed report with:
    - Coverage percentage
    - Missing numbers (in source but not in KG)
    - Hallucinated numbers (in KG but not in source)
    - Number frequency analysis
    - Location mapping (which chunk contains which number)
    """

    def __init__(self):
        """Initialize numeric validator."""
        self.normalizer = BanglaNumeralNormalizer()

    def validate_extraction(
        self,
        source_document: str,
        entities: List[Dict],
        relations: List[Dict],
        validation_level: str = "STRICT"
    ) -> Dict:
        """
        Validate that ALL numbers from source are in KG.

        Args:
            source_document: Original document text
            entities: Extracted entities
            relations: Extracted relations
            validation_level: "STRICT", "MODERATE", or "LENIENT"

        Returns:
            {
                'status': 'PASS' or 'FAIL',
                'validation_level': 'STRICT',
                'numeric_coverage': 1.0,
                'hallucination_rate': 0.0,
                'total_source_numbers': 45,
                'total_kg_numbers': 45,
                'missing_numbers': [],
                'hallucinated_numbers': [],
                'number_frequency': {...},
                'recommendations': [...]
            }
        """

        # Step 1: Extract numbers from source
        source_numbers = self._extract_numbers_with_context(source_document)

        # Step 2: Extract numbers from KG
        kg_numbers = self._extract_numbers_from_kg(entities, relations)

        # Step 3: Normalize numbers for comparison
        source_set = set(source_numbers.keys())
        kg_set = set(kg_numbers.keys())

        # Also normalize to English for cross-language comparison
        source_normalized = {
            self.normalizer.bangla_to_english(num): num
            for num in source_set
        }
        kg_normalized = {
            self.normalizer.bangla_to_english(num): num
            for num in kg_set
        }

        # Step 4: Compute coverage (using normalized numbers)
        matched_normalized = set(source_normalized.keys()) & set(kg_normalized.keys())
        missing_normalized = set(source_normalized.keys()) - set(kg_normalized.keys())
        hallucinated_normalized = set(kg_normalized.keys()) - set(source_normalized.keys())

        # Map back to original forms
        missing_numbers = [source_normalized[n] for n in missing_normalized]
        hallucinated_numbers = [kg_normalized[n] for n in hallucinated_normalized]

        if source_normalized:
            numeric_coverage = len(matched_normalized) / len(source_normalized)
        else:
            numeric_coverage = 1.0

        if kg_normalized:
            hallucination_rate = len(hallucinated_normalized) / len(kg_normalized)
        else:
            hallucination_rate = 0.0

        # Step 5: Determine validation status
        status = self._determine_status(
            numeric_coverage,
            hallucination_rate,
            validation_level
        )

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            missing_numbers,
            hallucinated_numbers,
            source_numbers,
            kg_numbers
        )

        # Step 7: Analyze number frequency
        frequency_analysis = self._analyze_frequency(source_numbers, kg_numbers)

        return {
            'status': status,
            'validation_level': validation_level,
            'numeric_coverage': numeric_coverage,
            'hallucination_rate': hallucination_rate,
            'total_source_numbers': len(source_normalized),
            'total_kg_numbers': len(kg_normalized),
            'matched_numbers': len(matched_normalized),
            'missing_numbers': missing_numbers,
            'hallucinated_numbers': hallucinated_numbers,
            'missing_number_contexts': {
                num: source_numbers[num] for num in missing_numbers
            },
            'hallucinated_number_sources': {
                num: kg_numbers[num] for num in hallucinated_numbers
            },
            'frequency_analysis': frequency_analysis,
            'recommendations': recommendations
        }

    def validate_chunks(
        self,
        chunks: List[Dict],
        chunk_extractions: List[Dict],
        validation_level: str = "STRICT"
    ) -> Dict:
        """
        Validate extraction at chunk level.

        Ensures each chunk's numbers are preserved in its extraction.

        Args:
            chunks: Original chunks (from TableAwareChunker)
            chunk_extractions: Extraction results per chunk
            validation_level: Validation strictness

        Returns:
            {
                'overall_status': 'PASS',
                'chunk_validations': [
                    {
                        'chunk_id': 'chunk_001',
                        'status': 'PASS',
                        'coverage': 1.0,
                        ...
                    }
                ],
                'statistics': {...}
            }
        """

        chunk_validations = []

        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            chunk_content = chunk['content']

            # Find corresponding extraction
            extraction = next(
                (e for e in chunk_extractions if e.get('chunk_id') == chunk_id),
                None
            )

            if not extraction:
                # No extraction for this chunk
                chunk_validations.append({
                    'chunk_id': chunk_id,
                    'status': 'MISSING_EXTRACTION',
                    'coverage': 0.0,
                    'hallucination_rate': 0.0
                })
                continue

            # Validate this chunk
            entities = extraction.get('entities', [])
            relations = extraction.get('relations', [])

            result = self.validate_extraction(
                source_document=chunk_content,
                entities=entities,
                relations=relations,
                validation_level=validation_level
            )

            chunk_validations.append({
                'chunk_id': chunk_id,
                'status': result['status'],
                'coverage': result['numeric_coverage'],
                'hallucination_rate': result['hallucination_rate'],
                'missing_numbers': result['missing_numbers'],
                'hallucinated_numbers': result['hallucinated_numbers']
            })

        # Compute overall statistics
        stats = self._compute_chunk_statistics(chunk_validations)

        # Determine overall status
        overall_status = 'PASS' if all(
            cv['status'] in ['PASS', 'MISSING_EXTRACTION']
            for cv in chunk_validations
        ) else 'FAIL'

        return {
            'overall_status': overall_status,
            'chunk_validations': chunk_validations,
            'statistics': stats
        }

    def _extract_numbers_with_context(
        self,
        text: str,
        context_window: int = 50
    ) -> Dict[str, List[str]]:
        """
        Extract numbers from text with surrounding context.

        Args:
            text: Source text
            context_window: Characters before/after number

        Returns:
            {
                '120': ['...CSE বিভাগে 120টি আসন...', '...total 120 seats...'],
                '4.00': ['...GPA 4.00 প্রয়োজন...']
            }
        """

        number_contexts = defaultdict(list)

        # Find all numbers with their positions
        for match in re.finditer(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', text):
            number = match.group(0)
            start = match.start()
            end = match.end()

            # Extract context
            context_start = max(0, start - context_window)
            context_end = min(len(text), end + context_window)
            context = text[context_start:context_end]

            number_contexts[number].append(context)

        return dict(number_contexts)

    def _extract_numbers_from_kg(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Extract numbers from knowledge graph with sources.

        Args:
            entities: Entity list
            relations: Relation list

        Returns:
            {
                '120': ['entity: CSE', 'relation: CSE has 120 seats'],
                '4.00': ['entity: GPA requirement']
            }
        """

        number_sources = defaultdict(list)

        # Extract from entities
        for entity in entities:
            entity_name = entity.get('entity_name', '')
            description = entity.get('description', '')

            # Find numbers in entity name
            for num in re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', entity_name):
                number_sources[num].append(f"entity_name: {entity_name}")

            # Find numbers in description
            for num in re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', description):
                number_sources[num].append(f"entity_desc: {entity_name}")

        # Extract from relations
        for relation in relations:
            content = relation.get('content', '')

            # Find numbers in relation content
            for num in re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', content):
                number_sources[num].append(f"relation: {content[:50]}...")

        return dict(number_sources)

    def _determine_status(
        self,
        numeric_coverage: float,
        hallucination_rate: float,
        validation_level: str
    ) -> str:
        """
        Determine validation status based on level.

        Args:
            numeric_coverage: Percentage of source numbers in KG
            hallucination_rate: Percentage of KG numbers not in source
            validation_level: "STRICT", "MODERATE", or "LENIENT"

        Returns:
            "PASS" or "FAIL"
        """

        if validation_level == "STRICT":
            # Production requirement: 100% coverage, 0% hallucination
            if numeric_coverage == 1.0 and hallucination_rate == 0.0:
                return "PASS"
            else:
                return "FAIL"

        elif validation_level == "MODERATE":
            # Development: 95%+ coverage, <5% hallucination
            if numeric_coverage >= 0.95 and hallucination_rate < 0.05:
                return "PASS"
            else:
                return "FAIL"

        elif validation_level == "LENIENT":
            # Early testing: 90%+ coverage, <10% hallucination
            if numeric_coverage >= 0.90 and hallucination_rate < 0.10:
                return "PASS"
            else:
                return "FAIL"

        else:
            return "FAIL"

    def _generate_recommendations(
        self,
        missing_numbers: List[str],
        hallucinated_numbers: List[str],
        source_numbers: Dict,
        kg_numbers: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on validation results."""

        recommendations = []

        if missing_numbers:
            recommendations.append(
                f"[CRITICAL] {len(missing_numbers)} numbers from source are missing in KG. "
                f"Check extraction completeness."
            )

            # Identify common patterns in missing numbers
            if any(self.normalizer.is_bangla_numeral(num) for num in missing_numbers):
                recommendations.append(
                    "[HINT] Some missing numbers are in Bangla. "
                    "Ensure LLM preserves Bangla numerals in output."
                )

            if any('.' in num for num in missing_numbers):
                recommendations.append(
                    "[HINT] Some missing numbers are decimals (GPAs/fees). "
                    "Check if LLM is rounding or dropping decimal points."
                )

        if hallucinated_numbers:
            recommendations.append(
                f"[CRITICAL] {len(hallucinated_numbers)} numbers in KG are not in source. "
                f"This indicates LLM hallucination or inference."
            )

            recommendations.append(
                "[ACTION] Review extraction prompt to prevent speculation. "
                "Use stricter constraints (ONLY extract what is EXPLICITLY mentioned)."
            )

        if not missing_numbers and not hallucinated_numbers:
            recommendations.append(
                "[SUCCESS] Perfect numeric accuracy achieved. All numbers validated."
            )

        return recommendations

    def _analyze_frequency(
        self,
        source_numbers: Dict,
        kg_numbers: Dict
    ) -> Dict:
        """
        Analyze number frequency distribution.

        Useful for debugging: if a number appears 5 times in source but only 2 times in KG,
        extraction is likely incomplete.
        """

        source_freq = {num: len(contexts) for num, contexts in source_numbers.items()}
        kg_freq = {num: len(sources) for num, sources in kg_numbers.items()}

        # Find numbers with frequency mismatch
        frequency_mismatches = []
        for num in source_freq:
            source_count = source_freq[num]
            kg_count = kg_freq.get(num, 0)

            if kg_count < source_count:
                frequency_mismatches.append({
                    'number': num,
                    'source_frequency': source_count,
                    'kg_frequency': kg_count,
                    'deficit': source_count - kg_count
                })

        return {
            'source_frequency': source_freq,
            'kg_frequency': kg_freq,
            'frequency_mismatches': frequency_mismatches
        }

    def _compute_chunk_statistics(self, chunk_validations: List[Dict]) -> Dict:
        """Compute aggregate statistics across chunks."""

        total_chunks = len(chunk_validations)
        passed_chunks = sum(1 for cv in chunk_validations if cv['status'] == 'PASS')
        failed_chunks = sum(1 for cv in chunk_validations if cv['status'] == 'FAIL')
        missing_extractions = sum(
            1 for cv in chunk_validations if cv['status'] == 'MISSING_EXTRACTION'
        )

        coverages = [
            cv['coverage'] for cv in chunk_validations
            if cv['status'] != 'MISSING_EXTRACTION'
        ]
        hallucination_rates = [
            cv['hallucination_rate'] for cv in chunk_validations
            if cv['status'] != 'MISSING_EXTRACTION'
        ]

        return {
            'total_chunks': total_chunks,
            'passed_chunks': passed_chunks,
            'failed_chunks': failed_chunks,
            'missing_extractions': missing_extractions,
            'pass_rate': passed_chunks / total_chunks if total_chunks > 0 else 0.0,
            'avg_coverage': sum(coverages) / len(coverages) if coverages else 0.0,
            'avg_hallucination_rate': (
                sum(hallucination_rates) / len(hallucination_rates)
                if hallucination_rates else 0.0
            ),
            'min_coverage': min(coverages) if coverages else 0.0,
            'max_coverage': max(coverages) if coverages else 0.0
        }
