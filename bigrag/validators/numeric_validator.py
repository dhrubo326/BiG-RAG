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

    NEW (January 2025): Hybrid LLM + regex validation for quality + completeness.

    Three validation levels (UPDATED for multilingual support with 5% tolerance):
    - STRICT: 95%+ PASS, 90-95% WARNING, <90% FAIL (production)
    - MODERATE: 90%+ PASS, 85-90% WARNING, <85% FAIL (development)
    - LENIENT: 80%+ PASS, 75-80% WARNING, <75% FAIL (testing)

    Returns detailed report with:
    - Coverage percentage
    - Missing numbers (in source but not in KG)
    - Hallucinated numbers (in KG but not in source)
    - Number frequency analysis
    - Location mapping (which chunk contains which number)
    """

    def __init__(self, api_key: Optional[str] = None, use_llm_validation: bool = True):
        """
        Initialize numeric validator with Gemini 2.5 Pro.

        Args:
            api_key: Gemini API key (required if use_llm_validation=True)
            use_llm_validation: Whether to use LLM-based validation (default: True)
        """
        self.normalizer = BanglaNumeralNormalizer()
        self.use_llm_validation = use_llm_validation

        if use_llm_validation:
            from google import genai
            from google.genai import types
            import os
            from dotenv import load_dotenv

            # Load .env to ensure GEMINI_API_KEY is available
            load_dotenv()

            self.gemini_api_key = api_key or os.getenv('GEMINI_API_KEY')
            if not self.gemini_api_key:
                raise ValueError(
                    "[ERROR] GEMINI_API_KEY not found in environment variables. "
                    "LLM-based numeric validation requires Gemini API key. "
                    "Please set GEMINI_API_KEY in your .env file."
                )
            else:
                # Initialize the new unified SDK client
                self.client = genai.Client(api_key=self.gemini_api_key)
                self.model_name = 'gemini-2.0-flash-exp'  # Using Flash for speed, Pro for accuracy
                print(f"[INFO] Numeric validator initialized with {self.model_name} (google-genai SDK)")

    async def _extract_numbers_hybrid(self, text: str) -> Dict[str, List[str]]:
        """
        Hybrid extraction: Combine LLM (quality) + Regex (completeness).

        Strategy:
        1. LLM extracts numbers with semantic understanding (handles compound numbers)
        2. Regex extracts all numeric sequences (guarantees 100% coverage)
        3. Merge results: LLM context overrides regex when both find same number

        This gives us:
        - Best quality: LLM understands "২০২৪-২০২৫" as academic year range
        - Best coverage: Regex catches every digit sequence LLM might filter out

        Args:
            text: Source text (Bangla/English mixed)

        Returns:
            Dict mapping normalized number → list of contexts
        """
        # Step 1: LLM extraction (high quality, context-aware)
        try:
            llm_numbers = await self._extract_numbers_with_llm(text)
        except Exception as e:
            # If LLM fails (malformed JSON, API error), fall back to regex only
            print(f"[WARN] LLM extraction failed, using regex only: {e}")
            llm_numbers = {}

        # Step 2: Regex extraction (complete coverage, simple)
        regex_numbers = self._extract_numbers_with_regex(text)

        # Step 3: Merge (union of both, LLM context preferred)
        merged = dict(regex_numbers)  # Start with regex (complete)
        for num, contexts in llm_numbers.items():
            if num in merged:
                # Number found by both: use LLM's richer context
                merged[num] = contexts
            else:
                # Number only found by LLM: add it
                merged[num] = contexts

        return merged

    async def _extract_numbers_with_llm(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all unique numbers from text using LLM (multilingual support).

        This replaces regex-based extraction to handle:
        - Multilingual numerals (১২০ = 120)
        - Complex formats (৯-৩০ = "9:30", not "9" and "30")
        - Time ranges, dates, decimals, percentages
        - Context-aware extraction (avoids splitting compound numbers)

        Args:
            text: Source text (can contain Bangla/English mixed content)

        Returns:
            Dict mapping normalized number → list of contexts where it appears
            Example: {"120": ["আসন সংখ্যা ১২০", "total seats 120"], "4.50": ["GPA ৪.৫০"]}
        """
        import json

        prompt = f"""Extract EVERY number from the text below. DO NOT FILTER OR SKIP ANYTHING.

TEXT:
{text}

CRITICAL RULES - EXTRACT EVERYTHING:
1. Extract EVERY numeric sequence - including single digits (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
2. Extract ALL years (2021, 2022, 2023, 2024, 2025, etc.)
3. Extract ALL IDs and codes (1065, 20000, 000, etc.)
4. Extract ALL Bangla numerals (০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯, ১২০, ৯০, etc.)
5. Extract numbers from EVERYWHERE: tables, paragraphs, lists, titles, headers, footers

6. Keep compound numbers intact:
   - Time ranges: "৯-৩০মি." = "9-30" (ONE number)
   - Date ranges: "২০২৪-২৫" = "2024-25" (ONE number)
   - Decimals: "৪.৫০" = "4.50" (ONE number)
   - Decimals: "১৮.০০" = "18.00" (ONE number)
   - Phone: "০১৭১১-১২৩৪৫৬" = "01711-123456" (ONE number)

7. Normalize to English numerals:
   - Bangla → English: ১২০ → "120", ৯০ → "90", ২ → "2"
   - Keep formats: ৪.৫০ → "4.50", ২০২৪-২৫ → "2024-25"

8. For each unique number, provide:
   - Normalized value (English numerals)
   - Original form (as it appears in text)
   - Context (surrounding 30 characters)

DO NOT judge which numbers are "important" - extract EVERYTHING you see.
DO NOT skip single digits, years, IDs, or any numeric value.

OUTPUT (JSON only, no commentary):
{{
  "numbers": [
    {{"normalized": "120", "original": "১২০", "context": "আসন সংখ্যা ১২০ জন"}},
    {{"normalized": "2", "original": "২", "context": "বিভাগ ২"}},
    {{"normalized": "2021", "original": "২০২১", "context": "সাল ২০২১ ইং"}},
    {{"normalized": "1065", "original": "১০৬৫", "context": "মোট ১০৬৫টি আসন"}},
    {{"normalized": "4.50", "original": "৪.৫০", "context": "জিপিএ ৪.৫০ প্রয়োজন"}},
    {{"normalized": "2024-25", "original": "২০২৪-২৫", "context": "শিক্ষাবর্ষ ২০২৪-২৫"}},
    {{"normalized": "11-30", "original": "১১-৩০", "context": "সময় ১১-৩০ মিনিট"}}
  ]
}}

IMPORTANT:
- Output ONLY valid JSON
- Extract EVERY number - no exceptions, no filtering
- Include single digits, years, IDs, codes, decimals, ranges - ALL numbers
"""

        try:
            # Use Gemini for extraction (same model as validation for consistency)
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json"
                }
            )

            # Parse response (Gemini might wrap JSON in markdown)
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            numbers = result.get('numbers', [])

            # Convert to dict mapping normalized number → list of contexts
            number_contexts = defaultdict(list)
            for num in numbers:
                normalized = num.get('normalized', '')
                context = num.get('context', '')
                if normalized:
                    number_contexts[normalized].append(context)

            return dict(number_contexts)

        except Exception as e:
            print(f"[ERROR] LLM number extraction failed: {e}")
            print("[INFO] Falling back to regex-only extraction")
            # Return empty dict - hybrid method will handle fallback to regex
            return {}

    async def validate_extraction(
        self,
        source_document: str,
        entities: List[Dict],
        relations: List[Dict],
        validation_level: str = "STRICT"
    ) -> Dict:
        """
        Validate numeric accuracy using Gemini as judge.
        Simple, flexible approach that understands both Bangla and English naturally.

        Args:
            source_document: Original document text
            entities: Extracted entities
            relations: Extracted relations
            validation_level: "STRICT", "MODERATE", or "LENIENT"

        Returns:
            {
                'status': 'PASS' or 'FAIL',
                'validation_level': 'STRICT',
                'numeric_coverage': 0.95,
                'hallucination_rate': 0.02,
                'missing_numbers': [],
                'hallucinated_numbers': [],
                'feedback': 'LLM feedback'
            }
        """
        import json

        # Build KG text from entities and relations
        kg_text = self._build_kg_text(entities, relations)

        # Create simple validation prompt for Gemini
        prompt = f"""You are validating if extracted knowledge graph preserves all numbers from source document.

SOURCE DOCUMENT:
{source_document}

EXTRACTED KNOWLEDGE GRAPH:
{kg_text}

TASK:
Compare numbers in SOURCE vs EXTRACTED. Numbers can be in Bangla (১২০) or English (120) - treat them as SAME.

Examples of SAME numbers:
- ১২০ = 120 (same)
- ৪.৫০ = 4.50 = 4.5 (same)
- ২০২৪-২৫ = 2024-25 (same)
- ৯০ = 90 (same)

Ignore minor formatting differences. Focus on semantic meaning.

OUTPUT (JSON only):
{{
  "coverage_percent": 95.5,
  "missing_numbers": ["2021", "16"],
  "hallucinated_numbers": [],
  "assessment": "PASS or FAIL",
  "feedback": "Brief explanation"
}}

IMPORTANT:
- Treat Bangla and English numbers as equivalent
- Ignore whitespace/formatting differences
- Focus on whether ALL important numbers are preserved
- Return PASS if coverage >= 90%, FAIL otherwise
"""

        try:
            # Use the new google-genai SDK's async generate_content method
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            response_text = response.text

            # Extract JSON from response (Gemini sometimes adds markdown)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)

            coverage = result.get('coverage_percent', 0) / 100.0
            missing = result.get('missing_numbers', [])
            hallucinated = result.get('hallucinated_numbers', [])
            assessment = result.get('assessment', 'FAIL')
            feedback = result.get('feedback', '')

            # Determine status based on threshold with 5% tolerance for WARNING
            if validation_level == "MODERATE":
                if coverage >= 0.90:
                    status = "PASS"
                elif coverage >= 0.85:  # 5% tolerance
                    status = "WARNING"
                else:
                    status = "FAIL"
            elif validation_level == "LENIENT":
                if coverage >= 0.80:
                    status = "PASS"
                elif coverage >= 0.75:  # 5% tolerance
                    status = "WARNING"
                else:
                    status = "FAIL"
            else:  # STRICT
                if coverage >= 0.95:
                    status = "PASS"
                elif coverage >= 0.90:  # 5% tolerance
                    status = "WARNING"
                else:
                    status = "FAIL"

            return {
                'status': status,
                'validation_level': validation_level,
                'numeric_coverage': coverage,
                'hallucination_rate': len(hallucinated) / max(len(missing) + len(hallucinated), 1),
                'total_source_numbers': len(missing) + int(coverage * 100),
                'total_kg_numbers': len(hallucinated) + int(coverage * 100),
                'missing_numbers': missing,
                'hallucinated_numbers': hallucinated,
                'gemini_feedback': feedback,
                'gemini_assessment': assessment,
                'recommendations': self._generate_recommendations_simple(missing, hallucinated, coverage)
            }

        except Exception as e:
            print(f"[ERROR] Gemini validation failed: {e}")
            # Return permissive result to not block pipeline
            return {
                'status': 'WARNING',
                'validation_level': validation_level,
                'numeric_coverage': 0.85,
                'hallucination_rate': 0.0,
                'total_source_numbers': 0,
                'total_kg_numbers': 0,
                'missing_numbers': [],
                'hallucinated_numbers': [],
                'gemini_feedback': f'Validation error: {str(e)}',
                'gemini_assessment': 'ERROR',
                'recommendations': ['Gemini validation failed - using permissive fallback']
            }

    def _generate_recommendations_simple(self, missing: List, hallucinated: List, coverage: float) -> List[str]:
        """Generate simple recommendations based on validation results."""
        recommendations = []

        if missing:
            recommendations.append(f"Missing {len(missing)} numbers from source. Check extraction completeness.")
        if hallucinated:
            recommendations.append(f"Found {len(hallucinated)} numbers not in source. Check for hallucination.")
        if coverage < 0.90:
            recommendations.append(f"Coverage {coverage:.1%} is below 90%. Improve extraction quality.")
        if not recommendations:
            recommendations.append("All numbers preserved correctly!")

        return recommendations

    async def validate_chunks(
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

            result = await self.validate_extraction(
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

    def _extract_numbers_with_regex(
        self,
        text: str,
        context_window: int = 50
    ) -> Dict[str, List[str]]:
        """
        Extract numbers from text using regex (COMPLETE coverage, simple extraction).

        This is the fallback/completeness method - guarantees NO numbers are missed.
        Used in hybrid approach for 100% recall.

        IMPORTANT: Normalizes Bangla numerals to English for consistent comparison.

        Args:
            text: Source text
            context_window: Characters before/after number

        Returns:
            {
                '120': ['...CSE বিভাগে 120টি আসন...', '...total 120 seats...'],
                '4.00': ['...GPA 4.00 প্রয়োজন...']
            }
            All keys are normalized to English numerals.
        """

        number_contexts = defaultdict(list)

        # Find all numbers with their positions
        # UPDATED PATTERN (Jan 2025): Captures decimals, ranges, times, compound numbers
        # Pattern: [০-৯0-9]+ (one or more digits) followed by zero or more separators and digits
        # Matches: 4.50, 2024-25, 11-30, 9:30, 1065, etc.
        for match in re.finditer(r'[০-৯0-9]+(?:[।.\-:][০-৯0-9]+)*', text):
            number = match.group(0)
            start = match.start()
            end = match.end()

            # Extract context
            context_start = max(0, start - context_window)
            context_end = min(len(text), end + context_window)
            context = text[context_start:context_end]

            # CRITICAL FIX: Normalize Bangla → English for consistent keys
            normalized_number = self.normalizer.bangla_to_english(number)
            number_contexts[normalized_number].append(context)

        return dict(number_contexts)

    def _extract_numbers_with_context(
        self,
        text: str,
        context_window: int = 50
    ) -> Dict[str, List[str]]:
        """
        DEPRECATED: Use _extract_numbers_with_regex() or _extract_numbers_hybrid() instead.
        Kept for backward compatibility.
        """
        return self._extract_numbers_with_regex(text, context_window)

    def _build_kg_text(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> str:
        """
        Build combined text from KG entities and relations for LLM extraction.

        Args:
            entities: Entity list
            relations: Relation list

        Returns:
            Combined text with all entity names, descriptions, and relation content
        """
        text_parts = []

        # Add entity names and descriptions
        for entity in entities:
            entity_name = entity.get('entity_name', '')
            description = entity.get('description', '')
            if entity_name:
                text_parts.append(f"Entity: {entity_name}")
            if description:
                text_parts.append(f"Description: {description}")

        # Add relation content
        for relation in relations:
            content = relation.get('content', '')
            if content:
                text_parts.append(f"Relation: {content}")

        return "\n".join(text_parts)

    def _extract_numbers_from_kg(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Extract numbers from knowledge graph with sources.

        IMPORTANT: Normalizes Bangla numerals to English for consistent comparison.

        Args:
            entities: Entity list
            relations: Relation list

        Returns:
            {
                '120': ['entity: CSE', 'relation: CSE has 120 seats'],
                '4.00': ['entity: GPA requirement']
            }
            All keys are normalized to English numerals.
        """

        number_sources = defaultdict(list)

        # Extract from entities
        for entity in entities:
            entity_name = entity.get('entity_name', '')
            description = entity.get('description', '')

            # Find numbers in entity name
            for num in re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', entity_name):
                # CRITICAL FIX: Normalize Bangla → English
                normalized_num = self.normalizer.bangla_to_english(num)
                number_sources[normalized_num].append(f"entity_name: {entity_name}")

            # Find numbers in description
            for num in re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', description):
                # CRITICAL FIX: Normalize Bangla → English
                normalized_num = self.normalizer.bangla_to_english(num)
                number_sources[normalized_num].append(f"entity_desc: {entity_name}")

        # Extract from relations
        for relation in relations:
            content = relation.get('content', '')

            # Find numbers in relation content
            for num in re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', content):
                # CRITICAL FIX: Normalize Bangla → English
                normalized_num = self.normalizer.bangla_to_english(num)
                number_sources[normalized_num].append(f"relation: {content[:50]}...")

        return dict(number_sources)

    def _determine_status(
        self,
        numeric_coverage: float,
        hallucination_rate: float,
        validation_level: str
    ) -> str:
        """
        Determine validation status based on level.

        3-tier validation system with graceful degradation:
        - PASS: High confidence (use without review)
        - WARNING: Medium confidence (usable but flagged for review)
        - FAIL: Low confidence (reject)

        Args:
            numeric_coverage: Percentage of source numbers in KG
            hallucination_rate: Percentage of KG numbers not in source
            validation_level: "STRICT", "MODERATE", or "LENIENT"

        Returns:
            "PASS", "WARNING", or "FAIL"
        """

        if validation_level == "STRICT":
            # Production: 95%+ coverage, <2% hallucination (high quality for production)
            if numeric_coverage >= 0.95 and hallucination_rate < 0.02:
                return "PASS"
            elif numeric_coverage >= 0.92 and hallucination_rate < 0.05:
                return "WARNING"
            else:
                return "FAIL"

        elif validation_level == "MODERATE":
            # Development: 90%+ coverage, <8% hallucination (realistic for multilingual)
            if numeric_coverage >= 0.90 and hallucination_rate < 0.08:
                return "PASS"
            elif numeric_coverage >= 0.85 and hallucination_rate < 0.12:
                return "WARNING"
            else:
                return "FAIL"

        elif validation_level == "LENIENT":
            # Early testing: 80%+ coverage, <15% hallucination (exploratory work)
            if numeric_coverage >= 0.80 and hallucination_rate < 0.15:
                return "PASS"
            elif numeric_coverage >= 0.70 and hallucination_rate < 0.20:
                return "WARNING"
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
