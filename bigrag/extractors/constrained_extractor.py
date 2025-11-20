"""
Constrained LLM Paragraph Extractor for Production Knowledge Graph

Triple-constraint approach for 99%+ accuracy on non-table content:
1. MUST-EXTRACT validation (every number in source must appear in output)
2. NO-HALLUCINATION validation (no numbers/facts not in source)
3. SEMANTIC validation (entities must be mentioned in source text)

Designed for educational admission documents with zero tolerance for errors.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI

from bigrag.bangla_utils import BanglaNumeralNormalizer
from bigrag.error_recovery import ExtractionErrorHandler


class ConstrainedLLMExtractor:
    """
    LLM entity/relation extraction with strict validation.

    Key innovation: Triple-constraint validation
    - Pre-extraction: Extract all numbers/facts from source
    - Post-extraction: Verify 100% of numbers appear in output
    - Retry on validation failure (up to 3 attempts)

    If validation fails after retries → REJECT extraction
    Better to have NO extraction than WRONG extraction.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize constrained extractor.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o-mini for cost efficiency)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.normalizer = BanglaNumeralNormalizer()

    async def extract_from_paragraph(
        self,
        paragraph_text: str,
        chunk_id: str,
        metadata: Optional[Dict] = None,
        language: str = "English"
    ) -> Optional[Dict]:
        """
        Extract entities and relations from paragraph text.

        Args:
            paragraph_text: Source text (non-table content)
            chunk_id: Chunk identifier
            metadata: Optional metadata (title, category, etc.)
            language: Output language (English/Bangla)

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'validation': {
                    'status': 'PASS',
                    'numeric_coverage': 1.0,
                    'hallucination_score': 0.0,
                    'attempts': 1
                }
            }
            OR None if validation fails after all retries
        """

        # Pre-extraction: Extract ground truth numbers
        source_numbers = self._extract_numbers_from_text(paragraph_text)
        source_facts = self._extract_key_facts(paragraph_text)

        # Attempt extraction with validation (up to 3 tries)
        for attempt in range(1, 4):
            # Create extraction prompt
            prompt = self._create_extraction_prompt(
                paragraph_text,
                language,
                metadata
            )

            # Call LLM
            try:
                llm_response = await self._call_llm(prompt)
            except Exception as e:
                print(f"[ERROR] LLM call failed (attempt {attempt}): {e}")
                if attempt == 3:
                    return None
                continue

            # Parse response
            try:
                extraction = json.loads(llm_response)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse LLM response (attempt {attempt}): {e}")
                if attempt == 3:
                    return None
                continue

            # Validate extraction
            validation_result = self._validate_extraction(
                source_text=paragraph_text,
                source_numbers=source_numbers,
                source_facts=source_facts,
                extraction=extraction
            )

            # Check if validation passed
            if validation_result['status'] == 'PASS':
                # Add metadata
                extraction['validation'] = validation_result
                extraction['validation']['attempts'] = attempt
                extraction['metadata'] = {
                    'chunk_id': chunk_id,
                    'extraction_method': 'constrained_llm',
                    'language': language,
                    **(metadata or {})
                }

                return extraction

            # Log validation failure
            print(f"[WARN] Validation failed (attempt {attempt}/{3}):")
            print(f"  Numeric coverage: {validation_result.get('numeric_coverage', 0):.2%}")
            print(f"  Missing numbers: {validation_result.get('missing_numbers', [])}")
            print(f"  Hallucinated numbers: {validation_result.get('hallucinated_numbers', [])}")

            # If last attempt, return None
            if attempt == 3:
                print(f"[ERROR] Extraction rejected after {attempt} attempts (validation failed)")
                return None

        return None

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with error recovery."""
        async def extract():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        return await ExtractionErrorHandler.retry_with_backoff(
            extract,
            max_retries=3,
            base_delay=2.0,
            max_delay=10.0
        )

    def _create_extraction_prompt(
        self,
        paragraph_text: str,
        language: str,
        metadata: Optional[Dict]
    ) -> str:
        """
        Create extraction prompt with strict constraints.

        CRITICAL RULES:
        1. Extract ONLY entities/relations mentioned in text
        2. Preserve EXACT numeric values (no conversion, no rounding)
        3. No speculation or inference beyond text
        4. Mark completeness_score honestly (0-10)
        """

        context_info = ""
        if metadata:
            if 'title' in metadata:
                context_info += f"\nDocument Title: {metadata['title']}"
            if 'category' in metadata:
                context_info += f"\nCategory: {metadata['category']}"

        prompt = f"""You are an ULTRA-PRECISE entity and relation extractor for academic admission documents.

TASK: Extract entities and knowledge segments from this paragraph.

CRITICAL CONSTRAINTS (ZERO TOLERANCE):
1. Extract ONLY what is EXPLICITLY mentioned in text (no inference)
2. Preserve EXACT numeric values:
   - "১২০ জন" → entity_name: "১২০"
   - "৪.০০ GPA" → entity_name: "৪.০০"
   - DO NOT convert Bangla to English
   - DO NOT round or approximate
3. Output language: {language}
4. Completeness score (0-10):
   - 10: Complete, self-contained knowledge
   - 7-9: Mostly complete, minor context missing
   - 5-6: Partial information
   - 0-4: Fragment or unclear

{context_info}

INPUT TEXT:
{paragraph_text}

OUTPUT FORMAT (JSON only):
{{
  "entities": [
    {{
      "entity_name": "কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং",
      "entity_type": "department",
      "description": "একটি প্রকৌশল বিভাগ",
      "key_score": 90
    }}
  ],
  "relations": [
    {{
      "content": "কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগে ১২০টি আসন রয়েছে।",
      "completeness_score": 10
    }}
  ]
}}

ENTITY TYPES (educational domain):
- department, faculty, university
- department_code, seat_count
- gpa_requirement, eligibility
- fee, deadline, time, event, location
- person, organization, concept

IMPORTANT:
- Output ONLY valid JSON (no commentary)
- Include EVERY number mentioned in text
- Do NOT add information not in text
- Mark completeness honestly
"""
        return prompt

    def _extract_numbers_from_text(self, text: str) -> set:
        """
        Extract all numbers from text (Bangla + English).

        Returns set of number strings for validation.
        """
        # Match Bangla and English numbers (integers and decimals)
        numbers = re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', text)
        return set(numbers)

    def _extract_key_facts(self, text: str) -> List[str]:
        """
        Extract key factual statements for validation.

        Simple heuristic: sentences with numbers are key facts.
        """
        # Split into sentences
        sentences = re.split(r'[।.\n]', text)

        # Keep sentences with numbers
        key_facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and re.search(r'[০-৯0-9]', sentence):
                key_facts.append(sentence)

        return key_facts

    def _validate_extraction(
        self,
        source_text: str,
        source_numbers: set,
        source_facts: List[str],
        extraction: Dict
    ) -> Dict:
        """
        Triple-constraint validation.

        Checks:
        1. Numeric coverage: 100% of source numbers in extraction
        2. No hallucination: No numbers in extraction not in source
        3. Semantic validity: Entity names mentioned in source

        Returns:
            {
                'status': 'PASS' or 'FAIL',
                'numeric_coverage': 1.0,
                'hallucination_score': 0.0,
                'semantic_validity': 1.0,
                'missing_numbers': [],
                'hallucinated_numbers': [],
                'hallucinated_entities': []
            }
        """

        # Extract numbers from extraction
        extraction_numbers = set()
        entities = extraction.get('entities', [])
        relations = extraction.get('relations', [])

        for entity in entities:
            nums = re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', entity.get('entity_name', ''))
            extraction_numbers.update(nums)

            desc_nums = re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', entity.get('description', ''))
            extraction_numbers.update(nums)

        for relation in relations:
            nums = re.findall(r'[০-৯0-9]+(?:\.[০-৯0-9]+)?', relation.get('content', ''))
            extraction_numbers.update(nums)

        # CRITICAL FIX: Normalize numbers to English for comparison
        # LLM may output Bangla numerals even when asked for English
        source_normalized = {self.normalizer.bangla_to_english(n): n for n in source_numbers}
        extraction_normalized = {self.normalizer.bangla_to_english(n): n for n in extraction_numbers}

        # Check 1: Numeric coverage (must be 100%)
        if source_normalized:
            matched_normalized = set(source_normalized.keys()) & set(extraction_normalized.keys())
            numeric_coverage = len(matched_normalized) / len(source_normalized)
            missing_normalized = set(source_normalized.keys()) - set(extraction_normalized.keys())
            missing_numbers = [source_normalized[n] for n in missing_normalized]
        else:
            numeric_coverage = 1.0
            missing_numbers = []

        # Check 2: No hallucination
        hallucinated_normalized = set(extraction_normalized.keys()) - set(source_normalized.keys())
        hallucinated_numbers = [extraction_normalized[n] for n in hallucinated_normalized]
        hallucination_score = len(hallucinated_numbers) / max(len(extraction_numbers), 1)

        # Check 3: Semantic validity (entity names must appear in source)
        hallucinated_entities = []
        for entity in entities:
            entity_name = entity.get('entity_name', '')
            # Normalize for comparison
            entity_normalized = self.normalizer.normalize_for_comparison(entity_name)
            source_normalized = self.normalizer.normalize_for_comparison(source_text)

            # Check if entity name appears in source (fuzzy match for numbers)
            if not self._is_mentioned_in_text(entity_normalized, source_normalized):
                # For numbers, be more lenient (they might be normalized)
                if not re.match(r'^[0-9]+(?:\.[0-9]+)?$', entity_normalized):
                    hallucinated_entities.append(entity_name)

        semantic_validity = 1.0 - (len(hallucinated_entities) / max(len(entities), 1))

        # Determine overall status
        # PASS requires:
        # - 100% numeric coverage (strict)
        # - 0% number hallucination (strict)
        # - 90%+ semantic validity (slightly lenient for name variations)
        if (numeric_coverage == 1.0 and
            hallucination_score == 0.0 and
            semantic_validity >= 0.9):
            status = 'PASS'
        else:
            status = 'FAIL'

        return {
            'status': status,
            'numeric_coverage': numeric_coverage,
            'hallucination_score': hallucination_score,
            'semantic_validity': semantic_validity,
            'missing_numbers': missing_numbers,
            'hallucinated_numbers': hallucinated_numbers,
            'hallucinated_entities': hallucinated_entities
        }

    def _is_mentioned_in_text(self, entity: str, text: str) -> bool:
        """
        Check if entity is mentioned in text (case-insensitive, normalized).

        Args:
            entity: Entity name (normalized)
            text: Source text (normalized)

        Returns:
            True if entity appears in text
        """
        # Simple substring check (case-insensitive)
        entity_lower = entity.lower()
        text_lower = text.lower()

        # Check exact match
        if entity_lower in text_lower:
            return True

        # Check partial match for multi-word entities (at least 50% words match)
        entity_words = entity_lower.split()
        if len(entity_words) > 1:
            matched_words = sum(1 for word in entity_words if word in text_lower)
            if matched_words / len(entity_words) >= 0.5:
                return True

        return False


class BatchConstrainedExtractor:
    """
    Process multiple paragraph chunks in batch with validation tracking.

    Aggregates validation statistics across chunks.
    """

    def __init__(self, extractor: ConstrainedLLMExtractor):
        """
        Initialize batch extractor.

        Args:
            extractor: ConstrainedLLMExtractor instance
        """
        self.extractor = extractor

    async def extract_from_chunks(
        self,
        chunks: List[Dict],
        language: str = "English"
    ) -> Dict:
        """
        Extract from multiple chunks with validation tracking.

        Args:
            chunks: List of chunk dicts (from TableAwareChunker)
            language: Output language

        Returns:
            {
                'extractions': [
                    {
                        'chunk_id': 'chunk_001',
                        'entities': [...],
                        'relations': [...],
                        'validation': {...}
                    }
                ],
                'statistics': {
                    'total_chunks': 10,
                    'successful_extractions': 9,
                    'failed_extractions': 1,
                    'avg_numeric_coverage': 1.0,
                    'avg_attempts': 1.2
                }
            }
        """

        extractions = []
        failed_chunks = []

        for chunk in chunks:
            # Skip table chunks (handled separately)
            if chunk.get('type') == 'table':
                continue

            chunk_id = chunk['chunk_id']
            content = chunk['content']
            metadata = chunk.get('metadata', {})

            # Extract with validation
            result = await self.extractor.extract_from_paragraph(
                paragraph_text=content,
                chunk_id=chunk_id,
                metadata=metadata,
                language=language
            )

            if result:
                extractions.append({
                    'chunk_id': chunk_id,
                    'entities': result.get('entities', []),
                    'relations': result.get('relations', []),
                    'validation': result.get('validation', {}),
                    'metadata': result.get('metadata', {})
                })
            else:
                failed_chunks.append(chunk_id)
                print(f"[ERROR] Extraction failed for chunk {chunk_id}")

        # Compute statistics
        stats = self._compute_statistics(extractions, failed_chunks, len(chunks))

        return {
            'extractions': extractions,
            'statistics': stats,
            'failed_chunks': failed_chunks
        }

    def _compute_statistics(
        self,
        extractions: List[Dict],
        failed_chunks: List[str],
        total_chunks: int
    ) -> Dict:
        """Compute validation statistics across all chunks."""

        if not extractions:
            return {
                'total_chunks': total_chunks,
                'successful_extractions': 0,
                'failed_extractions': len(failed_chunks),
                'success_rate': 0.0,
                'avg_numeric_coverage': 0.0,
                'avg_hallucination_score': 0.0,
                'avg_semantic_validity': 0.0,
                'avg_attempts': 0.0
            }

        # Extract validation metrics
        numeric_coverages = []
        hallucination_scores = []
        semantic_validities = []
        attempts = []

        for ext in extractions:
            val = ext.get('validation', {})
            numeric_coverages.append(val.get('numeric_coverage', 0.0))
            hallucination_scores.append(val.get('hallucination_score', 0.0))
            semantic_validities.append(val.get('semantic_validity', 0.0))
            attempts.append(val.get('attempts', 1))

        return {
            'total_chunks': total_chunks,
            'successful_extractions': len(extractions),
            'failed_extractions': len(failed_chunks),
            'success_rate': len(extractions) / total_chunks if total_chunks > 0 else 0.0,
            'avg_numeric_coverage': sum(numeric_coverages) / len(numeric_coverages),
            'avg_hallucination_score': sum(hallucination_scores) / len(hallucination_scores),
            'avg_semantic_validity': sum(semantic_validities) / len(semantic_validities),
            'avg_attempts': sum(attempts) / len(attempts)
        }
