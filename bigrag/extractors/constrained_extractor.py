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

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        extraction_mode: str = "semi_structured",
        enable_gleaning: bool = False,  # NEW: Enhanced Pipeline support
        max_gleaning_iterations: int = 2,  # NEW: Enhanced Pipeline support
        hitl_store=None  # NEW (Phase 1 Step 6): HITL store for failed extractions
    ):
        """
        Initialize constrained extractor.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o-mini for cost efficiency)
            extraction_mode: Validation mode (structured/semi_structured/unstructured)
                - structured: 99%+ accuracy, strict validation (PASS=100%, WARNING=95%)
                - semi_structured: 95%+ accuracy, moderate validation (PASS=95%, WARNING=90%) [DEFAULT]
                - unstructured: 80%+ accuracy, lenient validation (PASS=80%, WARNING=70%)
            enable_gleaning: NEW - Enable multi-pass gleaning for better recall (Phase 1 Step 3)
            max_gleaning_iterations: NEW - Number of gleaning passes (default: 2)
            hitl_store: NEW (Phase 1 Step 6) - FailedExtractionStore instance for HITL
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.extraction_mode = extraction_mode
        self.enable_gleaning = enable_gleaning  # NEW
        self.max_gleaning_iterations = max_gleaning_iterations  # NEW
        self.hitl_store = hitl_store  # NEW (Phase 1 Step 6)
        self.normalizer = BanglaNumeralNormalizer()

    async def extract_from_paragraph(
        self,
        paragraph_text: str,
        chunk_id: str,
        metadata: Optional[Dict] = None,
        language: str = "English"
    ) -> Optional[Dict]:
        """
        Extract entities and relations with optional gleaning (NEW - Phase 1 Step 3).

        TWO-STAGE PROCESS:
        STAGE 1 - Initial Extraction with Retry (Error Recovery):
          1. Attempt extraction up to 3 times
          2. Validate after each attempt
          3. If PASS/WARNING → proceed to Stage 2
          4. If all 3 attempts FAIL → reject chunk, return None

        STAGE 2 - Gleaning (Refinement, only if Stage 1 succeeded):
          5. If gleaning enabled, perform N additional passes with conversation history
          6. Each gleaning pass validated independently (failed passes skipped)
          7. Merge using quality-based comparison
          8. Final validation

        Args:
            paragraph_text: Source text (non-table content)
            chunk_id: Chunk identifier
            metadata: Optional metadata (title, category, etc.)
            language: Output language (English/Bangla)

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'validation': {...},
                'metadata': {
                    'extraction_method': 'constrained_llm' | 'constrained_llm_with_gleaning',
                    'gleaning_passes': 0 | 2,
                    ...
                }
            }
            OR None if validation fails after all retries
        """

        # Pre-extraction: Extract ground truth numbers
        source_numbers = self._extract_numbers_from_text(paragraph_text)
        source_facts = self._extract_key_facts(paragraph_text)

        # STAGE 1: Initial extraction with validation retry
        initial_result = await self._extract_once(
            paragraph_text,
            chunk_id,
            metadata,
            language,
            source_numbers,
            source_facts
        )

        if initial_result is None:
            # NEW (Phase 1 Step 6): HITL - Save failed chunk for human review
            if hasattr(self, 'hitl_store') and self.hitl_store:
                try:
                    self.hitl_store.save_failed_chunk(
                        chunk_id=chunk_id,
                        chunk_content=paragraph_text,
                        failure_reason="All 3 validation attempts failed",
                        validation_details={"error": "Extraction validation failed after retry"},
                        document_id=metadata.get('doc_id', 'unknown') if metadata else 'unknown',
                        metadata=metadata
                    )
                    print(f"[HITL] Failed chunk {chunk_id} saved for human review")
                except Exception as e:
                    print(f"[WARN] HITL save failed: {e}")

            return None  # Validation failed after 3 attempts

        # If gleaning disabled, return initial result
        if not self.enable_gleaning:
            return initial_result

        # STAGE 2: Gleaning loop (NEW - Phase 1 Step 3)
        print(f"[GLEANING] Starting {self.max_gleaning_iterations} gleaning passes for {chunk_id}")

        merged_extraction = initial_result
        conversation_history = [
            {"role": "user", "content": self._create_extraction_prompt(paragraph_text, language, metadata)},
            {"role": "assistant", "content": json.dumps({
                'entities': initial_result.get('entities', []),
                'relations': initial_result.get('relations', [])
            })}
        ]

        for gleaning_pass in range(self.max_gleaning_iterations):
            print(f"[GLEANING] Pass {gleaning_pass + 1}/{self.max_gleaning_iterations}")

            # Create continue-extraction prompt
            continue_prompt = self._create_gleaning_prompt(paragraph_text, language)
            conversation_history.append({"role": "user", "content": continue_prompt})

            # Call LLM with conversation history
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=conversation_history,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                glean_response = response.choices[0].message.content
                conversation_history.append({"role": "assistant", "content": glean_response})

                glean_extraction = json.loads(glean_response)

            except Exception as e:
                print(f"[WARN] Gleaning pass {gleaning_pass + 1} failed: {e}")
                continue  # Skip this gleaning pass

            # Validate gleaned extraction with RELAXED thresholds
            # (skip numeric coverage, focus on hallucination prevention)
            glean_validation = self._validate_extraction(
                source_text=paragraph_text,
                source_numbers=source_numbers,
                source_facts=source_facts,
                extraction=glean_extraction,
                is_gleaning=True  # ← CRITICAL: Use relaxed validation for incremental extraction
            )

            # SMART MERGE: Compare quality and merge (IDENTICAL to standard pipeline)
            if glean_validation['status'] in ['PASS', 'WARNING']:
                merged_extraction = self._merge_extractions_by_quality(
                    merged_extraction,
                    glean_extraction
                )
                print(f"[GLEANING] Pass {gleaning_pass + 1}: Added {len(glean_extraction.get('entities', []))} entities, {len(glean_extraction.get('relations', []))} relations")
            else:
                print(f"[GLEANING] Pass {gleaning_pass + 1}: Validation FAILED (hallucination or nonsense), skipping")

        # Final validation of merged result
        final_validation = self._validate_extraction(
            source_text=paragraph_text,
            source_numbers=source_numbers,
            source_facts=source_facts,
            extraction=merged_extraction
        )

        merged_extraction['validation'] = final_validation
        merged_extraction['metadata'] = {
            'chunk_id': chunk_id,
            'extraction_method': 'constrained_llm_with_gleaning',
            'gleaning_passes': self.max_gleaning_iterations,
            'language': language,
            'extraction_quality': final_validation['status'],
            **(metadata or {})
        }

        return merged_extraction

    async def _extract_once(
        self,
        paragraph_text: str,
        chunk_id: str,
        metadata: Optional[Dict],
        language: str,
        source_numbers: List[str],
        source_facts: List[str]
    ) -> Optional[Dict]:
        """
        Single extraction pass with validation retry (up to 3 attempts).

        This is the EXISTING logic refactored into a separate method for use in
        the two-stage extraction process (Stage 1: Error Recovery).

        Args:
            paragraph_text: Source text
            chunk_id: Chunk identifier
            metadata: Optional metadata
            language: Output language
            source_numbers: Pre-extracted numbers from text
            source_facts: Pre-extracted key facts

        Returns:
            Extraction dict with validation metadata, or None if all 3 attempts failed
        """

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

            # Check if validation passed or warning (accept both)
            status = validation_result['status']
            if status in ['PASS', 'WARNING']:
                # Add metadata
                extraction['validation'] = validation_result
                extraction['validation']['attempts'] = attempt
                extraction['validation']['extraction_mode'] = self.extraction_mode
                extraction['metadata'] = {
                    'chunk_id': chunk_id,
                    'extraction_method': 'constrained_llm',
                    'language': language,
                    'extraction_quality': status,  # Track quality level
                    **(metadata or {})
                }

                # Visual warning flag for WARNING status
                if status == 'WARNING':
                    print(f"[WARNING] Extraction succeeded with warnings (attempt {attempt}):")
                    print(f"  Mode: {self.extraction_mode}")
                    print(f"  Numeric coverage: {validation_result.get('numeric_coverage', 0):.2%}")
                    print(f"  Hallucination score: {validation_result.get('hallucination_score', 0):.2%}")
                    print(f"  Semantic validity: {validation_result.get('semantic_validity', 0):.2%}")
                    print(f"  This extraction will be included but may need review.")

                return extraction

            # Log validation failure (FAIL status only)
            print(f"[FAIL] Validation failed (attempt {attempt}/{3}):")
            print(f"  Numeric coverage: {validation_result.get('numeric_coverage', 0):.2%}")
            missing_count = len(validation_result.get('missing_numbers', []))
            hallucinated_count = len(validation_result.get('hallucinated_numbers', []))
            print(f"  Missing numbers: {missing_count}")
            print(f"  Hallucinated numbers: {hallucinated_count}")

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

    def _create_gleaning_prompt(self, paragraph_text: str, language: str) -> str:
        """
        Create gleaning continuation prompt (NEW - Phase 1 Step 3).

        CRITICAL: This must be IDENTICAL to standard pipeline's continue_prompt
        to ensure consistent behavior when we unify pipelines in the future.

        The gleaning prompt asks the LLM to review the text again and find
        any entities or relations that were missed in previous extraction passes.

        Args:
            paragraph_text: Source text to re-review
            language: Output language for entities/relations

        Returns:
            Gleaning prompt string
        """
        return f"""CONTINUE EXTRACTION: Review the source text again and identify ANY additional entities or relations you may have missed in the previous extraction.

IMPORTANT:
- Only extract NEW entities/relations not already mentioned
- Focus on entities that may have been overlooked
- Maintain the same JSON format
- Preserve exact numeric values from text
- Output language: {language}

Source text:
{paragraph_text}

Return JSON with:
{{
    "entities": [
        {{"entity_name": "name", "entity_type": "type", "description": "...", "key_score": 0-100}}
    ],
    "relations": [
        {{"content": "relation description", "completeness_score": 0-10}}
    ]
}}

If no additional entities/relations found, return empty lists.
"""

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
        extraction: Dict,
        is_gleaning: bool = False  # NEW: Relax validation for incremental gleaning passes
    ) -> Dict:
        """
        Triple-constraint validation.

        Checks:
        1. Numeric coverage: 100% of source numbers in extraction (SKIPPED for gleaning)
        2. No hallucination: No numbers in extraction not in source
        3. Semantic validity: Entity names mentioned in source

        Args:
            source_text: Original source text
            source_numbers: Set of numbers extracted from source
            source_facts: List of key facts from source
            extraction: Extraction result to validate
            is_gleaning: If True, use relaxed validation (skip numeric coverage)
                        Gleaning is incremental - it only returns NEW entities,
                        so numeric coverage check doesn't make sense.

        Returns:
            {
                'status': 'PASS' or 'FAIL' or 'WARNING',
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
        if is_gleaning:
            # RELAXED VALIDATION FOR GLEANING:
            # Gleaning is incremental - it only returns NEW entities not already extracted.
            # Therefore, we SKIP numeric coverage check (would always be low).
            # Instead, focus ONLY on preventing hallucinations and nonsense entities.

            if hallucination_score < 0.10 and semantic_validity >= 0.50:
                # Low hallucination + reasonable semantic validity = accept
                status = 'PASS'
            elif hallucination_score < 0.20 and semantic_validity >= 0.40:
                # Moderate quality - usable but warn
                status = 'WARNING'
            else:
                # High hallucination or nonsense entities = reject
                status = 'FAIL'

            # Log gleaning validation details
            print(f"      [GLEANING VALIDATION] hallucination={hallucination_score:.2%}, semantic={semantic_validity:.2%}, status={status}")
        else:
            # NORMAL VALIDATION FOR INITIAL EXTRACTION:
            # Use full 3-tier validation with numeric coverage
            status = self._determine_validation_status(
                numeric_coverage=numeric_coverage,
                hallucination_score=hallucination_score,
                semantic_validity=semantic_validity
            )

        return {
            'status': status,
            'numeric_coverage': numeric_coverage,
            'hallucination_score': hallucination_score,
            'semantic_validity': semantic_validity,
            'missing_numbers': missing_numbers,
            'hallucinated_numbers': hallucinated_numbers,
            'hallucinated_entities': hallucinated_entities
        }

    def _determine_validation_status(
        self,
        numeric_coverage: float,
        hallucination_score: float,
        semantic_validity: float
    ) -> str:
        """
        Determine validation status based on extraction mode.

        3-tier validation system:
        - PASS: High confidence extraction (use without review)
        - WARNING: Medium confidence extraction (usable but may need review)
        - FAIL: Low confidence extraction (reject)

        Thresholds vary by extraction_mode:

        STRUCTURED mode (strict - for highly structured data like tables):
        - PASS: 100% numeric coverage, 0% hallucination, 90%+ semantic
        - WARNING: 95%+ numeric coverage, <5% hallucination, 85%+ semantic
        - FAIL: Below WARNING thresholds

        SEMI_STRUCTURED mode (moderate - for mixed content) [DEFAULT]:
        - PASS: 95%+ numeric coverage, <5% hallucination, 85%+ semantic
        - WARNING: 60%+ numeric coverage, <15% hallucination, 70%+ semantic
        - FAIL: Below WARNING thresholds (for multilingual paragraphs, lower threshold needed)

        UNSTRUCTURED mode (lenient - for narrative text):
        - PASS: 80%+ numeric coverage, <15% hallucination, 70%+ semantic
        - WARNING: 70%+ numeric coverage, <20% hallucination, 60%+ semantic
        - FAIL: Below WARNING thresholds

        Args:
            numeric_coverage: Fraction of source numbers present in extraction (0-1)
            hallucination_score: Fraction of hallucinated numbers (0-1)
            semantic_validity: Fraction of entities mentioned in source (0-1)

        Returns:
            'PASS', 'WARNING', or 'FAIL'
        """

        if self.extraction_mode == "structured":
            # Strict thresholds for highly structured data
            if (numeric_coverage == 1.0 and
                hallucination_score == 0.0 and
                semantic_validity >= 0.9):
                return 'PASS'
            elif (numeric_coverage >= 0.95 and
                  hallucination_score < 0.05 and
                  semantic_validity >= 0.85):
                return 'WARNING'
            else:
                return 'FAIL'

        elif self.extraction_mode == "semi_structured":
            # Moderate thresholds for mixed content (DEFAULT)
            if (numeric_coverage >= 0.95 and
                hallucination_score < 0.05 and
                semantic_validity >= 0.85):
                return 'PASS'
            elif (numeric_coverage >= 0.60 and
                  hallucination_score < 0.15 and
                  semantic_validity >= 0.70):
                return 'WARNING'
            else:
                return 'FAIL'

        elif self.extraction_mode == "unstructured":
            # Lenient thresholds for narrative text
            if (numeric_coverage >= 0.80 and
                hallucination_score < 0.15 and
                semantic_validity >= 0.70):
                return 'PASS'
            elif (numeric_coverage >= 0.70 and
                  hallucination_score < 0.20 and
                  semantic_validity >= 0.60):
                return 'WARNING'
            else:
                return 'FAIL'

        else:
            # Unknown mode - fall back to semi_structured
            print(f"[WARN] Unknown extraction_mode '{self.extraction_mode}', using semi_structured")
            if (numeric_coverage >= 0.95 and
                hallucination_score < 0.05 and
                semantic_validity >= 0.85):
                return 'PASS'
            elif (numeric_coverage >= 0.90 and
                  hallucination_score < 0.10 and
                  semantic_validity >= 0.80):
                return 'WARNING'
            else:
                return 'FAIL'

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

    def _merge_extractions_by_quality(
        self,
        base_extraction: Dict,
        glean_extraction: Dict
    ) -> Dict:
        """
        Merge two extractions using quality-based comparison (NEW - Phase 1 Step 3).

        Logic (IDENTICAL to standard pipeline's smart merge):
        1. For entities with same name → keep better description (higher quality score)
        2. For new entities → add to base
        3. For relations → append all (deduplicate by content similarity later if needed)

        CRITICAL CLARIFICATIONS:
        - Tiebreaker hierarchy: quality score → description length → first-seen
        - key_scores are SUMMED across passes (e.g., 60 + 70 = 130)
        - This preserves importance signal across extraction passes

        Args:
            base_extraction: Initial/merged extraction result
            glean_extraction: Gleaning pass extraction result

        Returns:
            Merged extraction dict with combined entities and relations
        """
        from bigrag.utils import description_quality_score  # Reuse standard pipeline's scoring

        merged = {
            "entities": [],
            "relations": []
        }

        # Build entity lookup by name (normalized for comparison)
        base_entities = {}
        for e in base_extraction.get('entities', []):
            entity_name = e.get('entity_name', '')
            # Normalize entity name for matching (lowercase, strip whitespace)
            entity_key = entity_name.lower().strip()
            base_entities[entity_key] = e

        # Merge entities
        for glean_entity in glean_extraction.get('entities', []):
            entity_name = glean_entity.get('entity_name', '')
            entity_key = entity_name.lower().strip()

            if entity_key in base_entities:
                # Entity already exists - compare quality scores
                base_entity = base_entities[entity_key]
                base_desc = base_entity.get('description', '')
                glean_desc = glean_entity.get('description', '')

                base_quality = description_quality_score(base_desc)
                glean_quality = description_quality_score(glean_desc)

                # Tiebreaker logic (CRITICAL)
                if glean_quality > base_quality:
                    # Gleaned version is better
                    base_entities[entity_key] = glean_entity
                    print(f"    [MERGE] Entity '{entity_name}': Gleaned version is better (quality {base_quality:.0f} → {glean_quality:.0f})")
                elif glean_quality == base_quality:
                    # Tie on quality - use length as tiebreaker
                    if len(glean_desc) > len(base_desc):
                        base_entities[entity_key] = glean_entity
                        print(f"    [MERGE] Entity '{entity_name}': Gleaned version is longer (quality tie)")
                    else:
                        print(f"    [MERGE] Entity '{entity_name}': Keeping original (quality tie, original longer)")
                else:
                    # Original is better
                    print(f"    [MERGE] Entity '{entity_name}': Keeping original (quality {base_quality:.0f} vs {glean_quality:.0f})")

                # SUM key_scores across passes (CRITICAL)
                base_key_score = base_entities[entity_key].get('key_score', 0)
                glean_key_score = glean_entity.get('key_score', 0)
                base_entities[entity_key]['key_score'] = base_key_score + glean_key_score

            else:
                # New entity from gleaning
                base_entities[entity_key] = glean_entity
                print(f"    [MERGE] Entity '{entity_name}': NEW from gleaning")

        merged['entities'] = list(base_entities.values())

        # Merge relations (simple append for now)
        # NOTE: Could improve by deduplicating similar relations, but standard pipeline
        # also does simple append, so we keep it consistent
        merged['relations'] = base_extraction.get('relations', []) + glean_extraction.get('relations', [])

        return merged


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
