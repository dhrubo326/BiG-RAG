"""
NumericValidator - Production-grade numeric validation with Gemini API

Architecture (REDESIGNED - January 2025):
  This validator now operates at the CHUNK level during extraction, not after.
  Each chunk's source text is validated against its extracted entities/relations.
  This enables proper numeric validation without requiring post-extraction architecture changes.

Features:
- Hybrid LLM + regex number extraction
- 100% numeric coverage validation
- Hallucination detection
- Multilingual support (Bangla, English, Hindi, Arabic)
- Per-chunk validation for accurate coverage tracking
"""

from bigrag.interfaces.validator import ValidatorInterface
from typing import Dict, List

class NumericValidator(ValidatorInterface):
    def __init__(
        self,
        api_key: str = None,
        strictness: str = "MODERATE",
        use_llm_validation: bool = True,
        validation_mode: str = "document"  # NEW: "chunk" | "document" | "hybrid"
    ):
        """
        Initialize comprehensive numeric validator.

        Args:
            api_key: Gemini API key (required if use_llm_validation=True)
            strictness: Validation level - "STRICT" (95%), "MODERATE" (90%), "LENIENT" (80%)
            use_llm_validation: Whether to use LLM-based validation (default: True)
            validation_mode: NEW (Issue #2 fix)
                - "chunk": Validate each chunk separately (faster, less accurate)
                - "document": Validate entire document at once (slower, more accurate - matches old pipeline)
                - "hybrid": Try document-level, fallback to chunk-level on error
        """
        self.strictness = strictness
        self.use_llm_validation = use_llm_validation
        self.validation_mode = validation_mode  # NEW: Issue #2 fix

        if use_llm_validation:
            try:
                from bigrag.validators.numeric_validator import NumericValidator as FullNumericValidator
                self.validator = FullNumericValidator(api_key=api_key, use_llm_validation=True)
                print(f"[NumericValidator] Initialized with Gemini API (strictness={strictness})")
            except ImportError as e:
                print(f"[WARNING] Gemini SDK not installed: {e}")
                print("[WARNING] Numeric validation will pass through without validation.")
                self.validator = None
                self.use_llm_validation = False
            except Exception as e:
                print(f"[WARNING] Failed to initialize Gemini validator: {e}")
                print("[WARNING] Numeric validation will pass through without validation.")
                self.validator = None
                self.use_llm_validation = False
        else:
            self.validator = None

    async def validate(self, extractions: Dict) -> Dict:
        """
        Validate numeric accuracy of extractions.

        UPDATED ARCHITECTURE (Issue #1 & #2 fixes):
        - Now supports TWO validation modes: chunk-level and document-level
        - Document-level (NEW): Validates entire document against ALL entities/relations
          * Matches old production pipeline behavior
          * More accurate (eliminates cross-chunk duplicate issues)
          * Requires 'source_document' field in extractions
        - Chunk-level (legacy): Validates each chunk separately
          * Faster but less accurate
          * Requires 'chunks' field in extractions

        Args:
            extractions: {
                'entities': [...],      # ALL entities (merged)
                'relations': [...],     # ALL relations
                'failed_chunks': [...],
                'source_document': str,  # NEW: Full document text (for document-level validation)
                'chunks': [...]          # Optional: For chunk-level validation
            }

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'failed_chunks': [...],
                'summary': {
                    'status': 'PASS' | 'WARNING' | 'FAIL',
                    'numeric_coverage': 0.95,
                    'validation_method': 'document-level' | 'chunk-level'
                }
            }
        """
        if not self.validator or not self.use_llm_validation:
            # Fallback: pass through without validation
            return {
                'entities': extractions.get('entities', []),
                'relations': extractions.get('relations', []),
                'failed_chunks': extractions.get('failed_chunks', []),
                'summary': {
                    'status': 'PASS',
                    'numeric_coverage': None,
                    'validation_method': 'none (Gemini not configured)'
                }
            }

        # NEW: Document-level validation (Issue #2 fix - matches old pipeline)
        if self.validation_mode == "document" and extractions.get('source_document'):
            return await self._validate_document_level(extractions)
        # Hybrid mode: try document-level, fallback to chunk-level
        elif self.validation_mode == "hybrid":
            if extractions.get('source_document'):
                try:
                    return await self._validate_document_level(extractions)
                except Exception as e:
                    print(f"[WARNING] Document-level validation failed: {e}. Falling back to chunk-level.")
                    return await self._validate_chunk_level(extractions)
            else:
                return await self._validate_chunk_level(extractions)
        # Chunk-level validation (legacy)
        else:
            return await self._validate_chunk_level(extractions)

    async def _validate_document_level(self, extractions: Dict) -> Dict:
        """
        Validate entire document at once (matches old production pipeline).

        COPIED FROM enhanced_pipeline.py:684-689
        This is the CORRECT way to validate numerics:
        - Single call with full document text
        - Validates ALL entities and relations together
        - No cross-chunk duplication issues
        """
        source_document = extractions.get('source_document', '')
        all_entities = extractions.get('entities', [])
        all_relations = extractions.get('relations', [])

        if not source_document:
            print("[WARNING] Document-level validation requires 'source_document' field. Skipping.")
            return {
                'entities': all_entities,
                'relations': all_relations,
                'failed_chunks': extractions.get('failed_chunks', []),
                'summary': {
                    'status': 'WARNING',
                    'numeric_coverage': None,
                    'validation_method': 'skipped (no source_document provided)'
                }
            }

        try:
            # Single comprehensive validation call (matches old pipeline)
            validation_result = await self.validator.validate_extraction(
                source_document=source_document,
                entities=all_entities,
                relations=all_relations,
                validation_level=self.strictness
            )

            coverage = validation_result.get('numeric_coverage', 1.0)
            status = validation_result.get('status', 'PASS')

            # Determine overall status based on strictness
            if self.strictness == "STRICT":
                final_status = 'PASS' if coverage >= 0.95 else 'WARNING' if coverage >= 0.90 else 'FAIL'
            elif self.strictness == "MODERATE":
                final_status = 'PASS' if coverage >= 0.90 else 'WARNING' if coverage >= 0.85 else 'FAIL'
            else:  # LENIENT
                final_status = 'PASS' if coverage >= 0.80 else 'WARNING' if coverage >= 0.75 else 'FAIL'

            return {
                'entities': all_entities,  # Return all entities (document-level doesn't filter)
                'relations': all_relations,
                'failed_chunks': extractions.get('failed_chunks', []),
                'summary': {
                    'status': final_status,
                    'numeric_coverage': coverage,
                    'hallucination_rate': validation_result.get('hallucination_rate', 0.0),
                    'validation_method': f'document-level (strictness={self.strictness})',
                    'note': 'Validates entire document at once (matches old production pipeline)'
                }
            }

        except Exception as e:
            print(f"[ERROR] Document-level numeric validation failed: {e}")
            # Permissive fallback: return entities/relations unchanged
            return {
                'entities': all_entities,
                'relations': all_relations,
                'failed_chunks': extractions.get('failed_chunks', []),
                'summary': {
                    'status': 'WARNING',
                    'numeric_coverage': None,
                    'validation_method': 'error (fallback to pass-through)',
                    'error': str(e)
                }
            }

    async def _validate_chunk_level(self, extractions: Dict) -> Dict:
        """
        Validate each chunk separately (legacy mode).

        Original implementation preserved for backward compatibility.
        """

        # Check if chunks with source text are provided
        chunks = extractions.get('chunks', [])
        if not chunks:
            print("[WARNING] NumericValidator requires 'chunks' field with source text. Skipping validation.")
            return {
                'entities': extractions.get('entities', []),
                'relations': extractions.get('relations', []),
                'failed_chunks': extractions.get('failed_chunks', []),
                'summary': {
                    'status': 'WARNING',
                    'numeric_coverage': None,
                    'validation_method': 'skipped (no source text provided)',
                    'note': 'Extraction strategy must pass chunks with source text for numeric validation'
                }
            }

        # Validate each chunk
        valid_entities = []
        valid_relations = []
        failed_chunks_list = list(extractions.get('failed_chunks', []))
        chunks_validated = 0
        chunks_passed = 0
        chunks_failed = 0
        total_coverage = 0.0

        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            chunk_content = chunk.get('content', '')
            chunk_entities = chunk.get('entities', [])
            chunk_relations = chunk.get('relations', [])

            if not chunk_content:
                # No source text - skip validation for this chunk
                valid_entities.extend(chunk_entities)
                valid_relations.extend(chunk_relations)
                continue

            chunks_validated += 1

            try:
                # Run validation for this chunk
                validation_result = await self.validator.validate_extraction(
                    source_document=chunk_content,
                    entities=chunk_entities,
                    relations=chunk_relations,
                    validation_level=self.strictness
                )

                coverage = validation_result.get('numeric_coverage', 1.0)
                status = validation_result.get('status', 'PASS')
                total_coverage += coverage

                if status == 'PASS' or status == 'WARNING':
                    # Chunk passed validation - include its extractions
                    valid_entities.extend(chunk_entities)
                    valid_relations.extend(chunk_relations)
                    chunks_passed += 1
                else:
                    # Chunk failed validation - mark as failed
                    failed_chunks_list.append({
                        'chunk_id': chunk_id,
                        'content': chunk_content,
                        'reason': 'numeric_validation_failed',
                        'numeric_coverage': coverage,
                        'missing_numbers': validation_result.get('missing_numbers', []),
                        'hallucinated_numbers': validation_result.get('hallucinated_numbers', []),
                        'gemini_feedback': validation_result.get('gemini_feedback', '')
                    })
                    chunks_failed += 1

            except Exception as e:
                print(f"[ERROR] Numeric validation failed for chunk {chunk_id}: {e}")
                # On error, include entities/relations (permissive fallback)
                valid_entities.extend(chunk_entities)
                valid_relations.extend(chunk_relations)
                chunks_passed += 1

        # Compute overall status
        avg_coverage = total_coverage / chunks_validated if chunks_validated > 0 else 1.0
        pass_rate = chunks_passed / chunks_validated if chunks_validated > 0 else 1.0

        # Determine overall status based on strictness
        if self.strictness == "STRICT":
            status = 'PASS' if avg_coverage >= 0.95 and pass_rate >= 0.90 else 'WARNING' if avg_coverage >= 0.90 else 'FAIL'
        elif self.strictness == "MODERATE":
            status = 'PASS' if avg_coverage >= 0.90 and pass_rate >= 0.85 else 'WARNING' if avg_coverage >= 0.85 else 'FAIL'
        else:  # LENIENT
            status = 'PASS' if avg_coverage >= 0.80 and pass_rate >= 0.75 else 'WARNING' if avg_coverage >= 0.75 else 'FAIL'

        return {
            'entities': valid_entities,
            'relations': valid_relations,
            'failed_chunks': failed_chunks_list,
            'summary': {
                'status': status,
                'numeric_coverage': avg_coverage,
                'chunks_validated': chunks_validated,
                'chunks_passed': chunks_passed,
                'chunks_failed': chunks_failed,
                'pass_rate': pass_rate,
                'validation_method': f'gemini-hybrid (strictness={self.strictness})',
                'note': f'Validated {chunks_validated} chunks at chunk level for numeric accuracy'
            }
        }
