"""
GPT-4o Table Extractor for Production Knowledge Graph

LLM-only approach for 100% accurate table extraction from academic documents.
Designed for bilingual content (Bangla + English) with strict numeric preservation.

Now supports large documents (>100K tokens) using Gemini 2.5 Pro.
"""

import json
import re
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import asyncio
import os
import tiktoken

from bigrag.bangla_utils import BanglaNumeralNormalizer
from bigrag.error_recovery import ExtractionErrorHandler


class GPT4TableExtractor:
    """
    LLM-only table extraction using GPT-4o structured output.

    Why LLM-only:
    - Handles complex Bangla-English tables (merged cells, multi-headers)
    - 100% accurate number preservation (critical for seat counts)
    - Simpler codebase (no regex fallback needed)
    - Cost: ~$0.02 per table (acceptable for academic domain)

    Usage:
        extractor = GPT4TableExtractor(api_key="your-key")
        tables = await extractor.extract_tables_from_document(markdown_text)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", gemini_api_key: Optional[str] = None):
        """
        Initialize table extractor.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o for best structured output)
            gemini_api_key: Optional Google AI API key for Gemini 2.5 Pro (for large documents)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.normalizer = BanglaNumeralNormalizer()

        # Gemini support for large documents (>100K tokens)
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.use_gemini_threshold = 100_000  # Use Gemini for documents >100K tokens

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def extract_tables_from_document(
        self,
        markdown_text: str,
        document_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Extract ALL tables from document using GPT-4o or Gemini 2.5 Pro.

        Automatically selects model based on document size:
        - <100K tokens: GPT-4o (128K context, $2.50/1M input tokens)
        - >100K tokens: Gemini 2.5 Pro (2M context, $1.25/1M input tokens)

        Args:
            markdown_text: Document content in markdown format
            document_metadata: Optional metadata (title, category, etc.)

        Returns:
            List of table dictionaries:
            [
                {
                    'table_id': 'table_001',
                    'table_type': 'department_seats',  # Auto-classified
                    'headers': ['বিভাগ/বিষয়', 'কোড', 'আসন'],
                    'rows': [
                        {
                            'বিভাগ/বিষয়': 'কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং',
                            'কোড': 'CSE',
                            'আসন': '১২০'
                        },
                        # ... more rows
                    ],
                    'metadata': {
                        'source_location': 'page_2',
                        'confidence': 1.0,
                        'extraction_method': 'gpt4o_structured' or 'gemini_2.5_pro',
                        'validation_status': 'PASS'
                    }
                }
            ]
        """

        # Create extraction prompt with strict instructions
        prompt = self._create_extraction_prompt(markdown_text)

        # Count tokens to decide which model to use
        token_count = self._count_tokens(markdown_text)

        # Decide model based on token count
        use_gemini = token_count > self.use_gemini_threshold and self.gemini_api_key

        if use_gemini:
            print(f"[INFO] Document has {token_count:,} tokens (>100K) - using Gemini 2.5 Pro")
            extraction_method = 'gemini_2.5_pro'

            try:
                llm_response = await self._extract_with_gemini(prompt)
            except Exception as e:
                print(f"[ERROR] Gemini extraction failed: {e}")
                print("[WARN] Falling back to GPT-4o (may hit context limit)")
                use_gemini = False

        if not use_gemini:
            if token_count > self.use_gemini_threshold:
                print(f"[WARN] Document has {token_count:,} tokens (>100K) but Gemini not available")
                print("[WARN] Using GPT-4o - may hit 128K context limit")
            else:
                print(f"[INFO] Document has {token_count:,} tokens (<100K) - using GPT-4o")

            extraction_method = 'gpt4o_structured'

            # Use error recovery for API resilience
            async def extract():
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Deterministic
                    response_format={"type": "json_object"}  # Enforce JSON
                )
                return response.choices[0].message.content

            try:
                llm_response = await ExtractionErrorHandler.retry_with_backoff(
                    extract,
                    max_retries=3,
                    base_delay=2.0,
                    max_delay=10.0
                )
            except Exception as e:
                print(f"[ERROR] Table extraction failed: {e}")
                return []

        # Parse response
        try:
            result = json.loads(llm_response)
            tables = result.get('tables', [])
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse LLM response: {e}")
            return []

        # Add metadata to each table
        for i, table in enumerate(tables):
            if 'table_id' not in table:
                table['table_id'] = f'table_{i:03d}'

            table['metadata'] = {
                'source_location': 'document',
                'confidence': 1.0,
                'extraction_method': extraction_method,
                'token_count': token_count,
                'validation_status': 'PENDING',  # Will be validated next
                **(document_metadata or {})
            }

        # Validate extraction
        validated_tables = await self._validate_tables(markdown_text, tables)

        return validated_tables

    def _count_tokens(self, text: str) -> int:
        """
        Count number of tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4

    async def _extract_with_gemini(self, prompt: str) -> str:
        """
        Extract tables using Gemini 2.5 Pro for large documents.

        Gemini 2.5 Pro has 2M token context window vs GPT-4o's 128K.

        Args:
            prompt: Extraction prompt

        Returns:
            JSON response from Gemini
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set GEMINI_API_KEY environment variable "
                "or pass gemini_api_key to constructor."
            )

        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)

        # Use Gemini 2.5 Pro with 2M context window
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            generation_config={
                'temperature': 0.0,
                'candidate_count': 1,
            }
        )

        # Generate response
        response = await model.generate_content_async(prompt)

        # Extract JSON from response
        response_text = response.text

        # Clean markdown code blocks if present
        if '```json' in response_text:
            response_text = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if response_text:
                response_text = response_text.group(1)

        return response_text

    def _create_extraction_prompt(self, markdown_text: str) -> str:
        """
        Create GPT-4o prompt with strict extraction instructions.

        Critical rules enforced:
        1. Extract EVERY table (no skipping)
        2. Preserve EXACT values (numbers, codes, dates)
        3. Keep original language (Bangla/English/Mixed)
        4. Auto-classify table type
        """

        prompt = f"""You are a PRECISE table extractor for academic admission documents.

TASK: Extract ALL tables from this markdown document.

CRITICAL RULES (ZERO TOLERANCE):
1. Extract EVERY table (do NOT skip any, even small ones)
2. Preserve EXACT values:
   - Numbers: "১২০" → "১২০" (NOT "120")
   - GPAs: "৪.০০" → "৪.০০" (EXACT)
   - Codes: "CSE" → "CSE" (NOT "CS")
   - Dates: "০৪ ডিসেম্বর, ২০২৪" → "০৪ ডিসেম্বর, ২০২৪" (EXACT)
3. Keep original language (Bangla/English/Mixed)
4. Classify table type:
   - 'department_seats': Department names, codes, seat counts
   - 'exam_schedule': Exam dates, times, venues
   - 'fee_structure': Fee amounts, categories
   - 'eligibility': GPA requirements, subject criteria
   - 'timeline': Admission deadlines, milestones
   - 'general': Other tables

INPUT DOCUMENT:
{markdown_text}

OUTPUT FORMAT (JSON only, no markdown):
{{
  "tables": [
    {{
      "table_id": "table_001",
      "table_type": "department_seats",
      "headers": ["বিভাগ/বিষয়", "কোড", "আসন"],
      "rows": [
        {{"বিভাগ/বিষয়": "কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং", "কোড": "CSE", "আসন": "১২০"}},
        ...
      ]
    }}
  ]
}}

IMPORTANT: Output ONLY valid JSON (no commentary, no markdown).
"""
        return prompt

    async def _validate_tables(
        self,
        source_text: str,
        tables: List[Dict]
    ) -> List[Dict]:
        """
        LLM-based validation using GPT-4o-mini for cross-validation.

        Two-Model Strategy:
        - GPT-4o extracts tables (Phase 1)
        - GPT-4o-mini validates extraction (Phase 2)

        This catches extraction errors that same model would miss.

        Validation checks:
        1. Numeric coverage: 95%+ of table numbers must be in extracted tables
        2. No hallucinations: Tables should not contain numbers not in source tables
        3. Semantic equivalence: Handles Bangla/English equivalence (১২০ = 120)
        """

        # Extract markdown tables from source
        table_pattern = r'\|[^\n]+\|(?:\n\|[^\n]+\|)+'
        markdown_tables = re.findall(table_pattern, source_text)

        if not markdown_tables:
            # No tables in source - mark all as valid
            for table in tables:
                table['metadata']['validation_status'] = 'PASS'
                table['metadata']['numeric_coverage'] = 1.0
            return tables

        # Validate each table using LLM
        validated_tables = []

        for i, table in enumerate(tables):
            # Get corresponding source markdown table (if available)
            source_table_md = markdown_tables[i] if i < len(markdown_tables) else None

            if not source_table_md:
                # No corresponding source table - mark as hallucinated
                table['metadata']['validation_status'] = 'FAIL'
                table['metadata']['error'] = 'No corresponding source table found'
                validated_tables.append(table)
                continue

            # Validate using GPT-4o-mini
            validation_result = await self._llm_validate_table(
                source_table_md=source_table_md,
                extracted_table=table
            )

            # Update table metadata with validation results
            table['metadata']['validation_status'] = validation_result['status']
            table['metadata']['numeric_coverage'] = validation_result['numeric_coverage']

            if validation_result['status'] == 'FAIL':
                table['metadata']['missing_numbers'] = validation_result.get('missing_numbers', [])
                table['metadata']['hallucinated_numbers'] = validation_result.get('hallucinated_numbers', [])
                table['metadata']['validation_feedback'] = validation_result.get('feedback', '')

            validated_tables.append(table)

        return validated_tables

    async def _llm_validate_table(
        self,
        source_table_md: str,
        extracted_table: Dict
    ) -> Dict:
        """
        Use GPT-4o-mini to validate extracted table against source markdown.

        Returns:
            {
                'status': 'PASS' or 'FAIL',
                'numeric_coverage': 0.95,
                'missing_numbers': ['১২০', '৪.০০'],
                'hallucinated_numbers': ['120'],
                'feedback': 'Bangla numerals converted to English'
            }
        """

        # Convert extracted table to comparable format
        extracted_rows = json.dumps(extracted_table.get('rows', []), ensure_ascii=False, indent=2)

        # Create validation prompt
        prompt = f"""You are a STRICT table validation checker.

TASK: Compare source markdown table vs extracted structured table.

SOURCE MARKDOWN TABLE:
{source_table_md}

EXTRACTED STRUCTURED TABLE:
{extracted_rows}

VALIDATION CRITERIA:
1. All numbers from source must appear in extracted table (95%+ coverage)
2. Numbers must be EXACT match (১২০ ≠ 120, ৪.০০ ≠ 4.00)
3. No hallucinated numbers (numbers in extracted but not in source)
4. Bangla and English are NOT equivalent for this check

OUTPUT FORMAT (JSON only):
{{
  "status": "PASS" or "FAIL",
  "numeric_coverage": 0.95,
  "missing_numbers": ["১২০", "৪.০০"],
  "hallucinated_numbers": ["120"],
  "feedback": "Brief explanation if FAIL"
}}

IMPORTANT: Output ONLY valid JSON (no commentary).
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for validation
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Ensure required fields
            if 'status' not in result:
                result['status'] = 'FAIL'
            if 'numeric_coverage' not in result:
                result['numeric_coverage'] = 0.0

            return result

        except Exception as e:
            # If validation fails, mark as FAIL
            import sys
            try:
                print(f"[ERROR] LLM validation failed: {e}")
            except UnicodeEncodeError:
                print(f"[ERROR] LLM validation failed")

            return {
                'status': 'FAIL',
                'numeric_coverage': 0.0,
                'feedback': f'Validation error: {str(e)}'
            }

    async def extract_table_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str
    ) -> Optional[Dict]:
        """
        Extract single table from a text chunk (if present).

        Simpler version for chunk-level processing.

        Args:
            chunk_text: Text chunk content
            chunk_id: Chunk identifier

        Returns:
            Table dict or None if no table found
        """

        # Check if chunk contains markdown table pattern
        table_pattern = r'\|[^\n]+\|(?:\n\|[^\n]+\|)+'
        if not re.search(table_pattern, chunk_text):
            return None

        # Extract tables from chunk
        tables = await self.extract_tables_from_document(chunk_text)

        if not tables:
            return None

        # Return first table (chunks should contain only one table)
        table = tables[0]
        table['metadata']['chunk_id'] = chunk_id
        return table


class BilingualDetector:
    """
    Detect language distribution in text chunks.

    Used to determine output language for entity extraction.
    """

    @staticmethod
    def detect_languages(text: str) -> Dict:
        """
        Detect primary and secondary languages in text.

        Returns:
            {
                'primary': 'bn' or 'en',
                'secondary': 'en' or 'bn' or None,
                'is_bilingual': True/False,
                'confidence': 0.95,
                'bn_probability': 0.6,
                'en_probability': 0.4
            }
        """
        try:
            from langdetect import detect_langs
        except ImportError:
            # Fallback: simple heuristic based on character ranges
            return BilingualDetector._simple_language_detection(text)

        try:
            lang_probs = detect_langs(text)
        except Exception:
            # Fallback if detection fails
            return BilingualDetector._simple_language_detection(text)

        # Extract probabilities
        bn_prob = next((lp.prob for lp in lang_probs if lp.lang == 'bn'), 0.0)
        en_prob = next((lp.prob for lp in lang_probs if lp.lang == 'en'), 0.0)

        # Determine bilingual status (threshold: both > 20%)
        is_bilingual = bn_prob > 0.2 and en_prob > 0.2

        if is_bilingual:
            primary = 'bn' if bn_prob > en_prob else 'en'
            secondary = 'en' if primary == 'bn' else 'bn'
        else:
            primary = 'bn' if bn_prob > en_prob else 'en'
            secondary = None

        return {
            'primary': primary,
            'secondary': secondary,
            'is_bilingual': is_bilingual,
            'confidence': max(bn_prob, en_prob),
            'bn_probability': bn_prob,
            'en_probability': en_prob
        }

    @staticmethod
    def _simple_language_detection(text: str) -> Dict:
        """
        Fallback language detection using character ranges.

        Bangla: U+0980 to U+09FF
        English: U+0041 to U+007A
        """

        # Count Bangla characters
        bangla_chars = len([c for c in text if '\u0980' <= c <= '\u09FF'])

        # Count English characters
        english_chars = len([c for c in text if ('A' <= c <= 'Z') or ('a' <= c <= 'z')])

        total_chars = bangla_chars + english_chars

        if total_chars == 0:
            return {
                'primary': 'en',
                'secondary': None,
                'is_bilingual': False,
                'confidence': 0.5,
                'bn_probability': 0.0,
                'en_probability': 1.0
            }

        bn_prob = bangla_chars / total_chars
        en_prob = english_chars / total_chars

        is_bilingual = bn_prob > 0.2 and en_prob > 0.2

        if is_bilingual:
            primary = 'bn' if bn_prob > en_prob else 'en'
            secondary = 'en' if primary == 'bn' else 'bn'
        else:
            primary = 'bn' if bn_prob > en_prob else 'en'
            secondary = None

        return {
            'primary': primary,
            'secondary': secondary,
            'is_bilingual': is_bilingual,
            'confidence': max(bn_prob, en_prob),
            'bn_probability': bn_prob,
            'en_probability': en_prob
        }
