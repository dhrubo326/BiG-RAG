"""
SemanticChunker - Table-aware semantic chunking strategy.

COPIED FROM: bigrag/preprocessors/smart_chunker.py::TableAwareChunker
This implementation uses the PROVEN semantic chunking logic from the old production pipeline.

Key features (all copied from tested production code):
- Table detection with GPT-4 (preserves table structure)
- Semantic boundary chunking (respects paragraphs)
- 3-case accumulation logic (under/overflow/hard-limit)
- Tolerance factor (1.3x overflow for coherence)
- Asymmetric overlap (position-dependent)
- Bilingual sentence detection (English + Bengali)
- Sentence-level fallback for large paragraphs
"""

from bigrag.interfaces.chunker import ChunkerInterface
from typing import List, Dict, Optional
import hashlib
import re
import logging

logger = logging.getLogger(__name__)


class SemanticChunker(ChunkerInterface):
    """
    Table-aware semantic chunking (slow, accurate).

    Copied from TableAwareChunker (smart_chunker.py) - tested in production.
    """

    def __init__(self, api_key: str, chunk_size: int = 1200, overlap: int = 100, enable_table_detection: bool = True):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_table_detection = enable_table_detection

        # Import table extractor (only if enabled)
        if self.enable_table_detection:
            # CRITICAL: Fail-fast if table detection enabled but dependencies missing
            if not api_key:
                raise ValueError(
                    "[SemanticChunker] enable_table_detection=True requires api_key. "
                    "Either provide OpenAI API key OR set enable_table_detection=False."
                )

            try:
                from bigrag.preprocessors.table_extractor import GPT4TableExtractor
            except ImportError as e:
                raise ImportError(
                    "[SemanticChunker] GPT4TableExtractor not available. "
                    "Install required dependencies OR set enable_table_detection=False. "
                    f"Original error: {e}"
                ) from e

            # Initialize table extractor (will raise error if initialization fails)
            self.table_extractor = GPT4TableExtractor(api_key=api_key)
        else:
            self.table_extractor = None

    async def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk document with table detection and semantic boundaries.

        COPIED FROM: smart_chunker.py::TableAwareChunker.chunk_document()

        Algorithm:
        1. Extract tables first (if enabled)
        2. Remove tables from text
        3. Chunk remaining text with semantic boundaries (proven logic)
        4. Add table chunks
        5. Return combined chunks

        Args:
            text: Document content
            metadata: Optional metadata

        Returns:
            List of chunk dicts with chunk_method='semantic' tracking
        """
        logger.info(f"[SemanticChunker] Starting chunk process for document (length: {len(text)} chars)")

        chunks = []

        # Step 1: Detect tables if extractor available
        if self.table_extractor:
            # CRITICAL: No try/except - fail-fast if table extraction fails
            # User must fix the issue (API key, rate limits, document format, etc.)

            # Import language detector
            from bigrag.preprocessors.table_extractor import BilingualDetector

            # Extract tables (will raise exception if fails - NO silent fallback)
            tables = await self.table_extractor.extract_tables_from_document(
                text,
                document_metadata=metadata
            )

            logger.info(f"[SemanticChunker] Detected {len(tables)} tables")

            # Add table chunks (using same approach as TableAwareChunker)
            for table in tables:
                # Convert table to natural language (CRITICAL for embedding quality)
                # NOTE: Using local method (copied from smart_chunker.py) - no dependency on archived code
                nl_content = SemanticChunker._table_to_natural_language(table)

                chunk_id = hashlib.md5(nl_content.encode()).hexdigest()[:16]

                # Detect language in table content
                lang_info = BilingualDetector.detect_languages(nl_content)

                chunks.append({
                    'chunk_id': f'chunk-{chunk_id}',
                    'type': 'table',
                    'content': nl_content,
                    'structured_data': table,
                    'metadata': {
                        **(metadata or {}),
                        'language_info': lang_info,
                        'chunk_method': 'semantic'  # Track method
                    }
                })

            # Remove table markdown from text for paragraph chunking
            table_pattern = r'\|[^\n]+\|(?:\n\|[^\n]+\|)+'
            text = re.sub(table_pattern, '', text)
            logger.info(f"[SemanticChunker] Text length after table removal: {len(text)} chars")

        # Step 2: Chunk remaining text with FULL semantic boundaries logic
        # COPIED FROM smart_chunker.py::_chunk_with_semantic_boundaries()
        text_chunks = self._chunk_with_semantic_boundaries(
            text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            tolerance_factor=1.3  # Same as old system
        )

        # Step 3: Convert text chunks to chunk dicts
        from bigrag.preprocessors.table_extractor import BilingualDetector

        for chunk_text in text_chunks:
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]
            lang_info = BilingualDetector.detect_languages(chunk_text)

            chunks.append({
                'chunk_id': f'chunk-{chunk_id}',
                'type': 'paragraph',
                'content': chunk_text,
                'metadata': {
                    **(metadata or {}),
                    'language_info': lang_info,
                    'chunk_method': 'semantic'  # Track method
                }
            })

        # Log final statistics
        table_count = len([c for c in chunks if c['type'] == 'table'])
        para_count = len([c for c in chunks if c['type'] == 'paragraph'])
        logger.info(f"[SemanticChunker] Final: {len(chunks)} total chunks ({table_count} tables, {para_count} paragraphs)")

        return chunks

    # ============================================
    # COPIED FROM smart_chunker.py (lines 226-343)
    # Full semantic boundaries chunking logic
    # ============================================

    def _chunk_with_semantic_boundaries(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100,
        tolerance_factor: float = 1.3
    ) -> List[str]:
        """
        COPIED FROM smart_chunker.py:226-343

        Chunk text respecting semantic boundaries.

        CRITICAL FEATURES (from production code):
        1. ACCUMULATION DECISION: 3-case logic before adding next paragraph
           - If current + next <= chunk_size: Keep accumulating
           - If chunk_size < current + next <= chunk_size * tolerance:
             * If current >= chunk_size: Flush now (already large enough)
             * If current < chunk_size: Allow overflow (preserve paragraph)
           - If current + next > chunk_size * tolerance: MUST flush (hard limit)

        2. ASYMMETRIC OVERLAP: Position-dependent overlap
           - First chunk: 0 before + overlap after
           - Middle chunks: overlap before + overlap after
           - Last chunk: overlap before + 0 after
           - Single chunk: 0 overlap

        3. BILINGUAL SENTENCE DETECTION: English (. ! ?) + Bengali (।)

        4. SENTENCE-LEVEL FALLBACK: Split large paragraphs at sentence boundaries

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap in tokens on EACH side
            tolerance_factor: Overflow tolerance multiplier (default: 1.3 = 30%)

        Returns:
            List of text chunks with semantic boundaries preserved
        """
        from bigrag.utils import count_tokens_fast, split_by_paragraphs, split_by_sentences, get_overlap_text

        if not text or not text.strip():
            return []

        # Step 1: Split by paragraphs (double newline)
        paragraphs = split_by_paragraphs(text)

        if not paragraphs:
            return []

        # Step 2: Accumulate paragraphs into chunks with semantic boundaries
        chunks = []
        current_chunk_paragraphs = []
        current_tokens = 0

        max_allowed = int(chunk_size * tolerance_factor)  # e.g., 1300 tokens

        for para_idx, para in enumerate(paragraphs):
            para_tokens = count_tokens_fast(para)

            # Case 1: Paragraph fits within tolerance as standalone
            if para_tokens <= max_allowed:
                # Check if adding this paragraph would exceed limits
                new_total = current_tokens + para_tokens

                # CRITICAL: 3-case accumulation decision logic
                if new_total <= chunk_size:
                    # Under target - keep accumulating
                    current_chunk_paragraphs.append(para)
                    current_tokens = new_total

                elif new_total <= max_allowed:
                    # In overflow zone (e.g., 1000-1300)
                    if current_tokens >= chunk_size:
                        # Current already large enough - flush it
                        if current_chunk_paragraphs:
                            chunks.append('\n\n'.join(current_chunk_paragraphs))
                        # Start new chunk with this paragraph
                        current_chunk_paragraphs = [para]
                        current_tokens = para_tokens
                    else:
                        # Current still small - allow overflow for coherence
                        current_chunk_paragraphs.append(para)
                        current_tokens = new_total

                else:
                    # Exceeds hard limit (> 1300) - MUST flush
                    if current_chunk_paragraphs:
                        chunks.append('\n\n'.join(current_chunk_paragraphs))
                    # Start new chunk with this paragraph
                    current_chunk_paragraphs = [para]
                    current_tokens = para_tokens

            else:
                # Case 2: Paragraph too large (> tolerance) - split by sentences
                # First, flush current chunk if any
                if current_chunk_paragraphs:
                    chunks.append('\n\n'.join(current_chunk_paragraphs))
                    current_chunk_paragraphs = []
                    current_tokens = 0

                # Split large paragraph into sentence-based chunks
                sentence_chunks = self._split_paragraph_by_sentences(
                    para,
                    chunk_size=chunk_size,
                    max_allowed=max_allowed
                )
                chunks.extend(sentence_chunks)

        # Flush last chunk
        if current_chunk_paragraphs:
            chunks.append('\n\n'.join(current_chunk_paragraphs))

        # Step 3: Add asymmetric overlap
        chunks_with_overlap = self._add_asymmetric_overlap(chunks, overlap)

        return chunks_with_overlap

    def _split_paragraph_by_sentences(
        self,
        paragraph: str,
        chunk_size: int,
        max_allowed: int
    ) -> List[str]:
        """
        COPIED FROM smart_chunker.py:345-396

        Split a large paragraph into chunks at sentence boundaries.
        Handles both English (. ! ?) and Bengali (।) sentence endings.

        Args:
            paragraph: Large paragraph to split
            chunk_size: Target chunk size in tokens
            max_allowed: Maximum allowed tokens (with tolerance)

        Returns:
            List of sentence-based chunks
        """
        from bigrag.utils import split_by_sentences, count_tokens_fast

        sentences = split_by_sentences(paragraph)

        if not sentences:
            # Fallback: return as-is if no sentence boundaries found
            return [paragraph]

        chunks = []
        current_chunk_sentences = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens_fast(sent)

            if current_tokens + sent_tokens <= max_allowed and current_chunk_sentences:
                # Can add to current chunk
                current_chunk_sentences.append(sent)
                current_tokens += sent_tokens
            else:
                # Need to flush current chunk
                if current_chunk_sentences:
                    chunks.append(' '.join(current_chunk_sentences))

                # Start new chunk with this sentence
                current_chunk_sentences = [sent]
                current_tokens = sent_tokens

        # Flush last chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))

        return chunks

    def _add_asymmetric_overlap(
        self,
        chunks: List[str],
        overlap: int
    ) -> List[str]:
        """
        COPIED FROM smart_chunker.py:398-460

        Add asymmetric overlap to chunks based on position.

        CRITICAL: Overlap depends on chunk position:
        - First chunk: 0 before + overlap after
        - Middle chunks: overlap before + overlap after
        - Last chunk: overlap before + 0 after
        - Single chunk: 0 before + 0 after

        Args:
            chunks: List of chunks without overlap
            overlap: Overlap in tokens on each side

        Returns:
            List of chunks with asymmetric overlap
        """
        from bigrag.utils import get_overlap_text

        if not chunks:
            return []

        if len(chunks) == 1:
            # Single chunk - no overlap
            return chunks

        chunks_with_overlap = []

        for i, chunk in enumerate(chunks):
            overlap_parts = [chunk]  # Start with main chunk content

            if i == 0:
                # First chunk: 0 before + overlap after
                if i + 1 < len(chunks):
                    # Get overlap from next chunk (beginning)
                    next_overlap = get_overlap_text(chunks[i + 1], overlap, direction='start')
                    if next_overlap:
                        overlap_parts.append(f"\n[... {next_overlap}]")

            elif i == len(chunks) - 1:
                # Last chunk: overlap before + 0 after
                prev_overlap = get_overlap_text(chunks[i - 1], overlap, direction='end')
                if prev_overlap:
                    overlap_parts.insert(0, f"[{prev_overlap} ...]\n")

            else:
                # Middle chunk: overlap before + overlap after
                prev_overlap = get_overlap_text(chunks[i - 1], overlap, direction='end')
                if prev_overlap:
                    overlap_parts.insert(0, f"[{prev_overlap} ...]\n")

                if i + 1 < len(chunks):
                    next_overlap = get_overlap_text(chunks[i + 1], overlap, direction='start')
                    if next_overlap:
                        overlap_parts.append(f"\n[... {next_overlap}]")

            chunks_with_overlap.append(''.join(overlap_parts))

        return chunks_with_overlap

    # ============================================
    # COPIED FROM smart_chunker.py (lines 502-627)
    # Table to natural language conversion
    # ============================================

    @staticmethod
    def _table_to_natural_language(table_data: Dict) -> str:
        """
        COPIED FROM smart_chunker.py:502-553

        Convert structured table to natural language.

        This is CRITICAL for embedding quality:
        - Natural language embeds better than raw structured data
        - Preserves semantic meaning for retrieval
        - Makes table content searchable

        Example Input:
        {
            'table_type': 'department_seats',
            'headers': ['বিভাগ/বিষয়', 'কোড', 'আসন'],
            'rows': [
                {'বিভাগ/বিষয়': 'কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং', 'কোড': 'CSE', 'আসন': '১২০'}
            ]
        }

        Example Output:
        "কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগের কোড CSE এবং আসন সংখ্যা ১২০।"
        """
        table_type = table_data.get('table_type', 'general')
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])

        sentences = []

        # Add table header as context
        if headers:
            header_text = f"সারণী: {', '.join(headers)}"
            sentences.append(header_text)

        # Convert each row to natural sentence
        for row in rows:
            if table_type == 'department_seats':
                sentence = SemanticChunker._format_department_row(row)
            elif table_type == 'fee_structure':
                sentence = SemanticChunker._format_fee_row(row)
            elif table_type == 'exam_schedule':
                sentence = SemanticChunker._format_schedule_row(row)
            elif table_type == 'eligibility':
                sentence = SemanticChunker._format_eligibility_row(row)
            else:
                # Generic format
                sentence = SemanticChunker._format_generic_row(row)

            if sentence:
                sentences.append(sentence)

        return '\n'.join(sentences)

    @staticmethod
    def _format_department_row(row: Dict) -> str:
        """
        COPIED FROM smart_chunker.py:555-576

        Format department_seats table row.
        """
        # Try Bangla keys first, then English
        dept = (
            row.get('বিভাগ/বিষয়') or
            row.get('বিভাগ') or
            row.get('Department') or
            row.get('বিষয়') or
            ''
        )
        code = row.get('কোড') or row.get('Code') or ''
        seats = row.get('আসন') or row.get('Seats') or row.get('আসন সংখ্যা') or ''

        if dept and code and seats:
            return f"{dept} বিভাগের কোড {code} এবং আসন সংখ্যা {seats}।"
        elif dept and seats:
            return f"{dept} বিভাগের আসন সংখ্যা {seats}।"
        else:
            # Fallback
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            return ", ".join(parts) + "।"

    @staticmethod
    def _format_fee_row(row: Dict) -> str:
        """
        COPIED FROM smart_chunker.py:578-588

        Format fee_structure table row.
        """
        category = row.get('গ্রুপ') or row.get('Category') or row.get('বিভাগ') or ''
        fee = row.get('ফি') or row.get('Fee') or row.get('Amount') or ''

        if category and fee:
            return f"{category} ভর্তি পরীক্ষার ফি {fee} টাকা।"
        else:
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            return ", ".join(parts) + "।"

    @staticmethod
    def _format_schedule_row(row: Dict) -> str:
        """
        COPIED FROM smart_chunker.py:590-609

        Format exam_schedule table row.
        """
        event = row.get('Event') or row.get('ইভেন্ট') or ''
        date = row.get('Date') or row.get('তারিখ') or ''
        time = row.get('Time') or row.get('সময়') or ''

        parts = []
        if event:
            parts.append(event)
        if date:
            parts.append(f"তারিখ: {date}")
        if time:
            parts.append(f"সময়: {time}")

        if parts:
            return ", ".join(parts) + "।"
        else:
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            return ", ".join(parts) + "।"

    @staticmethod
    def _format_eligibility_row(row: Dict) -> str:
        """
        COPIED FROM smart_chunker.py:611-621

        Format eligibility table row.
        """
        criteria = row.get('Criteria') or row.get('শর্ত') or ''
        requirement = row.get('Requirement') or row.get('প্রয়োজনীয়তা') or ''

        if criteria and requirement:
            return f"{criteria}: {requirement}।"
        else:
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            return ", ".join(parts) + "।"

    @staticmethod
    def _format_generic_row(row: Dict) -> str:
        """
        COPIED FROM smart_chunker.py:623-627

        Generic row formatting (fallback).
        """
        parts = [f"{k}: {v}" for k, v in row.items() if v]
        return ", ".join(parts) + "।"
