"""
Table-Aware Smart Chunker for Production Knowledge Graph

Preserves table integrity during document chunking to prevent data loss.
Ensures tables are never split across chunk boundaries.
"""

import re
from typing import List, Dict, Optional, Tuple
from bigrag.preprocessors.table_extractor import GPT4TableExtractor, BilingualDetector

# Simple text chunking function (since split_text_by_token_size doesn't exist in utils)
def split_text_by_token_size(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple character-based chunking (approximation of token chunking).
    For production, replace with proper tiktoken-based chunking.
    """
    # Approximate: 1 token ≈ 4 characters
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4

    chunks = []
    start = 0

    while start < len(text):
        end = start + char_chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - char_overlap

        if start >= len(text):
            break

    return chunks


class TableAwareChunker:
    """
    Chunk documents while preserving table integrity.

    Strategy:
    1. Extract tables FIRST (using GPT4TableExtractor)
    2. Replace tables with placeholders in text
    3. Chunk remaining text normally (1200 tokens, 100 overlap)
    4. Insert table chunks separately (each table = 1 chunk)
    5. Maintain table-to-chunk mapping

    This ensures:
    - Tables are NEVER split across chunks
    - Each table preserves full context
    - Table data can be converted to natural language for embedding
    """

    def __init__(self, table_extractor: GPT4TableExtractor):
        """
        Initialize chunker with table extractor.

        Args:
            table_extractor: GPT4TableExtractor instance for table detection
        """
        self.table_extractor = table_extractor

    async def chunk_document(
        self,
        markdown_text: str,
        chunk_size: int = 1000,  # NEW: Reduced from 1200 to 1000
        overlap: int = 100,
        metadata: Optional[Dict] = None,
        use_semantic_chunking: bool = True,  # NEW: Enable semantic chunking
        debug_chunking: bool = False  # NEW: Debug mode
    ) -> List[Dict]:
        """
        Chunk document with table awareness and semantic boundaries.

        NEW (Phase 1 Step 2): Semantic boundary-aware chunking
        - Respects paragraph boundaries (double newline)
        - Respects sentence boundaries (. ! ? ।)
        - Asymmetric overlap based on chunk position
        - Configurable via use_semantic_chunking parameter

        Args:
            markdown_text: Document content in markdown format
            chunk_size: Target chunk size in tokens (default: 1000, reduced from 1200)
            overlap: Overlap between chunks in tokens (default: 100)
            metadata: Optional document metadata (title, category, tags)
            use_semantic_chunking: NEW - Use semantic boundaries (default: True)
            debug_chunking: NEW - Log chunk boundaries for debugging (default: False)

        Returns:
            List of chunk dictionaries:
            [
                {
                    'chunk_id': 'chunk_001',
                    'type': 'table',  # or 'paragraph'
                    'content': '...',  # Natural language
                    'structured_data': {...},  # Only for tables
                    'metadata': {
                        'title': '...',
                        'category': '...',
                        'language_info': {...}
                        'chunk_method': 'semantic' | 'fixed'  # NEW
                    }
                }
            ]
        """

        # Step 1: Extract all tables
        tables = await self.table_extractor.extract_tables_from_document(
            markdown_text,
            document_metadata=metadata
        )

        # Step 2: Replace tables with placeholders and track positions
        text_with_placeholders, table_positions = self._replace_tables_with_placeholders(
            markdown_text,
            len(tables)
        )

        # Step 3: Chunk non-table text with semantic boundaries
        if use_semantic_chunking:
            # NEW: Semantic boundary-aware chunking
            text_chunks = self._chunk_with_semantic_boundaries(
                text_with_placeholders,
                chunk_size=chunk_size,
                overlap=overlap
            )
            chunk_method = 'semantic'
        else:
            # Fallback: Fixed token-window chunking (old behavior)
            text_chunks = split_text_by_token_size(
                text_with_placeholders,
                chunk_size,
                overlap
            )
            chunk_method = 'fixed'

        if debug_chunking:
            # NEW: Debug logging
            from bigrag.utils import count_tokens_fast
            print(f"\n[DEBUG CHUNKING] Document chunked into {len(text_chunks)} chunks (method: {chunk_method})")
            for i, chunk in enumerate(text_chunks):
                tokens = count_tokens_fast(chunk)
                print(f"  Chunk {i}: {tokens} tokens")
                print(f"    Preview: {chunk[:100]}...")
                print()

        # Step 4: Create chunk objects
        chunks = []
        chunk_id = 0

        for text_chunk in text_chunks:
            # Check if chunk contains table placeholder
            table_matches = re.findall(r'<<<TABLE_(\d+)>>>', text_chunk)

            if table_matches:
                # This chunk contains one or more table placeholders
                for table_idx_str in table_matches:
                    table_idx = int(table_idx_str)

                    if table_idx < len(tables):
                        table_data = tables[table_idx]

                        # Convert table to natural language
                        nl_content = self._table_to_natural_language(table_data)

                        # Detect language
                        lang_info = BilingualDetector.detect_languages(nl_content)

                        chunks.append({
                            'chunk_id': f'chunk_{chunk_id:04d}',
                            'type': 'table',
                            'content': nl_content,
                            'structured_data': table_data,
                            'metadata': {
                                **(metadata or {}),
                                'table_id': table_data['table_id'],
                                'table_type': table_data['table_type'],
                                'extraction_confidence': table_data['metadata']['confidence'],
                                'validation_status': table_data['metadata']['validation_status'],
                                'language_info': lang_info
                            }
                        })

                        chunk_id += 1

                # Also create chunk for non-placeholder text in this chunk
                text_without_placeholders = re.sub(r'<<<TABLE_\d+>>>', '', text_chunk).strip()
                if text_without_placeholders:
                    lang_info = BilingualDetector.detect_languages(text_without_placeholders)
                    chunks.append({
                        'chunk_id': f'chunk_{chunk_id:04d}',
                        'type': 'paragraph',
                        'content': text_without_placeholders,
                        'structured_data': None,
                        'metadata': {
                            **(metadata or {}),
                            'language_info': lang_info,
                            'chunk_method': chunk_method  # NEW: Track chunking method
                        }
                    })
                    chunk_id += 1

            else:
                # Regular text chunk (no tables)
                lang_info = BilingualDetector.detect_languages(text_chunk)

                chunks.append({
                    'chunk_id': f'chunk_{chunk_id:04d}',
                    'type': 'paragraph',
                    'content': text_chunk,
                    'structured_data': None,
                    'metadata': {
                        **(metadata or {}),
                        'language_info': lang_info,
                        'chunk_method': chunk_method  # NEW: Track chunking method
                    }
                })

                chunk_id += 1

        return chunks

    def _chunk_with_semantic_boundaries(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100,
        tolerance_factor: float = 1.3
    ) -> List[str]:
        """
        NEW (Phase 1 Step 2): Chunk text respecting semantic boundaries.

        CRITICAL CLARIFICATIONS (from Technical Spec):

        1. ACCUMULATION DECISION: Decision made BEFORE adding next paragraph
           - If current + next <= chunk_size: Keep accumulating
           - If chunk_size < current + next <= chunk_size * tolerance:
             * If current >= chunk_size: Flush now (already large enough)
             * If current < chunk_size: Allow overflow (preserve paragraph)
           - If current + next > chunk_size * tolerance: MUST flush (hard limit)

        2. ASYMMETRIC OVERLAP: Depends on chunk position
           - First chunk: 0 before + overlap after = overlap total
           - Middle chunks: overlap before + overlap after = overlap * 2 total
           - Last chunk: overlap before + 0 after = overlap total
           - Single chunk: 0 before + 0 after = 0 total

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in tokens (default: 1000)
            overlap: Overlap in tokens on EACH side (default: 100)
            tolerance_factor: Overflow tolerance multiplier (default: 1.3 = 30%)

        Returns:
            List of text chunks with semantic boundaries preserved

        Example:
            Input: 3 paragraphs (800, 400, 900 tokens)
            Output: 2 chunks
                Chunk 1: Para1 (800) + Para2 (400) = 1200 tokens [within tolerance]
                Chunk 2: Para3 (900) + overlap from Para2 (100) = 1000 tokens
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

        max_allowed = int(chunk_size * tolerance_factor)  # 1300 tokens

        for para_idx, para in enumerate(paragraphs):
            para_tokens = count_tokens_fast(para)

            # Case 1: Paragraph fits within tolerance as standalone
            if para_tokens <= max_allowed:
                # Check if adding this paragraph would exceed limits
                new_total = current_tokens + para_tokens

                # CRITICAL: Accumulation decision logic
                if new_total <= chunk_size:
                    # Under target - keep accumulating
                    current_chunk_paragraphs.append(para)
                    current_tokens = new_total

                elif new_total <= max_allowed:
                    # In overflow zone (1000-1300)
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
                # Case 2: Paragraph too large (> 1300 tokens) - split by sentences
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

    def _replace_tables_with_placeholders(
        self,
        markdown_text: str,
        num_tables: int
    ) -> Tuple[str, List[Tuple[int, int, int]]]:
        """
        Replace markdown tables with placeholders.

        Args:
            markdown_text: Original markdown content
            num_tables: Expected number of tables

        Returns:
            (text_with_placeholders, table_positions)
            table_positions: [(start, end, table_idx), ...]
        """

        text_with_placeholders = markdown_text
        table_positions = []

        # Markdown table pattern: lines starting with |
        table_pattern = r'\|[^\n]+\|(?:\n\|[^\n]+\|)+'

        # Find all tables
        for i, match in enumerate(re.finditer(table_pattern, markdown_text)):
            if i >= num_tables:
                break

            placeholder = f'<<<TABLE_{i:03d}>>>'
            table_positions.append((match.start(), match.end(), i))

            # Replace first occurrence only
            text_with_placeholders = text_with_placeholders.replace(
                match.group(0),
                placeholder,
                1
            )

        return text_with_placeholders, table_positions

    @staticmethod
    def _table_to_natural_language(table_data: Dict) -> str:
        """
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
                sentence = TableAwareChunker._format_department_row(row)
            elif table_type == 'fee_structure':
                sentence = TableAwareChunker._format_fee_row(row)
            elif table_type == 'exam_schedule':
                sentence = TableAwareChunker._format_schedule_row(row)
            elif table_type == 'eligibility':
                sentence = TableAwareChunker._format_eligibility_row(row)
            else:
                # Generic format
                sentence = TableAwareChunker._format_generic_row(row)

            if sentence:
                sentences.append(sentence)

        return '\n'.join(sentences)

    @staticmethod
    def _format_department_row(row: Dict) -> str:
        """Format department_seats table row."""
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
        """Format fee_structure table row."""
        category = row.get('গ্রুপ') or row.get('Category') or row.get('বিভাগ') or ''
        fee = row.get('ফি') or row.get('Fee') or row.get('Amount') or ''

        if category and fee:
            return f"{category} ভর্তি পরীক্ষার ফি {fee} টাকা।"
        else:
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            return ", ".join(parts) + "।"

    @staticmethod
    def _format_schedule_row(row: Dict) -> str:
        """Format exam_schedule table row."""
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
        """Format eligibility table row."""
        criteria = row.get('Criteria') or row.get('শর্ত') or ''
        requirement = row.get('Requirement') or row.get('প্রয়োজনীয়তা') or ''

        if criteria and requirement:
            return f"{criteria}: {requirement}।"
        else:
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            return ", ".join(parts) + "।"

    @staticmethod
    def _format_generic_row(row: Dict) -> str:
        """Generic row formatting (fallback)."""
        parts = [f"{k}: {v}" for k, v in row.items() if v]
        return ", ".join(parts) + "।"
