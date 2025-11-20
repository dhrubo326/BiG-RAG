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
        chunk_size: int = 1200,
        overlap: int = 100,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk document with table awareness.

        Args:
            markdown_text: Document content in markdown format
            chunk_size: Maximum chunk size in tokens (default: 1200)
            overlap: Overlap between chunks in tokens (default: 100)
            metadata: Optional document metadata (title, category, tags)

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

        # Step 3: Chunk non-table text
        text_chunks = split_text_by_token_size(
            text_with_placeholders,
            chunk_size,
            overlap
        )

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
                            'language_info': lang_info
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
                        'language_info': lang_info
                    }
                })

                chunk_id += 1

        return chunks

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
