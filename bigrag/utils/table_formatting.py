"""
Table Formatting Utilities for BiG-RAG

Converts structured table data to natural language for embedding quality.
Extracted from TableAwareChunker to eliminate dependencies on archived code.
"""

from typing import Dict, List


class TableFormatter:
    """
    Convert structured table data to natural language.

    CRITICAL for embedding quality:
    - Natural language embeds better than raw structured data
    - Preserves semantic meaning for retrieval
    - Makes table content searchable
    """

    @staticmethod
    def table_to_natural_language(table_data: Dict) -> str:
        """
        Convert structured table to natural language.

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

        Args:
            table_data: Structured table dict from GPT4TableExtractor

        Returns:
            Natural language representation of table
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
                sentence = TableFormatter._format_department_row(row)
            elif table_type == 'fee_structure':
                sentence = TableFormatter._format_fee_row(row)
            elif table_type == 'exam_schedule':
                sentence = TableFormatter._format_schedule_row(row)
            elif table_type == 'eligibility':
                sentence = TableFormatter._format_eligibility_row(row)
            else:
                # Generic format
                sentence = TableFormatter._format_generic_row(row)

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
