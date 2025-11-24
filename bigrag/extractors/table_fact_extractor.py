"""
Table Fact Extractor for Production Knowledge Graph

Deterministic conversion of structured table data to knowledge graph facts.
100% accurate, no LLM needed (tables are already structured by GPT-4o).
"""

import re
from typing import Dict, List, Optional


class TableFactExtractor:
    """
    Convert structured table data to knowledge graph facts.

    100% deterministic (no LLM involved).

    Key design:
    - Each table row → 1 relation (knowledge segment)
    - Each table cell → 1 entity (with context)
    - Maintains full row context for entity descriptions

    This ensures:
    - No information loss from tables
    - 100% numeric accuracy (no LLM hallucinations)
    - Entities have rich context from full row
    """

    @staticmethod
    def extract_facts_from_table(
        table_data: Dict,
        chunk_id: str
    ) -> Dict:
        """
        Convert each table row to relations and entities.

        Args:
            table_data: Structured table from GPT4TableExtractor
            chunk_id: Source chunk identifier

        Returns:
            {
                'relations': [...],
                'entities': [...],
                'confidence': 1.0,
                'extraction_method': 'rule_based_table',
                'stats': {
                    'num_rows': 10,
                    'num_relations': 10,
                    'num_entities': 30
                }
            }

        Example:
            Input table row:
            {'বিভাগ/বিষয়': 'CSE', 'কোড': 'CSE', 'আসন': '120'}

            Output:
            relations: [
                {
                    'role': 'relation',
                    'content': 'CSE বিভাগের কোড CSE এবং আসন সংখ্যা 120।',
                    'completeness_score': 10,
                    'source_id': chunk_id,
                    'metadata': {...}
                }
            ]
            entities: [
                {
                    'entity_name': 'CSE',
                    'entity_type': 'department',
                    'description': 'CSE হল একটি বিভাগের নাম।',
                    'weight': 95.0,
                    'source_id': chunk_id,
                    'metadata': {...}
                },
                {
                    'entity_name': '120',
                    'entity_type': 'seat_count',
                    'description': '120 হল CSE বিভাগের আসন সংখ্যা।',
                    'weight': 95.0,
                    'source_id': chunk_id,
                    'metadata': {...}
                }
            ]
        """

        relations = []
        entities = []

        table_type = table_data.get('table_type', 'general')
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])

        for row_idx, row in enumerate(rows):
            # Create ONE relation per row
            relation_content = TableFactExtractor._row_to_sentence(
                headers,
                row,
                table_type
            )

            # Generate relation ID FIRST (needed for entity linking)
            from bigrag.utils import compute_mdhash_id
            from bigrag.constants import RELATION_PREFIX
            relation_id = compute_mdhash_id(relation_content, prefix=RELATION_PREFIX)

            relation = {
                'role': 'relation',
                'content': relation_content,
                'description': relation_content,  # Required for BiG-RAG retrieval
                'completeness_score': 10,  # 100% complete (from structured table)
                'source_id': chunk_id,
                'hyper_relation': relation_id,  # Add relation ID for consistency
                'metadata': {
                    'extraction_method': 'table_row',
                    'table_id': table_data.get('table_id', 'unknown'),
                    'table_type': table_type,
                    'row_index': row_idx,
                    'structured_fact': row,  # Preserve original row data
                    'linked_entities': []  # Track entities from this row (for bipartite edges)
                }
            }

            # Extract entities from each cell
            for col_name, cell_value in row.items():
                entity = TableFactExtractor._cell_to_entity(
                    col_name,
                    cell_value,
                    row,  # Full row context
                    chunk_id,
                    table_type,
                    relation_id  # Pass relation ID to link entities
                )
                if entity:
                    entities.append(entity)
                    # Link entity to relation using entity_id (Option B3: survives name changes)
                    relation['metadata']['linked_entities'].append(entity['entity_id'])

            # Add relation after populating linked_entities
            relations.append(relation)

        return {
            'relations': relations,
            'entities': entities,
            'confidence': 1.0,
            'extraction_method': 'rule_based_table',
            'stats': {
                'num_rows': len(rows),
                'num_relations': len(relations),
                'num_entities': len(entities)
            }
        }

    @staticmethod
    def _row_to_sentence(
        headers: List[str],
        row: Dict,
        table_type: str
    ) -> str:
        """
        Convert table row to natural language sentence.

        Template varies by table_type for better readability.

        This is the RELATION content (knowledge segment).
        """

        if table_type == 'department_seats':
            # Example: "কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগের কোড CSE এবং আসন সংখ্যা ১২০।"
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
                parts = [f"{k}: {v}" for k, v in row.items() if v]
                return ", ".join(parts) + "।"

        elif table_type == 'fee_structure':
            # Example: "Engineering ভর্তি পরীক্ষার ফি ১১০০ টাকা।"
            category = row.get('গ্রুপ') or row.get('Category') or row.get('বিভাগ') or ''
            fee = row.get('ফি') or row.get('Fee') or row.get('Amount') or ''

            if category and fee:
                return f"{category} ভর্তি পরীক্ষার ফি {fee} টাকা।"
            else:
                parts = [f"{k}: {v}" for k, v in row.items() if v]
                return ", ".join(parts) + "।"

        elif table_type == 'exam_schedule':
            # Example: "ভর্তি পরীক্ষা তারিখ: ০৪ ডিসেম্বর, ২০২৪, সময়: ১০:০০ AM।"
            event = row.get('Event') or row.get('ইভেন্ট') or ''
            date = row.get('Date') or row.get('তারিখ') or ''
            time = row.get('Time') or row.get('সময়') or ''
            venue = row.get('Venue') or row.get('স্থান') or ''

            parts = []
            if event:
                parts.append(event)
            if date:
                parts.append(f"তারিখ: {date}")
            if time:
                parts.append(f"সময়: {time}")
            if venue:
                parts.append(f"স্থান: {venue}")

            if parts:
                return ", ".join(parts) + "।"
            else:
                parts = [f"{k}: {v}" for k, v in row.items() if v]
                return ", ".join(parts) + "।"

        elif table_type == 'eligibility':
            # Example: "SSC জিপিএ প্রয়োজন: ৪.০০।"
            criteria = row.get('Criteria') or row.get('শর্ত') or ''
            requirement = row.get('Requirement') or row.get('প্রয়োজনীয়তা') or ''

            if criteria and requirement:
                return f"{criteria} প্রয়োজন: {requirement}।"
            else:
                parts = [f"{k}: {v}" for k, v in row.items() if v]
                return ", ".join(parts) + "।"

        else:
            # Generic template
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            return ", ".join(parts) + "।"

    @staticmethod
    def _cell_to_entity(
        col_name: str,
        cell_value: str,
        full_row: Dict,
        chunk_id: str,
        table_type: str,
        relation_id: str = None
    ) -> Optional[Dict]:
        """
        Convert table cell to entity node.

        Entity type inference based on column name + table type.

        Args:
            col_name: Column name/header
            cell_value: Cell content
            full_row: Full row data (for context)
            chunk_id: Source chunk ID
            table_type: Table classification

        Returns:
            Entity dict or None if cell should not be an entity
        """

        # Skip empty cells
        if not cell_value or str(cell_value).strip() == '':
            return None

        # Infer entity type
        entity_type = TableFactExtractor._infer_entity_type(
            col_name,
            cell_value,
            table_type
        )

        # Create description with full row context
        description = TableFactExtractor._create_entity_description(
            col_name,
            cell_value,
            full_row,
            table_type
        )

        # Generate stable entity ID based on description hash (Option B3)
        from bigrag.utils import compute_mdhash_id
        from bigrag.constants import ENTITY_PREFIX
        entity_id = compute_mdhash_id(description, prefix=ENTITY_PREFIX)

        return {
            'entity_id': entity_id,  # Stable ID (survives name changes during entity linking)
            'entity_name': str(cell_value).strip(),
            'entity_type': entity_type,
            'description': description,
            'weight': 95.0,  # High weight (from structured data)
            'source_id': chunk_id,
            'hyper_relation': relation_id,  # Link to parent relation (prevents orphan entities)
            'metadata': {
                'extraction_method': 'table_cell',
                'table_column': col_name,
                'table_type': table_type
            }
        }

    @staticmethod
    def _infer_entity_type(
        col_name: str,
        cell_value: str,
        table_type: str
    ) -> str:
        """
        Map column names to domain-specific entity types.

        EDUCATIONAL DOMAIN TYPES:
        - department: Academic departments (CSE, EEE, etc.)
        - faculty: Academic faculties
        - university: University names
        - department_code: Department abbreviations
        - seat_count: Number of seats/admissions
        - gpa_requirement: GPA requirements
        - eligibility: Eligibility criteria
        - fee: Fee amounts
        - deadline: Dates/deadlines
        - number: Generic numeric values
        - concept: Generic concepts

        Args:
            col_name: Column name
            cell_value: Cell content
            table_type: Table classification

        Returns:
            Entity type string
        """

        col_lower = col_name.lower()

        # Educational domain type mapping
        type_map = {
            # Bangla department/organization
            'বিভাগ': 'department',
            'বিষয়': 'department',
            'অনুষদ': 'faculty',
            'বিশ্ববিদ্যালয়': 'university',

            # Bangla attributes
            'কোড': 'department_code',
            'আসন': 'seat_count',
            'ফি': 'fee',
            'তারিখ': 'deadline',
            'সময়': 'time',
            'জিপিএ': 'gpa_requirement',
            'শর্ত': 'eligibility',

            # English
            'department': 'department',
            'subject': 'department',
            'faculty': 'faculty',
            'university': 'university',
            'code': 'department_code',
            'seats': 'seat_count',
            'seat': 'seat_count',
            'fee': 'fee',
            'amount': 'fee',
            'date': 'deadline',
            'deadline': 'deadline',
            'time': 'time',
            'gpa': 'gpa_requirement',
            'requirement': 'eligibility',
            'criteria': 'eligibility',
            'event': 'event',
            'venue': 'location'
        }

        # Check column name first
        for key, entity_type in type_map.items():
            if key in col_lower:
                return entity_type

        # Fallback: check if pure number (Bangla or English)
        if re.match(r'^[০-৯0-9]+(?:\.[০-৯0-9]+)?$', cell_value):
            return 'number'

        # Default: concept
        return 'concept'

    @staticmethod
    def _create_entity_description(
        col_name: str,
        cell_value: str,
        full_row: Dict,
        table_type: str
    ) -> str:
        """
        Create entity description with full row context.

        This is CRITICAL for:
        - Entity disambiguation (which "120"?)
        - Semantic search (embed with context)
        - Human readability

        Example:
        col_name = "আসন"
        cell_value = "১২০"
        full_row = {"বিভাগ": "CSE", "কোড": "CSE", "আসন": "১২০"}

        Output: "১২০ হল CSE বিভাগের আসন সংখ্যা।"

        Args:
            col_name: Column name
            cell_value: Cell content
            full_row: Full row data
            table_type: Table classification

        Returns:
            Natural language description
        """

        if table_type == 'department_seats':
            dept = (
                full_row.get('বিভাগ/বিষয়') or
                full_row.get('বিভাগ') or
                full_row.get('Department') or
                full_row.get('বিষয়') or
                ''
            )

            if 'আসন' in col_name or 'Seat' in col_name:
                if dept:
                    return f"{cell_value} হল {dept} বিভাগের আসন সংখ্যা।"
                else:
                    return f"{cell_value} হল একটি আসন সংখ্যা।"

            elif 'কোড' in col_name or 'Code' in col_name:
                if dept:
                    return f"{cell_value} হল {dept} বিভাগের কোড।"
                else:
                    return f"{cell_value} হল একটি বিভাগের কোড।"

            elif 'বিভাগ' in col_name or 'Department' in col_name or 'বিষয়' in col_name:
                return f"{cell_value} হল একটি বিভাগের নাম।"

            else:
                return f"{cell_value} হল {col_name} এর মান।"

        elif table_type == 'fee_structure':
            category = full_row.get('গ্রুপ') or full_row.get('Category') or full_row.get('বিভাগ') or ''

            if 'ফি' in col_name or 'Fee' in col_name or 'Amount' in col_name:
                if category:
                    return f"{cell_value} হল {category} এর ভর্তি পরীক্ষার ফি।"
                else:
                    return f"{cell_value} হল ভর্তি পরীক্ষার ফি।"

            elif 'গ্রুপ' in col_name or 'Category' in col_name:
                return f"{cell_value} হল একটি ভর্তি পরীক্ষার গ্রুপ।"

            else:
                return f"{cell_value} হল {col_name} এর মান।"

        elif table_type == 'exam_schedule':
            event = full_row.get('Event') or full_row.get('ইভেন্ট') or ''

            if 'Date' in col_name or 'তারিখ' in col_name:
                if event:
                    return f"{cell_value} হল {event} এর তারিখ।"
                else:
                    return f"{cell_value} হল একটি তারিখ।"

            elif 'Time' in col_name or 'সময়' in col_name:
                if event:
                    return f"{cell_value} হল {event} এর সময়।"
                else:
                    return f"{cell_value} হল একটি সময়।"

            elif 'Venue' in col_name or 'স্থান' in col_name:
                if event:
                    return f"{cell_value} হল {event} এর স্থান।"
                else:
                    return f"{cell_value} হল একটি স্থান।"

            else:
                return f"{cell_value} হল {col_name} এর মান।"

        elif table_type == 'eligibility':
            criteria = full_row.get('Criteria') or full_row.get('শর্ত') or ''

            if 'Requirement' in col_name or 'প্রয়োজনীয়তা' in col_name:
                if criteria:
                    return f"{cell_value} হল {criteria} এর প্রয়োজনীয়তা।"
                else:
                    return f"{cell_value} হল একটি প্রয়োজনীয়তা।"

            elif 'Criteria' in col_name or 'শর্ত' in col_name:
                return f"{cell_value} হল একটি যোগ্যতার শর্ত।"

            else:
                return f"{cell_value} হল {col_name} এর মান।"

        else:
            # Generic description
            return f"{cell_value} হল {col_name} এর মান।"
