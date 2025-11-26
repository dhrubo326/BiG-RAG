"""
Test synthetic relation generation for orphan entity fix.

Verifies that synthetic English relations are generated for table entities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigrag.extractors.table_fact_extractor import TableFactExtractor


def test_synthetic_relation_generation():
    """Test that synthetic relations are created for department table"""

    print("="*80)
    print("TEST: Synthetic Relation Generation for Orphan Entity Fix")
    print("="*80)

    # Simulate table data (from KUET document)
    table_data = {
        'table_id': 'test_table_1',
        'table_type': 'department_seats',
        'headers': ['বিভাগ/বিষয়', 'কোড', 'আসন'],
        'rows': [
            {
                'বিভাগ/বিষয়': 'সিভিল ইঞ্জিনিয়ারিং',
                'কোড': 'CE',
                'আসন': '120'
            },
            {
                'বিভাগ/বিষয়': 'কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং',
                'কোড': 'CSE',
                'আসন': '120'
            }
        ]
    }

    chunk_id = 'chunk-test123'

    # Extract facts
    result = TableFactExtractor.extract_facts_from_table(table_data, chunk_id)

    print(f"\nExtraction Results:")
    print(f"  Total relations: {len(result['relations'])}")
    print(f"  Total entities: {len(result['entities'])}")

    # Check for synthetic relations
    synthetic_relations = [
        r for r in result['relations']
        if r.get('metadata', {}).get('extraction_method') == 'synthetic_cross_lingual'
    ]

    print(f"\n  Synthetic relations: {len(synthetic_relations)}")

    # Verify synthetic relations
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    if len(synthetic_relations) == 0:
        print("[FAIL] No synthetic relations generated!")
        return False

    # Check first synthetic relation
    synth_rel = synthetic_relations[0]
    print("\nSample Synthetic Relation:")
    print("  Content: [Contains department name, code, and seats]")
    print(f"  Linked entities: {len(synth_rel['metadata']['linked_entities'])}")
    print(f"  Completeness score: {synth_rel['completeness_score']}")

    # Verify entity names are in relation content
    entities = result['entities']
    dept_entities = [e for e in entities if e.get('entity_type') == 'department']

    print(f"\n  Department entities found: {len(dept_entities)}")

    success = True
    for dept in dept_entities:
        dept_name = dept['entity_name']

        # Find synthetic relation mentioning this department
        matching_synth = None
        for sr in synthetic_relations:
            if dept_name in sr['content']:
                matching_synth = sr
                break

        if matching_synth:
            print(f"  [OK] Department has synthetic relation")
        else:
            print(f"  [FAIL] Department missing synthetic relation")
            success = False

    # Overall result
    print("\n" + "="*80)
    if success:
        print("[SUCCESS] Synthetic relation generation working correctly!")
        print("Expected result: 19 orphan entities -> 0 orphan entities after rebuild")
    else:
        print("[FAIL] Synthetic relation generation has issues")
    print("="*80)

    return success


if __name__ == '__main__':
    success = test_synthetic_relation_generation()
    sys.exit(0 if success else 1)
