"""
Test Bangla Number Normalization Fix

Verifies that the CRITICAL bug (Bangla number mismatch) is fixed:
- Source: "120" (English)
- Extraction: "১২০" (Bangla)
- Validator should match them correctly
"""

from bigrag.validators.numeric_validator import NumericValidator


def test_bangla_number_normalization():
    """Test that Bangla numbers are correctly normalized during validation."""

    print("\n" + "="*80)
    print("Test: Bangla Number Normalization Fix")
    print("="*80)

    validator = NumericValidator()

    # Test Case 1: English source, Bangla extraction
    print("\nTest Case 1: English Source → Bangla Extraction")
    print("-" * 80)

    source_doc = "CSE বিভাগে 120টি আসন রয়েছে।"  # English "120"

    entities = [
        {
            'entity_name': 'CSE',
            'entity_type': 'department_code',
            'description': 'Computer Science and Engineering'
        },
        {
            'entity_name': '১২০',  # Bangla numeral!
            'entity_type': 'seat_count',
            'description': 'CSE বিভাগের আসন সংখ্যা'
        }
    ]

    relations = [
        {
            'content': 'CSE বিভাগে ১২০টি আসন রয়েছে।'  # Bangla "১২০"
        }
    ]

    result = validator.validate_extraction(
        source_document=source_doc,
        entities=entities,
        relations=relations,
        validation_level="STRICT"
    )

    print(f"Source numbers found: {result['total_source_numbers']}")
    print(f"KG numbers found: {result['total_kg_numbers']}")
    print(f"Numeric coverage: {result['numeric_coverage']:.2%}")
    print(f"Hallucination rate: {result['hallucination_rate']:.2%}")
    print(f"Status: {result['status']}")

    # Assert validation passes (100% coverage, 0% hallucination)
    assert result['numeric_coverage'] == 1.0, \
        f"Expected 100% coverage, got {result['numeric_coverage']:.2%}"
    assert result['hallucination_rate'] == 0.0, \
        f"Expected 0% hallucination, got {result['hallucination_rate']:.2%}"
    assert result['status'] == 'PASS', \
        f"Expected PASS, got {result['status']}"

    print("\n[OK] Bangla number correctly matched to English '120'")

    # Test Case 2: Bangla source, English extraction
    print("\nTest Case 2: Bangla Source → English Extraction")
    print("-" * 80)

    source_doc = "CSE বিভাগে ১২০টি আসন রয়েছে।"  # Bangla "১২০"

    entities = [
        {
            'entity_name': '120',  # English numeral!
            'entity_type': 'seat_count',
            'description': 'CSE seats'
        }
    ]

    relations = [
        {
            'content': 'CSE has 120 seats.'  # English "120"
        }
    ]

    result = validator.validate_extraction(
        source_document=source_doc,
        entities=entities,
        relations=relations,
        validation_level="STRICT"
    )

    print(f"Source numbers found: {result['total_source_numbers']}")
    print(f"KG numbers found: {result['total_kg_numbers']}")
    print(f"Numeric coverage: {result['numeric_coverage']:.2%}")
    print(f"Hallucination rate: {result['hallucination_rate']:.2%}")
    print(f"Status: {result['status']}")

    assert result['numeric_coverage'] == 1.0, \
        f"Expected 100% coverage, got {result['numeric_coverage']:.2%}"
    assert result['hallucination_rate'] == 0.0, \
        f"Expected 0% hallucination, got {result['hallucination_rate']:.2%}"
    assert result['status'] == 'PASS', \
        f"Expected PASS, got {result['status']}"

    print("\n[OK] English number '120' correctly matched to Bangla '১২০'")

    # Test Case 3: Mixed numbers (multiple)
    print("\nTest Case 3: Mixed Numbers (Multiple)")
    print("-" * 80)

    source_doc = "CSE: 120 আসন, EEE: ১১০ আসন, ME: 100 আসন"  # Mix of English and Bangla

    entities = [
        {'entity_name': '১২০', 'entity_type': 'seat_count', 'description': 'CSE'},
        {'entity_name': '110', 'entity_type': 'seat_count', 'description': 'EEE'},
        {'entity_name': '১০০', 'entity_type': 'seat_count', 'description': 'ME'},
    ]

    relations = []

    result = validator.validate_extraction(
        source_document=source_doc,
        entities=entities,
        relations=relations,
        validation_level="STRICT"
    )

    print(f"Source numbers found: {result['total_source_numbers']}")
    print(f"KG numbers found: {result['total_kg_numbers']}")
    print(f"Numeric coverage: {result['numeric_coverage']:.2%}")
    print(f"Hallucination rate: {result['hallucination_rate']:.2%}")
    print(f"Status: {result['status']}")

    assert result['numeric_coverage'] == 1.0, \
        f"Expected 100% coverage, got {result['numeric_coverage']:.2%}"
    assert result['hallucination_rate'] == 0.0, \
        f"Expected 0% hallucination, got {result['hallucination_rate']:.2%}"
    assert result['status'] == 'PASS', \
        f"Expected PASS, got {result['status']}"

    print("\n[OK] Mixed Bangla/English numbers correctly matched")

    # Test Case 4: Hallucination detection still works
    print("\nTest Case 4: Hallucination Detection Still Works")
    print("-" * 80)

    source_doc = "CSE বিভাগে 120টি আসন রয়েছে।"

    entities = [
        {'entity_name': '120', 'entity_type': 'seat_count', 'description': 'CSE'},
        {'entity_name': '125', 'entity_type': 'seat_count', 'description': 'EEE'},  # Hallucinated!
    ]

    relations = []

    result = validator.validate_extraction(
        source_document=source_doc,
        entities=entities,
        relations=relations,
        validation_level="STRICT"
    )

    print(f"Source numbers found: {result['total_source_numbers']}")
    print(f"KG numbers found: {result['total_kg_numbers']}")
    print(f"Numeric coverage: {result['numeric_coverage']:.2%}")
    print(f"Hallucination rate: {result['hallucination_rate']:.2%}")
    print(f"Hallucinated numbers: {result['hallucinated_numbers']}")
    print(f"Status: {result['status']}")

    assert result['hallucination_rate'] == 0.5, \
        f"Expected 50% hallucination (1 out of 2), got {result['hallucination_rate']:.2%}"
    assert '125' in result['hallucinated_numbers'], \
        f"'125' should be detected as hallucinated"
    assert result['status'] == 'FAIL', \
        f"Expected FAIL due to hallucination, got {result['status']}"

    print("\n[OK] Hallucination '125' correctly detected as not in source")

    return True


def main():
    """Run Bangla number normalization test."""

    print("\n" + "="*80)
    print("BANGLA NUMBER NORMALIZATION FIX - CRITICAL BUG TEST")
    print("="*80)
    print("\nTesting that Bangla numerals match English numerals")
    print("(Bangla 120 should match English 120)")
    print("This was the BLOCKING bug preventing validation from working.")
    print("\n" + "="*80)

    try:
        test_bangla_number_normalization()

        print("\n" + "="*80)
        print("ALL TESTS PASSED")
        print("="*80)
        print("\nCRITICAL BUG FIXED:")
        print("  [OK] Bangla numerals normalized to English in extraction methods")
        print("  [OK] English source '120' matches Bangla extraction '১২০'")
        print("  [OK] Bangla source '১২০' matches English extraction '120'")
        print("  [OK] Mixed Bangla/English numbers handled correctly")
        print("  [OK] Hallucination detection still works")
        print("\nValidation system now works correctly for bilingual content!")
        print("="*80)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
