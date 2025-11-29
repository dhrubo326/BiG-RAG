"""
Test script to verify enable_numeric_validation=False works correctly.

This test ensures that when numeric validation is disabled:
1. Main validation is skipped
2. Gleaning validation is skipped
3. Final validation is skipped
4. All chunks are accepted without HITL failures
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor
import os


async def test_validation_disabled():
    """Test that validation can be completely disabled."""

    print("[TEST] Testing enable_numeric_validation=False")
    print("=" * 80)

    # Test document with numbers (should normally fail if missing numbers)
    test_text = """
    KUET Admission Requirements:
    - Minimum GPA: 4.00 in SSC
    - Minimum GPA: 4.00 in HSC
    - Total GP in 4 subjects: 18.00
    - Admission test date: January 11, 2025
    - Total seats: 1,065
    - Application deadline: December 14, 2024
    """

    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        with open('openai_api_key.txt') as f:
            api_key = f.read().strip()

    # Test 1: With validation ENABLED (default)
    print("\n[TEST 1] Validation ENABLED (enable_numeric_validation=True)")
    print("-" * 80)

    extractor_enabled = ConstrainedLLMExtractor(
        api_key=api_key,
        model="gpt-4o-mini",
        extraction_mode="semi_structured",
        enable_gleaning=True,
        max_gleaning_iterations=2,
        enable_numeric_validation=True  # Validation ON
    )

    result_enabled = await extractor_enabled.extract_from_paragraph(
        paragraph_text=test_text,
        chunk_id="test_chunk_enabled",
        language="English"
    )

    if result_enabled:
        validation = result_enabled.get('validation', {})
        print(f"[RESULT] Status: {validation.get('status')}")
        print(f"[RESULT] Numeric Coverage: {validation.get('numeric_coverage', 0):.2%}")
        print(f"[RESULT] Entities: {len(result_enabled.get('entities', []))}")
        print(f"[RESULT] Relations: {len(result_enabled.get('relations', []))}")
        if validation.get('skipped'):
            print("[ERROR] Validation should NOT be skipped when enabled!")
            return False
    else:
        print("[RESULT] Extraction FAILED (validation rejected)")

    # Test 2: With validation DISABLED
    print("\n[TEST 2] Validation DISABLED (enable_numeric_validation=False)")
    print("-" * 80)

    extractor_disabled = ConstrainedLLMExtractor(
        api_key=api_key,
        model="gpt-4o-mini",
        extraction_mode="semi_structured",
        enable_gleaning=True,
        max_gleaning_iterations=2,
        enable_numeric_validation=False  # Validation OFF
    )

    result_disabled = await extractor_disabled.extract_from_paragraph(
        paragraph_text=test_text,
        chunk_id="test_chunk_disabled",
        language="English"
    )

    if result_disabled:
        validation = result_disabled.get('validation', {})
        print(f"[RESULT] Status: {validation.get('status')}")
        print(f"[RESULT] Numeric Coverage: {validation.get('numeric_coverage', 0):.2%}")
        print(f"[RESULT] Entities: {len(result_disabled.get('entities', []))}")
        print(f"[RESULT] Relations: {len(result_disabled.get('relations', []))}")
        print(f"[RESULT] Skipped: {validation.get('skipped', False)}")

        # Verify validation was actually skipped
        if not validation.get('skipped'):
            print("[ERROR] Validation should be SKIPPED when disabled!")
            return False

        if validation.get('status') != 'PASS':
            print("[ERROR] Status should be PASS when validation is skipped!")
            return False

    else:
        print("[ERROR] Extraction should SUCCEED when validation is disabled!")
        return False

    print("\n" + "=" * 80)
    print("[PASS] All tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    result = asyncio.run(test_validation_disabled())
    sys.exit(0 if result else 1)
