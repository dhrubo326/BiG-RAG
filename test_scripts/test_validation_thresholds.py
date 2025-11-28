"""
Test VALIDATION_THRESHOLDS constant is properly integrated

This test verifies that:
1. VALIDATION_THRESHOLDS can be imported from features.py
2. It has the correct structure
3. UnifiedPipeline can use it when validation is enabled
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_validation_thresholds_import():
    """Test 1: VALIDATION_THRESHOLDS can be imported"""
    print("\n[TEST 1] Importing VALIDATION_THRESHOLDS...")

    try:
        from bigrag.pipeline.features import VALIDATION_THRESHOLDS
        print("  [OK] VALIDATION_THRESHOLDS imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Failed to import VALIDATION_THRESHOLDS: {e}")
        return False


def test_validation_thresholds_structure():
    """Test 2: VALIDATION_THRESHOLDS has correct structure"""
    print("\n[TEST 2] Verifying VALIDATION_THRESHOLDS structure...")

    from bigrag.pipeline.features import VALIDATION_THRESHOLDS

    # Check required strictness levels
    required_levels = ["STRICT", "MODERATE", "LENIENT"]
    for level in required_levels:
        if level not in VALIDATION_THRESHOLDS:
            print(f"  [FAIL] Missing strictness level: {level}")
            return False
    print(f"  [OK] All strictness levels present: {required_levels}")

    # Check required threshold keys
    required_keys = [
        "numeric_coverage_min",
        "entity_quality_min",
        "relation_completeness_min",
        "entity_name_min_length",
        "relation_description_min_length",
        "allow_generic_types"
    ]

    for level in required_levels:
        thresholds = VALIDATION_THRESHOLDS[level]
        for key in required_keys:
            if key not in thresholds:
                print(f"  [FAIL] Missing key '{key}' in {level}")
                return False

    print(f"  [OK] All required threshold keys present in each level")

    # Verify threshold values make sense (STRICT > MODERATE > LENIENT)
    strict = VALIDATION_THRESHOLDS["STRICT"]
    moderate = VALIDATION_THRESHOLDS["MODERATE"]
    lenient = VALIDATION_THRESHOLDS["LENIENT"]

    if not (strict["entity_quality_min"] > moderate["entity_quality_min"] > lenient["entity_quality_min"]):
        print(f"  [FAIL] entity_quality_min values not properly ordered")
        return False

    if not (strict["relation_completeness_min"] > moderate["relation_completeness_min"] > lenient["relation_completeness_min"]):
        print(f"  [FAIL] relation_completeness_min values not properly ordered")
        return False

    print("  [OK] Threshold values properly ordered (STRICT > MODERATE > LENIENT)")

    return True


def test_base_pipeline_imports():
    """Test 3: UnifiedPipeline can import VALIDATION_THRESHOLDS"""
    print("\n[TEST 3] Testing UnifiedPipeline imports VALIDATION_THRESHOLDS...")

    try:
        from bigrag.pipeline.base_pipeline import UnifiedPipeline
        print("  [OK] UnifiedPipeline imports successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Failed to import UnifiedPipeline: {e}")
        return False


def test_validation_with_quality_preset():
    """Test 4: Quality preset enables validation (which uses VALIDATION_THRESHOLDS)"""
    print("\n[TEST 4] Testing quality preset with validation enabled...")

    from bigrag.pipeline.features import PipelineFeatures

    # Quality preset should have validation enabled
    quality_features = PipelineFeatures.from_preset(
        "quality",
        openai_api_key="test-key",
        gemini_api_key=None
    )

    if not quality_features.enable_entity_validation:
        print("  [FAIL] quality preset should have enable_entity_validation=True")
        return False

    if not quality_features.enable_relation_validation:
        print("  [FAIL] quality preset should have enable_relation_validation=True")
        return False

    print("  [OK] quality preset has validation enabled")

    # Check validation_strictness is set
    if quality_features.validation_strictness not in ["STRICT", "MODERATE", "LENIENT"]:
        print(f"  [FAIL] Invalid validation_strictness: {quality_features.validation_strictness}")
        return False

    print(f"  [OK] validation_strictness set to: {quality_features.validation_strictness}")

    return True


def main():
    """Run all validation threshold tests"""
    print("=" * 70)
    print("VALIDATION_THRESHOLDS Integration Tests")
    print("=" * 70)

    tests = [
        ("Import VALIDATION_THRESHOLDS", test_validation_thresholds_import),
        ("VALIDATION_THRESHOLDS structure", test_validation_thresholds_structure),
        ("UnifiedPipeline imports", test_base_pipeline_imports),
        ("Quality preset validation", test_validation_with_quality_preset),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  [ERROR] Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All VALIDATION_THRESHOLDS tests passed!")
        print("The missing constant has been fixed and validated.")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
