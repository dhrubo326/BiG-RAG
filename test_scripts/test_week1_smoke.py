"""
Week 1 Smoke Tests for Modular Pipeline

These are SMOKE TESTS only - verify no crashes, basic functionality works.
Comprehensive testing happens in Week 4.

Tests:
1. Can PipelineFeatures be imported?
2. Can all 3 presets be created?
3. Do presets validate without errors?
4. Can UnifiedPipeline be instantiated?
5. Does quality_scoring module work?
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_features():
    """Test 1: Can we import PipelineFeatures?"""
    print("[TEST 1] Importing PipelineFeatures...")
    try:
        from bigrag.pipeline.features import PipelineFeatures
        print("  [OK] PipelineFeatures imported successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_create_presets():
    """Test 2: Can we create all 3 presets?"""
    print("\n[TEST 2] Creating all presets...")
    from bigrag.pipeline.features import PipelineFeatures

    presets_tested = 0
    for preset_name in ["standard", "quality", "balanced"]:
        try:
            features = PipelineFeatures.from_preset(preset_name, openai_api_key="test-key")
            print(f"  [OK] {preset_name} preset created")
            presets_tested += 1
        except Exception as e:
            print(f"  [FAIL] {preset_name} preset failed: {e}")

    return presets_tested == 3


def test_preset_validation():
    """Test 3: Do presets validate without errors?"""
    print("\n[TEST 3] Validating presets...")
    from bigrag.pipeline.features import PipelineFeatures

    all_valid = True
    for preset_name in ["standard", "quality", "balanced"]:
        try:
            features = PipelineFeatures.from_preset(preset_name, openai_api_key="test-key")
            warnings = features.validate()
            if warnings:
                print(f"  [WARN] {preset_name} has {len(warnings)} warnings:")
                for w in warnings:
                    print(f"    - {w}")
            else:
                print(f"  [OK] {preset_name} validated with no warnings")
        except Exception as e:
            print(f"  [FAIL] {preset_name} validation failed: {e}")
            all_valid = False

    return all_valid


def test_import_unified_pipeline():
    """Test 4: Can we import and instantiate UnifiedPipeline?"""
    print("\n[TEST 4] Importing UnifiedPipeline...")
    try:
        from bigrag.pipeline import PipelineFeatures, UnifiedPipeline

        # Try to instantiate with standard preset
        features = PipelineFeatures.from_preset("standard", openai_api_key="test-key")
        pipeline = UnifiedPipeline(features, dataset_path=None)

        print("  [OK] UnifiedPipeline instantiated successfully")
        print(f"  [INFO] Detected preset: {pipeline._detect_preset()}")
        print(f"  [INFO] Features: {pipeline._summarize_features()}")
        return True

    except Exception as e:
        print(f"  [FAIL] UnifiedPipeline instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_scoring():
    """Test 5: Does quality scoring work?"""
    print("\n[TEST 5] Testing quality scoring...")
    try:
        from bigrag.utils import description_quality_score

        # Test with good description
        desc1 = "Albert Einstein was a German physicist who developed the theory of relativity in 1905"
        score1 = description_quality_score(desc1)
        assert score1 > 50, f"Expected high score for detailed description, got {score1}"
        print(f"  [OK] Good description scored {score1:.1f} (>50)")

        # Test with poor description
        desc2 = "A thing"
        score2 = description_quality_score(desc2)
        assert score2 < 30, f"Expected low score for vague description, got {score2}"
        print(f"  [OK] Poor description scored {score2:.1f} (<30)")

        # Test with empty description
        score3 = description_quality_score("")
        assert score3 == 0, f"Expected 0 for empty description, got {score3}"
        print(f"  [OK] Empty description scored {score3:.1f} (=0)")

        return True

    except Exception as e:
        print(f"  [FAIL] Quality scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Week 1 Smoke Tests - Modular Pipeline")
    print("=" * 60)

    results = {
        "Import PipelineFeatures": test_import_features(),
        "Create all presets": test_create_presets(),
        "Validate presets": test_preset_validation(),
        "Import UnifiedPipeline": test_import_unified_pipeline(),
        "Quality scoring": test_quality_scoring()
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All Week 1 smoke tests passed!")
        print("Ready to proceed to Week 2 (Integration)")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} tests failed")
        print("Fix failures before proceeding to Week 2")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
