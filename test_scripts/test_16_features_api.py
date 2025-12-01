"""
Test Script for 16-Feature Indexing API

Tests the new unified_indexing.py endpoint with 16 independent features.
"""

from bigrag.config import IndexingConfig

def test_indexing_config_creation():
    """Test creating IndexingConfig with 16 features"""
    print("\n=== Test 1: Create IndexingConfig with all 16 features ===")

    try:
        config = IndexingConfig(
            # Group A: Chunking (2)
            chunking_strategy="semantic",
            enable_table_detection=True,

            # Group B: Extraction (4)
            extraction_strategy="gleaning",
            enable_table_fact_extraction=False,
            enable_multilingual=True,

            # Group C: Validation (3)
            enable_numeric_validation=False,
            enable_entity_validation=True,
            enable_relation_validation=True,

            # Group D: Merging (2)
            enable_entity_merging=True,
            enable_fuzzy_matching=True,

            # Group E: Quality (3)
            enable_hitl=True,
            enable_orphan_linking=True,
            enable_quality_scoring=True,

            # API keys (required for some features)
            openai_api_key="test_key",
            dataset_path="./test_dataset"
        )

        print("[OK] IndexingConfig created successfully")
        print(f"  - Chunking: {config.chunking_strategy}")
        print(f"  - Table detection: {config.enable_table_detection}")
        print(f"  - Extraction: {config.extraction_strategy}")
        print(f"  - Multilingual: {config.enable_multilingual}")
        print(f"  - Numeric validation: {config.enable_numeric_validation}")
        print(f"  - Entity validation: {config.enable_entity_validation}")
        print(f"  - Relation validation: {config.enable_relation_validation}")
        print(f"  - Entity merging: {config.enable_entity_merging}")
        print(f"  - Fuzzy matching: {config.enable_fuzzy_matching}")
        print(f"  - HITL: {config.enable_hitl}")
        print(f"  - Orphan linking: {config.enable_orphan_linking}")
        print(f"  - Quality scoring: {config.enable_quality_scoring}")

    except Exception as e:
        print(f"[FAIL] Failed to create IndexingConfig: {e}")
        return False

    return True


def test_dependency_validation():
    """Test dependency validation in IndexingConfig"""
    print("\n=== Test 2: Dependency Validation ===")

    # Test 1: Table fact extraction without table detection (should fail)
    print("\nTest 2.1: Table fact extraction without table detection")
    try:
        config = IndexingConfig(
            enable_table_fact_extraction=True,
            enable_table_detection=False,  # Missing dependency
            openai_api_key="test_key"
        )
        print("[FAIL] Should have raised ValueError for missing table detection")
        return False
    except ValueError as e:
        print(f"[OK] Correctly caught dependency error: {str(e)[:100]}...")

    # Test 2: Table detection with semantic chunking without API key (should fail)
    print("\nTest 2.2: Table detection with semantic chunking without API key")
    try:
        config = IndexingConfig(
            chunking_strategy="semantic",
            enable_table_detection=True,
            openai_api_key=None  # Missing API key
        )
        print("[FAIL] Should have raised ValueError for missing API key")
        return False
    except ValueError as e:
        print(f"[OK] Correctly caught dependency error: {str(e)[:100]}...")

    # Test 3: Valid configuration (should pass)
    print("\nTest 2.3: Valid configuration with all dependencies")
    try:
        config = IndexingConfig(
            chunking_strategy="semantic",
            enable_table_detection=True,
            enable_table_fact_extraction=True,
            openai_api_key="test_key"
        )
        print("[OK] Valid configuration accepted")
    except ValueError as e:
        print(f"[FAIL] Valid config rejected: {e}")
        return False

    return True


def test_presets():
    """Test preset configurations"""
    print("\n=== Test 3: Preset Configurations ===")

    # Test fast preset
    print("\nTest 3.1: Fast preset")
    try:
        config = IndexingConfig.preset_fast(openai_api_key="test_key")
        print(f"[OK] Fast preset created")
        print(f"  - Chunking: {config.chunking_strategy} (expected: token)")
        print(f"  - Table detection: {config.enable_table_detection} (expected: False)")
        print(f"  - Extraction: {config.extraction_strategy} (expected: strict)")
        print(f"  - Fuzzy matching: {config.enable_fuzzy_matching} (expected: False)")

        assert config.chunking_strategy == "token", "Fast preset should use token chunking"
        assert config.enable_table_detection == False, "Fast preset should disable table detection"
        assert config.extraction_strategy == "strict", "Fast preset should use strict extraction"
        assert config.enable_fuzzy_matching == False, "Fast preset should disable fuzzy matching"

    except Exception as e:
        print(f"[FAIL] Fast preset failed: {e}")
        return False

    # Test balanced preset
    print("\nTest 3.2: Balanced preset")
    try:
        config = IndexingConfig.preset_balanced(openai_api_key="test_key")
        print(f"[OK] Balanced preset created")
        print(f"  - Chunking: {config.chunking_strategy} (expected: semantic)")
        print(f"  - Table detection: {config.enable_table_detection} (expected: True)")
        print(f"  - Extraction: {config.extraction_strategy} (expected: gleaning)")

        assert config.chunking_strategy == "semantic", "Balanced preset should use semantic chunking"
        assert config.enable_table_detection == True, "Balanced preset should enable table detection"
        assert config.extraction_strategy == "gleaning", "Balanced preset should use gleaning extraction"

    except Exception as e:
        print(f"[FAIL] Balanced preset failed: {e}")
        return False

    # Test quality preset
    print("\nTest 3.3: Quality preset")
    try:
        config = IndexingConfig.preset_quality(
            openai_api_key="test_key",
            gemini_api_key="test_gemini_key"
        )
        print(f"[OK] Quality preset created")
        print(f"  - Chunking: {config.chunking_strategy} (expected: semantic)")
        print(f"  - Table detection: {config.enable_table_detection} (expected: True)")
        print(f"  - Table fact extraction: {config.enable_table_fact_extraction} (expected: True)")
        print(f"  - Numeric validation: {config.enable_numeric_validation} (expected: True)")
        print(f"  - Fuzzy matching: {config.enable_fuzzy_matching} (expected: True)")

        assert config.chunking_strategy == "semantic", "Quality preset should use semantic chunking"
        assert config.enable_table_detection == True, "Quality preset should enable table detection"
        assert config.enable_table_fact_extraction == True, "Quality preset should enable table fact extraction"
        assert config.enable_numeric_validation == True, "Quality preset should enable numeric validation"
        assert config.enable_fuzzy_matching == True, "Quality preset should enable fuzzy matching"

    except Exception as e:
        print(f"[FAIL] Quality preset failed: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing 16-Feature Indexing System")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("IndexingConfig Creation", test_indexing_config_creation()))
    results.append(("Dependency Validation", test_dependency_validation()))
    results.append(("Preset Configurations", test_presets()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
