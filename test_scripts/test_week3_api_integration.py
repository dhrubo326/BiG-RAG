"""
Week 3 API Integration Smoke Tests

Tests the integration of modular pipeline (Weeks 1-2) with BiGRAG core
and backend API endpoints.

All 5 tests should pass to confirm Week 3 implementation is complete.
"""

import sys
import os
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_pipeline_features():
    """Test 1: Can we import PipelineFeatures from bigrag?"""
    print("[TEST 1] Importing PipelineFeatures...")
    try:
        from bigrag.pipeline.features import PipelineFeatures
        from bigrag import BiGRAG

        # Verify BiGRAG accepts pipeline_features parameter
        import inspect
        sig = inspect.signature(BiGRAG.__init__)
        assert 'pipeline_features' in sig.parameters, "BiGRAG should have pipeline_features parameter"

        print("  [OK] PipelineFeatures importable and BiGRAG accepts pipeline_features")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bigrag_default_pipeline():
    """Test 2: Does BiGRAG default to standard preset when pipeline_features=None?"""
    print("\n[TEST 2] Testing BiGRAG default pipeline initialization...")
    try:
        from bigrag import BiGRAG
        from bigrag.pipeline.features import PipelineFeatures

        # Create BiGRAG without pipeline_features (should default to standard)
        rag = BiGRAG(
            working_dir="./test_temp_bigrag",
            pipeline_features=None  # Should default to standard preset
        )

        # Verify pipeline_features was set
        assert rag.pipeline_features is not None, "pipeline_features should not be None"
        assert isinstance(rag.pipeline_features, PipelineFeatures), "Should be PipelineFeatures instance"

        # Verify it's using standard preset characteristics
        assert rag.pipeline_features.enable_gleaning == True, "Standard preset should enable gleaning"
        assert rag.pipeline_features.enable_table_detection == False, "Standard preset should not enable table detection"

        print("  [OK] BiGRAG defaults to standard preset")

        # Cleanup
        import shutil
        if os.path.exists("./test_temp_bigrag"):
            shutil.rmtree("./test_temp_bigrag")

        return True
    except Exception as e:
        print(f"  [FAIL] BiGRAG initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bigrag_with_presets():
    """Test 3: Can BiGRAG be initialized with each preset?"""
    print("\n[TEST 3] Testing BiGRAG with all 3 presets...")

    from bigrag import BiGRAG
    from bigrag.pipeline.features import PipelineFeatures

    presets_tested = 0

    for preset_name in ["standard", "quality", "balanced"]:
        try:
            features = PipelineFeatures.from_preset(
                preset_name,
                openai_api_key="test-key"
            )

            rag = BiGRAG(
                working_dir=f"./test_temp_{preset_name}",
                pipeline_features=features
            )

            assert rag.pipeline_features is not None
            print(f"  [OK] {preset_name} preset works")
            presets_tested += 1

            # Cleanup
            import shutil
            if os.path.exists(f"./test_temp_{preset_name}"):
                shutil.rmtree(f"./test_temp_{preset_name}")
        except Exception as e:
            print(f"  [FAIL] {preset_name} preset failed: {e}")
            import traceback
            traceback.print_exc()

    return presets_tested == 3


def test_no_deprecated_parameters():
    """Test 4: Verify deprecated parameters are removed"""
    print("\n[TEST 4] Verifying deprecated parameters are removed...")
    try:
        from bigrag import BiGRAG
        import inspect

        # Get BiGRAG __init__ signature
        sig = inspect.signature(BiGRAG.__init__)
        params = list(sig.parameters.keys())

        # Check deprecated parameters are removed
        assert 'use_production_pipeline' not in params, "use_production_pipeline should be removed"
        assert 'use_enhanced_pipeline' not in params, "use_enhanced_pipeline should be removed"
        assert 'production_pipeline_config' not in params, "production_pipeline_config should be removed"
        assert 'enhanced_pipeline_config' not in params, "enhanced_pipeline_config should be removed"

        # Check new parameter exists
        assert 'pipeline_features' in params, "pipeline_features should exist"

        print("  [OK] All deprecated parameters removed, pipeline_features added")
        return True
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_integration():
    """Test 5: Basic API route import test (routes should compile)"""
    print("\n[TEST 5] Testing API routes compile...")
    try:
        # Just test that routes can be imported
        # Full API testing requires server running
        import backend.api.routes.documents as doc_routes
        import backend.api.routes.datasets as dataset_routes

        # Check that upload endpoint exists
        assert hasattr(doc_routes.router, 'routes'), "Documents router should have routes"

        # Check datasets router
        assert hasattr(dataset_routes.router, 'routes'), "Datasets router should have routes"

        print("  [OK] API routes compile successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] API import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Week 3 smoke tests"""
    print("="*70)
    print("Week 3 API Integration Smoke Tests")
    print("="*70)

    results = {
        "Import PipelineFeatures": test_import_pipeline_features(),
        "BiGRAG default pipeline": test_bigrag_default_pipeline(),
        "BiGRAG with presets": test_bigrag_with_presets(),
        "No deprecated parameters": test_no_deprecated_parameters(),
        "API integration": test_api_integration()
    }

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All Week 3 API integration tests passed!")
        print("Week 3 implementation complete and verified.")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} tests failed")
        print("Fix failures before proceeding to Week 4.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
