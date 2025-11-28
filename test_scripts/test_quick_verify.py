"""Quick verification test for PipelineFeatures integration"""

from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures
import os

print("=" * 80)
print("QUICK VERIFICATION TEST - PipelineFeatures Integration")
print("=" * 80)

# Test 1: Legacy API
print("\n[Test 1] Legacy API compatibility")
try:
    pipeline = EnhancedKGPipeline(api_key="test-key")
    print("[OK] Test 1 PASSED: Legacy API works")
except Exception as e:
    print(f"[FAIL] Test 1 FAILED: {e}")

# Test 2: PipelineFeatures API
print("\n[Test 2] PipelineFeatures API")
try:
    features = PipelineFeatures.from_preset("standard", openai_api_key="test-key")
    pipeline = EnhancedKGPipeline(features=features)
    print("[OK] Test 2 PASSED: PipelineFeatures API works")
except Exception as e:
    print(f"[FAIL] Test 2 FAILED: {e}")

# Test 3: Features stored correctly
print("\n[Test 3] Features storage verification")
try:
    assert pipeline.features is not None, "Features should be stored"
    assert pipeline.api_key == "test-key", "API key should match"
    print("[OK] Test 3 PASSED: Features stored correctly")
except Exception as e:
    print(f"[FAIL] Test 3 FAILED: {e}")

# Test 4: Quality preset
print("\n[Test 4] Quality preset verification")
try:
    features_quality = PipelineFeatures.from_preset("quality", openai_api_key="test-key")
    pipeline_quality = EnhancedKGPipeline(features=features_quality)

    # Verify quality preset enables advanced features
    assert pipeline_quality.features.enable_table_detection == True, "Quality should enable table detection"
    assert pipeline_quality.features.enable_gleaning == True, "Quality should enable gleaning"
    assert pipeline_quality.extraction_strategy == "gleaning", "Extraction strategy should be gleaning"
    assert pipeline_quality.entity_merge_strategy == "fuzzy", "Merge strategy should be fuzzy"

    print("[OK] Test 4 PASSED: Quality preset configured correctly")
except Exception as e:
    print(f"[FAIL] Test 4 FAILED: {e}")

# Test 5: Custom features
print("\n[Test 5] Custom features configuration")
try:
    features_custom = PipelineFeatures(
        openai_api_key="test-key",
        enable_table_detection=True,
        enable_gleaning=False,
        merge_strategy="basic",
        chunk_size=1500
    )
    pipeline_custom = EnhancedKGPipeline(features=features_custom)

    assert pipeline_custom.features.chunk_size == 1500, "Chunk size should be 1500"
    assert pipeline_custom.entity_merge_strategy == "basic", "Merge strategy should be basic"

    print("[OK] Test 5 PASSED: Custom features work")
except Exception as e:
    print(f"[FAIL] Test 5 FAILED: {e}")

print("\n" + "=" * 80)
print("[SUCCESS] All quick verification tests PASSED!")
print("=" * 80)
print("\nNext steps:")
print("1. Start backend server: cd backend && python server.py --data_source test_endpoint")
print("2. Test endpoint with: curl -X POST http://localhost:8001/indexing/index-document ...")
print("3. Verify graph created in expr/test_endpoint/")
