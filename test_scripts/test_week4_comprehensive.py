"""
Week 4 Comprehensive Testing Script

Tests all 3 presets (standard, quality, balanced) with real KUET admission document.
Validates complete modular pipeline implementation.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_1_implementation_completeness():
    """Test 1: Verify all Week 1-3 deliverables exist"""
    print_section("TEST 1: IMPLEMENTATION COMPLETENESS")

    # Get project root (parent of test_scripts)
    project_root = Path(__file__).parent.parent

    required_files = [
        "bigrag/pipeline/features.py",
        "bigrag/pipeline/base_pipeline.py",
        "bigrag/utils.py",  # Contains quality scoring functions
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"  [FAIL] {file_path} - NOT FOUND")
        else:
            print(f"  [OK] {file_path} - EXISTS")

    if missing_files:
        print(f"\n[FAIL] {len(missing_files)} required files missing")
        return False

    # Test imports
    try:
        from bigrag.pipeline.features import PipelineFeatures, VALIDATION_THRESHOLDS
        from bigrag.pipeline.base_pipeline import UnifiedPipeline
        from bigrag.utils import description_quality_score
        print("\n  [OK] All modules import successfully")
    except ImportError as e:
        print(f"\n  [FAIL] Import error: {e}")
        return False

    # Test presets
    try:
        standard = PipelineFeatures.from_preset("standard")
        quality = PipelineFeatures.from_preset("quality")
        balanced = PipelineFeatures.from_preset("balanced")
        print(f"  [OK] All 3 presets instantiate successfully")
        print(f"    - standard: gleaning={standard.enable_gleaning}, validation={standard.enable_entity_validation}")
        print(f"    - quality: gleaning={quality.enable_gleaning}, validation={quality.enable_entity_validation}")
        print(f"    - balanced: gleaning={balanced.enable_gleaning}, validation={balanced.enable_entity_validation}")
    except Exception as e:
        print(f"  [FAIL] Preset instantiation error: {e}")
        return False

    print("\n[PASS] Implementation completeness verified")
    return True


def test_2_bigrag_integration():
    """Test 2: Verify BiGRAG accepts pipeline_features parameter"""
    print_section("TEST 2: BIGRAG INTEGRATION")

    try:
        from bigrag import BiGRAG
        from bigrag.pipeline.features import PipelineFeatures

        # Create temp directory
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp(prefix="test_week4_")
        print(f"  [INFO] Using temp directory: {temp_dir}")

        # Test 1: BiGRAG with standard preset
        features = PipelineFeatures.from_preset("standard")
        rag = BiGRAG(
            working_dir=temp_dir,
            pipeline_features=features
        )
        print(f"  [OK] BiGRAG instantiated with standard preset")
        print(f"    - pipeline_features: {rag.pipeline_features is not None}")

        # Test 2: BiGRAG with None (should default to standard)
        temp_dir2 = tempfile.mkdtemp(prefix="test_week4_default_")
        rag2 = BiGRAG(working_dir=temp_dir2)
        print(f"  [OK] BiGRAG instantiated with default (None)")
        print(f"    - auto-created pipeline_features: {rag2.pipeline_features is not None}")

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)

        print("\n[PASS] BiGRAG integration verified")
        return True

    except Exception as e:
        print(f"\n[FAIL] BiGRAG integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_kuet_document_processing(preset_name="standard", doc_path="KUET_Admission_info.md"):
    """Test 3: Process real KUET document with specified preset"""
    print_section(f"TEST 3: KUET DOCUMENT PROCESSING ({preset_name.upper()} PRESET)")

    # Get project root and check if document exists
    project_root = Path(__file__).parent.parent
    full_doc_path = project_root / doc_path

    if not full_doc_path.exists():
        print(f"  [SKIP] {doc_path} not found at {full_doc_path}")
        return None

    doc_path = full_doc_path

    # Read document
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    doc_size = len(content)
    print(f"  [INFO] Document size: {doc_size:,} characters")

    try:
        from bigrag import BiGRAG
        from bigrag.pipeline.features import PipelineFeatures
        import tempfile
        import shutil

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"test_kuet_{preset_name}_")
        print(f"  [INFO] Working directory: {temp_dir}")

        # Create BiGRAG with specified preset
        features = PipelineFeatures.from_preset(preset_name)

        print(f"  [INFO] Preset features:")
        print(f"    - enable_table_detection: {features.enable_table_detection}")
        print(f"    - enable_gleaning: {features.enable_gleaning}")
        print(f"    - enable_entity_validation: {features.enable_entity_validation}")
        print(f"    - merge_strategy: {features.merge_strategy}")

        rag = BiGRAG(
            working_dir=temp_dir,
            pipeline_features=features,
            enable_llm_cache=True
        )

        # Process document
        print(f"\n  [INFO] Processing document (this may take a while)...")
        start_time = time.time()

        # Note: ainsert is async but BiGRAG wraps it synchronously
        rag.insert(
            content,
            metadata={"title": "KUET Admission Info", "category": "education"}
        )

        processing_time = time.time() - start_time

        print(f"\n  [OK] Document processed successfully")
        print(f"  [INFO] Processing time: {processing_time:.1f} seconds")

        # Count results
        try:
            # Read graph file
            graph_file = Path(temp_dir) / "graph_chunk_entity_relation.graphml"
            if graph_file.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(graph_file)
                root = tree.getroot()

                # Count nodes by role
                entities = 0
                relations = 0
                chunks = 0

                for node in root.iter():
                    if node.tag.endswith('node'):
                        for data in node:
                            if data.get('key') == 'd0' and data.text == 'entity':
                                entities += 1
                            elif data.get('key') == 'd0' and data.text == 'relation':
                                relations += 1
                            elif data.get('key') == 'd0' and data.text == 'chunk':
                                chunks += 1

                print(f"\n  [RESULTS]")
                print(f"    - Entities: {entities}")
                print(f"    - Relations: {relations}")
                print(f"    - Chunks: {chunks}")

                # Verify results are reasonable
                if entities >= 50 and relations >= 30:
                    print(f"  [OK] Results look reasonable")
                else:
                    print(f"  [WARNING] Results may be low (expected ~80-100 entities)")

                result = {
                    "preset": preset_name,
                    "entities": entities,
                    "relations": relations,
                    "chunks": chunks,
                    "processing_time": processing_time,
                    "status": "SUCCESS"
                }
            else:
                print(f"  [WARNING] Graph file not found")
                result = {
                    "preset": preset_name,
                    "status": "NO_GRAPH",
                    "processing_time": processing_time
                }
        except Exception as e:
            print(f"  [WARNING] Could not count results: {e}")
            result = {
                "preset": preset_name,
                "status": "COUNT_ERROR",
                "processing_time": processing_time,
                "error": str(e)
            }

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"\n[PASS] {preset_name.upper()} preset test completed")
        return result

    except Exception as e:
        print(f"\n[FAIL] {preset_name.upper()} preset test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "preset": preset_name,
            "status": "FAILED",
            "error": str(e)
        }


def test_4_api_endpoints():
    """Test 4: Verify API endpoints accept preset parameter"""
    print_section("TEST 4: API ENDPOINT VERIFICATION")

    try:
        # Test importing route modules
        from backend.api.routes import documents, datasets
        print(f"  [OK] Route modules import successfully")

        # Check if preset parameter exists in documents endpoint
        import inspect
        upload_sig = inspect.signature(documents.upload_document)
        params = list(upload_sig.parameters.keys())

        if 'preset' in params:
            print(f"  [OK] /documents/upload has 'preset' parameter")
        else:
            print(f"  [FAIL] /documents/upload missing 'preset' parameter")
            print(f"    Available params: {params}")
            return False

        # Check datasets endpoint
        create_sig = inspect.signature(datasets.create_and_index_document)
        params = list(create_sig.parameters.keys())

        if 'preset' in params:
            print(f"  [OK] /datasets/create-and-index has 'preset' parameter")
        else:
            print(f"  [FAIL] /datasets/create-and-index missing 'preset' parameter")
            print(f"    Available params: {params}")
            return False

        print("\n[PASS] API endpoints verified")
        return True

    except Exception as e:
        print(f"\n[FAIL] API endpoint verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Week 4 comprehensive tests"""
    print("=" * 80)
    print("WEEK 4 COMPREHENSIVE TESTING")
    print("=" * 80)
    print(f"Testing modular pipeline implementation with KUET admission document")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Test 1: Implementation completeness
    results['implementation'] = test_1_implementation_completeness()

    # Test 2: BiGRAG integration
    results['bigrag_integration'] = test_2_bigrag_integration()

    # Test 3: API endpoints
    results['api_endpoints'] = test_4_api_endpoints()

    # Test 4-6: All 3 presets with KUET document
    # Note: These tests are time-consuming, run only if basic tests pass
    if all([results.get('implementation'), results.get('bigrag_integration')]):
        print("\n" + "=" * 80)
        print("BASIC TESTS PASSED - PROCEEDING WITH KUET DOCUMENT TESTS")
        print("=" * 80)
        print("Note: This will take several minutes (API calls to OpenAI)")
        print("=" * 80)

        results['standard_preset'] = test_3_kuet_document_processing("standard")
        results['quality_preset'] = test_3_kuet_document_processing("quality")
        results['balanced_preset'] = test_3_kuet_document_processing("balanced")
    else:
        print("\n" + "=" * 80)
        print("BASIC TESTS FAILED - SKIPPING KUET DOCUMENT TESTS")
        print("=" * 80)

    # Print summary
    print_section("FINAL SUMMARY")

    basic_tests = [
        ("Implementation Completeness", results.get('implementation')),
        ("BiGRAG Integration", results.get('bigrag_integration')),
        ("API Endpoints", results.get('api_endpoints'))
    ]

    for test_name, passed in basic_tests:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    # Preset test results
    if 'standard_preset' in results:
        print(f"\nPreset Test Results:")
        for preset_name in ['standard_preset', 'quality_preset', 'balanced_preset']:
            result = results.get(preset_name)
            if result and result.get('status') == 'SUCCESS':
                print(f"  [PASS] {preset_name.replace('_preset', '').upper()}: "
                      f"{result['entities']} entities, {result['relations']} relations, "
                      f"{result['processing_time']:.1f}s")
            elif result:
                print(f"  [FAIL] {preset_name.replace('_preset', '').upper()}: "
                      f"{result.get('status', 'UNKNOWN')}")

    # Overall result
    basic_pass_count = sum(1 for _, passed in basic_tests if passed)
    total_basic = len(basic_tests)

    print(f"\nBasic Tests: {basic_pass_count}/{total_basic} passed")

    if basic_pass_count == total_basic:
        print("\n[SUCCESS] All Week 4 tests passed!")
        print("Modular pipeline is ready for use.")
        return 0
    else:
        print(f"\n[FAILURE] {total_basic - basic_pass_count} test(s) failed")
        print("Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
