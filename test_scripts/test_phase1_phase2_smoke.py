"""
Phase 1 + Phase 2 Comprehensive Smoke Tests

Tests all wrapper modules and full pipeline implementation.
This verifies that the complete modular pipeline works end-to-end.
"""

import sys
import os
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_all_modules():
    """Test 1: Can we import all pipeline modules?"""
    print("[TEST 1] Importing all pipeline modules...")
    modules_tested = 0
    modules_total = 8

    try:
        from bigrag.pipeline.features import PipelineFeatures
        print("  [OK] PipelineFeatures")
        modules_tested += 1
    except Exception as e:
        print(f"  [FAIL] PipelineFeatures: {e}")

    try:
        from bigrag.pipeline.chunkers import TokenChunker, TableChunker
        print("  [OK] TokenChunker, TableChunker")
        modules_tested += 1
    except Exception as e:
        print(f"  [FAIL] Chunkers: {e}")

    try:
        from bigrag.pipeline.extractors import LLMExtractor
        print("  [OK] LLMExtractor")
        modules_tested += 1
    except Exception as e:
        print(f"  [FAIL] LLMExtractor: {e}")

    try:
        from bigrag.pipeline.validators import EntityValidator
        print("  [OK] EntityValidator")
        modules_tested += 1
    except Exception as e:
        print(f"  [FAIL] EntityValidator: {e}")

    try:
        from bigrag.pipeline.mergers import BasicMerger, FuzzyMerger
        print("  [OK] BasicMerger, FuzzyMerger")
        modules_tested += 1
    except Exception as e:
        print(f"  [FAIL] Mergers: {e}")

    try:
        from bigrag.pipeline.postprocessors import OrphanLinker
        print("  [OK] OrphanLinker")
        modules_tested += 1
    except Exception as e:
        print(f"  [FAIL] OrphanLinker: {e}")

    try:
        from bigrag.pipeline import UnifiedPipeline
        print("  [OK] UnifiedPipeline")
        modules_tested += 1
    except Exception as e:
        print(f"  [FAIL] UnifiedPipeline: {e}")

    try:
        from bigrag.utils import description_quality_score
        print("  [OK] description_quality_score")
        modules_tested += 1
    except Exception as e:
        print(f"  [FAIL] description_quality_score: {e}")

    return modules_tested == modules_total


def test_instantiate_all_presets():
    """Test 2: Can we instantiate all 3 presets?"""
    print("\n[TEST 2] Instantiating all presets...")
    from bigrag.pipeline import PipelineFeatures, UnifiedPipeline

    presets_tested = 0
    for preset_name in ["standard", "quality", "balanced"]:
        try:
            features = PipelineFeatures.from_preset(preset_name, openai_api_key="test-key")
            pipeline = UnifiedPipeline(features, dataset_path=None)
            print(f"  [OK] {preset_name} preset instantiated")
            presets_tested += 1
        except Exception as e:
            print(f"  [FAIL] {preset_name} preset failed: {e}")
            import traceback
            traceback.print_exc()

    return presets_tested == 3


async def test_token_chunker():
    """Test 3: Does TokenChunker work?"""
    print("\n[TEST 3] Testing TokenChunker...")
    try:
        from bigrag.pipeline.chunkers import TokenChunker

        chunker = TokenChunker(chunk_size=100, overlap=10)
        test_text = "This is a test document. " * 50  # ~500 words

        chunks = await chunker.chunk(test_text, metadata={"title": "Test"})

        assert len(chunks) > 0, "Expected at least one chunk"
        assert all(isinstance(c, dict) for c in chunks), "All chunks should be dicts"
        assert all('content' in c for c in chunks), "All chunks should have content"
        assert all('chunk_id' in c for c in chunks), "All chunks should have chunk_id"

        print(f"  [OK] Created {len(chunks)} chunks")
        return True

    except Exception as e:
        print(f"  [FAIL] TokenChunker failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_basic_merger():
    """Test 4: Does BasicMerger work?"""
    print("\n[TEST 4] Testing BasicMerger...")
    try:
        from bigrag.pipeline.mergers import BasicMerger

        merger = BasicMerger()

        # Create test entities with duplicates
        entities = [
            {"entity_name": "Albert Einstein", "entity_type": "Person", "weight": 1.0},
            {"entity_name": "albert einstein", "entity_type": "Person", "weight": 1.0},  # Duplicate
            {"entity_name": "Physics", "entity_type": "Field", "weight": 1.0}
        ]

        merged = await merger.merge(entities, relations=[])

        assert len(merged) < len(entities), "Expected deduplication to reduce count"
        print(f"  [OK] Merged {len(entities)} -> {len(merged)} entities")
        return True

    except Exception as e:
        print(f"  [FAIL] BasicMerger failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_pipeline_stub():
    """Test 5: Does full pipeline process_document work (minimal test)?"""
    print("\n[TEST 5] Testing full pipeline with minimal document...")
    try:
        from bigrag.pipeline import PipelineFeatures, UnifiedPipeline

        # Use standard preset (simplest)
        features = PipelineFeatures.from_preset("standard", openai_api_key="test-key")
        pipeline = UnifiedPipeline(features)

        # Test with very short document
        test_doc = "Albert Einstein developed the theory of relativity."

        result = await pipeline.process_document(test_doc, metadata={"title": "Test"})

        # Verify result structure
        assert 'chunks' in result, "Result should have chunks"
        assert 'entities' in result, "Result should have entities"
        assert 'relations' in result, "Result should have relations"
        assert 'statistics' in result, "Result should have statistics"
        assert 'pipeline_metadata' in result, "Result should have pipeline_metadata"

        print(f"  [OK] Pipeline processed document:")
        print(f"       Chunks: {result['statistics']['total_chunks']}")
        print(f"       Entities: {result['statistics']['total_entities']}")
        print(f"       Relations: {result['statistics']['total_relations']}")
        print(f"       Preset: {result['pipeline_metadata']['preset']}")

        return True

    except Exception as e:
        print(f"  [FAIL] Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 1+2 smoke tests."""
    print("=" * 70)
    print("Phase 1 + Phase 2 Comprehensive Smoke Tests")
    print("=" * 70)

    # Sync tests
    results = {
        "Import all modules": test_import_all_modules(),
        "Instantiate all presets": test_instantiate_all_presets()
    }

    # Async tests
    loop = asyncio.get_event_loop()
    results["TokenChunker"] = loop.run_until_complete(test_token_chunker())
    results["BasicMerger"] = loop.run_until_complete(test_basic_merger())
    results["Full pipeline"] = loop.run_until_complete(test_full_pipeline_stub())

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All Phase 1+2 smoke tests passed!")
        print("Phase 1 + Phase 2 implementation complete and verified.")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} tests failed")
        print("Fix failures before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
