"""
Test script for unified pipeline indexing with KUET document.
Tests all 3 presets (standard, quality, balanced) with real document.

NO EMOJIS - Windows console compatibility.
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_results(preset_name, stats, processing_time):
    """Print processing results"""
    print(f"\n[{preset_name.upper()} PRESET RESULTS]")
    print(f"  Processing time: {processing_time:.1f} seconds")
    print(f"  Chunks created: {stats.get('chunks', 0)}")
    print(f"  Entities extracted: {stats.get('entities', 0)}")
    print(f"  Relations extracted: {stats.get('relations', 0)}")
    if stats.get('validation'):
        print(f"  Validation status: {stats['validation'].get('status', 'N/A')}")


async def test_preset(preset_name, doc_content, doc_metadata):
    """Test indexing with a specific preset"""
    from bigrag import BiGRAG
    from bigrag.pipeline.features import PipelineFeatures
    import tempfile
    import shutil

    print_header(f"Testing {preset_name.upper()} Preset")

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix=f"test_unified_{preset_name}_")
    print(f"[INFO] Working directory: {temp_dir}")

    try:
        # Create pipeline features
        features = PipelineFeatures.from_preset(
            preset_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY")
        )

        print(f"[INFO] Pipeline configuration:")
        print(f"  - enable_table_detection: {features.enable_table_detection}")
        print(f"  - enable_gleaning: {features.enable_gleaning}")
        print(f"  - enable_entity_validation: {features.enable_entity_validation}")
        print(f"  - enable_relation_validation: {features.enable_relation_validation}")
        print(f"  - merge_strategy: {features.merge_strategy}")
        print(f"  - validation_strictness: {features.validation_strictness}")

        # Create BiGRAG instance
        rag = BiGRAG(
            working_dir=temp_dir,
            pipeline_features=features,
            enable_llm_cache=True
        )

        # Process document
        print(f"\n[INFO] Indexing document...")
        start_time = time.time()

        # Use ainsert directly to capture pipeline results
        await rag.ainsert(
            doc_content,
            metadata=doc_metadata
        )

        processing_time = time.time() - start_time

        # Count results from graph file
        stats = {"chunks": 0, "entities": 0, "relations": 0}
        graph_file = Path(temp_dir) / "graph_chunk_entity_relation.graphml"

        if graph_file.exists():
            import xml.etree.ElementTree as ET
            tree = ET.parse(graph_file)
            root = tree.getroot()

            # Count nodes by role
            for node in root.iter():
                if node.tag.endswith('node'):
                    for data in node:
                        if data.get('key') == 'd0':
                            role = data.text
                            if role == 'entity':
                                stats['entities'] += 1
                            elif role == 'relation':
                                stats['relations'] += 1
                            elif role == 'chunk':
                                stats['chunks'] += 1

        print_results(preset_name, stats, processing_time)

        # Verify results are reasonable
        if stats['entities'] >= 30 and stats['relations'] >= 15:
            print(f"\n[PASS] {preset_name.upper()} preset indexing successful")
            return True, stats, processing_time
        else:
            print(f"\n[WARNING] {preset_name.upper()} preset produced low results")
            print(f"  Expected: 30+ entities, 15+ relations")
            print(f"  Got: {stats['entities']} entities, {stats['relations']} relations")
            return False, stats, processing_time

    except Exception as e:
        print(f"\n[FAIL] {preset_name.upper()} preset failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, 0.0

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Run all preset tests"""
    print("=" * 80)
    print("UNIFIED PIPELINE INDEXING TEST")
    print("=" * 80)
    print(f"Testing with KUET Admission document")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY environment variable not set")
        print("Please set it before running tests:")
        print("  set OPENAI_API_KEY=your-key-here")
        return 1

    # Load document
    project_root = Path(__file__).parent.parent
    doc_path = project_root / "KUET_Admission_info.md"

    if not doc_path.exists():
        print(f"\n[ERROR] Document not found: {doc_path}")
        return 1

    with open(doc_path, 'r', encoding='utf-8') as f:
        doc_content = f.read()

    doc_metadata = {
        "title": "KUET Admission Info 2024-2025",
        "category": "education",
        "tags": ["university", "admission", "engineering"]
    }

    print(f"\n[INFO] Document loaded: {len(doc_content)} characters")

    # Test all 3 presets
    results = {}

    # Test 1: Standard preset (fast)
    success, stats, time_taken = await test_preset("standard", doc_content, doc_metadata)
    results['standard'] = {'success': success, 'stats': stats, 'time': time_taken}

    # Test 2: Balanced preset (medium)
    success, stats, time_taken = await test_preset("balanced", doc_content, doc_metadata)
    results['balanced'] = {'success': success, 'stats': stats, 'time': time_taken}

    # Test 3: Quality preset (slow, accurate)
    success, stats, time_taken = await test_preset("quality", doc_content, doc_metadata)
    results['quality'] = {'success': success, 'stats': stats, 'time': time_taken}

    # Print summary
    print_header("FINAL SUMMARY")

    for preset_name, result in results.items():
        status = "[PASS]" if result['success'] else "[FAIL]"
        stats = result['stats']
        time_taken = result['time']

        print(f"{status} {preset_name.upper()}: "
              f"{stats.get('entities', 0)} entities, "
              f"{stats.get('relations', 0)} relations, "
              f"{time_taken:.1f}s")

    # Overall result
    all_passed = all(r['success'] for r in results.values())

    if all_passed:
        print("\n[SUCCESS] All presets passed!")
        print("\nUnified pipeline is working correctly.")
        print("You can now use it via BiGRAG API or directly.")
        return 0
    else:
        print("\n[FAILURE] Some presets failed")
        print("Check error messages above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
