"""
Test Production Pipeline Integration in BiGRAG.ainsert()

This script tests the newly implemented _process_document_with_production_pipeline() method
to verify that BiGRAG can use the production pipeline when use_production_pipeline=True.

Usage:
    python test_production_integration.py
"""

import asyncio
import os
import shutil
from pathlib import Path
from bigrag import BiGRAG


async def test_production_integration():
    """Test BiGRAG with production pipeline enabled."""

    print("\n" + "="*80)
    print("PRODUCTION PIPELINE INTEGRATION TEST")
    print("="*80)

    # Step 1: Setup test directory
    working_dir = "./expr/production_integration_test"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    print(f"\n[Setup] Working directory: {working_dir}")

    # Step 2: Check API key
    api_key_file = "openai_api_key.txt"
    if not os.path.exists(api_key_file):
        print(f"\n[ERROR] {api_key_file} not found!")
        print("Production pipeline requires OpenAI API key.")
        return False

    print(f"[Setup] API key file found: {api_key_file}")

    # Step 3: Load test document (KUET admission)
    kuet_file = "KUET_Admission_info.md"
    if not os.path.exists(kuet_file):
        print(f"\n[ERROR] {kuet_file} not found!")
        return False

    with open(kuet_file, 'r', encoding='utf-8') as f:
        kuet_doc = f.read()

    print(f"[Setup] Test document loaded: {len(kuet_doc)} characters")

    # Step 4: Initialize BiGRAG with production pipeline ENABLED
    print("\n[Test] Initializing BiGRAG with production pipeline ENABLED...")
    rag = BiGRAG(
        working_dir=working_dir,
        use_production_pipeline=True,  # ENABLE PRODUCTION PIPELINE
        production_pipeline_config={
            "validation_level": "MODERATE",  # 95%+ accuracy
            "enable_entity_linking": True,
            "extraction_mode": "semi_structured"
        }
    )
    print("[Test] BiGRAG initialized with production pipeline")

    # Step 5: Insert document (should use production pipeline)
    print("\n[Test] Inserting document via BiGRAG.ainsert()...")
    print("This should trigger _process_document_with_production_pipeline()")
    print("-" * 80)

    try:
        await rag.ainsert(
            [kuet_doc],
            metadata=[{
                'title': 'KUET Admission 2024-25',
                'category': 'university_admission',
                'tags': ['engineering', 'admission', 'KUET', 'Bangladesh'],
            }]
        )
        print("\n[Test] Document insertion complete!")

    except Exception as e:
        print(f"\n[ERROR] Document insertion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Verify all files were created
    print("\n[Verify] Checking created files...")
    expected_files = [
        "graph_chunk_entity_relation.graphml",
        "vdb_entities.json",
        "vdb_relations.json",
        "vdb_chunks.json",
        "kv_store_text_chunks.json",
        "kv_store_full_docs.json",
    ]

    all_created = True
    for filename in expected_files:
        filepath = Path(working_dir) / filename
        if filepath.exists():
            size = filepath.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
            print(f"  [OK] {filename} ({size_str})")
        else:
            print(f"  [FAIL] {filename} NOT FOUND")
            all_created = False

    if not all_created:
        print("\n[FAIL] Some files were not created!")
        return False

    # Step 7: Test query
    print("\n[Verify] Testing query functionality...")
    from bigrag.base import QueryParam

    query = "How many seats in CSE department?"
    try:
        contexts = await rag.aquery(
            query=query,
            param=QueryParam(mode="hybrid", top_k=3)
        )
        print(f"  Query: {query}")
        print(f"  Retrieved {len(contexts)} contexts")
        print(f"  [OK] Query successful")

    except Exception as e:
        print(f"  [ERROR] Query failed: {e}")
        return False

    # Step 8: Success summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nWorking directory: {working_dir}")
    print(f"Production pipeline: ENABLED")
    print(f"Files created: {len(expected_files)}/{len(expected_files)}")
    print(f"Query test: PASSED")
    print("\n[SUCCESS] Production pipeline integration working correctly!")
    print("\nNext: Compare with standard pipeline to verify backward compatibility")

    return True


async def test_backward_compatibility():
    """Test BiGRAG with production pipeline DISABLED (standard pipeline)."""

    print("\n" + "="*80)
    print("BACKWARD COMPATIBILITY TEST")
    print("="*80)

    # Step 1: Setup test directory
    working_dir = "./expr/standard_pipeline_test"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    print(f"\n[Setup] Working directory: {working_dir}")

    # Step 2: Load test document
    kuet_file = "KUET_Admission_info.md"
    with open(kuet_file, 'r', encoding='utf-8') as f:
        kuet_doc = f.read()

    print(f"[Setup] Test document loaded: {len(kuet_doc)} characters")

    # Step 3: Initialize BiGRAG with production pipeline DISABLED (default)
    print("\n[Test] Initializing BiGRAG with production pipeline DISABLED (default)...")
    rag = BiGRAG(
        working_dir=working_dir,
        use_production_pipeline=False,  # STANDARD PIPELINE (default)
    )
    print("[Test] BiGRAG initialized with standard pipeline")

    # Step 4: Insert document (should use standard pipeline)
    print("\n[Test] Inserting document via BiGRAG.ainsert()...")
    print("This should use the existing standard extraction code")
    print("-" * 80)

    try:
        await rag.ainsert(
            [kuet_doc],
            metadata=[{
                'title': 'KUET Admission 2024-25',
                'category': 'university_admission',
                'tags': ['engineering', 'admission', 'KUET', 'Bangladesh'],
            }]
        )
        print("\n[Test] Document insertion complete!")

    except Exception as e:
        print(f"\n[ERROR] Document insertion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Verify files created
    print("\n[Verify] Checking created files...")
    expected_files = [
        "graph_chunk_entity_relation.graphml",
        "vdb_entities.json",
        "vdb_relations.json",
        "vdb_chunks.json",
        "kv_store_text_chunks.json",
        "kv_store_full_docs.json",
    ]

    all_created = True
    for filename in expected_files:
        filepath = Path(working_dir) / filename
        if filepath.exists():
            size = filepath.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
            print(f"  [OK] {filename} ({size_str})")
        else:
            print(f"  [FAIL] {filename} NOT FOUND")
            all_created = False

    if not all_created:
        print("\n[FAIL] Some files were not created!")
        return False

    print("\n[SUCCESS] Backward compatibility verified - standard pipeline still works!")
    return True


async def main():
    """Main entry point."""

    try:
        # Test 1: Production pipeline integration
        success1 = await test_production_integration()

        # Test 2: Backward compatibility
        success2 = await test_backward_compatibility()

        if success1 and success2:
            print("\n" + "="*80)
            print("ALL TESTS PASSED")
            print("="*80)
            print("\n[OK] Production pipeline integration complete")
            print("[OK] Backward compatibility maintained")
            print("\nYou can now use BiGRAG with production pipeline by setting:")
            print("  use_production_pipeline=True")
            return 0
        else:
            print("\n" + "="*80)
            print("SOME TESTS FAILED")
            print("="*80)
            return 1

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Unexpected failure: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
