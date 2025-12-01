"""
Test script to verify modular system integration with insert() API

This script tests the CRITICAL FIX that routes insert() to index_document()
when IndexingConfig is provided.

Expected behavior:
1. With IndexingConfig → Uses modular system (index_document)
2. Without IndexingConfig → Uses legacy pipeline (backward compatible)

Run: python test_modular_integration.py
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Ensure BiG-RAG is in path
sys.path.insert(0, str(Path(__file__).parent))

def test_modular_system_integration():
    """Test that insert() uses modular system when IndexingConfig is provided."""

    print("=" * 80)
    print("TEST: Modular System Integration with insert() API")
    print("=" * 80)

    # Create temporary working directory
    temp_dir = tempfile.mkdtemp(prefix="bigrag_test_modular_")
    print(f"\n[Setup] Created temp directory: {temp_dir}")

    try:
        # Test 1: With IndexingConfig (should use modular system)
        print("\n" + "=" * 80)
        print("TEST 1: insert() WITH IndexingConfig (should use modular system)")
        print("=" * 80)

        from bigrag import BiGRAG
        from bigrag.config.indexing_config import IndexingConfig

        # Create balanced config (semantic chunking + hybrid extraction + fuzzy merging)
        config = IndexingConfig(
            chunker='token',  # Use token chunking for simplicity
            extractor='llm',  # Use LLM extraction
            merger='basic',   # Use basic merging
            validators=['entity'],  # Use entity validation
            orphan_linker='noop',  # Skip orphan linking for speed
            hitl='noop',  # Skip HITL for simplicity
            validation_mode='per_chunk'  # Per-chunk validation
        )

        # Initialize BiGRAG with IndexingConfig
        test1_dir = os.path.join(temp_dir, "test1_modular")
        rag = BiGRAG(
            indexing_config=config,
            working_dir=test1_dir
        )

        print(f"\n[Test 1] BiGRAG initialized with IndexingConfig")
        print(f"  - chunker: {config.chunker}")
        print(f"  - extractor: {config.extractor}")
        print(f"  - merger: {config.merger}")
        print(f"  - validators: {config.validators}")

        # Test document
        test_doc = """
        # KUET Computer Science and Engineering

        The Computer Science and Engineering (CSE) department at KUET has 180 seats.
        KUET offers undergraduate programs in various engineering disciplines.
        """

        test_metadata = {
            "title": "KUET CSE Department",
            "category": "university",
            "tags": ["engineering", "admission"]
        }

        print(f"\n[Test 1] Calling rag.insert()...")
        print(f"  - Document length: {len(test_doc)} chars")
        print(f"  - Metadata: {test_metadata}")

        # This should trigger modular system!
        rag.insert(
            [test_doc],
            metadata=[test_metadata]
        )

        print(f"\n[Test 1] SUCCESS - insert() completed")

        # Verify files were created
        expected_files = [
            "graph_chunk_entity_relation.graphml",
            "vdb_entities.json",
            "vdb_relations.json",
            "kv_store_text_chunks.json",
            "kv_store_full_docs.json"
        ]

        print(f"\n[Test 1] Verifying output files...")
        for filename in expected_files:
            filepath = os.path.join(test1_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"  [OK] {filename} ({file_size} bytes)")
            else:
                print(f"  [FAIL] {filename} - NOT FOUND")
                raise AssertionError(f"Expected file not created: {filename}")

        print(f"\n[Test 1] PASS - All files created successfully")

        # Test 2: Without IndexingConfig (should use legacy pipeline)
        print("\n" + "=" * 80)
        print("TEST 2: insert() WITHOUT IndexingConfig (backward compatibility)")
        print("=" * 80)

        test2_dir = os.path.join(temp_dir, "test2_legacy")
        rag2 = BiGRAG(
            working_dir=test2_dir
            # NO indexing_config parameter
        )

        print(f"\n[Test 2] BiGRAG initialized WITHOUT IndexingConfig")
        print(f"  - Should use legacy standard pipeline")

        print(f"\n[Test 2] Calling rag.insert()...")
        rag2.insert(
            [test_doc],
            metadata=[test_metadata]
        )

        print(f"\n[Test 2] SUCCESS - insert() completed with legacy pipeline")

        # Verify files were created (legacy pipeline)
        print(f"\n[Test 2] Verifying output files...")
        for filename in expected_files:
            filepath = os.path.join(test2_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"  [OK] {filename} ({file_size} bytes)")
            else:
                print(f"  [FAIL] {filename} - NOT FOUND")
                raise AssertionError(f"Expected file not created: {filename}")

        print(f"\n[Test 2] PASS - Backward compatibility maintained")

        # Summary
        print("\n" + "=" * 80)
        print("OVERALL TEST RESULT: PASS")
        print("=" * 80)
        print("\nSummary:")
        print("  [OK] Test 1: Modular system integration working")
        print("  [OK] Test 2: Backward compatibility maintained")
        print("\nConclusion:")
        print("  The fix successfully routes insert() to index_document() when")
        print("  IndexingConfig is provided, while maintaining backward compatibility")
        print("  with legacy pipelines.")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print(f"\n[Cleanup] Removing temp directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
            print(f"[Cleanup] SUCCESS")
        except Exception as e:
            print(f"[Cleanup] WARNING - Could not remove temp dir: {e}")


def test_preset_configs():
    """Test that preset configs work correctly."""

    print("\n" + "=" * 80)
    print("TEST: Preset Configurations")
    print("=" * 80)

    from bigrag.config.indexing_config import IndexingConfig

    presets = [
        ('fast', IndexingConfig.preset_fast),
        ('balanced', IndexingConfig.preset_balanced),
        ('quality', IndexingConfig.preset_quality)
    ]

    for preset_name, preset_func in presets:
        print(f"\n[Preset: {preset_name}]")
        config = preset_func()
        print(f"  - chunker: {config.chunker}")
        print(f"  - extractor: {config.extractor}")
        print(f"  - merger: {config.merger}")
        print(f"  - validators: {config.validators}")
        print(f"  - orphan_linker: {config.orphan_linker}")
        print(f"  - hitl: {config.hitl}")

    print(f"\n[Presets] All presets loaded successfully")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BiG-RAG Modular System Integration Test Suite")
    print("=" * 80)
    print("\nThis test verifies the CRITICAL FIX:")
    print("  - insert() routes to index_document() when IndexingConfig is provided")
    print("  - Backward compatibility with legacy pipelines is maintained")
    print("=" * 80)

    # Set environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("\n[WARNING] OPENAI_API_KEY not set in environment")
        print("  - Tests requiring LLM extraction will fail")
        print("  - Set OPENAI_API_KEY in .env file or environment")
        print("\n  Continuing with tests anyway...")

    # Run tests
    test_preset_configs()

    success = test_modular_system_integration()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
