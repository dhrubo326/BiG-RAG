"""
Test script for the new modular indexing system.

This script verifies that the BiGRAG class can be initialized with IndexingConfig
and that the index_document() method works correctly with different strategies.
"""

import asyncio
import os
from bigrag import BiGRAG
from bigrag.config import IndexingConfig


async def test_modular_indexing():
    """Test the modular indexing system with a simple document."""

    # Load API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        try:
            with open('openai_api_key.txt') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print("[ERROR] OpenAI API key not found. Please set OPENAI_API_KEY or create openai_api_key.txt")
            return False

    # Test document
    test_document = """
    # KUET Computer Science and Engineering

    The Department of Computer Science and Engineering (CSE) at Khulna University of
    Engineering & Technology (KUET) is one of the leading engineering departments in Bangladesh.

    ## Admission Information
    - Total Seats: 120
    - Minimum GPA: 4.50 in SSC
    - Minimum GPA: 4.50 in HSC
    - Admission Test Date: January 15, 2025

    ## Faculty
    The department has 45 faculty members, including 12 professors and 20 assistant professors.
    """

    metadata = {
        "title": "KUET CSE Department Overview",
        "category": "Education",
        "tags": ["university", "engineering", "computer-science"]
    }

    print("\n" + "="*80)
    print("MODULAR INDEXING SYSTEM TEST")
    print("="*80 + "\n")

    # Test 1: Fast preset (token chunking, strict extraction, no validation)
    print("[TEST 1] Fast Preset (minimal processing)")
    print("-" * 80)

    config_fast = IndexingConfig.preset_fast(
        openai_api_key=api_key,
        dataset_path="./expr/test_modular_fast"
    )

    rag_fast = BiGRAG(
        indexing_config=config_fast,
        working_dir="./expr/test_modular_fast"
    )

    try:
        result_fast = await rag_fast.index_document(test_document, metadata)

        print(f"\n[RESULTS - Fast Preset]")
        print(f"  Chunks: {result_fast['statistics']['total_chunks']}")
        print(f"  Entities: {result_fast['statistics']['total_entities']}")
        print(f"  Relations: {result_fast['statistics']['total_relations']}")
        print(f"  Validation: {result_fast['validation']['status']}")
        print("[PASS] Fast preset test completed successfully!\n")
    except Exception as e:
        print(f"[FAIL] Fast preset test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Balanced preset (semantic chunking, gleaning, fuzzy merging)
    print("\n[TEST 2] Balanced Preset (recommended for production)")
    print("-" * 80)

    config_balanced = IndexingConfig.preset_balanced(
        openai_api_key=api_key,
        dataset_path="./expr/test_modular_balanced"
    )

    rag_balanced = BiGRAG(
        indexing_config=config_balanced,
        working_dir="./expr/test_modular_balanced"
    )

    try:
        result_balanced = await rag_balanced.index_document(test_document, metadata)

        print(f"\n[RESULTS - Balanced Preset]")
        print(f"  Chunks: {result_balanced['statistics']['total_chunks']}")
        print(f"  Entities: {result_balanced['statistics']['total_entities']}")
        print(f"  Relations: {result_balanced['statistics']['total_relations']}")
        print(f"  Synthetic Relations: {result_balanced['statistics']['synthetic_relations']}")
        print(f"  Orphan Entities: {result_balanced['statistics']['orphan_entities']}")
        print(f"  Validation: {result_balanced['validation']['status']}")
        print("[PASS] Balanced preset test completed successfully!\n")
    except Exception as e:
        print(f"[FAIL] Balanced preset test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Custom configuration
    print("\n[TEST 3] Custom Configuration")
    print("-" * 80)

    config_custom = IndexingConfig(
        chunker="token",
        extractor="strict",
        validators=[],  # No validation
        merger="basic",
        hitl="noop",
        orphan_linker="noop",
        openai_api_key=api_key,
        dataset_path="./expr/test_modular_custom"
    )

    rag_custom = BiGRAG(
        indexing_config=config_custom,
        working_dir="./expr/test_modular_custom"
    )

    try:
        result_custom = await rag_custom.index_document(test_document, metadata)

        print(f"\n[RESULTS - Custom Configuration]")
        print(f"  Chunks: {result_custom['statistics']['total_chunks']}")
        print(f"  Entities: {result_custom['statistics']['total_entities']}")
        print(f"  Relations: {result_custom['statistics']['total_relations']}")
        print(f"  Validation: {result_custom['validation']['status']}")
        print("[PASS] Custom configuration test completed successfully!\n")
    except Exception as e:
        print(f"[FAIL] Custom configuration test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nThe modular indexing system is working correctly!")
    print("You can now use:")
    print("  - IndexingConfig.preset_fast() for quick indexing")
    print("  - IndexingConfig.preset_balanced() for production (recommended)")
    print("  - IndexingConfig.preset_quality() for highest accuracy")
    print("  - Custom IndexingConfig for fine-grained control")
    print("\n")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_modular_indexing())
    exit(0 if success else 1)
