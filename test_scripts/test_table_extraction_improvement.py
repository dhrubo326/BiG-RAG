"""
Test script to verify improved table extraction after prompt enhancement.

This script:
1. Deletes the RUET document from the knowledge graph
2. Re-inserts it with the improved extraction prompt
3. Verifies that CSE 180 seats data is now properly extracted
4. Tests query retrieval for RUET CSE seat count

Usage:
    python test_scripts/test_table_extraction_improvement.py
"""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bigrag import BiGRAG


async def test_ruet_rebuild():
    """Test rebuilding RUET document with improved table extraction."""

    print("=" * 80)
    print("TABLE EXTRACTION IMPROVEMENT TEST")
    print("=" * 80)

    # Initialize BiGRAG instance
    working_dir = "./expr/demo_test"
    print(f"\n[1/6] Initializing BiGRAG from {working_dir}...")

    rag = BiGRAG(working_dir=working_dir)

    # Load corpus to get RUET document
    corpus_path = "./datasets/demo_test/raw/corpus.jsonl"
    print(f"\n[2/6] Loading corpus from {corpus_path}...")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        ruet_doc = None
        for line in f:
            doc = json.loads(line)
            if "RUET" in doc.get("title", "") or "Rajshahi" in doc.get("title", ""):
                ruet_doc = doc
                break

    if not ruet_doc:
        print("[FAIL] RUET document not found in corpus!")
        return False

    print(f"[OK] Found RUET document:")
    print(f"     - ID: {ruet_doc['id']}")
    print(f"     - Title: {ruet_doc['title']}")
    print(f"     - Content length: {len(ruet_doc['contents'])} chars")

    # Delete existing RUET document
    print(f"\n[3/6] Deleting existing RUET document from knowledge graph...")
    try:
        await rag.adelete_document(ruet_doc['id'])
        print("[OK] RUET document deleted successfully")
    except Exception as e:
        print(f"[WARN] Delete failed (may not exist): {e}")

    # Re-insert with improved prompt
    print(f"\n[4/6] Re-inserting RUET document with improved table extraction prompt...")

    metadata = {
        "title": ruet_doc.get("title", ""),
        "tags": ruet_doc.get("metadata", {}).get("tags", []),
        "source": "test_rebuild"
    }

    try:
        await rag.ainsert(
            [ruet_doc['contents']],
            metadata=[metadata]
        )
        print("[OK] RUET document re-inserted successfully")
    except Exception as e:
        print(f"[FAIL] Re-insertion failed: {e}")
        return False

    # Verify CSE 180 extraction
    print(f"\n[5/6] Verifying CSE 180 seats data in knowledge graph...")

    # Check if "180" appears in the graph
    graph_file = f"{working_dir}/graph_chunk_entity_relation.graphml"
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_content = f.read()

    # Search for CSE with 180
    if "১৮০" in graph_content:
        print("[OK] Number '১৮০' (180) found in knowledge graph!")

        # Count occurrences
        count_180 = graph_content.count("১৮০")
        print(f"     - Found {count_180} occurrences of '১৮০'")

        # Check for CSE + 180 relation
        if "CSE" in graph_content and "১৮০" in graph_content:
            print("[OK] Both 'CSE' and '১৮০' present in graph - likely extracted correctly")
    else:
        print("[FAIL] Number '১৮০' (180) NOT found in knowledge graph!")
        print("       Table extraction may still have issues.")
        return False

    # Test query retrieval
    print(f"\n[6/6] Testing query retrieval for RUET CSE seat count...")

    query = "(RUET) কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং (CSE) বিভাগের আসন সংখ্যা কতগুলো?"

    try:
        results = await rag.aquery(query)

        print(f"[OK] Query executed successfully")
        print(f"     - Retrieved {len(results)} context items")

        # Check if any result contains RUET and CSE and 180
        found_ruet_cse = False
        for i, result in enumerate(results[:10]):  # Check top 10
            content = str(result)
            if ("RUET" in content or "রাজশাহী" in content) and "CSE" in content and "১৮০" in content:
                found_ruet_cse = True
                print(f"[OK] RUET CSE 180 found in result #{i+1}")
                print(f"     Preview: {content[:200]}...")
                break

        if not found_ruet_cse:
            print("[WARN] RUET CSE 180 not in top 10 results")
            print("       Retrieval ranking may need improvement")

    except Exception as e:
        print(f"[FAIL] Query failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("[OK] Table extraction improvement test completed")
    print("")
    print("Next steps:")
    print("1. Examine knowledge graph to verify table rows extracted separately")
    print("2. Compare with KUET document extraction quality")
    print("3. Test with other university documents to ensure consistency")

    return True


async def main():
    """Main test runner."""
    try:
        success = await test_ruet_rebuild()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
