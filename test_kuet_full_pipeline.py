"""
Test Complete Production Pipeline with KUET Document

Tests:
1. Educational pipeline end-to-end
2. Extraction (table + paragraph facts)
3. Entity canonicalization (KUET departments)
4. Validation (numeric + consistency)
5. Bipartite graph construction
6. Vector indexing
7. Query retrieval
"""

import asyncio
import sys
import codecs
from pathlib import Path

# Windows UTF-8 fix
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from bigrag.educational_pipeline import build_educational_kg


async def test_kuet_pipeline():
    """Test complete pipeline with KUET admission document."""

    print("=" * 80)
    print("KUET ADMISSION DOCUMENT - FULL PRODUCTION PIPELINE TEST")
    print("=" * 80)
    print()

    # Step 1: Load KUET document
    print("[STEP 1] Loading KUET document...")
    kuet_path = Path("KUET_Admission_info.md")

    if not kuet_path.exists():
        print(f"[ERROR] File not found: {kuet_path}")
        return

    kuet_content = kuet_path.read_text(encoding='utf-8')
    print(f"  Document size: {len(kuet_content)} characters")
    print(f"  Lines: {len(kuet_content.splitlines())}")
    print()

    # Step 2: Prepare metadata
    print("[STEP 2] Preparing metadata...")
    metadata = {
        'title': 'KUET Admission 2024-25',
        'category': 'university_admission',
        'tags': ['kuet', 'engineering', 'admission', 'bangladesh'],
        'university': 'KUET',
        'academic_year': '2024-2025'
    }
    print(f"  Metadata: {metadata}")
    print()

    # Step 3: Load API key
    print("[STEP 3] Loading OpenAI API key...")
    try:
        api_key_path = Path("openai_api_key.txt")
        if api_key_path.exists():
            api_key = api_key_path.read_text().strip()
            print(f"  API key loaded: {api_key[:8]}...")
        else:
            print("[ERROR] openai_api_key.txt not found")
            return
    except Exception as e:
        print(f"[ERROR] Failed to load API key: {e}")
        return
    print()

    # Step 4: Run educational pipeline
    print("[STEP 4] Running production pipeline...")
    print("  This will:")
    print("  - Extract tables using GPT-4o")
    print("  - Extract paragraphs with constrained LLM")
    print("  - Canonicalize entities (KUET departments)")
    print("  - Validate numeric accuracy (99%+ target)")
    print("  - Validate cross-chunk consistency")
    print("  - Build bipartite graph")
    print("  - Index to vector DBs")
    print()
    print("-" * 80)

    try:
        rag, results = await build_educational_kg(
            markdown_documents=[kuet_content],
            document_metadata=[metadata],
            api_key=api_key,
            working_dir="./expr/kuet_test",
            validation_level="STRICT",  # 99%+ accuracy required
            enable_entity_linking=True,
            chunk_token_size=1200,
            chunk_overlap=100
        )

        print("-" * 80)
        print()

        # Step 5: Check results
        print("[STEP 5] Analyzing results...")

        if not results or len(results) == 0:
            print("[ERROR] No results returned from pipeline")
            print("This likely means document validation failed or extraction error occurred.")
            print("\nCheck logs above for:")
            print("  - Validation failures")
            print("  - Extraction errors")
            print("  - Graph construction issues")
            return False

        result = results[0]

        validation = result['validation']
        stats = result['statistics']

        print(f"\n{'='*80}")
        print("EXTRACTION RESULTS")
        print(f"{'='*80}")
        print(f"  Total Entities: {stats['total_entities']}")
        print(f"  Total Relations: {stats['total_relations']}")
        print(f"  Total Chunks: {stats['total_chunks']}")
        print(f"    - Table chunks: {stats['table_chunks']}")
        print(f"    - Paragraph chunks: {stats['paragraph_chunks']}")
        print(f"  Entity Merge Reduction: {stats['entity_merge_reduction']}")
        print()

        # Validation results
        print(f"{'='*80}")
        print("VALIDATION RESULTS")
        print(f"{'='*80}")

        numeric = validation['numeric']
        consistency = validation['consistency']
        overall = validation['overall_status']

        print(f"\n[NUMERIC VALIDATION]")
        print(f"  Status: {numeric['status']}")
        print(f"  Coverage: {numeric['numeric_coverage']:.2%}")
        print(f"  Hallucination Rate: {numeric['hallucination_rate']:.2%}")
        print(f"  Source Numbers: {numeric['total_source_numbers']}")
        print(f"  KG Numbers: {numeric['total_kg_numbers']}")
        print(f"  Matched: {numeric['matched_numbers']}")

        if numeric['missing_numbers']:
            print(f"  [WARN] Missing Numbers ({len(numeric['missing_numbers'])}): {numeric['missing_numbers'][:10]}")

        if numeric.get('frequency_mismatches'):
            print(f"\n  [WARN] Frequency Mismatches:")
            for num, mismatch in list(numeric['frequency_mismatches'].items())[:5]:
                print(f"    {num}: source={mismatch['source']}, kg={mismatch['kg']}, deficit={mismatch['deficit']}")

        print(f"\n[CONSISTENCY VALIDATION]")
        print(f"  Status: {consistency['status']}")
        print(f"  Consistency Score: {consistency['consistency_score']:.2%}")
        print(f"  Total Issues: {consistency['total_issues']}")
        print(f"    - Entity Conflicts: {consistency['entity_conflicts']}")
        print(f"    - Numeric Conflicts: {consistency['numeric_conflicts']}")
        print(f"    - Relation Contradictions: {consistency['relation_contradictions']}")
        print(f"    - Reference Errors: {consistency['reference_errors']}")

        print(f"\n[OVERALL STATUS]")
        if overall == 'PASS':
            print(f"  Status: PASS (Production Ready)")
        else:
            print(f"  Status: FAIL (Needs Review)")

        print()

        # Step 6: Sample entities
        print(f"{'='*80}")
        print("SAMPLE EXTRACTED ENTITIES (First 10)")
        print(f"{'='*80}")

        for i, entity in enumerate(result['entities'][:10], 1):
            print(f"\n{i}. {entity['entity_name']}")
            print(f"   Type: {entity['entity_type']}")
            print(f"   Description: {entity['description'][:80]}...")
            print(f"   Weight: {entity.get('weight', 0)}")
            print(f"   Source: {entity.get('source_id', 'unknown')}")

        print()

        # Step 7: Sample relations
        print(f"{'='*80}")
        print("SAMPLE EXTRACTED RELATIONS (First 5)")
        print(f"{'='*80}")

        for i, relation in enumerate(result['relations'][:5], 1):
            print(f"\n{i}. {relation['content'][:100]}...")
            print(f"   Completeness: {relation.get('completeness_score', 0)}/10")
            print(f"   Source: {relation.get('source_id', 'unknown')}")

        print()

        # Step 8: Test query
        print(f"{'='*80}")
        print("QUERY TEST")
        print(f"{'='*80}")

        test_queries = [
            "KUET CSE আসন সংখ্যা কত?",
            "EEE বিভাগের আসন সংখ্যা কত?",
            "ভর্তি পরীক্ষার তারিখ কবে?",
            "How many seats in Civil Engineering?"
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                contexts = rag.query(query, param={'top_k': 3})
                print(f"  Retrieved {len(contexts)} contexts:")
                for j, ctx in enumerate(contexts[:2], 1):
                    print(f"    {j}. {ctx[:100]}...")
            except Exception as e:
                print(f"  [ERROR] Query failed: {e}")

        print()

        # Step 9: Final summary
        print(f"{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")

        print(f"\nExtraction:")
        print(f"  Entities: {stats['total_entities']}")
        print(f"  Relations: {stats['total_relations']}")
        print(f"  Entity Deduplication: {stats['entity_merge_reduction']} merged")

        print(f"\nValidation:")
        print(f"  Numeric Coverage: {numeric['numeric_coverage']:.2%} {'(PASS)' if numeric['numeric_coverage'] >= 0.99 else '(FAIL)'}")
        print(f"  Consistency Score: {consistency['consistency_score']:.2%} {'(PASS)' if consistency['consistency_score'] >= 0.90 else '(FAIL)'}")
        print(f"  Overall: {overall}")

        print(f"\nGraph Construction:")
        print(f"  Working Directory: ./expr/kuet_test")
        print(f"  Files Created:")
        print(f"    - graph_chunk_entity_relation.graphml")
        print(f"    - vdb_entities.json")
        print(f"    - vdb_relations.json")
        print(f"    - vdb_chunks.json")
        print(f"    - kv_store_full_docs.json")
        print(f"    - kv_store_text_chunks.json")

        print(f"\nQuery Test: {'PASS' if contexts else 'FAIL'}")

        if overall == 'PASS':
            print(f"\n{'='*80}")
            print("SUCCESS: Production pipeline working correctly!")
            print(f"{'='*80}")
            return True
        else:
            print(f"\n{'='*80}")
            print("WARNING: Validation failed. Review extraction quality.")
            print(f"{'='*80}")
            return False

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_kuet_pipeline())
    sys.exit(0 if success else 1)
