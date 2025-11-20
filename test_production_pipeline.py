"""
Test script for Week 1-2 Production KG Pipeline

Tests all modules independently:
- Table extraction (GPT-4o)
- Smart chunking
- Table fact extraction
- Paragraph extraction with validation
- Numeric validation
- Consistency validation
"""

import asyncio
import os
import sys

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add bigrag to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bigrag.preprocessors.table_extractor import GPT4TableExtractor, BilingualDetector
from bigrag.preprocessors.smart_chunker import TableAwareChunker
from bigrag.extractors.table_fact_extractor import TableFactExtractor
from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
from bigrag.validators.numeric_validator import NumericValidator
from bigrag.validators.consistency_validator import ConsistencyValidator


async def test_pipeline():
    """Test the complete production pipeline."""

    print("=" * 80)
    print("BiG-RAG Production Pipeline Test")
    print("=" * 80)
    print()

    # Sample document with table and paragraph content
    sample_doc = """# Demo Test Document

## Department Information

| Department | Code | Seats |
|-----------|------|-------|
| Computer Science and Engineering | CSE | 120 |
| Electrical and Electronic Engineering | EEE | 120 |
| Civil Engineering | CE | 60 |

## Admission Information

The admission test will be held on December 4, 2024. The total number of seats available is 300 across all departments. The Computer Science and Engineering department has 120 seats, which is the largest department.

## Fee Structure

The admission fee is 1100 Taka for all candidates. The application deadline is November 15, 2024.
"""

    print("Sample Document:")
    print("-" * 80)
    print(sample_doc)
    print("-" * 80)
    print()

    # Read API key
    api_key_file = "openai_api_key.txt"
    if not os.path.exists(api_key_file):
        print(f"[ERROR] {api_key_file} not found!")
        print("Please create openai_api_key.txt with your OpenAI API key")
        return

    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()

    if not api_key or api_key == "your-api-key-here":
        print("[ERROR] Please set a valid OpenAI API key in openai_api_key.txt")
        return

    print(f"[OK] API key loaded from {api_key_file}")
    print()

    # Initialize components
    try:
        # Step 1: Table Extraction
        print("[STEP 1/7] Table Extraction (GPT-4o)")
        print("-" * 80)

        table_extractor = GPT4TableExtractor(api_key=api_key, model="gpt-4o-mini")
        print("[OK] GPT4TableExtractor initialized")

        metadata = {
            "title": "Demo Test Document",
            "category": "admission",
            "tags": ["test", "demo"]
        }

        tables = await table_extractor.extract_tables_from_document(
            sample_doc,
            document_metadata=metadata
        )

        print(f"[RESULT] Found {len(tables)} table(s)")
        for i, table in enumerate(tables):
            print(f"  Table {i+1}:")
            print(f"    - Type: {table['table_type']}")
            print(f"    - Headers: {table['headers']}")
            print(f"    - Rows: {len(table['rows'])}")
            print(f"    - Validation: {table['metadata']['validation_status']}")
            if table['metadata']['validation_status'] == 'PASS':
                print(f"    - Coverage: {table['metadata'].get('numeric_coverage', 1.0):.2%}")
        print()

        # Step 2: Language Detection
        print("[STEP 2/7] Language Detection")
        print("-" * 80)

        lang_info = BilingualDetector.detect_languages(sample_doc)
        print(f"[RESULT] Primary language: {lang_info['primary']}")
        print(f"  - English probability: {lang_info['en_probability']:.2%}")
        print(f"  - Bangla probability: {lang_info['bn_probability']:.2%}")
        print(f"  - Is bilingual: {lang_info['is_bilingual']}")
        print()

        # Step 3: Smart Chunking
        print("[STEP 3/7] Smart Chunking (Table-Aware)")
        print("-" * 80)

        chunker = TableAwareChunker(table_extractor)
        print("[OK] TableAwareChunker initialized")

        chunks = await chunker.chunk_document(
            sample_doc,
            chunk_size=1200,
            overlap=100,
            metadata=metadata
        )

        print(f"[RESULT] Created {len(chunks)} chunk(s)")
        table_chunks = [c for c in chunks if c['type'] == 'table']
        paragraph_chunks = [c for c in chunks if c['type'] == 'paragraph']
        print(f"  - Table chunks: {len(table_chunks)}")
        print(f"  - Paragraph chunks: {len(paragraph_chunks)}")
        print()

        # Step 4: Table Fact Extraction
        print("[STEP 4/7] Table Fact Extraction (Deterministic)")
        print("-" * 80)

        all_entities = []
        all_relations = []

        for chunk in table_chunks:
            facts = TableFactExtractor.extract_facts_from_table(
                chunk['structured_data'],
                chunk['chunk_id']
            )
            all_entities.extend(facts['entities'])
            all_relations.extend(facts['relations'])
            print(f"[RESULT] {chunk['chunk_id']}:")
            print(f"  - Entities: {facts['stats']['num_entities']}")
            print(f"  - Relations: {facts['stats']['num_relations']}")
            print(f"  - Extraction: {facts['extraction_method']}")
            print(f"  - Confidence: {facts['confidence']}")
        print()

        # Step 5: Paragraph Extraction with Validation
        print("[STEP 5/7] Paragraph Extraction (Constrained LLM)")
        print("-" * 80)

        llm_extractor = ConstrainedLLMExtractor(api_key=api_key, model="gpt-4o-mini")
        print("[OK] ConstrainedLLMExtractor initialized")

        batch_extractor = BatchConstrainedExtractor(llm_extractor)

        batch_result = await batch_extractor.extract_from_chunks(
            paragraph_chunks,
            language="English"
        )

        print(f"[RESULT] Batch extraction statistics:")
        stats = batch_result['statistics']
        print(f"  - Total chunks: {stats['total_chunks']}")
        print(f"  - Successful: {stats['successful_extractions']}")
        print(f"  - Failed: {stats['failed_extractions']}")
        print(f"  - Success rate: {stats['success_rate']:.2%}")
        print(f"  - Avg numeric coverage: {stats['avg_numeric_coverage']:.2%}")
        print(f"  - Avg hallucination: {stats['avg_hallucination_score']:.2%}")
        print(f"  - Avg semantic validity: {stats['avg_semantic_validity']:.2%}")
        print(f"  - Avg attempts per chunk: {stats['avg_attempts']:.1f}")

        for extraction in batch_result['extractions']:
            all_entities.extend(extraction['entities'])
            all_relations.extend(extraction['relations'])
            print(f"  {extraction['chunk_id']}: {len(extraction['entities'])} entities, {len(extraction['relations'])} relations")

        if batch_result['failed_chunks']:
            print(f"[WARN] Failed chunks: {batch_result['failed_chunks']}")
        print()

        # Step 6: Numeric Validation
        print("[STEP 6/7] Numeric Validation")
        print("-" * 80)

        num_validator = NumericValidator()
        print("[OK] NumericValidator initialized")

        num_result = num_validator.validate_extraction(
            source_document=sample_doc,
            entities=all_entities,
            relations=all_relations,
            validation_level="STRICT"
        )

        print(f"[RESULT] Validation Status: {num_result['status']}")
        print(f"  - Validation Level: {num_result['validation_level']}")
        print(f"  - Numeric Coverage: {num_result['numeric_coverage']:.2%}")
        print(f"  - Hallucination Rate: {num_result['hallucination_rate']:.2%}")
        print(f"  - Total Source Numbers: {num_result['total_source_numbers']}")
        print(f"  - Total KG Numbers: {num_result['total_kg_numbers']}")
        print(f"  - Matched Numbers: {num_result['matched_numbers']}")

        if num_result['missing_numbers']:
            print(f"  [WARN] Missing Numbers: {num_result['missing_numbers']}")

        if num_result['hallucinated_numbers']:
            print(f"  [ERROR] Hallucinated Numbers: {num_result['hallucinated_numbers']}")

        if num_result['frequency_analysis']['frequency_mismatches']:
            print(f"  [INFO] Frequency Mismatches:")
            for mismatch in num_result['frequency_analysis']['frequency_mismatches']:
                print(f"    - '{mismatch['number']}': source={mismatch['source_frequency']}, kg={mismatch['kg_frequency']}, deficit={mismatch['deficit']}")

        print()

        # Step 7: Consistency Validation
        print("[STEP 7/7] Cross-Chunk Consistency Validation")
        print("-" * 80)

        cons_validator = ConsistencyValidator()
        print("[OK] ConsistencyValidator initialized")

        cons_result = cons_validator.validate_consistency(
            entities=all_entities,
            relations=all_relations,
            validation_level="STRICT"
        )

        print(f"[RESULT] Validation Status: {cons_result['status']}")
        print(f"  - Consistency Score: {cons_result['consistency_score']:.2%}")
        print(f"  - Total Entities: {cons_result['total_entities']}")
        print(f"  - Total Relations: {cons_result['total_relations']}")
        print(f"  - Total Issues: {cons_result['total_issues']}")
        print(f"  - Entity Conflicts: {len(cons_result['entity_conflicts'])}")
        print(f"  - Numeric Conflicts: {len(cons_result['numeric_conflicts'])}")
        print(f"  - Relation Contradictions: {len(cons_result['relation_contradictions'])}")
        print(f"  - Reference Errors: {len(cons_result['reference_errors'])}")

        if cons_result['entity_conflicts']:
            print(f"  [WARN] Entity Conflicts:")
            for conflict in cons_result['entity_conflicts'][:3]:  # Show first 3
                print(f"    - {conflict['type']}: {conflict['entity_name']} ({conflict['severity']})")

        if cons_result['numeric_conflicts']:
            print(f"  [ERROR] Numeric Conflicts:")
            for conflict in cons_result['numeric_conflicts'][:3]:  # Show first 3
                print(f"    - {conflict['type']}: {conflict['entity_name']} - {conflict['property']} ({conflict['severity']})")

        print()

        # Final Summary
        print("=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Total Entities Extracted: {len(all_entities)}")
        print(f"Total Relations Extracted: {len(all_relations)}")
        print()
        print(f"Numeric Validation: {num_result['status']}")
        print(f"  - Coverage: {num_result['numeric_coverage']:.2%}")
        print(f"  - Hallucination: {num_result['hallucination_rate']:.2%}")
        print()
        print(f"Consistency Validation: {cons_result['status']}")
        print(f"  - Consistency: {cons_result['consistency_score']:.2%}")
        print(f"  - Issues: {cons_result['total_issues']}")
        print()

        # Overall status
        if num_result['status'] == 'PASS' and cons_result['status'] == 'PASS':
            print("=" * 80)
            print("[SUCCESS] PIPELINE READY FOR PRODUCTION")
            print("=" * 80)
            print()
            print("All validations passed!")
            print("- Tables extracted correctly")
            print("- Numbers preserved with 100% accuracy")
            print("- No hallucinations detected")
            print("- No consistency conflicts")
            print()
            print("Next steps:")
            print("1. Test on larger documents")
            print("2. Implement Phase 3 (Entity Merging)")
            print("3. Integrate with BiGRAG.ainsert()")
            return True
        else:
            print("=" * 80)
            print("[FAILURE] VALIDATION FAILED")
            print("=" * 80)
            print()
            if num_result['status'] == 'FAIL':
                print("[ERROR] Numeric validation failed:")
                for rec in num_result['recommendations']:
                    print(f"  - {rec}")

            if cons_result['status'] == 'FAIL':
                print("[ERROR] Consistency validation failed:")
                for rec in cons_result['recommendations']:
                    print(f"  - {rec}")

            print()
            print("Please review the issues above and re-run the test.")
            return False

    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_pipeline())
    sys.exit(0 if success else 1)
