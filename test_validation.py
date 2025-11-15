"""Test validation functions directly"""
import asyncio
import logging
from bigrag.operate import _handle_single_entity_extraction, _handle_single_hyperrelation_extraction
from bigrag.utils import split_string_by_multi_markers

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

async def test_validation():
    # Sample LLM output (from debug_extraction.py)
    sample_output = """("relation"<|>Lionel Messi continues to make headlines in Major League Soccer with Inter Miami.<|>10)##("entity"<|>LIONEL MESSI<|>person<|>Widely regarded as one of the greatest football players of all time, currently playing for Inter Miami.<|>90)##("entity"<|>INTER MIAMI<|>organization<|>Soccer club based in Miami, Florida, that competes in Major League Soccer.<|>85)##"""

    # Parse records (simulating what operate.py does)
    records = split_string_by_multi_markers(sample_output, ["##", "<|COMPLETE|>"])

    print(f"Parsed {len(records)} records\n")

    # Test relation extraction
    for i, record in enumerate(records, 1):
        # Extract content from parentheses
        import re
        record_match = re.search(r"\((.*)\)", record)
        if not record_match:
            print(f"Record {i}: No parentheses match")
            continue

        record_content = record_match.group(1)

        # Split by delimiter
        parts = split_string_by_multi_markers(record_content, ["<|>"])

        print(f"Record {i}: {len(parts)} parts")
        print(f"  First field: {repr(parts[0])}")

        if parts[0] == '"relation"':
            print(f"  Testing relation extraction...")
            result = await _handle_single_hyperrelation_extraction(parts, "test-chunk-001")
            if result:
                print(f"  [SUCCESS] Relation extracted: {result['hyper_relation_content'][:50]}...")
            else:
                print(f"  [FAIL] Relation rejected by validation")

        elif parts[0] == '"entity"':
            print(f"  Testing entity extraction...")
            # For entities, we need a relation context
            now_hyper_relation = "rel-test123"  # Simulated relation ID
            result = await _handle_single_entity_extraction(parts, "test-chunk-001", now_hyper_relation)
            if result:
                print(f"  [SUCCESS] Entity extracted: {result['entity_name']}")
            else:
                print(f"  [FAIL] Entity rejected by validation")

        print()

if __name__ == "__main__":
    asyncio.run(test_validation())
