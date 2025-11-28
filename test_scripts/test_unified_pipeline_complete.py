"""
Test Unified Pipeline with All Fixes

Tests:
1. TableFactExtractor integration
2. Full numeric validation
3. Source_id assignment
4. Hyper-relation linking
5. Complete KUET document processing

Expected Results:
- Entities: 20-30
- Relations: 10-20
- Edges: 30-60+
- Orphans: <20%
- GraphML: 500-1000+ lines
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

from bigrag.pipeline.features import PipelineFeatures
from bigrag.pipeline.base_pipeline import UnifiedPipeline


async def test_unified_pipeline():
    """Test unified pipeline with quality preset."""
    print("=" * 80)
    print("Testing Unified Pipeline - Quality Preset")
    print("=" * 80)

    # Load KUET document
    kuet_path = project_root / "KUET_Admission_info.md"
    if not kuet_path.exists():
        print(f"[ERROR] KUET document not found at {kuet_path}")
        return

    with open(kuet_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"\n[1] Loaded KUET document: {len(content)} characters\n")

    # Initialize quality preset
    try:
        features = PipelineFeatures.from_preset(
            preset="quality",
            openai_api_key=None  # Will use .env
        )
        print("[2] Initialized quality preset:")
        print(f"    - Table detection: {features.enable_table_detection}")
        print(f"    - Table fact extraction: {features.enable_table_fact_extraction}")
        print(f"    - Gleaning: {features.enable_gleaning} (x{features.max_gleaning_iterations})")
        print(f"    - Numeric validation: {features.enable_numeric_validation}")
        print(f"    - Validation strictness: {features.validation_strictness}")
        print(f"    - Merge strategy: {features.merge_strategy}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize features: {e}")
        return

    # Initialize pipeline
    try:
        pipeline = UnifiedPipeline(features)
        print("\n[3] Initialized UnifiedPipeline\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # Process document
    print("[4] Processing document...\n")
    print("-" * 80)

    try:
        result = await pipeline.process_document(
            content=content,
            metadata={
                "title": "KUET Admission Information",
                "category": "education",
                "tags": ["kuet", "admission", "engineering"]
            }
        )
        print("-" * 80)
        print("\n[5] Processing complete!\n")

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    chunks = result.get('chunks', [])
    entities = result.get('entities', [])
    relations = result.get('relations', [])
    validation = result.get('validation', {})
    stats = result.get('statistics', {})

    print(f"\n[Chunks] {len(chunks)} total")
    print(f"[Entities] {len(entities)} total")
    print(f"[Relations] {len(relations)} total")
    print(f"[Avg Entities/Chunk] {stats.get('avg_entities_per_chunk', 0):.2f}")

    # Validation report
    print(f"\n[Validation Status] {validation.get('status', 'UNKNOWN')}")
    if validation.get('warnings'):
        print(f"[Validation Warnings] {len(validation['warnings'])} warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")

    # Numeric validation details
    if 'numeric_validation' in validation:
        numeric = validation['numeric_validation']
        print(f"\n[Numeric Validation]")
        print(f"  Status: {numeric.get('status', 'UNKNOWN')}")
        print(f"  Message: {numeric.get('message', 'No message')}")

    # Entity validation details
    if 'filtered_entities' in validation:
        print(f"\n[Entity Validation]")
        print(f"  Original: {validation['original_entities']}")
        print(f"  Filtered: {validation['filtered_entities']}")
        print(f"  Final: {validation['final_entities']}")

    # Relation validation details
    if 'filtered_relations' in validation:
        print(f"\n[Relation Validation]")
        print(f"  Original: {validation['original_relations']}")
        print(f"  Filtered: {validation['filtered_relations']}")
        print(f"  Final: {validation['final_relations']}")

    # Check for required fields
    print("\n" + "=" * 80)
    print("FIELD VERIFICATION")
    print("=" * 80)

    # Check entities
    entities_with_source_id = sum(1 for e in entities if e.get('source_id'))
    entities_with_entity_id = sum(1 for e in entities if e.get('entity_id'))
    entities_with_hyper_relation = sum(1 for e in entities if e.get('hyper_relation'))

    print(f"\n[Entities]")
    print(f"  With source_id: {entities_with_source_id}/{len(entities)} ({entities_with_source_id/len(entities)*100:.1f}%)")
    print(f"  With entity_id: {entities_with_entity_id}/{len(entities)} ({entities_with_entity_id/len(entities)*100:.1f}%)")
    print(f"  With hyper_relation: {entities_with_hyper_relation}/{len(entities)} ({entities_with_hyper_relation/len(entities)*100:.1f}%)")

    orphan_count = len(entities) - entities_with_hyper_relation
    orphan_ratio = orphan_count / len(entities) if entities else 0
    print(f"  Orphan entities: {orphan_count} ({orphan_ratio*100:.1f}%)")

    # Check relations
    relations_with_source_id = sum(1 for r in relations if r.get('source_id'))
    relations_with_relation_id = sum(1 for r in relations if r.get('relation_id'))
    relations_with_linked_entities = sum(
        1 for r in relations
        if r.get('metadata', {}).get('linked_entities')
    )

    print(f"\n[Relations]")
    print(f"  With source_id: {relations_with_source_id}/{len(relations)} ({relations_with_source_id/len(relations)*100:.1f}%)")
    print(f"  With relation_id: {relations_with_relation_id}/{len(relations)} ({relations_with_relation_id/len(relations)*100:.1f}%)")
    print(f"  With linked_entities: {relations_with_linked_entities}/{len(relations)} ({relations_with_linked_entities/len(relations)*100:.1f}%)")

    # Extraction method breakdown
    print("\n" + "=" * 80)
    print("EXTRACTION METHOD BREAKDOWN")
    print("=" * 80)

    entity_methods = {}
    for entity in entities:
        method = entity.get('metadata', {}).get('extraction_method', 'unknown')
        entity_methods[method] = entity_methods.get(method, 0) + 1

    relation_methods = {}
    for relation in relations:
        method = relation.get('metadata', {}).get('extraction_method', 'unknown')
        relation_methods[method] = relation_methods.get(method, 0) + 1

    print(f"\n[Entity Extraction Methods]")
    for method, count in entity_methods.items():
        print(f"  {method}: {count}")

    print(f"\n[Relation Extraction Methods]")
    for method, count in relation_methods.items():
        print(f"  {method}: {count}")

    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)

    criteria = {
        "Entities count": (len(entities) >= 20, f"{len(entities)} >= 20"),
        "Relations count": (len(relations) >= 10, f"{len(relations)} >= 10"),
        "Orphan ratio": (orphan_ratio < 0.2, f"{orphan_ratio*100:.1f}% < 20%"),
        "All entities have source_id": (entities_with_source_id == len(entities), f"{entities_with_source_id}/{len(entities)}"),
        "All entities have entity_id": (entities_with_entity_id == len(entities), f"{entities_with_entity_id}/{len(entities)}"),
        "All relations have source_id": (relations_with_source_id == len(relations), f"{relations_with_source_id}/{len(relations)}"),
        "All relations have relation_id": (relations_with_relation_id == len(relations), f"{relations_with_relation_id}/{len(relations)}"),
        "Validation passed": (validation.get('status') in ['PASSED', 'WARNING'], validation.get('status', 'UNKNOWN'))
    }

    passed = 0
    total = len(criteria)

    for criterion, (result, details) in criteria.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {criterion}: {details}")
        if result:
            passed += 1

    print(f"\n[Overall] {passed}/{total} criteria passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n[SUCCESS] All criteria met! Unified pipeline is working correctly.")
    else:
        print(f"\n[PARTIAL] {total - passed} criteria failed. Review above for details.")

    return result


if __name__ == "__main__":
    result = asyncio.run(test_unified_pipeline())
