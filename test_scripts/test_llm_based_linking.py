"""
Test script for LLM-based entity-relation linking.

This script tests whether the new entity-relation linking mechanism works correctly:
1. LLM outputs linked_entities as entity_name array
2. StrictExtractor populates metadata.linked_entities from LLM output
3. Step 6.5 converts entity_names to entity_ids after merge

Expected result: Relations should have accurate entity links, reducing synthetic orphan relations.
"""

import asyncio
import json
from bigrag import BiGRAG
from bigrag.base import QueryParam

async def test_llm_linking():
    print("[TEST] LLM-Based Entity-Relation Linking")
    print("=" * 80)

    # Sample test document with clear entity-relation structure
    test_document = """
    কুয়েটের প্রকৌশল অনুষদে তিনটি বিভাগ রয়েছে।

    কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং (CSE) বিভাগে ১৮০টি আসন রয়েছে।
    ইলেকট্রিক্যাল এন্ড ইলেকট্রনিক ইঞ্জিনিয়ারিং (EEE) বিভাগে ১৫০টি আসন রয়েছে।
    সিভিল ইঞ্জিনিয়ারিং (CE) বিভাগে ১২০টি আসন রয়েছে।
    """

    # Initialize BiGRAG with production pipeline
    print("\n[1/5] Initializing BiGRAG...")
    rag = BiGRAG(
        working_dir="./test_llm_linking_output",
        enable_llm_cache=False
    )

    # Index document
    print("\n[2/5] Indexing test document...")
    metadata = {
        "title": "KUET Department Information",
        "category": "education"
    }

    result = await rag.ainsert(
        [test_document],
        metadata=[metadata]
    )

    print(f"  [OK] Indexed {result.get('documents_processed', 0)} documents")

    # Retrieve graph statistics from storage
    print("\n[3/5] Analyzing graph structure...")

    # Get entities and relations from graph storage
    entities = []
    relations = []

    # Read from graph storage (NetworkX GraphML)
    try:
        import networkx as nx
        graph_path = "./test_llm_linking_output/graph_chunk_entity_relation.graphml"
        G = nx.read_graphml(graph_path)

        for node_id, node_data in G.nodes(data=True):
            if node_data.get('role') == 'entity':
                entities.append({
                    'entity_id': node_id,
                    'entity_name': node_data.get('name'),
                    'entity_type': node_data.get('entity_type')
                })
            elif node_data.get('role') == 'relation':
                # Get linked entities from edges
                linked_entities = []
                for edge in G.edges(node_id, data=True):
                    _, target, edge_data = edge
                    if G.nodes[target].get('role') == 'entity':
                        linked_entities.append(target)

                relations.append({
                    'relation_id': node_id,
                    'content': node_data.get('description', '')[:100],
                    'linked_entities': linked_entities,
                    'linking_source': 'graph_edges'  # From graph structure
                })

    except Exception as e:
        print(f"  [WARN] Could not read GraphML: {e}")

    print(f"  Total entities: {len(entities)}")
    print(f"  Total relations: {len(relations)}")

    # Analyze linking quality
    print("\n[4/5] Analyzing entity-relation linking quality...")

    relations_with_links = [r for r in relations if r.get('linked_entities')]
    relations_without_links = [r for r in relations if not r.get('linked_entities')]

    print(f"  Relations with entity links: {len(relations_with_links)} ({len(relations_with_links)/len(relations)*100:.1f}%)")
    print(f"  Relations without links (orphans): {len(relations_without_links)} ({len(relations_without_links)/len(relations)*100:.1f}%)")

    # Show sample linked relation
    if relations_with_links:
        sample = relations_with_links[0]
        print(f"\n  Sample linked relation:")
        print(f"    Content: {sample['content']}")
        print(f"    Linked entities: {len(sample['linked_entities'])} entities")
        for eid in sample['linked_entities'][:3]:
            entity = next((e for e in entities if e['entity_id'] == eid), None)
            if entity:
                print(f"      - {entity['entity_name']} ({entity['entity_type']})")

    # Check for synthetic relations
    print("\n[5/5] Checking for synthetic relation generation...")

    synthetic_relations = [r for r in relations if 'synthetic' in r.get('content', '').lower()]
    extracted_relations = [r for r in relations if 'synthetic' not in r.get('content', '').lower()]

    print(f"  Extracted relations: {len(extracted_relations)} ({len(extracted_relations)/len(relations)*100:.1f}%)")
    print(f"  Synthetic relations: {len(synthetic_relations)} ({len(synthetic_relations)/len(relations)*100:.1f}%)")

    # Expected result
    print("\n" + "=" * 80)
    print("[EXPECTED RESULT]")
    print("  - All extracted relations should have linked_entities populated by LLM")
    print("  - Synthetic relations should be <20% (ideally <10%)")
    print("  - Entity linking should be accurate (no incorrect associations)")

    # Actual result
    print("\n[ACTUAL RESULT]")
    if len(synthetic_relations) / len(relations) < 0.20:
        print("  [PASS] Synthetic relation rate is acceptable (<20%)")
    else:
        print("  [FAIL] Synthetic relation rate is too high (>=20%)")

    if len(relations_without_links) == 0:
        print("  [PASS] All relations have entity links")
    else:
        print(f"  [WARN] {len(relations_without_links)} relations missing entity links")

    print("\n[TEST COMPLETE]")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_llm_linking())
