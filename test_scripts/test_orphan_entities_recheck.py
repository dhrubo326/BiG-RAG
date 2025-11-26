"""
Quick test to check if orphan entities issue is fixed in kuet_unified graph

Compares before/after rebuild to verify fix effectiveness
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bigrag import BiGRAG
from bigrag.storage import NetworkXStorage


async def test_orphan_entities():
    """Test for orphan entities and relations in the graph"""

    print("\n" + "="*80)
    print("ORPHAN ENTITY DETECTION TEST - kuet_unified Graph")
    print("="*80)

    # Initialize BiGRAG
    project_root = Path(__file__).parent.parent
    expr_path = project_root / "expr" / "kuet_unified"

    print(f"\n[INIT] Loading graph from: {expr_path}")

    rag = BiGRAG(working_dir=str(expr_path))

    # Get graph
    graph_storage = rag.chunk_entity_relation_graph
    if not isinstance(graph_storage, NetworkXStorage):
        print("[ERROR] Graph storage is not NetworkXStorage!")
        return

    graph = graph_storage._graph

    print(f"[INIT] Graph loaded successfully")
    print(f"[STATS] Total nodes: {graph.number_of_nodes()}")
    print(f"[STATS] Total edges: {graph.number_of_edges()}")

    # Collect entities and relations
    entities = []
    relations = []

    for node_id, node_data in graph.nodes(data=True):
        role = node_data.get('role', 'unknown')
        if role == 'entity':
            entities.append({
                'id': node_id,
                'name': node_data.get('entity_name', 'NO_NAME'),
                'type': node_data.get('entity_type', 'UNKNOWN'),
                'degree': graph.degree(node_id),
            })
        elif role == 'relation':
            relations.append({
                'id': node_id,
                'content': node_data.get('content', 'NO_CONTENT')[:80],
                'degree': graph.degree(node_id),
            })

    print(f"\n[STATS] Entity nodes: {len(entities)}")
    print(f"[STATS] Relation nodes: {len(relations)}")

    # Find orphans
    orphan_entities = [e for e in entities if e['degree'] == 0]
    orphan_relations = [r for r in relations if r['degree'] == 0]

    print(f"\n" + "="*80)
    print("ORPHAN ENTITY ANALYSIS")
    print("="*80)

    print(f"\n[RESULT] Orphan entities: {len(orphan_entities)} out of {len(entities)}")
    print(f"[RESULT] Orphan entity rate: {len(orphan_entities)/len(entities)*100:.1f}%")

    if orphan_entities:
        print(f"\n[ORPHANS] Found {len(orphan_entities)} orphan entities:")
        print(f"[ORPHANS] Listing all orphan entities:\n")

        for i, entity in enumerate(orphan_entities, 1):
            # Safely print entity name (handle Unicode)
            try:
                print(f"  {i}. {entity['name']} (type: {entity['type']})")
            except UnicodeEncodeError:
                print(f"  {i}. [Unicode entity] (type: {entity['type']})")
    else:
        print("\n[SUCCESS] NO ORPHAN ENTITIES FOUND!")
        print("[SUCCESS] All entities are connected to at least one relation!")

    print(f"\n" + "="*80)
    print("ORPHAN RELATION ANALYSIS")
    print("="*80)

    print(f"\n[RESULT] Orphan relations: {len(orphan_relations)} out of {len(relations)}")
    print(f"[RESULT] Orphan relation rate: {len(orphan_relations)/len(relations)*100:.1f}%")

    if orphan_relations:
        print(f"\n[ORPHANS] Found {len(orphan_relations)} orphan relations:")
        print(f"[ORPHANS] Listing all orphan relations:\n")

        for i, relation in enumerate(orphan_relations, 1):
            # Safely print relation content
            try:
                print(f"  {i}. {relation['content']}...")
            except UnicodeEncodeError:
                print(f"  {i}. [Unicode relation content]...")
    else:
        print("\n[SUCCESS] NO ORPHAN RELATIONS FOUND!")
        print("[SUCCESS] All relations are connected to at least one entity!")

    # Check bipartite property
    print(f"\n" + "="*80)
    print("BIPARTITE PROPERTY CHECK")
    print("="*80)

    bipartite_violations = 0
    entity_to_entity = 0
    relation_to_relation = 0

    for edge in graph.edges():
        source_role = graph.nodes[edge[0]].get('role', 'unknown')
        target_role = graph.nodes[edge[1]].get('role', 'unknown')

        if source_role == 'entity' and target_role == 'entity':
            entity_to_entity += 1
        elif source_role == 'relation' and target_role == 'relation':
            relation_to_relation += 1

        # Valid: relation <-> entity
        if not ((source_role == 'relation' and target_role == 'entity') or
                (source_role == 'entity' and target_role == 'relation')):
            bipartite_violations += 1

    print(f"\n[RESULT] Bipartite violations: {bipartite_violations}")
    print(f"[RESULT] Entity->Entity edges: {entity_to_entity}")
    print(f"[RESULT] Relation->Relation edges: {relation_to_relation}")

    if bipartite_violations == 0:
        print("\n[SUCCESS] Graph maintains bipartite property!")
    else:
        print(f"\n[ERROR] Found {bipartite_violations} bipartite violations!")

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nPrevious test results (from 15:52):")
    print(f"  - Orphan entities: 19/87 (21.8%)")
    print(f"  - Orphan relations: 4/54 (7.4%)")

    print(f"\nCurrent test results (from 17:05 rebuild):")
    print(f"  - Orphan entities: {len(orphan_entities)}/{len(entities)} ({len(orphan_entities)/len(entities)*100:.1f}%)")
    print(f"  - Orphan relations: {len(orphan_relations)}/{len(relations)} ({len(orphan_relations)/len(relations)*100:.1f}%)")

    # Calculate improvement
    old_orphan_entity_rate = 19/87 * 100
    new_orphan_entity_rate = len(orphan_entities)/len(entities)*100 if entities else 0
    improvement = old_orphan_entity_rate - new_orphan_entity_rate

    print(f"\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if len(orphan_entities) == 0:
        print("\n[EXCELLENT] Assistant's claim is TRUE!")
        print("[EXCELLENT] Orphan entity problem is COMPLETELY FIXED!")
        print(f"[EXCELLENT] Improvement: {improvement:.1f} percentage points reduction!")
    elif len(orphan_entities) < 19:
        print(f"\n[GOOD] Assistant's claim is PARTIALLY TRUE!")
        print(f"[GOOD] Orphan entities reduced from 19 to {len(orphan_entities)}")
        print(f"[GOOD] Improvement: {improvement:.1f} percentage points reduction")
    else:
        print(f"\n[ISSUE] Assistant's claim is FALSE!")
        print(f"[ISSUE] Orphan entities NOT reduced (still {len(orphan_entities)})")
        print(f"[ISSUE] No improvement detected")

    # Check entity type distribution
    print(f"\n" + "="*80)
    print("ENTITY TYPE DISTRIBUTION")
    print("="*80)

    from collections import Counter
    type_dist = Counter(e['type'] for e in entities)

    print(f"\nTop entity types:")
    for etype, count in type_dist.most_common(10):
        orphan_count = len([e for e in orphan_entities if e['type'] == etype])
        print(f"  - {etype}: {count} total, {orphan_count} orphans ({orphan_count/count*100:.1f}%)")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_orphan_entities())
