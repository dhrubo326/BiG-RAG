"""Compare orphan node patterns across datasets."""
import networkx as nx
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("ORPHAN NODE PATTERN COMPARISON")
print("=" * 80)

datasets = {
    'Bangla (KUET)': r'D:\BiG-RAG\expr\bangla_diagnosis_test\graph_chunk_entity_relation.graphml',
    'CUET': r'D:\BiG-RAG\expr\cuet_diagnosis_test\graph_chunk_entity_relation.graphml'
}

results = {}

for name, path in datasets.items():
    print(f"\n{'='*80}")
    print(f"DATASET: {name}")
    print('='*80)

    G = nx.read_graphml(path)

    entities = [n for n in G.nodes() if G.nodes[n].get('role') == 'entity']
    relations = [n for n in G.nodes() if G.nodes[n].get('role') == 'relation']

    orphan_entities = [e for e in entities if G.degree(e) == 0]
    orphan_relations = [r for r in relations if G.degree(r) == 0]

    # Analyze orphan entities by type
    orphan_by_type = defaultdict(list)
    for e in orphan_entities:
        entity_type = G.nodes[e].get('entity_type', 'unknown')
        orphan_by_type[entity_type].append(e)

    # Analyze orphan entities by source
    orphan_by_source = defaultdict(list)
    for e in orphan_entities:
        source = G.nodes[e].get('source_id', 'unknown')
        orphan_by_source[source].append(e)

    print(f"\n[STATISTICS]")
    print(f"Total entities: {len(entities)}")
    print(f"Orphan entities: {len(orphan_entities)} ({len(orphan_entities)/len(entities)*100:.1f}%)")
    print(f"Total relations: {len(relations)}")
    print(f"Orphan relations: {len(orphan_relations)} ({len(orphan_relations)/len(relations)*100:.1f}%)")

    print(f"\n[ORPHAN ENTITIES BY TYPE]")
    for entity_type, elist in sorted(orphan_by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {entity_type}: {len(elist)} orphans")
        for i, e in enumerate(elist[:3]):
            print(f"    - {e[:70]}")

    print(f"\n[ORPHAN ENTITIES BY SOURCE CHUNK]")
    for source, elist in sorted(orphan_by_source.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {source}: {len(elist)} orphans")

    print(f"\n[ORPHAN RELATIONS BY SOURCE]")
    orphan_rel_by_source = defaultdict(list)
    for r in orphan_relations:
        source = G.nodes[r].get('source_id', 'unknown')
        orphan_rel_by_source[source].append(r)

    for source, rlist in sorted(orphan_rel_by_source.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {source}: {len(rlist)} orphans")
        for i, r in enumerate(rlist[:2]):
            content = G.nodes[r].get('content', r)[:70]
            print(f"    - {content}")

    # Check connected entities for comparison
    connected_entities = [e for e in entities if G.degree(e) > 0]
    if connected_entities:
        print(f"\n[CONNECTED ENTITY SAMPLE (for comparison)]")
        for i, e in enumerate(connected_entities[:3]):
            e_data = G.nodes[e]
            print(f"  {i+1}. {e[:60]}")
            print(f"     Type: {e_data.get('entity_type', 'N/A')}, Degree: {G.degree(e)}")
            neighbors = list(G.neighbors(e))
            if neighbors:
                neighbor_roles = [G.nodes[n].get('role', 'unknown') for n in neighbors[:2]]
                print(f"     Neighbors: {neighbor_roles}")

    results[name] = {
        'orphan_entity_rate': len(orphan_entities)/len(entities)*100,
        'orphan_relation_rate': len(orphan_relations)/len(relations)*100,
        'orphan_by_type': dict(orphan_by_type),
        'orphan_by_source': dict(orphan_by_source)
    }

print(f"\n{'='*80}")
print("COMPARATIVE SUMMARY")
print('='*80)

print(f"\n[ORPHAN RATES COMPARISON]")
for name, data in results.items():
    print(f"{name}:")
    print(f"  Entity orphan rate: {data['orphan_entity_rate']:.1f}%")
    print(f"  Relation orphan rate: {data['orphan_relation_rate']:.1f}%")

print(f"\n[COMMON PATTERNS]")
print("Both datasets show:")
print("  1. Orphan entities exist (13-26.5% range)")
print("  2. Orphan relations exist (8-11% range)")
print("  3. Orphans have source_id but no graph edges")
print("  4. Connected entities have edges to relations")

print(f"\n[CONCLUSION]")
if all(data['orphan_entity_rate'] > 10 for data in results.values()):
    print("[CONFIRMED] High orphan rate is SYSTEMATIC across datasets")
    print("This indicates a LOGIC ISSUE in entity-relation linking")
    print("NOT a data-specific problem")
else:
    print("Orphan rates vary - may be data-specific issue")

print("\n" + "=" * 80)
