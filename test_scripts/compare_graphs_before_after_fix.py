"""Compare old vs new graph to verify TableFactExtractor fix."""
import networkx as nx
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("COMPARING GRAPHS: BEFORE vs AFTER TableFactExtractor Fix")
print("=" * 80)

# Load both graphs
old_graph_path = r'D:\BiG-RAG\expr\bangla_diagnosis_test\graph_chunk_entity_relation.graphml'
new_graph_path = r'D:\BiG-RAG\expr\bangla_diagnosis_test_v2\graph_chunk_entity_relation.graphml'

print("\n[LOADING GRAPHS]")
print(f"Old graph: {old_graph_path}")
print(f"New graph: {new_graph_path}")

G_old = nx.read_graphml(old_graph_path)
G_new = nx.read_graphml(new_graph_path)

print("\n[BASIC STATISTICS]")
print(f"\nOld graph:")
print(f"  Total nodes: {G_old.number_of_nodes()}")
print(f"  Total edges: {G_old.number_of_edges()}")

print(f"\nNew graph:")
print(f"  Total nodes: {G_new.number_of_nodes()}")
print(f"  Total edges: {G_new.number_of_edges()}")

# Analyze node types
def analyze_graph(G, label):
    entities = [n for n in G.nodes() if G.nodes[n].get('role') == 'entity']
    relations = [n for n in G.nodes() if G.nodes[n].get('role') == 'relation']
    chunks = [n for n in G.nodes() if G.nodes[n].get('role') == 'chunk']

    orphan_entities = [e for e in entities if G.degree(e) == 0]
    orphan_relations = [r for r in relations if G.degree(r) == 0]

    print(f"\n[{label}]")
    print(f"Entities: {len(entities)}")
    print(f"  Connected: {len(entities) - len(orphan_entities)}")
    print(f"  Orphan: {len(orphan_entities)} ({len(orphan_entities)/len(entities)*100:.1f}%)")
    print(f"Relations: {len(relations)}")
    print(f"  Connected: {len(relations) - len(orphan_relations)}")
    print(f"  Orphan: {len(orphan_relations)} ({len(orphan_relations)/len(relations)*100:.1f}%)")
    print(f"Chunks: {len(chunks)}")

    return {
        'entities': entities,
        'relations': relations,
        'chunks': chunks,
        'orphan_entities': orphan_entities,
        'orphan_relations': orphan_relations
    }

old_stats = analyze_graph(G_old, "OLD GRAPH (before fix)")
new_stats = analyze_graph(G_new, "NEW GRAPH (after fix)")

# Compare orphan entities
print("\n" + "=" * 80)
print("ORPHAN ENTITY COMPARISON")
print("=" * 80)

old_orphans = set(old_stats['orphan_entities'])
new_orphans = set(new_stats['orphan_entities'])

print(f"\nOld orphan count: {len(old_orphans)}")
print(f"New orphan count: {len(new_orphans)}")
print(f"Reduction: {len(old_orphans) - len(new_orphans)} entities")
print(f"Reduction rate: {(len(old_orphans) - len(new_orphans))/len(old_orphans)*100:.1f}%")

# Entities that are NO LONGER orphans (FIXED)
fixed_entities = old_orphans - new_orphans
print(f"\n[FIXED ENTITIES] ({len(fixed_entities)} entities)")
for e in sorted(fixed_entities)[:20]:
    if e in G_new.nodes():
        e_data = G_new.nodes[e]
        entity_type = e_data.get('entity_type', 'N/A')
        degree = G_new.degree(e)
        print(f"  - {e[:60]} (type={entity_type}, degree={degree})")
    else:
        print(f"  - {e[:60]} (NOT IN NEW GRAPH)")

# Entities that are STILL orphans
still_orphan = old_orphans & new_orphans
print(f"\n[STILL ORPHAN] ({len(still_orphan)} entities)")
for e in sorted(still_orphan)[:20]:
    e_data = G_new.nodes[e]
    entity_type = e_data.get('entity_type', 'N/A')
    source_id = e_data.get('source_id', 'N/A')
    print(f"  - {e[:60]} (type={entity_type}, source={source_id})")

# NEW orphans (should be 0 or very few)
new_only_orphans = new_orphans - old_orphans
if new_only_orphans:
    print(f"\n[NEW ORPHANS - UNEXPECTED!] ({len(new_only_orphans)} entities)")
    for e in sorted(new_only_orphans)[:10]:
        e_data = G_new.nodes[e]
        entity_type = e_data.get('entity_type', 'N/A')
        source_id = e_data.get('source_id', 'N/A')
        print(f"  - {e[:60]} (type={entity_type}, source={source_id})")

# Check specific table entities mentioned in analysis
print("\n" + "=" * 80)
print("TABLE ENTITY VERIFICATION")
print("=" * 80)

table_entities = [
    '"CIVIL ENGINEERING"',
    '"COMPUTER SCIENCE AND ENGINEERING"',
    '"CSE"',
    '"120"',
    '"ELECTRICAL AND ELECTRONIC ENGINEERING"'
]

print("\nChecking previously orphaned table entities:")
for e_name in table_entities:
    if e_name in G_old.nodes():
        old_degree = G_old.degree(e_name)
        new_degree = G_new.degree(e_name) if e_name in G_new.nodes() else -1

        status = "FIXED" if old_degree == 0 and new_degree > 0 else \
                 "STILL ORPHAN" if old_degree == 0 and new_degree == 0 else \
                 "ALREADY OK" if old_degree > 0 else "NOT IN NEW"

        print(f"  {e_name:50s} | Old degree: {old_degree:2d} | New degree: {new_degree:2d} | [{status}]")

# Edge comparison
print("\n" + "=" * 80)
print("EDGE ANALYSIS")
print("=" * 80)

old_edges = G_old.number_of_edges()
new_edges = G_new.number_of_edges()
edge_increase = new_edges - old_edges

print(f"\nOld graph edges: {old_edges}")
print(f"New graph edges: {new_edges}")
print(f"Edge increase: {edge_increase} ({edge_increase/old_edges*100:.1f}%)")

# Relation comparison
print("\n" + "=" * 80)
print("RELATION ORPHAN COMPARISON")
print("=" * 80)

old_rel_orphans = set(old_stats['orphan_relations'])
new_rel_orphans = set(new_stats['orphan_relations'])

print(f"\nOld orphan relations: {len(old_rel_orphans)}")
print(f"New orphan relations: {len(new_rel_orphans)}")
print(f"Change: {len(new_rel_orphans) - len(old_rel_orphans)}")

if len(new_rel_orphans) < len(old_rel_orphans):
    fixed_relations = old_rel_orphans - new_rel_orphans
    print(f"\n[FIXED RELATIONS] ({len(fixed_relations)})")
    for r in sorted(fixed_relations)[:5]:
        r_content = G_new.nodes[r].get('content', r)[:80]
        print(f"  - {r_content}")

# Final verdict
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

expected_reduction = 14  # From analysis: 14/22 orphans were table entities
actual_reduction = len(old_orphans) - len(new_orphans)

print(f"\nExpected orphan reduction: ~{expected_reduction} entities (64% of 22)")
print(f"Actual orphan reduction: {actual_reduction} entities")
print(f"Reduction percentage: {actual_reduction/len(old_orphans)*100:.1f}%")

if actual_reduction >= expected_reduction * 0.8:  # 80% of expected
    print("\n[SUCCESS] Fix worked as expected!")
    print("- Table entities are now connected to their relations")
    print("- Orphan rate reduced significantly")
    print("- Remaining orphans are likely from LLM extraction gaps")
elif actual_reduction > 0:
    print("\n[PARTIAL SUCCESS] Fix worked partially")
    print("- Some table entities are now connected")
    print("- Less reduction than expected - may need investigation")
else:
    print("\n[FAILED] Fix did not work")
    print("- Orphan count did not decrease")
    print("- Need to investigate why fix wasn't effective")

print("\n" + "=" * 80)
