"""Compare v1 (old) vs v3 (latest after fix) to verify the entity ID fix worked."""
import networkx as nx
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("GRAPH COMPARISON: v1 (OLD) vs v3 (AFTER FIX)")
print("=" * 80)

# Load both graphs
v1_path = r'D:\BiG-RAG\expr\bangla_diagnosis_test\graph_chunk_entity_relation.graphml'
v3_path = r'D:\BiG-RAG\expr\bangla_diagnosis_test_v3\graph_chunk_entity_relation.graphml'

print(f"\n[LOADING]")
print(f"v1 (old): {v1_path}")
print(f"v3 (new): {v3_path}")

G_v1 = nx.read_graphml(v1_path)
G_v3 = nx.read_graphml(v3_path)

# Analyze node types
def analyze_graph(G, label):
    entities = [n for n in G.nodes() if G.nodes[n].get('role') == 'entity']
    relations = [n for n in G.nodes() if G.nodes[n].get('role') == 'relation']
    chunks = [n for n in G.nodes() if G.nodes[n].get('role') == 'chunk']

    orphan_entities = [e for e in entities if G.degree(e) == 0]
    orphan_relations = [r for r in relations if G.degree(r) == 0]

    print(f"\n[{label}]")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"")
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

v1_stats = analyze_graph(G_v1, "v1 - OLD (before fix)")
v3_stats = analyze_graph(G_v3, "v3 - NEW (after fix)")

# Calculate improvements
print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS")
print("=" * 80)

orphan_reduction = len(v1_stats['orphan_entities']) - len(v3_stats['orphan_entities'])
orphan_reduction_pct = (orphan_reduction / len(v1_stats['orphan_entities']) * 100) if len(v1_stats['orphan_entities']) > 0 else 0
edge_increase = G_v3.number_of_edges() - G_v1.number_of_edges()

print(f"\n[ENTITY ORPHANS]")
print(f"v1: {len(v1_stats['orphan_entities'])} orphans")
print(f"v3: {len(v3_stats['orphan_entities'])} orphans")
print(f"Reduction: {orphan_reduction} entities ({orphan_reduction_pct:.1f}%)")

print(f"\n[RELATION ORPHANS]")
rel_orphan_reduction = len(v1_stats['orphan_relations']) - len(v3_stats['orphan_relations'])
print(f"v1: {len(v1_stats['orphan_relations'])} orphans")
print(f"v3: {len(v3_stats['orphan_relations'])} orphans")
print(f"Change: {rel_orphan_reduction}")

print(f"\n[EDGES]")
print(f"v1: {G_v1.number_of_edges()} edges")
print(f"v3: {G_v3.number_of_edges()} edges")
print(f"Increase: {edge_increase} edges ({edge_increase/G_v1.number_of_edges()*100:.1f}%)")

# Check specific table entities that were orphans before
print("\n" + "=" * 80)
print("DEPARTMENT ENTITY VERIFICATION (Table Entities)")
print("=" * 80)

# These were orphan entities in v1 from table extraction
table_dept_codes = [
    'CIVIL ENGINEERING',
    'COMPUTER SCIENCE AND ENGINEERING',
    'ELECTRICAL AND ELECTRONIC ENGINEERING',
    'MECHANICAL ENGINEERING',
    'ARCHITECTURE',
    'CHEMICAL ENGINEERING'
]

print("\nChecking previously orphaned department entities:")
for dept in table_dept_codes:
    # Check v1
    v1_found = False
    v1_degree = 0
    for node_id in G_v1.nodes():
        node_data = G_v1.nodes[node_id]
        if node_data.get('role') == 'entity':
            # Check if entity_name matches (v1 uses name-based IDs)
            entity_name = node_data.get('entity_name', node_id.strip('"').upper())
            if entity_name == dept or node_id.strip('"').upper() == dept:
                v1_found = True
                v1_degree = G_v1.degree(node_id)
                break

    # Check v3
    v3_found = False
    v3_degree = 0
    v3_node_id = None
    for node_id in G_v3.nodes():
        node_data = G_v3.nodes[node_id]
        if node_data.get('role') == 'entity':
            # Check entity_name attribute (v3 uses entity_id as node ID)
            entity_name = node_data.get('entity_name', '')
            if entity_name == dept or entity_name.upper() == dept:
                v3_found = True
                v3_degree = G_v3.degree(node_id)
                v3_node_id = node_id
                break

    status = "NOT FOUND" if not v1_found and not v3_found else \
             "FIXED" if v1_degree == 0 and v3_degree > 0 else \
             "STILL ORPHAN" if v1_degree == 0 and v3_degree == 0 else \
             "ALREADY OK" if v1_degree > 0 and v3_degree > 0 else \
             "REGRESSED" if v1_degree > 0 and v3_degree == 0 else "UNKNOWN"

    print(f"  {dept:50s} | v1: deg={v1_degree:2d} | v3: deg={v3_degree:2d} | [{status}]")

# Show which entities were fixed
print("\n" + "=" * 80)
print("ENTITIES THAT ARE NO LONGER ORPHANS (FIXED)")
print("=" * 80)

v1_orphans = set(v1_stats['orphan_entities'])
v3_orphans = set(v3_stats['orphan_entities'])

# Build mapping from v1 orphan names to v3 node IDs
v1_orphan_names = {}
for orphan_id in v1_orphans:
    node_data = G_v1.nodes[orphan_id]
    entity_name = node_data.get('entity_name', orphan_id.strip('"').upper())
    v1_orphan_names[entity_name] = orphan_id

v3_entity_name_to_id = {}
for node_id in G_v3.nodes():
    node_data = G_v3.nodes[node_id]
    if node_data.get('role') == 'entity':
        entity_name = node_data.get('entity_name', '')
        v3_entity_name_to_id[entity_name] = node_id

# Find entities that were orphan in v1 but connected in v3
fixed_entities = []
for entity_name, v1_id in v1_orphan_names.items():
    if entity_name in v3_entity_name_to_id:
        v3_id = v3_entity_name_to_id[entity_name]
        v3_degree = G_v3.degree(v3_id)
        if v3_degree > 0:  # Connected in v3
            fixed_entities.append({
                'name': entity_name,
                'v1_id': v1_id,
                'v3_id': v3_id,
                'v3_degree': v3_degree,
                'type': G_v3.nodes[v3_id].get('entity_type', 'N/A')
            })

print(f"\nTotal fixed: {len(fixed_entities)}")
if fixed_entities:
    print("\nFixed entities (first 20):")
    for i, entity in enumerate(fixed_entities[:20], 1):
        print(f"  {i:2d}. {entity['name'][:60]:60s} (type={entity['type']:20s}, degree={entity['v3_degree']})")

# Still orphan
still_orphan = []
for entity_name, v1_id in v1_orphan_names.items():
    if entity_name in v3_entity_name_to_id:
        v3_id = v3_entity_name_to_id[entity_name]
        v3_degree = G_v3.degree(v3_id)
        if v3_degree == 0:  # Still orphan in v3
            still_orphan.append({
                'name': entity_name,
                'type': G_v3.nodes[v3_id].get('entity_type', 'N/A'),
                'source': G_v3.nodes[v3_id].get('source_id', 'N/A')
            })

print("\n" + "=" * 80)
print("ENTITIES STILL ORPHAN (Need Further Investigation)")
print("=" * 80)
print(f"\nTotal still orphan: {len(still_orphan)}")
if still_orphan:
    print("\nStill orphan entities (first 20):")
    for i, entity in enumerate(still_orphan[:20], 1):
        print(f"  {i:2d}. {entity['name'][:60]:60s} (type={entity['type']:20s}, src={entity['source']})")

# Final verdict
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

expected_reduction = 14  # From analysis: 14/22 orphans were table entities
actual_reduction = orphan_reduction

print(f"\nExpected orphan reduction: ~{expected_reduction} entities (64% of 22)")
print(f"Actual orphan reduction: {actual_reduction} entities ({orphan_reduction_pct:.1f}%)")

if actual_reduction >= expected_reduction * 0.8:  # 80% of expected
    print("\n[SUCCESS] Fix worked as expected!")
    print("  - Table entities are now connected to their relations")
    print("  - Orphan rate reduced significantly")
    print("  - Entity ID remapping system is working correctly")
elif actual_reduction > 0:
    print("\n[PARTIAL SUCCESS] Fix worked partially")
    print("  - Some table entities are now connected")
    print("  - Less reduction than expected - may need investigation")
else:
    print("\n[FAILED] Fix did not work")
    print("  - Orphan count did not decrease")
    print("  - Need to investigate why fix wasn't effective")

print("\n" + "=" * 80)
