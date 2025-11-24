"""Deep trace of orphan entity cause."""
import networkx as nx
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("TRACING ORPHAN ENTITY ROOT CAUSE")
print("=" * 80)

# Load Bangla graph
G = nx.read_graphml(r'D:\BiG-RAG\expr\bangla_diagnosis_test\graph_chunk_entity_relation.graphml')

# Find all entities and relations
entities = [n for n in G.nodes() if G.nodes[n].get('role') == 'entity']
relations = [n for n in G.nodes() if G.nodes[n].get('role') == 'relation']

# Separate orphan and connected entities
orphan_entities = [e for e in entities if G.degree(e) == 0]
connected_entities = [e for e in entities if G.degree(e) > 0]

print(f"\n[STATISTICS]")
print(f"Total entities: {len(entities)}")
print(f"Connected entities: {len(connected_entities)}")
print(f"Orphan entities: {len(orphan_entities)}")

# Analyze orphan entities in detail
print(f"\n[ORPHAN ENTITY ANALYSIS]")
print("Checking if orphan entities have source_id attribute...")

orphan_with_source = 0
orphan_without_source = 0

for e_id in orphan_entities[:5]:
    e = G.nodes[e_id]
    source_id = e.get('source_id', None)
    entity_type = e.get('entity_type', 'N/A')
    weight = e.get('weight', 'N/A')

    print(f"\nOrphan: {e_id[:60]}")
    print(f"  Entity type: {entity_type}")
    print(f"  Weight: {weight}")
    print(f"  Source ID: {source_id}")
    print(f"  Degree: {G.degree(e_id)}")

    if source_id:
        orphan_with_source += 1
    else:
        orphan_without_source += 1

print(f"\nOrphans with source_id: {orphan_with_source}")
print(f"Orphans without source_id: {orphan_without_source}")

# Now analyze connected entities
print(f"\n[CONNECTED ENTITY ANALYSIS]")
print("Checking what connected entities have...")

for e_id in connected_entities[:5]:
    e = G.nodes[e_id]
    source_id = e.get('source_id', None)
    neighbors = list(G.neighbors(e_id))

    print(f"\nConnected: {e_id[:60]}")
    print(f"  Source ID: {source_id}")
    print(f"  Degree: {G.degree(e_id)}")
    print(f"  Neighbors ({len(neighbors)}):")
    for n in neighbors:
        n_role = G.nodes[n].get('role', 'unknown')
        n_content = G.nodes[n].get('content', n)[:40]
        print(f"    - {n_role}: {n_content}")

# Check edges to see if they have source_id
print(f"\n[EDGE ANALYSIS]")
sample_edges = list(G.edges())[:5]
for src, dst in sample_edges:
    edge_data = G.edges[src, dst]
    print(f"\nEdge: {src[:40]} <-> {dst[:40]}")
    print(f"  Edge attributes: {list(edge_data.keys())}")
    print(f"  Edge source_id: {edge_data.get('source_id', 'N/A')}")
    print(f"  Edge weight: {edge_data.get('weight', 'N/A')}")

# KEY INSIGHT: Check if orphan entities have different source_ids than connected ones
print(f"\n[SOURCE ID DISTRIBUTION]")
orphan_sources = {}
for e_id in orphan_entities:
    source = G.nodes[e_id].get('source_id', 'unknown')
    orphan_sources[source] = orphan_sources.get(source, 0) + 1

connected_sources = {}
for e_id in connected_entities:
    source = G.nodes[e_id].get('source_id', 'unknown')
    connected_sources[source] = connected_sources.get(source, 0) + 1

print("Orphan entity sources:")
for source, count in sorted(orphan_sources.items(), key=lambda x: x[1], reverse=True):
    print(f"  {source}: {count} orphans")

print("\nConnected entity sources:")
for source, count in sorted(connected_sources.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {source}: {count} connected")

# Check relations from orphan source chunks
print(f"\n[RELATIONS FROM ORPHAN SOURCE CHUNKS]")
for source in orphan_sources.keys():
    # Find relations from this source
    source_relations = [r for r in relations if G.nodes[r].get('source_id') == source]
    print(f"\nChunk {source}:")
    print(f"  Orphan entities: {orphan_sources[source]}")
    print(f"  Relations from this chunk: {len(source_relations)}")

    if source_relations:
        for r_id in source_relations[:2]:
            r = G.nodes[r_id]
            content = r.get('content', r_id)[:60]
            degree = G.degree(r_id)
            print(f"    Relation: {content}... (degree={degree})")

print("\n" + "=" * 80)
