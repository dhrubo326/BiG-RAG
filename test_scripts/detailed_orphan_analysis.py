"""Detailed analysis of orphan nodes."""
import networkx as nx
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("DETAILED ORPHAN NODE ANALYSIS")
print("=" * 80)

# Load graph
G = nx.read_graphml(r'D:\BiG-RAG\expr\bangla_diagnosis_test\graph_chunk_entity_relation.graphml')

# Identify all node types
entities = [n for n in G.nodes() if G.nodes[n].get('role') == 'entity']
relations = [n for n in G.nodes() if G.nodes[n].get('role') == 'relation']
chunks = [n for n in G.nodes() if n.startswith('chunk-')]

print(f"\n[NODE TYPE COUNTS]")
print(f"Entities: {len(entities)}")
print(f"Relations: {len(relations)}")
print(f"Chunks: {len(chunks)}")
print(f"Total: {G.number_of_nodes()}")

# Check orphans
orphan_entities = [e for e in entities if G.degree(e) == 0]
orphan_relations = [r for r in relations if G.degree(r) == 0]
orphan_chunks = [c for c in chunks if G.degree(c) == 0]

print(f"\n[ORPHAN COUNTS]")
print(f"Orphan entities: {len(orphan_entities)}")
print(f"Orphan relations: {len(orphan_relations)}")
print(f"Orphan chunks: {len(orphan_chunks)}")

# Analyze orphan entities
print(f"\n[ORPHAN ENTITIES DETAIL]")
for i, e_id in enumerate(orphan_entities[:10]):
    e = G.nodes[e_id]
    print(f"\n{i+1}. Entity ID: {e_id[:60]}")
    print(f"   Type: {e.get('entity_type', 'N/A')}")
    print(f"   Weight: {e.get('weight', 'N/A')}")
    print(f"   Source: {e.get('source_id', 'N/A')}")
    print(f"   Description: {e.get('description', 'N/A')[:80]}")
    print(f"   Degree: {G.degree(e_id)}")
    print(f"   Neighbors: {list(G.neighbors(e_id))}")

# Analyze orphan relations
print(f"\n[ORPHAN RELATIONS DETAIL]")
for i, r_id in enumerate(orphan_relations[:5]):
    r = G.nodes[r_id]
    content = r.get('content', r_id)
    print(f"\n{i+1}. Relation: {content[:80]}")
    print(f"   Weight: {r.get('weight', 'N/A')}")
    print(f"   Source: {r.get('source_id', 'N/A')}")
    print(f"   Degree: {G.degree(r_id)}")
    print(f"   Neighbors: {list(G.neighbors(r_id))}")

# Check a few connected entities for comparison
connected_entities = [e for e in entities if G.degree(e) > 0][:5]
print(f"\n[CONNECTED ENTITIES (for comparison)]")
for i, e_id in enumerate(connected_entities):
    e = G.nodes[e_id]
    print(f"\n{i+1}. Entity ID: {e_id[:60]}")
    print(f"   Degree: {G.degree(e_id)}")
    print(f"   Neighbors: {list(G.neighbors(e_id))[:3]}")

# Check edge structure
print(f"\n[EDGE ANALYSIS]")
print(f"Total edges: {G.number_of_edges()}")
sample_edges = list(G.edges())[:5]
print(f"Sample edges:")
for i, (src, dst) in enumerate(sample_edges):
    src_role = G.nodes[src].get('role', 'unknown')
    dst_role = G.nodes[dst].get('role', 'unknown')
    print(f"  {i+1}. {src[:40]} ({src_role}) <-> {dst[:40]} ({dst_role})")

print("\n" + "=" * 80)
