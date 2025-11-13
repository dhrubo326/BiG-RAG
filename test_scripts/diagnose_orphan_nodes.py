"""
Diagnostic script to analyze orphan nodes in the football graph.

This script will:
1. Load the graph
2. Identify all orphan nodes (degree = 0)
3. Break them down by type (entity, relation, chunk)
4. Show what nodes would be included with current sampling logic
5. Help determine if UI display is correct
"""

import networkx as nx
from pathlib import Path

# Load graph
graph_file = Path("d:/BiG-RAG/expr/football/graph_chunk_entity_relation.graphml")
print(f"Loading graph from: {graph_file}")
G = nx.read_graphml(str(graph_file))

print(f"\n[OVERALL STATS]")
print(f"Total nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")

# Collect all nodes with their metadata
all_nodes = []
entity_count = 0
relation_count = 0
chunk_count = 0
orphan_count = 0

for node_id, attrs in G.nodes(data=True):
    role = attrs.get("role", "")
    node_type = "entity"  # default

    if role == "entity":
        node_type = "entity"
        entity_count += 1
    elif role == "relation":
        node_type = "relation"
        relation_count += 1
    elif role == "chunk" or node_id.startswith("chunk-"):
        node_type = "chunk"
        chunk_count += 1

    connections = G.degree(node_id)
    if connections == 0:
        orphan_count += 1

    weight = float(attrs.get("weight", 0.0))

    node = {
        "id": node_id,
        "type": node_type,
        "role": role,
        "weight": weight,
        "connections": connections,
        "name": attrs.get("name", "")[:50]
    }
    all_nodes.append(node)

print(f"\nEntities: {entity_count}")
print(f"Relations: {relation_count}")
print(f"Chunks: {chunk_count}")
print(f"Total orphan nodes (degree=0): {orphan_count}")

# Break down orphan nodes by type
orphan_nodes = [n for n in all_nodes if n["connections"] == 0]
orphan_entities = [n for n in orphan_nodes if n["type"] == "entity"]
orphan_relations = [n for n in orphan_nodes if n["type"] == "relation"]
orphan_chunks = [n for n in orphan_nodes if n["type"] == "chunk"]

print(f"\n[ORPHAN NODE BREAKDOWN]")
print(f"Total orphan nodes: {len(orphan_nodes)}")
print(f"  - Orphan entities: {len(orphan_entities)}")
print(f"  - Orphan relations: {len(orphan_relations)}")
print(f"  - Orphan chunks: {len(orphan_chunks)}")

# Show top 10 orphan nodes of each type
print(f"\n[TOP 10 ORPHAN ENTITIES BY WEIGHT]")
top_orphan_entities = sorted(orphan_entities, key=lambda x: x["weight"], reverse=True)[:10]
for i, node in enumerate(top_orphan_entities, 1):
    print(f"  {i}. [{node['weight']:.1f}] {node['name']}")

print(f"\n[TOP 10 ORPHAN RELATIONS BY WEIGHT]")
top_orphan_relations = sorted(orphan_relations, key=lambda x: x["weight"], reverse=True)[:10]
for i, node in enumerate(top_orphan_relations, 1):
    print(f"  {i}. [{node['weight']:.1f}] {node['name']}")

print(f"\n[TOP 10 ORPHAN CHUNKS BY WEIGHT]")
top_orphan_chunks = sorted(orphan_chunks, key=lambda x: x["weight"], reverse=True)[:10]
for i, node in enumerate(top_orphan_chunks, 1):
    print(f"  {i}. [{node['weight']:.1f}] {node['name']}")

# Simulate current sampling logic (20% cap, no type balancing)
limit = 1000
max_orphans_to_include = min(len(orphan_nodes), int(limit * 0.2))
included_orphans_current = orphan_nodes[:max_orphans_to_include]

print(f"\n[CURRENT SAMPLING LOGIC (file order, 20% cap)]")
print(f"Limit: {limit}")
print(f"Max orphans to include: {max_orphans_to_include}")
print(f"Orphans included: {len(included_orphans_current)}")

# Count by type
included_entities = [n for n in included_orphans_current if n["type"] == "entity"]
included_relations = [n for n in included_orphans_current if n["type"] == "relation"]
included_chunks = [n for n in included_orphans_current if n["type"] == "chunk"]

print(f"  - Entities included: {len(included_entities)} of {len(orphan_entities)} ({100*len(included_entities)/len(orphan_entities):.1f}%)")
print(f"  - Relations included: {len(included_relations)} of {len(orphan_relations)} ({100*len(included_relations)/len(orphan_relations):.1f}%)")
print(f"  - Chunks included: {len(included_chunks)} of {len(orphan_chunks)} ({100*len(included_chunks)/len(orphan_chunks):.1f}%)")

# Simulate proposed balanced sampling logic
print(f"\n[PROPOSED BALANCED SAMPLING]")
if len(orphan_nodes) > 0:
    entity_orphan_limit = int(max_orphans_to_include * len(orphan_entities) / len(orphan_nodes))
    relation_orphan_limit = int(max_orphans_to_include * len(orphan_relations) / len(orphan_nodes))
    chunk_orphan_limit = max_orphans_to_include - entity_orphan_limit - relation_orphan_limit

    print(f"Entity orphans to include: {entity_orphan_limit} of {len(orphan_entities)}")
    print(f"Relation orphans to include: {relation_orphan_limit} of {len(orphan_relations)}")
    print(f"Chunk orphans to include: {chunk_orphan_limit} of {len(orphan_chunks)}")

# For small graphs (< 1000 nodes), check if all orphans would be included anyway
if G.number_of_nodes() < limit:
    print(f"\n[IMPORTANT NOTE]")
    print(f"This is a small graph ({G.number_of_nodes()} nodes < limit {limit})")
    print(f"ALL nodes including ALL orphans will be displayed (no sampling needed)")
    print(f"The 20% cap and balancing logic won't apply here!")

print(f"\n[RECOMMENDATION]")
if G.number_of_nodes() < limit:
    print(f"For this small graph, current logic should display ALL orphan nodes correctly.")
    print(f"Check the UI to verify all {len(orphan_entities)} orphan entities and {len(orphan_relations)} orphan relations are visible.")
else:
    print(f"For large graphs, consider:")
    print(f"1. Increasing the 20% cap OR adding a 'show all orphans' option for debugging")
    print(f"2. Implementing type-balanced orphan selection")
    print(f"3. Adding UI filters to show/hide orphan nodes by type")
