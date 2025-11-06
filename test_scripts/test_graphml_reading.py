"""
Test script to debug why backend isn't returning relation nodes and edges
"""

import networkx as nx
from collections import Counter

# Load the graph
print("Loading graph...")
G = nx.read_graphml("expr/demo_test/graph_chunk_entity_relation.graphml")

print(f"\nTotal nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")

# Check node types
node_roles = []
for node_id, attrs in G.nodes(data=True):
    role = attrs.get("role", "MISSING")
    node_roles.append(role)

role_counts = Counter(node_roles)
print(f"\nNode role distribution:")
for role, count in role_counts.items():
    print(f"  {role}: {count}")

# Sample nodes by type
print(f"\nSample entity node:")
for node_id, attrs in G.nodes(data=True):
    if attrs.get("role") == "entity":
        print(f"  ID: {node_id}")
        print(f"  Attrs: {attrs}")
        break

print(f"\nSample relation node (bipartite_edge):")
for node_id, attrs in G.nodes(data=True):
    if attrs.get("role") == "bipartite_edge":
        print(f"  ID: {node_id[:100]}...")
        print(f"  Attrs: {attrs}")
        break

# Check edges
print(f"\nSample edges:")
for source, target, attrs in list(G.edges(data=True))[:3]:
    print(f"  {source[:50]}... -> {target[:50]}...")
    print(f"    Attrs: {attrs}")

# Check if edges connect bipartite_edge nodes to entity nodes
print(f"\nEdge type analysis:")
edge_types = []
for source, target in G.edges():
    source_role = G.nodes[source].get("role", "MISSING")
    target_role = G.nodes[target].get("role", "MISSING")
    edge_types.append(f"{source_role} -> {target_role}")

edge_type_counts = Counter(edge_types)
print("Edge connections:")
for edge_type, count in edge_type_counts.most_common():
    print(f"  {edge_type}: {count}")
