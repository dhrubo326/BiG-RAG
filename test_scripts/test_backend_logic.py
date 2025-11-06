"""
Test the exact same logic as the backend to see where the issue is
"""

import networkx as nx
import html

def _extract_relation_text(node_id: str) -> str:
    """Extract clean relation text from bipartite_edge node IDs."""
    text = node_id
    if text.startswith("<bipartite_edge>"):
        text = text[len("<bipartite_edge>"):]
    text = html.unescape(text)
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text.strip()

# Load graph
print("Loading graph...")
G = nx.read_graphml("expr/demo_test/graph_chunk_entity_relation.graphml")
print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")

# Simulate backend logic
all_nodes = []
entity_count = 0
relation_count = 0
chunk_count = 0

print("\nProcessing nodes (first 10)...")
for i, (node_id, attrs) in enumerate(G.nodes(data=True)):
    role = attrs.get("role", "")
    node_type = "entity"  # default

    if role == "entity":
        node_type = "entity"
        entity_count += 1
    elif role == "bipartite_edge":
        node_type = "relation"
        relation_count += 1
    elif role == "chunk" or node_id.startswith("chunk-"):
        node_type = "chunk"
        chunk_count += 1

    # Extract label
    if node_type == "relation":
        label = _extract_relation_text(node_id)
    else:
        label = attrs.get("name", node_id)

    node = {
        "id": node_id,
        "label": label,
        "type": node_type,
        "weight": float(attrs.get("weight", 0.0)),
        "role": role
    }
    all_nodes.append(node)

    if i < 10:
        print(f"  Node {i+1}: type={node_type}, role={role}, label={label[:50] if len(label) > 50 else label}")

print(f"\nTotal processed: {len(all_nodes)} nodes")
print(f"  Entities: {entity_count}")
print(f"  Relations: {relation_count}")
print(f"  Chunks: {chunk_count}")

# Sample top 50 by weight (like backend does with top_weighted strategy)
limit = 50
sampled_nodes = sorted(all_nodes, key=lambda x: x["weight"], reverse=True)[:limit]

print(f"\nAfter sampling (top {limit} by weight):")
sampled_entity_count = len([n for n in sampled_nodes if n["type"] == "entity"])
sampled_relation_count = len([n for n in sampled_nodes if n["type"] == "relation"])
print(f"  Entities: {sampled_entity_count}")
print(f"  Relations: {sampled_relation_count}")

# Show the weights
print(f"\nWeight distribution in sampled nodes:")
print(f"  Min weight: {min(n['weight'] for n in sampled_nodes)}")
print(f"  Max weight: {max(n['weight'] for n in sampled_nodes)}")

print(f"\nRelation node weights:")
relation_weights = [n['weight'] for n in all_nodes if n['type'] == 'relation']
print(f"  Min: {min(relation_weights)}")
print(f"  Max: {max(relation_weights)}")
print(f"  Avg: {sum(relation_weights)/len(relation_weights):.1f}")

print(f"\nEntity node weights:")
entity_weights = [n['weight'] for n in all_nodes if n['type'] == 'entity']
print(f"  Min: {min(entity_weights)}")
print(f"  Max: {max(entity_weights)}")
print(f"  Avg: {sum(entity_weights)/len(entity_weights):.1f}")

# Check edges
sampled_node_ids = {node["id"] for node in sampled_nodes}
edges = []
for source, target, attrs in G.edges(data=True):
    if source in sampled_node_ids and target in sampled_node_ids:
        edges.append({
            "source": source,
            "target": target,
            "weight": float(attrs.get("weight", 1.0))
        })

print(f"\nEdges between sampled nodes: {len(edges)}")
