"""Analyze CUET diagnosis test graph quality."""
import json
import networkx as nx
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("CUET KNOWLEDGE GRAPH QUALITY ANALYSIS")
print("=" * 80)

# Load GraphML
graph_path = r'D:\BiG-RAG\expr\cuet_diagnosis_test\graph_chunk_entity_relation.graphml'
G = nx.read_graphml(graph_path)

# Count nodes by role
entities = [n for n in G.nodes() if G.nodes[n].get('role') == 'entity']
relations = [n for n in G.nodes() if G.nodes[n].get('role') == 'relation']
chunks = [n for n in G.nodes() if n.startswith('chunk-')]

print(f"\n[GRAPH STRUCTURE]")
print(f"Total nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")
print(f"Entity nodes: {len(entities)}")
print(f"Relation nodes: {len(relations)}")
print(f"Chunk nodes: {len(chunks)}")

# Load vector DBs
vdb_entities_path = r'D:\BiG-RAG\expr\cuet_diagnosis_test\vdb_entities.json'
vdb_relations_path = r'D:\BiG-RAG\expr\cuet_diagnosis_test\vdb_relations.json'
vdb_chunks_path = r'D:\BiG-RAG\expr\cuet_diagnosis_test\vdb_chunks.json'

with open(vdb_entities_path, encoding='utf-8') as f:
    vdb_entities = json.load(f)
with open(vdb_relations_path, encoding='utf-8') as f:
    vdb_relations = json.load(f)
with open(vdb_chunks_path, encoding='utf-8') as f:
    vdb_chunks = json.load(f)

print(f"\n[VECTOR DATABASE INDEXING]")
print(f"Entities indexed: {len(vdb_entities.get('data', []))}")
print(f"Relations indexed: {len(vdb_relations.get('data', []))}")
print(f"Chunks indexed: {len(vdb_chunks.get('data', []))}")

# Sample entities
print(f"\n[SAMPLE ENTITIES]")
for i, entity_id in enumerate(entities[:5]):
    entity = G.nodes[entity_id]
    entity_type = entity.get('entity_type', 'N/A')
    weight = entity.get('weight', 'N/A')
    degree = G.degree(entity_id)
    print(f"{i+1}. {entity_id[:60]}")
    print(f"   Type: {entity_type}, Weight: {weight}, Degree: {degree}")

# Sample relations
print(f"\n[SAMPLE RELATIONS]")
for i, rel_id in enumerate(relations[:5]):
    rel = G.nodes[rel_id]
    content = rel.get('content', 'N/A')[:80]
    weight = rel.get('weight', 'N/A')
    degree = G.degree(rel_id)
    print(f"{i+1}. {content}...")
    print(f"   Weight: {weight}, Degree: {degree}")

# Check orphan nodes
orphan_entities = [e for e in entities if G.degree(e) == 0]
orphan_relations = [r for r in relations if G.degree(r) == 0]

print(f"\n[ORPHAN NODES]")
print(f"Orphan entities: {len(orphan_entities)} ({len(orphan_entities)/len(entities)*100:.1f}%)")
print(f"Orphan relations: {len(orphan_relations)} ({len(orphan_relations)/len(relations)*100:.1f}%)")

if len(orphan_entities) > 0:
    print(f"\nOrphan entity sample (first 10):")
    for i, e in enumerate(orphan_entities[:10]):
        e_data = G.nodes[e]
        print(f"  {i+1}. {e[:80]}")
        print(f"      Type: {e_data.get('entity_type', 'N/A')}, Source: {e_data.get('source_id', 'N/A')}")

if len(orphan_relations) > 0:
    print(f"\nOrphan relation sample (first 10):")
    for i, r in enumerate(orphan_relations[:10]):
        rel_content = G.nodes[r].get('content', r)
        rel_source = G.nodes[r].get('source_id', 'N/A')
        print(f"  {i+1}. {rel_content[:80]}")
        print(f"      Source: {rel_source}")

# Load chunks
chunks_path = r'D:\BiG-RAG\expr\cuet_diagnosis_test\kv_store_text_chunks.json'
with open(chunks_path, encoding='utf-8') as f:
    chunks_data = json.load(f)

print(f"\n[TEXT CHUNKS]")
print(f"Total chunks: {len(chunks_data)}")
for i, (chunk_id, chunk) in enumerate(list(chunks_data.items())[:3]):
    content = chunk.get('content', '')[:60]
    title = chunk.get('doc_title', 'N/A')
    print(f"{i+1}. {content}...")
    print(f"   Title: {title}")

# Quality metrics
print(f"\n[QUALITY METRICS]")
print(f"Entities/Chunk ratio: {len(entities)/len(chunks_data):.2f}")
print(f"Relations/Chunk ratio: {len(relations)/len(chunks_data):.2f}")
print(f"Avg edges per entity: {sum([G.degree(e) for e in entities])/len(entities):.2f}")
print(f"Avg edges per relation: {sum([G.degree(r) for r in relations])/len(relations):.2f}")

# Edge type analysis
print(f"\n[EDGE TYPE ANALYSIS]")
sample_edges = list(G.edges())[:10]
print(f"Sample edges (first 10):")
for i, (src, dst) in enumerate(sample_edges):
    src_role = G.nodes[src].get('role', 'unknown')
    dst_role = G.nodes[dst].get('role', 'unknown')
    print(f"  {i+1}. {src[:30]} ({src_role}) <-> {dst[:30]} ({dst_role})")

# Critical issues
print(f"\n[CRITICAL ISSUES]")
issues = []

vdb_entities_count = len(vdb_entities.get('data', []))
vdb_relations_count = len(vdb_relations.get('data', []))
vdb_chunks_count = len(vdb_chunks.get('data', []))

if vdb_entities_count == 0:
    issues.append("[FAIL] Entity vector DB has no data")
elif vdb_entities_count != len(entities):
    issues.append(f"[WARN] Entity VDB count ({vdb_entities_count}) != Graph entities ({len(entities)})")

if vdb_relations_count == 0:
    issues.append("[FAIL] Relation vector DB has no data")
elif vdb_relations_count != len(relations):
    issues.append(f"[WARN] Relation VDB count ({vdb_relations_count}) != Graph relations ({len(relations)})")

if vdb_chunks_count == 0:
    issues.append("[FAIL] Chunk vector DB has no data")

if len(orphan_entities) > len(entities) * 0.1:
    issues.append(f"[WARN] High orphan entity rate: {len(orphan_entities)/len(entities)*100:.1f}%")
if len(orphan_relations) > len(relations) * 0.1:
    issues.append(f"[WARN] High orphan relation rate: {len(orphan_relations)/len(relations)*100:.1f}%")

if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  [OK] No critical issues found")

print("\n" + "=" * 80)
