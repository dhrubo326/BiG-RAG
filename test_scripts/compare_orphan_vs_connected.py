"""Compare orphan vs connected entities from SAME chunk."""
import networkx as nx
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("COMPARING ORPHAN VS CONNECTED ENTITIES FROM SAME CHUNK")
print("=" * 80)

# Load graph
G = nx.read_graphml(r'D:\BiG-RAG\expr\bangla_diagnosis_test\graph_chunk_entity_relation.graphml')

# Focus on chunk_0001 (has both orphan and connected entities)
chunk_id = "chunk_0001"

entities = [n for n in G.nodes() if G.nodes[n].get('role') == 'entity']
relations = [n for n in G.nodes() if G.nodes[n].get('role') == 'relation']

chunk_entities = [e for e in entities if G.nodes[e].get('source_id') == chunk_id]
chunk_relations = [r for r in relations if G.nodes[r].get('source_id') == chunk_id]

orphan_entities = [e for e in chunk_entities if G.degree(e) == 0]
connected_entities = [e for e in chunk_entities if G.degree(e) > 0]

print(f"\n[CHUNK {chunk_id} ANALYSIS]")
print(f"Total entities from this chunk: {len(chunk_entities)}")
print(f"  - Connected: {len(connected_entities)}")
print(f"  - Orphan: {len(orphan_entities)}")
print(f"Relations from this chunk: {len(chunk_relations)}")

# Analyze relations to see which entities they connect to
print(f"\n[RELATION CONNECTION ANALYSIS]")
for i, r_id in enumerate(chunk_relations[:3]):
    r = G.nodes[r_id]
    content = r.get('content', r_id)[:80]
    neighbors = list(G.neighbors(r_id))

    print(f"\nRelation {i+1}: {content}")
    print(f"  Degree: {G.degree(r_id)}")
    print(f"  Connected entities ({len(neighbors)}):")
    for n in neighbors:
        n_data = G.nodes[n]
        n_role = n_data.get('role', 'unknown')
        entity_type = n_data.get('entity_type', 'N/A')
        print(f"    - {n[:60]} (role={n_role}, type={entity_type})")

# Key insight: Check if orphan entities appear in the content of relations
print(f"\n[ORPHAN ENTITY NAMES IN RELATION CONTENT]")
for e_id in orphan_entities[:5]:
    print(f"\nOrphan entity: {e_id}")

    # Check if this entity name appears in any relation content from this chunk
    found_in_relations = []
    for r_id in chunk_relations:
        r_content = G.nodes[r_id].get('content', '')
        if e_id.lower() in r_content.lower() or e_id[:20] in r_content:
            found_in_relations.append((r_id, r_content[:80]))

    if found_in_relations:
        print(f"  Found in {len(found_in_relations)} relation(s) content:")
        for r_id, content in found_in_relations[:2]:
            print(f"    - {content}")
    else:
        print(f"  NOT found in any relation content (this may explain orphan status)")

# Load the actual chunk content to see what was extracted
print(f"\n[ACTUAL CHUNK CONTENT]")
chunks_path = r'D:\BiG-RAG\expr\bangla_diagnosis_test\kv_store_text_chunks.json'
with open(chunks_path, encoding='utf-8') as f:
    chunks_data = json.load(f)

# Find chunk_0001
chunk_content = None
for chunk_key, chunk_data in chunks_data.items():
    if chunk_data.get('chunk_order_index') == 1 or 'chunk-' in chunk_key:
        # This might be chunk_0001
        if 'সিভিল ইঞ্জিনিয়ারিং' in chunk_data.get('content', ''):
            chunk_content = chunk_data.get('content', '')
            print(f"Found chunk content ({len(chunk_content)} chars)")
            print(f"Preview: {chunk_content[:200]}")
            break

if chunk_content:
    # Check if orphan entity names appear in chunk content
    print(f"\n[ORPHAN ENTITIES IN CHUNK CONTENT]")
    for e_id in orphan_entities[:5]:
        if e_id in chunk_content:
            print(f"  ✓ '{e_id[:40]}' found in chunk content")
        else:
            print(f"  ✗ '{e_id[:40]}' NOT in chunk content")

print("\n" + "=" * 80)
print("HYPOTHESIS:")
print("Orphan entities were extracted by LLM BUT the linking phase failed because:")
print("1. Entity extraction happened without immediate relation context")
print("2. The relation was created but entity wasn't linked to it")
print("3. The _merge_edges_then_upsert function only creates edges for entities")
print("   that have the 'hyper_relation' field in their node data")
print("=" * 80)
