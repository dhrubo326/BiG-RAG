"""
Simple KG Diagnostic: Check VDB-Graph ID Matching
"""

import sys
sys.path.insert(0, 'D:/BiG-RAG')

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
from bigrag import BiGRAG
import json

async def diagnose():
    print("=" * 80)
    print("SIMPLE KG DIAGNOSTIC: VDB-GRAPH ID MATCHING")
    print("=" * 80)

    # Initialize BiGRAG
    rag = BiGRAG(
        working_dir="D:/BiG-RAG/expr/kuet_production",
        addon_params={"language": "Bangla"}
    )

    graph = rag.chunk_entity_relation_graph

    # Get all entity IDs from VDB
    with open("D:/BiG-RAG/expr/kuet_production/vdb_entities.json", 'r', encoding='utf-8') as f:
        entity_vdb_data = json.load(f)

    vdb_entity_ids = set(item['__id__'] for item in entity_vdb_data['data'])

    # Get all entity IDs from Graph
    graph_entity_ids = {n for n, d in graph._graph.nodes(data=True) if d.get('role') == 'entity'}

    print(f"\n[Check 1] Entity ID Counts:")
    print(f"  VDB entries: {len(vdb_entity_ids)}")
    print(f"  Graph nodes: {len(graph_entity_ids)}")

    print(f"\n[Check 2] ID Overlap:")
    print(f"  IDs in BOTH:    {len(vdb_entity_ids & graph_entity_ids)}")
    print(f"  IDs in VDB only:  {len(vdb_entity_ids - graph_entity_ids)}")
    print(f"  IDs in Graph only: {len(graph_entity_ids - vdb_entity_ids)}")

    # Sample IDs
    print(f"\n[Check 3] Sample VDB Entity IDs:")
    for eid in list(vdb_entity_ids)[:5]:
        print(f"  {eid}")
        if eid in graph_entity_ids:
            print(f"    [OK] Found in graph")
        else:
            print(f"    [FAIL] NOT in graph")

    print(f"\n[Check 4] Sample Graph Entity IDs:")
    for eid in list(graph_entity_ids)[:5]:
        print(f"  {eid}")
        if eid in vdb_entity_ids:
            print(f"    [OK] Found in VDB")
        else:
            print(f"    [FAIL] NOT in VDB")

    # Check relation IDs
    with open("D:/BiG-RAG/expr/kuet_production/vdb_relations.json", 'r', encoding='utf-8') as f:
        relation_vdb_data = json.load(f)

    vdb_relation_ids = set(item['__id__'] for item in relation_vdb_data['data'])
    graph_relation_ids = {n for n, d in graph._graph.nodes(data=True) if d.get('role') == 'relation'}

    print(f"\n[Check 5] Relation ID Counts:")
    print(f"  VDB entries: {len(vdb_relation_ids)}")
    print(f"  Graph nodes: {len(graph_relation_ids)}")

    print(f"\n[Check 6] Relation ID Overlap:")
    print(f"  IDs in BOTH:    {len(vdb_relation_ids & graph_relation_ids)}")
    print(f"  IDs in VDB only:  {len(vdb_relation_ids - graph_relation_ids)}")
    print(f"  IDs in Graph only: {len(graph_relation_ids - vdb_relation_ids)}")

    # Sample relation IDs
    print(f"\n[Check 7] Sample VDB Relation IDs:")
    for rid in list(vdb_relation_ids)[:3]:
        print(f"  {rid}")
        if rid in graph_relation_ids:
            print(f"    [OK] Found in graph")
        else:
            print(f"    [FAIL] NOT in graph")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

    # Verdict
    entity_match_rate = len(vdb_entity_ids & graph_entity_ids) / len(vdb_entity_ids) * 100 if vdb_entity_ids else 0
    relation_match_rate = len(vdb_relation_ids & graph_relation_ids) / len(vdb_relation_ids) * 100 if vdb_relation_ids else 0

    print(f"\nVERDICT:")
    print(f"  Entity ID match rate: {entity_match_rate:.1f}%")
    print(f"  Relation ID match rate: {relation_match_rate:.1f}%")

    if entity_match_rate == 100 and relation_match_rate == 100:
        print(f"  [OK] VDB and Graph IDs are perfectly aligned!")
    else:
        print(f"  [PROBLEM] VDB and Graph IDs are NOT aligned!")
        print(f"  This is why Path A and Path B retrieval return 0 results.")

if __name__ == "__main__":
    asyncio.run(diagnose())
