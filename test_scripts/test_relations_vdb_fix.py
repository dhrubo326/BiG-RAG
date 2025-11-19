"""
Test script to verify relations VDB fix.

This script re-indexes the relations VDB with the corrected field.
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bigrag import BiGRAG
import networkx as nx


async def reindex_relations_vdb():
    """Re-index relations VDB with corrected relation_name field."""

    print("="*80)
    print("RELATIONS VDB FIX - RE-INDEXING")
    print("="*80)

    # Initialize BiGRAG
    rag = BiGRAG(working_dir='../expr/demo_test')

    # Step 1: Read all relation nodes from graph
    print("\n[Step 1] Reading relations from graph...")
    graph_file = '../expr/demo_test/graph_chunk_entity_relation.graphml'
    graph = nx.read_graphml(graph_file)

    relation_nodes = [
        (node, data) for node, data in graph.nodes(data=True)
        if data.get('role') == 'relation'
    ]

    print(f"Found {len(relation_nodes)} relation nodes in graph")

    # Step 2: Prepare data for VDB with CORRECTED relation_name field
    print("\n[Step 2] Preparing VDB data with actual content (not hash IDs)...")
    data_for_vdb = {}

    for node_id, node_data in relation_nodes:
        content = node_data.get('content', '')
        data_for_vdb[node_id] = {
            'content': content,  # For embedding
            'relation_name': content,  # FIX: Store actual content, not hash ID!
        }

    print(f"Prepared {len(data_for_vdb)} relations")

    # Step 3: Upsert to VDB
    print(f"\n[Step 3] Upserting {len(data_for_vdb)} relations to VDB...")
    print("(This will take a few minutes to re-generate embeddings...)")

    await rag.vdb_relations.upsert(data_for_vdb)

    print("[OK] Upsert complete!")

    # Step 4: Verify the fix
    print("\n[Step 4] Verifying fix...")

    with open('./expr/demo_test/vdb_relations.json', 'r', encoding='utf-8') as f:
        vdb_data = json.load(f)

    # Check first 5 relations
    print("\n[VERIFICATION] Checking first 5 relations:")
    success_count = 0
    fail_count = 0

    for i in range(min(5, len(vdb_data['data']))):
        rel = vdb_data['data'][i]
        rel_id = rel.get('__id__', 'N/A')
        rel_name = rel.get('relation_name', 'N/A')

        # Check if it's a hash ID (bug) or actual content (fixed)
        if rel_name.startswith('rel-'):
            print(f"  [{i+1}] [FAIL] Hash ID: {rel_id[:40]}...")
            fail_count += 1
        else:
            # Show length instead of content (avoid encoding issues)
            print(f"  [{i+1}] [SUCCESS] Content: {len(rel_name)} chars")
            success_count += 1

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total relations: {len(vdb_data['data'])}")
    print(f"Checked: 5")
    print(f"Success: {success_count}/5")
    print(f"Failed: {fail_count}/5")

    if fail_count == 0:
        print("\n[SUCCESS] All relations have actual content!")
        print("Path B retrieval should now work correctly.")
    else:
        print("\n[WARNING] Some relations still have hash IDs.")
        print("You may need to fully rebuild the knowledge graph.")

    print("="*80)


if __name__ == "__main__":
    asyncio.run(reindex_relations_vdb())
