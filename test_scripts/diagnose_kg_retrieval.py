"""
Diagnostic Script: Knowledge Graph Retrieval Issue
Investigates why Path A and Path B return 0 results while Path C works.

Usage:
    python test_scripts/diagnose_kg_retrieval.py
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
from bigrag.base import QueryParam
import json

async def diagnose_kg_retrieval():
    """
    Step-by-step diagnosis of KG retrieval process.
    """
    print("=" * 80)
    print("KNOWLEDGE GRAPH RETRIEVAL DIAGNOSIS")
    print("=" * 80)

    # Initialize BiGRAG
    print("\n[Step 1] Initializing BiGRAG...")
    rag = BiGRAG(
        working_dir="D:/BiG-RAG/expr/kuet_production",
        addon_params={"language": "Bangla"}
    )

    # Check graph structure
    print("\n[Step 2] Analyzing Graph Structure...")
    graph = rag.chunk_entity_relation_graph

    print(f"  Total nodes: {graph._graph.number_of_nodes()}")
    print(f"  Total edges: {graph._graph.number_of_edges()}")

    # Sample nodes
    nodes = list(graph._graph.nodes(data=True))[:10]
    print(f"\n[Step 3] Sample Nodes (first 10):")
    for node_id, attrs in nodes:
        print(f"    {node_id}: {attrs.get('role', 'unknown')} - name={attrs.get('entity_name', attrs.get('name', 'N/A'))}")

    # Check entity nodes specifically
    entity_nodes = [(n, d) for n, d in graph._graph.nodes(data=True) if d.get('role') == 'entity']
    relation_nodes = [(n, d) for n, d in graph._graph.nodes(data=True) if d.get('role') == 'relation']

    print(f"\n[Step 4] Node Distribution:")
    print(f"  Entity nodes: {len(entity_nodes)}")
    print(f"  Relation nodes: {len(relation_nodes)}")
    print(f"  Other nodes: {len(nodes) - len(entity_nodes) - len(relation_nodes)}")

    # Sample entity and relation
    if entity_nodes:
        sample_entity = entity_nodes[0]
        print(f"\n[Step 5] Sample Entity Node:")
        print(f"  ID: {sample_entity[0]}")
        print(f"  Attributes: {json.dumps(sample_entity[1], indent=4, ensure_ascii=False)}")

    if relation_nodes:
        sample_relation = relation_nodes[0]
        print(f"\n[Step 6] Sample Relation Node:")
        print(f"  ID: {sample_relation[0]}")
        print(f"  Attributes: {json.dumps(sample_relation[1], indent=4, ensure_ascii=False)}")

    # Check VDB structure
    print(f"\n[Step 7] Vector DB Structure:")

    # Check entity VDB (use private __storage attribute)
    entity_vdb_data = rag.vdb_entities._client._NanoVectorDB__storage
    print(f"  Entity VDB entries: {len(entity_vdb_data)}")
    if len(entity_vdb_data) > 0:
        sample_entity_vdb = list(entity_vdb_data.items())[0]
        print(f"    Sample key: {sample_entity_vdb[0]}")
        print(f"    Sample data keys: {list(sample_entity_vdb[1].keys())}")
        if 'entity_name' in sample_entity_vdb[1]:
            print(f"    entity_name: {sample_entity_vdb[1]['entity_name']}")

    # Check relation VDB (use private __storage attribute)
    relation_vdb_data = rag.vdb_relations._client._NanoVectorDB__storage
    print(f"  Relation VDB entries: {len(relation_vdb_data)}")
    if len(relation_vdb_data) > 0:
        sample_relation_vdb = list(relation_vdb_data.items())[0]
        print(f"    Sample key: {sample_relation_vdb[0]}")
        print(f"    Sample data keys: {list(sample_relation_vdb[1].keys())}")
        if 'relation_name' in sample_relation_vdb[1]:
            print(f"    relation_name: {sample_relation_vdb[1]['relation_name']}")

    # Test Path A: Entity retrieval
    print(f"\n[Step 8] Testing Path A (Entity Retrieval)...")
    query = "KUET এর EEE বিভাগে সিট কত?"

    # Search entity VDB
    from bigrag.llm import openai_embedding
    query_embedding = await openai_embedding([query])

    entity_results = rag.vdb_entities._client.query(
        query=query_embedding[0],
        top_k=10
    )

    print(f"  Entity VDB search returned: {len(entity_results)} results")
    if entity_results:
        print(f"  Top 3 entities:")
        for i, result in enumerate(entity_results[:3], 1):
            entity_id = result.get('__id__')
            entity_name = result.get('entity_name', 'N/A')
            distance = result.get('__metrics__', {}).get('distance', 'N/A')
            print(f"    {i}. ID={entity_id}, name={entity_name}, distance={distance}")

            # Check if this entity exists in graph
            if graph._graph.has_node(entity_id):
                print(f"       ✅ Found in graph")
                node_data = graph._graph.nodes[entity_id]
                print(f"       Graph node name: {node_data.get('entity_name', node_data.get('name', 'N/A'))}")
            else:
                print(f"       ❌ NOT found in graph (THIS IS THE PROBLEM!)")

    # Test Path B: Relation retrieval
    print(f"\n[Step 9] Testing Path B (Relation Retrieval)...")

    relation_results = rag.vdb_relations._client.query(
        query=query_embedding[0],
        top_k=10
    )

    print(f"  Relation VDB search returned: {len(relation_results)} results")
    if relation_results:
        print(f"  Top 3 relations:")
        for i, result in enumerate(relation_results[:3], 1):
            relation_id = result.get('__id__')
            relation_name = result.get('relation_name', 'N/A')
            distance = result.get('__metrics__', {}).get('distance', 'N/A')
            print(f"    {i}. ID={relation_id}, name={relation_name}, distance={distance}")

            # Check if this relation exists in graph
            if graph._graph.has_node(relation_id):
                print(f"       ✅ Found in graph")
            else:
                print(f"       ❌ NOT found in graph (THIS IS THE PROBLEM!)")

    # Compare VDB IDs with Graph IDs
    print(f"\n[Step 10] ID Mismatch Analysis:")

    vdb_entity_ids = set(rag.vdb_entities._client._NanoVectorDB__storage.keys())
    graph_entity_ids = {n for n, d in graph._graph.nodes(data=True) if d.get('role') == 'entity'}

    print(f"  Entity IDs in VDB: {len(vdb_entity_ids)}")
    print(f"  Entity nodes in Graph: {len(graph_entity_ids)}")
    print(f"  IDs in VDB but NOT in Graph: {len(vdb_entity_ids - graph_entity_ids)}")
    print(f"  IDs in Graph but NOT in VDB: {len(graph_entity_ids - vdb_entity_ids)}")

    if vdb_entity_ids - graph_entity_ids:
        print(f"\n  Sample VDB-only IDs (first 5):")
        for vdb_id in list(vdb_entity_ids - graph_entity_ids)[:5]:
            print(f"    {vdb_id}")

    if graph_entity_ids - vdb_entity_ids:
        print(f"\n  Sample Graph-only IDs (first 5):")
        for graph_id in list(graph_entity_ids - vdb_entity_ids)[:5]:
            node_data = graph._graph.nodes[graph_id]
            print(f"    {graph_id} - name={node_data.get('entity_name', node_data.get('name', 'N/A'))}")

    # Check for naming mismatches
    print(f"\n[Step 11] Checking VDB-Graph Linking...")

    # Get first entity from VDB
    if vdb_entity_ids:
        sample_vdb_id = list(vdb_entity_ids)[0]
        vdb_entry = rag.vdb_entities._client._NanoVectorDB__storage[sample_vdb_id]

        print(f"  VDB Entry:")
        print(f"    ID: {sample_vdb_id}")
        print(f"    entity_name: {vdb_entry.get('entity_name', 'N/A')}")

        # Try to find corresponding graph node
        if graph._graph.has_node(sample_vdb_id):
            graph_node = graph._graph.nodes[sample_vdb_id]
            print(f"  Graph Node (by ID match):")
            print(f"    entity_name: {graph_node.get('entity_name', 'N/A')}")
            print(f"    name: {graph_node.get('name', 'N/A')}")
            print(f"    ✅ VDB and Graph are linked correctly")
        else:
            # Try finding by name
            entity_name_in_vdb = vdb_entry.get('entity_name', '')
            matching_graph_nodes = [
                (n, d) for n, d in graph._graph.nodes(data=True)
                if d.get('entity_name') == entity_name_in_vdb or d.get('name') == entity_name_in_vdb
            ]

            if matching_graph_nodes:
                print(f"  ❌ ID mismatch, but found by name:")
                print(f"    Graph ID: {matching_graph_nodes[0][0]}")
                print(f"    VDB ID:  {sample_vdb_id}")
                print(f"    CONCLUSION: IDs don't match between VDB and Graph!")
            else:
                print(f"  ❌ Entity not found in graph at all!")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(diagnose_kg_retrieval())
