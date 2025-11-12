"""
Graph Statistics Service

Functions for retrieving knowledge graph statistics.
"""

import os
import json
from pathlib import Path
from typing import Optional
from bigrag.utils import logger

from .registry import registry


# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def compute_connectivity_analysis(G):
    """
    Compute detailed connectivity analysis for a graph.

    Args:
        G: NetworkX graph

    Returns:
        dict with connectivity metrics
    """
    import networkx as nx

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    # Separate entity and relation nodes
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get('role') == 'entity']
    relation_nodes = [n for n, d in G.nodes(data=True) if d.get('role') == 'relation']

    # Count orphan nodes (degree = 0)
    orphan_entities = sum(1 for n in entity_nodes if G.degree(n) == 0)
    orphan_relations = sum(1 for n in relation_nodes if G.degree(n) == 0)

    # Count nodes by connection degree
    single_connection = sum(1 for n in G.nodes() if G.degree(n) == 1)
    low_connection = sum(1 for n in G.nodes() if 2 <= G.degree(n) <= 5)
    high_connection = sum(1 for n in G.nodes() if G.degree(n) > 5)

    # Average degree
    degrees = [G.degree(n) for n in G.nodes()]
    avg_degree = sum(degrees) / total_nodes if total_nodes > 0 else 0.0

    # Graph density
    density = nx.density(G) if total_nodes > 1 else 0.0

    # Connected components analysis
    if G.is_directed():
        # For directed graphs, use weakly connected components
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    num_components = len(components)
    largest_component_size = max([len(c) for c in components]) if components else 0

    return {
        "total_nodes": total_nodes,
        "total_graph_edges": total_edges,
        "orphan_entities": orphan_entities,
        "orphan_relations": orphan_relations,
        "single_connection_nodes": single_connection,
        "low_connection_nodes": low_connection,
        "high_connection_nodes": high_connection,
        "avg_degree": round(avg_degree, 2),
        "density": round(density, 4),
        "largest_component_size": largest_component_size,
        "num_components": num_components
    }


async def get_graph_statistics(working_dir: str, dataset: Optional[str] = None):
    """
    Get knowledge graph statistics including connectivity analysis.

    Returns:
    - Global statistics (all datasets)
    - Per-dataset breakdown
    - Document counts by status
    - Entity, edge, and chunk counts
    - Connectivity analysis (orphans, degree distribution, components)
    """
    try:
        from ..models.models import DatasetStats, GraphStatsResponse, ConnectivityAnalysis

        datasets_to_query = []
        if dataset:
            datasets_to_query = [dataset]
        else:
            # Get all datasets
            working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
            expr_dir = PROJECT_ROOT / working_dir_base
            if expr_dir.exists():
                datasets_to_query = [
                    d.name for d in expr_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.')
                ]

        # Collect stats per dataset
        dataset_stats_list = []
        global_totals = {
            "total_documents": 0,
            "total_entities": 0,
            "total_relations": 0,
            "total_edges": 0,  # Deprecated: alias for total_relations
            "total_chunks": 0
        }

        for ds in datasets_to_query:
            # Get registry stats
            reg_stats = await registry.get_stats(ds)

            # Get KG file counts
            working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
            graph_file = str(PROJECT_ROOT / working_dir_base / ds / "graph_chunk_entity_relation.graphml")
            chunks_file = str(PROJECT_ROOT / working_dir_base / ds / "kv_store_text_chunks.json")
            full_docs_file = str(PROJECT_ROOT / working_dir_base / ds / "kv_store_full_docs.json")

            entities_count = 0
            edges_count = 0
            chunks_count = 0
            tokens_count = 0
            connectivity_data = None

            # Count entities and edges from GraphML file
            if os.path.exists(graph_file):
                try:
                    import networkx as nx
                    G = nx.read_graphml(graph_file)

                    for node, attrs in G.nodes(data=True):
                        role = attrs.get("role", "")
                        if role == "entity":
                            entities_count += 1
                        elif role == "relation":
                            edges_count += 1

                    # Compute connectivity analysis
                    connectivity_metrics = compute_connectivity_analysis(G)
                    connectivity_data = ConnectivityAnalysis(**connectivity_metrics)

                except Exception as e:
                    logger.warning(f"Failed to read GraphML for dataset {ds}: {e}")

            # Count chunks from KV storage
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    chunks_count = len(chunks)
                    tokens_count = sum(c.get("tokens", 0) for c in chunks.values())

            # Check if we need to count documents from kv_store_full_docs.json
            doc_count = reg_stats["total"]
            if doc_count == 0 and os.path.exists(full_docs_file):
                try:
                    with open(full_docs_file, 'r', encoding='utf-8') as f:
                        full_docs = json.load(f)
                        doc_count = len(full_docs)
                        reg_stats["indexed"] = doc_count
                        reg_stats["total"] = doc_count
                except Exception as e:
                    logger.debug(f"Could not read full_docs file: {e}")

            # Create dataset stats
            ds_stats = DatasetStats(
                dataset=ds,
                total_documents=doc_count,
                indexed_documents=reg_stats["indexed"],
                pending_documents=reg_stats["pending"],
                failed_documents=reg_stats["failed"],
                total_chunks=chunks_count,
                total_entities=entities_count,
                total_relations=edges_count,  # Relation node count
                total_edges=edges_count,  # Deprecated: alias for backward compatibility
                total_tokens=tokens_count,
                connectivity=connectivity_data  # Include connectivity analysis
            )

            dataset_stats_list.append(ds_stats)

            # Add to global totals
            global_totals["total_documents"] += doc_count
            global_totals["total_entities"] += entities_count
            global_totals["total_relations"] += edges_count
            global_totals["total_edges"] += edges_count  # Deprecated: alias
            global_totals["total_chunks"] += chunks_count

        return GraphStatsResponse(
            success=True,
            total_datasets=len(dataset_stats_list),
            global_stats=global_totals,
            datasets=dataset_stats_list
        )

    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to get graph statistics: {str(e)}")
