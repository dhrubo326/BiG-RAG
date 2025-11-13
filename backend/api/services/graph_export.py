"""
Graph Export Service

Functions for exporting and querying graph data.
"""

import os
import json
import html
import random
from pathlib import Path
from typing import Optional
from bigrag.utils import logger
import networkx as nx
from fastapi import HTTPException


# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _extract_relation_text(node_id: str) -> str:
    """
    Extract clean relation text from relation node IDs.

    Example:
        Input: '<relation>"Netflix is an American streaming service"'
        Output: 'Netflix is an American streaming service'
    """
    # Remove <relation> prefix if present
    text = node_id
    if text.startswith("<relation>"):
        text = text[len("<relation>"):]

    # Decode HTML entities (e.g., &quot; -> ")
    text = html.unescape(text)

    # Remove surrounding quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    return text.strip()


async def export_graph_for_cytoscape(
    data_source: str,
    limit: int = 1000,
    node_types: Optional[list] = None,
    edge_types: Optional[list] = None,
    min_weight: float = 0.0,
    sample_strategy: str = "top_weighted",
    include_all_orphans: bool = False
):
    """
    Export the knowledge graph for a dataset in Cytoscape-compatible format.

    OPTIMIZED FOR LARGE GRAPHS - Implements sampling and filtering to prevent browser freezing.

    Parameters:
    - data_source: Dataset name (e.g., "SingleTopic", "HotpotQA")
    - limit: Maximum nodes to return (default: 1000, max: 5000)
    - node_types: List of node types to include
    - edge_types: List of edge types to include (not currently used)
    - min_weight: Minimum node weight threshold (0.0-1.0)
    - sample_strategy: "top_weighted" (highest weight), "random", "diverse" (balanced types)
    - include_all_orphans: If True, include ALL orphan nodes regardless of limit (for debugging)

    Returns:
    - nodes: Sampled list of graph nodes
    - edges: Edges connecting sampled nodes
    - stats: Full graph statistics (unsampled)
    - sampling_info: Information about sampling applied
    - orphan_breakdown: Detailed breakdown of orphan nodes by type
    """
    try:
        # Enforce maximum limit for browser performance
        MAX_LIMIT = 5000
        if limit and limit > MAX_LIMIT:
            limit = MAX_LIMIT
            logger.warning(f"Limit exceeded max, capping at {MAX_LIMIT}")

        # Get graph file path
        working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
        graph_file = str(PROJECT_ROOT / working_dir_base / data_source / "graph_chunk_entity_relation.graphml")
        chunks_file = str(PROJECT_ROOT / working_dir_base / data_source / "kv_store_text_chunks.json")

        if not os.path.exists(graph_file):
            raise HTTPException(
                status_code=404,
                detail=f"Graph file not found for dataset '{data_source}'. "
                       f"Please ensure the dataset is built. Path: {graph_file}"
            )

        # Read the graph
        logger.info(f"Loading graph from {graph_file}")
        G = nx.read_graphml(graph_file)
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        logger.info(f"Graph loaded: {total_nodes} nodes, {total_edges} edges")

        # Parse node type filter
        allowed_types = None
        if node_types:
            type_map = {
                'entity': 'entity',
                'relation': 'relation',
                'chunk': 'chunk'
            }
            allowed_types = [type_map.get(t, t) for t in node_types]

        # First pass: collect all nodes with metadata (lightweight)
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

            # Calculate connections (degree) - CRITICAL for orphan detection
            connections = G.degree(node_id)
            if connections == 0:
                orphan_count += 1

            # Apply filters
            if allowed_types and role not in allowed_types:
                continue

            weight = float(attrs.get("weight", 0.0))
            if weight < min_weight:
                continue

            # Extract label and content based on node type
            if node_type == "relation":
                # For relation nodes: use d1 (content) field as main data
                content = attrs.get("content", "")
                if not content:
                    # Fallback: extract from node_id (legacy support)
                    content = _extract_relation_text(node_id)
                # Create short label (first 50 chars)
                label = content[:50] + "..." if len(content) > 50 else content
            elif node_type == "entity":
                # For entity nodes: use name, store description separately
                label = attrs.get("name", node_id)
                # Remove quotes if present
                if label.startswith('"') and label.endswith('"'):
                    label = label[1:-1]
            else:
                # For chunk nodes
                label = attrs.get("name", node_id)

            # Lightweight node object (detailed fields loaded later for sampled nodes)
            node = {
                "id": node_id,
                "label": label,
                "name": label,
                "type": node_type,
                "description": "",  # Load later for sampled nodes only
                "content": "",      # Load later for relation nodes
                "weight": weight,
                "connections": connections,  # NEW: degree count for orphan detection
                "sourceId": attrs.get("source_id", ""),
                "role": role,
                "metadata": {
                    "entityType": attrs.get("entity_type", ""),
                    "role": role,
                }
            }
            all_nodes.append(node)

        logger.info(f"After filtering: {len(all_nodes)} nodes")

        # ✅ NEW: Identify orphan nodes (important for debugging)
        orphan_nodes = [n for n in all_nodes if n["connections"] == 0]
        non_orphan_nodes = [n for n in all_nodes if n["connections"] > 0]

        # Breakdown orphan nodes by type
        orphan_entities = [n for n in orphan_nodes if n["type"] == "entity"]
        orphan_relations = [n for n in orphan_nodes if n["type"] == "relation"]
        orphan_chunks = [n for n in orphan_nodes if n["type"] == "chunk"]

        logger.info(f"Found {len(orphan_nodes)} total orphan nodes:")
        logger.info(f"  - {len(orphan_entities)} orphan entity nodes")
        logger.info(f"  - {len(orphan_relations)} orphan relation nodes")
        logger.info(f"  - {len(orphan_chunks)} orphan chunk nodes")

        # Apply sampling strategy
        sampled_nodes = all_nodes
        sampling_applied = False

        if len(all_nodes) > limit:
            sampling_applied = True
            logger.info(f"Applying {sample_strategy} sampling to get {limit} nodes from {len(all_nodes)}")

            # ✅ NEW: Handle orphan node inclusion
            if include_all_orphans:
                # DEBUG MODE: Include ALL orphan nodes regardless of limit
                logger.info(f"[DEBUG MODE] Including ALL {len(orphan_nodes)} orphan nodes (include_all_orphans=True)")
                included_orphans = orphan_nodes
                remaining_limit = max(0, limit - len(included_orphans))
                logger.info(f"Including {len(included_orphans)} orphan nodes + {remaining_limit} regular nodes")
            else:
                # NORMAL MODE: Include orphan nodes up to 20% of limit with type-balanced selection
                max_orphans_to_include = min(len(orphan_nodes), int(limit * 0.2))

                # Type-balanced orphan selection
                if len(orphan_nodes) > 0:
                    entity_orphan_limit = int(max_orphans_to_include * len(orphan_entities) / len(orphan_nodes))
                    relation_orphan_limit = int(max_orphans_to_include * len(orphan_relations) / len(orphan_nodes))
                    chunk_orphan_limit = max_orphans_to_include - entity_orphan_limit - relation_orphan_limit

                    # Sort each type by weight (highest first) and take top N
                    selected_entity_orphans = sorted(orphan_entities, key=lambda x: x["weight"], reverse=True)[:entity_orphan_limit]
                    selected_relation_orphans = sorted(orphan_relations, key=lambda x: x["weight"], reverse=True)[:relation_orphan_limit]
                    selected_chunk_orphans = sorted(orphan_chunks, key=lambda x: x["weight"], reverse=True)[:chunk_orphan_limit]

                    included_orphans = selected_entity_orphans + selected_relation_orphans + selected_chunk_orphans

                    logger.info(f"Type-balanced orphan selection: {len(selected_entity_orphans)} entities, "
                               f"{len(selected_relation_orphans)} relations, {len(selected_chunk_orphans)} chunks")
                else:
                    included_orphans = []

                remaining_limit = limit - len(included_orphans)
                logger.info(f"Including {len(included_orphans)} orphan nodes + {remaining_limit} regular nodes")

            if sample_strategy == "top_weighted":
                # Sort by weight descending, take top N
                sampled_regular = sorted(non_orphan_nodes, key=lambda x: x["weight"], reverse=True)[:remaining_limit]
                sampled_nodes = included_orphans + sampled_regular

            elif sample_strategy == "random":
                # Random sampling
                if remaining_limit < len(non_orphan_nodes):
                    sampled_regular = random.sample(non_orphan_nodes, remaining_limit)
                else:
                    sampled_regular = non_orphan_nodes
                sampled_nodes = included_orphans + sampled_regular

            elif sample_strategy == "diverse":
                # Balanced sampling across node types (excluding already included orphans)
                entities = [n for n in non_orphan_nodes if n["type"] == "entity"]
                relations = [n for n in non_orphan_nodes if n["type"] == "relation"]
                chunks = [n for n in non_orphan_nodes if n["type"] == "chunk"]

                # Allocate proportionally
                total = len(non_orphan_nodes)
                entity_limit = int(remaining_limit * len(entities) / total) if total > 0 else 0
                relation_limit = int(remaining_limit * len(relations) / total) if total > 0 else 0
                chunk_limit = remaining_limit - entity_limit - relation_limit

                sampled_entities = sorted(entities, key=lambda x: x["weight"], reverse=True)[:entity_limit]
                sampled_relations = sorted(relations, key=lambda x: x["weight"], reverse=True)[:relation_limit]
                sampled_chunks = sorted(chunks, key=lambda x: x["weight"], reverse=True)[:chunk_limit]

                sampled_nodes = included_orphans + sampled_entities + sampled_relations + sampled_chunks

            logger.info(f"Sampled {len(sampled_nodes)} nodes (including {len(included_orphans)} orphans)")

        # Get node IDs for edge filtering
        sampled_node_ids = {node["id"] for node in sampled_nodes}

        # Load chunk metadata only if we're including chunks (memory optimization)
        chunks_data = {}
        if any(n["type"] == "chunk" for n in sampled_nodes) and os.path.exists(chunks_file):
            logger.info(f"Loading chunk descriptions from {chunks_file}")
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

        # Load descriptions only for sampled nodes (performance optimization)
        for node in sampled_nodes:
            node_attrs = G.nodes[node["id"]]

            if node["type"] == "entity":
                # For entity nodes: use d5 (description) field
                node["description"] = node_attrs.get("description", "")
                node["entityType"] = node_attrs.get("entity_type", "unknown")

            elif node["type"] == "relation":
                # For relation nodes: use d1 (content) field as main data
                content = node_attrs.get("content", "")
                if not content:
                    # Fallback: extract from node_id (legacy support)
                    content = _extract_relation_text(node["id"])
                node["content"] = content  # Full content
                node["description"] = content[:200] + "..." if len(content) > 200 else content

            elif node["type"] == "chunk":
                # For chunk nodes: try KV store first, then GraphML
                if node["id"] in chunks_data:
                    node["description"] = chunks_data[node["id"]].get("content", "")[:500]
                    node["content"] = chunks_data[node["id"]].get("content", "")
                else:
                    content = node_attrs.get("content", "")
                    node["description"] = content[:500]
                    node["content"] = content

        # Extract edges only between sampled nodes
        edges = []
        for source, target, attrs in G.edges(data=True):
            if source in sampled_node_ids and target in sampled_node_ids:
                edge = {
                    "id": f"{source}_{target}",
                    "source": source,
                    "target": target,
                    "label": attrs.get("label", ""),
                    "weight": float(attrs.get("weight", 1.0)),
                    "type": attrs.get("type", "")
                }
                edges.append(edge)

        # Calculate full stats (unsampled)
        stats = {
            "totalNodes": total_nodes,
            "totalEdges": total_edges,
            "entities": entity_count,
            "relations": relation_count,
            "chunks": chunk_count,
            "documents": chunk_count,
            "orphanNodes": orphan_count,  # NEW: nodes with no connections
        }

        # Sampling info
        sampling_info = {
            "sampling_applied": sampling_applied,
            "strategy": sample_strategy if sampling_applied else None,
            "requested_limit": limit,
            "nodes_returned": len(sampled_nodes),
            "edges_returned": len(edges),
            "filters_applied": {
                "node_types": node_types,
                "min_weight": min_weight
            }
        }

        # ✅ NEW: Orphan breakdown statistics
        orphan_breakdown = {
            "total": len(orphan_nodes),
            "entities": len(orphan_entities),
            "relations": len(orphan_relations),
            "chunks": len(orphan_chunks),
            "included_in_response": len([n for n in sampled_nodes if n["connections"] == 0]),
            "include_all_orphans_mode": include_all_orphans
        }

        logger.info(f"Graph export complete: {len(sampled_nodes)} nodes, {len(edges)} edges (sampled: {sampling_applied})")
        logger.info(f"Orphan breakdown: {orphan_breakdown['total']} total ({orphan_breakdown['entities']} entities, "
                   f"{orphan_breakdown['relations']} relations, {orphan_breakdown['chunks']} chunks)")

        return {
            "success": True,
            "dataset": data_source,
            "nodes": sampled_nodes,
            "edges": edges,
            "stats": stats,
            "sampling_info": sampling_info,
            "orphan_breakdown": orphan_breakdown
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export graph: {str(e)}")


async def get_node_neighbors(node_id: str, depth: int = 1, working_dir: str = None, data_source: Optional[str] = None):
    """
    Get the subgraph containing a node and its neighbors.

    Parameters:
    - node_id: ID of the central node
    - depth: Number of hops to traverse (default: 1)
    - working_dir: Working directory (not used, kept for compatibility)
    - data_source: Dataset name

    Returns:
    - nodes: List of nodes in the subgraph
    - edges: List of edges connecting the nodes
    """
    try:
        if not data_source:
            raise HTTPException(status_code=400, detail="data_source parameter is required")

        # Get graph file path
        working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
        graph_file = str(PROJECT_ROOT / working_dir_base / data_source / "graph_chunk_entity_relation.graphml")

        if not os.path.exists(graph_file):
            raise HTTPException(status_code=404, detail=f"Graph file not found for dataset '{data_source}'")

        # Read the graph
        G = nx.read_graphml(graph_file)

        if node_id not in G:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in graph")

        # Get neighbors up to specified depth
        subgraph_nodes = {node_id}
        current_layer = {node_id}

        for _ in range(depth):
            next_layer = set()
            for node in current_layer:
                neighbors = set(G.neighbors(node))
                next_layer.update(neighbors)
            subgraph_nodes.update(next_layer)
            current_layer = next_layer

        # Create subgraph
        H = G.subgraph(subgraph_nodes)

        # Extract nodes and edges
        nodes = []
        for node, attrs in H.nodes(data=True):
            role = attrs.get("role", "")
            node_type = "entity" if role == "entity" else "relation" if role == "relation" else "chunk"
            connections = G.degree(node)  # Use full graph for accurate degree count

            # Prepare node data based on type
            node_data = {
                "id": node,
                "type": node_type,
                "weight": float(attrs.get("weight", 0.0)),
                "connections": connections,
            }

            if node_type == "entity":
                # Entity nodes: use d5 (description)
                label = attrs.get("name", node)
                if label.startswith('"') and label.endswith('"'):
                    label = label[1:-1]
                node_data.update({
                    "label": label,
                    "description": attrs.get("description", ""),
                    "entityType": attrs.get("entity_type", "unknown"),
                })
            elif node_type == "relation":
                # Relation nodes: use d1 (content)
                content = attrs.get("content", "")
                if not content:
                    content = _extract_relation_text(node)
                label = content[:50] + "..." if len(content) > 50 else content
                node_data.update({
                    "label": label,
                    "content": content,
                    "description": content[:200] + "..." if len(content) > 200 else content,
                })
            else:
                # Chunk nodes
                node_data.update({
                    "label": attrs.get("name", node),
                    "description": attrs.get("content", "")[:500],
                    "content": attrs.get("content", ""),
                })

            nodes.append(node_data)

        edges = []
        for source, target, attrs in H.edges(data=True):
            edges.append({
                "id": f"{source}_{target}",
                "source": source,
                "target": target,
                "label": attrs.get("label", ""),
                "weight": float(attrs.get("weight", 1.0)),
            })

        return {
            "success": True,
            "central_node": node_id,
            "depth": depth,
            "nodes": nodes,
            "edges": edges
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get node neighbors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get node neighbors: {str(e)}")


async def search_graph_nodes(query: str, limit: int = 20, working_dir: str = None, data_source: Optional[str] = None):
    """
    Search for nodes in the graph by text query.

    Parameters:
    - query: Search query (searches node names and descriptions)
    - limit: Maximum number of results (default: 20)
    - working_dir: Working directory (not used, kept for compatibility)
    - data_source: Dataset name

    Returns:
    - nodes: List of matching nodes
    """
    try:
        # Handle empty query gracefully
        if not query or not query.strip():
            return {
                "success": True,
                "query": query,
                "results_count": 0,
                "nodes": [],
                "message": "Empty query provided"
            }

        if not data_source:
            raise HTTPException(status_code=400, detail="data_source parameter is required")

        # Get graph file path
        working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
        graph_file = str(PROJECT_ROOT / working_dir_base / data_source / "graph_chunk_entity_relation.graphml")

        if not os.path.exists(graph_file):
            raise HTTPException(status_code=404, detail=f"Graph file not found for dataset '{data_source}'")

        # Read the graph
        G = nx.read_graphml(graph_file)

        # Search nodes (simple substring match)
        query_lower = query.strip().lower()
        matching_nodes = []

        for node, attrs in G.nodes(data=True):
            role = attrs.get("role", "")
            node_type = "entity" if role == "entity" else "relation" if role == "relation" else "chunk"
            connections = G.degree(node)

            # Search in appropriate fields based on node type
            match = False
            if node_type == "entity":
                name = attrs.get("name", "").lower()
                description = attrs.get("description", "").lower()
                match = query_lower in name or query_lower in description
            elif node_type == "relation":
                # For relations: search in content (d1) field
                content = attrs.get("content", "").lower()
                if not content:
                    content = _extract_relation_text(node).lower()
                match = query_lower in content
            else:
                # For chunks
                name = attrs.get("name", "").lower()
                content = attrs.get("content", "").lower()
                match = query_lower in name or query_lower in content

            if match:
                # Build node data based on type
                node_data = {
                    "id": node,
                    "type": node_type,
                    "weight": float(attrs.get("weight", 0.0)),
                    "connections": connections,
                }

                if node_type == "entity":
                    label = attrs.get("name", node)
                    if label.startswith('"') and label.endswith('"'):
                        label = label[1:-1]
                    node_data.update({
                        "label": label,
                        "description": attrs.get("description", ""),
                        "entityType": attrs.get("entity_type", "unknown"),
                    })
                elif node_type == "relation":
                    content = attrs.get("content", "")
                    if not content:
                        content = _extract_relation_text(node)
                    label = content[:50] + "..." if len(content) > 50 else content
                    node_data.update({
                        "label": label,
                        "content": content,
                        "description": content[:200] + "..." if len(content) > 200 else content,
                    })
                else:
                    node_data.update({
                        "label": attrs.get("name", node),
                        "description": attrs.get("content", "")[:500],
                        "content": attrs.get("content", ""),
                    })

                matching_nodes.append(node_data)

                if len(matching_nodes) >= limit:
                    break

        return {
            "success": True,
            "query": query,
            "results_count": len(matching_nodes),
            "nodes": matching_nodes
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search nodes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search nodes: {str(e)}")
