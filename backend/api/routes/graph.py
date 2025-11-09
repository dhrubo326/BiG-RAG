"""
Graph management routes
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from bigrag.utils import logger

from ..core.dependencies import get_working_dir, get_data_source
from ..models.models import GraphStatsResponse
from ..services.graph_stats import get_graph_statistics
from ..services.graph_export import (
    export_graph_for_cytoscape,
    get_node_neighbors,
    search_graph_nodes
)


router = APIRouter(prefix="/graph", tags=["Graph Management"])


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(dataset: Optional[str] = None):
    """
    Get knowledge graph statistics.

    **Returns:**
    - Total entities, relations, chunks
    - Graph density metrics
    - Dataset information

    **Example usage:**
    ```bash
    curl "http://localhost:8001/graph/stats"
    curl "http://localhost:8001/graph/stats?dataset=2WikiMultiHopQA"
    ```
    """
    try:
        working_dir = get_working_dir()
        stats = await get_graph_statistics(working_dir, dataset)
        return stats

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get graph stats: {str(e)}")


@router.get("/export")
async def export_graph(
    data_source: str,
    limit: Optional[int] = 1000,
    node_types: Optional[str] = None,
    edge_types: Optional[str] = None,
    min_weight: Optional[float] = 0.0,
    sample_strategy: Optional[str] = "top_weighted"
):
    """
    Export the knowledge graph for a dataset in Cytoscape-compatible format.

    **OPTIMIZED FOR LARGE GRAPHS** - Implements sampling and filtering to prevent browser freezing.

    **Parameters:**
    - data_source: Dataset name (e.g., "SingleTopic", "HotpotQA")
    - limit: Maximum nodes to return (default: 1000, max: 5000)
    - node_types: Comma-separated node types to include (e.g., "entity,relation")
    - edge_types: Comma-separated edge types to include
    - min_weight: Minimum node weight threshold (0.0-1.0)
    - sample_strategy: "top_weighted" (highest weight), "random", "diverse" (balanced types)

    **Returns:**
    - nodes: Sampled list of graph nodes
    - edges: Edges connecting sampled nodes
    - stats: Full graph statistics (unsampled)
    - sampling_info: Information about sampling applied

    **Example usage:**
    ```bash
    # Get top 1000 nodes (default)
    curl "http://localhost:8001/graph/export?data_source=SingleTopic"

    # Get top 500 entities and relations only
    curl "http://localhost:8001/graph/export?data_source=SingleTopic&limit=500&node_types=entity,relation"

    # Get high-weight nodes (> 0.5)
    curl "http://localhost:8001/graph/export?data_source=SingleTopic&min_weight=0.5"

    # Use diverse sampling strategy
    curl "http://localhost:8001/graph/export?data_source=demo_test&limit=1000&sample_strategy=diverse"
    ```
    """
    try:
        graph_data = await export_graph_for_cytoscape(
            data_source=data_source,
            limit=limit,
            node_types=node_types.split(',') if node_types else None,
            edge_types=edge_types.split(',') if edge_types else None,
            min_weight=min_weight,
            sample_strategy=sample_strategy
        )
        return graph_data

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"Failed to export graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to export graph: {str(e)}")


@router.get("/subgraph/neighbors")
async def get_neighbors(
    node_id: str,
    depth: int = 1,
    data_source: Optional[str] = None
):
    """
    Get the subgraph containing a node and its neighbors.

    **Parameters:**
    - node_id: Node ID (required)
    - depth: Neighborhood depth (1 = immediate neighbors, 2 = neighbors of neighbors)
    - data_source: Dataset name (defaults to current)

    **Example usage:**
    ```bash
    curl "http://localhost:8001/graph/subgraph/neighbors?node_id=entity_123&depth=2"
    ```

    **Returns:** Subgraph in Cytoscape.js format
    """
    try:
        working_dir = get_working_dir()
        subgraph = await get_node_neighbors(
            node_id=node_id,
            depth=depth,
            working_dir=working_dir,
            data_source=data_source
        )
        return subgraph

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"Failed to get neighbors: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get neighbors: {str(e)}")


@router.get("/subgraph/search")
async def search_nodes(
    q: str,
    limit: int = 20,
    data_source: Optional[str] = None
):
    """
    Search for nodes in the graph by text query.

    **Parameters:**
    - q: Search query (required)
    - limit: Maximum results to return (default: 20)
    - data_source: Dataset name (defaults to current)

    **Example usage:**
    ```bash
    curl "http://localhost:8001/graph/subgraph/search?q=artificial+intelligence&limit=10"
    ```

    **Returns:** List of matching nodes with metadata
    """
    try:
        working_dir = get_working_dir()
        results = await search_graph_nodes(
            query=q,
            limit=limit,
            working_dir=working_dir,
            data_source=data_source
        )
        return results

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"Failed to search nodes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to search nodes: {str(e)}")
