import api from './api';
import type { CytoscapeNode, CytoscapeEdge, GraphStats } from '../types';
import { API_ENDPOINTS } from '../utils/constants';

interface GraphData {
  nodes: CytoscapeNode[];
  edges: CytoscapeEdge[];
  stats?: GraphStats;
}

/**
 * Get the full graph data for a dataset
 */
export const getGraphData = async (dataSource: string): Promise<GraphData> => {
  const response = await api.get(API_ENDPOINTS.GRAPH_EXPORT, {
    params: { data_source: dataSource },
  });

  // Transform the data to Cytoscape format
  const graphData = response.data;

  const nodes: CytoscapeNode[] = (graphData.nodes || []).map((node: any) => ({
    data: {
      id: node.id,
      label: node.label || node.name || node.id,
      type: node.type || 'entity',
      description: node.description,
      weight: node.weight,
      source_id: node.source_id,
      metadata: node.metadata,
    },
    position: node.position,
  }));

  const edges: CytoscapeEdge[] = (graphData.edges || []).map((edge: any) => ({
    data: {
      id: edge.id || `${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      weight: edge.weight,
      type: edge.type,
    },
  }));

  // Calculate stats if not provided
  const stats: GraphStats = graphData.stats || {
    totalNodes: nodes.length,
    totalEdges: edges.length,
    entities: nodes.filter(n => n.data.type === 'entity').length,
    relations: nodes.filter(n => n.data.type === 'relation').length,
    chunks: nodes.filter(n => n.data.type === 'chunk').length,
    documents: nodes.filter(n => n.data.type === 'document').length,
  };

  return { nodes, edges, stats };
};

/**
 * Get a subgraph for a specific query
 */
export const getSubgraph = async (
  query: string,
  topK: number = 10
): Promise<GraphData> => {
  const response = await api.post(API_ENDPOINTS.SEARCH, {
    queries: [query],
    param: {
      top_k: topK,
      mode: 'hybrid',
      enable_reranking: true,
    },
  });

  // Extract nodes and edges from the search results
  // This is a simplified version - actual implementation would depend on backend response format
  const results = response.data.results?.[0] || {};

  const nodes: CytoscapeNode[] = [];
  const edges: CytoscapeEdge[] = [];

  // Convert search results to graph format
  if (results.graph) {
    // If backend provides graph structure
    const graphData = results.graph;

    (graphData.nodes || []).forEach((node: any) => {
      nodes.push({
        data: {
          id: node.id,
          label: node.label || node.name,
          type: node.type,
          description: node.description,
          weight: node.weight,
          source_id: node.source_id,
        },
      });
    });

    (graphData.edges || []).forEach((edge: any) => {
      edges.push({
        data: {
          id: `${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          weight: edge.weight,
        },
      });
    });
  } else if (results.contexts) {
    // If backend only provides contexts, create a simple graph
    results.contexts.forEach((context: string, index: number) => {
      nodes.push({
        data: {
          id: `context-${index}`,
          label: context.substring(0, 50) + '...',
          type: 'chunk',
          description: context,
        },
      });
    });
  }

  return { nodes, edges };
};

/**
 * Export graph in different formats
 */
export const exportGraph = async (format: 'graphml' | 'json' | 'csv'): Promise<Blob> => {
  const response = await api.get(`${API_ENDPOINTS.GRAPH_EXPORT}.${format}`, {
    responseType: 'blob',
  });

  return response.data;
};

/**
 * Get node neighbors
 */
export const getNodeNeighbors = async (
  nodeId: string,
  depth: number = 1
): Promise<GraphData> => {
  const response = await api.get(`${API_ENDPOINTS.GRAPH_SUBGRAPH}/neighbors`, {
    params: {
      node_id: nodeId,
      depth,
    },
  });

  const nodes: CytoscapeNode[] = (response.data.nodes || []).map((node: any) => ({
    data: {
      id: node.id,
      label: node.label,
      type: node.type,
      description: node.description,
      weight: node.weight,
    },
  }));

  const edges: CytoscapeEdge[] = (response.data.edges || []).map((edge: any) => ({
    data: {
      id: `${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      weight: edge.weight,
    },
  }));

  return { nodes, edges };
};

/**
 * Search nodes by text query
 */
export const searchNodes = async (
  query: string,
  limit: number = 20
): Promise<CytoscapeNode[]> => {
  const response = await api.get(`${API_ENDPOINTS.GRAPH_SUBGRAPH}/search`, {
    params: {
      q: query,
      limit,
    },
  });

  return (response.data.nodes || []).map((node: any) => ({
    data: {
      id: node.id,
      label: node.label,
      type: node.type,
      description: node.description,
      weight: node.weight,
    },
  }));
};