import { useCallback, useEffect, useState } from 'react';
import useGraphStore from '../stores/graph';
import { getGraphData, getSubgraph, exportGraph, type GraphLoadOptions } from '../services/graph';
import type { GraphExportOptions } from '../types';
import { toast } from 'sonner';

export const useGraph = () => {
  const {
    nodes,
    edges,
    selectedNode,
    layout,
    filters,
    isLoading,
    error,
    searchQuery,
    stats,
    setNodes,
    setEdges,
    selectNode,
    setLayout,
    updateFilters,
    setSearchQuery,
    setLoading,
    setError,
    setStats,
    clearGraph,
    getFilteredNodes,
    getFilteredEdges,
  } = useGraphStore();

  const [cytoscapeInstance, setCytoscapeInstance] = useState<any>(null);

  // Load full graph data with optional sampling/filtering
  const loadGraph = useCallback(
    async (dataSource: string, options: GraphLoadOptions = {}) => {
      setLoading(true);
      setError(null);

      console.log('[useGraph] Loading graph:', dataSource, 'with options:', options);

      try {
        const data = await getGraphData(dataSource, options);

        if (data.nodes && data.edges) {
          console.log(`[useGraph] Loaded ${data.nodes.length} nodes, ${data.edges.length} edges`);
          setNodes(data.nodes);
          setEdges(data.edges);
          setStats(data.stats || null);

          // Don't show success toast if sampling notification was already shown
          if (!data.samplingInfo?.sampling_applied) {
            toast.success(`Graph loaded: ${data.nodes.length} nodes`);
          }
        } else {
          throw new Error('Invalid graph data format');
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load graph';
        console.error('[useGraph] Error loading graph:', err);
        setError(message);
        toast.error(message);
      } finally {
        setLoading(false);
      }
    },
    [setNodes, setEdges, setStats, setLoading, setError]
  );

  // Load subgraph for a query
  const loadSubgraph = useCallback(
    async (query: string, topK = 10) => {
      setLoading(true);
      setError(null);

      try {
        const data = await getSubgraph(query, topK);

        if (data.nodes && data.edges) {
          setNodes(data.nodes);
          setEdges(data.edges);
          toast.success(`Loaded subgraph for "${query}"`);
        } else {
          throw new Error('Invalid subgraph data');
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load subgraph';
        setError(message);
        toast.error(message);
      } finally {
        setLoading(false);
      }
    },
    [setNodes, setEdges, setLoading, setError]
  );

  // Export graph in different formats
  const handleExportGraph = useCallback(
    async (options: GraphExportOptions) => {
      if (!cytoscapeInstance) {
        toast.error('Graph not initialized');
        return;
      }

      try {
        setLoading(true);

        if (options.format === 'png') {
          // Export as PNG using Cytoscape
          const png = cytoscapeInstance.png({
            quality: options.quality || 1,
            bg: options.background || '#ffffff',
          });

          // Download PNG
          const link = document.createElement('a');
          link.href = png;
          link.download = `graph-export-${Date.now()}.png`;
          link.click();

          toast.success('Graph exported as PNG');
        } else if (options.format === 'json') {
          // Export as JSON
          const data = {
            nodes: getFilteredNodes(),
            edges: getFilteredEdges(),
            layout,
            filters,
          };

          const json = JSON.stringify(data, null, 2);
          const blob = new Blob([json], { type: 'application/json' });
          const url = URL.createObjectURL(blob);

          const link = document.createElement('a');
          link.href = url;
          link.download = `graph-export-${Date.now()}.json`;
          link.click();

          URL.revokeObjectURL(url);
          toast.success('Graph exported as JSON');
        } else if (options.format === 'graphml') {
          // Export as GraphML - call backend API
          await exportGraph(options.format);
          toast.success('Graph exported as GraphML');
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to export graph';
        toast.error(message);
      } finally {
        setLoading(false);
      }
    },
    [cytoscapeInstance, getFilteredNodes, getFilteredEdges, layout, filters, setLoading]
  );

  // Fit graph to viewport
  const fitGraph = useCallback(() => {
    if (cytoscapeInstance) {
      cytoscapeInstance.fit();
    }
  }, [cytoscapeInstance]);

  // Center on selected node
  const centerOnNode = useCallback(
    (nodeId: string) => {
      if (cytoscapeInstance) {
        const node = cytoscapeInstance.$(`#${nodeId}`);
        if (node.length > 0) {
          cytoscapeInstance.animate({
            center: { eles: node },
            zoom: 2,
            duration: 500,
          });
        }
      }
    },
    [cytoscapeInstance]
  );

  // Expand node neighbors
  const expandNode = useCallback(
    async (nodeId: string) => {
      if (!selectedNode) return;

      try {
        // In a real implementation, this would fetch additional neighbors from the backend
        toast.info('Expanding node neighbors...');
        // await loadNodeNeighbors(nodeId);
      } catch (err) {
        toast.error('Failed to expand node');
      }
    },
    [selectedNode]
  );

  // Search for nodes
  const searchNodes = useCallback(
    (query: string) => {
      setSearchQuery(query);

      if (cytoscapeInstance && query) {
        // Highlight matching nodes
        const matching = cytoscapeInstance.nodes().filter((node: any) => {
          const label = node.data('label') || '';
          const description = node.data('description') || '';
          return (
            label.toLowerCase().includes(query.toLowerCase()) ||
            description.toLowerCase().includes(query.toLowerCase())
          );
        });

        // Reset all nodes first
        cytoscapeInstance.nodes().removeClass('highlighted');

        // Highlight matching nodes
        matching.addClass('highlighted');

        // Center on first match if any
        if (matching.length > 0) {
          cytoscapeInstance.animate({
            center: { eles: matching[0] },
            zoom: 2,
            duration: 500,
          });
        }
      }
    },
    [cytoscapeInstance, setSearchQuery]
  );

  // Apply layout
  const applyLayout = useCallback(
    (layoutName?: string) => {
      if (!cytoscapeInstance) return;

      const layoutToApply = layoutName || layout;

      const layoutOptions: Record<string, any> = {
        'cose-bilkent': {
          name: 'cose-bilkent',
          idealEdgeLength: 100,
          nodeRepulsion: 400000,
          edgeElasticity: 0.45,
          nestingFactor: 0.1,
          gravity: 0.25,
          numIter: 2500,
          tile: true,
          animate: true,
          animationDuration: 1000,
        },
        dagre: {
          name: 'dagre',
          rankDir: 'TB',
          nodeSep: 50,
          edgeSep: 10,
          rankSep: 100,
          animate: true,
          animationDuration: 1000,
        },
        fcose: {
          name: 'fcose',
          idealEdgeLength: 100,
          nodeRepulsion: 4500,
          edgeElasticity: 0.45,
          numIter: 2500,
          tile: true,
          animate: true,
          animationDuration: 1000,
        },
        grid: {
          name: 'grid',
          rows: undefined,
          cols: undefined,
          position: undefined,
          animate: true,
          animationDuration: 500,
        },
        circle: {
          name: 'circle',
          animate: true,
          animationDuration: 500,
        },
        concentric: {
          name: 'concentric',
          minNodeSpacing: 40,
          animate: true,
          animationDuration: 500,
        },
        breadthfirst: {
          name: 'breadthfirst',
          directed: true,
          spacingFactor: 1.5,
          animate: true,
          animationDuration: 500,
        },
      };

      const layoutConfig = layoutOptions[layoutToApply] || layoutOptions.grid;

      try {
        const layout = cytoscapeInstance.layout(layoutConfig);
        layout.run();

        if (layoutName) {
          setLayout(layoutName as any);
        }
      } catch (err) {
        console.error('Failed to apply layout:', err);
        toast.error('Failed to apply layout');
      }
    },
    [cytoscapeInstance, layout, setLayout]
  );

  // Initialize Cytoscape instance
  const initializeCytoscape = useCallback((cy: any) => {
    setCytoscapeInstance(cy);

    // Set up event handlers
    cy.on('tap', 'node', (evt: any) => {
      const node = evt.target;
      selectNode({
        data: node.data(),
        position: node.position(),
      });
    });

    cy.on('tap', (evt: any) => {
      if (evt.target === cy) {
        selectNode(null);
      }
    });

    cy.on('mouseover', 'node', (evt: any) => {
      evt.target.addClass('hover');
    });

    cy.on('mouseout', 'node', (evt: any) => {
      evt.target.removeClass('hover');
    });
  }, [selectNode]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (cytoscapeInstance) {
        cytoscapeInstance.destroy();
      }
    };
  }, [cytoscapeInstance]);

  return {
    // State
    nodes: getFilteredNodes(),
    edges: getFilteredEdges(),
    selectedNode,
    layout,
    filters,
    isLoading,
    error,
    searchQuery,
    stats,

    // Actions
    loadGraph,
    loadSubgraph,
    selectNode,
    setLayout,
    updateFilters,
    searchNodes,
    clearGraph,
    exportGraph: handleExportGraph,

    // Cytoscape operations
    initializeCytoscape,
    fitGraph,
    centerOnNode,
    expandNode,
    applyLayout,
  };
};