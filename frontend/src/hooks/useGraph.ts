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

  // ✅ PHASE 2: Expand node neighbors - fetch and merge into existing graph
  const expandNode = useCallback(
    async (nodeId: string, depth: number = 1) => {
      if (!cytoscapeInstance) {
        toast.error('Graph not initialized');
        return;
      }

      try {
        setLoading(true);
        toast.info('Fetching neighbors...');

        // Import the getNodeNeighbors function from graph service
        const { getNodeNeighbors } = await import('../services/graph');

        const neighborData = await getNodeNeighbors(nodeId, depth);

        if (!neighborData.nodes || neighborData.nodes.length === 0) {
          toast.warning('No new neighbors found');
          return;
        }

        // Get existing node IDs to avoid duplicates
        const existingNodeIds = new Set(nodes.map(n => n.data.id));
        const existingEdgeIds = new Set(edges.map(e => e.data.id));

        // Filter out nodes and edges that already exist
        const newNodes = neighborData.nodes.filter(n => !existingNodeIds.has(n.data.id));
        const newEdges = neighborData.edges.filter(e => !existingEdgeIds.has(e.data.id));

        if (newNodes.length === 0 && newEdges.length === 0) {
          toast.info('All neighbors already visible');
          return;
        }

        // Add new nodes and edges to the store
        setNodes([...nodes, ...newNodes]);
        setEdges([...edges, ...newEdges]);

        // Add new elements to Cytoscape instance
        newNodes.forEach(node => {
          cytoscapeInstance.add({ group: 'nodes', data: node.data });
        });
        newEdges.forEach(edge => {
          cytoscapeInstance.add({ group: 'edges', data: edge.data });
        });

        // Run layout on new elements
        const newElements = cytoscapeInstance.$(
          newNodes.map(n => `#${n.data.id}`).join(', ')
        );

        // Apply incremental layout
        const layoutOptions = {
          name: 'cose-bilkent',
          animate: true,
          animationDuration: 500,
          fit: false, // Don't fit entire graph, just position new nodes
          randomize: false,
        };

        cytoscapeInstance.layout(layoutOptions as any).run();

        // Highlight the expanded node and its new neighbors
        cytoscapeInstance.$(`#${nodeId}`).addClass('highlighted');
        newElements.addClass('highlighted');

        // Remove highlight after 2 seconds
        setTimeout(() => {
          cytoscapeInstance.$('.highlighted').removeClass('highlighted');
        }, 2000);

        toast.success(`Added ${newNodes.length} nodes, ${newEdges.length} edges`);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to expand node';
        console.error('[useGraph] Error expanding node:', err);
        toast.error(message);
      } finally {
        setLoading(false);
      }
    },
    [cytoscapeInstance, nodes, edges, setNodes, setEdges, setLoading]
  );

  // ✅ PHASE 2: Enhanced search with better highlighting and navigation
  const searchNodes = useCallback(
    (query: string) => {
      setSearchQuery(query);

      if (!cytoscapeInstance) return;

      if (!query || query.trim().length === 0) {
        // Clear all highlights if search is empty
        cytoscapeInstance.nodes().removeClass('highlighted');
        cytoscapeInstance.edges().removeClass('highlighted');
        return;
      }

      try {
        const searchTerm = query.toLowerCase().trim();

        // Highlight matching nodes
        const matching = cytoscapeInstance.nodes().filter((node: any) => {
          const label = node.data('label') || '';
          const description = node.data('description') || '';
          const type = node.data('type') || '';
          const sourceId = node.data('source_id') || '';

          return (
            label.toLowerCase().includes(searchTerm) ||
            description.toLowerCase().includes(searchTerm) ||
            type.toLowerCase().includes(searchTerm) ||
            sourceId.toLowerCase().includes(searchTerm)
          );
        });

        // Reset all highlights first
        cytoscapeInstance.nodes().removeClass('highlighted');
        cytoscapeInstance.edges().removeClass('highlighted');

        if (matching.length === 0) {
          toast.warning('No nodes found matching your search');
          return;
        }

        // Highlight matching nodes
        matching.addClass('highlighted');

        // Also highlight edges connected to matching nodes
        const connectedEdges = matching.connectedEdges();
        connectedEdges.addClass('highlighted');

        // Dim non-matching nodes for better focus
        cytoscapeInstance.nodes().not(matching).style('opacity', 0.3);
        cytoscapeInstance.edges().not(connectedEdges).style('opacity', 0.2);

        // Show bounding box around all matching nodes
        const bbox = matching.boundingBox();
        cytoscapeInstance.animate({
          fit: {
            eles: matching,
            padding: 50,
          },
          duration: 500,
        });

        // Show toast with search results
        toast.success(`Found ${matching.length} matching node${matching.length !== 1 ? 's' : ''}`);

        // Store matched nodes for navigation (next/previous)
        (cytoscapeInstance as any)._searchResults = matching;
        (cytoscapeInstance as any)._searchIndex = 0;

      } catch (err) {
        console.error('[useGraph] Error searching nodes:', err);
        toast.error('Search failed');
      }
    },
    [cytoscapeInstance, setSearchQuery]
  );

  // ✅ PHASE 2: Navigate through search results
  const navigateSearchResults = useCallback(
    (direction: 'next' | 'prev') => {
      if (!cytoscapeInstance) return;

      const results = (cytoscapeInstance as any)._searchResults;
      let currentIndex = (cytoscapeInstance as any)._searchIndex || 0;

      if (!results || results.length === 0) {
        toast.warning('No search results to navigate');
        return;
      }

      // Update index
      if (direction === 'next') {
        currentIndex = (currentIndex + 1) % results.length;
      } else {
        currentIndex = (currentIndex - 1 + results.length) % results.length;
      }

      (cytoscapeInstance as any)._searchIndex = currentIndex;

      // Center on current result
      const currentNode = results[currentIndex];
      cytoscapeInstance.animate({
        center: { eles: currentNode },
        zoom: Math.max(cytoscapeInstance.zoom(), 1.5),
        duration: 300,
      });

      // Pulse animation on current node
      currentNode.addClass('pulse');
      setTimeout(() => {
        currentNode.removeClass('pulse');
      }, 1000);

      toast.info(`Result ${currentIndex + 1} of ${results.length}`);
    },
    [cytoscapeInstance]
  );

  // Clear search highlights
  const clearSearch = useCallback(() => {
    if (!cytoscapeInstance) return;

    cytoscapeInstance.nodes().removeClass('highlighted');
    cytoscapeInstance.edges().removeClass('highlighted');
    cytoscapeInstance.nodes().style('opacity', 1);
    cytoscapeInstance.edges().style('opacity', 1);

    (cytoscapeInstance as any)._searchResults = null;
    (cytoscapeInstance as any)._searchIndex = 0;

    setSearchQuery('');
  }, [cytoscapeInstance, setSearchQuery]);

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

    // ✅ PHASE 2: New search navigation functions
    navigateSearchResults,
    clearSearch,
  };
};