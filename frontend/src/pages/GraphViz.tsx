/**
 * Graph Visualization Page
 *
 * Interactive bipartite graph visualization using Cytoscape.js:
 * - Entity nodes (blue)
 * - Relation nodes (red)
 * - Document chunk nodes (green)
 * - Interactive exploration (zoom, pan, click)
 * - Layout controls
 */

import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { Loader2 } from 'lucide-react';
import type { Core } from 'cytoscape';
import { useGraph } from '../hooks/useGraph';
import GraphCanvas from '../components/graph/GraphCanvas';
import GraphToolbar from '../components/graph/GraphToolbar';
import NodeInfoPanel from '../components/graph/NodeInfoPanel';
import GraphErrorBoundary from '../components/graph/GraphErrorBoundary'; // ✅ PHASE 2
import GraphTooltip from '../components/graph/GraphTooltip'; // ✅ PHASE 4.2
import type { GraphLayout, CytoscapeNode } from '../types';

export function GraphViz() {
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
    canLoadMore, // ✅ PHASE 4.1
    loadGraph,
    loadMoreNodes, // ✅ PHASE 4.1
    selectNode,
    setLayout,
    updateFilters,
    searchNodes,
    exportGraph,
    initializeCytoscape,
    fitGraph,
    applyLayout,
    clearGraph, // ✅ PHASE 2: For error recovery
  } = useGraph();

  const [cyInstance, setCyInstance] = useState<Core | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  // ✅ PHASE 4.2: Tooltip state
  const [hoveredNode, setHoveredNode] = useState<CytoscapeNode | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [tooltipConnectionCount, setTooltipConnectionCount] = useState<number | undefined>();

  // Load graph on mount with performance-optimized settings
  useEffect(() => {
    // Load with sampling for large graphs (default: top 1000 nodes)
    const loadGraphData = async () => {
      try {
        await loadGraph('SingleTopic', {
          limit: 1000, // Show top 1000 nodes
          sampleStrategy: 'diverse', // Get balanced mix of entities, relations, chunks
        });
      } catch (err) {
        console.error('[GraphViz] Failed to load graph:', err);
        toast.error('Failed to load graph. Please try again.');
      }
    };

    loadGraphData();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ✅ PHASE 2: Handle error boundary reset
  const handleErrorReset = () => {
    // Clear graph state and reload
    clearGraph();
    loadGraph('SingleTopic', {
      limit: 1000,
      sampleStrategy: 'diverse',
    });
  };

  // Handle Cytoscape initialization
  const handleCyReady = (cy: Core) => {
    setCyInstance(cy);
    initializeCytoscape(cy);

    // ✅ PHASE 4.2: Add tooltip event listeners
    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      const renderedPosition = node.renderedPosition();

      // Get connection count
      const degree = node.degree();

      setHoveredNode({
        data: node.data(),
        position: node.position(),
      });
      setTooltipPosition({
        x: renderedPosition.x,
        y: renderedPosition.y,
      });
      setTooltipConnectionCount(degree);
    });

    cy.on('mouseout', 'node', () => {
      setHoveredNode(null);
    });

    // Hide tooltip when panning/zooming
    cy.on('viewport', () => {
      setHoveredNode(null);
    });
  };

  // Handle zoom controls
  const handleZoomIn = () => {
    if (cyInstance) {
      const currentZoom = cyInstance.zoom();
      cyInstance.zoom({
        level: currentZoom * 1.2,
        renderedPosition: { x: cyInstance.width() / 2, y: cyInstance.height() / 2 },
      });
    }
  };

  const handleZoomOut = () => {
    if (cyInstance) {
      const currentZoom = cyInstance.zoom();
      cyInstance.zoom({
        level: currentZoom * 0.8,
        renderedPosition: { x: cyInstance.width() / 2, y: cyInstance.height() / 2 },
      });
    }
  };

  // Handle refresh
  const handleRefresh = () => {
    if (cyInstance) {
      applyLayout(layout);
      fitGraph();
    }
  };

  // Handle layout change
  const handleLayoutChange = (newLayout: GraphLayout) => {
    setLayout(newLayout);
    if (cyInstance) {
      applyLayout(newLayout);
    }
  };

  // Handle export
  const handleExport = async (format: 'png' | 'json' | 'graphml') => {
    await exportGraph({ format });
  };

  // Handle help modal
  const handleHelp = () => {
    setShowHelp(true);
  };

  // Handle view document
  const handleViewDocument = (docId: string) => {
    // Navigate to documents page with the specific document
    window.location.href = `/documents?id=${docId}`;
  };

  // Handle find similar
  const handleFindSimilar = async (nodeId: string) => {
    toast.info('Finding similar nodes...');
    // This would typically make an API call to find similar nodes
    // For now, just search for the node label
    const node = nodes.find((n) => n.data.id === nodeId);
    if (node) {
      searchNodes(node.data.label);
    }
  };

  // Handle expand node
  const handleExpandNode = async (_nodeId: string) => {
    toast.info('Expanding node neighbors...');
    // This would typically fetch additional neighbors from the backend
    // For now, just a placeholder
  };

  return (
    <GraphErrorBoundary onReset={handleErrorReset}>
      <div className="h-[calc(100vh-8rem)] flex flex-col relative">
      {/* Toolbar */}
      <GraphToolbar
        layout={layout}
        filters={filters}
        searchQuery={searchQuery}
        onLayoutChange={handleLayoutChange}
        onFiltersChange={updateFilters}
        onSearchChange={searchNodes}
        onExport={handleExport}
        onFit={fitGraph}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onRefresh={handleRefresh}
        onHelp={handleHelp}
      />

      {/* Main Content Area */}
      <div className="flex-1 relative">
        {/* ✨ IMPROVED: Loading State with better design */}
        {isLoading && (
          <div className="absolute inset-0 bg-gradient-to-br from-white/95 to-gray-50/95 dark:from-gray-900/95 dark:to-gray-800/95 backdrop-blur-sm flex items-center justify-center z-20">
            <div className="flex flex-col items-center gap-4 bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700">
              <div className="relative">
                <Loader2 className="w-12 h-12 animate-spin text-blue-500" />
                <div className="absolute inset-0 w-12 h-12 animate-ping text-blue-300 opacity-20">
                  <Loader2 className="w-full h-full" />
                </div>
              </div>
              <div className="text-center space-y-1">
                <p className="text-base font-semibold text-gray-900 dark:text-gray-100">Loading Graph</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Building visualization...</p>
              </div>
            </div>
          </div>
        )}

        {/* ✨ IMPROVED: Error State with better design */}
        {error && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 animate-in slide-in-from-top duration-300">
            <div className="bg-red-100 dark:bg-red-900/70 backdrop-blur-sm text-red-800 dark:text-red-200 px-6 py-3 rounded-xl shadow-lg border border-red-300 dark:border-red-700 flex items-center gap-3 max-w-md">
              <div className="flex-shrink-0">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              </div>
              <p className="text-sm font-medium">{error}</p>
            </div>
          </div>
        )}

        {/* Graph Canvas */}
        <GraphCanvas
          nodes={nodes}
          edges={edges}
          onNodeSelect={selectNode}
          onReady={handleCyReady}
          layout={layout}
          className="w-full h-full"
        />

        {/* ✅ PHASE 4.2: Tooltip on hover */}
        <GraphTooltip
          node={hoveredNode}
          position={tooltipPosition}
          connectionCount={tooltipConnectionCount}
        />

        {/* Node Info Panel */}
        {selectedNode && (
          <NodeInfoPanel
            node={selectedNode}
            onClose={() => selectNode(null)}
            onViewDocument={handleViewDocument}
            onFindSimilar={handleFindSimilar}
            onExpandNode={handleExpandNode}
            cytoscapeInstance={cyInstance} // ✅ PHASE 2: Pass Cytoscape instance for connection stats
          />
        )}

        {/* ✨ IMPROVED: Stats Badge with better design */}
        {stats && (
          <div className="absolute bottom-4 left-4 space-y-2">
            <div className="bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 p-4 text-sm">
              <div className="flex items-center gap-2 mb-3 pb-2 border-b border-gray-200 dark:border-gray-700">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <h4 className="font-semibold text-gray-900 dark:text-gray-100">Graph Stats</h4>
              </div>
              <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 dark:text-gray-400 text-xs">Nodes:</span>
                  <span className="font-bold text-gray-900 dark:text-gray-100">{stats.totalNodes.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 dark:text-gray-400 text-xs">Edges:</span>
                  <span className="font-bold text-gray-900 dark:text-gray-100">{stats.totalEdges.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span className="font-medium text-gray-900 dark:text-gray-100">{stats.entities.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 transform rotate-45"></div>
                  <span className="font-medium text-gray-900 dark:text-gray-100">{stats.relations.toLocaleString()}</span>
                </div>
                {stats.chunks > 0 && (
                  <div className="flex items-center gap-2 col-span-2">
                    <div className="w-3 h-3 bg-green-500 rounded-sm"></div>
                    <span className="text-gray-500 dark:text-gray-400 text-xs">Chunks:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">{stats.chunks.toLocaleString()}</span>
                  </div>
                )}
              </div>
            </div>

            {/* ✅ PHASE 4.1: Load More Button */}
            {canLoadMore && !isLoading && (
              <button
                onClick={() => loadMoreNodes()}
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow-lg transition-colors flex items-center justify-center gap-2 font-medium"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Load More Nodes
              </button>
            )}
          </div>
        )}

        {/* ✨ IMPROVED: Legend with better design */}
        <div className="absolute bottom-4 right-4 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 p-4 text-sm">
          <div className="flex items-center gap-2 mb-3 pb-2 border-b border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-gray-100">Node Types</h4>
          </div>
          <div className="space-y-2.5">
            <div className="flex items-center gap-3 group hover:bg-blue-50 dark:hover:bg-blue-900/20 px-2 py-1 rounded transition-colors">
              <div className="w-5 h-5 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full border-2 border-blue-700 shadow-sm"></div>
              <span className="text-gray-700 dark:text-gray-300 font-medium">Entity</span>
            </div>
            <div className="flex items-center gap-3 group hover:bg-red-50 dark:hover:bg-red-900/20 px-2 py-1 rounded transition-colors">
              <div className="w-5 h-5 bg-gradient-to-br from-red-500 to-red-600 border-2 border-red-700 shadow-sm transform rotate-45"></div>
              <span className="text-gray-700 dark:text-gray-300 font-medium">Relation</span>
            </div>
            <div className="flex items-center gap-3 group hover:bg-green-50 dark:hover:bg-green-900/20 px-2 py-1 rounded transition-colors">
              <div className="w-5 h-5 bg-gradient-to-br from-green-500 to-green-600 rounded border-2 border-green-700 shadow-sm"></div>
              <span className="text-gray-700 dark:text-gray-300 font-medium">Chunk</span>
            </div>
            <div className="flex items-center gap-3 group hover:bg-purple-50 dark:hover:bg-purple-900/20 px-2 py-1 rounded transition-colors">
              <div className="w-5 h-5 bg-gradient-to-br from-purple-500 to-purple-600 rounded-md border-2 border-purple-700 shadow-sm"></div>
              <span className="text-gray-700 dark:text-gray-300 font-medium">Document</span>
            </div>
          </div>
        </div>
      </div>

      {/* Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md">
            <h3 className="text-lg font-semibold mb-4">Graph Visualization Help</h3>
            <div className="space-y-2 text-sm">
              <p>
                <strong>Navigation:</strong>
              </p>
              <ul className="list-disc list-inside ml-2 space-y-1">
                <li>Click and drag to pan</li>
                <li>Scroll to zoom</li>
                <li>Click nodes to view details</li>
                <li>Double-click to expand neighbors</li>
              </ul>
              <p className="mt-3">
                <strong>Keyboard Shortcuts:</strong>
              </p>
              <ul className="list-disc list-inside ml-2 space-y-1">
                <li>
                  <kbd>Ctrl+F</kbd>: Focus search
                </li>
                <li>
                  <kbd>Ctrl+Z</kbd>: Undo layout
                </li>
                <li>
                  <kbd>Ctrl+R</kbd>: Reset zoom
                </li>
                <li>
                  <kbd>Space</kbd>: Fit graph to view
                </li>
                <li>
                  <kbd>Delete</kbd>: Remove selected node
                </li>
              </ul>
            </div>
            <button
              onClick={() => setShowHelp(false)}
              className="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
      </div>
    </GraphErrorBoundary>
  );
}
