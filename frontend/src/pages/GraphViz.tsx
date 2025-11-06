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
import type { GraphLayout } from '../types';

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
    loadGraph,
    selectNode,
    setLayout,
    updateFilters,
    searchNodes,
    exportGraph,
    initializeCytoscape,
    fitGraph,
    applyLayout,
  } = useGraph();

  const [cyInstance, setCyInstance] = useState<Core | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  // Load graph on mount with performance-optimized settings
  useEffect(() => {
    // Load with sampling for large graphs (default: top 1000 nodes)
    loadGraph('SingleTopic', {
      limit: 1000, // Show top 1000 nodes by weight
      sampleStrategy: 'top_weighted', // Get most important nodes
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle Cytoscape initialization
  const handleCyReady = (cy: Core) => {
    setCyInstance(cy);
    initializeCytoscape(cy);
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
        {/* Loading State */}
        {isLoading && (
          <div className="absolute inset-0 bg-white/80 dark:bg-gray-900/80 flex items-center justify-center z-20">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
              <p className="text-sm text-gray-600 dark:text-gray-400">Loading graph...</p>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-100 dark:bg-red-900/50 text-red-800 dark:text-red-200 px-4 py-2 rounded-lg z-20">
            {error}
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

        {/* Node Info Panel */}
        {selectedNode && (
          <NodeInfoPanel
            node={selectedNode}
            onClose={() => selectNode(null)}
            onViewDocument={handleViewDocument}
            onFindSimilar={handleFindSimilar}
            onExpandNode={handleExpandNode}
          />
        )}

        {/* Stats Badge */}
        {stats && (
          <div className="absolute bottom-4 left-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 text-sm">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <span className="text-gray-500">Nodes:</span>
              <span className="font-medium">{stats.totalNodes}</span>
              <span className="text-gray-500">Edges:</span>
              <span className="font-medium">{stats.totalEdges}</span>
              <span className="text-gray-500">Entities:</span>
              <span className="font-medium">{stats.entities}</span>
              <span className="text-gray-500">Relations:</span>
              <span className="font-medium">{stats.relations}</span>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="absolute bottom-4 right-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 text-sm">
          <div className="font-medium mb-2">Legend</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
              <span>Entity</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 transform rotate-45"></div>
              <span>Relation</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500"></div>
              <span>Chunk</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-purple-500 rounded-sm"></div>
              <span>Document</span>
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
  );
}
