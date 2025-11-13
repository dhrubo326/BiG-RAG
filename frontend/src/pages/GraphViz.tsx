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
import { Loader2, Search, X } from 'lucide-react';
import type { Core } from 'cytoscape';
import { useGraph } from '../hooks/useGraph';
import GraphCanvas from '../components/graph/GraphCanvas';
import GraphToolbar from '../components/graph/GraphToolbar';
import NodeInfoPanel from '../components/graph/NodeInfoPanel';
import GraphErrorBoundary from '../components/graph/GraphErrorBoundary'; // ✅ PHASE 2
import GraphTooltip from '../components/graph/GraphTooltip'; // ✅ PHASE 4.2
import type { GraphLayout, CytoscapeNode } from '../types';
import api from '../services/api';
import type { HealthResponse } from '../types/api';

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
    orphanBreakdown,  // ✅ NEW
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

  // Dataset selection state
  const [selectedDataset, setSelectedDataset] = useState<string>(''); // Empty until server responds
  const [serverDataset, setServerDataset] = useState<string>('');
  const [isLoadingDataset, setIsLoadingDataset] = useState(true);
  const [availableDatasets] = useState<string[]>([
    'demo_test',      // Server default first
    'SingleTopic',
    '2WikiMultiHopQA',
    'HotpotQA',
    'Musique',
    'NQ',
    'PopQA',
    'TriviaQA'
  ]);

  // ✅ NEW: Debug mode state
  const [debugMode, setDebugMode] = useState(false);
  const [showOrphanPanel, setShowOrphanPanel] = useState(false);

  // Search state
  const [localSearchQuery, setLocalSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<CytoscapeNode[]>([]);

  // ✅ PHASE 4.2: Tooltip state
  const [hoveredNode, setHoveredNode] = useState<CytoscapeNode | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [tooltipConnectionCount, setTooltipConnectionCount] = useState<number | undefined>();

  // Fetch server's default dataset from health check
  useEffect(() => {
    const fetchServerDataset = async () => {
      setIsLoadingDataset(true);
      try {
        const response = await api.get<HealthResponse>('/');
        if (response.data.dataset) {
          setServerDataset(response.data.dataset);
          setSelectedDataset(response.data.dataset); // Use server's dataset as default
        } else {
          // Fallback to first dataset if server doesn't return one
          setSelectedDataset(availableDatasets[0]);
        }
      } catch (error) {
        console.error('Failed to fetch server dataset:', error);
        // Fallback to first dataset on error
        setSelectedDataset(availableDatasets[0]);
      } finally {
        setIsLoadingDataset(false);
      }
    };

    fetchServerDataset();
  }, [availableDatasets]);

  // Load graph on mount with performance-optimized settings
  useEffect(() => {
    // Only load if dataset is set (wait for server fetch to complete)
    if (!selectedDataset) return;

    console.log('[GraphViz] Loading graph with selectedDataset:', selectedDataset, 'debugMode:', debugMode);

    const loadGraphData = async () => {
      try {
        await loadGraph(selectedDataset, {
          limit: 1000, // Show top 1000 nodes
          sampleStrategy: 'diverse', // Get balanced mix of entities, relations, chunks
          includeAllOrphans: debugMode, // ✅ NEW: Use debug mode
        });
      } catch (err) {
        console.error('[GraphViz] Failed to load graph:', err);
        toast.error('Failed to load graph. Please try again.');
      }
    };

    loadGraphData();
  }, [selectedDataset, debugMode, loadGraph]); // Reload when dataset or debug mode changes

  // ✅ NEW: Update Cytoscape when orphan filter changes
  useEffect(() => {
    if (!cyInstance) return;

    if (filters.showOrphans) {
      // Show all nodes
      cyInstance.nodes().style('display', 'element');
    } else {
      // Hide orphan nodes (connections = 0)
      cyInstance.nodes().forEach((node: any) => {
        const connections = node.data('connections') || 0;
        if (connections === 0) {
          node.style('display', 'none');
        } else {
          node.style('display', 'element');
        }
      });
    }
  }, [filters.showOrphans, cyInstance]);

  // ✅ PHASE 2: Handle error boundary reset
  const handleErrorReset = () => {
    // Clear graph state and reload
    clearGraph();
    loadGraph(selectedDataset, {
      limit: 1000,
      sampleStrategy: 'diverse',
    });
  };

  // Handle dataset change
  const handleDatasetChange = (newDataset: string) => {
    console.log('[GraphViz] Dataset changed from', selectedDataset, 'to', newDataset);
    setSelectedDataset(newDataset);
    toast.info(`Switching to ${newDataset} dataset...`);
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

  // Handle search with results dropdown
  const handleSearchChange = (query: string) => {
    setLocalSearchQuery(query);

    // Search through nodes if query is at least 3 characters
    if (query.length >= 3) {
      const query_lower = query.toLowerCase();
      const results = nodes.filter((node) => {
        const label = (node.data.label || '').toLowerCase();
        const description = (node.data.description || '').toLowerCase();
        const content = (node.data.content || '').toLowerCase();
        return label.includes(query_lower) || description.includes(query_lower) || content.includes(query_lower);
      }).slice(0, 20); // Limit to 20 results

      setSearchResults(results);
    } else {
      setSearchResults([]);
    }

    // Also call the hook's search function for highlighting
    searchNodes(query);
  };

  // Handle search result selection
  const selectAndZoomToNode = (nodeId: string) => {
    setSearchResults([]); // Close dropdown
    setLocalSearchQuery(''); // Clear search

    // Select the node
    selectNode(nodes.find(n => n.data.id === nodeId) || null);

    // Zoom to the node in Cytoscape
    if (cyInstance) {
      const node = cyInstance.$id(nodeId);
      if (node.length > 0) {
        cyInstance.animate({
          center: { eles: node },
          zoom: 2,
        }, {
          duration: 500,
        });
      }
    }
  };

  return (
    <GraphErrorBoundary onReset={handleErrorReset}>
      <div className="h-[calc(100vh-8rem)] flex flex-col relative">
      {/* Top Bar: Dataset + Search + Quick Stats */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3">
        <div className="flex items-center gap-4">
          {/* Dataset Selector */}
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
              Dataset:
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => handleDatasetChange(e.target.value)}
              disabled={isLoadingDataset}
              className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoadingDataset ? (
                <option value="">Loading server default...</option>
              ) : (
                availableDatasets.map((dataset) => (
                  <option key={dataset} value={dataset}>
                    {dataset}
                    {dataset === serverDataset && ' (Server Default)'}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Search Bar (Wide, prominent) */}
          <div className="flex-1 max-w-2xl relative">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search entities, relations, or chunks (min 3 characters)..."
                value={localSearchQuery}
                onChange={(e) => handleSearchChange(e.target.value)}
                className="w-full px-4 py-2 pl-10 pr-10 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
              />
              {localSearchQuery && (
                <button
                  onClick={() => handleSearchChange('')}
                  className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Search Results Dropdown */}
            {searchResults.length > 0 && (
              <div className="absolute z-50 mt-1 w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl max-h-96 overflow-y-auto">
                <div className="p-2 text-xs text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  Found {searchResults.length} results
                </div>
                {searchResults.map((result) => (
                  <button
                    key={result.data.id}
                    onClick={() => selectAndZoomToNode(result.data.id)}
                    className="w-full px-4 py-3 text-left hover:bg-gray-100 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-700 transition-colors"
                  >
                    <div className="flex items-start gap-3">
                      {/* Node Type Indicator */}
                      <div
                        className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                          result.data.type === 'entity'
                            ? 'bg-blue-500'
                            : result.data.type === 'relation'
                            ? 'bg-red-500'
                            : 'bg-green-500'
                        }`}
                      >
                        {result.data.type === 'entity'
                          ? 'E'
                          : result.data.type === 'relation'
                          ? 'R'
                          : 'C'}
                      </div>

                      {/* Node Info */}
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm text-gray-900 dark:text-gray-100 truncate">
                          {result.data.label}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 truncate mt-0.5">
                          {result.data.type === 'entity'
                            ? `Type: ${result.data.entityType || 'unknown'} • Weight: ${result.data.weight}`
                            : `Weight: ${result.data.weight} • ${result.data.connections || 0} connections`}
                        </div>
                        {result.data.connections === 0 && (
                          <span className="inline-block mt-1 px-2 py-0.5 text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 rounded">
                            Orphan Node
                          </span>
                        )}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Quick Stats */}
          {stats && (
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="font-semibold text-gray-900 dark:text-gray-100">
                  {stats.entities.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 transform rotate-45"></div>
                <span className="font-semibold text-gray-900 dark:text-gray-100">
                  {stats.relations.toLocaleString()}
                </span>
              </div>
              {stats.orphanNodes !== undefined && stats.orphanNodes > 0 && (
                <div className="text-xs text-yellow-600 dark:text-yellow-400 flex items-center gap-1">
                  <span className="w-2 h-2 bg-yellow-500 rounded-full"></span>
                  {stats.orphanNodes} orphan
                </div>
              )}
            </div>
          )}

          {/* ✅ NEW: Orphan Debug Button */}
          {orphanBreakdown && orphanBreakdown.total > 0 && (
            <div className="relative">
              <button
                onClick={() => setShowOrphanPanel(!showOrphanPanel)}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-yellow-100 hover:bg-yellow-200 dark:bg-yellow-900/30 dark:hover:bg-yellow-900/50 text-yellow-800 dark:text-yellow-200 rounded-lg border border-yellow-300 dark:border-yellow-700 transition-colors"
              >
                <span className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></span>
                <span className="font-medium">Orphan Debug</span>
                <svg className={`w-4 h-4 transition-transform ${showOrphanPanel ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* Dropdown Panel */}
              {showOrphanPanel && (
                <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 p-4 text-sm z-50">
                  <div className="flex items-center justify-between mb-3 pb-3 border-b border-gray-200 dark:border-gray-700">
                    <h4 className="font-semibold text-gray-900 dark:text-gray-100">Orphan Nodes Debug</h4>
                    <button
                      onClick={() => setShowOrphanPanel(false)}
                      className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Stats */}
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400">Total:</span>
                      <span className="font-bold text-gray-900 dark:text-gray-100">{orphanBreakdown.total}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400">Entities:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">{orphanBreakdown.entities}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400">Relations:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">{orphanBreakdown.relations}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400">Chunks:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">{orphanBreakdown.chunks}</span>
                    </div>
                    <div className="flex items-center justify-between pt-2 border-t border-gray-200 dark:border-gray-700">
                      <span className="text-xs text-gray-600 dark:text-gray-400">Displayed:</span>
                      <span className="font-bold text-gray-900 dark:text-gray-100">{orphanBreakdown.included_in_response}</span>
                    </div>
                  </div>

                  {/* Show/Hide Toggle */}
                  <div className="mb-3 pb-3 border-b border-gray-200 dark:border-gray-700">
                    <button
                      onClick={() => updateFilters({ showOrphans: !filters.showOrphans })}
                      className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                        filters.showOrphans
                          ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                          : 'bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100'
                      }`}
                    >
                      {filters.showOrphans ? 'Hide Orphan Nodes' : 'Show Orphan Nodes'}
                    </button>
                  </div>

                  {/* Debug Mode Toggle */}
                  <div className="mb-3 pb-3 border-b border-gray-200 dark:border-gray-700">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={debugMode}
                        onChange={(e) => {
                          console.log('[GraphViz] Debug mode toggled:', e.target.checked, 'for dataset:', selectedDataset);
                          setDebugMode(e.target.checked);
                          toast.info(e.target.checked ? 'Loading ALL orphan nodes...' : 'Reloading with 20% cap...');
                        }}
                        className="w-4 h-4 text-yellow-600 bg-gray-100 border-gray-300 rounded focus:ring-yellow-500"
                      />
                      <span className="text-xs font-medium text-gray-900 dark:text-gray-100">
                        Debug Mode (Show ALL)
                      </span>
                    </label>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 ml-6">
                      Bypass 20% cap for large graphs
                    </p>
                    {orphanBreakdown.include_all_orphans_mode && (
                      <div className="text-xs text-green-600 dark:text-green-400 font-medium mt-2 ml-6">
                        ✓ Debug mode active
                      </div>
                    )}
                  </div>

                  {/* CSV Export Button */}
                  <button
                    onClick={() => {
                      if (!cyInstance) {
                        toast.error('Graph not loaded');
                        return;
                      }

                      const orphanNodesData: any[] = [];
                      cyInstance.nodes().forEach((node: any) => {
                        const connections = node.data('connections') || 0;
                        if (connections === 0) {
                          orphanNodesData.push({
                            type: node.data('type'),
                            label: (node.data('label') || '').replace(/"/g, '""'),
                            weight: node.data('weight') || 0,
                            connections: connections,
                            source_id: node.data('source_id') || node.data('sourceId') || '',
                          });
                        }
                      });

                      if (orphanNodesData.length === 0) {
                        toast.info('No orphan nodes to export');
                        return;
                      }

                      const csv = ['Type,Label,Weight,Connections,Source ID'].concat(
                        orphanNodesData.map(n =>
                          `${n.type},"${n.label}",${n.weight},${n.connections},"${n.source_id}"`
                        )
                      ).join('\n');

                      const blob = new Blob([csv], { type: 'text/csv' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `orphan-nodes-${selectedDataset}-${new Date().toISOString().slice(0,10)}.csv`;
                      a.click();
                      URL.revokeObjectURL(url);

                      toast.success(`Exported ${orphanNodesData.length} orphan nodes`);
                      setShowOrphanPanel(false);
                    }}
                    className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Export to CSV
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Toolbar */}
      <GraphToolbar
        layout={layout}
        filters={filters}
        onLayoutChange={handleLayoutChange}
        onFiltersChange={updateFilters}
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
