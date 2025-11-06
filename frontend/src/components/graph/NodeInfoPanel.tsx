import React, { useState, useEffect } from 'react';
import { X, ExternalLink, Search, FileText, Network, TrendingUp } from 'lucide-react';
import type { CytoscapeNode } from '../../types';
import { capitalize, formatScore } from '../../utils/formatters';

interface NodeInfoPanelProps {
  node: CytoscapeNode | null;
  onClose: () => void;
  onViewDocument?: (docId: string) => void;
  onFindSimilar?: (nodeId: string) => void;
  onExpandNode?: (nodeId: string) => void;
  cytoscapeInstance?: any;
}

interface ConnectionStats {
  neighbors: number;
  incomingEdges: number;
  outgoingEdges: number;
  totalEdges: number;
  connectedEntities: number;
  connectedRelations: number;
  connectedChunks: number;
}

const NodeInfoPanel: React.FC<NodeInfoPanelProps> = ({
  node,
  onClose,
  onViewDocument,
  onFindSimilar,
  onExpandNode,
  cytoscapeInstance,
}) => {
  const [connectionStats, setConnectionStats] = useState<ConnectionStats | null>(null);
  const [isVisible, setIsVisible] = useState(false);

  // Calculate connection statistics
  useEffect(() => {
    if (!node || !cytoscapeInstance) {
      setConnectionStats(null);
      return;
    }

    try {
      const cyNode = cytoscapeInstance.$(`#${CSS.escape(node.data.id)}`);
      if (!cyNode || cyNode.length === 0) {
        setConnectionStats(null);
        return;
      }

      // Get connected elements
      const connectedEdges = cyNode.connectedEdges();
      const neighbors = cyNode.neighborhood('node');

      // Count by type
      const connectedEntities = neighbors.filter('[type="entity"]').length;
      const connectedRelations = neighbors.filter('[type="relation"]').length;
      const connectedChunks = neighbors.filter('[type="chunk"]').length;

      // Incoming/outgoing edges
      const incomingEdges = cyNode.connectedEdges(`[target="${CSS.escape(node.data.id)}"]`).length;
      const outgoingEdges = cyNode.connectedEdges(`[source="${CSS.escape(node.data.id)}"]`).length;

      setConnectionStats({
        neighbors: neighbors.length,
        incomingEdges,
        outgoingEdges,
        totalEdges: connectedEdges.length,
        connectedEntities,
        connectedRelations,
        connectedChunks,
      });
    } catch (err) {
      console.error('Error calculating connection stats:', err);
      setConnectionStats(null);
    }
  }, [node, cytoscapeInstance]);

  // ✅ NEW: Smooth slide-in animation
  useEffect(() => {
    if (node) {
      // Delay to trigger animation
      setTimeout(() => setIsVisible(true), 10);
    } else {
      setIsVisible(false);
    }
  }, [node]);

  if (!node) return null;

  const { data } = node;
  const typeColor = {
    entity: 'blue',
    relation: 'red',
    chunk: 'green',
    document: 'purple',
  };

  const typeColorClass = typeColor[data.type as keyof typeof typeColor] || 'gray';

  return (
    <>
      {/* ✅ IMPROVED: Overlay for mobile */}
      <div
        className={`fixed inset-0 bg-black/20 backdrop-blur-sm z-40 transition-opacity duration-300 lg:hidden ${
          isVisible ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
      />

      {/* ✅ IMPROVED: Panel with slide-in animation and proper positioning */}
      <div
        className={`
          fixed top-0 right-0 h-full w-96 max-w-full
          bg-white/98 dark:bg-gray-900/98 backdrop-blur-sm
          border-l border-gray-200 dark:border-gray-700
          shadow-2xl
          z-50
          transform transition-transform duration-300 ease-out
          ${isVisible ? 'translate-x-0' : 'translate-x-full'}
          overflow-y-auto
          flex flex-col
        `}
      >
        {/* ✅ IMPROVED: Sticky header with gradient */}
        <div className="sticky top-0 z-10 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 border-b border-gray-200 dark:border-gray-700 px-6 py-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full bg-${typeColorClass}-500 animate-pulse`}></div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Node Details
              </h3>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors group"
              aria-label="Close panel"
            >
              <X className="w-5 h-5 text-gray-500 group-hover:text-gray-900 dark:group-hover:text-gray-100 transition-colors" />
            </button>
          </div>
        </div>

        {/* ✅ IMPROVED: Content with better spacing */}
        <div className="flex-1 px-6 py-6 space-y-6">
          {/* ✅ IMPROVED: Node Type Badge with icon */}
          <div className="flex items-center gap-2">
            <span
              className={`
                inline-flex items-center px-3 py-1.5 rounded-lg text-sm font-medium
                bg-${typeColorClass}-100 dark:bg-${typeColorClass}-900/30
                text-${typeColorClass}-800 dark:text-${typeColorClass}-200
                border border-${typeColorClass}-200 dark:border-${typeColorClass}-700
              `}
            >
              {data.type === 'entity' && '⚫'}
              {data.type === 'relation' && '🔶'}
              {data.type === 'chunk' && '📄'}
              {data.type === 'document' && '📋'}
              <span className="ml-1.5">{capitalize(data.type)}</span>
            </span>
          </div>

          {/* ✅ IMPROVED: Node Name with better typography */}
          <div className="bg-gradient-to-br from-gray-50 to-white dark:from-gray-800 dark:to-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <label className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
              Name
            </label>
            <p className="mt-2 text-base font-medium text-gray-900 dark:text-gray-100 leading-relaxed">
              {data.label || 'Unnamed'}
            </p>
          </div>

          {/* Description */}
          {data.description && (
            <div className="bg-gradient-to-br from-blue-50 to-white dark:from-blue-900/10 dark:to-gray-900 rounded-xl p-4 border border-blue-100 dark:border-blue-900/30">
              <label className="text-xs font-semibold text-blue-700 dark:text-blue-400 uppercase tracking-wide">
                Description
              </label>
              <p className="mt-2 text-sm text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap">
                {data.description}
              </p>
            </div>
          )}

          {/* ✅ IMPROVED: Stats Grid */}
          <div className="grid grid-cols-2 gap-3">
            {/* Weight */}
            {data.weight !== undefined && (
              <div className="bg-gradient-to-br from-purple-50 to-white dark:from-purple-900/10 dark:to-gray-900 rounded-lg p-3 border border-purple-100 dark:border-purple-900/30">
                <label className="text-xs font-medium text-purple-600 dark:text-purple-400 flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" />
                  Weight
                </label>
                <p className="mt-1 text-lg font-bold text-gray-900 dark:text-gray-100">
                  {formatScore(data.weight)}
                </p>
              </div>
            )}

            {/* Source Document */}
            {data.source_id && (
              <div className="bg-gradient-to-br from-green-50 to-white dark:from-green-900/10 dark:to-gray-900 rounded-lg p-3 border border-green-100 dark:border-green-900/30">
                <label className="text-xs font-medium text-green-600 dark:text-green-400 flex items-center gap-1">
                  <FileText className="w-3 h-3" />
                  Source
                </label>
                <p className="mt-1 text-xs font-mono text-gray-700 dark:text-gray-300 truncate" title={data.source_id}>
                  {data.source_id.split('-').pop()}
                </p>
              </div>
            )}
          </div>

          {/* ✅ IMPROVED: Connection Statistics */}
          {connectionStats && (
            <div className="bg-gradient-to-br from-gray-50 to-white dark:from-gray-800 dark:to-gray-900 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-4">
                <Network className="w-5 h-5 text-blue-500" />
                <label className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                  Connections
                </label>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 dark:text-gray-400">Neighbors</span>
                  <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
                    {connectionStats.neighbors}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 dark:text-gray-400">Total Edges</span>
                  <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
                    {connectionStats.totalEdges}
                  </span>
                </div>
                {connectionStats.connectedEntities > 0 && (
                  <div className="flex flex-col">
                    <span className="text-xs text-blue-600 dark:text-blue-400">⚫ Entities</span>
                    <span className="text-base font-semibold text-blue-700 dark:text-blue-300">
                      {connectionStats.connectedEntities}
                    </span>
                  </div>
                )}
                {connectionStats.connectedRelations > 0 && (
                  <div className="flex flex-col">
                    <span className="text-xs text-red-600 dark:text-red-400">🔶 Relations</span>
                    <span className="text-base font-semibold text-red-700 dark:text-red-300">
                      {connectionStats.connectedRelations}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Metadata */}
          {data.metadata && Object.keys(data.metadata).filter(k => k !== 'role').length > 0 && (
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
              <label className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2 block">
                Metadata
              </label>
              <div className="space-y-1.5">
                {Object.entries(data.metadata)
                  .filter(([key]) => key !== 'role')
                  .map(([key, value]) => (
                    <div key={key} className="text-xs flex justify-between">
                      <span className="font-medium text-gray-600 dark:text-gray-400">
                        {key}:
                      </span>
                      <span className="text-gray-900 dark:text-gray-100 font-mono max-w-[60%] truncate" title={String(value)}>
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* ✅ IMPROVED: Node ID (collapsed by default) */}
          <details className="group">
            <summary className="text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-700 dark:hover:text-gray-200 transition-colors">
              Show Node ID
            </summary>
            <div className="mt-2 p-3 bg-gray-100 dark:bg-gray-800 rounded-lg">
              <p className="text-xs font-mono text-gray-600 dark:text-gray-300 break-all">
                {data.id}
              </p>
            </div>
          </details>
        </div>

        {/* ✅ IMPROVED: Action Buttons - sticky bottom */}
        <div className="sticky bottom-0 bg-gradient-to-t from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-4 space-y-2 shadow-lg">
          {data.source_id && onViewDocument && (
            <button
              onClick={() => onViewDocument(data.source_id!)}
              className="w-full px-4 py-2.5 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2 shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
            >
              <FileText className="w-4 h-4" />
              View Document
            </button>
          )}

          {onFindSimilar && (
            <button
              onClick={() => onFindSimilar(data.id)}
              className="w-full px-4 py-2.5 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
            >
              <Search className="w-4 h-4" />
              Find Similar
            </button>
          )}

          {onExpandNode && (
            <button
              onClick={() => onExpandNode(data.id)}
              className="w-full px-4 py-2.5 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
            >
              <ExternalLink className="w-4 h-4" />
              Expand Neighbors
            </button>
          )}
        </div>
      </div>
    </>
  );
};

export default NodeInfoPanel;
