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
  // ✅ PHASE 2: Add Cytoscape instance for querying connections
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

  // ✅ PHASE 2: Calculate connection statistics
  useEffect(() => {
    if (!node || !cytoscapeInstance) {
      setConnectionStats(null);
      return;
    }

    try {
      const cyNode = cytoscapeInstance.$(`#${node.data.id}`);
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
      const incomingEdges = cyNode.connectedEdges('[target="' + node.data.id + '"]').length;
      const outgoingEdges = cyNode.connectedEdges('[source="' + node.data.id + '"]').length;

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

  if (!node) return null;

  const { data } = node;
  const typeColor = {
    entity: 'blue',
    relation: 'red',
    chunk: 'green',
    document: 'purple',
  };

  return (
    <div className="absolute top-0 right-0 w-80 h-full bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-lg overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Node Details</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Node Type Badge */}
        <div>
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-${
              typeColor[data.type as keyof typeof typeColor]
            }-100 text-${typeColor[data.type as keyof typeof typeColor]}-800`}
          >
            {capitalize(data.type)}
          </span>
        </div>

        {/* Node Name */}
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            Name
          </label>
          <p className="mt-1 text-sm text-gray-900 dark:text-gray-100">
            {data.label || 'Unnamed'}
          </p>
        </div>

        {/* Node ID */}
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            ID
          </label>
          <p className="mt-1 text-xs font-mono text-gray-600 dark:text-gray-300">
            {data.id}
          </p>
        </div>

        {/* Description */}
        {data.description && (
          <div>
            <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Description
            </label>
            <p className="mt-1 text-sm text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
              {data.description}
            </p>
          </div>
        )}

        {/* Weight */}
        {data.weight !== undefined && (
          <div>
            <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Weight
            </label>
            <p className="mt-1 text-sm text-gray-900 dark:text-gray-100">
              {formatScore(data.weight)}
            </p>
          </div>
        )}

        {/* Source Document */}
        {data.source_id && (
          <div>
            <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Source Document
            </label>
            <p className="mt-1 text-sm text-gray-900 dark:text-gray-100">
              {data.source_id}
            </p>
          </div>
        )}

        {/* Metadata */}
        {data.metadata && Object.keys(data.metadata).length > 0 && (
          <div>
            <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Metadata
            </label>
            <div className="mt-1 space-y-1">
              {Object.entries(data.metadata).map(([key, value]) => (
                <div key={key} className="text-sm">
                  <span className="font-medium text-gray-600 dark:text-gray-400">
                    {key}:
                  </span>{' '}
                  <span className="text-gray-900 dark:text-gray-100">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Position */}
        {node.position && (
          <div>
            <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Position
            </label>
            <p className="mt-1 text-sm text-gray-900 dark:text-gray-100">
              X: {Math.round(node.position.x)}, Y: {Math.round(node.position.y)}
            </p>
          </div>
        )}

        {/* ✅ PHASE 2: Connection Statistics */}
        {connectionStats && (
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <Network className="w-4 h-4 text-blue-500" />
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Connections
              </label>
            </div>
            <div className="space-y-1.5 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Neighbors:</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {connectionStats.neighbors}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Total Edges:</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {connectionStats.totalEdges}
                </span>
              </div>
              {connectionStats.incomingEdges > 0 && (
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Incoming:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {connectionStats.incomingEdges}
                  </span>
                </div>
              )}
              {connectionStats.outgoingEdges > 0 && (
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Outgoing:</span>
                  <span className="font-medium text-blue-600 dark:text-blue-400">
                    {connectionStats.outgoingEdges}
                  </span>
                </div>
              )}
              {connectionStats.connectedEntities > 0 && (
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Entities:</span>
                  <span className="font-medium text-blue-600 dark:text-blue-400">
                    {connectionStats.connectedEntities}
                  </span>
                </div>
              )}
              {connectionStats.connectedRelations > 0 && (
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Relations:</span>
                  <span className="font-medium text-red-600 dark:text-red-400">
                    {connectionStats.connectedRelations}
                  </span>
                </div>
              )}
              {connectionStats.connectedChunks > 0 && (
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Chunks:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {connectionStats.connectedChunks}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700 space-y-2">
          {data.source_id && onViewDocument && (
            <button
              onClick={() => onViewDocument(data.source_id!)}
              className="w-full px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center gap-2"
            >
              <FileText className="w-4 h-4" />
              View Document
            </button>
          )}

          {onFindSimilar && (
            <button
              onClick={() => onFindSimilar(data.id)}
              className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center justify-center gap-2"
            >
              <Search className="w-4 h-4" />
              Find Similar
            </button>
          )}

          {onExpandNode && (
            <button
              onClick={() => onExpandNode(data.id)}
              className="w-full px-3 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center justify-center gap-2"
            >
              <ExternalLink className="w-4 h-4" />
              Expand Neighbors
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default NodeInfoPanel;