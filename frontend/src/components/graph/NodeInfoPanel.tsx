import React from 'react';
import { X, ExternalLink, Search, FileText } from 'lucide-react';
import type { CytoscapeNode } from '../../types';
import { capitalize, formatScore } from '../../utils/formatters';

interface NodeInfoPanelProps {
  node: CytoscapeNode | null;
  onClose: () => void;
  onViewDocument?: (docId: string) => void;
  onFindSimilar?: (nodeId: string) => void;
  onExpandNode?: (nodeId: string) => void;
}

const NodeInfoPanel: React.FC<NodeInfoPanelProps> = ({
  node,
  onClose,
  onViewDocument,
  onFindSimilar,
  onExpandNode,
}) => {
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