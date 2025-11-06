/**
 * Graph Tooltip Component
 *
 * Shows tooltips on node hover with:
 * - Node label
 * - Node type
 * - Description
 * - Weight
 * - Connection count
 */

import React from 'react';
import { createPortal } from 'react-dom';
import type { CytoscapeNode } from '../../types';

interface GraphTooltipProps {
  node: CytoscapeNode | null;
  position: { x: number; y: number };
  connectionCount?: number;
}

const GraphTooltip: React.FC<GraphTooltipProps> = ({ node, position, connectionCount }) => {
  if (!node) return null;

  const { data } = node;

  // Determine node type label and color
  const typeConfig = {
    entity: { label: 'Entity', color: 'blue' },
    relation: { label: 'Relation', color: 'red' },
    chunk: { label: 'Chunk', color: 'green' },
    document: { label: 'Document', color: 'purple' },
  };

  const config = typeConfig[data.type as keyof typeof typeConfig] || typeConfig.entity;

  // Tooltip component
  const tooltip = (
    <div
      className="fixed z-50 pointer-events-none"
      style={{
        left: `${position.x + 15}px`,
        top: `${position.y + 15}px`,
      }}
    >
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl border border-gray-200 dark:border-gray-700 p-4 min-w-[320px] max-w-md animate-in fade-in slide-in-from-bottom-2 duration-200">
        {/* Node Type Badge */}
        <div className="flex items-center gap-2 mb-2">
          <span
            className={`px-2 py-0.5 text-xs font-medium rounded-full bg-${config.color}-100 dark:bg-${config.color}-900/30 text-${config.color}-700 dark:text-${config.color}-300`}
          >
            {config.label}
          </span>
          {data.weight !== undefined && (
            <span className="text-xs text-gray-500 dark:text-gray-400">
              Weight: {data.weight.toFixed(2)}
            </span>
          )}
        </div>

        {/* Node Label */}
        <div className="font-semibold text-base text-gray-900 dark:text-gray-100 mb-2">
          {data.label || data.id}
        </div>

        {/* Description (if available) */}
        {data.description && (
          <div className="text-xs text-gray-600 dark:text-gray-400 mb-2 line-clamp-4 leading-relaxed">
            {data.description}
          </div>
        )}

        {/* Metadata */}
        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500 dark:text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
          {connectionCount !== undefined && (
            <div className="flex items-center gap-1 whitespace-nowrap">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              <span>{connectionCount} connection{connectionCount !== 1 ? 's' : ''}</span>
            </div>
          )}
          {data.source_id && (
            <div className="flex items-center gap-1 min-w-0 flex-1">
              <svg className="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span className="truncate text-xs" title={data.source_id}>
                {data.source_id.length > 20 ? `${data.source_id.substring(0, 20)}...` : data.source_id}
              </span>
            </div>
          )}
        </div>

        {/* Entity Type (for entities) */}
        {data.type === 'entity' && data.metadata?.entity_type && (
          <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
            <span className="text-xs text-gray-500 dark:text-gray-400">
              Type: <span className="font-medium text-gray-700 dark:text-gray-300">{data.metadata.entity_type}</span>
            </span>
          </div>
        )}

        {/* Tooltip Arrow */}
        <div
          className="absolute w-2 h-2 bg-white dark:bg-gray-800 border-l border-t border-gray-200 dark:border-gray-700 transform rotate-45"
          style={{
            left: '-4px',
            top: '12px',
          }}
        />
      </div>
    </div>
  );

  // Render tooltip in portal to avoid z-index issues
  return createPortal(tooltip, document.body);
};

export default GraphTooltip;
