/**
 * ContextsList Component
 *
 * Displays all contexts retrieved by the agent for debugging.
 */

import { useState } from 'react';
import { Search, ChevronDown, ChevronUp } from 'lucide-react';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import type { AgentContextItem } from '../../types/api';

interface ContextsListProps {
  contexts: AgentContextItem[];
}

export function ContextsList({ contexts }: ContextsListProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedContexts, setExpandedContexts] = useState<Set<number>>(new Set());

  // Filter contexts by search term
  const filteredContexts = contexts.filter((ctx) =>
    ctx.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
    ctx.source?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const toggleContext = (index: number) => {
    const newExpanded = new Set(expandedContexts);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedContexts(newExpanded);
  };

  const expandAll = () => {
    setExpandedContexts(new Set(filteredContexts.map((_, i) => i)));
  };

  const collapseAll = () => {
    setExpandedContexts(new Set());
  };

  return (
    <div className="space-y-4">
      {/* Search and Controls */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <Input
            type="text"
            placeholder="Search contexts..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <button
          onClick={expandAll}
          className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          Expand All
        </button>
        <button
          onClick={collapseAll}
          className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          Collapse All
        </button>
      </div>

      {/* Results Count */}
      <div className="text-sm text-gray-600 dark:text-gray-400">
        Showing {filteredContexts.length} of {contexts.length} contexts
      </div>

      {/* Contexts List */}
      <div className="space-y-2">
        {filteredContexts.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            {searchTerm ? 'No contexts match your search' : 'No contexts retrieved'}
          </div>
        ) : (
          filteredContexts.map((ctx, index) => {
            const isExpanded = expandedContexts.has(index);

            return (
              <div
                key={index}
                className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-800"
              >
                {/* Context Header */}
                <button
                  onClick={() => toggleContext(index)}
                  className="w-full p-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <div className="flex items-center gap-3 flex-1">
                    <span className="text-sm font-semibold text-gray-500 dark:text-gray-400">
                      #{index + 1}
                    </span>
                    <div className="text-left flex-1">
                      {/* Source */}
                      {ctx.source && (
                        <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                          Source: {ctx.source}
                        </div>
                      )}
                      {/* Preview */}
                      <p className="text-sm line-clamp-1">
                        {ctx.text}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 ml-2">
                    {/* Relevance Score */}
                    {ctx.relevance_score !== undefined && (
                      <Badge variant="outline" className="text-xs">
                        {ctx.relevance_score.toFixed(3)}
                      </Badge>
                    )}
                    {/* Metadata Type */}
                    {ctx.metadata?.type && (
                      <Badge className="text-xs">
                        {ctx.metadata.type}
                      </Badge>
                    )}
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                </button>

                {/* Context Details */}
                {isExpanded && (
                  <div className="p-3 pt-0 space-y-3 border-t border-gray-100 dark:border-gray-700">
                    {/* Full Text */}
                    <div>
                      <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">
                        Text
                      </div>
                      <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                        {ctx.text}
                      </p>
                    </div>

                    {/* Metadata */}
                    {Object.keys(ctx.metadata).length > 0 && (
                      <div>
                        <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">
                          Metadata
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                          <pre className="text-xs overflow-x-auto">
                            {JSON.stringify(ctx.metadata, null, 2)}
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
