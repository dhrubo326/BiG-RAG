import { useState } from 'react';
import { FileText, Sparkles, Link, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import type { RetrievedContext } from '../../types';
import { formatScore } from '../../utils/formatters';

interface RetrievalVizProps {
  contexts: RetrievedContext[];
  thinking?: string | null;
  isLoading?: boolean;
}

const RetrievalViz: React.FC<RetrievalVizProps> = ({
  contexts,
  thinking,
  isLoading = false,
}) => {
  const [expandedContexts, setExpandedContexts] = useState<Set<number>>(new Set());
  const [activeTab, setActiveTab] = useState<'contexts' | 'graph' | 'trace'>('contexts');

  const toggleContext = (index: number) => {
    const newExpanded = new Set(expandedContexts);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedContexts(newExpanded);
  };

  // Group contexts by path
  const groupedContexts = contexts.reduce((acc, ctx, index) => {
    const path = ctx.path || (ctx.type === 'entity' ? 'Path A' : ctx.type === 'relation' ? 'Path B' : 'Path C');
    if (!acc[path]) {
      acc[path] = [];
    }
    acc[path].push({ ...ctx, index });
    return acc;
  }, {} as Record<string, (RetrievedContext & { index: number })[]>);

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setActiveTab('contexts')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'contexts'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
        >
          Contexts
        </button>
        <button
          onClick={() => setActiveTab('graph')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'graph'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
        >
          Graph
        </button>
        <button
          onClick={() => setActiveTab('trace')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'trace'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
        >
          Trace
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <div className="flex items-center gap-2 text-gray-500">
              <Sparkles className="w-4 h-4 animate-pulse" />
              <span className="text-sm">Retrieving contexts...</span>
            </div>
          </div>
        ) : activeTab === 'contexts' ? (
          <div className="space-y-4">
            {contexts.length === 0 ? (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No contexts retrieved</p>
              </div>
            ) : (
              Object.entries(groupedContexts).map(([path, ctxs]) => (
                <div key={path} className="space-y-2">
                  <h4 className="font-medium text-sm text-gray-700 dark:text-gray-300">
                    {path} ({ctxs.length})
                  </h4>
                  {ctxs.map((ctx) => (
                    <div
                      key={ctx.index}
                      className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700"
                    >
                      <button
                        onClick={() => toggleContext(ctx.index)}
                        className="w-full flex items-start gap-2 text-left"
                      >
                        {expandedContexts.has(ctx.index) ? (
                          <ChevronDown className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        ) : (
                          <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        )}

                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            {ctx.type === 'entity' && (
                              <span className="px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 text-xs rounded">
                                Entity
                              </span>
                            )}
                            {ctx.type === 'relation' && (
                              <span className="px-2 py-0.5 bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300 text-xs rounded">
                                Relation
                              </span>
                            )}
                            {ctx.type === 'chunk' && (
                              <span className="px-2 py-0.5 bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300 text-xs rounded">
                                Chunk
                              </span>
                            )}
                            <span className="text-xs text-gray-500">
                              Score: {formatScore(ctx.score)}
                            </span>
                          </div>

                          <p className={`text-sm text-gray-700 dark:text-gray-300 ${
                            expandedContexts.has(ctx.index) ? '' : 'line-clamp-2'
                          }`}>
                            {ctx.content}
                          </p>

                          {expandedContexts.has(ctx.index) && (
                            <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500">
                              <div>Source: {ctx.source}</div>
                              {ctx.metadata && Object.keys(ctx.metadata).length > 0 && (
                                <div className="mt-1">
                                  Metadata: {JSON.stringify(ctx.metadata)}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </button>
                    </div>
                  ))}
                </div>
              ))
            )}
          </div>
        ) : activeTab === 'graph' ? (
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <h4 className="font-medium mb-3 flex items-center gap-2">
                <Link className="w-4 h-4" />
                Retrieved Subgraph
              </h4>
              <div className="h-64 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg flex items-center justify-center">
                <div className="text-center text-gray-500 dark:text-gray-400">
                  <p className="text-sm mb-2">Mini graph visualization</p>
                  <button className="text-xs text-blue-500 hover:text-blue-600 flex items-center gap-1 mx-auto">
                    <ExternalLink className="w-3 h-3" />
                    View in full graph
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {thinking ? (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <h4 className="font-medium mb-3">Retrieval Trace</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-1.5"></div>
                    <div className="flex-1">
                      <p className="font-medium">Thinking</p>
                      <p className="text-gray-600 dark:text-gray-400 mt-1">{thinking}</p>
                    </div>
                  </div>

                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full mt-1.5"></div>
                    <div className="flex-1">
                      <p className="font-medium">Query Generated</p>
                      <p className="text-gray-600 dark:text-gray-400 mt-1">
                        Retrieved {contexts.length} contexts
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 bg-purple-500 rounded-full mt-1.5"></div>
                    <div className="flex-1">
                      <p className="font-medium">Response Generated</p>
                      <p className="text-gray-600 dark:text-gray-400 mt-1">
                        Answer synthesized from retrieved contexts
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <p>No trace information available</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default RetrievalViz;