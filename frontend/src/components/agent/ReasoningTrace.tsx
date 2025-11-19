/**
 * ReasoningTrace Component
 *
 * Visual debugging panel showing iteration-by-iteration reasoning steps.
 */

import { useState } from 'react';
import { ChevronDown, ChevronUp, Brain, Search, Zap, Database, CheckCircle2 } from 'lucide-react';
import { Badge } from '../ui/badge';
import type { ReasoningStep } from '../../types/api';

interface ReasoningTraceProps {
  reasoningTrace: ReasoningStep[];
}

export function ReasoningTrace({ reasoningTrace }: ReasoningTraceProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([1])); // Expand first step by default

  const toggleStep = (stepNum: number) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepNum)) {
      newExpanded.delete(stepNum);
    } else {
      newExpanded.add(stepNum);
    }
    setExpandedSteps(newExpanded);
  };

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    if (conf >= 0.6) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
  };

  return (
    <div className="space-y-4">
      {reasoningTrace.map((step) => {
        const isExpanded = expandedSteps.has(step.step);

        return (
          <div
            key={step.step}
            className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-800"
          >
            {/* Step Header */}
            <button
              onClick={() => toggleStep(step.step)}
              className="w-full p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                  <span className="text-sm font-bold text-blue-700 dark:text-blue-300">{step.step}</span>
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Iteration {step.step}</h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {step.planned_queries.length} queries • {step.executed_actions.length} actions • {formatTime(step.execution_time_ms)}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Badge className={getConfidenceColor(step.confidence)}>
                  {(step.confidence * 100).toFixed(0)}%
                </Badge>
                {isExpanded ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </div>
            </button>

            {/* Step Details */}
            {isExpanded && (
              <div className="p-4 pt-0 space-y-4 border-t border-gray-100 dark:border-gray-700">
                {/* Thought */}
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold mb-2">
                    <Brain className="w-4 h-4" />
                    Thought
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                    {step.thought}
                  </p>
                </div>

                {/* Planned Queries */}
                {step.planned_queries.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 text-sm font-semibold mb-2">
                      <Search className="w-4 h-4" />
                      Queries Planned ({step.planned_queries.length})
                    </div>
                    <div className="space-y-2">
                      {step.planned_queries.map((query, idx) => (
                        <div
                          key={idx}
                          className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded space-y-1"
                        >
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              {query.language}
                            </Badge>
                            <span className="text-sm font-medium">{query.query}</span>
                          </div>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            Reason: {query.reason}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Executed Actions */}
                {step.executed_actions.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 text-sm font-semibold mb-2">
                      <Zap className="w-4 h-4" />
                      Actions Executed ({step.executed_actions.length})
                    </div>
                    <div className="space-y-2">
                      {step.executed_actions.map((action, idx) => (
                        <div
                          key={idx}
                          className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded flex items-center justify-between"
                        >
                          <div>
                            <div className="text-sm font-medium">{action.action_type}</div>
                            <div className="text-xs text-gray-600 dark:text-gray-400">
                              {action.query} ({action.language})
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-semibold text-blue-600 dark:text-blue-400">
                              {action.num_results} results
                            </div>
                            <div className="text-xs text-gray-500">
                              {formatTime(action.execution_time_ms)}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Observations */}
                {step.observations.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 text-sm font-semibold mb-2">
                      <Database className="w-4 h-4" />
                      Observations ({step.observations.length})
                    </div>
                    <div className="space-y-3">
                      {step.observations.map((obs, idx) => (
                        <div
                          key={idx}
                          className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded space-y-2"
                        >
                          <div className="text-sm font-medium">{obs.query}</div>
                          {obs.summary && (
                            <p className="text-xs text-gray-600 dark:text-gray-400 italic">
                              {obs.summary}
                            </p>
                          )}
                          <div className="text-xs text-gray-500">
                            {obs.contexts.length} contexts retrieved (showing top 3)
                          </div>
                          <div className="space-y-1">
                            {obs.contexts.slice(0, 3).map((ctx, ctxIdx) => (
                              <div
                                key={ctxIdx}
                                className="text-xs bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700"
                              >
                                <div className="flex items-center justify-between mb-1">
                                  <span className="font-medium text-gray-600 dark:text-gray-400">
                                    Context {ctxIdx + 1}
                                  </span>
                                  {ctx.relevance_score !== undefined && (
                                    <span className="text-blue-600 dark:text-blue-400">
                                      {ctx.relevance_score.toFixed(3)}
                                    </span>
                                  )}
                                </div>
                                <p className="text-gray-700 dark:text-gray-300 line-clamp-2">
                                  {ctx.text}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Variables Stored */}
                {Object.keys(step.variables_stored).length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 text-sm font-semibold mb-2">
                      <CheckCircle2 className="w-4 h-4" />
                      Variables Stored ({Object.keys(step.variables_stored).length})
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded space-y-1">
                      {Object.entries(step.variables_stored).map(([key, value]) => (
                        <div key={key} className="flex items-start gap-2 text-sm">
                          <span className="font-mono text-blue-600 dark:text-blue-400 font-semibold">
                            {key}:
                          </span>
                          <span className="text-gray-700 dark:text-gray-300">
                            {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
