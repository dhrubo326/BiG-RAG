/**
 * AgentAnswer Component
 *
 * Displays the final answer from the agent with confidence and metadata.
 */

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CheckCircle2, AlertCircle, Clock, Zap, DollarSign, Hash } from 'lucide-react';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import type { AgentResponse } from '../../types/api';

interface AgentAnswerProps {
  response: AgentResponse;
}

export function AgentAnswer({ response }: AgentAnswerProps) {
  const { answer, confidence, metadata, limitations } = response;

  // Format execution time
  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  // Format cost
  const formatCost = (usd: number) => {
    if (usd < 0.01) return `$${(usd * 1000).toFixed(2)}`;
    return `$${usd.toFixed(4)}`;
  };

  // Get confidence color
  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'text-green-600 dark:text-green-400';
    if (conf >= 0.6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  // Get stopped reason badge
  const getStoppedReasonBadge = (reason: string) => {
    switch (reason) {
      case 'complete':
        return <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Complete</Badge>;
      case 'high_confidence':
        return <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">High Confidence</Badge>;
      case 'max_iterations':
        return <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">Max Iterations</Badge>;
      default:
        return <Badge>{reason}</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Answer */}
      <div>
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-green-600" />
          Answer
        </h3>
        <div className="prose prose-sm dark:prose-invert max-w-none bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {answer}
          </ReactMarkdown>
        </div>
      </div>

      {/* Confidence */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold">Confidence</h3>
          <span className={`text-sm font-bold ${getConfidenceColor(confidence)}`}>
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        <Progress value={confidence * 100} className="h-2" />
      </div>

      {/* Limitations */}
      {limitations && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-yellow-800 dark:text-yellow-200 mb-1">Limitations</h4>
              <p className="text-sm text-yellow-700 dark:text-yellow-300">{limitations}</p>
            </div>
          </div>
        </div>
      )}

      {/* Metadata */}
      <div>
        <h3 className="text-sm font-semibold mb-3">Execution Metadata</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Iterations */}
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <Hash className="w-4 h-4" />
              <span className="text-xs font-medium">Iterations</span>
            </div>
            <p className="text-lg font-bold">{response.total_iterations}</p>
          </div>

          {/* Queries */}
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <Zap className="w-4 h-4" />
              <span className="text-xs font-medium">Queries</span>
            </div>
            <p className="text-lg font-bold">{metadata.queries_executed}</p>
          </div>

          {/* Execution Time */}
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <Clock className="w-4 h-4" />
              <span className="text-xs font-medium">Time</span>
            </div>
            <p className="text-lg font-bold">{formatTime(metadata.execution_time_ms)}</p>
          </div>

          {/* Cost */}
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <DollarSign className="w-4 h-4" />
              <span className="text-xs font-medium">Cost</span>
            </div>
            <p className="text-lg font-bold">{formatCost(metadata.total_cost_usd)}</p>
          </div>
        </div>

        {/* Additional metadata */}
        <div className="mt-4 flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
          <span>Model: <strong>{metadata.model_used}</strong></span>
          <span>Tokens: <strong>{metadata.total_tokens.toLocaleString()}</strong></span>
          <span>Stopped: {getStoppedReasonBadge(metadata.stopped_reason)}</span>
        </div>
      </div>
    </div>
  );
}
