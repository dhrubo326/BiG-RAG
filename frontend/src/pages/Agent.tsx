/**
 * Agent Page
 *
 * Multi-hop reasoning agent interface with:
 * - Question input and parameter configuration
 * - Visual debugging with reasoning trace
 * - Answer display with confidence metrics
 * - Full context inspection
 */

import { Brain, AlertCircle } from 'lucide-react';
import { useAgent } from '../hooks/useAgent';
import { AgentInput } from '../components/agent/AgentInput';
import { AgentAnswer } from '../components/agent/AgentAnswer';
import { ReasoningTrace } from '../components/agent/ReasoningTrace';
import { ContextsList } from '../components/agent/ContextsList';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';

export function Agent() {
  const agent = useAgent();

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Brain className="w-6 h-6 text-purple-600" />
              <h1 className="text-2xl font-bold">Multi-Hop Reasoning Agent</h1>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Complex questions requiring iterative query execution and reasoning
            </p>
          </div>

          {/* Agent Status */}
          {agent.agentInfo && (
            <div className="text-right">
              <div className="flex items-center gap-2 justify-end">
                <div className={`w-2 h-2 rounded-full ${agent.agentReady ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm font-medium">
                  {agent.agentReady ? 'Ready' : 'Not Ready'}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Model: {agent.agentInfo.default_model}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <div className="container mx-auto px-6 py-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column: Input */}
            <div>
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-lg font-semibold mb-4">Query Configuration</h2>
                <AgentInput
                  question={agent.question}
                  language={agent.language}
                  maxIterations={agent.maxIterations}
                  agentModel={agent.agentModel}
                  enableParallel={agent.enableParallel}
                  topKPerQuery={agent.topKPerQuery}
                  numKgInContext={agent.numKgInContext}
                  numChunksInContext={agent.numChunksInContext}
                  enableReranking={agent.enableReranking}
                  confidenceThreshold={agent.confidenceThreshold}
                  isLoading={agent.isLoading}
                  agentReady={agent.agentReady}
                  setQuestion={agent.setQuestion}
                  setLanguage={agent.setLanguage}
                  setMaxIterations={agent.setMaxIterations}
                  setAgentModel={agent.setAgentModel}
                  setEnableParallel={agent.setEnableParallel}
                  setTopKPerQuery={agent.setTopKPerQuery}
                  setNumKgInContext={agent.setNumKgInContext}
                  setNumChunksInContext={agent.setNumChunksInContext}
                  setEnableReranking={agent.setEnableReranking}
                  setConfidenceThreshold={agent.setConfidenceThreshold}
                  onSubmit={agent.submitQuery}
                  onReset={agent.resetParameters}
                />
              </div>
            </div>

            {/* Right Column: Results */}
            <div>
              {/* Error State */}
              {agent.error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="text-sm font-semibold text-red-800 dark:text-red-200 mb-1">Error</h3>
                      <p className="text-sm text-red-700 dark:text-red-300">{agent.error}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Loading State */}
              {agent.isLoading && (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8">
                  <div className="flex flex-col items-center justify-center gap-4">
                    <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
                    <div className="text-center">
                      <h3 className="font-semibold mb-1">Processing Query...</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                        The agent is iteratively planning queries, retrieving contexts, and reasoning.
                      </p>
                      <p className="text-xs text-gray-400 dark:text-gray-500">
                        This typically takes 3-5 minutes. Please wait...
                      </p>
                      <div className="mt-4 flex items-center gap-2 justify-center">
                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '600ms' }} />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Results */}
              {!agent.isLoading && agent.response && (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                  <Tabs defaultValue="answer" className="w-full">
                    <div className="border-b border-gray-200 dark:border-gray-700 px-6 pt-4">
                      <TabsList>
                        <TabsTrigger value="answer">Answer</TabsTrigger>
                        <TabsTrigger value="trace">Reasoning Trace</TabsTrigger>
                        <TabsTrigger value="contexts">
                          Contexts ({agent.response.contexts_used.length})
                        </TabsTrigger>
                      </TabsList>
                    </div>

                    <div className="p-6">
                      <TabsContent value="answer" className="mt-0">
                        <AgentAnswer response={agent.response} />
                      </TabsContent>

                      <TabsContent value="trace" className="mt-0">
                        <ReasoningTrace reasoningTrace={agent.response.reasoning_trace} />
                      </TabsContent>

                      <TabsContent value="contexts" className="mt-0">
                        <ContextsList contexts={agent.response.contexts_used} />
                      </TabsContent>
                    </div>
                  </Tabs>
                </div>
              )}

              {/* Empty State */}
              {!agent.isLoading && !agent.response && !agent.error && (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8">
                  <div className="text-center">
                    <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="font-semibold mb-2">No Results Yet</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Enter a question and submit to see the agent's reasoning process.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
