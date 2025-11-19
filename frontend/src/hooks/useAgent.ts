/**
 * useAgent Hook
 *
 * Manages state and logic for the multi-hop reasoning agent.
 */

import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import {
  queryAgent,
  getAgentHealth,
  getAgentInfo,
  createAgentRequest,
} from '../services/agent';
import type {
  AgentRequest,
  AgentResponse,
  AgentHealthResponse,
  AgentInfo,
} from '../types/api';

export function useAgent() {
  // Agent status
  const [agentReady, setAgentReady] = useState(false);
  const [agentInfo, setAgentInfo] = useState<AgentInfo | null>(null);
  const [healthStatus, setHealthStatus] = useState<AgentHealthResponse | null>(null);

  // Request parameters
  const [question, setQuestion] = useState('');
  const [language, setLanguage] = useState('auto');
  const [maxIterations, setMaxIterations] = useState(3);
  const [agentModel, setAgentModel] = useState('gpt-4o');
  const [enableParallel, setEnableParallel] = useState(true);
  const [topKPerQuery, setTopKPerQuery] = useState(60);
  const [numKgInContext, setNumKgInContext] = useState(15);
  const [numChunksInContext, setNumChunksInContext] = useState(5);
  const [enableReranking, setEnableReranking] = useState(false);
  const [enableVariableStorage, setEnableVariableStorage] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.8);

  // Response state
  const [response, setResponse] = useState<AgentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load agent info on mount
  useEffect(() => {
    loadAgentStatus();
  }, []);

  /**
   * Load agent health and info
   */
  const loadAgentStatus = async () => {
    try {
      const [health, info] = await Promise.all([
        getAgentHealth(),
        getAgentInfo(),
      ]);

      setHealthStatus(health);
      setAgentInfo(info);
      setAgentReady(health.ready);

      if (!health.ready) {
        toast.error('Agent is not ready. Make sure OpenAI API key is configured.');
      }
    } catch (err) {
      console.error('Failed to load agent status:', err);
      setAgentReady(false);
      toast.error('Failed to connect to agent endpoint');
    }
  };

  /**
   * Submit a query to the agent
   */
  const submitQuery = async () => {
    if (!question.trim()) {
      toast.error('Please enter a question');
      return;
    }

    if (!agentReady) {
      toast.error('Agent is not ready');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const request: AgentRequest = createAgentRequest(question, {
        language,
        max_iterations: maxIterations,
        agent_model: agentModel,
        enable_parallel: enableParallel,
        top_k_per_query: topKPerQuery,
        num_kg_in_context: numKgInContext,
        num_chunks_in_context: numChunksInContext,
        enable_reranking: enableReranking,
        enable_variable_storage: enableVariableStorage,
        confidence_threshold: confidenceThreshold,
      });

      console.log('[Agent] Submitting query:', request);

      const result = await queryAgent(request);

      console.log('[Agent] Response:', result);

      setResponse(result);
      toast.success('Query completed successfully!');
    } catch (err: any) {
      console.error('[Agent] Query failed:', err);
      console.error('[Agent] Error response:', err.response);

      // Handle FastAPI validation errors (422)
      let errorMessage = 'Failed to query agent';

      if (err.response?.status === 422) {
        // Pydantic validation error
        const validationErrors = err.response?.data?.detail;
        if (Array.isArray(validationErrors)) {
          errorMessage = 'Validation Error: ' + validationErrors.map((e: any) =>
            `${e.loc?.join('.')} - ${e.msg}`
          ).join(', ');
        } else if (typeof validationErrors === 'string') {
          errorMessage = validationErrors;
        } else {
          errorMessage = 'Request validation failed. Check console for details.';
        }
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
      toast.error(errorMessage, { duration: 5000 });
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Reset form and response
   */
  const reset = () => {
    setQuestion('');
    setResponse(null);
    setError(null);
  };

  /**
   * Reset parameters to default values
   */
  const resetParameters = () => {
    setLanguage('auto');
    setMaxIterations(3);
    setAgentModel('gpt-4o');
    setEnableParallel(true);
    setTopKPerQuery(60);
    setNumKgInContext(15);
    setNumChunksInContext(5);
    setEnableReranking(false);
    setEnableVariableStorage(true);
    setConfidenceThreshold(0.8);
    toast.success('Parameters reset to defaults');
  };

  return {
    // Agent status
    agentReady,
    agentInfo,
    healthStatus,
    loadAgentStatus,

    // Request parameters
    question,
    setQuestion,
    language,
    setLanguage,
    maxIterations,
    setMaxIterations,
    agentModel,
    setAgentModel,
    enableParallel,
    setEnableParallel,
    topKPerQuery,
    setTopKPerQuery,
    numKgInContext,
    setNumKgInContext,
    numChunksInContext,
    setNumChunksInContext,
    enableReranking,
    setEnableReranking,
    enableVariableStorage,
    setEnableVariableStorage,
    confidenceThreshold,
    setConfidenceThreshold,

    // Response state
    response,
    isLoading,
    error,

    // Actions
    submitQuery,
    reset,
    resetParameters,
  };
}
