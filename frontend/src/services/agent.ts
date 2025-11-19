/**
 * Agent API Service
 *
 * Handles API calls to the multi-hop reasoning agent endpoints.
 */

import api from './api';
import type {
  AgentRequest,
  AgentResponse,
  AgentHealthResponse,
  AgentInfo,
} from '../types/api';

/**
 * Query the agent with a question.
 *
 * @param request Agent request parameters
 * @returns Agent response with answer and reasoning trace
 */
export async function queryAgent(request: AgentRequest): Promise<AgentResponse> {
  // Log the exact request being sent
  console.log('[AgentService] Sending request to /agent/query:');
  console.log(JSON.stringify(request, null, 2));

  const response = await api.post<AgentResponse>('/agent/query', request);
  return response.data;
}

/**
 * Check if the agent is ready.
 *
 * @returns Agent health status
 */
export async function getAgentHealth(): Promise<AgentHealthResponse> {
  const response = await api.get<AgentHealthResponse>('/agent/health');
  return response.data;
}

/**
 * Get agent information and capabilities.
 *
 * @returns Agent info
 */
export async function getAgentInfo(): Promise<AgentInfo> {
  const response = await api.get<AgentInfo>('/agent/info');
  return response.data;
}

/**
 * Create a default agent request with common parameters.
 *
 * @param question The question to ask
 * @param overrides Optional parameter overrides
 * @returns Agent request object
 */
export function createAgentRequest(
  question: string,
  overrides?: Partial<AgentRequest>
): AgentRequest {
  return {
    question,
    language: 'auto',
    max_iterations: 3,
    agent_model: 'gpt-4o',
    enable_parallel: true,
    top_k_per_query: 60,
    num_kg_in_context: 15,
    num_chunks_in_context: 5,
    enable_reranking: false,
    enable_variable_storage: true,
    confidence_threshold: 0.8,
    ...overrides,
  };
}
