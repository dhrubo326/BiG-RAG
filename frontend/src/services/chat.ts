import api from './api';
import type { QueryParams, QueryResponse, RetrievedContext } from '../types';
import { API_ENDPOINTS } from '../utils/constants';

/**
 * Ask a question and get a response with retrieval
 * Uses /api/unified/chat endpoint for full RAG pipeline (retrieval + answer generation)
 */
export const askQuestion = async (params: QueryParams): Promise<QueryResponse> => {
  // Get active dataset from localStorage (settings store)
  const settingsStore = localStorage.getItem('settings-store');
  let activeDataset = 'demo_test'; // Default fallback

  if (settingsStore) {
    try {
      const settings = JSON.parse(settingsStore);
      activeDataset = settings.state?.activeDataset || 'demo_test';
    } catch (e) {
      console.warn('Failed to parse settings:', e);
    }
  }

  const response = await api.post(API_ENDPOINTS.UNIFIED_CHAT, {
    messages: [
      {
        role: 'user',
        content: params.query,
      },
    ],
    use_rag: true,
    force_subgraphs: [activeDataset], // Use active dataset
    mode: params.mode || 'hybrid',
    top_k: params.top_k || 60,
    enable_reranking: params.enable_reranking !== undefined ? params.enable_reranking : true,

    // LLM parameters
    model: params.model || 'gpt-4o-mini',
    temperature: params.temperature || 0.7,
    max_tokens: params.max_tokens || 4096,
    llm_provider: params.llm_provider || 'openai',
    language: params.language,

    // Output mode: get both answer and contexts
    output_mode: 'answer_with_context',
    include_metadata: true,
  });

  const data = response.data;

  // Map contexts from unified API response
  const contexts: RetrievedContext[] = (data.contexts || []).map((ctx: any, index: number) => ({
    content: ctx.content || '',
    score: ctx.score || 0,
    source: ctx.source || `Source ${index + 1}`,
    type: ctx.type || 'chunk',
    metadata: ctx.metadata || {},
  }));

  return {
    answer: data.answer || '',
    contexts: contexts,
    thinking: undefined,
    metadata: {
      model: data.llm_metrics?.model || params.model || 'unknown',
      retrieval_time: data.retrieval_metrics?.latency_ms || 0,
      generation_time: data.llm_metrics?.latency_ms || 0,
      total_tokens: data.llm_metrics?.total_tokens || 0,
    },
  };
};

/**
 * Stream chat response (for real-time streaming)
 */
export const streamChat = async (
  params: QueryParams,
  onChunk: (chunk: string) => void,
  onContexts?: (contexts: RetrievedContext[]) => void,
  abortSignal?: AbortSignal
): Promise<void> => {
  const response = await fetch(`${api.defaults.baseURL}${API_ENDPOINTS.ASK}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question: params.query,
      top_k: params.top_k,
      mode: params.mode,
      enable_reranking: params.enable_reranking,
      model: params.model,
      temperature: params.temperature,
      stream: true, // Enable streaming
    }),
    signal: abortSignal,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Response body is not readable');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  let fullResponse = '';
  let contexts: RetrievedContext[] = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process lines in buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.trim() === '') continue;

        // Parse SSE format
        if (line.startsWith('data: ')) {
          const data = line.slice(6);

          if (data === '[DONE]') {
            return;
          }

          try {
            const parsed = JSON.parse(data);

            if (parsed.type === 'context') {
              // Retrieved contexts
              contexts = parsed.contexts || [];
              if (onContexts) {
                onContexts(contexts);
              }
            } else if (parsed.type === 'chunk') {
              // Text chunk
              const chunk = parsed.content || '';
              fullResponse += chunk;
              onChunk(fullResponse);
            } else if (parsed.type === 'error') {
              throw new Error(parsed.message || 'Stream error');
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
};

/**
 * Get suggested questions based on current context
 */
export const getSuggestedQuestions = async (
  context?: string,
  limit: number = 5
): Promise<string[]> => {
  try {
    const response = await api.post('/api/suggest', {
      context,
      limit,
    });

    return response.data.suggestions || [];
  } catch {
    // Return default suggestions if API fails
    return [
      'What are the main topics in the knowledge graph?',
      'Can you summarize the key information?',
      'What relationships exist between entities?',
      'Show me recent updates to the knowledge base',
      'What questions can I ask about this data?',
    ];
  }
};

/**
 * Get chat history
 */
export const getChatHistory = async (
  limit: number = 50
): Promise<{ messages: any[]; total: number }> => {
  const response = await api.get('/api/chat/history', {
    params: { limit },
  });

  return {
    messages: response.data.messages || [],
    total: response.data.total || 0,
  };
};

/**
 * Save chat feedback
 */
export const saveFeedback = async (
  messageId: string,
  feedback: 'positive' | 'negative',
  comment?: string
): Promise<void> => {
  await api.post('/api/chat/feedback', {
    message_id: messageId,
    feedback,
    comment,
  });
};