import api from './api';
import type { QueryParams, QueryResponse, RetrievedContext } from '../types';
import { API_ENDPOINTS } from '../utils/constants';

/**
 * Ask a question and get a response with retrieval
 * Uses /chat/completions endpoint for full RAG pipeline (retrieval + answer generation)
 */
export const askQuestion = async (params: QueryParams): Promise<QueryResponse> => {
  const response = await api.post(API_ENDPOINTS.CHAT_COMPLETIONS, {
    model: params.model || 'gpt-4o-mini',
    messages: [
      {
        role: 'user',
        content: params.query,
      },
    ],
    temperature: params.temperature || 0.7,
    max_tokens: params.max_tokens || 4096,
    llm_provider: params.llm_provider,
    use_rag: true, // Enable RAG for knowledge retrieval
    enable_reranking: params.enable_reranking,
    top_k: params.top_k || 60,  // Retrieval count (default: 60)
    num_kg_in_context: params.num_kg_in_context || 15,  // KG output count (default: 15)
    num_chunks_in_context: params.num_chunks_in_context || 5,  // Chunk output count (default: 5)
    mode: params.mode || 'hybrid',  // Query mode (default: hybrid)
    language: params.language,  // Language override (optional)
  });

  // Transform OpenAI-compatible response to our QueryResponse type
  const data = response.data;
  const content = data.choices?.[0]?.message?.content || '';

  // Since /chat/completions returns the final answer, we need to extract contexts separately
  // For now, we'll use a separate call to /ask for contexts if needed
  let contexts: any[] = [];

  // Optionally fetch contexts for display (non-blocking)
  try {
    const contextResponse = await api.post(API_ENDPOINTS.ASK, {
      question: params.query,
      top_k: params.top_k || 60,  // Match chat/completions defaults
      num_kg_in_context: params.num_kg_in_context || 15,
      num_chunks_in_context: params.num_chunks_in_context || 5,
      mode: params.mode || 'hybrid',
      enable_reranking: params.enable_reranking,
      language: params.language,  // Pass language parameter
    });

    // Map retrieved_contexts from backend to our RetrievedContext type
    const retrievedContexts = contextResponse.data.retrieved_contexts || [];
    contexts = retrievedContexts.map((ctx: any, index: number) => ({
      content: ctx.context || '',
      score: ctx.coherence_score || 0,
      source: `Source ${ctx.rank || index + 1}`,
      type: ctx.type || 'chunk',  // Use actual type from backend (entity/relation/chunk)
      metadata: {
        rank: ctx.rank || index + 1,
        ...ctx.metadata,
      },
    }));
  } catch (err) {
    console.warn('Failed to fetch contexts for display:', err);
  }

  return {
    answer: content,
    contexts: contexts,
    thinking: undefined,
    metadata: {
      model: data.model || params.model || 'unknown',
      retrieval_time: 0,
      generation_time: 0,
      total_tokens: data.usage?.total_tokens || 0,
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