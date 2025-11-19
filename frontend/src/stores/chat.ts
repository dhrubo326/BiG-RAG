import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { ChatMessage, RetrievedContext, QueryParams } from '../types';
import { DEFAULT_CONFIG } from '../utils/constants';
import { generateId } from '../utils/formatters';

interface ChatState {
  // Data
  messages: ChatMessage[];
  retrievedContexts: RetrievedContext[];
  currentThinking: string | null;

  // UI State
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;

  // Settings
  model: string;
  temperature: number;
  topK: number;
  numKgInContext: number;
  numChunksInContext: number;
  enableReranking: boolean;
  queryMode: QueryParams['mode'];

  // Actions
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  updateMessage: (id: string, content: string) => void;
  deleteMessage: (id: string) => void;
  clearMessages: () => void;
  setRetrievedContexts: (contexts: RetrievedContext[]) => void;
  setThinking: (thinking: string | null) => void;
  setLoading: (loading: boolean) => void;
  setStreaming: (streaming: boolean) => void;
  setError: (error: string | null) => void;
  setModel: (model: string) => void;
  setTemperature: (temperature: number) => void;
  setTopK: (topK: number) => void;
  setNumKgInContext: (count: number) => void;
  setNumChunksInContext: (count: number) => void;
  setEnableReranking: (enable: boolean) => void;
  setQueryMode: (mode: QueryParams['mode']) => void;
  resetSettings: () => void;
}

const useChatStore = create<ChatState>()(
  devtools(
    persist(
      (set) => ({
        // Initial state
        messages: [],
        retrievedContexts: [],
        currentThinking: null,
        isLoading: false,
        isStreaming: false,
        error: null,
        model: DEFAULT_CONFIG.MODEL,
        temperature: DEFAULT_CONFIG.TEMPERATURE,
        topK: DEFAULT_CONFIG.TOP_K,
        numKgInContext: DEFAULT_CONFIG.NUM_KG_IN_CONTEXT,
        numChunksInContext: DEFAULT_CONFIG.NUM_CHUNKS_IN_CONTEXT,
        enableReranking: DEFAULT_CONFIG.ENABLE_RERANKING,
        queryMode: DEFAULT_CONFIG.QUERY_MODE,

        // Actions
        addMessage: (message) =>
          set((state) => ({
            messages: [
              ...state.messages,
              {
                ...message,
                id: generateId(),
                timestamp: new Date().toISOString(),
              },
            ],
          })),

        updateMessage: (id, content) =>
          set((state) => ({
            messages: state.messages.map((msg) =>
              msg.id === id ? { ...msg, content } : msg
            ),
          })),

        deleteMessage: (id) =>
          set((state) => ({
            messages: state.messages.filter((msg) => msg.id !== id),
          })),

        clearMessages: () =>
          set({
            messages: [],
            retrievedContexts: [],
            currentThinking: null,
            error: null,
          }),

        setRetrievedContexts: (contexts) =>
          set({ retrievedContexts: contexts }),

        setThinking: (thinking) => set({ currentThinking: thinking }),

        setLoading: (loading) => set({ isLoading: loading }),

        setStreaming: (streaming) => set({ isStreaming: streaming }),

        setError: (error) => set({ error }),

        setModel: (model) => set({ model }),

        setTemperature: (temperature) =>
          set({ temperature: Math.max(0, Math.min(1, temperature)) }),

        setTopK: (topK) => set({ topK: Math.max(1, Math.min(100, topK)) }),

        setNumKgInContext: (count) => set({ numKgInContext: Math.max(1, Math.min(30, count)) }),

        setNumChunksInContext: (count) => set({ numChunksInContext: Math.max(0, Math.min(20, count)) }),

        setEnableReranking: (enable) => set({ enableReranking: enable }),

        setQueryMode: (mode) => set({ queryMode: mode }),

        resetSettings: () =>
          set({
            model: DEFAULT_CONFIG.MODEL,
            temperature: DEFAULT_CONFIG.TEMPERATURE,
            topK: DEFAULT_CONFIG.TOP_K,
            numKgInContext: DEFAULT_CONFIG.NUM_KG_IN_CONTEXT,
            numChunksInContext: DEFAULT_CONFIG.NUM_CHUNKS_IN_CONTEXT,
            enableReranking: DEFAULT_CONFIG.ENABLE_RERANKING,
            queryMode: DEFAULT_CONFIG.QUERY_MODE,
          }),
      }),
      {
        name: 'chat-store',
        partialize: (state) => ({
          // Only persist messages and settings, not UI state
          messages: state.messages.slice(-20), // Keep last 20 messages
          model: state.model,
          temperature: state.temperature,
          topK: state.topK,
          numKgInContext: state.numKgInContext,
          numChunksInContext: state.numChunksInContext,
          enableReranking: state.enableReranking,
          queryMode: state.queryMode,
        }),
      }
    ),
    {
      name: 'chat-store',
    }
  )
);

export default useChatStore;