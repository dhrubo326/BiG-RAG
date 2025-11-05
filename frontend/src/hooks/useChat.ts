import { useCallback, useRef } from 'react';
import useChatStore from '../stores/chat';
import { askQuestion, streamChat } from '../services/chat';
import type { QueryParams } from '../types';
import { toast } from 'sonner';

export const useChat = () => {
  const {
    messages,
    retrievedContexts,
    currentThinking,
    isLoading,
    isStreaming,
    error,
    model,
    temperature,
    topK,
    enableReranking,
    queryMode,
    addMessage,
    updateMessage,
    deleteMessage,
    clearMessages,
    setRetrievedContexts,
    setThinking,
    setLoading,
    setStreaming,
    setError,
    setModel,
    setTemperature,
    setTopK,
    setEnableReranking,
    setQueryMode,
    resetSettings,
  } = useChatStore();

  const abortControllerRef = useRef<AbortController | null>(null);

  // Send a message to the chat API
  const sendMessage = useCallback(
    async (content: string) => {
      // Validate input
      if (!content.trim()) {
        toast.warning('Please enter a message');
        return;
      }

      // Add user message
      addMessage({
        role: 'user',
        content,
      });

      // Clear previous contexts
      setRetrievedContexts([]);
      setThinking(null);
      setError(null);

      // Prepare query parameters
      const params: QueryParams = {
        query: content,
        top_k: topK,
        mode: queryMode,
        enable_reranking: enableReranking,
        model,
        temperature,
        max_tokens: 4096,
      };

      try {
        setLoading(true);

        // Call the API
        const response = await askQuestion(params);

        // Set retrieved contexts
        if (response.contexts && response.contexts.length > 0) {
          setRetrievedContexts(response.contexts);
        }

        // Set thinking process if available
        if (response.thinking) {
          setThinking(response.thinking);
        }

        // Add assistant message
        addMessage({
          role: 'assistant',
          content: response.answer,
          metadata: {
            model,
            temperature,
            retrieval_count: response.contexts?.length || 0,
            thinking: response.thinking,
          },
        });
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Failed to get response';
        setError(errorMessage);
        toast.error(errorMessage);

        // Add error message to chat
        addMessage({
          role: 'assistant',
          content: `Error: ${errorMessage}`,
        });
      } finally {
        setLoading(false);
      }
    },
    [
      messages,
      model,
      temperature,
      topK,
      enableReranking,
      queryMode,
      addMessage,
      setRetrievedContexts,
      setThinking,
      setLoading,
      setError,
    ]
  );

  // Stream chat response (for future implementation)
  const streamMessage = useCallback(
    async (content: string) => {
      if (!content.trim()) {
        toast.warning('Please enter a message');
        return;
      }

      // Create abort controller for canceling stream
      abortControllerRef.current = new AbortController();

      // Add user message
      const userMessageId = Date.now().toString();
      addMessage({
        role: 'user',
        content,
      });

      // Add placeholder for assistant message
      const assistantMessageId = (Date.now() + 1).toString();
      addMessage({
        role: 'assistant',
        content: '',
      });

      // Clear previous contexts
      setRetrievedContexts([]);
      setThinking(null);
      setError(null);

      try {
        setStreaming(true);

        // Stream the response
        await streamChat(
          {
            query: content,
            top_k: topK,
            mode: queryMode,
            enable_reranking: enableReranking,
            model,
            temperature,
          },
          (chunk: string) => {
            // Update assistant message with streamed content
            updateMessage(assistantMessageId, chunk);
          },
          (contexts) => {
            setRetrievedContexts(contexts);
          },
          abortControllerRef.current.signal
        );
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          toast.info('Stream canceled');
        } else {
          const errorMessage =
            err instanceof Error ? err.message : 'Failed to stream response';
          setError(errorMessage);
          toast.error(errorMessage);
          updateMessage(assistantMessageId, `Error: ${errorMessage}`);
        }
      } finally {
        setStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [
      model,
      temperature,
      topK,
      enableReranking,
      queryMode,
      addMessage,
      updateMessage,
      setRetrievedContexts,
      setThinking,
      setStreaming,
      setError,
    ]
  );

  // Cancel streaming
  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setStreaming(false);
    }
  }, [setStreaming]);

  // Regenerate last response
  const regenerateLastResponse = useCallback(async () => {
    // Find the last user message
    const lastUserMessage = [...messages]
      .reverse()
      .find((msg) => msg.role === 'user');

    if (!lastUserMessage) {
      toast.warning('No message to regenerate');
      return;
    }

    // Remove the last assistant message if it exists
    const lastAssistantIndex = messages.findLastIndex(
      (msg) => msg.role === 'assistant'
    );
    if (lastAssistantIndex > -1) {
      const lastAssistantMessage = messages[lastAssistantIndex];
      if (lastAssistantMessage) {
        deleteMessage(lastAssistantMessage.id);
      }
    }

    // Resend the last user message
    await sendMessage(lastUserMessage.content);
  }, [messages, deleteMessage, sendMessage]);

  // Export chat history
  const exportChatHistory = useCallback(() => {
    const exportData = {
      messages,
      settings: {
        model,
        temperature,
        topK,
        enableReranking,
        queryMode,
      },
      exportedAt: new Date().toISOString(),
    };

    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `chat-history-${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);
    toast.success('Chat history exported');
  }, [messages, model, temperature, topK, enableReranking, queryMode]);

  // Import chat history
  const importChatHistory = useCallback(
    (file: File) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);

          if (data.messages && Array.isArray(data.messages)) {
            clearMessages();
            data.messages.forEach((msg: any) => {
              addMessage(msg);
            });

            if (data.settings) {
              if (data.settings.model) setModel(data.settings.model);
              if (data.settings.temperature !== undefined) {
                setTemperature(data.settings.temperature);
              }
              if (data.settings.topK !== undefined) setTopK(data.settings.topK);
              if (data.settings.enableReranking !== undefined) {
                setEnableReranking(data.settings.enableReranking);
              }
              if (data.settings.queryMode) setQueryMode(data.settings.queryMode);
            }

            toast.success('Chat history imported');
          } else {
            throw new Error('Invalid chat history format');
          }
        } catch (err) {
          toast.error('Failed to import chat history');
        }
      };

      reader.readAsText(file);
    },
    [
      clearMessages,
      addMessage,
      setModel,
      setTemperature,
      setTopK,
      setEnableReranking,
      setQueryMode,
    ]
  );

  // Get suggested questions based on context
  const getSuggestedQuestions = useCallback(() => {
    // This could be enhanced with AI-generated suggestions
    return [
      'What are the main entities in the knowledge graph?',
      'How many documents are indexed?',
      'Can you explain the retrieval process?',
      'What types of questions can I ask?',
      'Show me information about a specific topic',
    ];
  }, []);

  return {
    // State
    messages,
    retrievedContexts,
    currentThinking,
    isLoading,
    isStreaming,
    error,
    model,
    temperature,
    topK,
    enableReranking,
    queryMode,

    // Actions
    sendMessage,
    streamMessage,
    cancelStream,
    regenerateLastResponse,
    deleteMessage,
    clearMessages,
    exportChatHistory,
    importChatHistory,
    getSuggestedQuestions,

    // Settings
    setModel,
    setTemperature,
    setTopK,
    setEnableReranking,
    setQueryMode,
    resetSettings,
  };
};