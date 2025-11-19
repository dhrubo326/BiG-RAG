/**
 * Chat Page
 *
 * Interactive chat interface for:
 * - Querying the knowledge graph
 * - Real-time retrieval with BiG-RAG
 * - Viewing retrieved context
 * - Conversation history
 */

import { useState } from 'react';
import { useChat } from '../hooks/useChat';
import ChatWindow from '../components/chat/ChatWindow';
import ChatInput from '../components/chat/ChatInput';
import RetrievalViz from '../components/chat/RetrievalViz';
import ChatSettings from '../components/chat/ChatSettings';
import { Download, Upload, Trash2, Settings2 } from 'lucide-react';
import { toast } from 'sonner';

export function Chat() {
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
    sendMessage,
    // streamMessage, // For future streaming implementation
    cancelStream,
    regenerateLastResponse,
    deleteMessage,
    clearMessages,
    exportChatHistory,
    importChatHistory,
    getSuggestedQuestions,
    setModel,
    setTemperature,
    setTopK,
    setEnableReranking,
    setQueryMode,
    resetSettings,
  } = useChat();

  const [showSettings, setShowSettings] = useState(false);
  const [showRetrieval, setShowRetrieval] = useState(true);

  const suggestions = getSuggestedQuestions();

  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        importChatHistory(file);
      }
    };
    input.click();
  };

  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear all messages?')) {
      clearMessages();
      toast.success('Chat history cleared');
    }
  };

  const handleRegenerate = async () => {
    await regenerateLastResponse();
  };

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Chat with BiG-RAG</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Ask questions about your knowledge graph
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowRetrieval(!showRetrieval)}
              className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              {showRetrieval ? 'Hide' : 'Show'} Context
            </button>
            <button
              onClick={exportChatHistory}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Export chat"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={handleImport}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Import chat"
            >
              <Upload className="w-4 h-4" />
            </button>
            <button
              onClick={handleClearChat}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors text-red-500"
              title="Clear chat"
              disabled={messages.length === 0}
            >
              <Trash2 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Settings"
            >
              <Settings2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Chat Section */}
        <div className={`flex-1 flex flex-col ${showRetrieval ? 'mr-96' : ''}`}>
          {/* Messages */}
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            isStreaming={isStreaming}
            onDeleteMessage={deleteMessage}
            onRegenerateMessage={handleRegenerate}
          />

          {/* Error Display */}
          {error && (
            <div className="px-4 py-2 bg-red-100 dark:bg-red-900/50 text-red-800 dark:text-red-200 text-sm">
              {error}
            </div>
          )}

          {/* Input */}
          <ChatInput
            onSend={sendMessage}
            onCancel={isStreaming ? cancelStream : undefined}
            isLoading={isLoading}
            isStreaming={isStreaming}
            suggestions={messages.length === 0 ? suggestions : []}
            onSettingsClick={() => setShowSettings(!showSettings)}
          />
        </div>

        {/* Retrieval Panel - Fixed position within parent, stays visible when scrolling */}
        {showRetrieval && (
          <div className="absolute top-0 right-0 w-96 h-full border-l border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 overflow-hidden flex flex-col">
            <RetrievalViz
              contexts={retrievedContexts}
              thinking={currentThinking}
              isLoading={isLoading}
            />
          </div>
        )}

        {/* Settings Panel */}
        {showSettings && (
          <ChatSettings
            model={model}
            temperature={temperature}
            topK={topK}
            enableReranking={enableReranking}
            queryMode={queryMode || 'hybrid'}
            onModelChange={setModel}
            onTemperatureChange={setTemperature}
            onTopKChange={setTopK}
            onRerankingChange={setEnableReranking}
            onQueryModeChange={setQueryMode}
            onReset={resetSettings}
            onClose={() => setShowSettings(false)}
          />
        )}
      </div>
    </div>
  );
}
