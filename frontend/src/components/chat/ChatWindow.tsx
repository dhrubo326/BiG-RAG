import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import type { ChatMessage } from '../../types';

interface ChatWindowProps {
  messages: ChatMessage[];
  isLoading: boolean;
  isStreaming: boolean;
  onDeleteMessage?: (id: string) => void;
  onRegenerateMessage?: (id: string) => void;
}

const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  isStreaming,
  onDeleteMessage,
  onRegenerateMessage,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* Welcome message if no messages */}
      {messages.length === 0 && (
        <div className="text-center py-12">
          <h3 className="text-xl font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Welcome to BiG-RAG Chat
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            Ask questions about your knowledge graph and documents.
          </p>
        </div>
      )}

      {/* Messages */}
      {messages.map((message, index) => (
        <MessageBubble
          key={message.id}
          message={message}
          isLast={index === messages.length - 1}
          onDelete={onDeleteMessage}
          onRegenerate={onRegenerateMessage}
        />
      ))}

      {/* Loading indicator */}
      {isLoading && !isStreaming && (
        <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <div className="flex gap-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
          <span className="text-sm">Thinking...</span>
        </div>
      )}

      {/* Streaming indicator */}
      {isStreaming && (
        <div className="flex items-center gap-2 text-blue-500">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          <span className="text-sm">Generating response...</span>
        </div>
      )}

      {/* Auto-scroll anchor */}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatWindow;