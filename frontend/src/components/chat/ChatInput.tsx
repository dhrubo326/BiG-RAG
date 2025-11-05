import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Settings, StopCircle } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  onCancel?: () => void;
  isLoading?: boolean;
  isStreaming?: boolean;
  placeholder?: string;
  suggestions?: string[];
  onSettingsClick?: () => void;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onCancel,
  isLoading = false,
  isStreaming = false,
  placeholder = 'Type your question...',
  suggestions = [],
  onSettingsClick,
}) => {
  const [message, setMessage] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (message.trim() && !isLoading && !isStreaming) {
      onSend(message.trim());
      setMessage('');
      setShowSuggestions(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion);
    setShowSuggestions(false);
    textareaRef.current?.focus();
  };

  const isDisabled = isLoading || isStreaming;

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
      {/* Suggestions */}
      {suggestions.length > 0 && !message && (
        <div className="px-4 pt-3">
          <button
            onClick={() => setShowSuggestions(!showSuggestions)}
            className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            {showSuggestions ? 'Hide' : 'Show'} suggestions
          </button>
          {showSuggestions && (
            <div className="mt-2 flex flex-wrap gap-2">
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="text-xs px-3 py-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-4">
        <div className="flex items-end gap-2">
          {/* Attachment button (placeholder) */}
          <button
            type="button"
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
            title="Attach file (coming soon)"
            disabled
          >
            <Paperclip className="w-5 h-5" />
          </button>

          {/* Textarea */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={isDisabled}
              rows={1}
              className="w-full px-4 py-2 resize-none max-h-32 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>

          {/* Settings button */}
          {onSettingsClick && (
            <button
              type="button"
              onClick={onSettingsClick}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title="Chat settings"
            >
              <Settings className="w-5 h-5" />
            </button>
          )}

          {/* Send/Cancel button */}
          {isStreaming && onCancel ? (
            <button
              type="button"
              onClick={onCancel}
              className="p-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              title="Stop generation"
            >
              <StopCircle className="w-5 h-5" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={isDisabled || !message.trim()}
              className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Send message"
            >
              <Send className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Character count */}
        <div className="mt-1 text-xs text-gray-500 text-right">
          {message.length} / 4000
        </div>
      </form>
    </div>
  );
};

export default ChatInput;