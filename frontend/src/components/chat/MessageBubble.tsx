import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot, Copy, Check, RefreshCw, Trash2, ChevronDown, ChevronUp } from 'lucide-react';
import type { ChatMessage } from '../../types';
import { formatRelativeTime } from '../../utils/formatters';
import { toast } from 'sonner';

interface MessageBubbleProps {
  message: ChatMessage;
  isLast?: boolean;
  onDelete?: (id: string) => void;
  onRegenerate?: (id: string) => void;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isLast = false,
  onDelete,
  onRegenerate,
}) => {
  const [copied, setCopied] = useState(false);
  const [showThinking, setShowThinking] = useState(false);

  const isUser = message.role === 'user';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      toast.success('Copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      toast.error('Failed to copy');
    }
  };

  const handleDelete = () => {
    if (onDelete && window.confirm('Are you sure you want to delete this message?')) {
      onDelete(message.id);
    }
  };

  const handleRegenerate = () => {
    if (onRegenerate) {
      onRegenerate(message.id);
    }
  };

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
        }`}
      >
        {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-3xl ${isUser ? 'text-right' : 'text-left'}`}>
        <div
          className={`inline-block px-4 py-3 rounded-lg ${
            isUser
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
          }`}
        >
          {/* Message text */}
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>

          {/* Thinking process (for assistant messages) */}
          {!isUser && message.metadata?.thinking && (
            <div className="mt-3 pt-3 border-t border-gray-300 dark:border-gray-600">
              <button
                onClick={() => setShowThinking(!showThinking)}
                className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
              >
                {showThinking ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                <span>Thinking process</span>
              </button>
              {showThinking && (
                <div className="mt-2 text-xs text-gray-600 dark:text-gray-400 italic">
                  {message.metadata.thinking}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Message metadata and actions */}
        <div className="mt-1 flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
          <span>{formatRelativeTime(message.timestamp)}</span>

          {message.metadata?.model && (
            <span>• {message.metadata.model}</span>
          )}

          {message.metadata?.retrieval_count !== undefined && message.metadata.retrieval_count > 0 && (
            <span>• {message.metadata.retrieval_count} contexts retrieved</span>
          )}

          {/* Action buttons */}
          <div className={`flex items-center gap-1 ml-2 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
            <button
              onClick={handleCopy}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
              title="Copy message"
            >
              {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            </button>

            {!isUser && isLast && onRegenerate && (
              <button
                onClick={handleRegenerate}
                className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
                title="Regenerate response"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            )}

            {onDelete && (
              <button
                onClick={handleDelete}
                className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded transition-colors text-red-500"
                title="Delete message"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;