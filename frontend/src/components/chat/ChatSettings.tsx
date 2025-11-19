import { X } from 'lucide-react';
import { AVAILABLE_MODELS, QUERY_MODES } from '../../utils/constants';

interface ChatSettingsProps {
  model: string;
  temperature: number;
  topK: number;
  numKgInContext: number;
  numChunksInContext: number;
  enableReranking: boolean;
  queryMode: string;
  onModelChange: (model: string) => void;
  onTemperatureChange: (temp: number) => void;
  onTopKChange: (topK: number) => void;
  onNumKgInContextChange: (count: number) => void;
  onNumChunksInContextChange: (count: number) => void;
  onRerankingChange: (enable: boolean) => void;
  onQueryModeChange: (mode: any) => void;
  onReset: () => void;
  onClose: () => void;
}

const ChatSettings: React.FC<ChatSettingsProps> = ({
  model,
  temperature,
  topK,
  numKgInContext,
  numChunksInContext,
  enableReranking,
  queryMode,
  onModelChange,
  onTemperatureChange,
  onTopKChange,
  onNumKgInContextChange,
  onNumChunksInContextChange,
  onRerankingChange,
  onQueryModeChange,
  onReset,
  onClose,
}) => {
  return (
    <div className="absolute top-0 right-0 w-80 h-full bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-lg z-20">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold">Chat Settings</h3>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Settings */}
      <div className="p-4 space-y-6 overflow-y-auto h-[calc(100%-4rem)]">
        {/* Model Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Model
          </label>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
          >
            {AVAILABLE_MODELS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Temperature */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Temperature: {temperature.toFixed(1)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={temperature}
            onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Precise (0.0)</span>
            <span>Balanced (0.5)</span>
            <span>Creative (1.0)</span>
          </div>
        </div>

        {/* Top-K Retrieval (from vector DBs) */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Top-K Retrieval: {topK}
          </label>
          <input
            type="range"
            min="10"
            max="100"
            step="10"
            value={topK}
            onChange={(e) => onTopKChange(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>10</span>
            <span>50</span>
            <span>100</span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Items to retrieve from vector databases
          </p>
        </div>

        {/* KG Context Count */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            KG Context: {numKgInContext}
          </label>
          <input
            type="range"
            min="5"
            max="30"
            step="5"
            value={numKgInContext}
            onChange={(e) => onNumKgInContextChange(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>5</span>
            <span>15</span>
            <span>30</span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Knowledge graph relations in final context
          </p>
        </div>

        {/* Chunk Context Count */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Chunk Context: {numChunksInContext}
          </label>
          <input
            type="range"
            min="0"
            max="20"
            step="5"
            value={numChunksInContext}
            onChange={(e) => onNumChunksInContextChange(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0</span>
            <span>10</span>
            <span>20</span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Text chunks in final context
          </p>
        </div>

        {/* Query Mode */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Query Mode
          </label>
          <select
            value={queryMode}
            onChange={(e) => onQueryModeChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
          >
            {QUERY_MODES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Enable Reranking */}
        <div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={enableReranking}
              onChange={(e) => onRerankingChange(e.target.checked)}
              className="w-4 h-4 text-blue-500 border-gray-300 rounded focus:ring-blue-500"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Enable Semantic Reranking
            </span>
          </label>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Improves relevance but increases latency
          </p>
        </div>

        {/* Reset Button */}
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={onReset}
            className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
          >
            Reset to Defaults
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatSettings;