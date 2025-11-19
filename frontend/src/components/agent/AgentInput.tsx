/**
 * AgentInput Component
 *
 * Input form for the multi-hop reasoning agent.
 * Includes question textarea and advanced settings panel.
 */

import { useState } from 'react';
import { ChevronDown, ChevronUp, RotateCcw, Send } from 'lucide-react';
import { Button } from '../ui/button';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { Input } from '../ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';
import { Slider } from '../ui/slider';

interface AgentInputProps {
  // State
  question: string;
  language: string;
  maxIterations: number;
  agentModel: string;
  enableParallel: boolean;
  topKPerQuery: number;
  numKgInContext: number;
  numChunksInContext: number;
  enableReranking: boolean;
  confidenceThreshold: number;
  isLoading: boolean;
  agentReady: boolean;

  // Setters
  setQuestion: (value: string) => void;
  setLanguage: (value: string) => void;
  setMaxIterations: (value: number) => void;
  setAgentModel: (value: string) => void;
  setEnableParallel: (value: boolean) => void;
  setTopKPerQuery: (value: number) => void;
  setNumKgInContext: (value: number) => void;
  setNumChunksInContext: (value: number) => void;
  setEnableReranking: (value: boolean) => void;
  setConfidenceThreshold: (value: number) => void;

  // Actions
  onSubmit: () => void;
  onReset: () => void;
}

export function AgentInput(props: AgentInputProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    props.onSubmit();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Question Input */}
      <div>
        <Label htmlFor="question">Question</Label>
        <Textarea
          id="question"
          value={props.question}
          onChange={(e) => props.setQuestion(e.target.value)}
          placeholder="Ask a complex question requiring multi-hop reasoning...
Example: Who is the captain of the team that won the 2022 FIFA World Cup?"
          className="mt-1.5 min-h-[120px] resize-none"
          disabled={props.isLoading}
        />
      </div>

      {/* Language Selection */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="language">Language</Label>
          <Select
            value={props.language}
            onValueChange={props.setLanguage}
            disabled={props.isLoading}
          >
            <SelectTrigger id="language" className="mt-1.5">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto-detect</SelectItem>
              <SelectItem value="English">English</SelectItem>
              <SelectItem value="Bangla">Bangla</SelectItem>
              <SelectItem value="Hindi">Hindi</SelectItem>
              <SelectItem value="Arabic">Arabic</SelectItem>
              <SelectItem value="Chinese">Chinese</SelectItem>
              <SelectItem value="Spanish">Spanish</SelectItem>
              <SelectItem value="French">French</SelectItem>
              <SelectItem value="German">German</SelectItem>
              <SelectItem value="Japanese">Japanese</SelectItem>
              <SelectItem value="Korean">Korean</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="agent-model">Agent Model</Label>
          <Select
            value={props.agentModel}
            onValueChange={props.setAgentModel}
            disabled={props.isLoading}
          >
            <SelectTrigger id="agent-model" className="mt-1.5">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpt-4o">GPT-4o (Recommended)</SelectItem>
              <SelectItem value="gpt-4o-mini">GPT-4o Mini (Faster)</SelectItem>
              <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Advanced Settings Toggle */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline"
        >
          {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          Advanced Settings
        </button>
      </div>

      {/* Advanced Settings Panel */}
      {showAdvanced && (
        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg space-y-4 bg-gray-50 dark:bg-gray-800/50">
          {/* Iterations */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <Label htmlFor="max-iterations">Max Iterations</Label>
              <span className="text-sm text-gray-600 dark:text-gray-400">{props.maxIterations}</span>
            </div>
            <Slider
              id="max-iterations"
              min={1}
              max={5}
              step={1}
              value={[props.maxIterations]}
              onValueChange={(value) => props.setMaxIterations(value[0])}
              disabled={props.isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">Number of reasoning iterations (1-5)</p>
          </div>

          {/* Top-K per Query */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <Label htmlFor="top-k">Top-K per Query</Label>
              <span className="text-sm text-gray-600 dark:text-gray-400">{props.topKPerQuery}</span>
            </div>
            <Slider
              id="top-k"
              min={10}
              max={100}
              step={10}
              value={[props.topKPerQuery]}
              onValueChange={(value) => props.setTopKPerQuery(value[0])}
              disabled={props.isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">Items to retrieve from vector DBs (10-100)</p>
          </div>

          {/* Num KG in Context */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <Label htmlFor="num-kg">KG Relations in Context</Label>
              <span className="text-sm text-gray-600 dark:text-gray-400">{props.numKgInContext}</span>
            </div>
            <Slider
              id="num-kg"
              min={1}
              max={30}
              step={1}
              value={[props.numKgInContext]}
              onValueChange={(value) => props.setNumKgInContext(value[0])}
              disabled={props.isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">KG relations in final context (1-30)</p>
          </div>

          {/* Num Chunks in Context */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <Label htmlFor="num-chunks">Text Chunks in Context</Label>
              <span className="text-sm text-gray-600 dark:text-gray-400">{props.numChunksInContext}</span>
            </div>
            <Slider
              id="num-chunks"
              min={0}
              max={20}
              step={1}
              value={[props.numChunksInContext]}
              onValueChange={(value) => props.setNumChunksInContext(value[0])}
              disabled={props.isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">Text chunks in final context (0-20)</p>
          </div>

          {/* Confidence Threshold */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <Label htmlFor="confidence">Confidence Threshold</Label>
              <span className="text-sm text-gray-600 dark:text-gray-400">{props.confidenceThreshold.toFixed(2)}</span>
            </div>
            <Slider
              id="confidence"
              min={0}
              max={1}
              step={0.05}
              value={[props.confidenceThreshold]}
              onValueChange={(value) => props.setConfidenceThreshold(value[0])}
              disabled={props.isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">Early stopping threshold (0.0-1.0)</p>
          </div>

          {/* Checkboxes */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={props.enableReranking}
                onChange={(e) => props.setEnableReranking(e.target.checked)}
                disabled={props.isLoading}
                className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-sm">Enable Semantic Reranking (requires sentence-transformers)</span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={props.enableParallel}
                onChange={(e) => props.setEnableParallel(e.target.checked)}
                disabled={props.isLoading}
                className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-sm">Enable Parallel Query Execution</span>
            </label>
          </div>

          {/* Reset Button */}
          <div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={props.onReset}
              disabled={props.isLoading}
              className="w-full"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset to Defaults
            </Button>
          </div>
        </div>
      )}

      {/* Submit Button */}
      <div>
        <Button
          type="submit"
          disabled={props.isLoading || !props.agentReady || !props.question.trim()}
          className="w-full"
        >
          {props.isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              Processing...
            </>
          ) : (
            <>
              <Send className="w-4 h-4 mr-2" />
              {props.agentReady ? 'Submit Query' : 'Agent Not Ready'}
            </>
          )}
        </Button>
      </div>

      {!props.agentReady && (
        <p className="text-sm text-red-600 dark:text-red-400 text-center">
          Agent is not ready. Make sure OpenAI API key is configured in settings.
        </p>
      )}

      {props.agentReady && !props.isLoading && (
        <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
          Note: Agent queries take 3-5 minutes to complete due to multiple LLM iterations.
        </p>
      )}
    </form>
  );
}
