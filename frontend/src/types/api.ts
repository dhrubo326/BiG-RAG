// API request and response type definitions

// Base API response wrapper
export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Health check
export interface HealthResponse {
  status: string;
  message: string;
  dataset: string;  // Backend returns "dataset" not "data_source"
  api_version?: string;
  version?: string;
  embedding_mode?: string;
  default_llm_provider?: string;
  available_providers?: string[];
}

// Document types
export interface Document {
  id: string;
  title: string;
  content?: string;
  chunks?: TextChunk[];
  entities?: EntityInfo[];
  relations?: RelationInfo[];
  metadata?: DocumentMetadata;
  created_at?: string;
  updated_at?: string;
}

export interface DocumentMetadata {
  title?: string;
  source?: string;
  category?: string;
  tags?: string[];
  author?: string;
  date?: string;
  url?: string;
}

export interface TextChunk {
  id: string;
  content: string;
  position: number;
  metadata?: Record<string, any>;
}

export interface EntityInfo {
  name: string;
  type: string;
  description?: string;
  weight?: number;
  occurrences?: number;
}

export interface RelationInfo {
  subject: string;
  predicate: string;
  object: string;
  weight?: number;
  source_id?: string;
}

// Chat/Query types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    model?: string;
    temperature?: number;
    retrieval_count?: number;
    thinking?: string;
  };
}

export interface QueryParams {
  query: string;
  top_k?: number;  // Number of items to retrieve from vector DBs (default: 60)
  num_kg_in_context?: number;  // Number of KG items (relations) in final output (default: 15)
  num_chunks_in_context?: number;  // Number of chunks in final output (default: 5)
  mode?: 'local' | 'global' | 'hybrid' | 'naive';
  enable_reranking?: boolean;
  dataset?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  llm_provider?: string;
  language?: string;  // Language override (e.g., "Bangla", "English", "Hindi")
}

export interface QueryResponse {
  answer: string;
  contexts: RetrievedContext[];
  thinking?: string;
  metadata?: {
    model: string;
    retrieval_time: number;
    generation_time: number;
    total_tokens?: number;
  };
}

export interface RetrievedContext {
  content: string;
  score: number;
  source: string;
  type: 'entity' | 'relation' | 'chunk';
  path?: string;
  metadata?: Record<string, any>;
}

// Search types
export interface SearchRequest {
  queries: string[];
  param?: {
    top_k?: number;
    mode?: 'local' | 'global' | 'hybrid' | 'naive';
    enable_reranking?: boolean;
  };
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface SearchResult {
  query: string;
  contexts: string[];
  scores?: number[];
  metadata?: Record<string, any>;
}

// Statistics types
export interface SystemStats {
  documents: number;
  entities: number;
  relations: number;
  chunks: number;
  edges: number;
  dataset: string;
  last_updated?: string;
}

// Evaluation types
export interface EvaluationConfig {
  dataset: string;
  model: string;
  questions?: number[] | 'all';
  top_k: number;
  enable_reranking: boolean;
  temperature?: number;
}

export interface EvaluationResult {
  id: string;
  dataset: string;
  model: string;
  timestamp: string;
  metrics: {
    exact_match: number;
    f1_score: number;
    total_questions: number;
    correct_answers: number;
  };
  details?: QuestionResult[];
}

export interface QuestionResult {
  question: string;
  golden_answer: string;
  predicted_answer: string;
  exact_match: number;
  f1_score: number;
  retrieved_contexts?: string[];
}

// Job/Task types
export interface Job {
  id: string;
  type: 'build' | 'evaluation' | 'export';
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  message?: string;
  created_at: string;
  completed_at?: string;
  result?: any;
}

// Settings types
export interface UserSettings {
  language: 'en' | 'zh';
  theme: 'light' | 'dark' | 'auto';
  autoSave: boolean;
  defaultModel: string;
  defaultTopK: number;
  enableReranking: boolean;
  apiKeys?: {
    openai?: string;
    anthropic?: string;
    google?: string;
    xai?: string;
  };
}

// Dataset types
export interface Dataset {
  name: string;
  description?: string;
  documents: number;
  questions: number;
  built: boolean;
  size?: string;
  last_modified?: string;
}

// Agent types (Multi-hop Reasoning)
export interface AgentRequest {
  question: string;
  language?: string;  // Default: "auto"
  max_iterations?: number;  // Default: 3, range: 1-5
  agent_model?: string;  // Default: "gpt-4o"
  enable_parallel?: boolean;  // Default: true

  // BiG-RAG retrieval parameters
  top_k_per_query?: number;  // Default: 60, range: 10-100
  num_kg_in_context?: number;  // Default: 15, range: 1-30
  num_chunks_in_context?: number;  // Default: 5, range: 0-20
  enable_reranking?: boolean;  // Default: false

  // Advanced options
  enable_variable_storage?: boolean;  // Default: true
  confidence_threshold?: number;  // Default: 0.8, range: 0.0-1.0
  data_source?: string;  // Optional dataset override
}

export interface AgentResponse {
  answer: string;
  reasoning_trace: ReasoningStep[];
  total_iterations: number;
  contexts_used: AgentContextItem[];
  metadata: AgentMetadata;
  confidence: number;  // 0.0-1.0
  limitations?: string;
  variable_X?: Record<string, any>;  // NEW: Accumulated knowledge (simplified agent only, for debugging)
}

export interface ReasoningStep {
  step: number;  // 1-indexed
  thought: string;
  planned_queries: PlannedQuery[];
  executed_actions: ExecutedAction[];
  observations: AgentObservation[];
  variables_stored: Record<string, any>;
  confidence: number;  // 0.0-1.0
  execution_time_ms: number;
}

export interface PlannedQuery {
  query: string;
  language: string;
  reason: string;
}

export interface ExecutedAction {
  action_type: string;  // "search_bigrag" | "search_bigrag_skipped" | "search_bigrag_error"
  query: string;
  language: string;
  num_results: number;
  execution_time_ms: number;
}

export interface AgentObservation {
  query: string;
  contexts: AgentContextItem[];
  summary?: string;
}

export interface AgentContextItem {
  text: string;
  source?: string;
  metadata: Record<string, any>;
  relevance_score?: number;
}

export interface AgentMetadata {
  model_used: string;
  total_tokens: number;
  total_cost_usd: number;
  execution_time_ms: number;
  queries_executed: number;
  stopped_reason: 'max_iterations' | 'high_confidence' | 'complete';
}

export interface AgentHealthResponse {
  status: 'ready' | 'not_ready';
  message: string;
  ready: boolean;
  model?: string;
}

export interface AgentInfo {
  name: string;
  version: string;
  description: string;
  capabilities: string[];
  supported_languages: string[];
  max_iterations: number;
  default_model: string;
}