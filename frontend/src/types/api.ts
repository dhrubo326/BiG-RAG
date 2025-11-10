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
  top_k?: number;
  mode?: 'local' | 'global' | 'hybrid' | 'naive';
  enable_reranking?: boolean;
  dataset?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  llm_provider?: string;
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