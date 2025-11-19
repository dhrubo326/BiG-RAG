// Application constants

// API endpoints
export const API_ENDPOINTS = {
  // Health & Stats
  HEALTH: '/',
  STATS: '/stats',

  // Chat & Search
  ASK: '/ask',
  CHAT_COMPLETIONS: '/chat/completions',
  SEARCH: '/search',

  // Documents
  DOCUMENTS: '/documents',
  DOCUMENT_BY_ID: (id: string) => `/documents/${id}`,
  UPLOAD_DOCUMENT: '/documents/upload',
  DELETE_DOCUMENT: (id: string) => `/documents/${id}`,

  // Evaluation
  EVAL_GENERATE: '/eval/batch_generate',
  EVAL_RESULTS: '/eval/evaluate_results',

  // Graph
  GRAPH_EXPORT: '/graph/export',
  GRAPH_SUBGRAPH: '/graph/subgraph',

  // Jobs
  JOBS: '/jobs',
  JOB_BY_ID: (id: string) => `/jobs/${id}`,
} as const;

// Graph visualization colors
export const GRAPH_COLORS = {
  entity: '#3B82F6', // Blue
  relation: '#EF4444', // Red
  chunk: '#10B981', // Green
  document: '#8B5CF6', // Purple
  selected: '#FFA500', // Orange
  hover: '#FFD700', // Gold
} as const;

// Default configuration
export const DEFAULT_CONFIG = {
  TOP_K: 60, // Number of items to retrieve from vector DBs
  NUM_KG_IN_CONTEXT: 15, // Number of KG items (relations) in final context
  NUM_CHUNKS_IN_CONTEXT: 5, // Number of chunks in final context
  MAX_RESPONSE_LENGTH: 4096,
  MAX_PROMPT_LENGTH: 4096,
  TEMPERATURE: 0.7,
  MODEL: 'gpt-4o-mini',
  QUERY_MODE: 'hybrid' as const,
  ENABLE_RERANKING: false, // Default: false (requires sentence-transformers package)
  GRAPH_LAYOUT: 'cose-bilkent' as const,
  GRAPH_MIN_WEIGHT: 0.0, // ✅ Changed from 0.1 to 0.0 to show ALL nodes including low-weight orphans
} as const;

// UI Constants
export const UI_CONFIG = {
  TOAST_DURATION: 4000,
  DEBOUNCE_DELAY: 300,
  PAGINATION_SIZE: 20,
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  ALLOWED_FILE_TYPES: ['.txt', '.pdf', '.md', '.json', '.jsonl'],
  GRAPH_MAX_NODES: 1000,
  CHAT_MAX_MESSAGES: 100,
} as const;

// Models available for selection
export const AVAILABLE_MODELS = [
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
  { value: 'gpt-4o', label: 'GPT-4o' },
  { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
  { value: 'claude-3-5-sonnet', label: 'Claude 3.5 Sonnet' },
  { value: 'claude-3-opus', label: 'Claude 3 Opus' },
  { value: 'gemini-pro', label: 'Gemini Pro' },
] as const;

// Query modes
export const QUERY_MODES = [
  { value: 'hybrid', label: 'Hybrid (Recommended)' },
  { value: 'local', label: 'Local (Entity-based)' },
  { value: 'global', label: 'Global (Relation-based)' },
  { value: 'naive', label: 'Naive (Direct chunk)' },
] as const;

// Graph layouts
export const GRAPH_LAYOUTS = [
  { value: 'cose-bilkent', label: 'Cose-Bilkent (Bipartite)' },
  { value: 'dagre', label: 'Dagre (Hierarchical)' },
  { value: 'fcose', label: 'Force-directed' },
  { value: 'grid', label: 'Grid' },
  { value: 'circle', label: 'Circle' },
  { value: 'concentric', label: 'Concentric' },
  { value: 'breadthfirst', label: 'Breadthfirst' },
] as const;

// Languages
export const LANGUAGES = [
  { value: 'en', label: 'English' },
  { value: 'bn', label: 'বাংলা' },
] as const;

// Themes
export const THEMES = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'auto', label: 'Auto' },
] as const;