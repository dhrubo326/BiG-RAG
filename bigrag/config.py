"""
BiG-RAG Configuration Management

Loads configuration from environment variables (.env file).
Provides type-safe access to configuration values with sensible defaults.

Usage:
    from bigrag.config import config

    # Access values
    api_key = config.openai_api_key
    chunk_size = config.chunk_size
    enable_reranking = config.enable_reranking
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Try to use python-dotenv if available, otherwise use custom loader
try:
    from dotenv import load_dotenv
    # load_dotenv will automatically search for .env in parent directories
    load_dotenv()
except ImportError:
    # Fallback to custom loader if python-dotenv not installed
    def load_env_file(env_path: str = ".env") -> None:
        """
        Load environment variables from .env file

        Args:
            env_path: Path to .env file (default: searches for .env up the directory tree)
        """
        # Try to find .env in current or parent directories
        current = Path.cwd()
        env_file = None

        # Search up the directory tree for .env
        while current != current.parent:
            potential_env = current / env_path
            if potential_env.exists():
                env_file = potential_env
                break
            current = current.parent

        # Fallback to current directory
        if env_file is None:
            env_file = Path(env_path)
            if not env_file.exists():
                return

        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                        value = value[1:-1]

                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value

    # Load .env file on module import
    load_env_file()


def parse_bool(value: str) -> bool:
    """
    Parse boolean from string (handles true/false and 1/0).

    Supports multiple formats:
    - "true", "TRUE", "True" → True
    - "false", "FALSE", "False" → False
    - "1", "yes", "on" → True
    - "0", "no", "off" → False

    Args:
        value: String value to parse

    Returns:
        Boolean value

    Raises:
        ValueError: If value cannot be parsed as boolean
    """
    value_lower = value.lower().strip()
    if value_lower in ('true', '1', 'yes', 'on'):
        return True
    if value_lower in ('false', '0', 'no', 'off'):
        return False
    raise ValueError(f"Cannot parse '{value}' as boolean")


@dataclass
class BiGRAGConfig:
    """BiG-RAG Configuration with Environment Variable Support"""

    # ========================
    # Server Configuration
    # ========================
    host: str = field(default_factory=lambda: os.getenv('HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(os.getenv('PORT', '8001')))
    webui_title: str = field(default_factory=lambda: os.getenv('WEBUI_TITLE', 'BiG-RAG API'))
    webui_description: str = field(default_factory=lambda: os.getenv(
        'WEBUI_DESCRIPTION',
        'Bipartite Graph Retrieval-Augmented Generation'
    ))

    # ========================
    # Dataset Settings
    # ========================
    default_dataset: str = field(default_factory=lambda: os.getenv('DEFAULT_DATASET', 'demo_test'))
    input_dir: str = field(default_factory=lambda: os.getenv('INPUT_DIR', './datasets'))
    working_dir: str = field(default_factory=lambda: os.getenv('WORKING_DIR', './expr'))

    # ========================
    # Query Configuration
    # ========================
    top_k: int = field(default_factory=lambda: int(os.getenv('TOP_K', '5')))
    retrieval_mode: str = field(default_factory=lambda: os.getenv('RETRIEVAL_MODE', 'hybrid'))
    enable_reranking: bool = field(default_factory=lambda: parse_bool(os.getenv('ENABLE_RERANKING', 'false')))
    # Query Preprocessing
    # Enable automatic query preprocessing (typo correction, grammar fixing, language translation)
    # Set to 'false' to disable preprocessing globally (can still be overridden per-query)
    enable_query_preprocessing: bool = field(default_factory=lambda: parse_bool(os.getenv('ENABLE_QUERY_PREPROCESSING', 'true')))

    # Reranking Configuration
    rerank_provider: str = field(default_factory=lambda: os.getenv('RERANK_PROVIDER', 'local'))
    rerank_model: str = field(default_factory=lambda: os.getenv(
        'RERANK_MODEL',
        'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ))
    rerank_batch_size: int = field(default_factory=lambda: int(os.getenv('RERANK_BATCH_SIZE', '32')))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv('RERANK_TOP_K', '5')))

    # Jina AI Reranker
    jina_api_key: str = field(default_factory=lambda: os.getenv('JINA_API_KEY', ''))
    jina_rerank_model: str = field(default_factory=lambda: os.getenv(
        'JINA_RERANK_MODEL',
        'jina-reranker-v2-base-multilingual'
    ))
    jina_api_url: str = field(default_factory=lambda: os.getenv(
        'JINA_API_URL',
        'https://api.jina.ai/v1/rerank'
    ))

    # Custom Reranker API
    custom_rerank_api_url: str = field(default_factory=lambda: os.getenv('CUSTOM_RERANK_API_URL', ''))
    custom_rerank_api_key: str = field(default_factory=lambda: os.getenv('CUSTOM_RERANK_API_KEY', ''))

    max_context_items: int = field(default_factory=lambda: int(os.getenv('MAX_CONTEXT_ITEMS', '10')))
    enable_llm_cache: bool = field(default_factory=lambda: parse_bool(os.getenv('ENABLE_LLM_CACHE', 'true')))

    # ========================
    # Document Processing
    # ========================
    chunk_size: int = field(default_factory=lambda: int(os.getenv('CHUNK_SIZE', '1200')))
    chunk_overlap_size: int = field(default_factory=lambda: int(os.getenv('CHUNK_OVERLAP_SIZE', '100')))
    tiktoken_model: str = field(default_factory=lambda: os.getenv('TIKTOKEN_MODEL', 'gpt-4o'))
    entity_types: List[str] = field(default_factory=lambda: json.loads(
        os.getenv('ENTITY_TYPES', '["organization", "person", "geo", "time"]')
    ))
    default_language: str = field(default_factory=lambda: os.getenv('DEFAULT_LANGUAGE', 'English'))
    max_async: int = field(default_factory=lambda: int(os.getenv('MAX_ASYNC', '4')))
    enable_llm_cache_for_extract: bool = field(default_factory=lambda: parse_bool(os.getenv(
        'ENABLE_LLM_CACHE_FOR_EXTRACT',
        'true'
    )))

    # ========================
    # Embedding Configuration
    # ========================
    embedding_provider: str = field(default_factory=lambda: os.getenv('EMBEDDING_PROVIDER', 'openai'))
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large'))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv('EMBEDDING_DIM', '3072')))
    embedding_batch_size: int = field(default_factory=lambda: int(os.getenv('EMBEDDING_BATCH_SIZE', '10')))
    embedding_batch_num: int = field(default_factory=lambda: int(os.getenv('EMBEDDING_BATCH_NUM', '32')))  # Alias for backward compatibility with BiGRAG class
    embedding_max_async: int = field(default_factory=lambda: int(os.getenv('EMBEDDING_MAX_ASYNC', '8')))
    embedding_device: str = field(default_factory=lambda: os.getenv('EMBEDDING_DEVICE', 'cpu'))

    # ========================
    # LLM Configuration
    # ========================
    llm_provider: str = field(default_factory=lambda: os.getenv('LLM_PROVIDER', 'openai'))

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    openai_model: str = field(default_factory=lambda: os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))
    openai_base_url: str = field(default_factory=lambda: os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    openai_temperature: float = field(default_factory=lambda: float(os.getenv('OPENAI_TEMPERATURE', '0.0')))
    openai_max_tokens: int = field(default_factory=lambda: int(os.getenv('OPENAI_MAX_TOKENS', '4096')))

    # Anthropic (Claude)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY', ''))
    anthropic_model: str = field(default_factory=lambda: os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022'))
    anthropic_max_tokens: int = field(default_factory=lambda: int(os.getenv('ANTHROPIC_MAX_TOKENS', '4096')))

    # Google (Gemini)
    google_api_key: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY', ''))
    google_model: str = field(default_factory=lambda: os.getenv('GOOGLE_MODEL', 'gemini-pro'))
    google_max_tokens: int = field(default_factory=lambda: int(os.getenv('GOOGLE_MAX_TOKENS', '4096')))

    # xAI (Grok)
    xai_api_key: str = field(default_factory=lambda: os.getenv('XAI_API_KEY', ''))
    grok_model: str = field(default_factory=lambda: os.getenv('GROK_MODEL', 'grok-beta'))
    grok_max_tokens: int = field(default_factory=lambda: int(os.getenv('GROK_MAX_TOKENS', '4096')))

    llm_timeout: int = field(default_factory=lambda: int(os.getenv('LLM_TIMEOUT', '180')))

    # ========================
    # Storage Backend
    # ========================
    kv_storage: str = field(default_factory=lambda: os.getenv('KV_STORAGE', 'JsonKVStorage'))
    graph_storage: str = field(default_factory=lambda: os.getenv('GRAPH_STORAGE', 'NetworkXStorage'))
    vector_storage: str = field(default_factory=lambda: os.getenv('VECTOR_STORAGE', 'NanoVectorDBStorage'))

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv('NEO4J_URI', ''))
    neo4j_username: str = field(default_factory=lambda: os.getenv('NEO4J_USERNAME', 'neo4j'))
    neo4j_password: str = field(default_factory=lambda: os.getenv('NEO4J_PASSWORD', ''))
    neo4j_database: str = field(default_factory=lambda: os.getenv('NEO4J_DATABASE', 'neo4j'))

    # MongoDB
    mongo_uri: str = field(default_factory=lambda: os.getenv('MONGO_URI', ''))
    mongo_database: str = field(default_factory=lambda: os.getenv('MONGO_DATABASE', 'BiGRAG'))

    # Milvus
    milvus_uri: str = field(default_factory=lambda: os.getenv('MILVUS_URI', ''))
    milvus_db_name: str = field(default_factory=lambda: os.getenv('MILVUS_DB_NAME', 'bigrag'))

    # ChromaDB
    chromadb_host: str = field(default_factory=lambda: os.getenv('CHROMADB_HOST', 'localhost'))
    chromadb_port: int = field(default_factory=lambda: int(os.getenv('CHROMADB_PORT', '8000')))

    # ========================
    # Document Registry
    # ========================
    registry_path: str = field(default_factory=lambda: os.getenv('REGISTRY_PATH', './document_registry.json'))
    max_documents_per_dataset: int = field(default_factory=lambda: int(os.getenv('MAX_DOCUMENTS_PER_DATASET', '0')))
    job_cleanup_hours: int = field(default_factory=lambda: int(os.getenv('JOB_CLEANUP_HOURS', '24')))

    # ========================
    # Background Processing
    # ========================
    enable_async_processing: bool = field(default_factory=lambda: os.getenv(
        'ENABLE_ASYNC_PROCESSING',
        'true'
    ).lower() == 'true')
    max_parallel_jobs: int = field(default_factory=lambda: int(os.getenv('MAX_PARALLEL_JOBS', '2')))
    max_job_queue_size: int = field(default_factory=lambda: int(os.getenv('MAX_JOB_QUEUE_SIZE', '100')))
    job_timeout: int = field(default_factory=lambda: int(os.getenv('JOB_TIMEOUT', '600')))

    # ========================
    # RL Training (Optional)
    # ========================
    training_mode: bool = field(default_factory=lambda: parse_bool(os.getenv('TRAINING_MODE', 'false')))
    base_model: str = field(default_factory=lambda: os.getenv('BASE_MODEL', ''))
    rl_algorithm: str = field(default_factory=lambda: os.getenv('RL_ALGORITHM', 'grpo'))
    actor_lr: float = field(default_factory=lambda: float(os.getenv('ACTOR_LR', '5e-7')))
    critic_lr: float = field(default_factory=lambda: float(os.getenv('CRITIC_LR', '1e-5')))
    train_batch_size: int = field(default_factory=lambda: int(os.getenv('TRAIN_BATCH_SIZE', '128')))
    total_epochs: int = field(default_factory=lambda: int(os.getenv('TOTAL_EPOCHS', '1')))

    # ========================
    # Evaluation
    # ========================
    eval_metrics: List[str] = field(default_factory=lambda: os.getenv('EVAL_METRICS', 'em,f1,rouge_l').split(','))
    eval_batch_size: int = field(default_factory=lambda: int(os.getenv('EVAL_BATCH_SIZE', '32')))
    save_eval_results: bool = field(default_factory=lambda: parse_bool(os.getenv('SAVE_EVAL_RESULTS', 'true')))
    eval_results_dir: str = field(default_factory=lambda: os.getenv('EVAL_RESULTS_DIR', './evaluation_results'))

    # ========================
    # Advanced Settings
    # ========================
    debug: bool = field(default_factory=lambda: parse_bool(os.getenv('DEBUG', 'false')))
    enable_profiling: bool = field(default_factory=lambda: parse_bool(os.getenv('ENABLE_PROFILING', 'false')))
    cache_dir: str = field(default_factory=lambda: os.getenv('CACHE_DIR', './cache'))
    tiktoken_cache_dir: str = field(default_factory=lambda: os.getenv('TIKTOKEN_CACHE_DIR', './cache/tiktoken'))
    hf_home: str = field(default_factory=lambda: os.getenv('HF_HOME', './cache/huggingface'))

    api_retry_attempts: int = field(default_factory=lambda: int(os.getenv('API_RETRY_ATTEMPTS', '3')))
    api_retry_backoff: float = field(default_factory=lambda: float(os.getenv('API_RETRY_BACKOFF', '1.0')))
    api_retry_max_backoff: float = field(default_factory=lambda: float(os.getenv('API_RETRY_MAX_BACKOFF', '10.0')))

    # ========================
    # Health & Monitoring
    # ========================
    enable_health_check: bool = field(default_factory=lambda: os.getenv(
        'ENABLE_HEALTH_CHECK',
        'true'
    ).lower() == 'true')
    health_check_interval: int = field(default_factory=lambda: int(os.getenv('HEALTH_CHECK_INTERVAL', '60')))

    # ========================
    # Logging
    # ========================
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_dir: Optional[str] = field(default_factory=lambda: os.getenv('LOG_DIR', None))

    def __post_init__(self):
        """Validate configuration and load API keys from files if needed"""
        # Load API keys from files if not in environment
        self._load_api_key_from_file('OPENAI_API_KEY', 'openai_api_key.txt')
        self._load_api_key_from_file('ANTHROPIC_API_KEY', 'anthropic_api_key.txt')
        self._load_api_key_from_file('GOOGLE_API_KEY', 'google_api_key.txt')
        self._load_api_key_from_file('XAI_API_KEY', 'grok_api_key.txt')

        # Update instance attributes with loaded keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY', self.openai_api_key)
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', self.anthropic_api_key)
        self.google_api_key = os.getenv('GOOGLE_API_KEY', self.google_api_key)
        self.xai_api_key = os.getenv('XAI_API_KEY', self.xai_api_key)

        # Create cache directories if they don't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tiktoken_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.hf_home).mkdir(parents=True, exist_ok=True)

    def _load_api_key_from_file(self, env_key: str, filename: str) -> None:
        """Load API key from file if not in environment"""
        if os.getenv(env_key):
            return

        key_file = Path(filename)
        if key_file.exists():
            with open(key_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
                os.environ[env_key] = api_key

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with optional default"""
        return getattr(self, key, default)

    def update(self, **kwargs) -> None:
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def print_summary(self) -> None:
        """Print configuration summary (masks sensitive values)"""
        print("\n" + "="*80)
        print("BiG-RAG Configuration Summary")
        print("="*80)

        sensitive_keys = {
            'openai_api_key', 'anthropic_api_key', 'google_api_key', 'xai_api_key',
            'jina_api_key', 'custom_rerank_api_key',
            'neo4j_password', 'mongo_uri', 'milvus_uri'
        }

        for key, value in self.to_dict().items():
            if key in sensitive_keys:
                if value:
                    display_value = f"{value[:8]}..." if len(value) > 8 else "***"
                else:
                    display_value = "(not set)"
            else:
                display_value = value

            print(f"  {key}: {display_value}")

        print("="*80 + "\n")


# Global configuration instance
config = BiGRAGConfig()


# Convenience functions
def get_config() -> BiGRAGConfig:
    """Get global configuration instance"""
    return config


def reload_config() -> BiGRAGConfig:
    """Reload configuration from environment"""
    global config
    # Bug #3 Fix: load_env_file() only exists when python-dotenv is NOT installed
    # Try dotenv first, fall back to custom loader
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        load_env_file()
    config = BiGRAGConfig()
    return config


# ========================
# Phase 2.3: Multi-Hop Query Configuration
# ========================

"""
Dynamic configuration system for multi-hop graph traversal.

Provides dataset-specific and domain-specific defaults for hop counts.
Enables runtime registration of custom configurations without code changes.
"""

from .base import QueryParam as QueryParamBase

# Dataset-specific hop configuration
# Multi-hop datasets require 2-hop traversal, single-hop datasets use 1-hop
DATASET_HOP_CONFIG: Dict[str, int] = {
    # Multi-hop QA datasets (require 2-hop reasoning)
    "2WikiMultiHopQA": 2,
    "HotpotQA": 2,
    "Musique": 2,

    # Single-hop QA datasets
    "NQ": 1,
    "PopQA": 1,
    "TriviaQA": 1,

    # Demo/test datasets
    "SingleTopic": 1,
    "demo_test": 1,
}

# Domain-based hop configuration (fallback when dataset not in DATASET_HOP_CONFIG)
DOMAIN_HOP_CONFIG: Dict[str, int] = {
    "academic": 2,        # Academic papers often need multi-hop reasoning
    "scientific": 2,      # Scientific documents often require multi-hop
    "medical": 2,         # Medical knowledge requires multi-hop
    "legal": 2,           # Legal documents often need multi-hop
    "news": 1,            # News articles usually single-hop
    "qa": 2,              # QA datasets usually multi-hop
    "wiki": 2,            # Wikipedia often requires multi-hop
    "general": 1,         # General-purpose default
}


def get_default_query_param(
    dataset_name: Optional[str] = None,
    domain: Optional[str] = None,
    max_hops: Optional[int] = None,
    **kwargs
) -> QueryParamBase:
    """
    Get dataset-specific or domain-specific default query parameters.

    Priority (highest to lowest):
        1. Explicit max_hops parameter
        2. Dataset-specific configuration (DATASET_HOP_CONFIG)
        3. Domain-specific configuration (DOMAIN_HOP_CONFIG)
        4. Global default (1-hop)

    Args:
        dataset_name: Specific dataset name (e.g., "2WikiMultiHopQA", "HotpotQA")
        domain: Domain category (e.g., "academic", "scientific", "news")
        max_hops: Explicit hop count override (1-3), takes precedence over all defaults
        **kwargs: Additional QueryParam overrides (mode, top_k, enable_reranking, etc.)

    Returns:
        QueryParam with appropriate defaults based on dataset/domain

    Examples:
        # Explicit override (highest priority)
        >>> param = get_default_query_param(max_hops=3)
        >>> param.max_hops
        3

        # Dataset-specific default
        >>> param = get_default_query_param(dataset_name="2WikiMultiHopQA")
        >>> param.max_hops
        2

        # Domain-specific default
        >>> param = get_default_query_param(domain="academic")
        >>> param.max_hops
        2

        # Default (no matches)
        >>> param = get_default_query_param()
        >>> param.max_hops
        1

        # Override other parameters
        >>> param = get_default_query_param(dataset_name="HotpotQA", top_k=100, enable_reranking=False)
        >>> param.max_hops, param.top_k, param.enable_reranking
        (2, 100, False)
    """
    # Determine hop count based on priority
    if max_hops is not None:
        # Explicit override takes precedence
        hops = max_hops
    elif dataset_name and dataset_name in DATASET_HOP_CONFIG:
        # Dataset-specific configuration
        hops = DATASET_HOP_CONFIG[dataset_name]
    elif domain and domain in DOMAIN_HOP_CONFIG:
        # Domain-specific configuration
        hops = DOMAIN_HOP_CONFIG[domain]
    else:
        # Global default
        hops = 1

    # Create QueryParam with defaults
    param_dict = {
        "mode": "hybrid",
        "top_k": 60,
        "max_hops": hops,
        "enable_reranking": True,
    }

    # Apply any additional overrides from kwargs
    param_dict.update(kwargs)

    return QueryParamBase(**param_dict)


def register_dataset_hop_config(dataset_name: str, max_hops: int):
    """
    Register a custom dataset hop configuration at runtime.

    Enables dynamic configuration without modifying code.
    Useful for production systems with custom datasets.

    Args:
        dataset_name: Dataset identifier (case-sensitive)
        max_hops: Number of hops (1-3)

    Raises:
        ValueError: If max_hops not in range [1, 3]

    Examples:
        >>> register_dataset_hop_config("CustomMedicalQA", 2)
        >>> param = get_default_query_param("CustomMedicalQA")
        >>> param.max_hops
        2

        >>> register_dataset_hop_config("CustomNewsQA", 1)
        >>> param = get_default_query_param("CustomNewsQA")
        >>> param.max_hops
        1
    """
    if not 1 <= max_hops <= 3:
        raise ValueError(f"max_hops must be between 1 and 3, got {max_hops}")
    DATASET_HOP_CONFIG[dataset_name] = max_hops


def register_domain_hop_config(domain: str, max_hops: int):
    """
    Register a custom domain hop configuration at runtime.

    Enables dynamic configuration without modifying code.
    Useful for production systems with custom domain categories.

    Args:
        domain: Domain category (case-sensitive)
        max_hops: Number of hops (1-3)

    Raises:
        ValueError: If max_hops not in range [1, 3]

    Examples:
        >>> register_domain_hop_config("legal", 2)
        >>> param = get_default_query_param(domain="legal")
        >>> param.max_hops
        2

        >>> register_domain_hop_config("entertainment", 1)
        >>> param = get_default_query_param(domain="entertainment")
        >>> param.max_hops
        1
    """
    if not 1 <= max_hops <= 3:
        raise ValueError(f"max_hops must be between 1 and 3, got {max_hops}")
    DOMAIN_HOP_CONFIG[domain] = max_hops


def get_registered_datasets() -> Dict[str, int]:
    """
    Get all registered dataset configurations.

    Returns:
        Dictionary mapping dataset names to hop counts

    Example:
        >>> configs = get_registered_datasets()
        >>> configs["2WikiMultiHopQA"]
        2
    """
    return DATASET_HOP_CONFIG.copy()


def get_registered_domains() -> Dict[str, int]:
    """
    Get all registered domain configurations.

    Returns:
        Dictionary mapping domain categories to hop counts

    Example:
        >>> configs = get_registered_domains()
        >>> configs["academic"]
        2
    """
    return DOMAIN_HOP_CONFIG.copy()


# ========================
# Indexing Configuration (Strategy Pattern)
# ========================

@dataclass
class IndexingConfig:
    """
    Configuration for BiG-RAG indexing system using strategy pattern.

    Replaces PipelineFeatures with cleaner, strategy-focused configuration.
    Maps to 13 original pipeline features.
    """

    # ========== STRATEGIES ==========
    chunker: str = "semantic"
    """Chunking strategy: 'token' | 'semantic' | 'hybrid'"""

    extractor: str = "gleaning"
    """Extraction strategy: 'strict' | 'gleaning' | 'hybrid'"""

    validators: List[str] = field(default_factory=list)
    """Validation strategies: [] | ['numeric'] | ['entity'] | ['relation'] | ['numeric', 'entity', 'relation']

    Available validators:
    - 'numeric': Gemini-based numeric consistency validation (document-level or chunk-level)
    - 'entity': Entity quality validation (name length, description, generic type filtering)
    - 'relation': Relation completeness validation (description length, completeness score)
    - 'semantic': Legacy - validates BOTH entity AND relation (use 'entity' + 'relation' for granular control)
    """

    merger: str = "fuzzy"
    """Merging strategy: 'basic' | 'fuzzy' | 'hybrid'"""

    hitl: str = "file"
    """HITL strategy: 'file' | 'database' | 'noop'"""

    orphan_linker: str = "synthetic"
    """Orphan linking strategy: 'synthetic' | 'noop'"""

    # ========== PARAMETERS ==========
    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 100
    enable_table_detection: bool = True
    """Enable GPT-4 table extraction (used with semantic/hybrid chunking)"""

    # Extraction
    gleaning_iterations: int = 2
    extraction_concurrency: int = 16
    enable_table_fact_extraction: bool = True  # NEW: Issue #4 - explicit control
    """Enable rule-based table fact extraction (used with hybrid extractor)"""

    # Validation
    validation_strictness: str = "MODERATE"  # STRICT | MODERATE | LENIENT
    validation_mode: str = "document"  # NEW (Issue #2): "chunk" | "document" | "hybrid"
    """Numeric validation mode: 'chunk' (fast, less accurate) | 'document' (slow, more accurate) | 'hybrid' (fallback)"""

    # Quality
    enable_quality_scoring: bool = True

    # LLM Cache
    enable_llm_cache: bool = True

    # API Keys
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Dataset path (for HITL)
    dataset_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        import logging
        logger = logging.getLogger(__name__)

        # Validate strategy choices
        valid_chunkers = ['token', 'semantic', 'hybrid']
        if self.chunker not in valid_chunkers:
            raise ValueError(f"chunker must be one of {valid_chunkers}")

        valid_extractors = ['strict', 'gleaning', 'hybrid']
        if self.extractor not in valid_extractors:
            raise ValueError(f"extractor must be one of {valid_extractors}")

        valid_validators = ['numeric', 'semantic']
        for v in self.validators:
            if v not in valid_validators:
                raise ValueError(f"validator '{v}' invalid. Choose from {valid_validators}")

        valid_mergers = ['basic', 'fuzzy', 'hybrid']
        if self.merger not in valid_mergers:
            raise ValueError(f"merger must be one of {valid_mergers}")

        valid_hitl = ['file', 'database', 'noop']
        if self.hitl not in valid_hitl:
            raise ValueError(f"hitl must be one of {valid_hitl}")

        valid_orphan_linkers = ['synthetic', 'noop']
        if self.orphan_linker not in valid_orphan_linkers:
            raise ValueError(f"orphan_linker must be one of {valid_orphan_linkers}")

        # NEW: Validate table detection configuration (FAIL-FAST - no silent fallback)
        if self.enable_table_detection and self.chunker in ['semantic', 'hybrid']:
            if not self.openai_api_key:
                raise ValueError(
                    f"[IndexingConfig] chunker='{self.chunker}' with enable_table_detection=True "
                    "requires openai_api_key. Either:\n"
                    "  1. Provide openai_api_key parameter, OR\n"
                    "  2. Set enable_table_detection=False to disable table extraction"
                )

    @classmethod
    def preset_fast(cls, **kwargs) -> 'IndexingConfig':
        """
        Fast preset: token chunking, strict extraction, basic merging.

        Chunking: Token-based (no table detection)
        Extraction: Strict schema (single-pass, no gleaning)
        Validation: None
        Merging: Basic (exact match)
        HITL: Disabled
        Orphan Linking: Disabled

        Use for: Large corpora, speed-critical applications, simple text documents
        """
        return cls(
            chunker="token",
            extractor="strict",
            validators=[],
            merger="basic",
            hitl="noop",
            orphan_linker="noop",
            **kwargs
        )

    @classmethod
    def preset_balanced(cls, **kwargs) -> 'IndexingConfig':
        """
        Balanced preset: semantic chunking, gleaning, fuzzy merging.

        Chunking: Semantic boundaries (includes table detection - REQUIRES openai_api_key)
        Extraction: Gleaning (multi-pass with refinement)
        Validation: Semantic entity validation
        Merging: Fuzzy matching
        HITL: File-based review queue
        Orphan Linking: Synthetic relation generation

        Use for: General-purpose knowledge graphs, educational content, mixed documents
        Note: Set enable_table_detection=False to disable table extraction
        """
        return cls(
            chunker="semantic",
            extractor="gleaning",
            validators=["entity"],  # NEW (Issue #3): Granular entity-only validation
            merger="fuzzy",
            hitl="file",
            orphan_linker="synthetic",
            validation_strictness="LENIENT",
            **kwargs
        )

    @classmethod
    def preset_quality(cls, **kwargs) -> 'IndexingConfig':
        """
        Quality preset: all features enabled, strict validation.

        Chunking: Semantic boundaries (includes table detection - REQUIRES openai_api_key)
        Extraction: Hybrid (tables via table_fact_extractor + paragraphs via gleaning)
        Validation: Numeric + Semantic validation
        Merging: Fuzzy matching
        HITL: File-based review queue
        Orphan Linking: Synthetic relation generation
        Quality Scoring: Enabled

        Use for: High-value content, academic papers, technical documentation with tables
        Note: Highest accuracy but slower and more expensive (multi-pass extraction + validation)
        """
        return cls(
            chunker="semantic",
            extractor="hybrid",
            validators=["numeric", "entity", "relation"],  # NEW (Issue #3): All three validators
            merger="fuzzy",
            hitl="file",
            orphan_linker="synthetic",
            validation_strictness="MODERATE",
            validation_mode="document",  # NEW (Issue #2): Use document-level numeric validation
            enable_quality_scoring=True,
            **kwargs
        )
