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

    NEW (January 2025): Updated to 16 independent features with clear separation
    between boolean feature toggles and configuration parameters.

    Architecture: All features are independent with explicit dependency validation.
    No auto-enabling - user has full control with fail-fast error messages.
    """

    # ========================================
    # GROUP A: CHUNKING FEATURES (2 features)
    # ========================================
    chunking_strategy: str = "semantic"
    """Chunking strategy: 'token' (fast, fixed-size) | 'semantic' (slow, boundary-aware)"""

    enable_table_detection: bool = True
    """Enable GPT-4 table extraction during chunking (requires openai_api_key with semantic strategy)"""

    # ========================================
    # GROUP B: EXTRACTION FEATURES (4 features)
    # ========================================
    extraction_strategy: str = "gleaning"
    """Extraction strategy: 'strict' (single-pass) | 'gleaning' (multi-pass refinement)"""

    enable_table_fact_extraction: bool = False
    """Enable rule-based table→facts extraction (requires enable_table_detection=True)"""

    enable_multilingual: bool = True
    """Enable multilingual extraction (supports English, Bangla, Hindi, Arabic, etc.)"""

    # ========================================
    # GROUP C: VALIDATION FEATURES (3 features)
    # ========================================
    enable_numeric_validation: bool = False
    """Enable Gemini-based numeric accuracy validation (expensive, requires gemini_api_key)"""

    enable_entity_validation: bool = True
    """Enable entity quality validation (cheap, regex-based - default ON)"""

    enable_relation_validation: bool = True
    """Enable relation completeness validation (cheap, regex-based - default ON)"""

    # ========================================
    # GROUP D: MERGING FEATURES (2 features)
    # ========================================
    enable_entity_merging: bool = True
    """Enable entity deduplication (can disable to keep all duplicate entities)"""

    enable_fuzzy_matching: bool = True
    """Enable fuzzy string matching for merging (vs exact match only)"""

    # ========================================
    # GROUP E: QUALITY FEATURES (3 features)
    # ========================================
    enable_hitl: bool = True
    """Enable Human-in-the-Loop failure tracking (default ON, no cost)"""

    enable_orphan_linking: bool = True
    """Enable synthetic relation generation for orphan entities"""

    enable_quality_scoring: bool = True
    """Enable extraction quality metric tracking (default ON, cheap)"""

    # ========================================
    # PARAMETERS (not features - configuration values)
    # ========================================
    # Chunking parameters
    chunk_size: int = 1200
    chunk_overlap: int = 100

    # Extraction parameters
    gleaning_iterations: int = 2
    """Number of gleaning passes (only applies if extraction_strategy='gleaning')"""

    extraction_concurrency: int = 16
    """Max concurrent LLM API calls"""

    # Validation parameters
    validation_strictness: str = "MODERATE"
    """Validation strictness: 'STRICT' (99%) | 'MODERATE' (95%) | 'LENIENT' (80%)"""

    numeric_validation_mode: str = "document"
    """Numeric validation mode: 'chunk' | 'document' (only applies if enable_numeric_validation=True)"""

    # Merging parameters
    fuzzy_similarity_threshold: float = 0.9
    """Fuzzy matching threshold 0-1 (only applies if enable_fuzzy_matching=True)"""

    # Quality parameters
    enable_llm_cache: bool = True
    """Enable LLM response caching (recommended for cost/speed)"""

    # API Keys
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Dataset path (for HITL)
    dataset_path: Optional[str] = None

    def __post_init__(self):
        """
        Validate configuration with explicit dependency checks.

        NEW (January 2025): All features are independent, but some combinations
        require specific conditions. We validate these at config time with clear
        error messages (fail-fast, not silent fallback).
        """
        import logging
        logger = logging.getLogger(__name__)

        # ========================================
        # VALIDATE STRATEGY CHOICES
        # ========================================
        valid_chunking_strategies = ['token', 'semantic']
        if self.chunking_strategy not in valid_chunking_strategies:
            raise ValueError(
                f"[IndexingConfig] chunking_strategy='{self.chunking_strategy}' invalid. "
                f"Choose from: {valid_chunking_strategies}"
            )

        valid_extraction_strategies = ['strict', 'gleaning']
        if self.extraction_strategy not in valid_extraction_strategies:
            raise ValueError(
                f"[IndexingConfig] extraction_strategy='{self.extraction_strategy}' invalid. "
                f"Choose from: {valid_extraction_strategies}"
            )

        valid_strictness = ['STRICT', 'MODERATE', 'LENIENT']
        if self.validation_strictness not in valid_strictness:
            raise ValueError(
                f"[IndexingConfig] validation_strictness='{self.validation_strictness}' invalid. "
                f"Choose from: {valid_strictness}"
            )

        valid_numeric_modes = ['chunk', 'document']
        if self.numeric_validation_mode not in valid_numeric_modes:
            raise ValueError(
                f"[IndexingConfig] numeric_validation_mode='{self.numeric_validation_mode}' invalid. "
                f"Choose from: {valid_numeric_modes}"
            )

        # ========================================
        # DEPENDENCY CHECK 1: Table Fact Extraction → Table Detection
        # ========================================
        if self.enable_table_fact_extraction and not self.enable_table_detection:
            raise ValueError(
                "[IndexingConfig] enable_table_fact_extraction=True requires enable_table_detection=True.\n"
                "Reason: Cannot extract facts from tables that weren't detected.\n"
                "Fix: Set enable_table_detection=True OR disable table_fact_extraction."
            )

        # ========================================
        # DEPENDENCY CHECK 2: Table Detection with Semantic Chunking → API Key
        # ========================================
        if self.enable_table_detection and self.chunking_strategy == 'semantic':
            if not self.openai_api_key:
                raise ValueError(
                    "[IndexingConfig] chunking_strategy='semantic' with enable_table_detection=True requires openai_api_key.\n"
                    "Reason: GPT-4 is needed to detect and extract tables.\n"
                    "Fix: Provide openai_api_key OR set enable_table_detection=False."
                )

        # ========================================
        # DEPENDENCY CHECK 3: Fuzzy Matching → Merging Enabled
        # ========================================
        if self.enable_fuzzy_matching and not self.enable_entity_merging:
            raise ValueError(
                "[IndexingConfig] enable_fuzzy_matching=True requires enable_entity_merging=True.\n"
                "Reason: Fuzzy matching is a merging technique.\n"
                "Fix: Set enable_entity_merging=True OR disable fuzzy_matching."
            )

        # ========================================
        # DEPENDENCY CHECK 4: Numeric Validation → Gemini API Key
        # ========================================
        if self.enable_numeric_validation:
            if not self.gemini_api_key:
                logger.warning(
                    "[IndexingConfig] enable_numeric_validation=True but gemini_api_key not provided. "
                    "Numeric validation will be SKIPPED (pass-through). "
                    "To enable validation, provide gemini_api_key parameter."
                )

    @classmethod
    def preset_fast(cls, **kwargs) -> 'IndexingConfig':
        """
        Fast preset: Optimized for speed and low cost.

        Features:
        - Chunking: Token-based (no table detection)
        - Extraction: Single-pass strict extraction
        - Validation: Entity + Relation only (cheap, no numeric validation)
        - Merging: Exact match only (no fuzzy matching)
        - Quality: All ON (HITL, orphan linking, quality scoring)

        Use Cases:
        - Large corpora (100K+ documents)
        - Speed-critical applications
        - Simple text documents without tables
        - Budget-constrained projects

        Cost: ~$0.50 per 10K documents (gpt-4o-mini)
        Speed: ~2-3 minutes per 1K documents
        """
        return cls(
            # Chunking: Fast token-based
            chunking_strategy="token",
            enable_table_detection=False,  # No GPT-4 calls

            # Extraction: Single-pass
            extraction_strategy="strict",
            enable_table_fact_extraction=False,
            enable_multilingual=False,  # English-only for speed

            # Validation: Only cheap validations
            enable_numeric_validation=False,  # Expensive
            enable_entity_validation=True,    # Cheap
            enable_relation_validation=True,  # Cheap

            # Merging: Exact match only
            enable_entity_merging=True,
            enable_fuzzy_matching=False,  # Expensive

            # Quality: All ON (cheap)
            enable_hitl=True,
            enable_orphan_linking=True,
            enable_quality_scoring=True,

            **kwargs
        )

    @classmethod
    def preset_balanced(cls, **kwargs) -> 'IndexingConfig':
        """
        Balanced preset: Good quality/speed tradeoff (RECOMMENDED).

        Features:
        - Chunking: Semantic with table detection (requires openai_api_key)
        - Extraction: Multi-pass gleaning for quality
        - Validation: Entity + Relation (no expensive numeric validation)
        - Merging: Fuzzy matching for deduplication
        - Quality: All ON

        Use Cases:
        - General-purpose knowledge graphs
        - Educational content
        - Mixed documents (paragraphs + tables)
        - Production deployments

        Cost: ~$2-3 per 10K documents (gpt-4o-mini)
        Speed: ~5-8 minutes per 1K documents

        Note: Requires openai_api_key for table detection.
        Set enable_table_detection=False if no API key available.
        """
        return cls(
            # Chunking: Semantic with tables
            chunking_strategy="semantic",
            enable_table_detection=True,  # Requires API key

            # Extraction: Gleaning for quality
            extraction_strategy="gleaning",
            enable_table_fact_extraction=False,  # Use LLM for tables too
            enable_multilingual=True,

            # Validation: Cheap validations only
            enable_numeric_validation=False,
            enable_entity_validation=True,
            enable_relation_validation=True,

            # Merging: Fuzzy for deduplication
            enable_entity_merging=True,
            enable_fuzzy_matching=True,

            # Quality: All ON
            enable_hitl=True,
            enable_orphan_linking=True,
            enable_quality_scoring=True,

            **kwargs
        )

    @classmethod
    def preset_quality(cls, **kwargs) -> 'IndexingConfig':
        """
        Quality preset: Maximum accuracy with all features enabled.

        Features:
        - Chunking: Semantic with table detection (requires openai_api_key)
        - Extraction: Gleaning + table fact extraction
        - Validation: ALL validations (numeric + entity + relation)
        - Merging: Fuzzy matching with high threshold
        - Quality: All ON

        Use Cases:
        - High-value content
        - Academic papers
        - Technical documentation with tables
        - Legal documents
        - Medical records

        Cost: ~$5-8 per 10K documents (gpt-4o-mini + gemini)
        Speed: ~10-15 minutes per 1K documents

        Note: Requires openai_api_key (table detection) + gemini_api_key (numeric validation).
        """
        return cls(
            # Chunking: Semantic with tables
            chunking_strategy="semantic",
            enable_table_detection=True,  # Requires API key

            # Extraction: Gleaning + table facts
            extraction_strategy="gleaning",
            enable_table_fact_extraction=True,  # Rule-based table facts
            enable_multilingual=True,

            # Validation: ALL validations
            enable_numeric_validation=True,   # Expensive but accurate
            enable_entity_validation=True,
            enable_relation_validation=True,

            # Merging: Fuzzy with high threshold
            enable_entity_merging=True,
            enable_fuzzy_matching=True,
            fuzzy_similarity_threshold=0.95,  # Stricter

            # Quality: All ON
            enable_hitl=True,
            enable_orphan_linking=True,
            enable_quality_scoring=True,

            # Parameters
            validation_strictness="MODERATE",
            numeric_validation_mode="document",  # More accurate

            **kwargs
        )
