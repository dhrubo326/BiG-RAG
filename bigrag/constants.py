"""
BiG-RAG Code-Level Constants

This module contains code-level constants used throughout BiG-RAG.
These are defaults hardcoded in the implementation.

For deployment-specific configuration (environment variables),
see config.py which provides the BiGRAGConfig dataclass.

Separation of concerns:
- constants.py: Code defaults (rarely change, hardcoded)
- config.py: Deployment settings (vary by environment, from .env)
"""

# ========================
# Extraction Settings
# ========================

DEFAULT_CHUNK_SIZE = 1200
"""Default size of text chunks in tokens"""

DEFAULT_CHUNK_OVERLAP = 100
"""Default overlap between adjacent chunks in tokens"""

DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]
"""Default entity types for extraction"""

DEFAULT_MAX_ENTITY_TOKENS = 6000
"""Maximum tokens for entity context"""

DEFAULT_MAX_RELATION_TOKENS = 8000
"""Maximum tokens for relation context"""

# ========================
# Graph Settings
# ========================

GRAPH_FIELD_SEP = "<SEP>"
"""Separator for multi-value fields in graph nodes"""

BIPARTITE_EDGE_PREFIX = "rel-"
"""Prefix for bipartite edge (relation) node IDs"""

ENTITY_PREFIX = "ent-"
"""Prefix for entity node IDs in vector DB"""

CHUNK_PREFIX = "chunk-"
"""Prefix for text chunk IDs"""

# ========================
# Retrieval Settings
# ========================

DEFAULT_TOP_K_ENTITIES = 60
"""Default number of entities to retrieve (Path A)"""

DEFAULT_TOP_K_RELATIONS = 60
"""Default number of relations to retrieve (Path B)"""

DEFAULT_TOP_K_CHUNKS = 10
"""Default number of chunks to retrieve (Path C)"""

# ========================
# Embedding Settings
# ========================

DEFAULT_EMBEDDING_DIM = 1024
"""Default embedding dimension for FlagEmbedding"""

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
"""Default embedding model"""

# ========================
# Prompt Settings
# ========================

DEFAULT_LANGUAGE = "English"
"""Default language for entity extraction"""

DEFAULT_TUPLE_DELIMITER = "<|>"
"""Delimiter between tuple fields in extraction output"""

DEFAULT_RECORD_DELIMITER = "##"
"""Delimiter between records in extraction output"""

DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
"""Marker for completed extraction"""

# ========================
# Retry Settings (defaults if config not available)
# ========================

DEFAULT_MAX_RETRIES = 3
"""Default maximum retry attempts for transient failures"""

DEFAULT_RETRY_DELAY = 0.2
"""Default initial retry delay in seconds (exponential backoff)"""

# ========================
# Concurrency Settings
# ========================

DEFAULT_LLM_CONCURRENCY = 16
"""Default max concurrent LLM API calls (prevents rate limit errors)"""

# ========================
# Logging Settings
# ========================

DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024
"""Default max size for rotating log files (10MB)"""

DEFAULT_LOG_BACKUP_COUNT = 5
"""Default number of backup log files to keep"""
