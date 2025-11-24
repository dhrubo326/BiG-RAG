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

# =============================================================================
# Node ID Conventions
# =============================================================================
#
# BiG-RAG uses distinct ID formats for different node types to optimize
# performance and maintain consistency across storage layers (GraphML, Vector DB, KV Store).
#
# ENTITY NODES:
#   - Format: Canonical entity name (uppercase)
#   - Prefix: None (direct canonical name)
#   - Examples: "LIONEL MESSI", "BARCELONA", "PYTHON"
#   - Purpose: Human-readable for debugging and interpretability
#   - Storage: Graph nodes use name as ID, vector DB uses name as key
#
# RELATION NODES:
#   - Format: Hash-based with prefix "rel-"
#   - Hash: MD5 hash of relation content (32 hex characters)
#   - Examples: "rel-8c80df8f1fc71f13c4ffbe19fa22bf8f"
#   - Purpose: Collision-resistant, fast lookup, compact storage
#   - Storage: Graph nodes, vector DB keys, source_id references
#
# CHUNK REFERENCES:
#   - Format: Hash-based with prefix "chunk-"
#   - Hash: MD5 hash of chunk content (32 hex characters)
#   - Examples: "chunk-600f9c648bc602202ec663361837e416"
#   - Purpose: Consistent ID across graph and vector databases
#   - Storage: KV store keys, source_id fields, vector DB keys
#
# DOCUMENT IDs:
#   - Format: Hash-based with prefix "doc-"
#   - Hash: MD5 hash of document content (32 hex characters)
#   - Examples: "doc-ce2415fb73e5596b76d8f93f636c43a7"
#   - Purpose: Deduplication and consistent referencing
#   - Storage: KV store keys, full_doc_id references
#
# BENEFITS OF THIS DESIGN:
#   1. Fast Lookup: O(1) hash comparison vs O(n) string comparison for relations
#   2. Consistency: Hash IDs match vector database keys across storage layers
#   3. Compact Storage: GraphML files ~30-40% smaller than content-as-ID
#   4. Human Readability: Entity names remain interpretable for debugging
#   5. Collision Resistance: MD5 hashes virtually eliminate ID conflicts
#   6. Content Stability: Content changes automatically generate new IDs
#
# IMPLEMENTATION NOTES:
#   - Hash generation: Use bigrag.utils.compute_mdhash_id(content, prefix="rel-")
#   - Entity normalization: Use uppercase() for canonical names
#   - Case sensitivity: Entity IDs are case-insensitive (stored uppercase)
#   - Hash IDs: Must remain lowercase to match vector DB keys
#   - No uppercase transformation for hash-based IDs (breaks vector DB lookups)
#
# GRAPH STRUCTURE VALIDATION:
#   - Use NetworkXStorage.get_bipartite_metrics() for quality checks
#   - Validates bipartite property (no entity-entity or relation-relation edges)
#   - Reports violations and node/edge counts by type
#
# =============================================================================

# ========================
# Extraction Settings
# ========================

DEFAULT_CHUNK_SIZE = 1200
"""Default size of text chunks in tokens"""

DEFAULT_CHUNK_OVERLAP = 100
"""Default overlap between adjacent chunks in tokens"""

DEFAULT_ENTITY_TYPES = [
    "person",          # Individual humans, characters
    "organization",    # Companies, institutions, teams, groups
    "location",        # Places, geographic locations (replaces "geo")
    "event",           # Events, incidents, occurrences
    "concept",         # Abstract ideas, theories, principles
    "method",          # Processes, techniques, procedures
    "object",          # Physical objects, equipment, artifacts
    "data",            # Datasets, information, statistics
    "natural_object",  # Natural phenomena, biological entities
    "time",            # Time periods, dates, temporal references
    "category",        # Classifications, types, categories
]
"""Default entity types for extraction (expanded to match LightRAG for multilingual support)"""

DEFAULT_MAX_ENTITY_TOKENS = 6000
"""Maximum tokens for entity context"""

DEFAULT_MAX_RELATION_TOKENS = 8000
"""Maximum tokens for relation context"""

# ========================
# Graph Settings
# ========================

GRAPH_FIELD_SEP = "<SEP>"
"""Separator for multi-value fields in graph nodes"""

RELATION_PREFIX = "rel-"
"""Prefix for relation node IDs
Example: rel-8c80df8f1fc71f13c4ffbe19fa22bf8f"""

ENTITY_PREFIX = "entity-"
"""Prefix for entity node IDs (UNIFIED across all pipelines)
Example: entity-abc123 (hash-based, stable across name changes)
Both graph nodes and vector DB use entity_id with this prefix"""

CHUNK_PREFIX = "chunk-"
"""Prefix for text chunk IDs
Example: chunk-600f9c648bc602202ec663361837e416"""

DOCUMENT_PREFIX = "doc-"
"""Prefix for document IDs
Example: doc-ce2415fb73e5596b76d8f93f636c43a7"""

# Hash length for MD5-based IDs (32 hex characters)
HASH_ID_LENGTH = 32
"""Length of MD5 hash component in node IDs (32 hex characters)"""

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
