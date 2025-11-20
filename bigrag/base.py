from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar

import numpy as np

from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {
        "tokens": int,
        "content": str,
        "full_doc_id": str,
        "chunk_order_index": int,
        "doc_title": str,  # Document title for context
        "doc_metadata": dict,  # Additional metadata (tags, category, etc.)
    },
    total=False,  # Make doc_title and doc_metadata optional for backward compatibility
)

T = TypeVar("T")


@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    # Number of top-k items to retrieve; corresponds to entities in "local" mode and relationships in "global" mode.
    top_k: int = 60
    # Final output counts (applied after RRF scoring)
    num_kg_in_context: int = 15  # Number of KG items (relations) to include in final context
    num_chunks_in_context: int = 10  # Number of chunks to include in final context
    # Phase 2: Maximum number of hops for multi-hop graph traversal
    # 1-hop: Entity → Relation (single-hop reasoning)
    # 2-hop: Entity → Relation → Entity → Relation (multi-hop reasoning)
    # Recommended: 1 for single-hop datasets (NQ, PopQA, TriviaQA), 2 for multi-hop datasets (2WikiMultiHopQA, HotpotQA, Musique)
    max_hops: int = 1
    # Number of document chunks to retrieve.
    # top_n: int = 10
    # Number of tokens for the original chunks.
    max_token_for_text_unit: int = 4000
    # Number of tokens for the relationship descriptions
    max_token_for_global_context: int = 4000
    # Number of tokens for the entity descriptions
    max_token_for_local_context: int = 4000
    # Phase 3.4: Enable semantic reranking for chunk retrieval
    # Uses cross-encoder to rerank top-10 chunks → top-5 by relevance
    # Improves precision at cost of ~50-100ms latency
    # Default: False (requires sentence-transformers package and cross-encoder model)
    enable_reranking: bool = False
    # Query language override (optional)
    # If None, uses default from global_config["addon_params"]["language"]
    # Useful for mixed-language corpora or per-query language switching
    # Examples: "Bangla", "English", "Hindi", "Arabic", "Chinese"
    language: Union[str, None] = None
    # Query preprocessing control (per-query override)
    # If None, uses global default from ENABLE_QUERY_PREPROCESSING environment variable
    # Set to True to force preprocessing for this query, False to skip preprocessing
    # Useful for pre-processed queries or when you want to bypass preprocessing for specific queries
    enable_query_preprocessing: Union[bool, None] = None

    def __post_init__(self):
        """Validate QueryParam parameters at runtime"""
        valid_modes = ["local", "global", "hybrid", "naive"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {valid_modes}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.num_kg_in_context < 1:
            raise ValueError(f"num_kg_in_context must be >= 1, got {self.num_kg_in_context}")
        if self.num_chunks_in_context < 0:
            raise ValueError(f"num_chunks_in_context must be >= 0, got {self.num_chunks_in_context}")
        if not 1 <= self.max_hops <= 3:
            raise ValueError(f"max_hops must be between 1 and 3, got {self.max_hops}")
        if self.max_token_for_text_unit < 1:
            raise ValueError(f"max_token_for_text_unit must be >= 1, got {self.max_token_for_text_unit}")
        if self.max_token_for_global_context < 1:
            raise ValueError(f"max_token_for_global_context must be >= 1, got {self.max_token_for_global_context}")
        if self.max_token_for_local_context < 1:
            raise ValueError(f"max_token_for_local_context must be >= 1, got {self.max_token_for_local_context}")
        # Validate language if provided
        if self.language is not None and not isinstance(self.language, str):
            raise ValueError(f"language must be a string or None, got {type(self.language)}")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError

    async def delete(self, ids: list[str]):
        """Delete items by IDs from vector database"""
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> dict[str, T]:
        """Get multiple items by IDs. Returns dict mapping ID -> item. Missing IDs are not included."""
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def delete(self, id: str):
        """Delete a single item by ID"""
        raise NotImplementedError

    async def delete_many(self, ids: list[str]):
        """Delete multiple items by IDs"""
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in BiGRAG.")
