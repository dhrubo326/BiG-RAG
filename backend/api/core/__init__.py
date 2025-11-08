"""
Core module for BiG-RAG API

Provides dependency injection, manager classes, and configuration.
"""

from .managers import LLMProviderManager, EmbeddingManager
from .dependencies import (
    get_rag_instance, get_llm_manager, get_embedding_manager,
    set_rag_instance, set_llm_manager, set_embedding_manager,
    set_server_metadata, get_server_start_time, get_data_source, get_working_dir,
    RAGDep, LLMDep, EmbeddingDep
)

__all__ = [
    "LLMProviderManager",
    "EmbeddingManager",
    "get_rag_instance",
    "get_llm_manager",
    "get_embedding_manager",
    "set_rag_instance",
    "set_llm_manager",
    "set_embedding_manager",
    "set_server_metadata",
    "get_server_start_time",
    "get_data_source",
    "get_working_dir",
    "RAGDep",
    "LLMDep",
    "EmbeddingDep",
]
