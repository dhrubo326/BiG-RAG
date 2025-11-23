"""
FastAPI Dependency Injection

Provides singleton instances of BiGRAG, LLM Manager, and Embedding Manager
for use across all route modules.
"""

from typing import Annotated
from fastapi import Depends

from bigrag import BiGRAG
from .managers import LLMProviderManager, EmbeddingManager


# Singleton instances (set during server startup)
_rag_instance: BiGRAG = None
_llm_manager: LLMProviderManager = None
_embedding_manager: EmbeddingManager = None
_unified_executor = None  # NEW: Unified subgraph executor
_server_start_time: float = None
_data_source: str = None
_working_dir: str = None


def set_rag_instance(rag: BiGRAG):
    """Set the global RAG instance (called during startup)"""
    global _rag_instance
    _rag_instance = rag


def set_llm_manager(manager: LLMProviderManager):
    """Set the global LLM manager (called during startup)"""
    global _llm_manager
    _llm_manager = manager


def set_embedding_manager(manager: EmbeddingManager):
    """Set the global embedding manager (called during startup)"""
    global _embedding_manager
    _embedding_manager = manager


def set_server_metadata(start_time: float, data_source: str, working_dir: str):
    """Set server metadata (called during startup)"""
    global _server_start_time, _data_source, _working_dir
    _server_start_time = start_time
    _data_source = data_source
    _working_dir = working_dir


def get_rag_instance() -> BiGRAG:
    """Get the global RAG instance"""
    if _rag_instance is None:
        raise RuntimeError("RAG instance not initialized. Server not started properly.")
    return _rag_instance


def get_llm_manager() -> LLMProviderManager:
    """Get the global LLM manager"""
    if _llm_manager is None:
        raise RuntimeError("LLM manager not initialized. Server not started properly.")
    return _llm_manager


def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager"""
    if _embedding_manager is None:
        raise RuntimeError("Embedding manager not initialized. Server not started properly.")
    return _embedding_manager


def get_server_start_time() -> float:
    """Get server start time"""
    return _server_start_time


def get_data_source() -> str:
    """Get current data source"""
    return _data_source


def get_working_dir() -> str:
    """Get working directory"""
    return _working_dir


def set_unified_executor(executor):
    """Set the global unified executor (called during startup if --unified mode)"""
    global _unified_executor
    _unified_executor = executor


def get_unified_executor():
    """Get the global unified executor (None if not in unified mode)"""
    return _unified_executor


# Type aliases for dependency injection
RAGDep = Annotated[BiGRAG, Depends(get_rag_instance)]
LLMDep = Annotated[LLMProviderManager, Depends(get_llm_manager)]
EmbeddingDep = Annotated[EmbeddingManager, Depends(get_embedding_manager)]
