"""
Unified Subgraph System for BiG-RAG

This module provides a unified query interface across multiple subgraphs.
Queries are routed to relevant subgraphs using LLM-based routing, and results
are aggregated and returned.

Main Components:
- SubgraphRouter: Routes queries to relevant subgraphs using LLM analysis
- SubgraphCache: Lazy-loads and caches subgraph instances (LRU)
- UnifiedQueryExecutor: Executes queries across selected subgraphs and aggregates results

Usage:
    from bigrag.unified import UnifiedQueryExecutor

    executor = UnifiedQueryExecutor(
        registry_path="expr/subgraph_registry.json",
        llm_func=gpt_4o_mini_complete,
        max_cached_subgraphs=5
    )

    result = await executor.query("Who won the 2022 World Cup?")
"""

from .router import SubgraphRouter
from .cache import SubgraphCache
from .executor import UnifiedQueryExecutor

__all__ = [
    'SubgraphRouter',
    'SubgraphCache',
    'UnifiedQueryExecutor',
]

__version__ = '1.0.0'
