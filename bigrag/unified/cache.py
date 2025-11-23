"""
Subgraph Cache - Lazy-loading LRU cache for subgraph instances.

Manages BiGRAG instances for each subgraph with lazy loading and
LRU eviction to minimize memory usage.
"""

import asyncio
import logging
from collections import OrderedDict
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class SubgraphCache:
    """LRU cache for lazy-loaded BiGRAG subgraph instances."""

    def __init__(
        self,
        registry: Dict,
        max_size: int = 5,
        prewarm: Optional[List[str]] = None,
        bigrag_kwargs: Optional[Dict] = None
    ):
        """
        Initialize cache.

        Args:
            registry: Subgraph registry dict (from SubgraphRouter)
            max_size: Maximum number of subgraphs to keep in memory
            prewarm: Optional list of subgraph names to preload at startup
            bigrag_kwargs: Additional kwargs to pass to BiGRAG constructor
        """
        self.registry = registry
        self.max_size = max_size
        self.cache = OrderedDict()  # LRU cache: {subgraph_name: BiGRAG instance}
        self.bigrag_kwargs = bigrag_kwargs or {}

        # Track cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'loads': 0
        }

        # Store prewarm list for later (will be triggered by startup event)
        self.prewarm_list = prewarm
        if prewarm:
            logger.info(f"Prewarm list set: {prewarm} (will load during startup)")

    async def _prewarm(self, subgraph_names: List[str]):
        """Preload specified subgraphs into cache."""
        for name in subgraph_names:
            try:
                await self.get(name)
                logger.info(f"Prewarmed subgraph: {name}")
            except Exception as e:
                logger.error(f"Failed to prewarm subgraph '{name}': {e}")

    async def get(self, subgraph_name: str):
        """
        Get BiGRAG instance for subgraph (lazy load if not cached).

        Args:
            subgraph_name: Name of subgraph to load

        Returns:
            BiGRAG instance

        Raises:
            ValueError: If subgraph name not in registry
            FileNotFoundError: If subgraph directory not found
        """
        # Validate subgraph exists in registry
        if subgraph_name not in self.registry['subgraphs']:
            raise ValueError(
                f"Subgraph '{subgraph_name}' not found in registry. "
                f"Available: {list(self.registry['subgraphs'].keys())}"
            )

        # Cache hit - move to end (mark as recently used)
        if subgraph_name in self.cache:
            self.stats['hits'] += 1
            self.cache.move_to_end(subgraph_name)
            logger.debug(f"Cache HIT for subgraph '{subgraph_name}'")
            return self.cache[subgraph_name]

        # Cache miss - load from disk
        self.stats['misses'] += 1
        logger.info(f"Cache MISS for subgraph '{subgraph_name}' - loading from disk")

        subgraph_config = self.registry['subgraphs'][subgraph_name]
        subgraph_path = Path(subgraph_config['path'])

        # Validate subgraph directory exists
        if not subgraph_path.exists():
            raise FileNotFoundError(
                f"Subgraph directory not found: {subgraph_path}"
            )

        # Lazy import BiGRAG (avoid circular imports)
        from bigrag import BiGRAG

        # Load subgraph
        try:
            rag = BiGRAG(
                working_dir=str(subgraph_path),
                **self.bigrag_kwargs
            )
            self.stats['loads'] += 1

            # Add to cache
            self.cache[subgraph_name] = rag
            self.cache.move_to_end(subgraph_name)

            logger.info(
                f"Loaded subgraph '{subgraph_name}' from {subgraph_path} "
                f"(cache size: {len(self.cache)}/{self.max_size})"
            )

            # Evict LRU if cache full
            if len(self.cache) > self.max_size:
                await self._evict_lru()

            return rag

        except Exception as e:
            logger.error(f"Failed to load subgraph '{subgraph_name}': {e}")
            raise

    async def _evict_lru(self):
        """Evict least recently used subgraph from cache."""
        if not self.cache:
            return

        # Pop first item (least recently used)
        evicted_name, evicted_rag = self.cache.popitem(last=False)
        self.stats['evictions'] += 1

        logger.info(
            f"Evicted subgraph '{evicted_name}' from cache (LRU eviction) "
            f"- new cache size: {len(self.cache)}/{self.max_size}"
        )

        # Clean up BiGRAG instance to free memory
        try:
            del evicted_rag
        except Exception as e:
            logger.warning(f"Error cleaning up evicted subgraph: {e}")

    def clear(self):
        """Clear all cached subgraphs."""
        num_cleared = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {num_cleared} subgraphs from cache")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (
            self.stats['hits'] / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            **self.stats,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            'cached_subgraphs': list(self.cache.keys())
        }

    def contains(self, subgraph_name: str) -> bool:
        """Check if subgraph is currently cached (without loading it)."""
        return subgraph_name in self.cache

    async def preload(self, subgraph_names: List[str]):
        """Manually preload specified subgraphs."""
        await self._prewarm(subgraph_names)
