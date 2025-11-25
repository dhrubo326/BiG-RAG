"""
Unified Query Executor - Executes queries across selected subgraphs.

Coordinates routing, subgraph loading, query execution, and result aggregation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path

from .router import SubgraphRouter
from .cache import SubgraphCache
from bigrag import BiGRAG
from bigrag.base import QueryParam

logger = logging.getLogger(__name__)


class UnifiedQueryExecutor:
    """Executes queries across multiple subgraphs with routing and aggregation."""

    def __init__(
        self,
        registry_path: str = "expr/subgraph_registry.json",
        llm_func: Optional[Callable] = None,
        max_cached_subgraphs: int = 5,
        prewarm_subgraphs: Optional[List[str]] = None,
        enable_parallel: bool = True,
        bigrag_kwargs: Optional[Dict] = None
    ):
        """
        Initialize unified executor.

        Args:
            registry_path: Path to subgraph_registry.json
            llm_func: LLM function for routing (e.g., gpt_4o_mini_complete)
            max_cached_subgraphs: Max subgraphs to keep in memory
            prewarm_subgraphs: Subgraphs to preload at startup
            enable_parallel: Enable parallel querying of multiple subgraphs
            bigrag_kwargs: Additional kwargs for BiGRAG instances
        """
        self.registry_path = Path(registry_path)
        self.enable_parallel = enable_parallel

        # Initialize router
        self.router = SubgraphRouter(
            registry_path=str(self.registry_path),
            llm_func=llm_func
        )

        # Initialize cache
        self.cache = SubgraphCache(
            registry=self.router.registry,
            max_size=max_cached_subgraphs,
            prewarm=prewarm_subgraphs,
            bigrag_kwargs=bigrag_kwargs or {}
        )

        logger.info(
            f"Initialized UnifiedQueryExecutor with {len(self.router.list_subgraphs())} "
            f"subgraphs (cache_size={max_cached_subgraphs}, "
            f"parallel={enable_parallel})"
        )

    async def query(
        self,
        query: str,
        query_param: Optional[Any] = None,
        force_subgraphs: Optional[List[str]] = None,
        include_metadata: bool = True
    ) -> Dict:
        """
        Execute unified query across relevant subgraphs.

        Args:
            query: User query string
            query_param: QueryParam instance for BiGRAG (if None, uses defaults)
            force_subgraphs: Force specific subgraphs (bypass routing)
            include_metadata: Include routing/execution metadata in response

        Returns:
            Dict with keys:
                - query: Original query
                - routing: Routing decision metadata
                - results: List of combined results from all subgraphs
                - subgraph_results: Per-subgraph results (if include_metadata=True)
                - execution_time: Total execution time (if include_metadata=True)
        """
        start_time = time.time()

        # Step 1: Route query to relevant subgraph(s)
        routing_decision = await self.router.route(query, force_subgraphs=force_subgraphs)
        selected_subgraphs = routing_decision['subgraphs']

        logger.info(
            f"Query routed to {len(selected_subgraphs)} subgraph(s): "
            f"{selected_subgraphs}"
        )

        # Step 2: Query selected subgraphs (parallel or sequential)
        if self.enable_parallel and len(selected_subgraphs) > 1:
            subgraph_results = await self._query_parallel(
                selected_subgraphs, query, query_param
            )
        else:
            subgraph_results = await self._query_sequential(
                selected_subgraphs, query, query_param
            )

        # Step 3: Aggregate results
        aggregated = self._aggregate_results(subgraph_results)

        execution_time = time.time() - start_time

        # Build response
        response = {
            'query': query,
            'results': aggregated['combined_results']
        }

        if include_metadata:
            response['routing'] = routing_decision
            response['subgraph_results'] = aggregated['per_subgraph']
            response['execution_time'] = execution_time
            response['cache_stats'] = self.cache.get_stats()

        logger.info(
            f"Unified query completed in {execution_time:.2f}s - "
            f"returned {len(response['results'])} results from "
            f"{len(selected_subgraphs)} subgraph(s)"
        )

        return response

    async def _query_single_subgraph(
        self,
        subgraph_name: str,
        query: str,
        query_param: Optional[Any]
    ) -> Dict:
        """Query a single subgraph and return results with metadata."""
        try:
            start_time = time.time()

            # TEMPORARY: Load BiGRAG directly without caching to avoid issues
            subgraph_config = self.router.registry['subgraphs'][subgraph_name]
            subgraph_path = Path(subgraph_config['path'])

            # FIX: Convert relative paths to absolute (resolve from project root)
            if not subgraph_path.is_absolute():
                # Get project root: bigrag/unified/executor.py -> bigrag/ -> BiG-RAG/
                project_root = Path(__file__).parent.parent.parent
                subgraph_path = (project_root / subgraph_path).resolve()

            # Load BiGRAG instance
            rag = BiGRAG(
                working_dir=str(subgraph_path),
                **self.cache.bigrag_kwargs
            )

            # Use default QueryParam if not provided
            if query_param is None:
                query_param = QueryParam(
                    only_need_context=True,
                    top_k=10
                )

            # Execute query
            results = await rag.aquery(query, param=query_param)

            execution_time = time.time() - start_time

            return {
                'subgraph': subgraph_name,
                'success': True,
                'results': results,
                'num_results': len(results),
                'execution_time': execution_time,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error querying subgraph '{subgraph_name}': {e}")
            return {
                'subgraph': subgraph_name,
                'success': False,
                'results': [],
                'num_results': 0,
                'execution_time': 0.0,
                'error': str(e)
            }

    async def _query_parallel(
        self,
        subgraph_names: List[str],
        query: str,
        query_param: Optional[Any]
    ) -> List[Dict]:
        """Query multiple subgraphs in parallel."""
        logger.info(f"Querying {len(subgraph_names)} subgraphs in PARALLEL")

        tasks = [
            self._query_single_subgraph(sg, query, query_param)
            for sg in subgraph_names
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Exception querying subgraph '{subgraph_names[i]}': {result}"
                )
                processed_results.append({
                    'subgraph': subgraph_names[i],
                    'success': False,
                    'results': [],
                    'num_results': 0,
                    'execution_time': 0.0,
                    'error': str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _query_sequential(
        self,
        subgraph_names: List[str],
        query: str,
        query_param: Optional[Any]
    ) -> List[Dict]:
        """Query multiple subgraphs sequentially."""
        logger.info(f"Querying {len(subgraph_names)} subgraph(s) SEQUENTIALLY")

        results = []
        for sg_name in subgraph_names:
            result = await self._query_single_subgraph(sg_name, query, query_param)
            results.append(result)

        return results

    def _aggregate_results(self, subgraph_results: List[Dict]) -> Dict:
        """
        Aggregate results from multiple subgraphs.

        Simple concatenation strategy - more sophisticated ranking/deduplication
        can be added later.
        """
        combined_results = []
        per_subgraph = {}

        for sg_result in subgraph_results:
            subgraph_name = sg_result['subgraph']
            per_subgraph[subgraph_name] = sg_result

            if sg_result['success']:
                # Add subgraph metadata to each result
                for result in sg_result['results']:
                    result_with_meta = result.copy() if isinstance(result, dict) else result
                    if isinstance(result_with_meta, dict):
                        result_with_meta['_subgraph'] = subgraph_name
                        # Normalize coherence score to 'score' for consistent sorting
                        # BiGRAG returns '<coherence>' key, but API consumers expect 'score'
                        if '<coherence>' in result_with_meta and 'score' not in result_with_meta:
                            result_with_meta['score'] = result_with_meta['<coherence>']
                    combined_results.append(result_with_meta)

        # Sort by relevance score (descending)
        try:
            combined_results.sort(
                key=lambda x: x.get('score', 0.0) if isinstance(x, dict) else 0.0,
                reverse=True
            )
        except Exception:
            # If sorting fails, keep original order
            pass

        return {
            'combined_results': combined_results,
            'per_subgraph': per_subgraph
        }

    def get_available_subgraphs(self) -> List[str]:
        """List all available subgraph names."""
        return self.router.list_subgraphs()

    def get_subgraph_info(self, subgraph_name: str) -> Optional[Dict]:
        """Get metadata for a specific subgraph."""
        return self.router.get_subgraph_info(subgraph_name)

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear subgraph cache."""
        self.cache.clear()

    def reload_registry(self):
        """Reload subgraph registry from disk."""
        self.router.reload_registry()
        # Update cache registry reference
        self.cache.registry = self.router.registry
        logger.info("Registry reloaded in executor and cache")
