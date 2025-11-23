# Unified Subgraph System Implementation Plan
**Goal:** Build multi-subgraph query routing system WITHOUT changing KG building or retrieval
**Status:** Ready to Implement
**Last Updated:** 2025-01-22

---

## Overview

**What We're Building:**
```
Query → LLM Router → Select Subgraph(s) → Existing Retrieval → Synthesized Answer
```

**What We're NOT Touching:**
- ❌ KG building process (script_build.py stays same)
- ❌ Retrieval logic (bigrag/operate.py stays same)
- ❌ Entity extraction (bigrag/operate.py stays same)

**What We're Building:**
- ✅ Subgraph registry (subgraph_registry.json)
- ✅ LLM-based router (bigrag/unified/router.py)
- ✅ Lazy loading cache (bigrag/unified/cache.py)
- ✅ Unified query executor (bigrag/unified/executor.py)
- ✅ New API endpoints (backend/server.py)
- ✅ Update server.py to support unified mode

---

## Current State (Existing Subgraphs)

You already have these subgraphs:

```
expr/
├── demo_test/              # Existing subgraph 1
│   ├── graph_chunk_entity_relation.graphml
│   ├── vdb_entities.json
│   ├── vdb_relations.json
│   ├── vdb_chunks.json
│   └── kv_store_*.json
│
├── football/               # Existing subgraph 2
│   └── ... (same structure)
│
└── kuet_test/              # Existing subgraph 3
    └── ... (same structure)
```

**Goal:** Make these 3 subgraphs work together with automatic routing.

---

## Architecture: Unified Subgraph System

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                                │
│         "কুয়েটে CSE তে কতটি আসন আছে?"                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              UNIFIED QUERY ENDPOINT                          │
│         POST /api/unified/query                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM ROUTER                                  │
│  - Reads subgraph_registry.json                             │
│  - Analyzes query + subgraph metadata                        │
│  - Decides: ["kuet_test"] or ["demo_test", "football"]      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              SUBGRAPH CACHE (Lazy Loading)                   │
│  - Check if subgraph loaded in memory                        │
│  - If not: Load from disk (expr/kuet_test/)                 │
│  - If yes: Use cached BiGRAG instance                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         EXISTING BIGRAG QUERY (NO CHANGES)                   │
│  await rag.aquery(query, param=QueryParam(...))              │
│  - Uses existing retrieval logic (operate.py)                │
│  - Three-path retrieval (Entity + Relation + Chunk)          │
│  - Returns context items                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              RESULT SYNTHESIZER                              │
│  - Aggregates results from multiple subgraphs                │
│  - Formats final response                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  RESPONSE TO USER                            │
│  {                                                           │
│    "query": "...",                                           │
│    "routing": {"subgraphs": ["kuet_test"]},                 │
│    "answer": "120 seats",                                    │
│    "context": [...]                                          │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan (3 Steps)

### Step 1: Create Subgraph Registry (Manual)
**Effort:** 15 minutes
**Files:** `expr/subgraph_registry.json` (new - manual creation)
**Actions:**
- Copy-paste JSON template to `expr/subgraph_registry.json`
- Update aliases/topics for your specific subgraphs

### Step 2: Build Unified Query System
**Effort:** 4-6 hours
**Files:** (All NEW in `bigrag/unified/` directory)
- `bigrag/unified/__init__.py` - Package exports
- `bigrag/unified/router.py` - LLM-based routing logic
- `bigrag/unified/cache.py` - Lazy loading with LRU eviction
- `bigrag/unified/executor.py` - Multi-subgraph query execution

**Actions:**
- Create `bigrag/unified/` directory
- Copy-paste 4 files from this plan
- No changes to existing `bigrag/` files

### Step 3: Update Backend Server
**Effort:** 2-3 hours
**Files:**
- `backend/server.py` (modify - add unified mode)

**Actions:**
- Add imports: `from bigrag.unified import ...`
- Add new endpoints: `/api/unified/query`, `/api/unified/subgraphs`, `/api/unified/route`
- Add `initialize_unified_mode()` function
- Add `--unified` CLI flag
- Keep existing `/search` endpoint unchanged (backward compatible)

---

## Step 1: Create Subgraph Registry

### File: `expr/subgraph_registry.json`

```json
{
  "version": "1.0",
  "created_at": "2025-01-22T10:00:00Z",
  "subgraphs": {
    "demo_test": {
      "path": "expr/demo_test",
      "description": "Demo test dataset",
      "aliases": ["demo", "test"],
      "topics": ["general"],
      "enabled": true
    },
    "football": {
      "path": "expr/football",
      "description": "Football knowledge base - players, teams, world cup",
      "aliases": ["football", "soccer", "ফুটবল"],
      "topics": ["sports", "players", "teams", "world cup", "messi", "ronaldo"],
      "enabled": true
    },
    "kuet_test": {
      "path": "expr/kuet_test",
      "description": "KUET admission information - departments, seats, eligibility",
      "aliases": ["KUET", "kuet", "কুয়েট", "Khulna University of Engineering & Technology"],
      "topics": ["admission", "seats", "departments", "faq", "process", "eligibility"],
      "enabled": true
    }
  }
}
```

**Create this file manually** (copy-paste above content).

**What it contains:**
- List of available subgraphs
- Path to each subgraph directory
- Aliases for routing (including Bangla)
- Topics/keywords for LLM routing
- Enable/disable flag

---

## Step 2: Build Unified Query System

### File 1: `bigrag/unified/__init__.py`

```python
"""
Unified subgraph query system.

Provides multi-subgraph routing and querying without changing
existing KG building or retrieval logic.
"""

from .router import SubgraphRouter
from .cache import SubgraphCache
from .executor import UnifiedQueryExecutor

__all__ = [
    "SubgraphRouter",
    "SubgraphCache",
    "UnifiedQueryExecutor"
]
```

---

### File 2: `bigrag/unified/router.py`

```python
"""
LLM-based subgraph router.

Decides which subgraph(s) to query based on user query.
"""

import json
from typing import List, Dict
from pathlib import Path


class SubgraphRouter:
    """
    Routes queries to relevant subgraphs using LLM.

    Uses subgraph_registry.json to make informed decisions.
    """

    def __init__(
        self,
        registry_path: str = "expr/subgraph_registry.json",
        llm_func = None
    ):
        """
        Initialize router.

        Args:
            registry_path: Path to subgraph registry JSON
            llm_func: LLM function for routing decisions
        """
        self.registry = self._load_registry(registry_path)
        self.llm_func = llm_func

    def _load_registry(self, path: str) -> Dict:
        """Load subgraph registry from JSON."""
        registry_path = Path(path)

        if not registry_path.exists():
            raise FileNotFoundError(
                f"Subgraph registry not found: {path}\n"
                f"Create this file with subgraph metadata."
            )

        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)

        # Filter enabled subgraphs only
        enabled_subgraphs = {
            name: info
            for name, info in registry['subgraphs'].items()
            if info.get('enabled', True)
        }

        registry['subgraphs'] = enabled_subgraphs
        return registry

    async def route(self, query: str) -> Dict:
        """
        Decide which subgraph(s) to query.

        Args:
            query: User query

        Returns:
            {
                'subgraphs': ['kuet_test'],
                'reasoning': 'Query mentions KUET',
                'confidence': 0.95
            }
        """
        # Build routing prompt
        prompt = self._build_routing_prompt(query)

        # Ask LLM
        llm_response = await self.llm_func(
            prompt,
            max_tokens=200,
            temperature=0.0  # Deterministic routing
        )

        # Parse response
        routing_decision = self._parse_routing_response(llm_response)

        return routing_decision

    def _build_routing_prompt(self, query: str) -> str:
        """Build prompt for LLM router."""

        # Format subgraph info
        subgraph_info = []
        for name, info in self.registry['subgraphs'].items():
            subgraph_info.append(
                f"- {name}: {info['description']}\n"
                f"  Aliases: {', '.join(info['aliases'])}\n"
                f"  Topics: {', '.join(info['topics'])}"
            )

        subgraphs_text = "\n\n".join(subgraph_info)

        return f"""You are a query routing agent for a multi-subgraph knowledge system.

AVAILABLE SUBGRAPHS:
{subgraphs_text}

USER QUERY:
{query}

TASK: Determine which subgraph(s) are relevant to this query.

RULES:
1. If query mentions specific subgraph (e.g., "KUET", "football"), select that subgraph
2. If query compares multiple (e.g., "compare KUET and BUET"), select all mentioned
3. If query is general, select most relevant based on topics
4. Return 1-3 most relevant subgraphs (avoid selecting all)

OUTPUT FORMAT (JSON only, no explanation):
{{
  "subgraphs": ["kuet_test"],
  "reasoning": "Brief explanation",
  "confidence": 0.95
}}

OUTPUT:"""

    def _parse_routing_response(self, llm_response: str) -> Dict:
        """Parse LLM routing response."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(0))
            else:
                decision = json.loads(llm_response)

            # Validate subgraphs exist in registry
            valid_subgraphs = [
                sg for sg in decision.get('subgraphs', [])
                if sg in self.registry['subgraphs']
            ]

            if not valid_subgraphs:
                # Fallback: return all subgraphs
                valid_subgraphs = list(self.registry['subgraphs'].keys())
                reasoning = "LLM returned invalid subgraphs, using all as fallback"
            else:
                reasoning = decision.get('reasoning', 'LLM routing decision')

            return {
                'subgraphs': valid_subgraphs,
                'reasoning': reasoning,
                'confidence': decision.get('confidence', 0.5)
            }

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: return all subgraphs
            return {
                'subgraphs': list(self.registry['subgraphs'].keys()),
                'reasoning': f'Parsing error: {e}. Using all subgraphs as fallback.',
                'confidence': 0.0
            }
```

---

### File 3: `bigrag/unified/cache.py`

```python
"""
Subgraph cache with lazy loading and LRU eviction.
"""

from collections import OrderedDict
from typing import Optional
from pathlib import Path
import asyncio


class SubgraphCache:
    """
    LRU cache for lazy-loaded subgraphs.

    Only loads subgraphs when queried, keeps recently used in memory.
    """

    def __init__(
        self,
        registry: dict,
        max_size: int = 5,
        prewarm: Optional[list] = None
    ):
        """
        Initialize cache.

        Args:
            registry: Subgraph registry dict
            max_size: Maximum subgraphs to keep in memory
            prewarm: List of subgraph names to pre-load at startup
        """
        self.registry = registry
        self.max_size = max_size
        self.cache = OrderedDict()  # {subgraph_name: BiGRAG instance}

        # Pre-warm cache in background
        if prewarm:
            asyncio.create_task(self._prewarm(prewarm))

    async def _prewarm(self, subgraph_names: list):
        """Pre-load popular subgraphs in background."""
        print(f"[Cache] Pre-warming {len(subgraph_names)} subgraphs...")
        for name in subgraph_names:
            if name in self.registry['subgraphs']:
                await self.get(name)
        print(f"[Cache] Pre-warming complete!")

    async def get(self, subgraph_name: str):
        """
        Get subgraph, loading if needed (lazy loading).

        Args:
            subgraph_name: Name of subgraph (e.g., "kuet_test")

        Returns:
            BiGRAG instance for this subgraph
        """
        # Check if in cache
        if subgraph_name in self.cache:
            # Move to end (mark as recently used)
            self.cache.move_to_end(subgraph_name)
            print(f"[Cache HIT] {subgraph_name}")
            return self.cache[subgraph_name]

        # Cache miss - load subgraph
        print(f"[Cache MISS] Loading {subgraph_name}...")

        # Get path from registry
        if subgraph_name not in self.registry['subgraphs']:
            raise ValueError(f"Subgraph '{subgraph_name}' not found in registry")

        subgraph_path = self.registry['subgraphs'][subgraph_name]['path']

        # Load BiGRAG instance (uses existing BiGRAG - NO CHANGES)
        from bigrag import BiGRAG

        rag = BiGRAG(
            working_dir=subgraph_path,
            enable_llm_cache=True
        )

        # Add to cache
        self.cache[subgraph_name] = rag

        # Evict if cache full (LRU)
        if len(self.cache) > self.max_size:
            # Remove oldest (least recently used)
            evicted_name, evicted_rag = self.cache.popitem(last=False)
            print(f"[Cache EVICT] {evicted_name} (cache full, max={self.max_size})")
            # Explicitly free memory
            del evicted_rag

        print(f"[Cache ADD] {subgraph_name} (cache size: {len(self.cache)}/{self.max_size})")
        return rag

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'cached_subgraphs': list(self.cache.keys()),
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'available_subgraphs': list(self.registry['subgraphs'].keys())
        }
```

---

### File 4: `bigrag/unified/executor.py`

```python
"""
Unified query executor - queries multiple subgraphs and aggregates results.
"""

import asyncio
from typing import List, Dict
from bigrag.base import QueryParam


class UnifiedQueryExecutor:
    """
    Executes queries across selected subgraphs and aggregates results.

    Uses existing BiGRAG query logic - NO CHANGES to retrieval.
    """

    def __init__(
        self,
        router,
        cache,
        enable_parallel: bool = True
    ):
        """
        Initialize executor.

        Args:
            router: SubgraphRouter instance
            cache: SubgraphCache instance
            enable_parallel: Query multiple subgraphs in parallel
        """
        self.router = router
        self.cache = cache
        self.enable_parallel = enable_parallel

    async def query(
        self,
        query: str,
        query_param: QueryParam = None,
        force_subgraphs: List[str] = None
    ) -> Dict:
        """
        Execute unified query.

        Args:
            query: User query
            query_param: Query parameters for BiGRAG
            force_subgraphs: Override router, query specific subgraphs

        Returns:
            {
                'query': 'original query',
                'routing': {'subgraphs': [...], 'reasoning': '...'},
                'results': [context items from all subgraphs],
                'subgraph_results': {subgraph_name: [results]}
            }
        """
        # Step 1: Route query (or use forced subgraphs)
        if force_subgraphs:
            routing_decision = {
                'subgraphs': force_subgraphs,
                'reasoning': 'Forced by user',
                'confidence': 1.0
            }
        else:
            routing_decision = await self.router.route(query)

        selected_subgraphs = routing_decision['subgraphs']

        print(f"[Unified Query] Routing to: {selected_subgraphs}")

        # Step 2: Query selected subgraphs (parallel or sequential)
        if self.enable_parallel and len(selected_subgraphs) > 1:
            # Query in parallel
            subgraph_results = await asyncio.gather(*[
                self._query_single_subgraph(
                    subgraph_name,
                    query,
                    query_param
                )
                for subgraph_name in selected_subgraphs
            ])
        else:
            # Query sequentially
            subgraph_results = []
            for subgraph_name in selected_subgraphs:
                result = await self._query_single_subgraph(
                    subgraph_name,
                    query,
                    query_param
                )
                subgraph_results.append(result)

        # Step 3: Aggregate results
        aggregated = self._aggregate_results(
            subgraph_results,
            selected_subgraphs,
            routing_decision
        )

        return {
            'query': query,
            'routing': routing_decision,
            'results': aggregated['combined_results'],
            'subgraph_results': aggregated['per_subgraph']
        }

    async def _query_single_subgraph(
        self,
        subgraph_name: str,
        query: str,
        query_param: QueryParam
    ) -> Dict:
        """Query single subgraph (uses existing BiGRAG - NO CHANGES)."""

        # Load subgraph (lazy loading with cache)
        rag = await self.cache.get(subgraph_name)

        # Query using EXISTING BiGRAG logic (NO CHANGES)
        results = await rag.aquery(
            query,
            param=query_param or QueryParam()
        )

        # Add subgraph metadata to results
        for item in results:
            item['subgraph'] = subgraph_name

        print(f"[{subgraph_name}] Retrieved {len(results)} context items")

        return {
            'subgraph': subgraph_name,
            'results': results
        }

    def _aggregate_results(
        self,
        subgraph_results: List[Dict],
        selected_subgraphs: List[str],
        routing_decision: Dict
    ) -> Dict:
        """
        Aggregate results from multiple subgraphs.

        Strategy:
        - Single subgraph: Return as-is
        - Multiple subgraphs: Interleave or group by subgraph
        """
        # Collect all results
        all_results = []
        per_subgraph = {}

        for sg_result in subgraph_results:
            subgraph_name = sg_result['subgraph']
            results = sg_result['results']

            all_results.extend(results)
            per_subgraph[subgraph_name] = results

        # Sort by coherence score (existing BiGRAG scoring)
        all_results.sort(
            key=lambda x: x.get('<coherence>', 0),
            reverse=True
        )

        return {
            'combined_results': all_results,
            'per_subgraph': per_subgraph
        }
```

---

## Step 3: Update Backend Server

### File: `backend/server.py` (Modifications)

```python
# backend/server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
from pathlib import Path

# EXISTING IMPORTS (keep these)
from bigrag import BiGRAG
from bigrag.base import QueryParam

# NEW IMPORTS (add these)
from bigrag.unified import SubgraphRouter, SubgraphCache, UnifiedQueryExecutor

app = FastAPI(title="BiG-RAG API")

# Global instances
rag_instance = None  # For single-subgraph mode
unified_executor = None  # For unified mode


# ============================================================
# EXISTING ENDPOINTS (NO CHANGES)
# ============================================================

@app.post("/search")
async def search(queries: List[str], subgraph: Optional[str] = None):
    """
    EXISTING endpoint - works with single subgraph mode.

    Usage: python server.py --data_source demo_test
    """
    global rag_instance

    if rag_instance is None:
        raise HTTPException(
            status_code=500,
            detail="Server not initialized. Start with --data_source or --unified"
        )

    # Use existing BiGRAG query (NO CHANGES)
    results = await rag_instance.aquery(queries[0])
    return {"results": results}


# ============================================================
# NEW UNIFIED ENDPOINTS
# ============================================================

class UnifiedQueryRequest(BaseModel):
    query: str
    language: Optional[str] = "auto"
    force_subgraphs: Optional[List[str]] = None
    num_context: Optional[int] = 10


@app.post("/api/unified/query")
async def unified_query(request: UnifiedQueryRequest):
    """
    NEW unified query endpoint.

    Automatically routes to relevant subgraph(s) and returns results.

    Usage: python server.py --unified
    """
    global unified_executor

    if unified_executor is None:
        raise HTTPException(
            status_code=500,
            detail="Unified mode not enabled. Start with --unified flag"
        )

    # Execute unified query
    result = await unified_executor.query(
        query=request.query,
        query_param=QueryParam(
            only_need_context=True,
            num_kg_in_context=request.num_context,
            num_chunks_in_context=request.num_context
        ),
        force_subgraphs=request.force_subgraphs
    )

    return result


@app.get("/api/unified/subgraphs")
async def list_subgraphs():
    """List available subgraphs."""
    global unified_executor

    if unified_executor is None:
        raise HTTPException(
            status_code=500,
            detail="Unified mode not enabled"
        )

    return unified_executor.cache.get_stats()


@app.post("/api/unified/route")
async def test_routing(query: str):
    """Test routing decision without executing query (debug)."""
    global unified_executor

    if unified_executor is None:
        raise HTTPException(
            status_code=500,
            detail="Unified mode not enabled"
        )

    routing_decision = await unified_executor.router.route(query)
    return routing_decision


# ============================================================
# STARTUP LOGIC
# ============================================================

def initialize_single_mode(data_source: str):
    """Initialize single subgraph mode (EXISTING)."""
    global rag_instance

    working_dir = f"expr/{data_source}"

    print(f"[Single Mode] Loading subgraph: {data_source}")
    print(f"[Single Mode] Working directory: {working_dir}")

    rag_instance = BiGRAG(
        working_dir=working_dir,
        enable_llm_cache=True
    )

    print(f"[Single Mode] Ready! Use /search endpoint")


def initialize_unified_mode(
    registry_path: str = "expr/subgraph_registry.json",
    max_cached: int = 5,
    prewarm: List[str] = None
):
    """Initialize unified multi-subgraph mode (NEW)."""
    global unified_executor

    print(f"[Unified Mode] Initializing...")
    print(f"[Unified Mode] Registry: {registry_path}")

    # Import LLM function (use existing from BiGRAG)
    from bigrag.llm import gpt_4o_mini_complete

    # Initialize router
    router = SubgraphRouter(
        registry_path=registry_path,
        llm_func=gpt_4o_mini_complete
    )

    print(f"[Unified Mode] Found {len(router.registry['subgraphs'])} subgraphs:")
    for name in router.registry['subgraphs'].keys():
        print(f"  - {name}")

    # Initialize cache
    cache = SubgraphCache(
        registry=router.registry,
        max_size=max_cached,
        prewarm=prewarm
    )

    # Initialize executor
    unified_executor = UnifiedQueryExecutor(
        router=router,
        cache=cache,
        enable_parallel=True
    )

    print(f"[Unified Mode] Ready! Use /api/unified/* endpoints")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, help="Single subgraph mode")
    parser.add_argument("--unified", action="store_true", help="Enable unified mode")
    parser.add_argument("--registry", type=str, default="expr/subgraph_registry.json")
    parser.add_argument("--max_cached", type=int, default=5)
    parser.add_argument("--prewarm", type=str, nargs="+", help="Pre-load subgraphs")
    parser.add_argument("--port", type=int, default=8001)

    args = parser.parse_args()

    # Initialize based on mode
    if args.unified:
        initialize_unified_mode(
            registry_path=args.registry,
            max_cached=args.max_cached,
            prewarm=args.prewarm
        )
    elif args.data_source:
        initialize_single_mode(args.data_source)
    else:
        print("ERROR: Specify --data_source or --unified")
        exit(1)

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

---

## Usage Examples

### Mode 1: Single Subgraph (EXISTING - NO CHANGES)

```bash
# Start server for one subgraph
cd backend
python server.py --data_source demo_test

# Query
curl -X POST http://localhost:8001/search \
  -d '{"queries": ["test query"]}'
```

---

### Mode 2: Unified Multi-Subgraph (NEW)

```bash
# Start unified server
cd backend
python server.py --unified

# Query with automatic routing
curl -X POST http://localhost:8001/api/unified/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "কুয়েটে CSE তে কতটি আসন আছে?"
  }'

# Response
{
  "query": "কুয়েটে CSE তে কতটি আসন আছে?",
  "routing": {
    "subgraphs": ["kuet_test"],
    "reasoning": "Query mentions KUET",
    "confidence": 0.95
  },
  "results": [
    {
      "content": "KUET CSE has 120 seats",
      "subgraph": "kuet_test",
      "type": "relation",
      "coherence": 0.95
    }
  ]
}
```

---

### Mode 3: Unified with Pre-warming

```bash
# Pre-load popular subgraphs at startup
cd backend
python server.py --unified --prewarm kuet_test demo_test

# First query to kuet_test: ~150ms (pre-warmed, no loading)
# First query to football: ~1500ms (not pre-warmed, needs loading)
```

---

### Mode 4: Force Specific Subgraphs

```bash
# Override router, query specific subgraphs
curl -X POST http://localhost:8001/api/unified/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "CSE seats",
    "force_subgraphs": ["kuet_test", "demo_test"]
  }'
```

---

## Testing Plan

### Test 1: Single Subgraph Mode (Ensure No Regression)

```bash
# Should work exactly as before
python server.py --data_source demo_test
curl -X POST http://localhost:8001/search -d '{"queries": ["test"]}'
```

**Expected:** Works same as current system (no changes).

---

### Test 2: Unified Mode Basic Query

```bash
# Start unified
python server.py --unified

# Test KUET query
curl -X POST http://localhost:8001/api/unified/query \
  -d '{"query": "KUET CSE seats"}'

# Expected routing: ["kuet_test"]
```

---

### Test 3: Multi-Subgraph Query

```bash
# Query that needs multiple subgraphs
curl -X POST http://localhost:8001/api/unified/query \
  -d '{"query": "Compare demo_test and football"}'

# Expected routing: ["demo_test", "football"]
```

---

### Test 4: List Subgraphs

```bash
curl http://localhost:8001/api/unified/subgraphs

# Expected:
{
  "cached_subgraphs": ["kuet_test"],
  "cache_size": 1,
  "max_size": 5,
  "available_subgraphs": ["demo_test", "football", "kuet_test"]
}
```

---

### Test 5: Test Routing (Debug)

```bash
curl -X POST http://localhost:8001/api/unified/route \
  -d 'query=কুয়েটে CSE তে কতটি আসন আছে?'

# Expected:
{
  "subgraphs": ["kuet_test"],
  "reasoning": "Query mentions KUET (কুয়েট) and CSE",
  "confidence": 0.95
}
```

---

## File Structure Summary

```
BiG-RAG/
├── expr/
│   ├── subgraph_registry.json          # NEW - Manual creation
│   ├── demo_test/                       # Existing subgraph
│   ├── football/                        # Existing subgraph
│   └── kuet_test/                       # Existing subgraph
│
├── bigrag/
│   ├── unified/                         # NEW directory
│   │   ├── __init__.py                  # NEW
│   │   ├── router.py                    # NEW - LLM routing
│   │   ├── cache.py                     # NEW - Lazy loading
│   │   └── executor.py                  # NEW - Query execution
│   │
│   ├── bigrag.py                        # NO CHANGES
│   ├── operate.py                       # NO CHANGES
│   └── ...
│
├── backend/
│   └── server.py                        # MODIFIED - Add unified mode
│
└── UNIFIED_SUBGRAPH_IMPLEMENTATION_PLAN.md  # This file
```

---

## Implementation Checklist

### Phase 1: Setup (15 min)
- [ ] Create `expr/subgraph_registry.json` (copy-paste from plan)
- [ ] Update aliases/topics for demo_test, football, kuet_test
- [ ] Verify all 3 subgraphs exist in `expr/` directory

### Phase 2: Build Unified System (4-6 hours)
- [ ] Create directory: `mkdir bigrag/unified`
- [ ] Create `bigrag/unified/__init__.py` (copy-paste from plan)
- [ ] Create `bigrag/unified/router.py` (copy-paste from plan)
- [ ] Create `bigrag/unified/cache.py` (copy-paste from plan)
- [ ] Create `bigrag/unified/executor.py` (copy-paste from plan)
- [ ] Test imports: `python -c "from bigrag.unified import UnifiedQueryExecutor"`

### Phase 3: Update Server (2-3 hours)
- [ ] Backup current `backend/server.py` (copy to `server.py.backup`)
- [ ] Modify `backend/server.py` (add unified mode code from plan)
- [ ] Add new endpoints: `/api/unified/query`, `/api/unified/subgraphs`, `/api/unified/route`
- [ ] Add startup logic: `initialize_unified_mode()` function
- [ ] Test imports: `python backend/server.py --help`

### Phase 4: Testing (2 hours)
- [ ] **Test 1**: Single mode (no regression)
  - Run: `python server.py --data_source demo_test`
  - Query: `curl -X POST http://localhost:8001/search -d '{"queries": ["test"]}'`
  - Expected: Works same as before

- [ ] **Test 2**: Unified mode startup
  - Run: `python server.py --unified`
  - Check console: Should list 3 subgraphs (demo_test, football, kuet_test)
  - Expected: No errors, server starts

- [ ] **Test 3**: List subgraphs
  - Query: `curl http://localhost:8001/api/unified/subgraphs`
  - Expected: Returns list of 3 subgraphs

- [ ] **Test 4**: Test routing
  - Query: `curl -X POST http://localhost:8001/api/unified/route -d 'query=KUET CSE seats'`
  - Expected: Routes to `["kuet_test"]`

- [ ] **Test 5**: Unified query
  - Query: `curl -X POST http://localhost:8001/api/unified/query -d '{"query": "KUET CSE seats"}'`
  - Expected: Returns results from kuet_test subgraph

- [ ] **Test 6**: Multi-subgraph query
  - Query: `curl -X POST http://localhost:8001/api/unified/query -d '{"query": "Compare demo_test and football"}'`
  - Expected: Routes to both subgraphs, returns combined results

### Phase 5: Documentation (Optional - 1 hour)
- [ ] Update README with unified mode usage
- [ ] Document API endpoints in API docs

**Total Estimated Time:** 8-12 hours (without documentation: 6-11 hours)

---

## No Changes to These Files

✅ Keep existing files unchanged:
- `script_build.py` - KG building stays same
- `bigrag/bigrag.py` - Core BiGRAG stays same
- `bigrag/operate.py` - Retrieval logic stays same
- `bigrag/extractors/` - Entity extraction stays same
- All existing endpoints work as before

---

## Summary

**What We're Building:**
1. ✅ Subgraph registry (JSON file)
2. ✅ LLM-based router
3. ✅ Lazy loading cache
4. ✅ Unified query executor
5. ✅ New API endpoints

**What We're NOT Changing:**
- ❌ KG building (script_build.py)
- ❌ Retrieval (bigrag/operate.py)
- ❌ Entity extraction (bigrag/extractors/)

**Benefits:**
- 🎯 Works with existing subgraphs (demo_test, football, kuet_test)
- 🎯 No changes to existing code
- 🎯 Automatic routing with LLM
- 🎯 Memory efficient (lazy loading)
- 🎯 One port for all subgraphs

**Ready to start implementation?**
