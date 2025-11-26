# Unified Routing Enhancement Plan

**Date**: January 26, 2025
**Status**: PLAN MODE - For Review
**Context**: Improving unified routing system for production robustness

---

## Executive Summary

This plan addresses four critical improvements to the unified routing system:

1. **LRU Cache Enhancement**: Increase capacity from 5 to 10 subgraphs with proper pre-loading
2. **Topics Field Analysis**: Address mis-routing risks with similar university names
3. **Dynamic Registry Management**: API endpoints for CRUD operations on registry
4. **Hot-Reload Support**: Enable dynamic subgraph creation without server restart

---

## Current State Analysis

### 1. Cache System (`bigrag/unified/cache.py`)

**Current Implementation**:
- Uses `OrderedDict` for LRU eviction (GOOD)
- `max_size = 5` (default) - **TOO SMALL**
- Lazy loading on cache miss (GOOD)
- Prewarm support exists but incomplete

**Problems**:
- Production servers with many datasets will thrash cache with only 5 slots
- Pre-loading only happens if explicitly called (not automatic on startup)

**Code Reference**: [bigrag/unified/cache.py:23](d:\BiG-RAG\bigrag\unified\cache.py#L23)

```python
def __init__(self, max_size: int = 5, working_dir_base: str = "expr"):
    self.max_size = max_size  # ← Currently 5, need 10
```

---

### 2. Router System (`bigrag/unified/router.py`)

**Current Implementation**:
- LLM-based semantic routing using GPT-4o-mini
- Routing prompt includes: **description + aliases + topics**

**CRITICAL FINDING**: Topics ARE used in routing (line 75):

```python
Topics: {', '.join(config['topics'][:10])}  # Limit to first 10 topics
```

**Problems**:
- Similar universities will have overlapping topics:
  - KUET: `["CSE", "EEE", "admissions", "departments", ...]`
  - BUET: `["CSE", "EEE", "admissions", "departments", ...]` (hypothetical)
  - Question: "How many CSE seats?" → LLM might pick wrong university

**Routing Prompt** (lines 60-97):
```python
Available subgraphs:
1. football
   Description: Knowledge graph about football
   Aliases: football, soccer
   Topics: Lionel Messi, Barcelona, ...

2. kuet_test
   Description: KUET educational content
   Aliases: kuet, kuet_test
   Topics: CSE, EEE, admissions, departments, ...
```

**Code Reference**: [bigrag/unified/router.py:75](d:\BiG-RAG\bigrag\unified\router.py#L75)

---

### 3. Registry Structure (`expr/subgraph_registry.json`)

**Current Structure**:
```json
{
  "subgraphs": {
    "football": {
      "path": "expr/football",
      "description": "Knowledge graph about football",
      "aliases": ["football", "soccer"],
      "topics": [
        "Lionel Messi", "Barcelona", "Champions League",
        "La Liga", "Copa del Rey", "..."
      ],
      "enabled": true
    },
    "kuet_test": {
      "path": "expr/kuet_test",
      "description": "KUET educational content",
      "aliases": ["kuet", "kuet_test"],
      "topics": [
        "CSE", "EEE", "ME", "CE", "IPE", "admissions",
        "departments", "faculty", "curriculum", "..."
      ],
      "enabled": true
    }
  }
}
```

**Problems**:
- No API for CRUD operations (manual JSON editing required)
- No validation on updates
- Changes require server restart or manual `reload_registry()` call

---

### 4. Server Initialization (`backend/server.py`)

**Current Implementation** (lines 60-80):
```python
if args.unified:
    unified_executor = UnifiedQueryExecutor(
        registry_path=registry_path,
        working_dir_base=working_dir_base,
        llm_model_func=gpt_4o_mini_complete,
        max_cached=args.max_cached,  # Default: 5
        addon_params=addon_params
    )

    if args.prewarm:
        subgraphs = args.prewarm.split(',')
        await unified_executor.prewarm(subgraphs)
```

**Problems**:
- max_cached default is 5 (line 73)
- Prewarm only runs if `--prewarm` flag provided
- No automatic preloading based on registry

---

## Proposed Solutions

### Solution 1: LRU Cache Enhancement

**Goal**: Support up to 10 subgraphs with automatic pre-loading

#### Changes Required:

**A. Update Default max_size** (`backend/server.py:73`):
```python
# OLD
parser.add_argument('--max_cached', type=int, default=5)

# NEW
parser.add_argument('--max_cached', type=int, default=10,
                    help="Maximum number of subgraphs to cache (LRU policy)")
```

**B. Implement Smart Pre-warming** (`bigrag/unified/executor.py`):

Add new method `_auto_prewarm()`:
```python
async def _auto_prewarm(self, top_n: int = 10):
    """
    Automatically pre-load most important subgraphs on startup.

    Priority order:
    1. Subgraphs with auto_created=false (manually curated)
    2. Most recently created (by created_at timestamp)
    3. Up to top_n subgraphs
    """
    enabled_subgraphs = [
        (name, config)
        for name, config in self.registry["subgraphs"].items()
        if config.get("enabled", True)
    ]

    # Sort by: manual-first, then by created_at descending
    def sort_key(item):
        name, config = item
        is_manual = not config.get("auto_created", False)
        created_at = config.get("created_at", "1970-01-01T00:00:00")
        return (is_manual, created_at)

    sorted_subgraphs = sorted(enabled_subgraphs, key=sort_key, reverse=True)
    top_subgraphs = [name for name, _ in sorted_subgraphs[:top_n]]

    logger.info(f"[Auto-Prewarm] Pre-loading {len(top_subgraphs)} subgraphs: {top_subgraphs}")
    await self.cache.prewarm(top_subgraphs)
```

**C. Call Auto-Prewarm on Startup** (`backend/server.py`):
```python
if args.unified:
    unified_executor = UnifiedQueryExecutor(...)

    # Auto-prewarm enabled subgraphs
    await unified_executor._auto_prewarm(top_n=args.max_cached)

    # Manual prewarm still supported for override
    if args.prewarm:
        subgraphs = args.prewarm.split(',')
        await unified_executor.prewarm(subgraphs)
```

**Impact**:
- Production servers start with 10 most important subgraphs pre-loaded
- Reduces cold-start latency
- Manual prewarm still available for testing specific subgraphs

---

### Solution 2: Topics Field - Three Options

#### Option A: Remove Topics from Routing (RECOMMENDED)

**Rationale**: Topics cause ambiguity for similar domains (universities, companies, etc.)

**Changes** (`bigrag/unified/router.py:60-97`):
```python
# OLD (line 75)
Topics: {', '.join(config['topics'][:10])}

# NEW (remove topics entirely)
# (Remove line 75 completely)
```

**Updated Routing Prompt**:
```python
Available subgraphs:
1. football
   Description: Knowledge graph about football/soccer including teams, players, leagues
   Aliases: football, soccer

2. kuet_test
   Description: KUET (Khulna University of Engineering & Technology) educational content
   Aliases: kuet, kuet_test
```

**Impact**:
- Removes ambiguity for queries like "How many CSE seats?"
- Forces descriptions to be more specific (GOOD)
- Aliases provide additional matching hints

**Migration**: Enhance descriptions to be more specific:
```json
{
  "kuet_test": {
    "description": "KUET (Khulna University of Engineering & Technology) educational content including admissions, departments (CSE, EEE, ME, CE, IPE), curriculum, and faculty information",
    "aliases": ["kuet", "kuet_test", "Khulna University"]
  }
}
```

---

#### Option B: Use Topics with Subgraph Name Prefix

**Rationale**: Keep topics but disambiguate with subgraph name

**Changes** (`expr/subgraph_registry.json`):
```json
{
  "kuet_test": {
    "topics": [
      "KUET CSE", "KUET EEE", "KUET admissions",
      "KUET departments", "KUET curriculum"
    ]
  },
  "buet_test": {
    "topics": [
      "BUET CSE", "BUET EEE", "BUET admissions",
      "BUET departments", "BUET curriculum"
    ]
  }
}
```

**Impact**:
- Keeps topics for specificity
- Adds overhead to maintain prefixed topics
- Requires updating all existing registries

---

#### Option C: Add Domain Hint Field

**Rationale**: Separate field for domain classification

**Changes** (`bigrag/unified/router.py`):
```python
# New field in registry
{
  "kuet_test": {
    "domain": "education/university/bangladesh",
    "subdomain": "KUET"
  }
}

# Routing prompt includes domain
Domain: {config.get('domain', 'general')} > {config.get('subdomain', '')}
```

**Impact**:
- Hierarchical domain classification
- More complex routing logic
- Requires schema updates

---

**RECOMMENDATION**: **Option A** (Remove Topics)

- Simplest solution
- Forces better descriptions
- Eliminates ambiguity
- No schema changes needed

---

### Solution 3: Dynamic Registry Management API

**Goal**: Enable CRUD operations on registry without server restart

#### New Endpoints (`backend/api/routes/unified.py`)

**A. Get Registry**
```python
@router.get("/registry", summary="Get all subgraphs in registry")
async def get_registry():
    """
    Returns current subgraph registry.

    Example response:
    {
      "version": "1.0",
      "subgraphs": {
        "football": {...},
        "kuet_test": {...}
      }
    }
    """
    executor = get_unified_executor()
    if not executor:
        raise HTTPException(503, "Unified mode not enabled")

    return executor.registry
```

**B. Get Single Subgraph**
```python
@router.get("/registry/{subgraph_name}", summary="Get subgraph details")
async def get_subgraph(subgraph_name: str):
    """
    Returns details for a specific subgraph.

    Returns 404 if subgraph not found.
    """
    executor = get_unified_executor()
    if not executor:
        raise HTTPException(503, "Unified mode not enabled")

    if subgraph_name not in executor.registry["subgraphs"]:
        raise HTTPException(404, f"Subgraph '{subgraph_name}' not found")

    return executor.registry["subgraphs"][subgraph_name]
```

**C. Update Subgraph Metadata**
```python
class SubgraphUpdateRequest(BaseModel):
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    enabled: Optional[bool] = None

@router.put("/registry/{subgraph_name}", summary="Update subgraph metadata")
async def update_subgraph(subgraph_name: str, request: SubgraphUpdateRequest):
    """
    Update subgraph metadata (description, aliases, topics, enabled).

    Does NOT modify graph files, only registry metadata.
    Changes take effect immediately without restart.

    Example:
    PUT /api/unified/registry/kuet_test
    {
      "description": "Updated description",
      "enabled": false
    }
    """
    executor = get_unified_executor()
    if not executor:
        raise HTTPException(503, "Unified mode not enabled")

    if subgraph_name not in executor.registry["subgraphs"]:
        raise HTTPException(404, f"Subgraph '{subgraph_name}' not found")

    # Update fields
    config = executor.registry["subgraphs"][subgraph_name]
    if request.description is not None:
        config["description"] = request.description
    if request.aliases is not None:
        config["aliases"] = request.aliases
    if request.topics is not None:
        config["topics"] = request.topics
    if request.enabled is not None:
        config["enabled"] = request.enabled

    # Save to disk
    registry_path = Path("expr/subgraph_registry.json")
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(executor.registry, f, indent=2, ensure_ascii=False)

    # Hot-reload registry and router
    executor.reload_registry()

    # Clear cache for this subgraph (force reload on next query)
    if subgraph_name in executor.cache.cache:
        del executor.cache.cache[subgraph_name]
        logger.info(f"[Registry] Cleared cache for updated subgraph: {subgraph_name}")

    return {
        "success": True,
        "message": f"Subgraph '{subgraph_name}' updated successfully",
        "updated_config": config
    }
```

**D. Delete Subgraph**
```python
@router.delete("/registry/{subgraph_name}", summary="Remove subgraph from registry")
async def delete_subgraph(
    subgraph_name: str,
    delete_files: bool = False
):
    """
    Remove subgraph from registry.

    Parameters:
    - delete_files: If true, also delete graph files from expr/ directory

    WARNING: If delete_files=true, this is irreversible!
    """
    executor = get_unified_executor()
    if not executor:
        raise HTTPException(503, "Unified mode not enabled")

    if subgraph_name not in executor.registry["subgraphs"]:
        raise HTTPException(404, f"Subgraph '{subgraph_name}' not found")

    # Remove from registry
    config = executor.registry["subgraphs"].pop(subgraph_name)

    # Save to disk
    registry_path = Path("expr/subgraph_registry.json")
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(executor.registry, f, indent=2, ensure_ascii=False)

    # Hot-reload
    executor.reload_registry()

    # Clear from cache
    if subgraph_name in executor.cache.cache:
        del executor.cache.cache[subgraph_name]

    # Delete files if requested
    if delete_files:
        import shutil
        graph_dir = Path(config["path"])
        if graph_dir.exists():
            shutil.rmtree(graph_dir)
            logger.info(f"[Registry] Deleted graph directory: {graph_dir}")

    return {
        "success": True,
        "message": f"Subgraph '{subgraph_name}' removed from registry",
        "files_deleted": delete_files
    }
```

**Impact**:
- Full CRUD operations via REST API
- No manual JSON editing required
- Changes take effect immediately
- Production-safe

---

### Solution 4: Hot-Reload for Dynamic Subgraph Creation

**Goal**: `/datasets/create-and-index` endpoint automatically registers new subgraphs

#### Current Flow (`backend/api/routes/datasets.py:342-349`):

```python
# Step 12: Reload registry in unified executor (if dataset was just added)
if dataset_info["registry_updated"]:
    try:
        unified_executor.reload_registry()
        logger.info(f"[Create-and-Index] Reloaded unified executor registry")
    except Exception as e:
        logger.warning(f"[Create-and-Index] Failed to reload registry: {e}")
```

**GOOD**: Already calls `reload_registry()` after adding new dataset!

#### Enhancement Required:

**A. Add Cache Prewarm for New Subgraph** (`backend/api/routes/datasets.py`):
```python
# Step 12: Reload registry in unified executor (if dataset was just added)
if dataset_info["registry_updated"]:
    try:
        unified_executor.reload_registry()
        logger.info(f"[Create-and-Index] Reloaded unified executor registry")

        # NEW: Pre-load the new subgraph immediately
        await unified_executor.cache.get(data_source)
        logger.info(f"[Create-and-Index] Pre-loaded new subgraph: {data_source}")

    except Exception as e:
        logger.warning(f"[Create-and-Index] Failed to reload registry: {e}")
```

**B. Add Graph Build Completion Check**:

**Problem**: Currently, indexing happens in background. Registry updated immediately, but graph files not ready yet.

**Solution**: Add status tracking to ProcessingJob:
```python
# In backend/api/services/jobs.py
class ProcessingJob:
    status: JobStatus  # PENDING, INDEXING, BUILDING_GRAPH, COMPLETED, FAILED
    graph_ready: bool = False  # NEW field
```

**C. Update `/datasets/create-and-index` Response**:
```python
return DatasetIndexResponse(
    ...
    graph_ready=False,  # True only after graph build completes
    estimated_time_minutes=5  # Rough estimate
)
```

**Impact**:
- New datasets immediately available for queries (no restart)
- Frontend can show loading state while graph builds
- Cache automatically warms new subgraph

---

## Implementation Timeline

### Phase 1: Cache Enhancement (2-3 hours)
- [ ] Update default max_cached from 5 to 10
- [ ] Implement `_auto_prewarm()` method
- [ ] Add auto-prewarm call on server startup
- [ ] Test with 10+ subgraphs

### Phase 2: Topics Removal (1-2 hours)
- [ ] Remove topics from routing prompt
- [ ] Enhance descriptions in existing registries
- [ ] Update KUET and football descriptions
- [ ] Test routing accuracy

### Phase 3: Registry API (4-5 hours)
- [ ] Implement GET /api/unified/registry
- [ ] Implement GET /api/unified/registry/{name}
- [ ] Implement PUT /api/unified/registry/{name}
- [ ] Implement DELETE /api/unified/registry/{name}
- [ ] Add OpenAPI examples
- [ ] Test hot-reload behavior

### Phase 4: Hot-Reload Enhancement (2-3 hours)
- [ ] Add cache prewarm for new subgraphs
- [ ] Add graph_ready field to ProcessingJob
- [ ] Update `/datasets/create-and-index` response
- [ ] Test end-to-end dynamic creation flow

**Total Estimated Time**: 9-13 hours

---

## Testing Plan

### Test Case 1: LRU Cache with 10 Subgraphs
```bash
# Create 15 subgraphs
for i in {1..15}; do
  curl -X POST "http://localhost:8001/datasets/create" \
    -F "dataset_name=test_dataset_$i"
done

# Query each sequentially
for i in {1..15}; do
  curl -X POST "http://localhost:8001/api/unified/chat" \
    -d '{"messages": [{"role": "user", "content": "test"}], "force_subgraphs": ["test_dataset_'$i'"]}'
done

# Check cache stats
curl http://localhost:8001/api/unified/cache/stats

# Expected: 10 cached, 5 evicted (LRU policy)
```

### Test Case 2: Topics Removal - Routing Accuracy
```bash
# Scenario: Two universities with similar topics
# Create KUET and BUET subgraphs
# Query: "How many CSE seats at KUET?"

curl -X POST "http://localhost:8001/api/unified/chat" \
  -d '{
    "messages": [{"role": "user", "content": "How many CSE seats at KUET?"}]
  }'

# Expected: Routes to kuet_test (NOT buet_test)
# Verify response includes "subgraph_used": "kuet_test"
```

### Test Case 3: Dynamic Registry Update
```bash
# Update KUET description
curl -X PUT "http://localhost:8001/api/unified/registry/kuet_test" \
  -d '{
    "description": "KUET (Khulna University) - Updated description"
  }'

# Verify immediately takes effect (no restart)
curl -X GET "http://localhost:8001/api/unified/registry/kuet_test"

# Query should use new description for routing
curl -X POST "http://localhost:8001/api/unified/chat" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me about Khulna University"}]
  }'
```

### Test Case 4: Dynamic Subgraph Creation + Hot-Reload
```bash
# Create new dataset
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@new_doc.md" \
  -F "data_source=new_university"

# Wait for indexing (check status)
curl http://localhost:8001/status/<job_id>

# Query immediately (no server restart)
curl -X POST "http://localhost:8001/api/unified/chat" \
  -d '{
    "messages": [{"role": "user", "content": "Info about new university?"}]
  }'

# Expected: Routes to new_university subgraph successfully
```

---

## Breaking Changes

### None Expected

All changes are backwards-compatible:
- Existing `/api/unified/query` and `/api/unified/chat` endpoints unchanged
- Registry structure unchanged (only topics usage removed from routing)
- Old subgraphs continue to work
- Default max_cached increased (no breaking change)

---

## Migration Guide

### For Users with Existing Registries

**Step 1**: Enhance descriptions to be more specific (since topics removed):
```json
{
  "kuet_test": {
    "description": "KUET (Khulna University of Engineering & Technology) in Bangladesh - covers admissions, departments (CSE, EEE, ME, CE, IPE), curriculum, faculty, and campus information"
  }
}
```

**Step 2**: No code changes required - restart server to pick up new defaults

**Step 3**: Optionally use new registry API for future updates

---

## Decisions (January 26, 2025)

1. **LRU Cache Size**: Make configurable with default=10 ✅

2. **Topics Removal**: Option A (remove topics entirely) ✅

3. **Auto-Prewarm Priority**: Manual-first, then most-recent ✅

4. **Registry API Permissions**: No authentication (internal API) ✅

5. **Graph Build Time**: Return immediately with job_id (async pattern) ✅

---

## Conclusion

This plan makes the unified routing system production-ready with:
- Larger LRU cache (10 subgraphs)
- Automatic pre-loading on startup
- Improved routing accuracy (remove ambiguous topics)
- Full REST API for registry management
- Hot-reload for dynamic subgraph creation

No server restarts required for any registry operations.

**Please review and provide feedback on the open questions above.**
