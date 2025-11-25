# Backend API Update Plan for Enhanced Pipeline Integration

**Version:** 1.0
**Date:** 2025-01-25
**Goal:** Update backend API to support Enhanced Pipeline and unified subgraph architecture

---

## Overview

This plan outlines the backend API changes needed to integrate:
1. **Enhanced Pipeline** (from Production_pipeline_redesign_plan.md) - hybrid chunking + gleaning + validation
2. **Unified Subgraph Architecture** (from SUBGRAPH_MANAGEMENT_GUIDE.md) - multi-subgraph with LLM routing

**Key Principles:**
- ✅ **Unified endpoints remain unchanged** (`/api/unified/*`)
- ✅ **Only indexing/building endpoints need updates**
- ✅ **Backward compatibility** with existing graphs
- ✅ **Minimal breaking changes** for frontend

---

## Phase 1: Core Pipeline Integration (Week 1)

### 1.1 Update Dataset Creation Endpoint

**Current:** `/datasets/create-and-index`
**Status:** ✅ Already uses production pipeline
**Changes Needed:** Add Enhanced Pipeline configuration support

**File:** `backend/api/dataset_routes.py`

```python
# NEW: Add enhanced pipeline config parameter
@router.post("/datasets/create-and-index")
async def create_and_index_dataset(
    dataset_name: str,
    documents: List[Dict],
    process_async: bool = False,

    # NEW: Enhanced pipeline config
    use_enhanced_pipeline: bool = True,  # Default to enhanced
    enhanced_config: Optional[Dict] = None  # Override defaults
):
    """
    Create new dataset and index documents.

    Args:
        enhanced_config: {
            "extraction_strategy": "hybrid",  # strict | gleaning | hybrid
            "validation_level": "MODERATE",
            "enable_entity_linking": True,
            "chunk_size": 1000,
            "overlap": 100
        }
    """

    # Default config
    if enhanced_config is None:
        enhanced_config = {
            "extraction_strategy": "hybrid",
            "validation_level": "MODERATE",
            "enable_entity_linking": True,
            "chunk_size": 1000,
            "overlap": 100
        }

    # Pass config to BiGRAG
    rag = BiGRAG(
        working_dir=f"expr/{dataset_name}",
        use_enhanced_pipeline=use_enhanced_pipeline,
        enhanced_pipeline_config=enhanced_config
    )

    # ... rest of implementation
```

**Testing:**
```bash
# Test with enhanced pipeline (default)
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "test_enhanced",
    "documents": [{"content": "Test doc", "title": "Test"}]
  }'

# Test with custom config
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "test_custom",
    "documents": [{"content": "Test", "title": "Test"}],
    "enhanced_config": {
      "extraction_strategy": "gleaning",
      "chunk_size": 1200
    }
  }'
```

---

### 1.2 Update Document Upload Endpoint

**Current:** `/documents/upload`
**Status:** ⚠️ Single-subgraph only, not unified
**Changes Needed:**
1. Add dataset_name parameter for unified mode
2. Add enhanced pipeline config
3. Auto-update subgraph registry

**File:** `backend/api/document_routes.py`

```python
@router.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    dataset_name: str,  # NEW: Required for unified mode

    # Enhanced pipeline config
    use_enhanced_pipeline: bool = True,
    extraction_strategy: str = "hybrid",
    validation_level: str = "MODERATE",
    enable_entity_linking: bool = True
):
    """
    Upload document to specific subgraph.

    NEW: Works with unified subgraph architecture.
    """

    # Load subgraph
    subgraph_path = f"expr/{dataset_name}"
    if not os.path.exists(subgraph_path):
        raise HTTPException(404, f"Subgraph '{dataset_name}' not found")

    # Initialize BiGRAG with enhanced config
    rag = BiGRAG(
        working_dir=subgraph_path,
        use_enhanced_pipeline=use_enhanced_pipeline,
        enhanced_pipeline_config={
            "extraction_strategy": extraction_strategy,
            "validation_level": validation_level,
            "enable_entity_linking": enable_entity_linking
        }
    )

    # Process document
    content = await file.read()
    doc_text = content.decode('utf-8')

    await rag.ainsert(
        [doc_text],
        metadata=[{"title": file.filename, "source": "upload"}]
    )

    # NEW: Reload unified executor to pick up changes
    if hasattr(app.state, 'unified_executor'):
        await app.state.unified_executor.reload_subgraph(dataset_name)

    return {
        "status": "success",
        "dataset_name": dataset_name,
        "filename": file.filename
    }
```

**Migration Note:**
Old behavior (single subgraph mode) is **deprecated**. Frontend must now specify `dataset_name`.

**Frontend Migration:**
```typescript
// OLD (deprecated)
formData.append("file", file);
await fetch("/documents/upload", { method: "POST", body: formData });

// NEW (required)
formData.append("file", file);
formData.append("dataset_name", currentDataset); // Must specify
await fetch("/documents/upload", { method: "POST", body: formData });
```

---

### 1.3 Add Pipeline Status Endpoint (NEW)

**Purpose:** Check which pipeline a subgraph was built with

**File:** `backend/api/dataset_routes.py`

```python
@router.get("/datasets/{dataset_name}/pipeline-info")
async def get_pipeline_info(dataset_name: str):
    """
    Get pipeline configuration used to build this subgraph.

    Returns:
        {
            "pipeline_type": "enhanced" | "standard" | "unknown",
            "config": {
                "extraction_strategy": "hybrid",
                "validation_level": "MODERATE",
                ...
            },
            "compatible_with": ["enhanced-v1.0", "standard-v1.0"]
        }
    """

    subgraph_path = f"expr/{dataset_name}"
    graph_file = f"{subgraph_path}/graph_chunk_entity_relation.graphml"

    if not os.path.exists(graph_file):
        raise HTTPException(404, "Subgraph not found")

    # Read pipeline metadata from GraphML
    graph = nx.read_graphml(graph_file)
    pipeline_version = graph.graph.get('pipeline_version', 'unknown')

    # Determine pipeline type
    if 'enhanced' in pipeline_version.lower():
        pipeline_type = "enhanced"
    elif 'production' in pipeline_version.lower():
        pipeline_type = "production"  # Old name
    elif 'standard' in pipeline_version.lower():
        pipeline_type = "standard"
    else:
        pipeline_type = "unknown"

    return {
        "dataset_name": dataset_name,
        "pipeline_type": pipeline_type,
        "pipeline_version": pipeline_version,
        "compatible_with": graph.graph.get('backward_compatible', []),
        "created_at": graph.graph.get('created_at', 'unknown'),
        "config": graph.graph.get('pipeline_config', {})
    }
```

**Use Case:**
```bash
# Check pipeline used for KUET subgraph
curl "http://localhost:8001/datasets/KUET/pipeline-info"

# Response:
{
  "dataset_name": "KUET",
  "pipeline_type": "enhanced",
  "pipeline_version": "enhanced-v1.0",
  "compatible_with": ["standard-v1.0", "production-v1.0"],
  "config": {
    "extraction_strategy": "hybrid",
    "validation_level": "MODERATE"
  }
}
```

---

## Phase 2: Unified Subgraph Endpoints (Week 2)

### 2.1 Deprecate Single-Subgraph Mode Endpoints

**Current Endpoints (DEPRECATED):**
- `POST /search` - Single subgraph search
- `POST /chat/completions` - Single subgraph chat

**Status:** ⚠️ Mark as deprecated, keep for backward compatibility (6 months)

**File:** `backend/api/search_routes.py`

```python
@router.post("/search")
async def search_single_subgraph(request: SearchRequest):
    """
    DEPRECATED: Use /api/unified/query instead.

    This endpoint will be removed in v2.0 (July 2025).
    """
    warnings.warn(
        "POST /search is deprecated. Use POST /api/unified/query instead.",
        DeprecationWarning
    )

    # Fallback to unified endpoint
    return await unified_query(
        query=request.queries[0],
        dataset_name=app.state.current_dataset  # From startup
    )
```

**Migration Guide for Frontend:**
```typescript
// OLD (deprecated)
const response = await fetch("/search", {
  method: "POST",
  body: JSON.stringify({ queries: ["query"] })
});

// NEW (use unified endpoint)
const response = await fetch("/api/unified/query", {
  method: "POST",
  body: JSON.stringify({
    query: "query",
    dataset_name: "KUET"  // Optional (router decides if omitted)
  })
});
```

---

### 2.2 Update Unified Query Endpoint (NO CHANGES)

**Current:** `/api/unified/query`
**Status:** ✅ **No changes needed**
**Reason:** Retrieval logic unchanged, works with both pipelines

**Verification:**
```bash
# Should work with enhanced pipeline graphs
curl -X POST "http://localhost:8001/api/unified/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many seats in KUET CSE?",
    "dataset_name": "KUET"
  }'

# Should work with standard pipeline graphs
curl -X POST "http://localhost:8001/api/unified/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who won 2022 World Cup?",
    "dataset_name": "football"
  }'
```

---

### 2.3 Update Unified Chat Endpoint (NO CHANGES)

**Current:** `/api/unified/ask`
**Status:** ✅ **No changes needed**
**Reason:** Uses `/api/unified/query` internally

**Verification:**
```bash
curl -X POST "http://localhost:8001/api/unified/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are KUET CSE admission requirements?",
    "dataset_name": "KUET"
  }'
```

---

## Phase 3: Rebuild & Migration Endpoints (Week 2)

### 3.1 Add Graph Rebuild Endpoint with Pipeline Selection

**Purpose:** Rebuild existing subgraph with enhanced pipeline

**File:** `backend/api/dataset_routes.py`

```python
@router.post("/datasets/{dataset_name}/rebuild")
async def rebuild_subgraph(
    dataset_name: str,

    # Pipeline selection
    use_enhanced_pipeline: bool = True,
    enhanced_config: Optional[Dict] = None,

    # Options
    force: bool = False,  # Overwrite existing
    backup: bool = True   # Backup old graph before rebuild
):
    """
    Rebuild subgraph with enhanced pipeline.

    Use Cases:
    - Migrate old standard pipeline graph to enhanced
    - Apply new extraction strategy to existing corpus
    - Fix orphan nodes with validation

    Process:
    1. Backup existing graph (if backup=True)
    2. Load corpus from kv_store_full_docs.json
    3. Rebuild with enhanced pipeline
    4. Update subgraph registry
    5. Reload unified executor
    """

    subgraph_path = f"expr/{dataset_name}"

    if not os.path.exists(subgraph_path):
        raise HTTPException(404, f"Subgraph '{dataset_name}' not found")

    # Backup old graph
    if backup:
        backup_dir = f"{subgraph_path}_backup_{int(time.time())}"
        shutil.copytree(subgraph_path, backup_dir)
        logger.info(f"Backed up to {backup_dir}")

    # Load corpus
    corpus_file = f"{subgraph_path}/kv_store_full_docs.json"
    if not os.path.exists(corpus_file):
        raise HTTPException(400, "Corpus not found. Cannot rebuild.")

    with open(corpus_file) as f:
        corpus = json.load(f)

    documents = [doc['content'] for doc in corpus.values()]
    metadata = [doc.get('metadata', {}) for doc in corpus.values()]

    # Rebuild with enhanced pipeline
    rag = BiGRAG(
        working_dir=subgraph_path,
        use_enhanced_pipeline=use_enhanced_pipeline,
        enhanced_pipeline_config=enhanced_config or {
            "extraction_strategy": "hybrid",
            "validation_level": "MODERATE",
            "enable_entity_linking": True
        }
    )

    # Clear existing graph (if force=True)
    if force:
        await rag.clear_graph()

    # Re-insert all documents
    await rag.ainsert(documents, metadata=metadata)

    # Reload unified executor
    if hasattr(app.state, 'unified_executor'):
        await app.state.unified_executor.reload_subgraph(dataset_name)

    return {
        "status": "success",
        "dataset_name": dataset_name,
        "pipeline": "enhanced",
        "documents_processed": len(documents),
        "backup_created": backup
    }
```

**Use Case:**
```bash
# Rebuild KUET with enhanced pipeline
curl -X POST "http://localhost:8001/datasets/KUET/rebuild" \
  -H "Content-Type: application/json" \
  -d '{
    "use_enhanced_pipeline": true,
    "enhanced_config": {
      "extraction_strategy": "gleaning"
    }
  }'
```

---

### 3.2 Add Batch Migration Endpoint

**Purpose:** Migrate multiple subgraphs to enhanced pipeline

**File:** `backend/api/admin_routes.py` (NEW)

```python
@router.post("/admin/migrate-to-enhanced")
async def migrate_all_to_enhanced(
    subgraph_names: List[str],  # Which subgraphs to migrate
    backup: bool = True,
    parallel: bool = False  # Migrate in parallel (faster)
):
    """
    Migrate multiple subgraphs to enhanced pipeline.

    Example:
        POST /admin/migrate-to-enhanced
        {
            "subgraph_names": ["KUET", "BUET", "DU"],
            "backup": true,
            "parallel": false
        }
    """

    results = []

    if parallel:
        # Parallel migration (use asyncio.gather)
        tasks = [
            rebuild_subgraph(name, use_enhanced_pipeline=True, backup=backup)
            for name in subgraph_names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Sequential migration
        for name in subgraph_names:
            try:
                result = await rebuild_subgraph(
                    name,
                    use_enhanced_pipeline=True,
                    backup=backup
                )
                results.append({"subgraph": name, "status": "success"})
            except Exception as e:
                results.append({"subgraph": name, "status": "error", "error": str(e)})

    return {
        "total": len(subgraph_names),
        "results": results
    }
```

---

## Phase 4: Enhanced Pipeline Monitoring (Week 3)

### 4.1 Add Extraction Metrics Endpoint

**Purpose:** Track extraction quality per subgraph

**File:** `backend/api/monitoring_routes.py` (NEW)

```python
@router.get("/datasets/{dataset_name}/extraction-metrics")
async def get_extraction_metrics(dataset_name: str):
    """
    Get extraction quality metrics for subgraph.

    Returns:
        {
            "total_chunks": 120,
            "successful_extractions": 118,
            "failed_extractions": 2,
            "gleaning_improvements": 45,  # Entities found via gleaning
            "avg_validation_score": 0.92,
            "orphan_entities": 3,
            "orphan_relations": 5
        }
    """

    subgraph_path = f"expr/{dataset_name}"
    graph_file = f"{subgraph_path}/graph_chunk_entity_relation.graphml"

    if not os.path.exists(graph_file):
        raise HTTPException(404, "Subgraph not found")

    # Read graph
    graph = nx.read_graphml(graph_file)

    # Count metrics
    entities = [n for n, d in graph.nodes(data=True) if d.get('role') == 'entity']
    relations = [n for n, d in graph.nodes(data=True) if d.get('role') == 'relation']

    # Check for orphans
    orphan_entities = [e for e in entities if graph.degree(e) == 0]
    orphan_relations = [r for r in relations if graph.degree(r) == 0]

    # Read extraction metadata
    metadata = graph.graph.get('extraction_metadata', {})

    return {
        "dataset_name": dataset_name,
        "total_entities": len(entities),
        "total_relations": len(relations),
        "orphan_entities": len(orphan_entities),
        "orphan_relations": len(orphan_relations),
        "extraction_metadata": metadata
    }
```

---

### 4.2 Add Failed Extraction Endpoint (HITL System)

**Purpose:** View chunks that failed extraction validation

**File:** `backend/api/hitl_routes.py` (NEW)

```python
@router.get("/hitl/{dataset_name}/failed-extractions")
async def get_failed_extractions(
    dataset_name: str,
    limit: int = 50,
    skip: int = 0
):
    """
    Get failed extractions for human review.

    Returns list of chunks that failed validation.
    """

    failed_store_path = f"expr/{dataset_name}/failed_extractions/failed_chunks.json"

    if not os.path.exists(failed_store_path):
        return {"total": 0, "failures": []}

    with open(failed_store_path) as f:
        failures = json.load(f)

    # Pagination
    total = len(failures)
    paginated = failures[skip:skip+limit]

    return {
        "dataset_name": dataset_name,
        "total": total,
        "showing": len(paginated),
        "failures": paginated
    }

@router.post("/hitl/{dataset_name}/correct-extraction")
async def submit_extraction_correction(
    dataset_name: str,
    extraction_id: str,
    corrected_entities: List[Dict],
    corrected_relations: List[Dict]
):
    """
    Submit human-corrected extraction.

    This will:
    1. Update failed extraction record
    2. Insert corrected entities/relations to graph
    3. Mark as reviewed
    """

    # Implementation: Update graph with corrected data
    # ... (see Step 6 in Production_pipeline_redesign_plan.md)

    return {"status": "correction_applied", "extraction_id": extraction_id}
```

---

## Phase 5: API Documentation Update (Week 3)

### 5.1 Update OpenAPI Docs

**File:** `backend/server.py`

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="BiG-RAG API",
    version="2.0.0",  # Updated version
    description="""
    BiG-RAG API with Enhanced Pipeline and Unified Subgraph Architecture.

    ## Key Features
    - **Enhanced Pipeline**: Hybrid chunking + gleaning extraction + validation
    - **Unified Subgraphs**: Multi-subgraph with LLM routing
    - **Backward Compatible**: Works with standard pipeline graphs

    ## Migration Notes
    - `/search` (deprecated) → use `/api/unified/query`
    - `/chat/completions` (deprecated) → use `/api/unified/ask`
    - All indexing endpoints now require `dataset_name` parameter

    ## Pipeline Types
    - **Enhanced**: Hybrid strategy (recommended for new graphs)
    - **Standard**: Fast extraction (backward compatibility)
    """
)
```

---

### 5.2 Add API Migration Guide

**File:** `backend/API_MIGRATION_GUIDE.md` (NEW)

```markdown
# API Migration Guide (v1.0 → v2.0)

## Breaking Changes

### 1. Document Upload Requires dataset_name

**Before (v1.0):**
```bash
POST /documents/upload
FormData: file=document.pdf
```

**After (v2.0):**
```bash
POST /documents/upload
FormData:
  file=document.pdf
  dataset_name=KUET  # REQUIRED
```

### 2. Search Endpoint Deprecated

**Before (v1.0):**
```bash
POST /search
{ "queries": ["query"] }
```

**After (v2.0):**
```bash
POST /api/unified/query
{ "query": "query", "dataset_name": "KUET" }
```

### 3. Chat Endpoint Deprecated

**Before (v1.0):**
```bash
POST /chat/completions
{ "messages": [...] }
```

**After (v2.0):**
```bash
POST /api/unified/ask
{ "question": "...", "dataset_name": "KUET" }
```

## New Endpoints

- `POST /datasets/{name}/rebuild` - Rebuild with enhanced pipeline
- `GET /datasets/{name}/pipeline-info` - Check pipeline type
- `GET /datasets/{name}/extraction-metrics` - View quality metrics
- `GET /hitl/{name}/failed-extractions` - Human-in-the-loop review
```

---

## Summary of Changes

### Endpoints Added ✅

| Endpoint | Purpose | Phase |
|----------|---------|-------|
| `GET /datasets/{name}/pipeline-info` | Check pipeline type | 1.3 |
| `POST /datasets/{name}/rebuild` | Rebuild with enhanced | 3.1 |
| `POST /admin/migrate-to-enhanced` | Batch migration | 3.2 |
| `GET /datasets/{name}/extraction-metrics` | Quality metrics | 4.1 |
| `GET /hitl/{name}/failed-extractions` | HITL review | 4.2 |
| `POST /hitl/{name}/correct-extraction` | Submit corrections | 4.2 |

### Endpoints Modified 🔧

| Endpoint | Change | Phase |
|----------|--------|-------|
| `POST /datasets/create-and-index` | Add enhanced config | 1.1 |
| `POST /documents/upload` | Add dataset_name (required) | 1.2 |

### Endpoints Deprecated ⚠️

| Endpoint | Replacement | Removal Date |
|----------|-------------|--------------|
| `POST /search` | `POST /api/unified/query` | July 2025 |
| `POST /chat/completions` | `POST /api/unified/ask` | July 2025 |

### Endpoints Unchanged ✅

| Endpoint | Status |
|----------|--------|
| `POST /api/unified/query` | No changes |
| `POST /api/unified/ask` | No changes |
| `GET /api/unified/subgraphs` | No changes |
| `POST /api/unified/reload` | No changes |

---

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| **Week 1** | Core Pipeline Integration | Updated create-and-index, document upload, pipeline-info endpoint |
| **Week 2** | Unified Endpoints | Deprecation warnings, rebuild endpoint, migration endpoint |
| **Week 3** | Monitoring & Docs | Metrics endpoints, HITL endpoints, API docs |

---

## Testing Checklist

### Phase 1 Tests
- [ ] Create dataset with enhanced pipeline
- [ ] Upload document to specific subgraph
- [ ] Check pipeline info for subgraph
- [ ] Verify backward compatibility (standard graphs work)

### Phase 2 Tests
- [ ] Deprecation warnings appear for old endpoints
- [ ] Unified query works with enhanced graphs
- [ ] Unified chat works with enhanced graphs

### Phase 3 Tests
- [ ] Rebuild single subgraph
- [ ] Batch migrate multiple subgraphs
- [ ] Verify backup creation

### Phase 4 Tests
- [ ] View extraction metrics
- [ ] View failed extractions
- [ ] Submit correction for failed extraction

---

## Frontend Migration Tasks

### Required Changes
1. **Document Upload Component**
   - Add `dataset_name` to form data
   - Update API call to new endpoint

2. **Search Component**
   - Replace `/search` with `/api/unified/query`
   - Add optional `dataset_name` parameter

3. **Chat Component**
   - Replace `/chat/completions` with `/api/unified/ask`
   - Add optional `dataset_name` parameter

4. **Dataset Management**
   - Add "Rebuild with Enhanced" button
   - Show pipeline type badge (enhanced/standard)
   - Display extraction metrics

### Optional Enhancements
1. **HITL Review UI**
   - View failed extractions
   - Submit corrections
   - Track review progress

2. **Pipeline Selector**
   - Choose extraction strategy when creating dataset
   - Show recommended strategy based on content

---

## Backward Compatibility

### What Still Works ✅
- Existing graphs built with standard pipeline
- Unified query/chat endpoints
- Document deletion
- Graph visualization

### What Requires Migration ⚠️
- Frontend document upload calls (must add `dataset_name`)
- Frontend search calls (use unified endpoint)
- Scripts using old `/search` endpoint

### Migration Support
- Deprecation warnings (6-month grace period)
- Fallback to unified endpoints
- Backward-compatible graph format

---

## Current Backend Endpoint Audit

### Analyzed Route Files
- ✅ `datasets.py` - Dataset creation and indexing
- ✅ `documents.py` - Document upload, delete, rebuild, list
- ✅ `retrieval.py` - Search and ask endpoints
- ✅ `llm.py` - Chat completions with RAG
- ✅ `unified.py` - Unified multi-subgraph endpoints
- ✅ `graph.py` - Graph visualization and stats
- ✅ `evaluation.py` - Evaluation endpoints
- ✅ `jobs.py` - Background job status tracking
- ✅ `health.py` - Health checks

---

## Complete Endpoint Inventory & Update Plan

### Category A: Unified Endpoints (NO CHANGES - Already Production-Ready)

| Endpoint | Status | Notes |
|----------|--------|-------|
| `POST /api/unified/query` | ✅ Keep as-is | Works with enhanced pipeline graphs |
| `POST /api/unified/ask` | ✅ Keep as-is | Simple search interface |
| `POST /api/unified/route` | ✅ Keep as-is | Routing decision only |
| `GET /api/unified/subgraphs` | ✅ Keep as-is | List available subgraphs |
| `GET /api/unified/subgraphs/{name}` | ✅ Keep as-is | Get subgraph metadata |
| `GET /api/unified/cache/stats` | ✅ Keep as-is | Cache statistics |
| `POST /api/unified/cache/clear` | ✅ Keep as-is | Clear cache |
| `POST /api/unified/registry/reload` | ✅ Keep as-is | Reload registry |

**Reason:** These endpoints are retrieval-only and work with any graph structure (standard or enhanced pipeline).

---

### Category B: Dataset Management Endpoints (NEEDS ENHANCEMENT)

#### B1. `/datasets/create-and-index` (MODIFY)

**Current State:**
- ✅ Already uses production pipeline by default
- ✅ Auto-registers to subgraph registry
- ✅ Works in unified mode
- ❌ Missing enhanced pipeline config parameter

**Required Changes:**
```python
@router.post("/datasets/create-and-index")
async def create_and_index_document(
    # ... existing params ...

    # NEW: Enhanced pipeline config
    extraction_strategy: str = Form("hybrid", description="strict | gleaning | hybrid"),
    validation_level: str = Form("MODERATE", description="STRICT | MODERATE | LENIENT"),
    enable_entity_linking: bool = Form(True),
    chunk_size: int = Form(1000),
    overlap: int = Form(100)
):
```

**Implementation Priority:** Phase 1 (Week 1)

---

#### B2. `/datasets/create` (KEEP AS-IS)

**Current State:**
- ✅ Creates dataset structure only (no indexing)
- ✅ Updates subgraph registry
- ✅ Works in unified mode

**Required Changes:** None (doesn't involve indexing)

---

### Category C: Document Management Endpoints (NEEDS UPDATE)

#### C1. `/documents/upload` (MODIFY - Critical)

**Current State:**
- ✅ Works in both single and unified mode
- ✅ Supports production pipeline via `use_production_pipeline=true`
- ❌ `data_source` parameter optional (breaks unified mode assumption)
- ❌ Missing enhanced pipeline config options

**Required Changes:**

**Change 1: Make `data_source` required in unified mode**
```python
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(None),
    data_source: str = Form(..., description="REQUIRED: Dataset name"),  # Make required

    # Replace single boolean with config
    extraction_strategy: str = Form("hybrid"),
    validation_level: str = Form("MODERATE"),
    enable_entity_linking: bool = Form(True)
):
```

**Change 2: Remove single-mode fallback**
```python
# OLD (remove this)
if not data_source or data_source == "string":
    # Use current dataset from server startup

# NEW (always require)
if not data_source:
    raise HTTPException(400, "data_source parameter is required")
```

**Migration Impact:** ⚠️ **BREAKING CHANGE** for frontend
- Old frontend code will fail if `data_source` not provided
- Add deprecation warning first (6 months)

**Implementation Priority:** Phase 1 (Week 1)

---

#### C2. `/documents/rebuild` (MODIFY)

**Current State:**
- Uses injected RAG instance (single mode only)
- Rebuilds entire graph from corpus.jsonl

**Required Changes:**

**Add enhanced pipeline support:**
```python
@router.post("/documents/rebuild")
async def rebuild_graph(
    dataset_name: str = Form(..., description="Dataset to rebuild"),  # Make required
    use_enhanced_pipeline: bool = Form(True),
    enhanced_config: Optional[Dict] = Form(None),
    force: bool = Form(True),
    backup: bool = Form(True)
):
    # Load corpus
    # Create new BiGRAG instance with enhanced config
    # Rebuild
    # Update registry
```

**Implementation Priority:** Phase 3 (Week 2)

---

#### C3. `/documents` (LIST) - NO CHANGES

**Status:** ✅ Keep as-is (metadata only, no indexing)

---

#### C4. `/documents/{document_id}` (GET DETAILS) - MINOR UPDATE

**Current State:**
- Reads from registry + KG stats

**Required Changes:**
- Add `pipeline_info` field showing which pipeline was used

```python
return DocumentDetailResponse(
    # ... existing fields ...
    pipeline_info={
        "type": "enhanced",  # Read from GraphML
        "extraction_strategy": "hybrid",
        "created_at": "..."
    }
)
```

**Implementation Priority:** Phase 3 (Week 2)

---

#### C5. `/documents/{document_id}` (DELETE) - NO CHANGES

**Status:** ✅ Keep as-is (deletion logic unchanged)

---

### Category D: Retrieval Endpoints (NEEDS DEPRECATION)

#### D1. `POST /search` (DEPRECATE)

**Current State:**
- Single-subgraph only
- Not in `retrieval.py` (need to find)

**Action:** Mark as deprecated, redirect to `/api/unified/query`

```python
@router.post("/search")
async def search_single_subgraph(request: SearchRequest):
    """
    ⚠️ DEPRECATED: Use POST /api/unified/query instead.
    This endpoint will be removed in v2.0 (July 2025).
    """
    import warnings
    warnings.warn("POST /search is deprecated", DeprecationWarning)

    # Get unified executor
    executor = get_unified_executor()
    if not executor:
        raise HTTPException(503, "Single-mode deprecated. Use --unified mode.")

    # Redirect to unified endpoint
    return await unified_query(
        query=request.queries[0],
        force_subgraphs=[get_data_source()]  # Use current dataset
    )
```

**Implementation Priority:** Phase 2 (Week 2)

---

#### D2. `POST /ask` (EVALUATE - May Keep)

**Current State:**
- Returns context items (no LLM synthesis)
- Used by frontend for pure retrieval

**Action:** Keep but add unified mode support

```python
@router.post("/ask")
async def ask_question(request: AskRequest):
    # Check if unified mode
    executor = get_unified_executor()
    if executor:
        # Use unified endpoint
        return await unified_ask(request)
    else:
        # Use single-mode RAG (backward compat)
        rag = get_rag_instance()
        # ... existing code
```

**Implementation Priority:** Phase 2 (Week 2)

---

### Category E: LLM Endpoints (NEEDS UPDATE)

#### E1. `POST /chat/completions` (EVALUATE)

**Current State:**
- OpenAI-compatible chat endpoint
- Uses RAG retrieval + LLM synthesis
- Single-subgraph only

**Action:** Add unified mode support

```python
@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    executor = get_unified_executor()

    if executor:
        # UNIFIED MODE: Use unified retrieval
        result = await executor.query(
            query=user_prompt,
            query_param=QueryParam(...)
        )
        contexts = result['results']
    else:
        # SINGLE MODE: Use injected RAG
        rag = get_rag_instance()
        contexts = await rag.aquery(...)

    # ... rest of LLM synthesis logic
```

**Alternative:** Deprecate this endpoint, recommend `/api/unified/ask` instead

**Implementation Priority:** Phase 2 (Week 2)

---

### Category F: Graph & Stats Endpoints (NO CHANGES)

| Endpoint | Status | Notes |
|----------|--------|-------|
| `GET /graph/stats` | ✅ Keep | Stats work with any graph |
| `GET /graph/visualization` | ✅ Keep | Visualization unchanged |
| `POST /graph/export` | ✅ Keep | Export unchanged |

---

### Category G: Evaluation Endpoints (NO CHANGES)

| Endpoint | Status | Notes |
|----------|--------|-------|
| `POST /evaluation/run` | ✅ Keep | Uses unified retrieval |
| `GET /evaluation/results` | ✅ Keep | Results unchanged |

---

### Category H: Job Tracking (NO CHANGES)

| Endpoint | Status | Notes |
|----------|--------|-------|
| `GET /jobs/{job_id}/status` | ✅ Keep | Job tracking unchanged |
| `GET /jobs/list` | ✅ Keep | List all jobs |

---

### Category I: Health & System (NO CHANGES)

| Endpoint | Status | Notes |
|----------|--------|-------|
| `GET /health` | ✅ Keep | Health check unchanged |

---

## NEW Endpoints to Add

### I1. Pipeline Information Endpoint

**Purpose:** Check which pipeline a dataset was built with

```python
@router.get("/datasets/{dataset_name}/pipeline-info")
async def get_pipeline_info(dataset_name: str):
    """Get pipeline configuration for dataset"""
    # Read from GraphML metadata
    # Return pipeline type, config, version
```

**Implementation Priority:** Phase 1 (Week 1)

---

### I2. Rebuild Endpoint (Dataset-Scoped)

**Purpose:** Rebuild specific dataset with enhanced pipeline

```python
@router.post("/datasets/{dataset_name}/rebuild")
async def rebuild_dataset(
    dataset_name: str,
    use_enhanced_pipeline: bool = True,
    enhanced_config: Optional[Dict] = None,
    backup: bool = True
):
    """Rebuild dataset with enhanced pipeline"""
```

**Implementation Priority:** Phase 3 (Week 2)

---

### I3. Batch Migration Endpoint

**Purpose:** Migrate multiple datasets to enhanced pipeline

```python
@router.post("/admin/migrate-to-enhanced")
async def batch_migrate(
    dataset_names: List[str],
    backup: bool = True,
    parallel: bool = False
):
    """Migrate multiple datasets to enhanced pipeline"""
```

**Implementation Priority:** Phase 3 (Week 2)

---

### I4. Extraction Metrics Endpoint

**Purpose:** View extraction quality metrics

```python
@router.get("/datasets/{dataset_name}/extraction-metrics")
async def get_extraction_metrics(dataset_name: str):
    """Get extraction quality metrics for dataset"""
```

**Implementation Priority:** Phase 4 (Week 3)

---

### I5. Failed Extraction Endpoints (HITL)

**Purpose:** Human-in-the-loop review of failed extractions

```python
@router.get("/hitl/{dataset_name}/failed-extractions")
async def list_failed_extractions(...):
    """List chunks that failed validation"""

@router.post("/hitl/{dataset_name}/correct-extraction")
async def submit_correction(...):
    """Submit human-corrected extraction"""
```

**Implementation Priority:** Phase 4 (Week 3) - Optional

---

## Summary of Required Changes

### Critical Changes (Phase 1 - Week 1)

1. ✅ Add enhanced config to `/datasets/create-and-index`
2. ⚠️ Make `data_source` required in `/documents/upload`
3. ✅ Add `/datasets/{name}/pipeline-info` endpoint

### Important Changes (Phase 2 - Week 2)

4. ⚠️ Deprecate `/search` endpoint
5. ✅ Add unified mode support to `/ask`
6. ✅ Add unified mode support to `/chat/completions`
7. ✅ Update `/documents/rebuild` with enhanced config

### Nice-to-Have (Phase 3 - Week 2)

8. ✅ Add `/datasets/{name}/rebuild` endpoint
9. ✅ Add `/admin/migrate-to-enhanced` endpoint
10. ✅ Add pipeline info to document details

### Optional (Phase 4 - Week 3)

11. ✅ Add `/datasets/{name}/extraction-metrics`
12. ✅ Add HITL endpoints (`/hitl/*`)

---

## Breaking Changes Summary

### For Frontend

**1. `/documents/upload` now requires `data_source`**

Before:
```typescript
formData.append("file", file);
// No data_source needed
```

After:
```typescript
formData.append("file", file);
formData.append("data_source", currentDataset); // REQUIRED
```

**Migration Period:** 6 months (deprecation warnings)

---

**2. `/search` deprecated (use `/api/unified/query`)**

Before:
```typescript
POST /search { "queries": ["query"] }
```

After:
```typescript
POST /api/unified/query { "query": "query", "dataset_name": "..." }
```

**Migration Period:** 6 months (redirect with warnings)

---

**3. `/chat/completions` behavior change in unified mode**

Before:
```typescript
POST /chat/completions { "messages": [...] }
// Uses server startup dataset
```

After:
```typescript
POST /chat/completions { "messages": [...] }
// Auto-routes to relevant subgraph in unified mode
// OR use force_subgraphs parameter
```

**Migration:** Non-breaking (automatic routing in unified mode)

---

## Questions for Review

1. **Breaking Changes**: Is requiring `data_source` in `/documents/upload` acceptable? (Alternative: maintain old behavior for default dataset)

2. **Deprecation Timeline**: Is 6 months (July 2025) enough time for frontend migration?

3. **HITL Priority**: Should Phase 4 (HITL) be implemented now or deferred to Phase 2 of core pipeline work?

4. **Batch Migration**: Should we provide automatic migration on server startup (detect old graphs and offer migration)?

5. **Metrics Storage**: Should extraction metrics be stored in GraphML or separate JSON file?

6. **Chat Completions**: Should we deprecate `/chat/completions` in favor of unified endpoints, or keep with unified mode support?

7. **Ask Endpoint**: Keep `/ask` as-is (retrieval-only) or merge with `/api/unified/ask`?

---

## Next Steps

1. **Review this plan** - Confirm approach and timeline
2. **Prioritize phases** - Can defer Phase 4 if needed
3. **Update frontend** - Coordinate with UI team for breaking changes
4. **Test with real data** - Verify enhanced pipeline works with existing datasets
5. **Document API changes** - Update frontend/backend README files

**Ready to proceed?** Please review and provide feedback before implementation.
