# Backend Endpoint Guide

**Target Branch**: `feature/production-kg-educational`
**Goal**: Create `/indexing/index-document` endpoint with full feature control
**Effort**: 1 hour
**Risk**: LOW (new endpoint, doesn't affect existing code)

---

## Overview

We'll create a new endpoint that gives users full control over indexing features via parameters, without needing to understand presets.

**User Experience**:
```bash
# User just enables features they want
curl -F "need_table_extraction=true" \
     -F "need_gleaning=true" \
     -F "merge_strategy=fuzzy" \
     -F "file=@document.md"
```

**Backend translates** → PipelineFeatures → EnhancedPipeline

---

## File 1: Create New Endpoint

### Location: `backend/api/routes/unified_indexing.py`

**Action**: Create new file

**Full Content**:

```python
"""
Unified Indexing Endpoint with Feature Flags

Provides granular control over indexing without presets.
Users enable specific features via form parameters.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel

from bigrag.utils import logger
from bigrag.pipeline.features import PipelineFeatures
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag import BiGRAG
from ..services.utils import process_markdown, validate_file_upload, truncate_text
from ..services.kg_utils import compute_doc_id, add_document_to_corpus
from ..services.registry import registry


router = APIRouter(prefix="/indexing", tags=["Unified Indexing"])


# Response Model
class IndexingResponse(BaseModel):
    """Response for indexing endpoint"""
    success: bool
    message: str
    dataset_name: str
    document_id: str
    job_id: str
    filename: str
    title: str
    content_preview: str
    content_length: int
    status: str
    metadata: Optional[Dict[str, Any]] = None
    upload_date: str
    features_enabled: Dict[str, Any]
    estimated_time: str
    estimated_cost: str


@router.post("/index-document", response_model=IndexingResponse)
async def index_document_with_features(
    background_tasks: BackgroundTasks,

    # Required
    file: UploadFile = File(..., description="Document to index (.txt or .md)"),
    data_source: str = Form(..., description="Dataset name"),

    # Optional metadata
    title: str = Form(None, description="Document title"),
    metadata: str = Form(None, description="JSON metadata"),
    process_async: bool = Form(True, description="Process in background"),

    # Chunking features
    need_table_extraction: bool = Form(False, description="Extract tables with GPT-4"),
    need_dynamic_chunking: bool = Form(False, description="Semantic vs token chunking"),

    # Extraction features
    need_gleaning: bool = Form(False, description="Multi-pass extraction"),
    gleaning_iterations: int = Form(2, description="Gleaning passes"),
    need_table_fact_extraction: bool = Form(False, description="Rule-based table facts"),
    extraction_concurrency: int = Form(16, description="Parallel LLM calls"),

    # Validation features
    need_numeric_validation: bool = Form(False, description="Validate numeric accuracy"),
    need_semantic_validation: bool = Form(False, description="Filter low-quality entities"),
    validation_strictness: str = Form("MODERATE", description="STRICT|MODERATE|LENIENT"),

    # Merging features
    merge_strategy: str = Form("basic", description="basic|fuzzy|hybrid"),

    # Quality features
    enable_hitl: bool = Form(False, description="Human-in-the-loop"),
    enable_orphan_linking: bool = Form(False, description="Link orphan entities"),
    enable_quality_scoring: bool = Form(False, description="Track quality metrics")
):
    """
    Index document with explicit feature control.

    **No presets needed** - just enable features you want!

    Example (Fast & Cheap):
    ```bash
    curl -X POST "http://localhost:8001/indexing/index-document" \\
      -F "file=@doc.md" \\
      -F "data_source=my_data"
      # All defaults = fast, basic extraction
    ```

    Example (High Quality):
    ```bash
    curl -X POST "http://localhost:8001/indexing/index-document" \\
      -F "file=@doc.md" \\
      -F "data_source=my_data" \\
      -F "need_table_extraction=true" \\
      -F "need_gleaning=true" \\
      -F "need_numeric_validation=true" \\
      -F "merge_strategy=fuzzy"
    ```
    """
    try:
        # Step 1: Validate file
        if not file.filename.endswith(('.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .md files supported"
            )

        # Step 2: Read content
        content_bytes = await file.read()
        is_valid, error_msg = validate_file_upload(content_bytes, file.filename, max_size_mb=50)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        content_text = content_bytes.decode('utf-8')

        # Process markdown if needed
        if file.filename.endswith('.md'):
            try:
                content_text = process_markdown(content_text)
            except Exception as e:
                logger.warning(f"Markdown processing failed: {e}. Using raw content.")

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="Empty file")

        # Step 3: Parse metadata
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON metadata")

        # Step 4: Build PipelineFeatures from form parameters
        features = PipelineFeatures(
            # Chunking
            enable_table_detection=need_table_extraction,
            chunk_mode="semantic" if need_dynamic_chunking else "token",
            chunk_size=1200,
            chunk_overlap=100,

            # Extraction
            enable_gleaning=need_gleaning,
            max_gleaning_iterations=gleaning_iterations if need_gleaning else 1,
            enable_table_fact_extraction=need_table_fact_extraction,
            extraction_concurrency=extraction_concurrency,

            # Validation
            enable_numeric_validation=need_numeric_validation,
            enable_entity_validation=need_semantic_validation,
            enable_relation_validation=need_semantic_validation,
            validation_strictness=validation_strictness,

            # Merging
            merge_strategy=merge_strategy,
            enable_entity_merging=True,  # Always merge

            # Quality
            enable_hitl=enable_hitl,
            enable_orphan_linking=enable_orphan_linking,
            enable_quality_scoring=enable_quality_scoring,

            # API Keys
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            gemini_api_key=os.getenv('GEMINI_API_KEY')
        )

        # Validate features
        warnings = features.validate()
        if warnings:
            logger.warning(f"[Indexing] Feature warnings: {warnings}")

        # Step 5: Ensure dataset exists
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
        expr_dir = PROJECT_ROOT / "expr" / data_source
        dataset_dir = PROJECT_ROOT / "datasets" / data_source / "raw"
        expr_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Step 6: Generate IDs
        doc_id = compute_doc_id(content_text, prefix="doc")
        job_id = f"job-{compute_doc_id(doc_id + str(datetime.now()), prefix='')}"
        doc_title = title or file.filename

        # Step 7: Add to corpus
        await add_document_to_corpus(
            data_source=data_source,
            doc_id=doc_id,
            content=content_text,
            title=doc_title,
            metadata=doc_metadata
        )

        # Step 8: Add to registry
        await registry.add_document(
            document_id=doc_id,
            filename=file.filename,
            title=doc_title,
            content_length=len(content_text),
            dataset=data_source,
            metadata=doc_metadata,
            job_id=job_id,
            status="pending"
        )

        # Step 9: Estimate time and cost
        estimated_time, estimated_cost = _estimate_time_cost(features, len(content_text))

        # Step 10: Process document
        if process_async:
            # Background processing
            background_tasks.add_task(
                _process_with_bigrag,
                data_source,
                doc_id,
                content_text,
                doc_title,
                doc_metadata,
                features
            )
            status = "processing"
            logger.info(f"[Indexing] Queued job: {job_id}")
        else:
            # Synchronous processing
            await _process_with_bigrag(
                data_source,
                doc_id,
                content_text,
                doc_title,
                doc_metadata,
                features
            )
            status = "completed"
            logger.info(f"[Indexing] Completed synchronously")

        # Step 11: Build response
        return IndexingResponse(
            success=True,
            message=f"Document {'queued' if process_async else 'indexed'}",
            dataset_name=data_source,
            document_id=doc_id,
            job_id=job_id,
            filename=file.filename,
            title=doc_title,
            content_preview=truncate_text(content_text, 200),
            content_length=len(content_text),
            status=status,
            metadata=doc_metadata,
            upload_date=datetime.now().isoformat(),
            features_enabled={
                "table_extraction": need_table_extraction,
                "dynamic_chunking": need_dynamic_chunking,
                "gleaning": need_gleaning,
                "table_fact_extraction": need_table_fact_extraction,
                "numeric_validation": need_numeric_validation,
                "semantic_validation": need_semantic_validation,
                "merge_strategy": merge_strategy,
                "hitl": enable_hitl
            },
            estimated_time=estimated_time,
            estimated_cost=estimated_cost
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Indexing] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


async def _process_with_bigrag(
    dataset_name: str,
    document_id: str,
    content: str,
    title: str,
    metadata: Dict[str, Any],
    features: PipelineFeatures
) -> Dict[str, Any]:
    """
    Process document using BiGRAG with EnhancedPipeline.

    BiGRAG handles:
    - EnhancedPipeline initialization with features
    - Graph building from pipeline results
    - Storage to all backends (GraphML, vector DBs, KV stores)
    """
    try:
        logger.info(f"[Indexing] Processing {document_id}")

        # Initialize BiGRAG with features
        rag = BiGRAG(
            working_dir=f"expr/{dataset_name}",
            pipeline_features=features
        )

        # Insert document (BiGRAG uses EnhancedPipeline internally)
        await rag.ainsert(
            [content],
            metadatas=[{
                "title": title,
                "document_id": document_id,
                **metadata
            }]
        )

        logger.info(f"[Indexing] Completed: {document_id}")
        return {"success": True, "document_id": document_id}

    except Exception as e:
        logger.error(f"[Indexing] Failed: {e}")
        raise


def _estimate_time_cost(features: PipelineFeatures, content_length: int) -> tuple:
    """Estimate processing time and cost."""
    base_time = 30  # seconds
    base_cost = 0.05  # USD

    if features.enable_table_detection:
        base_time += 30
        base_cost += 0.05

    if features.enable_gleaning:
        base_time += 60 * features.max_gleaning_iterations
        base_cost += 0.10 * features.max_gleaning_iterations

    if features.enable_numeric_validation:
        base_time += 20
        base_cost += 0.05

    # Scale by document size
    scale = content_length / 40000
    total_time = int(base_time * scale)
    total_cost = base_cost * scale

    # Format
    if total_time < 60:
        time_str = f"{total_time}s"
    else:
        time_str = f"{total_time // 60}min {total_time % 60}s"

    cost_str = f"${total_cost:.3f}" if total_cost < 0.10 else f"${total_cost:.2f}"

    return time_str, cost_str
```

---

## File 2: Update Existing Endpoint (Optional)

### Location: `backend/api/routes/datasets.py`

**Endpoint**: `/datasets/create-and-index`

**Current**: Uses production pipeline (hardcoded)

**Update**: Add optional `features` parameter

**Changes**:

```python
# Line ~150 (in create_and_index endpoint)

# Add optional parameter
use_custom_features: bool = Form(False, description="Use custom features (not preset)")

# If custom features requested, build from parameters
if use_custom_features:
    features = PipelineFeatures(
        # Map from request parameters
        enable_table_detection=request.get('enable_table_detection', False),
        # ... etc
    )
else:
    # Use quality preset (default)
    features = PipelineFeatures.from_preset("quality", openai_api_key=os.getenv('OPENAI_API_KEY'))

# Use features when calling BiGRAG
rag = BiGRAG(
    working_dir=f"expr/{dataset_name}",
    pipeline_features=features  # NEW
)
```

---

## File 3: Register New Router

### Location: `backend/server.py`

**Add import**:
```python
from api.routes.unified_indexing import router as unified_indexing_router
```

**Register router**:
```python
# Around line 200 (where other routers are added)
app.include_router(unified_indexing_router)
```

---

## Testing the Endpoint

### Test 1: Basic Indexing (Fast)

```bash
curl -X POST "http://localhost:8001/indexing/index-document" \
  -F "file=@test.md" \
  -F "data_source=test_basic" \
  -F "title=Test Document"
```

**Expected**:
- Status: 200
- Response includes `estimated_time`, `estimated_cost`
- Graph created in `expr/test_basic/`

---

### Test 2: Quality Indexing (All Features)

```bash
curl -X POST "http://localhost:8001/indexing/index-document" \
  -F "file=@KUET_Admission_info.md" \
  -F "data_source=kuet_quality" \
  -F "title=KUET Admission" \
  -F "need_table_extraction=true" \
  -F "need_gleaning=true" \
  -F "gleaning_iterations=2" \
  -F "need_table_fact_extraction=true" \
  -F "need_numeric_validation=true" \
  -F "need_semantic_validation=true" \
  -F "merge_strategy=fuzzy" \
  -F "enable_hitl=true" \
  -F "enable_orphan_linking=true"
```

**Expected**:
- Graph has 40-60 edges (not 5!)
- Orphan warnings: 2-3 (not 12!)
- GraphML: 1800-2200 lines (not 408!)

---

### Test 3: Custom Mix

```bash
curl -X POST "http://localhost:8001/indexing/index-document" \
  -F "file=@doc.md" \
  -F "data_source=custom" \
  -F "need_table_extraction=true" \
  -F "need_numeric_validation=true" \
  -F "merge_strategy=fuzzy"
  # Only tables + validation + fuzzy merge
```

---

## Comparison: Old vs New

### Old Way (Preset-Based)

```bash
# All or nothing
curl -F "preset=quality"  # Gets ALL features
curl -F "preset=standard"  # Gets BASIC features
# Can't mix and match!
```

### New Way (Feature-Based)

```bash
# Pick exactly what you need
curl -F "need_table_extraction=true" \
     -F "need_numeric_validation=true"
     # Only these 2 features, nothing else!
```

---

## Summary

| Endpoint | Purpose | Flexibility | Use Case |
|----------|---------|-------------|----------|
| `/indexing/index-document` | NEW | Full control (15 flags) | Custom indexing needs |
| `/datasets/create-and-index` | Existing | Preset-based | Quick batch indexing |

Both endpoints now use **EnhancedPipeline with PipelineFeatures** under the hood!

---

## Next Steps

1. ✅ Create the endpoint file
2. ✅ Register router in server.py
3. ✅ Test with curl commands
4. ✅ Verify graph quality (see TESTING_CHECKLIST.md)
