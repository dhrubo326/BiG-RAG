# Quick Start: Clean Implementation on feature/production-kg-educational

**Goal**: Add PipelineFeatures interface to EnhancedPipeline (working code) instead of fixing buggy UnifiedPipeline rewrite.

**Time**: 1-2 hours

**Risk**: LOW (only adding interface, not changing logic)

---

## Overview

This guide walks you through implementing the clean solution:

1. Switch to `feature/production-kg-educational` branch
2. Apply changes from CHANGES_NEEDED.md to EnhancedPipeline
3. Create new endpoint from ENDPOINT_GUIDE.md
4. Test using TESTING_CHECKLIST.md
5. Clean up obsolete files

---

## Step 1: Switch Branch and Verify

### 1.1 Switch to Clean Branch

```bash
# Save any uncommitted work on current branch
git stash

# Switch to production branch
git checkout feature/production-kg-educational

# Pull latest
git pull origin feature/production-kg-educational

# Verify you're on correct branch
git branch
# Should show: * feature/production-kg-educational
```

### 1.2 Verify EnhancedPipeline Exists

```bash
# Check file exists
ls bigrag/enhanced_pipeline.py
# Should show: bigrag/enhanced_pipeline.py

# Check it's working code
python -c "from bigrag.enhanced_pipeline import EnhancedKGPipeline; print('OK')"
# Should print: OK
```

### 1.3 Check PipelineFeatures File

```bash
# Check if features.py exists on this branch
ls bigrag/pipeline/features.py
```

**If exists**: Skip to Step 2

**If missing**: Copy from current branch:
```bash
# Copy features.py from feature/production-kg-unified branch
git checkout feature/production-kg-unified -- bigrag/pipeline/features.py

# Verify copy
ls bigrag/pipeline/features.py
# Should exist now
```

---

## Step 2: Apply Changes to EnhancedPipeline

**Reference**: CHANGES_NEEDED.md

### 2.1 Add PipelineFeatures Import

**File**: `bigrag/enhanced_pipeline.py`

**Location**: After existing imports (around line 32)

**Add**:
```python
from bigrag.pipeline.features import PipelineFeatures
```

**Verify**:
```bash
grep "from bigrag.pipeline.features import PipelineFeatures" bigrag/enhanced_pipeline.py
# Should show the line
```

### 2.2 Update __init__ Method

**File**: `bigrag/enhanced_pipeline.py`

**Location**: Lines 61-145 (class EnhancedKGPipeline __init__ method)

**Current Signature**:
```python
def __init__(
    self,
    api_key: str,
    model: str = "gpt-4o-mini",
    validation_level: str = "MODERATE",
    enable_entity_linking: bool = True,
    entity_merge_strategy: str = "fuzzy",
    extraction_strategy: str = "hybrid",
    extraction_mode: str = "semi_structured",
    review_queue_path: str = "expr/human_review_queue.json",
    dataset_path: Optional[str] = None
):
```

**Replace With** (see CHANGES_NEEDED.md lines 48-130 for full code):
```python
def __init__(
    self,
    features: PipelineFeatures = None,  # NEW: Primary interface
    # Legacy parameters (backward compatible)
    api_key: str = None,
    model: str = "gpt-4o-mini",
    validation_level: str = "MODERATE",
    enable_entity_linking: bool = True,
    entity_merge_strategy: str = "fuzzy",
    extraction_strategy: str = "hybrid",
    extraction_mode: str = "semi_structured",
    review_queue_path: str = "expr/human_review_queue.json",
    dataset_path: Optional[str] = None
):
    """
    Initialize Enhanced KG Pipeline.

    NEW: Accepts PipelineFeatures for full control via feature flags.
    Legacy parameters still work for backward compatibility.

    Args:
        features: PipelineFeatures object (recommended - provides full control)
        api_key: OpenAI API key (legacy - required if features not provided)
        ... (rest of legacy params)
    """

    # NEW: Map from PipelineFeatures if provided
    if features:
        # API Keys
        self.api_key = features.openai_api_key
        self.gemini_api_key = features.gemini_api_key

        # Model (keep parameter, not in features yet)
        self.model = model

        # Validation
        self.validation_level = features.validation_strictness

        # Entity Linking/Merging
        self.enable_entity_linking = features.enable_entity_merging
        self.entity_merge_strategy = features.merge_strategy

        # Extraction Strategy (map from gleaning flag)
        if features.enable_gleaning:
            self.extraction_strategy = "gleaning"
        elif features.enable_table_fact_extraction:
            self.extraction_strategy = "hybrid"  # Tables + LLM
        else:
            self.extraction_strategy = "strict"  # LLM only

        # Extraction Mode (keep default for now)
        self.extraction_mode = extraction_mode

        # HITL
        self.review_queue_path = review_queue_path

        # Dataset path
        self.dataset_path = dataset_path or "expr/default"

        # Store features for later reference
        self.features = features

    else:
        # Legacy mode - use parameters as before
        if api_key is None:
            raise ValueError("Either 'features' or 'api_key' must be provided")

        self.api_key = api_key
        self.gemini_api_key = None  # Not available in legacy mode
        self.model = model
        self.validation_level = validation_level
        self.enable_entity_linking = enable_entity_linking
        self.entity_merge_strategy = entity_merge_strategy
        self.extraction_strategy = extraction_strategy
        self.extraction_mode = extraction_mode
        self.review_queue_path = review_queue_path
        self.dataset_path = dataset_path
        self.features = None

    # Rest of __init__ stays EXACTLY THE SAME (component initialization)
    # Just continue with existing code from line 146+
```

**IMPORTANT**: Keep rest of __init__ unchanged! Only add the if/else block at the top.

### 2.3 Update Chunking Call (Optional)

**File**: `bigrag/enhanced_pipeline.py`

**Location**: Around line 285 (in process_document method)

**Current Code**:
```python
chunks = await self.chunker.chunk_document(
    markdown_text,
    chunk_size=1000,  # Hardcoded
    overlap=100,      # Hardcoded
    metadata=metadata
)
```

**Replace With**:
```python
# Determine chunk parameters
if self.features:
    chunk_size = self.features.chunk_size
    chunk_overlap = self.features.chunk_overlap
    use_semantic = (self.features.chunk_mode == "semantic")
else:
    # Legacy defaults
    chunk_size = 1000
    chunk_overlap = 100
    use_semantic = False

# Call chunker with determined parameters
chunks = await self.chunker.chunk_document(
    markdown_text=markdown_text,
    chunk_size=chunk_size,
    overlap=chunk_overlap,
    metadata=metadata,
    use_semantic_chunking=use_semantic
)
```

### 2.4 Replace print() with logger (Optional)

**File**: `bigrag/enhanced_pipeline.py`

**Find and Replace**:
- Find: `print(` → Replace: `logger.info(`
- Find: `print(f"` → Replace: `logger.info(f"`

**Add import at top** (if not already present):
```python
from bigrag.utils import logger
```

**Why Optional**: Not critical for functionality, but better for production.

---

## Step 3: Create Backend Endpoint

**Reference**: ENDPOINT_GUIDE.md

### 3.1 Create Endpoint File

**File**: `backend/api/routes/unified_indexing.py`

**Action**: Create new file with full content from ENDPOINT_GUIDE.md (lines 35-390).

**Copy-paste entire content** - it's 390 lines of complete, working code.

**Key Sections**:
- Lines 35-60: Imports and router setup
- Lines 64-81: IndexingResponse model
- Lines 83-314: Main endpoint function
- Lines 316-357: Helper function for BiGRAG processing
- Lines 359-390: Time/cost estimation

**Verify**:
```bash
# Check file created
ls backend/api/routes/unified_indexing.py
# Should exist

# Check it's valid Python
python -m py_compile backend/api/routes/unified_indexing.py
# Should show no errors
```

### 3.2 Register Router in server.py

**File**: `backend/server.py`

**Add Import** (around line 30, after other route imports):
```python
from api.routes.unified_indexing import router as unified_indexing_router
```

**Register Router** (around line 200, where other routers are added):
```python
app.include_router(unified_indexing_router)
```

**Verify**:
```bash
grep "unified_indexing_router" backend/server.py
# Should show 2 lines (import + include_router)
```

### 3.3 Update BiGRAG to Accept PipelineFeatures

**File**: `bigrag/bigrag.py`

**Location**: __init__ method

**Check if pipeline_features parameter exists**:
```bash
grep "pipeline_features" bigrag/bigrag.py
```

**If missing**, add to __init__:
```python
def __init__(
    self,
    working_dir: str = "./index_default",
    pipeline_features: PipelineFeatures = None,  # NEW
    # ... rest of parameters
):
    self.pipeline_features = pipeline_features or PipelineFeatures.from_preset("standard")
```

**If exists**: Verify it's used when initializing EnhancedPipeline.

---

## Step 4: Test Implementation

**Reference**: TESTING_CHECKLIST.md

### 4.1 Quick Unit Test

```bash
cd test_scripts

# Create test file
cat > test_quick_verify.py << 'EOF'
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures
import os

# Test 1: Legacy API
try:
    pipeline = EnhancedKGPipeline(api_key="test-key")
    print("[OK] Test 1: Legacy API works")
except Exception as e:
    print(f"[FAIL] Test 1: {e}")

# Test 2: PipelineFeatures API
try:
    features = PipelineFeatures.from_preset("standard", openai_api_key="test-key")
    pipeline = EnhancedKGPipeline(features=features)
    print("[OK] Test 2: PipelineFeatures API works")
except Exception as e:
    print(f"[FAIL] Test 2: {e}")

# Test 3: Features stored
try:
    assert pipeline.features is not None, "Features should be stored"
    assert pipeline.api_key == "test-key", "API key should match"
    print("[OK] Test 3: Features stored correctly")
except Exception as e:
    print(f"[FAIL] Test 3: {e}")

print("\n[SUCCESS] All quick tests passed!")
EOF

python test_quick_verify.py
```

**Expected Output**:
```
[OK] Test 1: Legacy API works
[OK] Test 2: PipelineFeatures API works
[OK] Test 3: Features stored correctly

[SUCCESS] All quick tests passed!
```

**If any fail**: Go back to Step 2 and verify changes are correct.

### 4.2 Start Backend Server

```bash
# Terminal 1: Start server
cd backend
python server.py --data_source test_endpoint

# Wait for startup message
# Should see: "INFO: Application startup complete."
```

### 4.3 Test Endpoint (Basic)

```bash
# Terminal 2: Create test file
echo "KUET offers 1065 seats across 18 departments." > test_basic.md

# Call endpoint
curl -X POST "http://localhost:8001/indexing/index-document" \
  -F "file=@test_basic.md" \
  -F "data_source=test_endpoint" \
  -F "title=Basic Test"
```

**Expected Response**:
```json
{
  "success": true,
  "message": "Document queued",
  "document_id": "doc-...",
  "features_enabled": {
    "table_extraction": false,
    "gleaning": false
  },
  "estimated_time": "30s"
}
```

**If 500 error**: Check backend/api.log for errors.

### 4.4 Test Endpoint (Quality)

```bash
# Use real KUET document
curl -X POST "http://localhost:8001/indexing/index-document" \
  -F "file=@datasets/SingleTopic/raw/KUET_Admission_info.md" \
  -F "data_source=test_quality" \
  -F "title=KUET Admission" \
  -F "need_table_extraction=true" \
  -F "need_gleaning=true" \
  -F "need_numeric_validation=true" \
  -F "merge_strategy=fuzzy" \
  -F "process_async=false"
```

**Wait**: ~2-3 minutes

**Verify Graph Created**:
```bash
# Check graph size
wc -l expr/test_quality/graph_chunk_entity_relation.graphml
# Expected: 1800-2200 lines
```

**If graph is tiny (<500 lines)**: Entity linking issue - check Step 2.2 was applied correctly.

### 4.5 Run Full Test Suite

```bash
cd test_scripts

# Run all tests from TESTING_CHECKLIST.md
python test_enhanced_pipeline_legacy.py && \
python test_enhanced_pipeline_features.py && \
python test_enhanced_pipeline_quality.py && \
python test_full_pipeline_standard.py && \
python test_full_pipeline_quality.py

# Check exit code
echo $?
# Should be 0 (success)
```

**If any test fails**: See TESTING_CHECKLIST.md for debugging steps.

---

## Step 5: Clean Up Obsolete Files

**After all tests pass**, delete buggy rewrite files:

```bash
# Delete UnifiedPipeline files (obsolete)
rm bigrag/pipeline/base_pipeline.py
rm bigrag/pipeline/unified_pipeline.py

# Verify deleted
ls bigrag/pipeline/
# Should NOT show base_pipeline.py or unified_pipeline.py
```

**Commit Changes**:
```bash
git add -A
git commit -m "Add PipelineFeatures interface to EnhancedPipeline

- Update EnhancedPipeline.__init__ to accept PipelineFeatures
- Create /indexing/index-document endpoint with feature flags
- Delete obsolete UnifiedPipeline files (buggy rewrite)
- All tests passing (graph quality verified)"
```

---

## Step 6: Rebuild All Graphs (Optional)

**If you want fresh graphs with quality preset**:

```bash
# Rebuild KUET dataset
python script_build.py --data_source SingleTopic --use_production_pipeline

# Rebuild other datasets
python script_build.py --data_source 2WikiMultiHopQA
python script_build.py --data_source HotpotQA

# Wait for completion (2-4 hours per dataset)
```

**Verify**:
```bash
# Check graph quality
ls -lh expr/SingleTopic/graph_chunk_entity_relation.graphml
# Should be 1800-2200 lines
```

---

## Troubleshooting

### Issue 1: Import Error (PipelineFeatures not found)

**Error**:
```
ModuleNotFoundError: No module named 'bigrag.pipeline.features'
```

**Fix**:
```bash
# Copy features.py from unified branch
git checkout feature/production-kg-unified -- bigrag/pipeline/features.py
```

### Issue 2: API Key Error

**Error**:
```
ValueError: Either 'features' or 'api_key' must be provided
```

**Fix**:
```bash
# Set API key environment variable
export OPENAI_API_KEY="your-key-here"

# Or on Windows
set OPENAI_API_KEY=your-key-here
```

### Issue 3: Graph Too Small

**Symptom**: Graph only 400-500 lines (should be 1800-2200).

**Cause**: Entity linking timing issue.

**Fix**: Verify Step 2.2 was applied correctly - check that entity linking happens AFTER merging (around line 523-540 in enhanced_pipeline.py).

### Issue 4: Endpoint Returns 500

**Symptom**: curl command returns Internal Server Error.

**Debug**:
```bash
# Check backend logs
tail -f backend/api.log

# Look for Python exceptions
grep "Traceback" backend/api.log
```

**Common Causes**:
- BiGRAG not initialized correctly
- Pipeline features validation failed
- Missing API key

---

## Summary

**Total Time**: 1-2 hours

**Steps**:
1. ✅ Switch to feature/production-kg-educational branch
2. ✅ Apply 3 changes to EnhancedPipeline (import, __init__, chunking)
3. ✅ Create new endpoint (copy-paste from ENDPOINT_GUIDE.md)
4. ✅ Test (quick verify + full test suite)
5. ✅ Clean up (delete obsolete files)
6. ⚠️ Optional: Rebuild all graphs with quality preset

**Success Criteria**:
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] KUET graph: 1800-2200 lines
- [ ] Orphan rate: <20%
- [ ] Endpoint returns 200 OK
- [ ] Backend logs show no errors

**Next Steps After Success**:
1. Update frontend to use new endpoint
2. Create preset selector UI
3. Add real-time indexing progress
4. Deploy to production

---

## Quick Reference Commands

```bash
# Switch branch
git checkout feature/production-kg-educational

# Test implementation
cd test_scripts && python test_quick_verify.py

# Start server
cd backend && python server.py --data_source test_endpoint

# Test endpoint
curl -X POST "http://localhost:8001/indexing/index-document" \
  -F "file=@test.md" -F "data_source=test" -F "title=Test"

# Verify graph
wc -l expr/test/graph_chunk_entity_relation.graphml

# Commit
git add -A && git commit -m "Add PipelineFeatures to EnhancedPipeline"
```

---

**Questions?** See:
- CHANGES_NEEDED.md - Detailed code changes
- ENDPOINT_GUIDE.md - Endpoint implementation
- TESTING_CHECKLIST.md - Full test suite

**Ready?** Start with Step 1!
