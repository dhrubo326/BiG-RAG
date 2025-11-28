# Changes Needed for `feature/production-kg-educational` Branch

**Target Branch**: `feature/production-kg-educational`
**Goal**: Add PipelineFeatures support to EnhancedPipeline
**Effort**: 1-2 hours
**Risk**: LOW (only adding, not changing existing logic)

---

## Overview

We'll add PipelineFeatures interface to EnhancedPipeline WITHOUT changing its core logic. This gives API flexibility while keeping proven extraction/linking code.

---

## File 1: `bigrag/enhanced_pipeline.py`

### Change 1.1: Add Import (Line ~32)

**Location**: After other imports, before class definition

**Add**:
```python
from bigrag.pipeline.features import PipelineFeatures
```

---

### Change 1.2: Update `__init__` Method (Lines 61-145)

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

**New Signature** (add `features` as first parameter):
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

---

### Change 1.3: Update Chunking Call (Line ~285)

**Current Code**:
```python
chunks = await self.chunker.chunk_document(
    markdown_text,
    chunk_size=1000,  # Hardcoded
    overlap=100,      # Hardcoded
    metadata=metadata
)
```

**New Code** (use features if available):
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

---

### Change 1.4: Replace print() with logger (Throughout file)

**Find and Replace**:
```python
# Find:    print(
# Replace: logger.info(

# Find:    print(f"
# Replace: logger.info(f"

# Find:    print("=" * 80)
# Replace: logger.info("=" * 80)

# Find:    print("-" * 80)
# Replace: logger.info("-" * 80)
```

**Add import at top** (if not already present):
```python
from bigrag.utils import logger
```

---

## File 2: `bigrag/pipeline/features.py`

**Action**: Copy from current branch OR create new file

**Location**: `bigrag/pipeline/features.py`

**Content**: This file should already exist in current branch. If not, copy it:

```bash
git checkout feature/production-kg-unified -- bigrag/pipeline/features.py
```

Or manually create with 15 feature flags + 3 presets (standard, quality, balanced).

---

## Summary of Changes

| File | Lines Changed | Risk | Effort |
|------|--------------|------|--------|
| `enhanced_pipeline.py` | ~100 lines (mostly in __init__) | LOW | 45 min |
| `features.py` | Copy from other branch | NONE | 5 min |
| Total | | LOW | 50 min |

---

## Testing After Changes

**Test 1**: Legacy API still works
```python
from bigrag.enhanced_pipeline import EnhancedKGPipeline

pipeline = EnhancedKGPipeline(
    api_key="your-key",
    validation_level="MODERATE"
)
# Should work exactly as before
```

**Test 2**: New PipelineFeatures API works
```python
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures

features = PipelineFeatures.from_preset("quality", openai_api_key="your-key")
pipeline = EnhancedKGPipeline(features=features)
# Should work with new API
```

**Test 3**: Process document
```python
result = await pipeline.process_document(
    "KUET offers 1065 seats",
    metadata={"title": "Test"}
)

print(f"Entities: {len(result['entities'])}")
print(f"Relations: {len(result['relations'])}")
# Should extract entities and relations
```

---

## Next Steps

After making these changes:
1. ✅ Commit changes to `feature/production-kg-educational`
2. ✅ Move to ENDPOINT_GUIDE.md to update backend endpoints
3. ✅ Test with TESTING_CHECKLIST.md

---

## Notes

- ✅ **Backward Compatible**: Old code using legacy parameters still works
- ✅ **No Logic Changes**: Core extraction/linking logic UNTOUCHED
- ✅ **Low Risk**: Only adding new interface, not changing behavior
- ✅ **Clean**: Single source of truth (EnhancedPipeline)
