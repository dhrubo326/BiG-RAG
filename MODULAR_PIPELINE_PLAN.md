# Modular Unified Pipeline - Implementation Plan

**Version**: 1.0
**Date**: November 26, 2024
**Status**: Planning Phase

---

## 🎯 Executive Summary

**🚨 DEVELOPMENT PHILOSOPHY**: Build the ideal modular pipeline - **NO backward compatibility constraints**.

Replace all existing pipelines with **ONE modular unified pipeline** that:
- ✅ Imports proven components from `operate.py` (don't duplicate code)
- ✅ Integrates **all enhanced features** (table extraction, validation, smart chunking)
- ✅ Allows **mix-and-match features** via simple configuration flags
- ✅ Clean, modern architecture - **no legacy support needed**

**Core Philosophy**: "Build it right, not backward compatible"

**Project Status**:
- 🟢 **Development Phase** - APIs can change, graphs will be rebuilt, old code is reference only
- 🟢 **Old pipelines kept for emergency rollback only** - Not for gradual migration
- 🟢 **Focus**: Clean, modular, ideal implementation (not compatibility layers)

---

## 📊 Current State Analysis

### Existing Pipelines

| Pipeline | File | Accuracy | Speed | Features | Issues |
|----------|------|----------|-------|----------|---------|
| **Standard** | `operate.py` | 90-95% | Fast | Basic extraction, gleaning | ✅ Reliable but limited |
| **Production** | `production_pipeline.py` | 95-98% | Medium | + Table extraction, validation | ⚠️ DEPRECATED |
| **Enhanced** | `enhanced_pipeline.py` | 95-99% | Slow | + All features | ❌ High failure rate (58%) |

### Problems

1. **All-or-Nothing**: Must choose entire pipeline, can't mix features
2. **BUET Failure**: Enhanced pipeline rejected 70% of extractions (too strict)
3. **Code Duplication**: Same chunking/extraction logic in 3 places
4. **Confusing API**: Users don't know which pipeline to use

### Real-World Example

**BUET Document** (43K chars):
- Enhanced pipeline: 29 entities, 11 relations (58% failed validation) ❌
- Standard pipeline: Would get ~80-90 entities, ~70-80 relations ✅

**Need**: Use standard extraction + fuzzy merging (no validation)

---

## 🏗️ Proposed Architecture

### Core Concept: Feature Flags

Instead of choosing a pipeline, choose features:

```python
# OLD WAY (All or nothing)
rag = BiGRAG(use_enhanced_pipeline=True)  # Get ALL features or NONE

# NEW WAY (Modular)
rag = BiGRAG(
    pipeline_features=PipelineFeatures(
        enable_table_extraction=True,      # Yes, extract tables
        enable_numeric_validation=False,   # No, skip validation
        enable_fuzzy_merging=True,         # Yes, better entity merging
        enable_gleaning=False              # No, single-pass is fine
    )
)
```

### Feature Categories

#### 1. **Chunking Features**
- `enable_table_detection`: GPT-4 table extraction (from enhanced pipeline)
- `chunk_mode`: "token" (standard) | "semantic" (enhanced) | "hybrid"

#### 2. **Extraction Features**
- `enable_gleaning`: Multi-pass extraction with conversation history (standard pipeline)
- `enable_table_fact_extraction`: Rule-based table fact extraction (enhanced pipeline)
- `extraction_concurrency`: Parallel LLM calls (standard pipeline optimization)

#### 3. **Validation Features**
- `enable_numeric_validation`: Gemini-based numeric consistency check (enhanced pipeline)
- `enable_entity_validation`: Entity quality scoring (enhanced pipeline)
- `enable_relation_validation`: Relation completeness check (enhanced pipeline)
- `validation_strictness`: "STRICT" | "MODERATE" | "LENIENT"

#### 4. **Merging Features**
- `enable_entity_merging`: Entity deduplication (both pipelines)
- `merge_strategy`: "basic" (standard) | "fuzzy" (enhanced) | "hybrid"

#### 5. **Quality Features**
- `enable_hitl`: Save failed extractions for human review (enhanced pipeline)
- `enable_orphan_linking`: Post-merge orphan entity fixing (enhanced pipeline)
- `enable_quality_scoring`: Track extraction quality metrics (enhanced pipeline)

### Preset Configurations

For common use cases, provide presets:

```python
# Preset 1: STANDARD (Fast, Reliable)
PipelineFeatures.from_preset("standard")
# = Current standard pipeline behavior
# - Token chunking
# - Single-pass extraction with gleaning
# - Basic entity merging
# - No validation

# Preset 2: QUALITY (Slow, Accurate)
PipelineFeatures.from_preset("quality")
# = Current enhanced pipeline behavior
# - Table-aware chunking
# - Table fact extraction
# - All validations enabled
# - Fuzzy entity merging
# - HITL enabled

# Preset 3: BALANCED (Medium Speed/Quality)
PipelineFeatures.from_preset("balanced")
# = Strategic feature selection
# - Table detection enabled
# - Single-pass extraction (no gleaning)
# - Fuzzy merging
# - Validation LENIENT
# - HITL enabled
```

---

## 📁 New Directory Structure

```
bigrag/
├── pipeline/                          # NEW: Modular pipeline system
│   ├── __init__.py
│   ├── base_pipeline.py              # NEW: Core unified pipeline class
│   ├── features.py                   # NEW: Feature configuration dataclass
│   │
│   ├── chunkers/                     # Chunking modules
│   │   ├── __init__.py
│   │   ├── token_chunker.py          # EXTRACT from operate.py
│   │   ├── table_chunker.py          # MOVE from preprocessors/smart_chunker.py
│   │   └── semantic_chunker.py       # FUTURE: Semantic boundary detection
│   │
│   ├── extractors/                   # Extraction modules
│   │   ├── __init__.py
│   │   ├── llm_extractor.py          # EXTRACT from operate.py (gleaning support)
│   │   ├── table_extractor.py        # ALIAS to extractors/table_fact_extractor.py
│   │   └── hybrid_extractor.py       # NEW: Combines LLM + table extraction
│   │
│   ├── validators/                   # Validation modules
│   │   ├── __init__.py
│   │   ├── numeric_validator.py      # ALIAS to validators/numeric_validator.py
│   │   ├── entity_validator.py       # EXTRACT from enhanced_pipeline.py
│   │   └── relation_validator.py     # EXTRACT from enhanced_pipeline.py
│   │
│   ├── mergers/                      # Entity merging modules
│   │   ├── __init__.py
│   │   ├── basic_merger.py           # EXTRACT from operate.py
│   │   ├── fuzzy_merger.py           # ALIAS to merging/unified_merger.py
│   │   └── hybrid_merger.py          # FUTURE: Adaptive strategy
│   │
│   └── postprocessors/               # Post-processing modules
│       ├── __init__.py
│       ├── orphan_linker.py          # EXTRACT from enhanced_pipeline.py
│       └── quality_scorer.py         # EXTRACT from enhanced_pipeline.py
│
├── operate.py                        # KEEP: Standard pipeline (backward compatibility)
├── enhanced_pipeline.py              # KEEP: During migration, then deprecate
├── bigrag.py                         # MODIFY: Add unified pipeline support
│
├── extractors/                       # KEEP: Existing modules (used by pipeline/)
│   ├── constrained_extractor.py
│   └── table_fact_extractor.py
│
├── validators/                       # KEEP: Existing modules
│   └── numeric_validator.py
│
├── preprocessors/                    # KEEP: Existing modules
│   ├── table_extractor.py
│   └── smart_chunker.py
│
├── merging/                          # KEEP: Existing modules
│   ├── unified_merger.py
│   ├── canonicalization.py
│   └── entity_linker.py
│
└── hitl/                             # KEEP: Existing modules
    └── failed_extraction_store.py
```

**Key Principles**:
- **EXTRACT**: Move functions from monolithic files to modules (don't duplicate)
- **ALIAS**: Reference existing modules via imports (don't duplicate)
- **KEEP**: Preserve existing files for backward compatibility

---

## 🔧 Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Create Feature Configuration

**File**: `bigrag/pipeline/features.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PipelineFeatures:
    """
    Modular pipeline feature configuration.

    Each feature can be independently enabled/disabled.
    Use presets for common configurations.
    """

    # ========== CHUNKING FEATURES ==========
    enable_table_detection: bool = False
    """Use GPT-4 to detect and extract tables (enhanced pipeline)"""

    chunk_mode: str = "token"
    """Chunking strategy: 'token' (standard) | 'semantic' (enhanced) | 'hybrid'"""

    chunk_size: int = 1200
    """Chunk size in tokens (standard pipeline default)"""

    chunk_overlap: int = 100
    """Chunk overlap in tokens (standard pipeline default)"""

    # ========== EXTRACTION FEATURES ==========
    enable_gleaning: bool = True
    """Multi-pass extraction with conversation history (standard pipeline)"""

    max_gleaning_iterations: int = 2
    """Maximum gleaning passes (standard pipeline default)"""

    enable_table_fact_extraction: bool = False
    """Rule-based fact extraction from tables (enhanced pipeline)"""

    extraction_concurrency: int = 16
    """Parallel LLM API calls (standard pipeline default)"""

    # ========== VALIDATION FEATURES ==========
    enable_numeric_validation: bool = False
    """Gemini-based numeric consistency check (enhanced pipeline)"""

    enable_entity_validation: bool = False
    """Entity quality scoring and filtering (enhanced pipeline)"""

    enable_relation_validation: bool = False
    """Relation completeness validation (enhanced pipeline)"""

    validation_strictness: str = "MODERATE"
    """Validation level: 'STRICT' (99%) | 'MODERATE' (95%) | 'LENIENT' (80%)"""

    # ========== MERGING FEATURES ==========
    enable_entity_merging: bool = True
    """Entity deduplication (both pipelines)"""

    merge_strategy: str = "basic"
    """Merging approach: 'basic' (fast) | 'fuzzy' (accurate) | 'hybrid'"""

    # ========== QUALITY FEATURES ==========
    enable_hitl: bool = False
    """Save failed extractions for human review (enhanced pipeline)"""

    enable_orphan_linking: bool = False
    """Post-merge orphan entity linking (enhanced pipeline)"""

    enable_quality_scoring: bool = False
    """Track extraction quality metrics (enhanced pipeline)"""

    # ========== API KEYS ==========
    openai_api_key: Optional[str] = None
    """OpenAI API key for table extraction and LLM-based extraction"""

    gemini_api_key: Optional[str] = None
    """Gemini API key for numeric validation (optional, falls back to regex)"""

    # ========== METADATA ==========
    pipeline_version: str = "unified-v1.0"
    """Pipeline version identifier"""

    def validate(self) -> List[str]:
        """
        Validate feature dependencies and return warnings.

        Returns:
            List of warning messages for dependency issues

        Examples:
            >>> features = PipelineFeatures(enable_smart_chunking=True)
            >>> warnings = features.validate()
            >>> # Returns: ["smart_chunking requires table_detection"]
        """
        warnings = []

        # === CHUNKING DEPENDENCIES ===
        if self.chunk_mode == "semantic" and not self.enable_table_detection:
            warnings.append(
                "chunk_mode='semantic' requires enable_table_detection=True. "
                "Will fall back to 'token' mode."
            )

        # === EXTRACTION DEPENDENCIES ===
        if self.enable_table_fact_extraction and not self.enable_table_detection:
            warnings.append(
                "enable_table_fact_extraction requires enable_table_detection=True. "
                "Table fact extraction will be skipped."
            )

        # === API KEY DEPENDENCIES ===
        if self.enable_table_detection and not self.openai_api_key:
            warnings.append(
                "enable_table_detection requires OpenAI API key. "
                "Feature will be disabled at runtime unless OPENAI_API_KEY env var is set."
            )

        if self.enable_numeric_validation and not self.gemini_api_key:
            warnings.append(
                "enable_numeric_validation works best with Gemini API key. "
                "Will use regex-based validation as fallback (lower accuracy)."
            )

        # === MERGING DEPENDENCIES ===
        if self.merge_strategy == "fuzzy" and not self.enable_entity_merging:
            warnings.append(
                "merge_strategy='fuzzy' requires enable_entity_merging=True. "
                "Will be ignored."
            )

        # === POST-PROCESSING DEPENDENCIES ===
        if self.enable_orphan_linking and not self.enable_entity_merging:
            warnings.append(
                "enable_orphan_linking requires enable_entity_merging=True. "
                "Orphan linking needs merged entity graph to work correctly."
            )

        # === VALIDATION STRICTNESS ===
        valid_strictness = ["STRICT", "MODERATE", "LENIENT"]
        if self.validation_strictness not in valid_strictness:
            warnings.append(
                f"validation_strictness='{self.validation_strictness}' is invalid. "
                f"Choose from: {valid_strictness}. Defaulting to 'MODERATE'."
            )

        return warnings

    @classmethod
    def from_preset(cls, preset: str, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None) -> 'PipelineFeatures':
        """
        Create feature configuration from preset.

        Args:
            preset: 'standard' | 'quality' | 'balanced'

        Returns:
            PipelineFeatures instance

        Examples:
            >>> features = PipelineFeatures.from_preset("standard")
            >>> features.enable_gleaning
            True
            >>> features.enable_numeric_validation
            False
        """
        presets = {
            "standard": cls._preset_standard(),
            "quality": cls._preset_quality(),
            "balanced": cls._preset_balanced()
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")

        return presets[preset]

    @classmethod
    def _preset_standard(cls, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None) -> 'PipelineFeatures':
        """
        STANDARD preset: Fast, reliable (current standard pipeline).

        Use for:
        - Large documents where speed matters
        - Documents without complex tables
        - When validation is too strict (e.g., BUET)

        Performance: ~30-60 seconds for 40K document
        Accuracy: 90-95%
        Cost: ~$0.15 per 40K document (GPT-4o-mini extraction only)
        API Calls: ~20-30 (extraction + gleaning)
        """
        return cls(
            # Chunking
            enable_table_detection=False,
            chunk_mode="token",
            chunk_size=1200,
            chunk_overlap=100,

            # Extraction
            enable_gleaning=True,
            max_gleaning_iterations=2,
            enable_table_fact_extraction=False,
            extraction_concurrency=16,

            # Validation
            enable_numeric_validation=False,
            enable_entity_validation=False,
            enable_relation_validation=False,

            # Merging
            enable_entity_merging=True,
            merge_strategy="basic",

            # Quality
            enable_hitl=False,
            enable_orphan_linking=False,
            enable_quality_scoring=False,

            # API Keys
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key
        )

    @classmethod
    def _preset_quality(cls, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None) -> 'PipelineFeatures':
        """
        QUALITY preset: Slow, accurate (current enhanced pipeline).

        Use for:
        - Educational/technical documents with tables
        - When accuracy is critical
        - Small to medium documents (<50K chars)

        Performance: ~2-5 minutes for 40K document
        Accuracy: 95-99%
        Cost: ~$0.40-0.60 per 40K document (table extraction + validation + gleaning)
        API Calls: ~60-100 (table detection + extraction + validation + gleaning)
        """
        return cls(
            # Chunking
            enable_table_detection=True,
            chunk_mode="semantic",
            chunk_size=1200,
            chunk_overlap=100,

            # Extraction
            enable_gleaning=True,
            max_gleaning_iterations=2,
            enable_table_fact_extraction=True,
            extraction_concurrency=16,

            # Validation
            enable_numeric_validation=True,
            enable_entity_validation=True,
            enable_relation_validation=True,
            validation_strictness="MODERATE",

            # Merging
            enable_entity_merging=True,
            merge_strategy="fuzzy",

            # Quality
            enable_hitl=True,
            enable_orphan_linking=True,
            enable_quality_scoring=True,

            # API Keys
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key
        )

    @classmethod
    def _preset_balanced(cls, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None) -> 'PipelineFeatures':
        """
        BALANCED preset: Medium speed/quality.

        Use for:
        - General-purpose documents
        - When you want some validation but not too strict
        - Medium documents (20-50K chars)

        Performance: ~1-2 minutes for 40K document
        Accuracy: 92-96%
        Cost: ~$0.25-0.35 per 40K document (table extraction + single-pass extraction)
        API Calls: ~40-60 (table detection + extraction, no gleaning)
        """
        return cls(
            # Chunking
            enable_table_detection=True,
            chunk_mode="token",
            chunk_size=1200,
            chunk_overlap=100,

            # Extraction
            enable_gleaning=False,  # Single-pass for speed
            enable_table_fact_extraction=True,
            extraction_concurrency=16,

            # Validation
            enable_numeric_validation=False,
            enable_entity_validation=True,
            enable_relation_validation=False,
            validation_strictness="LENIENT",

            # Merging
            enable_entity_merging=True,
            merge_strategy="fuzzy",

            # Quality
            enable_hitl=True,
            enable_orphan_linking=False,
            enable_quality_scoring=False,

            # API Keys
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key
        )


# ========== VALIDATION THRESHOLDS ==========

# Validation strictness levels define quality thresholds for filtering entities and relations
VALIDATION_THRESHOLDS = {
    "STRICT": {
        "numeric_coverage_min": 0.95,      # 95% of numbers in source must be found in extraction
        "entity_quality_min": 0.90,         # 90% quality score (description completeness)
        "relation_completeness_min": 8.0,   # 8/10 completeness score
        "description_min_length": 20,       # Entity descriptions must be >= 20 chars
        "allow_generic_types": False,       # Reject generic types like "OTHER", "UNKNOWN"
    },
    "MODERATE": {
        "numeric_coverage_min": 0.85,      # 85% of numbers must be found
        "entity_quality_min": 0.75,         # 75% quality score
        "relation_completeness_min": 6.0,   # 6/10 completeness score
        "description_min_length": 10,       # Descriptions must be >= 10 chars
        "allow_generic_types": True,        # Allow generic types with warning
    },
    "LENIENT": {
        "numeric_coverage_min": 0.70,      # 70% of numbers must be found
        "entity_quality_min": 0.60,         # 60% quality score
        "relation_completeness_min": 4.0,   # 4/10 completeness score
        "description_min_length": 5,        # Descriptions must be >= 5 chars
        "allow_generic_types": True,        # Allow all types
    }
}

# Quality scoring formula (used by enable_quality_scoring feature)
# entity_quality_score = (
#     0.4 * description_completeness +  # How detailed is the description?
#     0.3 * context_relevance +          # Is entity mentioned in relevant context?
#     0.2 * source_count +               # How many chunks mention this entity?
#     0.1 * type_specificity             # Is type specific vs. generic?
# )


# ========== QUALITY SCORING ALGORITHM ==========

def description_quality_score(description: str) -> float:
    """
    Calculate quality score for entity description (used in gleaning merge).

    Scoring factors:
    - Length (40 points max): Longer descriptions are more detailed
    - Keyword density (30 points max): Informative words (who, what, where, when)
    - Specificity (30 points max): Numbers, dates, proper names

    Returns: Score 0-100

    Examples:
        >>> description_quality_score("KUET has 18 departments established in 1967.")
        85.0  # Good length, has numbers and dates

        >>> description_quality_score("University")
        10.0  # Too short, no details
    """
    if not description:
        return 0.0

    score = 0.0

    # Factor 1: Length (up to 40 points)
    length_score = min(len(description) / 5, 40)  # 200 chars = 40 points
    score += length_score

    # Factor 2: Keyword density (up to 30 points)
    informative_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which',
                         'কে', 'কি', 'কোথায়', 'কেন']
    keyword_count = sum(1 for word in informative_words if word in description.lower())
    keyword_score = min(keyword_count * 10, 30)
    score += keyword_score

    # Factor 3: Specificity (up to 30 points)
    import re
    has_numbers = bool(re.search(r'\d', description))
    has_dates = bool(re.search(r'\d{4}|\d{1,2}/\d{1,2}', description))
    has_names = bool(re.search(r'[A-Z][a-z]+|[অ-হ]{3,}', description))

    specificity_score = (
        (10 if has_numbers else 0) +
        (10 if has_dates else 0) +
        (10 if has_names else 0)
    )
    score += specificity_score

    return score
```

---

## 🔧 Implementation Notes for Developers

### Semantic Chunking Algorithm

**Critical Details** (for `enable_table_detection=True` + `chunk_mode="semantic"`):

#### 1. Accumulation Decision Logic

Decision to flush happens **BEFORE** adding next paragraph:

```python
# Pseudo-code for paragraph accumulation
current_tokens = count_tokens(current_chunk)
para_tokens = count_tokens(next_paragraph)

if current_tokens + para_tokens <= 1000:
    # Keep accumulating
    current_chunk.append(next_paragraph)
elif 1000 < current_tokens + para_tokens <= 1300:
    # Overflow zone (30% tolerance)
    if current_tokens >= 1000:
        flush_current_chunk()  # Already large enough
        current_chunk = [next_paragraph]
    else:
        # Allow overflow to preserve paragraph
        current_chunk.append(next_paragraph)
elif current_tokens + para_tokens > 1300:
    # Hard limit exceeded - MUST flush
    flush_current_chunk()
    current_chunk = [next_paragraph]
```

#### 2. Asymmetric Overlap Strategy

Overlap depends on chunk position:

| Chunk Position | Overlap Before | Overlap After | Total Overlap |
|----------------|----------------|---------------|---------------|
| First chunk | 0 tokens | 100 tokens | 100 tokens |
| Middle chunks | 100 tokens | 100 tokens | 200 tokens |
| Last chunk | 100 tokens | 0 tokens | 100 tokens |
| Single chunk | 0 tokens | 0 tokens | 0 tokens |

```python
# Overlap calculation
if chunk_index == 0:
    overlap_before = 0
    overlap_after = 100
elif chunk_index == total_chunks - 1:
    overlap_before = 100
    overlap_after = 0
else:
    overlap_before = 100
    overlap_after = 100
```

#### 3. Sentence Boundary Detection

```python
# Bengali sentence endings: । (dari), ! (exclamation), ? (question)
# English: . ! ?
sentence_pattern = r'([।!?।।]+\s*|[.!?]+\s+)'

# Split by sentence endings (keep delimiters)
sentences = re.split(sentence_pattern, text)
```

**Reference**: See `Production_pipeline_redesign_plan.md` Step 2 for full implementation.

---

### Gleaning Extraction Implementation

**Two-Stage Process** (when `enable_gleaning=True`):

#### STAGE 1: Initial Extraction with Retry (Error Recovery)

```python
# Attempt extraction up to 3 times
for attempt in range(3):
    result = await extract_once(chunk)
    validation = validate(result)

    if validation['status'] in ['PASS', 'WARNING']:
        break  # Proceed to Stage 2
    elif attempt == 2:
        # All 3 attempts failed
        if hitl_store:
            hitl_store.save_failed_chunk(chunk, validation)
        return None  # Skip this chunk
```

**Key Points**:
- Retries use **SAME prompt** (stateless error recovery)
- If any attempt passes → proceed to gleaning
- If all fail → save to HITL and skip

#### STAGE 2: Gleaning (Refinement, Only if Stage 1 Succeeded)

```python
# Only runs if initial extraction passed
merged_extraction = initial_result
conversation_history = [initial_prompt, initial_response]

for gleaning_pass in range(max_gleaning_iterations):
    # Create continue-extraction prompt
    continue_prompt = "CONTINUE EXTRACTION: Find any additional entities..."
    conversation_history.append({"role": "user", "content": continue_prompt})

    # Call LLM with conversation history
    glean_result = await llm.complete(conversation_history)
    conversation_history.append({"role": "assistant", "content": glean_result})

    # Validate gleaned extraction
    validation = validate(glean_result)

    if validation['status'] in ['PASS', 'WARNING']:
        # Merge using quality-based comparison
        merged_extraction = merge_by_quality(merged_extraction, glean_result)
```

#### Critical Clarifications

1. **Gleaning is REFINEMENT, not error recovery**
   - Only runs after successful initial extraction
   - Uses conversation history (NOT stateless retries)

2. **Quality Tiebreaker Hierarchy**
   - Primary: Quality score (use `description_quality_score()`)
   - Secondary: Description length
   - Tertiary: First-seen (stable sort)

3. **key_score Aggregation**
   ```python
   # IMPORTANT: SUM across passes, don't average
   merged_entity['key_score'] = base_score + glean_score  # e.g., 60 + 70 = 130
   ```

4. **Context Window Management**
   - Skip examples in gleaning prompts (~400 tokens saved)
   - Reserve 2000 tokens for response
   - Total context budget: ~4096 tokens

**Reference**: See `Production_pipeline_redesign_plan.md` Step 3 for full implementation.

---

## 📋 Implementation Decisions & Developer FAQ

**🚨 IMPORTANT**: This is a **greenfield development** - focus on ideal implementation, NOT backward compatibility.

### Decision 1: Code Reuse Strategy

**Question**: Should we copy code from `operate.py` or import it?

**ANSWER**: **IMPORT, NEVER COPY** ✅

**Reasoning**:
- `operate.py` has battle-tested functions (`chunking_by_token_size`, `extract_entities`, etc.)
- Copying = maintenance nightmare (bugs fixed in 2 places)
- Importing = single source of truth

**Example**:
```python
# ✅ CORRECT - Import and wrap
from bigrag.operate import chunking_by_token_size as _token_chunk_impl

class TokenChunker:
    def chunk(self, text, size, overlap):
        return _token_chunk_impl(text, size, overlap)

# ❌ WRONG - Copy-paste code
class TokenChunker:
    def chunk(self, text, size, overlap):
        # 200 lines of copied code from operate.py
```

**Action**: Always import from existing modules. Create thin wrappers if needed for interface consistency.

---

### Decision 2: Existing Component Reuse

**Question**: Which existing components should we reuse?

**ANSWER**: Reuse **ALL** existing well-tested components ✅

**Reuse Table**:

| Component | File | Status | Action |
|-----------|------|--------|--------|
| Token chunking | `operate.py::chunking_by_token_size` | ✅ Production | Import |
| Entity extraction | `operate.py::extract_entities` | ✅ Production | Import |
| **Table extraction** | `preprocessors/table_extractor.py` | ✅ Tested | Import |
| **Smart chunking** | `preprocessors/smart_chunker.py` (627 lines) | ✅ Tested | Import |
| **Constrained extraction** | `extractors/constrained_extractor.py` | ✅ Has gleaning | Import |
| **Table fact extraction** | `extractors/table_fact_extractor.py` | ✅ Tested | Import |
| **Numeric validation** | `validators/numeric_validator.py` | ✅ Tested | Import |
| **Entity merging** | `merging/entity_linker.py` | ✅ Multiple strategies | Import |
| **HITL system** | `hitl/failed_extraction_store.py` (460 lines) | ✅ Complete | Import |

**What's NEW** (needs implementation):
- `bigrag/pipeline/features.py` - PipelineFeatures dataclass + VALIDATION_THRESHOLDS
- `bigrag/pipeline/base_pipeline.py` - UnifiedPipeline orchestrator
- `bigrag/utils/quality_scoring.py` - description_quality_score() function

**That's it!** Only 3 new files. Everything else is imported.

---

### Decision 3: Directory Structure

**Question**: Follow exact structure in plan or adapt?

**ANSWER**: **Minimal new directories - leverage existing structure** ✅

**Final Structure**:
```
bigrag/
├── pipeline/                    # NEW - Only 2 files
│   ├── __init__.py
│   ├── features.py              # PipelineFeatures + VALIDATION_THRESHOLDS
│   └── base_pipeline.py         # UnifiedPipeline class
│
├── preprocessors/               # EXISTING - Import from here
│   ├── table_extractor.py       # Has: GPT4TableExtractor
│   └── smart_chunker.py         # Has: TableAwareChunker
│
├── extractors/                  # EXISTING - Import from here
│   ├── constrained_extractor.py # Has: ConstrainedLLMExtractor (with gleaning!)
│   └── table_fact_extractor.py  # Has: TableFactExtractor
│
├── validators/                  # EXISTING - Import from here
│   └── numeric_validator.py     # Has: NumericValidator
│
├── merging/                     # EXISTING - Import from here
│   ├── entity_linker.py         # Has: SimpleEntityLinker, ProductionEntityLinker
│   └── canonicalization.py      # Has: EntityCanonicalizationMap
│
├── hitl/                        # EXISTING - Import from here
│   └── failed_extraction_store.py  # Has: FailedExtractionStore (complete!)
│
├── utils/                       # NEW - Add 1 file
│   └── quality_scoring.py       # NEW: description_quality_score() function
│
├── operate.py                   # EXISTING - Import core functions
│
└── [OLD PIPELINES - IGNORE]     # For emergency rollback only
    ├── production_pipeline.py   # Don't touch, don't import
    ├── enhanced_pipeline.py     # Don't touch, don't import
    └── educational_pipeline.py  # Don't touch, don't import
```

**Why this structure**:
- ✅ Only 3 new files (minimal code to write)
- ✅ Maximum reuse of tested code
- ✅ Clear separation: `pipeline/` = orchestration, everything else = execution
- ✅ No complex migration or compatibility layers

---

### Decision 4: Gleaning Implementation

**Question**: Is gleaning already implemented?

**ANSWER**: **YES - Already perfect in ConstrainedLLMExtractor** ✅

**Current Implementation** (`extractors/constrained_extractor.py` lines 65-80):
- ✅ Stage 1: Retry with validation (up to 3 attempts)
- ✅ Stage 2: Gleaning refinement (if `enable_gleaning=True`)
- ✅ Quality-based merging

**What You Need to Do**:
1. Import `ConstrainedLLMExtractor`
2. Pass `enable_gleaning` parameter from `PipelineFeatures`
3. **That's it!**

**Example**:
```python
# In UnifiedPipeline.__init__
if self.features.enable_gleaning:
    self.extractor = ConstrainedLLMExtractor(
        api_key=self.features.openai_api_key,
        enable_gleaning=True,
        max_gleaning_iterations=self.features.max_gleaning_iterations,
        hitl_store=self.hitl_store
    )
```

**What's NEW**: Quality scoring function (`description_quality_score()`) - implement in `utils/quality_scoring.py`.

---

### Decision 5: Semantic Chunking

**Question**: Does TableAwareChunker already have semantic chunking?

**ANSWER**: **Check and verify, likely YES** ⚠️

**Action Steps**:
1. Read `bigrag/preprocessors/smart_chunker.py` (627 lines)
2. Check for:
   - ✅ Table-aware chunking (likely present)
   - ❓ Asymmetric overlap (first/middle/last chunks)
   - ❓ Bengali sentence detection (।!?)
   - ❓ 30% overflow tolerance
3. If missing features: Add them to `smart_chunker.py` directly
4. Import in `UnifiedPipeline`

**Usage**:
```python
# In UnifiedPipeline.__init__
if self.features.enable_table_detection:
    from bigrag.preprocessors.table_extractor import GPT4TableExtractor
    from bigrag.preprocessors.smart_chunker import TableAwareChunker

    self.table_extractor = GPT4TableExtractor(api_key=self.features.openai_api_key)
    self.chunker = TableAwareChunker(self.table_extractor)
else:
    # Use token-based chunking from operate.py
    from bigrag.operate import chunking_by_token_size
    self.chunker = lambda text: chunking_by_token_size(text, 1200, 100)
```

---

### Decision 6: HITL Integration

**Question**: Do we need to create HITL system?

**ANSWER**: **NO - Already exists and is perfect** ✅

**Existing Implementation**:
- File: `bigrag/hitl/failed_extraction_store.py` (460 lines)
- Has: `save_failed_chunk()`, `save_failed_table()`, `get_failed_extractions()`
- Matches plan exactly

**What You Need**:
1. ✅ Import `FailedExtractionStore`
2. ✅ Pass to extractors
3. ❓ Check if API endpoints exist (`backend/api/hitl_routes.py`)
   - If missing: Create endpoints from plan (lines 1246-1277)
   - If exists: Verify completeness

**Usage**:
```python
# In UnifiedPipeline.__init__
if self.features.enable_hitl:
    from bigrag.hitl.failed_extraction_store import FailedExtractionStore
    self.hitl_store = FailedExtractionStore(dataset_path)
else:
    self.hitl_store = None

# Pass to extractor
self.extractor = ConstrainedLLMExtractor(
    ...,
    hitl_store=self.hitl_store
)
```

---

### Decision 7: Error Handling Location

**Question**: Where should error handling live?

**ANSWER**: **Both levels - defense in depth** ✅

**High-Level (UnifiedPipeline)**:
```python
async def process_document(self, text, metadata):
    # Feature-level fallback
    try:
        if self.features.enable_table_detection:
            chunks = await self.semantic_chunker.chunk(text, metadata)
    except TableExtractionError as e:
        logger.warning(f"Table extraction failed: {e}. Falling back to token chunking.")
        chunks = await self.token_chunker.chunk(text)
```

**Low-Level (Individual Modules)**:
```python
# In table_extractor.py (already implemented)
async def extract_table(self, markdown):
    try:
        return await self.gpt4_call(markdown)
    except APIError:
        return None  # Let pipeline handle fallback
```

**Strategy**: Graceful degradation (not fail-fast). If optional feature fails, use simpler alternative.

---

### Decision 8: Testing Approach

**Question**: How to test?

**ANSWER**: **Test-driven with existing test data** ✅

**Test Data**:
- ✅ KUET admission info (40K chars, tables, bilingual) - `datasets/kuet_test/`
- ✅ BUET admission info (if available)
- ✅ SingleTopic dataset

**Success Criteria** (simplified - no backward compatibility needed):

**Phase 1 (Core Implementation)**:
- ✅ Standard preset completes KUET document without errors
- ✅ Quality preset completes KUET document without errors
- ✅ Balanced preset completes KUET document without errors
- ✅ All feature flags work (enable/disable doesn't crash)
- ✅ Results are reasonable (80-100 entities, 60-90 relations for KUET)

**Phase 2 (Feature Validation)**:
- ✅ Table extraction finds all tables in KUET doc
- ✅ Numeric validation catches hallucinations
- ✅ Entity merging reduces duplicates
- ✅ HITL captures failed extractions

**No need to compare with old pipelines** - just verify new pipeline works correctly!

---

### Decision 9: Implementation Priority

**Question**: What order to implement?

**ANSWER**: **Hybrid Testing Approach** ✅

**Testing Strategy**:
- **Weeks 1-3**: Smoke tests only (fast iteration, verify no crashes)
- **Week 4**: Comprehensive testing (all features, integration, performance, bug fixes)

**Week 1: Core Implementation** (Smoke tests only)
1. Create `bigrag/pipeline/features.py` (PipelineFeatures dataclass)
2. Create `bigrag/utils/quality_scoring.py` (description_quality_score)
3. Create `bigrag/pipeline/base_pipeline.py` (UnifiedPipeline class)
4. **Smoke test**: Can it instantiate? Does `from_preset()` work? No crashes?
5. Verify `smart_chunker.py` has all features (asymmetric overlap, Bengali, overflow)
6. If missing: Add features to `smart_chunker.py` directly

**Week 2: Feature Integration** (Smoke tests only)
7. Import and connect all existing modules
8. Implement `UnifiedPipeline.process_document()` orchestration
9. Add error handling (graceful degradation)
10. **Smoke test**: Does each preset instantiate? Can it process a simple doc? No crashes?

**Week 3: API Integration** (Smoke tests only)
11. Update `backend/server.py` - remove old parameters, add preset/features
12. Check if HITL endpoints exist, create if missing
13. **Smoke test**: Can API accept requests? Does it return something? No crashes?

**Week 4: Comprehensive Testing & Bug Fixes** (ALL testing happens here)
14. **Test all 3 presets** with KUET document (standard, quality, balanced)
15. **Test feature flags** - enable/disable each feature, verify no crashes
16. **Test error handling** - API failures, timeouts, missing API keys
17. **Test via API** - all endpoints, all parameter combinations
18. **Fix all bugs discovered** during testing
19. **Integration tests** - end-to-end document processing
20. **Performance benchmarks** - time, cost, accuracy for each preset
21. **Documentation updates** - CLAUDE.md, README.md with new usage examples
22. **Edge case testing** - empty docs, huge docs, special characters, Unicode

**Total**: 4 weeks to production-ready modular pipeline!

**Why Hybrid Approach?**
- Faster development in weeks 1-3 (no time spent on comprehensive tests)
- Catch critical crashes early (smoke tests)
- Fix all bugs together in week 4 (more efficient)
- Allows rapid iteration without test maintenance overhead

---

## ⚠️ Critical "Don't Do This" List

1. ❌ **DON'T copy code from `operate.py`** - Import functions, create thin wrappers
2. ❌ **DON'T reimplement existing modules** - `ConstrainedLLMExtractor`, `TableAwareChunker`, etc. are already perfect
3. ❌ **DON'T create backward compatibility layers** - We're in development, break things if needed
4. ❌ **DON'T touch old pipeline files** - They're for emergency rollback only
5. ❌ **DON'T create deep directory hierarchies** - Keep it flat (`pipeline/`, not `pipeline/chunkers/token/`)
6. ❌ **DON'T add migration scripts** - Users will rebuild graphs, no migration needed
7. ❌ **DON'T preserve old API endpoints** - Change them to match new design
8. ❌ **DON'T implement PipelineSelector execution** - It's recommendation-only, keep it that way
9. ❌ **DON'T skip quality_scoring.py** - It's small but critical for gleaning merge
10. ❌ **DON'T over-engineer** - 3 new files (features.py, base_pipeline.py, quality_scoring.py) + imports = done!

---

## ✅ Quick Start Checklist

**Before you start coding**:
- [ ] Read this entire plan
- [ ] Review existing components (1 hour):
  - [ ] `bigrag/operate.py` - Core functions
  - [ ] `bigrag/extractors/constrained_extractor.py` - Has gleaning!
  - [ ] `bigrag/preprocessors/smart_chunker.py` - Has semantic chunking?
  - [ ] `bigrag/hitl/failed_extraction_store.py` - HITL system
- [ ] Understand you're building from scratch (no migration)

**Week 1 Deliverables**:
- [ ] `bigrag/pipeline/features.py` (200 lines)
- [ ] `bigrag/pipeline/base_pipeline.py` (300 lines)
- [ ] `bigrag/utils/quality_scoring.py` (50 lines)
- [ ] Smoke tests pass (instantiation, no crashes)

**Week 2-3 Deliverables**:
- [ ] All modules integrated
- [ ] API endpoints updated
- [ ] Smoke tests pass (basic functionality, no crashes)

**Week 4 Deliverables**:
- [ ] Comprehensive tests pass (all presets, all features, edge cases)
- [ ] All bugs fixed
- [ ] Performance benchmarks complete
- [ ] Documentation updated

**You're ready to start! Focus on clean code, not compatibility.**

---

```

#### 1.2 Create Unified Pipeline Class

**File**: `bigrag/pipeline/base_pipeline.py`

```python
import asyncio
from typing import List, Dict, Tuple, Optional
from bigrag.utils import logger
from .features import PipelineFeatures

class UnifiedPipeline:
    """
    Unified modular knowledge graph pipeline.

    Combines standard and enhanced pipeline features with plug-and-play architecture.
    Based on proven standard pipeline (operate.py) with optional enhancements.

    Usage:
        features = PipelineFeatures.from_preset("standard")
        pipeline = UnifiedPipeline(features, api_key="your-key")
        result = await pipeline.process_document(content, metadata)

    Architecture:
        1. Chunking (required)
        2. Extraction (required)
        3. Validation (optional - based on features)
        4. Merging (optional - based on features)
        5. Post-processing (optional - based on features)
    """

    def __init__(
        self,
        features: PipelineFeatures,
        api_key: str,
        llm_model: str = "gpt-4o-mini",
        dataset_path: Optional[str] = None
    ):
        """
        Initialize unified pipeline with feature configuration.

        Args:
            features: Feature configuration (use PipelineFeatures.from_preset())
            api_key: OpenAI API key
            llm_model: LLM model for extraction
            dataset_path: Path for HITL storage (if enable_hitl=True)
        """
        self.features = features
        self.api_key = api_key
        self.llm_model = llm_model
        self.dataset_path = dataset_path

        logger.info(f"[Unified Pipeline] Initializing with preset: {self._detect_preset()}")
        logger.info(f"[Unified Pipeline] Features: {self._summarize_features()}")

        # Initialize components based on features
        self.chunker = self._init_chunker()
        self.extractor = self._init_extractor()
        self.validator = self._init_validator() if self._needs_validation() else None
        self.merger = self._init_merger()
        self.postprocessor = self._init_postprocessor() if self._needs_postprocessing() else None

    def _detect_preset(self) -> str:
        """Detect which preset was used (for logging)"""
        if self.features == PipelineFeatures.from_preset("standard"):
            return "STANDARD (fast, reliable)"
        elif self.features == PipelineFeatures.from_preset("quality"):
            return "QUALITY (slow, accurate)"
        elif self.features == PipelineFeatures.from_preset("balanced"):
            return "BALANCED (medium speed/quality)"
        else:
            return "CUSTOM"

    def _summarize_features(self) -> str:
        """Summarize enabled features (for logging)"""
        enabled = []
        if self.features.enable_table_detection:
            enabled.append("table_detection")
        if self.features.enable_gleaning:
            enabled.append("gleaning")
        if self.features.enable_numeric_validation:
            enabled.append("numeric_validation")
        if self.features.merge_strategy == "fuzzy":
            enabled.append("fuzzy_merging")
        if self.features.enable_hitl:
            enabled.append("hitl")

        return ", ".join(enabled) if enabled else "basic"

    def _init_chunker(self):
        """Initialize chunker based on features"""
        if self.features.enable_table_detection:
            from bigrag.pipeline.chunkers.table_chunker import TableChunker
            return TableChunker(
                api_key=self.api_key,
                chunk_mode=self.features.chunk_mode
            )
        else:
            from bigrag.pipeline.chunkers.token_chunker import TokenChunker
            return TokenChunker(
                chunk_size=self.features.chunk_size,
                overlap=self.features.chunk_overlap
            )

    def _init_extractor(self):
        """Initialize extractor based on features"""
        from bigrag.pipeline.extractors.llm_extractor import LLMExtractor
        return LLMExtractor(
            api_key=self.api_key,
            model=self.llm_model,
            enable_gleaning=self.features.enable_gleaning,
            max_iterations=self.features.max_gleaning_iterations,
            concurrency=self.features.extraction_concurrency,
            enable_table_facts=self.features.enable_table_fact_extraction,
            hitl_store=self._init_hitl() if self.features.enable_hitl else None
        )

    def _init_validator(self):
        """Initialize validators based on features"""
        if not self._needs_validation():
            return None

        from bigrag.pipeline.validators.entity_validator import EntityValidator
        return EntityValidator(
            enable_numeric=self.features.enable_numeric_validation,
            enable_entity_quality=self.features.enable_entity_validation,
            enable_relation_quality=self.features.enable_relation_validation,
            strictness=self.features.validation_strictness
        )

    def _init_merger(self):
        """Initialize merger based on features"""
        if self.features.merge_strategy == "fuzzy":
            from bigrag.pipeline.mergers.fuzzy_merger import FuzzyMerger
            return FuzzyMerger()
        else:
            from bigrag.pipeline.mergers.basic_merger import BasicMerger
            return BasicMerger()

    def _init_postprocessor(self):
        """Initialize post-processor if needed"""
        if not self._needs_postprocessing():
            return None

        from bigrag.pipeline.postprocessors.orphan_linker import OrphanLinker
        return OrphanLinker() if self.features.enable_orphan_linking else None

    def _init_hitl(self):
        """Initialize HITL store if enabled"""
        if not self.features.enable_hitl or not self.dataset_path:
            return None

        from bigrag.hitl.failed_extraction_store import FailedExtractionStore
        return FailedExtractionStore(self.dataset_path)

    def _needs_validation(self) -> bool:
        """Check if any validation is enabled"""
        return (
            self.features.enable_numeric_validation or
            self.features.enable_entity_validation or
            self.features.enable_relation_validation
        )

    def _needs_postprocessing(self) -> bool:
        """Check if post-processing is needed"""
        return self.features.enable_orphan_linking or self.features.enable_quality_scoring

    async def process_document(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process document through modular pipeline.

        Args:
            content: Document text
            metadata: Optional metadata (title, category, tags, etc.)

        Returns:
            dict: {
                'entities': List[Dict],
                'relations': List[Dict],
                'chunks': List[Dict],
                'validation': Dict,
                'statistics': Dict,
                'pipeline_metadata': Dict
            }
        """
        metadata = metadata or {}

        # Step 1: Chunking (always required)
        logger.info("[Pipeline] Step 1: Chunking...")
        chunks = await self.chunker.chunk(content, metadata)
        logger.info(f"[Pipeline] Created {len(chunks)} chunks")

        # Step 2: Extraction (always required)
        logger.info("[Pipeline] Step 2: Extraction...")
        entities, relations = await self.extractor.extract(chunks, metadata)
        logger.info(f"[Pipeline] Extracted {len(entities)} entities, {len(relations)} relations")

        # Step 3: Validation (optional)
        if self.validator:
            logger.info("[Pipeline] Step 3: Validation...")
            entities, relations, validation_report = await self.validator.validate(
                entities, relations, chunks
            )
            logger.info(f"[Pipeline] Validation: {validation_report['status']}")
        else:
            validation_report = {'status': 'SKIPPED', 'message': 'Validation disabled'}

        # Step 4: Merging (optional but recommended)
        if self.features.enable_entity_merging:
            logger.info(f"[Pipeline] Step 4: Merging (strategy: {self.features.merge_strategy})...")
            entities = await self.merger.merge(entities, relations)
            logger.info(f"[Pipeline] Merged to {len(entities)} unique entities")

        # Step 5: Post-processing (optional)
        if self.postprocessor:
            logger.info("[Pipeline] Step 5: Post-processing...")
            entities, relations = await self.postprocessor.process(entities, relations)

        # Build result
        return {
            'entities': entities,
            'relations': relations,
            'chunks': chunks,
            'validation': validation_report,
            'statistics': {
                'total_entities': len(entities),
                'total_relations': len(relations),
                'total_chunks': len(chunks)
            },
            'pipeline_metadata': {
                'version': self.features.pipeline_version,
                'preset': self._detect_preset(),
                'features_enabled': self._summarize_features()
            }
        }
```

### Phase 2: Extract Standard Pipeline Components (Week 2)

#### 2.1 Token Chunker Module

**File**: `bigrag/pipeline/chunkers/token_chunker.py`

```python
"""
Token-based chunking module (from standard pipeline).

Simple, fast, reliable chunking using token counts.
"""

from typing import List, Dict
from bigrag.utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken, compute_mdhash_id

class TokenChunker:
    """
    Token-based text chunker.

    Based on standard pipeline chunking_by_token_size() function.
    Splits text into overlapping chunks of fixed token size.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 100,
        tiktoken_model: str = "gpt-4o-mini"
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tiktoken_model = tiktoken_model

    async def chunk(
        self,
        content: str,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Chunk text into fixed-size token chunks with overlap.

        Args:
            content: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dicts with format:
            {
                'chunk_id': str,
                'content': str,
                'tokens': int,
                'chunk_order_index': int,
                'metadata': Dict
            }
        """
        metadata = metadata or {}

        # Tokenize content
        tokens = encode_string_by_tiktoken(content, model_name=self.tiktoken_model)

        # Create overlapping chunks
        chunks = []
        for index, start in enumerate(
            range(0, len(tokens), self.chunk_size - self.overlap)
        ):
            # Extract token slice
            chunk_tokens = tokens[start : start + self.chunk_size]

            # Decode back to text
            chunk_content = decode_tokens_by_tiktoken(
                chunk_tokens,
                model_name=self.tiktoken_model
            ).strip()

            # Create chunk ID
            chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")

            # Build chunk dict
            chunk = {
                'chunk_id': chunk_id,
                'content': chunk_content,
                'tokens': len(chunk_tokens),
                'chunk_order_index': index,
                'metadata': {
                    **metadata,
                    'chunking_method': 'token',
                    'chunk_size': self.chunk_size,
                    'overlap': self.overlap
                }
            }

            chunks.append(chunk)

        return chunks
```

*(Continue implementation in next sections...)*

---

## 🛡️ Error Handling Strategy

### Graceful Degradation Philosophy

**Core Principle**: If an optional feature fails, fall back to simpler alternative instead of failing the entire pipeline.

### Feature-Specific Error Handling

#### 1. **Table Extraction Failure**

**Scenario**: GPT-4 table extraction times out or returns invalid JSON

**Handling**:
```python
try:
    # Attempt table extraction
    tables = await self.table_extractor.extract(content)
except (TimeoutError, JSONDecodeError, APIError) as e:
    logger.warning(f"Table extraction failed: {e}. Falling back to token chunking.")
    # Graceful degradation: Treat entire content as plain text
    tables = []
    # Continue with token-based chunking
```

**User Impact**: Pipeline continues, but table structure is lost (acceptable for most use cases)

#### 2. **Numeric Validation Failure**

**Scenario**: Gemini API unavailable or quota exceeded

**Handling**:
```python
if self.features.enable_numeric_validation:
    try:
        validation_result = await self.numeric_validator.validate_llm(entities, source)
    except (APIError, QuotaExceededError) as e:
        logger.warning(f"LLM validation unavailable: {e}. Using regex fallback.")
        validation_result = self.numeric_validator.validate_regex(entities, source)
```

**User Impact**: Lower validation accuracy (regex-based), but pipeline completes

#### 3. **Entity Merging Failure**

**Scenario**: Fuzzy merging crashes due to memory or logic error

**Handling**:
```python
if self.features.merge_strategy == "fuzzy":
    try:
        merged_entities = await self.fuzzy_merger.merge(entities)
    except (MemoryError, RuntimeError) as e:
        logger.error(f"Fuzzy merging failed: {e}. Falling back to basic merging.")
        # Fallback to basic (hash-based) merging
        merged_entities = await self.basic_merger.merge(entities)
```

**User Impact**: More duplicate entities, but pipeline completes

#### 4. **Extraction Timeout**

**Scenario**: LLM extraction hangs or takes too long

**Handling**:
```python
# Set per-chunk timeout
async with asyncio.timeout(self.extraction_timeout):
    try:
        entities, relations = await self.extractor.extract(chunk)
    except asyncio.TimeoutError:
        logger.warning(f"Chunk {chunk_id} extraction timed out. Skipping.")
        # Add to failed extraction store if HITL enabled
        if self.hitl_store:
            await self.hitl_store.save_failed(chunk, reason="timeout")
        entities, relations = [], []
```

**User Impact**: Some chunks skipped, but most data extracted

#### 5. **API Key Missing**

**Scenario**: Feature requires API key but none provided

**Handling**:
```python
def _init_table_extractor(self):
    if not self.features.openai_api_key:
        # Check environment variable as fallback
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("Table extraction requires OPENAI_API_KEY. Feature disabled.")
            return None
    # Initialize extractor
    return GPT4TableExtractor(api_key=api_key)
```

**User Impact**: Feature silently disabled with warning

### Error Reporting Levels

**CRITICAL** (Pipeline fails):
- Invalid document format (empty, corrupted)
- All chunks fail extraction
- Storage backend unavailable

**ERROR** (Feature fails, pipeline continues):
- Table extraction fails → fallback to token chunking
- Fuzzy merging fails → fallback to basic merging
- Validation API unavailable → skip validation or use fallback

**WARNING** (Degraded performance):
- Single chunk extraction times out → skip chunk
- Numeric validation uses regex fallback → lower accuracy
- API key missing → feature disabled

**INFO** (Expected behavior):
- Gleaning iteration produces no new entities → stop early
- Orphan linking finds no orphans → skip processing

### Retry Strategy

**Retry with Backoff** (for transient API errors):
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError))
)
async def extract_with_retry(chunk):
    return await llm_extract(chunk)
```

**Don't Retry** (for permanent errors):
- Invalid API key → fail immediately
- Malformed response → skip and log
- Timeout → skip chunk (user may have limited time budget)

### Failed Extraction Queue (HITL)

When `enable_hitl=True`, failed extractions are saved for human review:

```python
# Structure of failed extraction record
{
    "chunk_id": "chunk-abc123",
    "content": "Original chunk text...",
    "failure_reason": "timeout" | "api_error" | "validation_failed",
    "error_message": "Extraction timed out after 60s",
    "timestamp": "2024-11-26T17:43:00Z",
    "retry_count": 2,
    "feature": "table_extraction"
}
```

**Review Workflow**:
1. Pipeline saves failed chunks to `{dataset}/failed_extractions/queue.json`
2. User reviews queue and fixes manually or adjusts configuration
3. Re-run pipeline with `retry_failed=True` flag

### Validation Error Summary

At end of pipeline, log validation summary:

```python
logger.info("=" * 80)
logger.info("PIPELINE VALIDATION SUMMARY")
logger.info("=" * 80)
logger.info(f"Total entities extracted: {total_entities}")
logger.info(f"  - PASS: {pass_count} ({pass_rate:.1%})")
logger.info(f"  - FAIL: {fail_count} ({fail_rate:.1%})")
logger.info(f"  - SKIPPED: {skip_count} (feature disabled)")
logger.info(f"Total relations extracted: {total_relations}")
logger.info(f"  - PASS: {rel_pass_count} ({rel_pass_rate:.1%})")
logger.info(f"  - FAIL: {rel_fail_count} ({rel_fail_rate:.1%})")
logger.info(f"Failed chunks: {failed_chunks} (saved to HITL queue)")
logger.info(f"Validation strictness: {self.features.validation_strictness}")
logger.info("=" * 80)
```

---

## 🔄 Human-in-the-Loop (HITL) System

### Overview

When `enable_hitl=True`, the pipeline captures and stores failed extractions for human review and later reprocessing.

**Problem Solved**:
- Failed chunks are no longer lost (previously only logged)
- Humans can review and correct extraction failures
- Pipeline continues gracefully while tracking what failed

### Storage Structure

```
expr/{dataset}/failed_extractions/
├── failed_chunks.json       # Paragraph extraction failures
├── failed_tables.json       # Table extraction failures
└── review_queue.json        # Pending human review
```

### Implementation

**File**: `bigrag/hitl/failed_extraction_store.py`

```python
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class FailedExtractionStore:
    """
    Store failed extractions for human review and later reprocessing.

    Usage:
        store = FailedExtractionStore("expr/my_dataset")
        store.save_failed_chunk(chunk_id, content, reason, validation, doc_id)
    """

    def __init__(self, dataset_path: str):
        self.base_path = Path(dataset_path) / "failed_extractions"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.chunks_file = self.base_path / "failed_chunks.json"
        self.tables_file = self.base_path / "failed_tables.json"
        self.queue_file = self.base_path / "review_queue.json"

    def save_failed_chunk(
        self,
        chunk_id: str,
        chunk_content: str,
        failure_reason: str,
        validation_details: Dict,
        document_id: str,
        metadata: Optional[Dict] = None
    ):
        """Save failed chunk extraction for human review."""

        failure_record = {
            "extraction_id": f"chunk_{chunk_id}_{datetime.now().timestamp()}",
            "type": "chunk",
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": chunk_content,
            "failure_reason": failure_reason,
            "validation_details": validation_details,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review"
        }

        self._append_to_file(self.chunks_file, failure_record)
        self._add_to_review_queue(failure_record)

    def save_failed_table(
        self,
        table_id: str,
        table_data: Dict,
        failure_reason: str,
        document_id: str
    ):
        """Save failed table extraction for human review."""

        failure_record = {
            "extraction_id": f"table_{table_id}_{datetime.now().timestamp()}",
            "type": "table",
            "table_id": table_id,
            "document_id": document_id,
            "table_data": table_data,
            "failure_reason": failure_reason,
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review"
        }

        self._append_to_file(self.tables_file, failure_record)
        self._add_to_review_queue(failure_record)

    def get_failed_extractions(self, document_id: Optional[str] = None) -> List[Dict]:
        """Retrieve all failed extractions (optionally filtered by document)."""

        all_failures = []

        # Load chunks
        if self.chunks_file.exists():
            with open(self.chunks_file) as f:
                all_failures.extend(json.load(f))

        # Load tables
        if self.tables_file.exists():
            with open(self.tables_file) as f:
                all_failures.extend(json.load(f))

        # Filter by document if specified
        if document_id:
            all_failures = [f for f in all_failures if f["document_id"] == document_id]

        return all_failures

    def mark_reviewed(self, extraction_id: str, corrected_data: Dict):
        """Mark extraction as human-reviewed with corrections."""
        # Update status in review_queue.json
        pass

    def _append_to_file(self, file_path: Path, record: Dict):
        """Append record to JSON file."""
        records = []
        if file_path.exists():
            with open(file_path) as f:
                records = json.load(f)

        records.append(record)

        with open(file_path, 'w') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def _add_to_review_queue(self, record: Dict):
        """Add to review queue."""
        self._append_to_file(self.queue_file, record)
```

### Integration Example

**In extractor** (when extraction fails):

```python
# bigrag/extractors/llm_extractor.py

async def extract(self, chunk):
    result = await self._attempt_extraction(chunk)

    if result is None and self.hitl_store:
        # Save failed chunk for human review
        await self.hitl_store.save_failed_chunk(
            chunk_id=chunk['chunk_id'],
            chunk_content=chunk['content'],
            failure_reason="All validation attempts failed",
            validation_details=last_validation_result,
            document_id=chunk.get('metadata', {}).get('doc_id'),
            metadata=chunk.get('metadata')
        )
        logger.warning(f"Chunk {chunk['chunk_id']} saved to HITL queue")

    return result  # Continue with other chunks
```

### API Endpoints

**File**: `backend/api/hitl_routes.py`

```python
from fastapi import APIRouter
from bigrag.hitl.failed_extraction_store import FailedExtractionStore

router = APIRouter(prefix="/hitl", tags=["Human-in-the-Loop"])

@router.get("/failed-extractions/{dataset_name}")
async def get_failed_extractions(dataset_name: str, document_id: Optional[str] = None):
    """Get failed extractions for human review."""
    store = FailedExtractionStore(f"expr/{dataset_name}")
    failures = store.get_failed_extractions(document_id)

    return {
        "dataset": dataset_name,
        "total_failures": len(failures),
        "failures": failures
    }

@router.post("/correct-extraction/{extraction_id}")
async def submit_correction(extraction_id: str, corrected_data: Dict):
    """Submit human-corrected extraction."""
    return {"status": "correction_saved", "extraction_id": extraction_id}

@router.post("/reprocess/{extraction_id}")
async def reprocess_extraction(extraction_id: str):
    """Reprocess corrected extraction into graph."""
    return {"status": "reprocessed", "extraction_id": extraction_id}
```

### Benefits

- **No data loss**: All failures captured for later review
- **Human oversight**: Domain experts can correct extraction errors
- **Quality improvement**: Failed cases inform prompt engineering
- **Debugging aid**: Understand why certain chunks fail validation

---

## 🎯 Success Criteria

### Must Have (MVP)

- [x] `PipelineFeatures` class with 3 presets
- [x] `UnifiedPipeline` class that routes to correct modules
- [x] `TokenChunker` extracted from standard pipeline
- [x] `LLMExtractor` extracted from standard pipeline
- [x] `BasicMerger` extracted from standard pipeline
- [x] Integration with `BiGRAG` class
- [x] Backward compatibility (old code works)
- [x] Documentation and examples

### Should Have (V1.1)

- [ ] All validators integrated
- [ ] Table chunker module
- [ ] Fuzzy merger module
- [ ] Orphan linker module
- [ ] API endpoint feature flags
- [ ] Migration guide

### Nice to Have (V2.0)

- [ ] Semantic chunker
- [ ] Hybrid merger (adaptive)
- [ ] Quality scorer
- [ ] Performance benchmarks
- [ ] Feature recommendation engine

---

## 📝 Implementation Notes

### Keep All Existing Features

From **Enhanced Pipeline**:
- ✅ Table extraction (GPT4TableExtractor)
- ✅ Smart chunking (TableAwareChunker)
- ✅ Numeric validation (NumericValidator)
- ✅ Entity validation (quality scoring)
- ✅ Fuzzy merging (UnifiedEntityMerger)
- ✅ HITL system (FailedExtractionStore)
- ✅ Orphan linking (post-merge)

From **Standard Pipeline**:
- ✅ Token chunking (chunking_by_token_size)
- ✅ Gleaning extraction (extract_entities)
- ✅ Basic merging (_merge_nodes_then_upsert)
- ✅ LLM concurrency control (semaphore)

### Don't Duplicate Code

**Rule**: If code exists, reference it. Don't copy.

**Example**:
```python
# ❌ BAD: Copy function from operate.py
def chunk_by_tokens(content):
    # ... copied 100 lines of code ...

# ✅ GOOD: Import and wrap
from bigrag.operate import chunking_by_token_size

class TokenChunker:
    async def chunk(self, content):
        return chunking_by_token_size(content, ...)
```

### Backward Compatibility Strategy

**Phase 1 (Migration Period - 3 months)**:
- Keep all 3 pipelines working
- Add deprecation warnings
- Auto-migrate configurations

**Phase 2 (Deprecation - After 3 months)**:
- Mark `use_enhanced_pipeline` as deprecated
- Keep `operate.py` (many external dependencies)
- Remove `enhanced_pipeline.py`

**Phase 3 (Cleanup - After 6 months)**:
- Archive old pipeline files
- Keep only unified pipeline

---

## 📚 Usage Examples

### Example 1: BUET Document (Fast Standard)

```python
from bigrag import BiGRAG
from bigrag.pipeline.features import PipelineFeatures

# Use standard preset (no validation)
rag = BiGRAG(
    working_dir="./expr/buet_unified",
    pipeline_features=PipelineFeatures.from_preset("standard")
)

# Process document
await rag.ainsert(buet_content, metadata={"title": "BUET Info"})

# Result: ~80-90 entities, ~70-80 relations (fast, no failures)
```

### Example 2: KUET Document (Quality Pipeline)

```python
# Use quality preset (all features)
rag = BiGRAG(
    working_dir="./expr/kuet_unified",
    pipeline_features=PipelineFeatures.from_preset("quality")
)

await rag.ainsert(kuet_content, metadata={"title": "KUET Admission"})

# Result: High accuracy with table extraction and validation
```

### Example 3: Custom Configuration

```python
# Custom: Tables but no validation
features = PipelineFeatures(
    enable_table_detection=True,
    enable_table_fact_extraction=True,
    enable_numeric_validation=False,  # Skip strict validation
    enable_entity_validation=False,
    merge_strategy="fuzzy"
)

rag = BiGRAG(
    working_dir="./expr/custom",
    pipeline_features=features
)
```

### Example 4: API Endpoint

```bash
# Use preset
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@document.md" \
  -F "data_source=my_dataset" \
  -F "pipeline_preset=standard"

# Custom features
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@document.md" \
  -F "data_source=my_dataset" \
  -F "enable_table_extraction=true" \
  -F "enable_validation=false" \
  -F "merge_strategy=fuzzy"
```

---

## 🚀 Migration Timeline

### Week 1: Core Infrastructure
- Create `bigrag/pipeline/` directory structure
- Implement `PipelineFeatures` class
- Implement `UnifiedPipeline` class (stub)
- Write comprehensive tests

### Week 2: Extract Standard Components
- Extract `TokenChunker` from `operate.py`
- Extract `LLMExtractor` from `operate.py`
- Extract `BasicMerger` from `operate.py`
- Connect to `UnifiedPipeline`

### Week 3: Integrate Enhanced Features
- Create wrapper for `TableChunker`
- Create wrapper for `FuzzyMerger`
- Create wrapper for validators
- Connect all to `UnifiedPipeline`

### Week 4: Integration & Testing
- Update `BiGRAG.__init__()` with feature support
- Update API endpoints with feature flags
- Write migration guide
- Test with BUET/KUET documents
- Document performance benchmarks

---

## 📊 Expected Improvements

### BUET Document (43K chars)

**Before** (Enhanced pipeline):
- Entities: 29
- Relations: 11
- Failures: 70%
- Time: 7 minutes

**After** (Standard preset):
- Entities: ~85
- Relations: ~75
- Failures: 5-10%
- Time: 30-60 seconds

### KUET Document (43K chars)

**Before** (Enhanced pipeline):
- Entities: 89
- Relations: 82
- Failures: 30%
- Time: 5 minutes

**After** (Quality preset):
- Entities: ~90
- Relations: ~85
- Failures: 10-15%
- Time: 2-3 minutes

---

## ❓ Open Questions

1. **Default Preset**: Should API default be "standard" or "balanced"?

2. **Feature Discovery**: How should users know which features to use?

3. **Performance Benchmarks**: Should we auto-recommend preset based on doc size?

4. **Validation Strictness**: Should LENIENT be default instead of MODERATE?

5. **Breaking Changes**: Is 3-month migration period sufficient?

---

## 📖 Next Steps

1. **Review this plan** - Provide feedback on architecture
2. **Approve presets** - Confirm "standard", "quality", "balanced" naming
3. **Begin implementation** - Start with Phase 1 (core infrastructure)
4. **Test on BUET** - Validate standard preset solves the problem
5. **Document features** - Create user guide for feature selection

---

**Document Version**: 1.0
**Last Updated**: November 26, 2024
**Status**: Awaiting Review
**Estimated Effort**: 4 weeks (1 developer)
