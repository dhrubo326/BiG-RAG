# Modular Unified Pipeline - Implementation Plan

**Version**: 1.0
**Date**: November 26, 2024
**Status**: Planning Phase

---

## 🎯 Executive Summary

Transform BiG-RAG from 3 separate pipelines into **ONE modular unified pipeline** that:
- ✅ Keeps proven **standard pipeline** (`operate.py`) as the reliable foundation
- ✅ Preserves **all enhanced features** (table extraction, validation, smart chunking)
- ✅ Allows **mix-and-match features** via simple configuration flags
- ✅ Maintains **100% backward compatibility** with existing code

**Core Philosophy**: "Use what you need, skip what you don't"

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
