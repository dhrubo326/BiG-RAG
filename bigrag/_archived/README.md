# Archived Pipeline Code

This directory contains the **OLD pipeline system** that has been superseded by the new **modular indexing system** as of January 2025.

---

## 📁 Archived Files

| File | Purpose | Replacement |
|------|---------|-------------|
| **enhanced_pipeline.py** | Old enhanced knowledge graph pipeline with all features | `BiGRAG` class with `IndexingConfig` |
| **production_pipeline.py** | Old production pipeline with table extraction | `IndexingConfig.preset_quality()` |
| **educational_pipeline.py** | Old educational pipeline variant | `IndexingConfig` with custom settings |
| **pipeline_selector.py** | Old pipeline selection logic | `IndexingConfig` presets (fast/balanced/quality) |

---

## ⚠️ Why Were These Archived?

### Problems with Old System

1. **Code Duplication**: 60% duplication across 4 pipeline files (~2750 duplicate lines)
2. **Tight Coupling**: Pipelines directly instantiated dependencies (table extractors, validators, etc.)
3. **Limited Flexibility**: Binary choice (standard vs. production) - no granular control
4. **Hard to Test**: Monolithic classes made unit testing difficult
5. **Hard to Extend**: Adding new features required modifying multiple files

### Benefits of New System

1. **Zero Duplication**: Strategy pattern eliminates code duplication
2. **Dependency Injection**: Strategies receive dependencies via constructor
3. **Granular Control**: 15+ feature flags vs. 2 presets
4. **Easy Testing**: Each strategy can be tested independently
5. **Easy Extension**: Add new strategies without modifying existing code

---

## 🔄 Migration Guide

### Old Way (Archived)

```python
from bigrag.pipeline.features import PipelineFeatures
from bigrag.enhanced_pipeline import EnhancedKGPipeline

features = PipelineFeatures(
    enable_table_detection=True,
    enable_gleaning=True,
    enable_numeric_validation=True,
    enable_entity_validation=True,
    validation_strictness="MODERATE"
)

pipeline = EnhancedKGPipeline(features, working_dir="./expr/my_dataset")
result = await pipeline.process_document(text, metadata)
```

### New Way (Modular)

```python
from bigrag import BiGRAG
from bigrag.config import IndexingConfig

# Option 1: Use preset
config = IndexingConfig.preset_quality(
    openai_api_key="your-key",
    gemini_api_key="your-key",
    dataset_path="./expr/my_dataset"
)

# Option 2: Custom configuration
config = IndexingConfig(
    chunker="semantic",        # table-aware chunking
    extractor="gleaning",      # multi-pass extraction
    validators=["numeric", "semantic"],  # both validators
    merger="fuzzy",            # entity deduplication
    hitl="file",              # save failures to file
    orphan_linker="synthetic", # link orphan entities
    validation_strictness="MODERATE",
    openai_api_key="your-key",
    gemini_api_key="your-key",
    dataset_path="./expr/my_dataset"
)

# Use BiGRAG with modular system
rag = BiGRAG(
    indexing_config=config,
    working_dir="./expr/my_dataset"
)

result = await rag.index_document(text, metadata)
```

---

## 🔀 Feature Flag Mapping

| Old PipelineFeatures Flag | New IndexingConfig Setting |
|---------------------------|----------------------------|
| `chunk_mode="token"` | `chunker="token"` |
| `chunk_mode="semantic"` | `chunker="semantic"` |
| `enable_table_detection=True` | `chunker="semantic"` or `"hybrid"` |
| `enable_gleaning=True` | `extractor="gleaning"` |
| `enable_table_fact_extraction=True` | `extractor="hybrid"` |
| `enable_numeric_validation=True` | `validators=["numeric"]` |
| `enable_entity_validation=True` | `validators=["semantic"]` |
| `validation_strictness="MODERATE"` | `validation_strictness="MODERATE"` |
| `enable_entity_merging=True` | `merger="basic"` or `"fuzzy"` |
| `merge_strategy="fuzzy"` | `merger="fuzzy"` |
| `enable_hitl=True` | `hitl="file"` |
| `enable_orphan_linking=True` | `orphan_linker="synthetic"` |

---

## 📊 Preset Equivalents

| Old Pipeline | New Preset | Description |
|--------------|------------|-------------|
| Standard (default) | `IndexingConfig.preset_fast()` | Token chunking, strict extraction, no validation |
| Production Pipeline | `IndexingConfig.preset_quality()` | Semantic chunking, gleaning, full validation |
| *NEW* | `IndexingConfig.preset_balanced()` | Middle ground (recommended for production) |

---

## 🛠️ Automated Migration

Use the migration helper module:

```python
from bigrag.migration import migrate_pipeline_features
from bigrag.pipeline.features import PipelineFeatures

# Old features
old_features = PipelineFeatures(
    enable_table_detection=True,
    enable_gleaning=True,
    enable_numeric_validation=True
)

# Auto-convert to new config
new_config = migrate_pipeline_features(old_features)

# Use with BiGRAG
from bigrag import BiGRAG
rag = BiGRAG(indexing_config=new_config, working_dir="./expr/my_dataset")
```

---

## ⚠️ Important Notes

1. **Do NOT import from `bigrag/_archived/`** - This code is deprecated
2. **Graphs are compatible** - Old graphs work with new system (no rebuild needed)
3. **Backend API updated** - All endpoints now use new modular system
4. **Test coverage** - New system has comprehensive unit tests

---

## 📚 Documentation

- **New System Docs**: See [MODULARITY_REFACTOR_PLAN.md](../MODULARITY_REFACTOR_PLAN.md)
- **Migration Guide**: See [bigrag/migration.py](../migration.py)
- **API Reference**: See [backend/README.md](../../backend/README.md)

---

## 🗓️ Archive Date

**January 30, 2025** - Archived as part of modular indexing system implementation.

---

**Questions?** Open an issue on GitHub or check the migration guide above.
