"""
Migration Helper Module

Provides backward compatibility and automatic migration from old PipelineFeatures
to new IndexingConfig system.

Usage:
    from bigrag.migration import migrate_pipeline_features
    from bigrag.pipeline.features import PipelineFeatures

    old_features = PipelineFeatures(enable_gleaning=True, ...)
    new_config = migrate_pipeline_features(old_features)
"""

from typing import List
from bigrag.config import IndexingConfig


def migrate_pipeline_features(features) -> IndexingConfig:
    """
    Migrate old PipelineFeatures to new IndexingConfig.

    Args:
        features: PipelineFeatures instance (from bigrag/pipeline/features.py)

    Returns:
        IndexingConfig instance with equivalent settings

    Example:
        >>> from bigrag.pipeline.features import PipelineFeatures
        >>> old = PipelineFeatures(enable_gleaning=True, enable_numeric_validation=True)
        >>> new = migrate_pipeline_features(old)
        >>> new.extractor  # Returns "gleaning"
        >>> new.validators  # Returns ["numeric"]
    """
    # Map chunker strategy
    chunker = _map_chunker(
        chunk_mode=getattr(features, 'chunk_mode', 'token'),
        enable_table_detection=getattr(features, 'enable_table_detection', False)
    )

    # Map extractor strategy
    extractor = _map_extractor(
        gleaning=getattr(features, 'enable_gleaning', False),
        table_facts=getattr(features, 'enable_table_fact_extraction', False)
    )

    # Map validators
    validators = _build_validator_list(
        numeric=getattr(features, 'enable_numeric_validation', False),
        semantic=getattr(features, 'enable_entity_validation', False)
    )

    # Map merger strategy
    merger = _map_merger(
        enable_merging=getattr(features, 'enable_entity_merging', True),
        merge_strategy=getattr(features, 'merge_strategy', 'basic')
    )

    # Map HITL strategy
    hitl = _map_hitl(
        enable_hitl=getattr(features, 'enable_hitl', False)
    )

    # Map orphan linker
    orphan_linker = _map_orphan_linker(
        enable_orphan_linking=getattr(features, 'enable_orphan_linking', True)
    )

    # Create IndexingConfig with mapped settings
    return IndexingConfig(
        chunker=chunker,
        extractor=extractor,
        validators=validators,
        merger=merger,
        hitl=hitl,
        orphan_linker=orphan_linker,
        # Parameters
        chunk_size=getattr(features, 'chunk_size', 1200),
        chunk_overlap=getattr(features, 'chunk_overlap', 100),
        gleaning_iterations=getattr(features, 'gleaning_iterations', 2),
        extraction_concurrency=getattr(features, 'extraction_concurrency', 16),
        validation_strictness=getattr(features, 'validation_strictness', 'MODERATE'),
        enable_quality_scoring=getattr(features, 'enable_quality_scoring', True),
        enable_llm_cache=getattr(features, 'enable_llm_cache', True),
        openai_api_key=getattr(features, 'openai_api_key', None),
        gemini_api_key=getattr(features, 'gemini_api_key', None),
        dataset_path=getattr(features, 'dataset_path', None)
    )


def _map_chunker(chunk_mode: str, enable_table_detection: bool) -> str:
    """
    Map old chunking settings to new chunker strategy.

    Logic:
    - token mode + no tables → "token"
    - semantic mode + tables → "semantic" (table-aware)
    - token mode + tables → "hybrid" (best of both)
    """
    if chunk_mode == "token" and not enable_table_detection:
        return "token"
    elif chunk_mode == "semantic" or enable_table_detection:
        return "semantic"  # Semantic chunker is table-aware
    else:
        return "token"


def _map_extractor(gleaning: bool, table_facts: bool) -> str:
    """
    Map old extraction settings to new extractor strategy.

    Logic:
    - Tables + paragraphs → "hybrid"
    - Gleaning enabled → "gleaning"
    - Default → "strict"
    """
    if table_facts:
        return "hybrid"  # Tables + paragraphs
    elif gleaning:
        return "gleaning"  # Multi-pass
    else:
        return "strict"  # Single-pass


def _build_validator_list(numeric: bool, semantic: bool) -> List[str]:
    """
    Map old validation flags to new validator list.

    Args:
        numeric: enable_numeric_validation flag
        semantic: enable_entity_validation flag (maps to semantic validator)

    Returns:
        List of validator names (e.g., ["numeric", "semantic"])
    """
    validators = []
    if numeric:
        validators.append('numeric')
    if semantic:
        validators.append('semantic')
    return validators


def _map_merger(enable_merging: bool, merge_strategy: str) -> str:
    """
    Map old merging settings to new merger strategy.

    Args:
        enable_merging: Whether entity merging is enabled
        merge_strategy: "basic", "fuzzy", or "hybrid"

    Returns:
        Merger strategy name
    """
    if not enable_merging:
        return "basic"  # No noop merger - always merge with basic
    return merge_strategy  # "basic", "fuzzy", or "hybrid"


def _map_hitl(enable_hitl: bool) -> str:
    """
    Map old HITL flag to new HITL strategy.

    Args:
        enable_hitl: Whether HITL is enabled

    Returns:
        "file" if enabled, "noop" if disabled
    """
    return "file" if enable_hitl else "noop"


def _map_orphan_linker(enable_orphan_linking: bool) -> str:
    """
    Map old orphan linking flag to new orphan linker strategy.

    Args:
        enable_orphan_linking: Whether orphan linking is enabled

    Returns:
        "synthetic" if enabled, "noop" if disabled
    """
    return "synthetic" if enable_orphan_linking else "noop"


# ========== BACKWARD COMPATIBILITY HELPERS ==========

def features_to_dict(features) -> dict:
    """
    Convert PipelineFeatures to dict for inspection.

    Useful for debugging migration issues.
    """
    return {
        'chunk_mode': getattr(features, 'chunk_mode', 'token'),
        'chunk_size': getattr(features, 'chunk_size', 1200),
        'chunk_overlap': getattr(features, 'chunk_overlap', 100),
        'enable_table_detection': getattr(features, 'enable_table_detection', False),
        'enable_gleaning': getattr(features, 'enable_gleaning', False),
        'gleaning_iterations': getattr(features, 'gleaning_iterations', 2),
        'enable_table_fact_extraction': getattr(features, 'enable_table_fact_extraction', False),
        'enable_numeric_validation': getattr(features, 'enable_numeric_validation', False),
        'enable_entity_validation': getattr(features, 'enable_entity_validation', False),
        'validation_strictness': getattr(features, 'validation_strictness', 'MODERATE'),
        'enable_entity_merging': getattr(features, 'enable_entity_merging', True),
        'merge_strategy': getattr(features, 'merge_strategy', 'basic'),
        'enable_hitl': getattr(features, 'enable_hitl', False),
        'enable_orphan_linking': getattr(features, 'enable_orphan_linking', True),
    }


def config_to_features_dict(config: IndexingConfig) -> dict:
    """
    Convert IndexingConfig back to feature flag dict (for comparison).

    Useful for testing migration correctness.
    """
    return {
        'chunker': config.chunker,
        'extractor': config.extractor,
        'validators': config.validators,
        'merger': config.merger,
        'hitl': config.hitl,
        'orphan_linker': config.orphan_linker,
        'chunk_size': config.chunk_size,
        'chunk_overlap': config.chunk_overlap,
        'gleaning_iterations': config.gleaning_iterations,
        'extraction_concurrency': config.extraction_concurrency,
        'validation_strictness': config.validation_strictness,
        'enable_quality_scoring': config.enable_quality_scoring,
        'enable_llm_cache': config.enable_llm_cache,
    }
