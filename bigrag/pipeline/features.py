from dataclasses import dataclass, field
from typing import Optional, List

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
        """
        warnings = []

        # Chunking dependencies
        if self.chunk_mode == "semantic" and not self.enable_table_detection:
            warnings.append(
                "chunk_mode='semantic' requires enable_table_detection=True. "
                "Falling back to 'token' chunking."
            )

        # Extraction dependencies
        if self.enable_gleaning and self.max_gleaning_iterations < 1:
            warnings.append(
                f"enable_gleaning=True but max_gleaning_iterations={self.max_gleaning_iterations}. "
                "Setting to 1."
            )

        # Validation dependencies
        if self.enable_numeric_validation and not self.gemini_api_key:
            warnings.append(
                "enable_numeric_validation=True but gemini_api_key not provided. "
                "Falling back to regex validation."
            )

        if self.validation_strictness not in ["STRICT", "MODERATE", "LENIENT"]:
            warnings.append(
                f"validation_strictness='{self.validation_strictness}' invalid. "
                "Must be STRICT/MODERATE/LENIENT. Using MODERATE."
            )

        # Merging dependencies
        if self.merge_strategy not in ["basic", "fuzzy", "hybrid"]:
            warnings.append(
                f"merge_strategy='{self.merge_strategy}' invalid. "
                "Must be basic/fuzzy/hybrid. Using basic."
            )

        # HITL dependencies
        if self.enable_hitl and not self.enable_entity_validation:
            warnings.append(
                "enable_hitl=True works best with enable_entity_validation=True. "
                "Consider enabling entity validation for better HITL quality."
            )

        return warnings

    @classmethod
    def from_preset(cls, preset: str, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None) -> 'PipelineFeatures':
        """
        Create PipelineFeatures from preset configuration.

        Args:
            preset: "standard" | "quality" | "balanced"
            openai_api_key: OpenAI API key (required for all presets)
            gemini_api_key: Gemini API key (optional, for numeric validation)

        Returns:
            PipelineFeatures instance

        Raises:
            ValueError: If preset is unknown
        """
        presets = {
            "standard": cls._preset_standard,
            "quality": cls._preset_quality,
            "balanced": cls._preset_balanced
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")

        return presets[preset](openai_api_key=openai_api_key, gemini_api_key=gemini_api_key)

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
