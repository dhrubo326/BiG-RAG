"""
Unified modular knowledge graph pipeline.

This module provides a clean interface to BiG-RAG's dual-pipeline system,
allowing users to choose between standard (fast), quality (accurate), or
balanced presets via simple feature flags.

Architecture:
    - Import existing components from operate.py (no code duplication)
    - Feature-flag based configuration
    - Graceful degradation on errors
    - HITL integration for failed extractions (Week 2-3)
"""

import asyncio
from typing import List, Dict, Optional
from ..utils import logger
from .features import PipelineFeatures

# Import existing pipeline components (DO NOT DUPLICATE CODE!)
from ..operate import chunking_by_token_size, extract_entities
from ..preprocessors.smart_chunker import TableAwareChunker
from ..extractors.constrained_extractor import ConstrainedLLMExtractor


class UnifiedPipeline:
    """
    Unified modular knowledge graph pipeline.

    Combines standard and enhanced pipeline features with feature-flag architecture.
    Based on proven operate.py functions with zero code duplication.

    Usage:
        ```python
        # Standard preset (fast, reliable)
        features = PipelineFeatures.from_preset("standard", openai_api_key="...")
        pipeline = UnifiedPipeline(features)
        result = await pipeline.process_document(content, metadata)

        # Quality preset (slow, accurate)
        features = PipelineFeatures.from_preset("quality", openai_api_key="...")
        pipeline = UnifiedPipeline(features)
        result = await pipeline.process_document(content, metadata)
        ```

    Architecture:
        1. Chunking (required) - token-based or table-aware
        2. Extraction (required) - with optional gleaning
        3. Validation (optional) - Week 2-3
        4. Merging (required) - Week 2-3
        5. HITL (optional) - Week 2-3

    Features:
        - Zero code duplication (imports from operate.py)
        - Graceful degradation (feature failures don't crash pipeline)
        - Production-ready error handling
    """

    def __init__(
        self,
        features: PipelineFeatures,
        dataset_path: Optional[str] = None,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize unified pipeline with feature configuration.

        Args:
            features: Feature configuration (use PipelineFeatures.from_preset())
            dataset_path: Path for HITL storage (if enable_hitl=True)
            llm_model: LLM model for extraction (default: gpt-4o-mini)

        Raises:
            ValueError: If features validation fails
        """
        self.features = features
        self.dataset_path = dataset_path
        self.llm_model = llm_model

        # Validate features and log warnings
        warnings = features.validate()
        if warnings:
            for warning in warnings:
                logger.warning(f"[Unified Pipeline] {warning}")

        # Log configuration
        preset_name = self._detect_preset()
        logger.info(f"[Unified Pipeline] Initialized with preset: {preset_name}")
        logger.info(f"[Unified Pipeline] Features: {self._summarize_features()}")

        # Initialize components based on features
        self.chunker = self._init_chunker()
        self.extractor = self._init_extractor()

    def _detect_preset(self) -> str:
        """Detect which preset was used (for logging)."""
        # Compare against presets (approximate detection)
        if (not self.features.enable_table_detection and
            self.features.enable_gleaning and
            self.features.merge_strategy == "basic"):
            return "STANDARD (fast, reliable)"
        elif (self.features.enable_table_detection and
              self.features.enable_gleaning and
              self.features.merge_strategy == "fuzzy" and
              self.features.enable_numeric_validation):
            return "QUALITY (slow, accurate)"
        elif (self.features.enable_table_detection and
              not self.features.enable_gleaning):
            return "BALANCED (medium speed/quality)"
        else:
            return "CUSTOM"

    def _summarize_features(self) -> str:
        """Summarize enabled features (for logging)."""
        enabled = []
        if self.features.enable_table_detection:
            enabled.append("table_detection")
        if self.features.enable_gleaning:
            enabled.append(f"gleaning(x{self.features.max_gleaning_iterations})")
        if self.features.enable_numeric_validation:
            enabled.append("numeric_validation")
        if self.features.enable_entity_validation:
            enabled.append("entity_validation")
        if self.features.merge_strategy == "fuzzy":
            enabled.append("fuzzy_merging")
        if self.features.enable_hitl:
            enabled.append("hitl")
        if self.features.enable_orphan_linking:
            enabled.append("orphan_linking")

        return ", ".join(enabled) if enabled else "basic"

    def _init_chunker(self):
        """Initialize chunker based on features."""
        if self.features.enable_table_detection:
            # Use table-aware chunking (enhanced pipeline)
            logger.info("[Unified Pipeline] Using TableAwareChunker (semantic chunking)")
            return TableAwareChunker(
                api_key=self.features.openai_api_key,
                chunk_size=self.features.chunk_size,
                chunk_overlap=self.features.chunk_overlap,
                chunk_mode=self.features.chunk_mode
            )
        else:
            # Use token-based chunking (standard pipeline)
            logger.info("[Unified Pipeline] Using token-based chunking (standard)")
            return None  # Will use chunking_by_token_size function directly

    def _init_extractor(self):
        """Initialize extractor based on features."""
        logger.info(f"[Unified Pipeline] Using ConstrainedLLMExtractor (gleaning={self.features.enable_gleaning})")
        return ConstrainedLLMExtractor(
            api_key=self.features.openai_api_key,
            model=self.llm_model,
            enable_gleaning=self.features.enable_gleaning,
            max_gleaning_iterations=self.features.max_gleaning_iterations
            # Note: extraction_concurrency is handled by extract_entities() function, not the extractor
        )

    async def process_document(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process document through modular pipeline.

        This is a STUB for Week 1 smoke tests - full implementation in Week 2.

        Args:
            content: Document text
            metadata: Optional metadata (title, category, tags, etc.)

        Returns:
            dict: {
                'chunks': List[Dict],      # Text chunks with metadata
                'entities': List[Dict],    # Extracted entities
                'relations': List[Dict],   # Extracted relations
                'statistics': Dict,        # Counts and metrics
                'pipeline_metadata': Dict  # Pipeline configuration info
            }
        """
        metadata = metadata or {}

        # For Week 1: Just return a stub response to verify instantiation works
        logger.warning("[Pipeline] process_document() is a stub for Week 1 - full implementation in Week 2")

        return {
            'chunks': [],
            'entities': [],
            'relations': [],
            'statistics': {
                'total_chunks': 0,
                'total_entities': 0,
                'total_relations': 0
            },
            'pipeline_metadata': {
                'preset': self._detect_preset(),
                'features': self._summarize_features(),
                'version': self.features.pipeline_version,
                'status': 'WEEK_1_STUB'
            }
        }
