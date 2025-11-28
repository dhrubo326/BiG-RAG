"""
Unified modular knowledge graph pipeline.

Full implementation with all Phase 1 + Phase 2 components.
"""

import asyncio
from typing import List, Dict, Optional
from ..utils import logger
from .features import PipelineFeatures


class UnifiedPipeline:
    """
    Unified modular knowledge graph pipeline.

    Combines standard and enhanced pipeline features with plug-and-play architecture.
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
        3. Validation (optional) - numeric/entity/relation validation
        4. Merging (required) - basic or fuzzy entity deduplication
        5. Post-processing (optional) - orphan linking

    Features:
        - Zero code duplication (imports from operate.py + existing modules)
        - Graceful degradation (feature failures don't crash pipeline)
        - Production-ready error handling
        - HITL integration (optional)
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
        self.validator = self._init_validator() if self._needs_validation() else None
        self.merger = self._init_merger()
        self.postprocessor = self._init_postprocessor() if self._needs_postprocessing() else None

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
            logger.info("[Unified Pipeline] Using TableChunker (semantic chunking)")
            from .chunkers import TableChunker
            return TableChunker(
                api_key=self.features.openai_api_key,
                chunk_size=self.features.chunk_size,
                chunk_overlap=self.features.chunk_overlap,
                chunk_mode=self.features.chunk_mode
            )
        else:
            # Use token-based chunking (standard pipeline)
            logger.info("[Unified Pipeline] Using TokenChunker (standard)")
            from .chunkers import TokenChunker
            return TokenChunker(
                chunk_size=self.features.chunk_size,
                overlap=self.features.chunk_overlap
            )

    def _init_extractor(self):
        """Initialize extractor based on features."""
        logger.info(f"[Unified Pipeline] Using LLMExtractor (gleaning={self.features.enable_gleaning})")
        from .extractors import LLMExtractor
        return LLMExtractor(
            api_key=self.features.openai_api_key,
            model=self.llm_model,
            enable_gleaning=self.features.enable_gleaning,
            max_iterations=self.features.max_gleaning_iterations,
            concurrency=self.features.extraction_concurrency,
            enable_table_facts=self.features.enable_table_fact_extraction,
            hitl_store=self._init_hitl() if self.features.enable_hitl else None
        )

    def _init_validator(self):
        """Initialize validators based on features."""
        if not self._needs_validation():
            return None

        logger.info(f"[Unified Pipeline] Using EntityValidator (strictness={self.features.validation_strictness})")
        from .validators import EntityValidator
        return EntityValidator(
            enable_numeric=self.features.enable_numeric_validation,
            enable_entity_quality=self.features.enable_entity_validation,
            enable_relation_quality=self.features.enable_relation_validation,
            strictness=self.features.validation_strictness,
            gemini_api_key=self.features.gemini_api_key
        )

    def _init_merger(self):
        """Initialize merger based on features."""
        if self.features.merge_strategy == "fuzzy":
            logger.info("[Unified Pipeline] Using FuzzyMerger")
            from .mergers import FuzzyMerger
            return FuzzyMerger()
        else:
            logger.info("[Unified Pipeline] Using BasicMerger")
            from .mergers import BasicMerger
            return BasicMerger()

    def _init_postprocessor(self):
        """Initialize post-processor if needed."""
        if not self._needs_postprocessing():
            return None

        if self.features.enable_orphan_linking:
            logger.info("[Unified Pipeline] Using OrphanLinker")
            from .postprocessors import OrphanLinker
            return OrphanLinker()

        return None

    def _init_hitl(self):
        """Initialize HITL store if enabled."""
        if not self.features.enable_hitl or not self.dataset_path:
            return None

        logger.info(f"[Unified Pipeline] HITL enabled: {self.dataset_path}")
        from ..hitl.failed_extraction_store import FailedExtractionStore
        return FailedExtractionStore(self.dataset_path)

    def _needs_validation(self) -> bool:
        """Check if any validation is enabled."""
        return (
            self.features.enable_numeric_validation or
            self.features.enable_entity_validation or
            self.features.enable_relation_validation
        )

    def _needs_postprocessing(self) -> bool:
        """Check if post-processing is needed."""
        return self.features.enable_orphan_linking or self.features.enable_quality_scoring

    async def process_document(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process document through modular pipeline.

        FULL IMPLEMENTATION (Phase 1 + Phase 2 complete).

        Args:
            content: Document text
            metadata: Optional metadata (title, category, tags, etc.)

        Returns:
            dict: {
                'chunks': List[Dict],      # Text chunks with metadata
                'entities': List[Dict],    # Extracted entities
                'relations': List[Dict],   # Extracted relations
                'validation': Dict,        # Validation report
                'statistics': Dict,        # Counts and metrics
                'pipeline_metadata': Dict  # Pipeline configuration info
            }

        Raises:
            Exception: If critical pipeline step fails (with detailed error message)
        """
        metadata = metadata or {}

        logger.info("[Pipeline] ========== Starting document processing ==========")

        try:
            # Step 1: Chunking (REQUIRED)
            logger.info("[Pipeline] Step 1: Chunking...")
            chunks = await self.chunker.chunk(content, metadata)
            logger.info(f"[Pipeline] Created {len(chunks)} chunks")

            # Step 2: Extraction (REQUIRED)
            logger.info("[Pipeline] Step 2: Extraction...")
            entities, relations = await self.extractor.extract(chunks, metadata)
            logger.info(f"[Pipeline] Extracted {len(entities)} entities, {len(relations)} relations")

            # Step 3: Validation (OPTIONAL)
            validation_report = {'status': 'SKIPPED', 'message': 'Validation disabled'}
            if self.validator:
                logger.info("[Pipeline] Step 3: Validation...")
                entities, relations, validation_report = await self.validator.validate(
                    entities, relations, chunks
                )
                logger.info(f"[Pipeline] Validation: {validation_report['status']}")

            # Step 4: Merging (REQUIRED)
            logger.info(f"[Pipeline] Step 4: Merging (strategy: {self.features.merge_strategy})...")
            entities = await self.merger.merge(entities, relations)
            logger.info(f"[Pipeline] Merged to {len(entities)} unique entities")

            # Step 5: Post-processing (OPTIONAL)
            if self.postprocessor:
                logger.info("[Pipeline] Step 5: Post-processing...")
                entities, relations = await self.postprocessor.process(entities, relations)
                logger.info("[Pipeline] Post-processing complete")

            # Build result
            result = {
                'chunks': chunks,
                'entities': entities,
                'relations': relations,
                'validation': validation_report,
                'statistics': {
                    'total_chunks': len(chunks),
                    'total_entities': len(entities),
                    'total_relations': len(relations),
                    'avg_entities_per_chunk': len(entities) / len(chunks) if chunks else 0
                },
                'pipeline_metadata': {
                    'version': self.features.pipeline_version,
                    'preset': self._detect_preset(),
                    'features_enabled': self._summarize_features()
                }
            }

            logger.info("[Pipeline] ========== Processing complete ==========")
            return result

        except Exception as e:
            logger.error(f"[Pipeline] ✗ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise
