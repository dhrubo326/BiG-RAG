"""
StrategyFactory - Build strategy instances from IndexingConfig.

This factory creates all strategy instances needed for the BiGRAG indexing system.
"""

from bigrag.config import IndexingConfig
from bigrag.interfaces import (
    ChunkerInterface,
    ExtractorInterface,
    ValidatorInterface,
    MergerInterface,
    HITLInterface,
    OrphanLinkerInterface,
)
from typing import Dict


class StrategyFactory:
    """Factory for creating strategy instances from IndexingConfig."""

    @staticmethod
    def build(config: IndexingConfig) -> Dict:
        """
        Build all strategies from config.

        Args:
            config: IndexingConfig

        Returns:
            {
                'chunker': ChunkerInterface,
                'extractor': ExtractorInterface,
                'validator': ValidatorInterface,
                'merger': MergerInterface,
                'hitl': HITLInterface,
                'orphan_linker': OrphanLinkerInterface
            }
        """
        return {
            'chunker': StrategyFactory.create_chunker(config),
            'extractor': StrategyFactory.create_extractor(config),
            'validator': StrategyFactory.create_validator(config),
            'merger': StrategyFactory.create_merger(config),
            'hitl': StrategyFactory.create_hitl(config),
            'orphan_linker': StrategyFactory.create_orphan_linker(config)
        }

    @staticmethod
    def create_chunker(config: IndexingConfig) -> ChunkerInterface:
        """
        Create chunker strategy from config.

        Checks registry first for custom strategies, falls back to built-in strategies.
        """
        # Check registry for custom strategies
        from bigrag.registry import StrategyRegistry
        try:
            custom_class = StrategyRegistry.get_chunker(config.chunking_strategy)
            # Instantiate custom strategy (pass config for flexibility)
            return custom_class(
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                api_key=config.openai_api_key
            )
        except KeyError:
            pass  # Not in registry, try built-in strategies

        # Built-in strategies
        if config.chunking_strategy == "token":
            from bigrag.strategies.chunking.token import TokenChunker
            return TokenChunker(
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap
            )
        elif config.chunking_strategy == "semantic":
            from bigrag.strategies.chunking.semantic import SemanticChunker
            # NEW: Pass HITL handler if available
            hitl_handler = StrategyFactory.create_hitl(config) if config.enable_hitl else None
            return SemanticChunker(
                api_key=config.openai_api_key,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                enable_table_detection=config.enable_table_detection,
                hitl_handler=hitl_handler  # NEW: Issue #3
            )
        elif config.chunking_strategy == "hybrid":
            from bigrag.strategies.chunking.hybrid import HybridChunker
            # NEW: Pass HITL handler if available
            hitl_handler = StrategyFactory.create_hitl(config) if config.enable_hitl else None
            return HybridChunker(
                api_key=config.openai_api_key,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                enable_table_detection=config.enable_table_detection,
                hitl_handler=hitl_handler  # NEW: Issue #3
            )
        else:
            raise ValueError(f"Unknown chunking_strategy: {config.chunking_strategy}. Register custom chunkers using StrategyRegistry.")

    @staticmethod
    def create_extractor(config: IndexingConfig) -> ExtractorInterface:
        """Create extractor strategy from config."""
        if config.extraction_strategy == "strict":
            from bigrag.strategies.extraction.strict import StrictExtractor
            return StrictExtractor(
                api_key=config.openai_api_key,
                concurrency=config.extraction_concurrency,
                enable_validation=config.enable_numeric_validation
            )
        elif config.extraction_strategy == "gleaning":
            from bigrag.strategies.extraction.gleaning import GleaningExtractor
            return GleaningExtractor(
                api_key=config.openai_api_key,
                max_iterations=config.gleaning_iterations,
                concurrency=config.extraction_concurrency,
                enable_validation=config.enable_numeric_validation
            )
        elif config.extraction_strategy == "hybrid":
            from bigrag.strategies.extraction.hybrid import HybridExtractor
            # NEW: Pass HITL handler and enable_table_fact_extraction
            hitl_handler = StrategyFactory.create_hitl(config) if config.enable_hitl else None
            return HybridExtractor(
                api_key=config.openai_api_key,
                gleaning_iterations=config.gleaning_iterations,
                concurrency=config.extraction_concurrency,
                enable_validation=config.enable_numeric_validation,
                enable_table_fact_extraction=config.enable_table_fact_extraction,  # NEW: Issue #4
                hitl_handler=hitl_handler  # NEW: Issue #7
            )
        else:
            raise ValueError(f"Unknown extraction_strategy: {config.extraction_strategy}")

    @staticmethod
    def create_validator(config: IndexingConfig) -> ValidatorInterface:
        """Create validator strategy from config (NEW: uses boolean flags)."""
        # Build list of validators based on boolean flags
        validators = []

        if config.enable_numeric_validation:
            from bigrag.strategies.validation.numeric import NumericValidator
            validators.append(NumericValidator(
                api_key=config.gemini_api_key,
                strictness=config.validation_strictness,
                validation_mode=config.numeric_validation_mode
            ))

        if config.enable_entity_validation:
            from bigrag.strategies.validation.entity import EntityValidator
            validators.append(EntityValidator(strictness=config.validation_strictness))

        if config.enable_relation_validation:
            from bigrag.strategies.validation.relation import RelationValidator
            validators.append(RelationValidator(strictness=config.validation_strictness))

        # Return appropriate validator based on count
        if not validators:
            from bigrag.strategies.validation.noop import NoOpValidator
            return NoOpValidator()
        elif len(validators) == 1:
            return validators[0]
        else:
            from bigrag.strategies.validation.composite import CompositeValidator
            return CompositeValidator(validators)

    @staticmethod
    def create_merger(config: IndexingConfig) -> MergerInterface:
        """Create merger strategy from config (NEW: uses boolean flags)."""
        if not config.enable_entity_merging:
            from bigrag.strategies.merging.noop import NoOpMerger
            return NoOpMerger()
        elif config.enable_fuzzy_matching:
            from bigrag.strategies.merging.fuzzy import FuzzyMerger
            return FuzzyMerger(fuzzy_threshold=config.fuzzy_similarity_threshold)
        else:
            from bigrag.strategies.merging.basic import BasicMerger
            return BasicMerger()

    @staticmethod
    def create_hitl(config: IndexingConfig) -> HITLInterface:
        """Create HITL strategy from config (NEW: uses boolean flag)."""
        if not config.enable_hitl:
            from bigrag.strategies.hitl.noop import NoOpHITL
            return NoOpHITL()
        else:
            from bigrag.strategies.hitl.file import FileHITL
            return FileHITL(dataset_path=config.dataset_path)

    @staticmethod
    def create_orphan_linker(config: IndexingConfig) -> OrphanLinkerInterface:
        """Create orphan linker strategy from config (NEW: uses boolean flag)."""
        if not config.enable_orphan_linking:
            from bigrag.strategies.orphan_linking.noop import NoOpOrphanLinker
            return NoOpOrphanLinker()
        else:
            from bigrag.strategies.orphan_linking.synthetic import SyntheticOrphanLinker
            return SyntheticOrphanLinker()
