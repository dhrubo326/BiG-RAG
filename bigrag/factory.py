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
        """Create chunker strategy from config."""
        if config.chunker == "token":
            from bigrag.strategies.chunking.token import TokenChunker
            return TokenChunker(
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap
            )
        elif config.chunker == "semantic":
            from bigrag.strategies.chunking.semantic import SemanticChunker
            return SemanticChunker(
                api_key=config.openai_api_key,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap
            )
        elif config.chunker == "hybrid":
            from bigrag.strategies.chunking.hybrid import HybridChunker
            return HybridChunker(
                api_key=config.openai_api_key,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown chunker: {config.chunker}")

    @staticmethod
    def create_extractor(config: IndexingConfig) -> ExtractorInterface:
        """Create extractor strategy from config."""
        if config.extractor == "strict":
            from bigrag.strategies.extraction.strict import StrictExtractor
            return StrictExtractor(
                api_key=config.openai_api_key,
                concurrency=config.extraction_concurrency,
                enable_validation='numeric' in config.validators
            )
        elif config.extractor == "gleaning":
            from bigrag.strategies.extraction.gleaning import GleaningExtractor
            return GleaningExtractor(
                api_key=config.openai_api_key,
                max_iterations=config.gleaning_iterations,
                concurrency=config.extraction_concurrency,
                enable_validation='numeric' in config.validators
            )
        elif config.extractor == "hybrid":
            from bigrag.strategies.extraction.hybrid import HybridExtractor
            return HybridExtractor(
                api_key=config.openai_api_key,
                gleaning_iterations=config.gleaning_iterations,
                concurrency=config.extraction_concurrency,
                enable_validation='numeric' in config.validators
            )
        else:
            raise ValueError(f"Unknown extractor: {config.extractor}")

    @staticmethod
    def create_validator(config: IndexingConfig) -> ValidatorInterface:
        """Create validator strategy from config."""
        if not config.validators:
            from bigrag.strategies.validation.noop import NoOpValidator
            return NoOpValidator()

        if len(config.validators) == 1:
            # Single validator
            if 'numeric' in config.validators:
                from bigrag.strategies.validation.numeric import NumericValidator
                return NumericValidator(
                    api_key=config.gemini_api_key,
                    strictness=config.validation_strictness
                )
            else:  # semantic
                from bigrag.strategies.validation.semantic import SemanticValidator
                return SemanticValidator(
                    strictness=config.validation_strictness
                )
        else:
            # Multiple validators - use composite
            from bigrag.strategies.validation.composite import CompositeValidator
            from bigrag.strategies.validation.numeric import NumericValidator
            from bigrag.strategies.validation.semantic import SemanticValidator

            validators = []
            if 'numeric' in config.validators:
                validators.append(NumericValidator(
                    api_key=config.gemini_api_key,
                    strictness=config.validation_strictness
                ))
            if 'semantic' in config.validators:
                validators.append(SemanticValidator(
                    strictness=config.validation_strictness
                ))

            return CompositeValidator(validators)

    @staticmethod
    def create_merger(config: IndexingConfig) -> MergerInterface:
        """Create merger strategy from config."""
        if config.merger == "basic":
            from bigrag.strategies.merging.basic import BasicMerger
            return BasicMerger()
        elif config.merger == "fuzzy":
            from bigrag.strategies.merging.fuzzy import FuzzyMerger
            return FuzzyMerger()
        elif config.merger == "hybrid":
            from bigrag.strategies.merging.hybrid import HybridMerger
            return HybridMerger()
        else:
            raise ValueError(f"Unknown merger: {config.merger}")

    @staticmethod
    def create_hitl(config: IndexingConfig) -> HITLInterface:
        """Create HITL strategy from config."""
        if config.hitl == "noop":
            from bigrag.strategies.hitl.noop import NoOpHITL
            return NoOpHITL()
        elif config.hitl == "file":
            from bigrag.strategies.hitl.file import FileHITL
            return FileHITL(dataset_path=config.dataset_path)
        elif config.hitl == "database":
            from bigrag.strategies.hitl.database import DatabaseHITL
            return DatabaseHITL()  # Connection string would come from config
        else:
            raise ValueError(f"Unknown hitl: {config.hitl}")

    @staticmethod
    def create_orphan_linker(config: IndexingConfig) -> OrphanLinkerInterface:
        """Create orphan linker strategy from config."""
        if config.orphan_linker == "noop":
            from bigrag.strategies.orphan_linking.noop import NoOpOrphanLinker
            return NoOpOrphanLinker()
        elif config.orphan_linker == "synthetic":
            from bigrag.strategies.orphan_linking.synthetic import SyntheticOrphanLinker
            return SyntheticOrphanLinker()
        else:
            raise ValueError(f"Unknown orphan_linker: {config.orphan_linker}")
