# BiG-RAG Modular Indexing System - Refactoring Plan

**Date**: January 30, 2025
**Goal**: Replace all pipeline variants with a single, modular BiG-RAG indexing system
**Philosophy**: One indexing process, infinitely configurable via strategy pattern

---

## Executive Summary

**Current State**: Multiple pipeline classes (StandardPipeline, ProductionPipeline, EnhancedKGPipeline) with duplicated code and confusing naming.

**Target State**: **Single BiGRAG class** with modular, plug-and-play components.

**Approach**:
- ✅ Keep all existing features (13 feature flags)
- ✅ Keep all storage structures (GraphML, JSON, vector DBs)
- ❌ Remove all pipeline variants (archive for reference)
- ❌ Remove backward compatibility (clean slate)
- ✅ Redesign function organization via Strategy Pattern

**Timeline**: 7-11 days of focused refactoring

---

## Design Principles

### 1. Single Entry Point
```python
from bigrag import BiGRAG

# That's it. One class, all features.
rag = BiGRAG(config)
await rag.index_document(text, metadata)
```

### 2. Strategy-Based Configuration
```python
from bigrag.config import IndexingConfig

config = IndexingConfig(
    # Chunking
    chunker="semantic",           # or "token", "hybrid"

    # Extraction
    extractor="gleaning",          # or "strict", "hybrid"
    gleaning_iterations=2,

    # Validation
    validators=["numeric", "semantic"],  # or [], ["numeric"], ["semantic"]
    validation_strictness="MODERATE",

    # Merging
    merger="fuzzy",                # or "basic", "hybrid"

    # Quality
    enable_hitl=True,
    enable_orphan_linking=True
)
```

### 3. Zero Code Duplication
```
OLD (3 pipelines):
- standard_pipeline.py       (800 lines)
- production_pipeline.py     (950 lines)
- enhanced_pipeline.py       (1000 lines)
Total: 2750 lines with 60%+ duplication

NEW (1 system):
- bigrag/indexer.py          (~300 lines - orchestration)
- bigrag/strategies/...      (~1200 lines - all strategies)
Total: 1500 lines, 0% duplication
```

---

## New Architecture

### Directory Structure

```
bigrag/
├── __init__.py                 # Export BiGRAG class
├── indexer.py                  # BiGRAG class (orchestrator, ~300 lines)
├── config.py                   # IndexingConfig (replaces PipelineFeatures)
│
├── interfaces/                 # Abstract base classes
│   ├── __init__.py
│   ├── chunker.py             # ChunkerInterface
│   ├── extractor.py           # ExtractorInterface
│   ├── validator.py           # ValidatorInterface
│   ├── merger.py              # MergerInterface
│   └── hitl.py                # HITLInterface
│
├── strategies/                 # Concrete implementations
│   ├── chunking/
│   │   ├── token.py           # TokenChunker
│   │   ├── semantic.py        # SemanticChunker (table-aware)
│   │   └── hybrid.py          # HybridChunker
│   │
│   ├── extraction/
│   │   ├── strict.py          # StrictExtractor (single-pass)
│   │   ├── gleaning.py        # GleaningExtractor (multi-pass)
│   │   └── hybrid.py          # HybridExtractor (tables + gleaning)
│   │
│   ├── validation/
│   │   ├── numeric.py         # NumericValidator
│   │   ├── semantic.py        # SemanticValidator (entity/relation)
│   │   ├── composite.py       # CompositeValidator (run multiple)
│   │   └── noop.py            # NoOpValidator (skip validation)
│   │
│   ├── merging/
│   │   ├── basic.py           # BasicMerger (exact match)
│   │   ├── fuzzy.py           # FuzzyMerger (edit distance + aliases)
│   │   └── hybrid.py          # HybridMerger (adaptive)
│   │
│   └── hitl/
│       ├── file.py            # FileHITL (save to JSON)
│       ├── database.py        # DatabaseHITL (future - SQLite)
│       └── noop.py            # NoOpHITL (disable)
│
├── factory.py                  # StrategyFactory (build strategies from config)
├── storage/                    # Existing storage backends (unchanged)
├── builders/                   # Existing graph builders (unchanged)
├── preprocessors/              # ARCHIVE: table_extractor.py, smart_chunker.py
├── extractors/                 # ARCHIVE: constrained_extractor.py, table_fact_extractor.py
├── merging/                    # ARCHIVE: entity_linker.py, unified_merger.py
└── validators/                 # ARCHIVE: numeric_validator.py

archive/                        # OLD CODE (reference only)
├── standard_pipeline.py
├── production_pipeline.py
└── enhanced_pipeline.py
```

---

## Core Classes

### 1. BiGRAG (Main Class)

**File**: `bigrag/indexer.py`

**Purpose**: Orchestrate indexing with dependency-injected strategies

**Size**: ~300 lines (thin coordinator)

```python
from typing import Dict, List, Optional
from bigrag.interfaces import (
    ChunkerInterface, ExtractorInterface, ValidatorInterface,
    MergerInterface, HITLInterface
)

class BiGRAG:
    """
    BiG-RAG: Bipartite Graph Retrieval-Augmented Generation.

    Single, modular indexing system with plug-and-play strategies.

    Usage:
        from bigrag import BiGRAG
        from bigrag.config import IndexingConfig

        config = IndexingConfig(
            chunker="semantic",
            extractor="gleaning",
            validators=["numeric", "semantic"],
            merger="fuzzy"
        )

        rag = BiGRAG(config)
        result = await rag.index_document(text, metadata)
    """

    def __init__(
        self,
        config: 'IndexingConfig',
        # Storage (existing - unchanged)
        graph_storage=None,
        vector_storage=None,
        kv_storage=None
    ):
        """
        Initialize BiGRAG indexing system.

        Args:
            config: IndexingConfig with feature flags and strategy choices
            graph_storage: Graph backend (default: NetworkX)
            vector_storage: Vector backend (default: NanoVectorDB)
            kv_storage: KV backend (default: JSON)
        """
        self.config = config

        # Build strategies from config (via factory)
        from bigrag.factory import StrategyFactory
        strategies = StrategyFactory.build(config)

        # Inject dependencies (no hard-coded implementations)
        self.chunker: ChunkerInterface = strategies['chunker']
        self.extractor: ExtractorInterface = strategies['extractor']
        self.validator: ValidatorInterface = strategies['validator']
        self.merger: MergerInterface = strategies['merger']
        self.hitl: HITLInterface = strategies['hitl']

        # Storage (existing code - unchanged)
        self.graph_storage = graph_storage or self._init_graph_storage()
        self.vector_storage = vector_storage or self._init_vector_storage()
        self.kv_storage = kv_storage or self._init_kv_storage()

    async def index_document(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Index a single document into BiG-RAG knowledge graph.

        Pipeline:
        1. Chunk document (strategy: token/semantic/hybrid)
        2. Extract entities + relations (strategy: strict/gleaning/hybrid)
        3. Validate extractions (strategy: numeric/semantic/composite/noop)
        4. Merge entities (strategy: basic/fuzzy/hybrid)
        5. Build bipartite graph
        6. Store to disk

        Args:
            text: Document content (markdown)
            metadata: Optional metadata (title, category, tags)

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'statistics': {...},
                'validation': {...}
            }
        """
        # Step 1: Chunk
        chunks = await self.chunker.chunk(text, metadata)

        # Step 2: Extract
        extractions = await self.extractor.extract(chunks)

        # Step 3: Validate
        validated = await self.validator.validate(extractions)

        # Step 4: Handle HITL failures
        if validated['failed_chunks']:
            await self.hitl.save_failures(
                validated['failed_chunks'],
                metadata=metadata
            )

        # Step 5: Merge entities
        merged_entities = await self.merger.merge(validated['entities'])

        # Step 6: Build graph (existing code)
        await self._build_graph(
            entities=merged_entities,
            relations=validated['relations'],
            chunks=chunks
        )

        # Step 7: Persist (existing code)
        await self._persist()

        return {
            'entities': merged_entities,
            'relations': validated['relations'],
            'statistics': self._compute_stats(merged_entities, validated),
            'validation': validated['summary']
        }

    async def query(self, query: str, **kwargs) -> List[Dict]:
        """Query the knowledge graph (existing code - unchanged)."""
        # Existing retrieval code from base.py
        pass

    async def delete_document(self, doc_id: str) -> None:
        """Delete document with cascade cleanup (existing code - unchanged)."""
        # Existing deletion code from bigrag.py
        pass
```

---

### 2. IndexingConfig (Configuration)

**File**: `bigrag/config.py`

**Purpose**: Replace PipelineFeatures with cleaner, strategy-focused config

**Size**: ~200 lines

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class IndexingConfig:
    """
    Configuration for BiGRAG indexing system.

    All features from old pipelines consolidated here.
    """

    # ========== STRATEGIES ==========
    chunker: str = "semantic"
    """Chunking strategy: 'token' | 'semantic' | 'hybrid'"""

    extractor: str = "gleaning"
    """Extraction strategy: 'strict' | 'gleaning' | 'hybrid'"""

    validators: List[str] = None
    """Validation strategies: [] | ['numeric'] | ['semantic'] | ['numeric', 'semantic']"""

    merger: str = "fuzzy"
    """Merging strategy: 'basic' | 'fuzzy' | 'hybrid'"""

    hitl: str = "file"
    """HITL strategy: 'file' | 'database' | 'noop'"""

    # ========== PARAMETERS ==========
    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 100

    # Extraction
    gleaning_iterations: int = 2
    extraction_concurrency: int = 16

    # Validation
    validation_strictness: str = "MODERATE"  # STRICT | MODERATE | LENIENT

    # Quality
    enable_orphan_linking: bool = True
    enable_quality_scoring: bool = True

    # API Keys
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.validators is None:
            self.validators = []  # Default: no validation

        # Validate strategy choices
        valid_chunkers = ['token', 'semantic', 'hybrid']
        if self.chunker not in valid_chunkers:
            raise ValueError(f"chunker must be one of {valid_chunkers}")

        valid_extractors = ['strict', 'gleaning', 'hybrid']
        if self.extractor not in valid_extractors:
            raise ValueError(f"extractor must be one of {valid_extractors}")

        valid_validators = ['numeric', 'semantic']
        for v in self.validators:
            if v not in valid_validators:
                raise ValueError(f"validator '{v}' invalid. Choose from {valid_validators}")

        valid_mergers = ['basic', 'fuzzy', 'hybrid']
        if self.merger not in valid_mergers:
            raise ValueError(f"merger must be one of {valid_mergers}")

    @classmethod
    def preset_fast(cls, **kwargs) -> 'IndexingConfig':
        """Fast preset: token chunking, strict extraction, basic merging."""
        return cls(
            chunker="token",
            extractor="strict",
            validators=[],
            merger="basic",
            hitl="noop",
            **kwargs
        )

    @classmethod
    def preset_balanced(cls, **kwargs) -> 'IndexingConfig':
        """Balanced preset: semantic chunking, gleaning, fuzzy merging."""
        return cls(
            chunker="semantic",
            extractor="gleaning",
            validators=["semantic"],
            merger="fuzzy",
            hitl="file",
            validation_strictness="LENIENT",
            **kwargs
        )

    @classmethod
    def preset_quality(cls, **kwargs) -> 'IndexingConfig':
        """Quality preset: all features enabled, strict validation."""
        return cls(
            chunker="semantic",
            extractor="hybrid",
            validators=["numeric", "semantic"],
            merger="fuzzy",
            hitl="file",
            validation_strictness="MODERATE",
            enable_orphan_linking=True,
            enable_quality_scoring=True,
            **kwargs
        )
```

---

### 3. StrategyFactory (Builder)

**File**: `bigrag/factory.py`

**Purpose**: Build strategy instances from config

**Size**: ~150 lines

```python
from bigrag.config import IndexingConfig
from bigrag.interfaces import (
    ChunkerInterface, ExtractorInterface, ValidatorInterface,
    MergerInterface, HITLInterface
)

class StrategyFactory:
    """Factory for creating strategy instances from IndexingConfig."""

    @staticmethod
    def build(config: IndexingConfig) -> dict:
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
                'hitl': HITLInterface
            }
        """
        return {
            'chunker': StrategyFactory._build_chunker(config),
            'extractor': StrategyFactory._build_extractor(config),
            'validator': StrategyFactory._build_validator(config),
            'merger': StrategyFactory._build_merger(config),
            'hitl': StrategyFactory._build_hitl(config)
        }

    @staticmethod
    def _build_chunker(config: IndexingConfig) -> ChunkerInterface:
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

    @staticmethod
    def _build_extractor(config: IndexingConfig) -> ExtractorInterface:
        if config.extractor == "strict":
            from bigrag.strategies.extraction.strict import StrictExtractor
            return StrictExtractor(
                api_key=config.openai_api_key,
                enable_validation='numeric' in config.validators
            )
        elif config.extractor == "gleaning":
            from bigrag.strategies.extraction.gleaning import GleaningExtractor
            return GleaningExtractor(
                api_key=config.openai_api_key,
                max_iterations=config.gleaning_iterations,
                enable_validation='numeric' in config.validators
            )
        elif config.extractor == "hybrid":
            from bigrag.strategies.extraction.hybrid import HybridExtractor
            return HybridExtractor(
                api_key=config.openai_api_key,
                gleaning_iterations=config.gleaning_iterations,
                enable_validation='numeric' in config.validators
            )

    @staticmethod
    def _build_validator(config: IndexingConfig) -> ValidatorInterface:
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
    def _build_merger(config: IndexingConfig) -> MergerInterface:
        if config.merger == "basic":
            from bigrag.strategies.merging.basic import BasicMerger
            return BasicMerger()
        elif config.merger == "fuzzy":
            from bigrag.strategies.merging.fuzzy import FuzzyMerger
            return FuzzyMerger()
        elif config.merger == "hybrid":
            from bigrag.strategies.merging.hybrid import HybridMerger
            return HybridMerger()

    @staticmethod
    def _build_hitl(config: IndexingConfig) -> HITLInterface:
        if config.hitl == "noop":
            from bigrag.strategies.hitl.noop import NoOpHITL
            return NoOpHITL()
        elif config.hitl == "file":
            from bigrag.strategies.hitl.file import FileHITL
            return FileHITL(dataset_path=...)  # From config
        elif config.hitl == "database":
            from bigrag.strategies.hitl.database import DatabaseHITL
            return DatabaseHITL(connection_string=...)
```

---

### 4. Strategy Interfaces

**File**: `bigrag/interfaces/chunker.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class ChunkerInterface(ABC):
    """Abstract interface for document chunking strategies."""

    @abstractmethod
    async def chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk document into processable segments.

        Args:
            text: Document content
            metadata: Optional metadata (title, category, tags)

        Returns:
            List of chunks: [
                {
                    'id': 'chunk-abc123',
                    'type': 'paragraph' | 'table',
                    'content': '...',
                    'metadata': {...}
                }
            ]
        """
        pass
```

**File**: `bigrag/interfaces/extractor.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict

class ExtractorInterface(ABC):
    """Abstract interface for entity/relation extraction strategies."""

    @abstractmethod
    async def extract(self, chunks: List[Dict]) -> Dict:
        """
        Extract entities and relations from chunks.

        Args:
            chunks: List of chunks from chunker

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'failed_chunks': [...]
            }
        """
        pass
```

**File**: `bigrag/interfaces/validator.py`

```python
from abc import ABC, abstractmethod
from typing import Dict

class ValidatorInterface(ABC):
    """Abstract interface for extraction validation strategies."""

    @abstractmethod
    async def validate(self, extractions: Dict) -> Dict:
        """
        Validate extractions (numeric coverage, semantic quality, etc.).

        Args:
            extractions: Output from ExtractorInterface.extract()

        Returns:
            {
                'entities': [...],       # Valid entities
                'relations': [...],      # Valid relations
                'failed_chunks': [...],  # Chunks that failed validation
                'summary': {
                    'status': 'PASS' | 'WARNING' | 'FAIL',
                    'numeric_coverage': 0.95,
                    'semantic_validity': 0.98
                }
            }
        """
        pass
```

**File**: `bigrag/interfaces/merger.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict

class MergerInterface(ABC):
    """Abstract interface for entity merging strategies."""

    @abstractmethod
    async def merge(self, entities: List[Dict]) -> List[Dict]:
        """
        Merge duplicate entities.

        Args:
            entities: List of entities from extractor

        Returns:
            List of merged entities (duplicates consolidated)
        """
        pass
```

**File**: `bigrag/interfaces/hitl.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class HITLInterface(ABC):
    """Abstract interface for Human-in-the-Loop strategies."""

    @abstractmethod
    async def save_failures(
        self,
        failed_chunks: List[Dict],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save failed extractions for human review.

        Args:
            failed_chunks: Chunks that failed validation
            metadata: Optional document metadata
        """
        pass
```

---

## Implementation Phases

### Phase 1: Setup Infrastructure (Day 1-2)

**Tasks**:
1. ✅ Create `bigrag/interfaces/` directory with 5 interface files
2. ✅ Create `bigrag/config.py` with IndexingConfig
3. ✅ Create `bigrag/factory.py` with StrategyFactory
4. ✅ Create `bigrag/indexer.py` with BiGRAG class skeleton
5. ✅ Create `bigrag/strategies/` directory structure

**Deliverables**:
- All interface files with docstrings
- IndexingConfig with 3 presets
- StrategyFactory with all build methods
- BiGRAG class with method signatures

**Status**: Ready to code (no dependencies)

---

### Phase 2: Implement Strategies (Day 3-7)

**Extract existing code into strategy classes**

#### Chunking Strategies (Day 3)

**File**: `bigrag/strategies/chunking/token.py`
```python
from bigrag.interfaces.chunker import ChunkerInterface

class TokenChunker(ChunkerInterface):
    """Token-based fixed-size chunking (fast, simple)."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk(self, text: str, metadata=None) -> List[Dict]:
        # Extract from smart_chunker.py (token chunking logic)
        # Archive original file
        pass
```

**File**: `bigrag/strategies/chunking/semantic.py`
```python
from bigrag.interfaces.chunker import ChunkerInterface

class SemanticChunker(ChunkerInterface):
    """Table-aware semantic chunking (slow, accurate)."""

    def __init__(self, api_key: str, chunk_size: int = 1200, overlap: int = 100):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Initialize table extractor
        from bigrag.preprocessors.table_extractor import GPT4TableExtractor
        self.table_extractor = GPT4TableExtractor(api_key=api_key)

    async def chunk(self, text: str, metadata=None) -> List[Dict]:
        # Extract from smart_chunker.py (semantic + table detection logic)
        # Archive original file
        pass
```

**File**: `bigrag/strategies/chunking/hybrid.py`
```python
class HybridChunker(ChunkerInterface):
    """Hybrid: detect tables first, then chunk remaining text."""
    # Combine TokenChunker + SemanticChunker logic
    pass
```

#### Extraction Strategies (Day 4-5)

**File**: `bigrag/strategies/extraction/strict.py`
```python
from bigrag.interfaces.extractor import ExtractorInterface

class StrictExtractor(ExtractorInterface):
    """Single-pass extraction without gleaning."""

    def __init__(self, api_key: str, enable_validation: bool = True):
        self.api_key = api_key
        self.enable_validation = enable_validation

        # Wrap existing ConstrainedLLMExtractor
        from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor
        self.llm_extractor = ConstrainedLLMExtractor(
            api_key=api_key,
            enable_gleaning=False,
            enable_numeric_validation=enable_validation
        )

    async def extract(self, chunks: List[Dict]) -> Dict:
        # Use existing extractor code
        # Archive original file
        pass
```

**File**: `bigrag/strategies/extraction/gleaning.py`
```python
class GleaningExtractor(ExtractorInterface):
    """Multi-pass extraction with conversation history."""

    def __init__(self, api_key: str, max_iterations: int = 2, enable_validation: bool = True):
        # Wrap existing ConstrainedLLMExtractor with gleaning=True
        pass
```

**File**: `bigrag/strategies/extraction/hybrid.py`
```python
class HybridExtractor(ExtractorInterface):
    """Tables use rule-based extraction, paragraphs use gleaning."""

    async def extract(self, chunks: List[Dict]) -> Dict:
        table_chunks = [c for c in chunks if c['type'] == 'table']
        paragraph_chunks = [c for c in chunks if c['type'] == 'paragraph']

        # Extract tables with TableFactExtractor
        # Extract paragraphs with GleaningExtractor
        # Combine results
        pass
```

#### Validation Strategies (Day 5-6)

**File**: `bigrag/strategies/validation/numeric.py`
```python
class NumericValidator(ValidatorInterface):
    """Validate numeric coverage using Gemini."""

    def __init__(self, api_key: str = None, strictness: str = "MODERATE"):
        # Wrap existing NumericValidator
        from bigrag.validators.numeric_validator import NumericValidator as OldValidator
        self.validator = OldValidator(api_key=api_key)
        self.strictness = strictness

    async def validate(self, extractions: Dict) -> Dict:
        # Extract logic from NumericValidator
        # Archive original file
        pass
```

**File**: `bigrag/strategies/validation/semantic.py`
```python
class SemanticValidator(ValidatorInterface):
    """Validate entity quality and relation completeness."""
    # Extract from entity_linker.py validation logic
    pass
```

**File**: `bigrag/strategies/validation/composite.py`
```python
class CompositeValidator(ValidatorInterface):
    """Run multiple validators in sequence."""

    def __init__(self, validators: List[ValidatorInterface]):
        self.validators = validators

    async def validate(self, extractions: Dict) -> Dict:
        result = extractions
        for validator in self.validators:
            result = await validator.validate(result)
        return result
```

**File**: `bigrag/strategies/validation/noop.py`
```python
class NoOpValidator(ValidatorInterface):
    """Skip validation (accept all extractions)."""

    async def validate(self, extractions: Dict) -> Dict:
        return {
            'entities': extractions['entities'],
            'relations': extractions['relations'],
            'failed_chunks': [],
            'summary': {
                'status': 'PASS',
                'numeric_coverage': 1.0,
                'semantic_validity': 1.0
            }
        }
```

#### Merging Strategies (Day 6)

**File**: `bigrag/strategies/merging/basic.py`
```python
class BasicMerger(MergerInterface):
    """Exact match merging only."""
    # Extract from unified_merger.py (basic mode)
    pass
```

**File**: `bigrag/strategies/merging/fuzzy.py`
```python
class FuzzyMerger(MergerInterface):
    """Fuzzy matching with edit distance + aliases."""
    # Extract from unified_merger.py (fuzzy mode)
    # Archive original file
    pass
```

**File**: `bigrag/strategies/merging/hybrid.py`
```python
class HybridMerger(MergerInterface):
    """Adaptive merging based on entity type."""
    # Extract from unified_merger.py (hybrid mode)
    pass
```

#### HITL Strategies (Day 7)

**File**: `bigrag/strategies/hitl/file.py`
```python
class FileHITL(HITLInterface):
    """Save failed extractions to JSON file."""
    # Extract from failed_extraction_store.py
    pass
```

**File**: `bigrag/strategies/hitl/noop.py`
```python
class NoOpHITL(HITLInterface):
    """Disable HITL (don't save failures)."""

    async def save_failures(self, failed_chunks, metadata=None):
        pass  # Do nothing
```

---

### Phase 3: Integrate & Test (Day 8-9)

**Tasks**:
1. Complete BiGRAG.index_document() implementation
2. Update API endpoint to use BiGRAG instead of EnhancedKGPipeline
3. Write unit tests for each strategy
4. Write integration tests for BiGRAG
5. Performance benchmarks (ensure no regression)

**Test Plan**:
```python
# Test each strategy independently
async def test_token_chunker():
    chunker = TokenChunker(chunk_size=1000, overlap=100)
    chunks = await chunker.chunk(test_text)
    assert len(chunks) > 0
    assert chunks[0]['type'] == 'paragraph'

# Test BiGRAG with different configs
async def test_bigrag_fast_preset():
    config = IndexingConfig.preset_fast(openai_api_key="...")
    rag = BiGRAG(config)
    result = await rag.index_document(test_text)
    assert result['statistics']['total_entities'] > 0

# Test strategy swapping
async def test_strategy_swap():
    config1 = IndexingConfig(chunker="token", ...)
    config2 = IndexingConfig(chunker="semantic", ...)

    rag1 = BiGRAG(config1)
    rag2 = BiGRAG(config2)

    result1 = await rag1.index_document(test_text)
    result2 = await rag2.index_document(test_text)

    # Different strategies should produce different results
    assert result1 != result2
```

---

### Phase 4: Archive & Cleanup (Day 10-11)

**Tasks**:
1. Move old pipeline files to `archive/`
2. Update all imports in `backend/` to use BiGRAG
3. Update documentation
4. Remove deprecated code references

**Archive Structure**:
```
archive/
├── pipelines/
│   ├── standard_pipeline.py
│   ├── production_pipeline.py
│   └── enhanced_pipeline.py
├── extractors/
│   ├── constrained_extractor.py
│   └── table_fact_extractor.py
├── merging/
│   ├── entity_linker.py
│   └── unified_merger.py
└── README.md  # Explains why these were archived
```

---

## API Endpoint Changes

### Before (Current)

```python
# backend/api/routes/unified_indexing.py
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures

features = PipelineFeatures(
    enable_gleaning=need_gleaning,
    enable_numeric_validation=need_numeric_validation,
    # ... 13 parameters
)

pipeline = EnhancedKGPipeline(features=features, dataset_path=expr_dir)
result = await pipeline.process_document(content_text, metadata)
```

### After (New)

```python
# backend/api/routes/unified_indexing.py
from bigrag import BiGRAG
from bigrag.config import IndexingConfig

config = IndexingConfig(
    # Strategies
    chunker="semantic" if need_table_extraction else "token",
    extractor="gleaning" if need_gleaning else "strict",
    validators=_build_validator_list(need_numeric_validation, need_semantic_validation),
    merger=merge_strategy,
    hitl="file" if enable_hitl else "noop",

    # Parameters
    gleaning_iterations=gleaning_iterations,
    validation_strictness=validation_strictness,
    enable_orphan_linking=enable_orphan_linking,

    # API Keys
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    gemini_api_key=os.getenv('GEMINI_API_KEY')
)

rag = BiGRAG(config, dataset_path=expr_dir)
result = await rag.index_document(content_text, metadata)
```

**Helper Function**:
```python
def _build_validator_list(numeric: bool, semantic: bool) -> List[str]:
    validators = []
    if numeric:
        validators.append('numeric')
    if semantic:
        validators.append('semantic')
    return validators
```

---

## Storage Structure (Unchanged)

All storage formats remain identical:
- ✅ GraphML: `graph_chunk_entity_relation.graphml`
- ✅ Vector DBs: `vdb_entities.json`, `vdb_relations.json`, `vdb_chunks.json`
- ✅ KV Stores: `kv_store_full_docs.json`, `kv_store_text_chunks.json`
- ✅ Registry: `subgraph_registry.json`

**No migration needed** - existing graphs work with new BiGRAG system.

---

## Benefits

### 1. Single Entry Point ✅
```python
from bigrag import BiGRAG  # That's it
```

### 2. Zero Code Duplication ✅
```
OLD: 2750 lines (60%+ duplication)
NEW: 1500 lines (0% duplication)
Savings: 45% fewer lines
```

### 3. True Modularity ✅
```python
# Swap strategies without code changes
config = IndexingConfig(
    chunker="my_custom_chunker",  # Your own implementation
    extractor="gleaning",
    merger="fuzzy"
)
```

### 4. Easy Testing ✅
```python
# Test each strategy in isolation
chunker = TokenChunker()
chunks = await chunker.chunk(test_text)
assert len(chunks) == expected_count
```

### 5. Clean Architecture ✅
```
BiGRAG (300 lines - orchestration only)
├── Strategies (1200 lines - business logic)
└── Interfaces (100 lines - contracts)

Total: 1600 lines, fully modular
```

---

## Migration Plan

### Week 1: Infrastructure
- Day 1-2: Create interfaces + config + factory

### Week 2: Strategies
- Day 3: Chunking strategies
- Day 4-5: Extraction strategies
- Day 5-6: Validation strategies
- Day 6: Merging strategies
- Day 7: HITL strategies

### Week 3: Integration
- Day 8-9: Complete BiGRAG + tests
- Day 10-11: Archive old code + cleanup

### Week 4: Documentation
- Update README.md
- Update API docs
- Create migration guide (for developers)

---

## Success Criteria

- ✅ Single BiGRAG class replaces all pipelines
- ✅ All 13 features working via IndexingConfig
- ✅ Zero code duplication
- ✅ All storage structures unchanged
- ✅ All tests passing
- ✅ Performance equivalent or better
- ✅ Old code archived (not deleted)

---

## Timeline

**Total**: 11 days focused work

**Breakdown**:
- Phase 1 (Infrastructure): 2 days
- Phase 2 (Strategies): 5 days
- Phase 3 (Integration): 2 days
- Phase 4 (Archive): 2 days

**Risk**: Low-Medium (parallel work possible, gradual integration)

**Recommendation**: **START Phase 1** - create interfaces and config (low risk, high value)
