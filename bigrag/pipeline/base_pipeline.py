"""
Unified modular knowledge graph pipeline.

Clean implementation using DIRECT IMPORTS (no wrapper modules).
Following MODULAR_PIPELINE_PLAN.md principle: "3 new files + imports = done!"
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from ..utils import logger
from .features import PipelineFeatures

# DIRECT IMPORTS from existing modules (no wrappers!)
from ..operate import chunking_by_token_size, extract_entities
from ..preprocessors.smart_chunker import TableAwareChunker
from ..preprocessors.table_extractor import GPT4TableExtractor
from ..extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
from ..validators.numeric_validator import NumericValidator
from ..merging.entity_linker import ProductionEntityLinker, SimpleEntityLinker, EntityCanonicalizationMap
from ..hitl.failed_extraction_store import FailedExtractionStore


class UnifiedPipeline:
    """
    Unified modular knowledge graph pipeline.

    Uses DIRECT IMPORTS from existing modules (no wrapper code).
    Only 3 new files: features.py, base_pipeline.py, quality_scoring.py

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
        1. Chunking - chunking_by_token_size() OR TableAwareChunker
        2. Extraction - ConstrainedLLMExtractor + BatchConstrainedExtractor
        3. Validation - NumericValidator (optional)
        4. Merging - SimpleEntityLinker OR ProductionEntityLinker
        5. Post-processing - Orphan detection (optional)

    Zero code duplication - everything imported from existing modules.
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

        # Initialize components based on features (DIRECT imports, no wrappers!)
        # IMPORTANT: Initialize hitl_store BEFORE extractor (extractor needs it)
        self.hitl_store = self._init_hitl() if self.features.enable_hitl else None
        self.chunker = self._init_chunker()
        self.extractor = self._init_extractor()
        self.validator = self._init_validator() if self._needs_validation() else None
        self.entity_linker = self._init_entity_linker()

    def _detect_preset(self) -> str:
        """Detect which preset was used (for logging)."""
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
        """Initialize chunker - DIRECT import, no wrapper."""
        if self.features.enable_table_detection:
            logger.info("[Unified Pipeline] Using TableAwareChunker (semantic chunking)")
            # Use existing TableAwareChunker DIRECTLY
            table_extractor = GPT4TableExtractor(api_key=self.features.openai_api_key)
            return TableAwareChunker(table_extractor=table_extractor)
        else:
            logger.info("[Unified Pipeline] Using token-based chunking (standard)")
            # Will use chunking_by_token_size() function directly in process_document()
            return None

    def _init_extractor(self):
        """Initialize extractor - DIRECT import, no wrapper."""
        logger.info(f"[Unified Pipeline] Using ConstrainedLLMExtractor (gleaning={self.features.enable_gleaning})")
        # Use existing ConstrainedLLMExtractor DIRECTLY
        return ConstrainedLLMExtractor(
            api_key=self.features.openai_api_key,
            model=self.llm_model,
            enable_gleaning=self.features.enable_gleaning,
            max_gleaning_iterations=self.features.max_gleaning_iterations,
            hitl_store=self.hitl_store if self.features.enable_hitl else None
        )

    def _init_validator(self):
        """Initialize validator - DIRECT import, no wrapper."""
        if not self._needs_validation():
            return None

        logger.info(f"[Unified Pipeline] Using NumericValidator (strictness={self.features.validation_strictness})")
        # Use existing NumericValidator DIRECTLY
        return NumericValidator(
            api_key=self.features.gemini_api_key,
            use_llm_validation=True if self.features.gemini_api_key else False
        )

    def _init_entity_linker(self):
        """Initialize entity linker - DIRECT import, no wrapper."""
        if self.features.merge_strategy == "fuzzy":
            logger.info("[Unified Pipeline] Using ProductionEntityLinker")
            # Use existing ProductionEntityLinker DIRECTLY
            canon_map = EntityCanonicalizationMap()
            return ProductionEntityLinker(
                canonicalization_map=canon_map,
                embedding_model=None,
                llm_func=None
            )
        else:
            logger.info("[Unified Pipeline] Using SimpleEntityLinker")
            # Use existing SimpleEntityLinker DIRECTLY
            canon_map = EntityCanonicalizationMap()
            return SimpleEntityLinker(canonicalization_map=canon_map)

    def _init_hitl(self):
        """Initialize HITL store - DIRECT import, no wrapper."""
        if not self.features.enable_hitl or not self.dataset_path:
            return None

        logger.info(f"[Unified Pipeline] HITL enabled: {self.dataset_path}")
        # Use existing FailedExtractionStore DIRECTLY
        return FailedExtractionStore(self.dataset_path)

    def _needs_validation(self) -> bool:
        """Check if any validation is enabled."""
        return (
            self.features.enable_numeric_validation or
            self.features.enable_entity_validation or
            self.features.enable_relation_validation
        )

    async def process_document(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process document through modular pipeline.

        Uses DIRECT function calls to existing modules (no wrappers).

        Args:
            content: Document text
            metadata: Optional metadata (title, category, tags, etc.)

        Returns:
            dict: {
                'chunks': List[Dict],
                'entities': List[Dict],
                'relations': List[Dict],
                'validation': Dict,
                'statistics': Dict,
                'pipeline_metadata': Dict
            }
        """
        metadata = metadata or {}
        logger.info("[Pipeline] ========== Starting document processing ==========")

        try:
            # Step 1: Chunking (REQUIRED)
            logger.info("[Pipeline] Step 1: Chunking...")
            chunks = await self._chunk_document(content, metadata)
            logger.info(f"[Pipeline] Created {len(chunks)} chunks")

            # Step 2: Extraction (REQUIRED)
            logger.info("[Pipeline] Step 2: Extraction...")
            entities, relations = await self._extract_entities_relations(chunks)
            logger.info(f"[Pipeline] Extracted {len(entities)} entities, {len(relations)} relations")

            # Step 3: Validation (OPTIONAL)
            validation_report = {'status': 'SKIPPED', 'message': 'Validation disabled'}
            if self.validator:
                logger.info("[Pipeline] Step 3: Validation...")
                entities, relations, validation_report = await self._validate(entities, relations, chunks)
                logger.info(f"[Pipeline] Validation: {validation_report.get('status', 'UNKNOWN')}")

            # Step 4: Merging (REQUIRED)
            logger.info(f"[Pipeline] Step 4: Merging (strategy: {self.features.merge_strategy})...")
            entities = await self._merge_entities(entities, relations)
            logger.info(f"[Pipeline] Merged to {len(entities)} unique entities")

            # Step 4.5: Add hyper-relation bidirectional linking (REQUIRED)
            self._add_hyper_relations(entities, relations)

            # Step 5: Post-processing (OPTIONAL)
            if self.features.enable_orphan_linking:
                logger.info("[Pipeline] Step 5: Post-processing (orphan detection)...")
                orphan_count = self._detect_orphans(entities, relations)
                logger.info(f"[Pipeline] Found {orphan_count} orphan entities")

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

    async def _chunk_document(self, content: str, metadata: Dict) -> List[Dict]:
        """Chunk document using selected strategy (DIRECT function calls)."""
        if self.chunker:
            # Use TableAwareChunker DIRECTLY with correct parameters
            chunks = await self.chunker.chunk_document(
                markdown_text=content,
                chunk_size=self.features.chunk_size,
                overlap=self.features.chunk_overlap,
                metadata=metadata,
                use_semantic_chunking=(self.features.chunk_mode == "semantic")
            )
            # Normalize format
            result_chunks = []
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    if 'chunk_order_index' not in chunk:
                        chunk['chunk_order_index'] = i
                    result_chunks.append(chunk)
                else:
                    result_chunks.append({
                        'content': str(chunk),
                        'metadata': metadata,
                        'chunk_order_index': i
                    })
            return result_chunks
        else:
            # Use chunking_by_token_size() function DIRECTLY from operate.py
            chunk_texts = chunking_by_token_size(
                content,
                max_token_size=self.features.chunk_size,
                overlap_token_size=self.features.chunk_overlap,
                tiktoken_model=self.llm_model,
                doc_title=metadata.get('title', ''),
                doc_metadata=metadata
            )
            # Convert to dict format
            return [
                {
                    'content': chunk,
                    'metadata': metadata,
                    'chunk_order_index': i
                }
                for i, chunk in enumerate(chunk_texts)
            ]

    async def _extract_entities_relations(self, chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relations (DIRECT use of existing extractors)."""
        # Use BatchConstrainedExtractor DIRECTLY
        batch_extractor = BatchConstrainedExtractor(self.extractor)

        # Ensure chunks have required fields (chunk_id, content, metadata)
        normalized_chunks = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                # Ensure chunk_id exists (use 4 digits to match TableAwareChunker)
                if 'chunk_id' not in chunk:
                    chunk['chunk_id'] = f"chunk_{i:04d}"

                # Ensure content is a string (not dict)
                content = chunk.get('content', chunk.get('text', ''))
                if isinstance(content, dict):
                    # If content is still a dict, try to extract text
                    content = str(content.get('text', content.get('content', str(content))))
                chunk['content'] = str(content)

                # Ensure metadata exists
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                normalized_chunks.append(chunk)
            else:
                normalized_chunks.append({
                    'chunk_id': f"chunk_{i:04d}",
                    'content': str(chunk),
                    'metadata': {}
                })

        # NEW: Table fact extraction (if enabled)
        # Process table chunks separately for 0% hallucination
        if self.features.enable_table_fact_extraction:
            from ..extractors.table_fact_extractor import TableFactExtractor

            table_chunks = []
            text_chunks = []

            for chunk in normalized_chunks:
                chunk_metadata = chunk.get('metadata', {})
                if chunk_metadata.get('contains_table') and chunk_metadata.get('table_data'):
                    table_chunks.append(chunk)
                else:
                    text_chunks.append(chunk)

            logger.info(f"[Pipeline] Extracting from {len(table_chunks)} table chunks + {len(text_chunks)} text chunks")
        else:
            text_chunks = normalized_chunks
            table_chunks = []

        # Call existing batch extraction method DIRECTLY for text chunks
        if text_chunks:
            result = await batch_extractor.extract_from_chunks(
                chunks=text_chunks,
                language="English"
            )
        else:
            result = {'extractions': []}

        # NEW: Extract from table chunks using rule-based approach
        table_extraction_count = 0
        if self.features.enable_table_fact_extraction and table_chunks:
            from ..extractors.table_fact_extractor import TableFactExtractor

            for table_chunk in table_chunks:
                chunk_id = table_chunk.get('chunk_id', '')
                table_data = table_chunk.get('metadata', {}).get('table_data')

                if table_data:
                    try:
                        facts = TableFactExtractor.extract_facts_from_table(
                            table_data=table_data,
                            chunk_id=chunk_id
                        )
                        # Add table extraction to result
                        result['extractions'].append({
                            'chunk_id': chunk_id,
                            'entities': facts['entities'],
                            'relations': facts['relations']
                        })
                        table_extraction_count += 1
                    except Exception as e:
                        logger.warning(f"[Pipeline] Table extraction failed for {chunk_id}: {e}")

            logger.info(f"[Pipeline] Extracted facts from {table_extraction_count} table chunks")

        # Aggregate results from extractions and add required fields
        # (Following production_pipeline.py pattern: lines 213-246)
        all_entities = []
        all_relations = []

        from bigrag.utils import compute_mdhash_id
        from bigrag.constants import ENTITY_PREFIX, RELATION_PREFIX

        for extraction in result.get('extractions', []):
            chunk_id = extraction.get('chunk_id', '')

            # Add source_id and entity_id to entities (production pipeline pattern)
            for entity in extraction.get('entities', []):
                if 'source_id' not in entity:
                    entity['source_id'] = chunk_id
                if 'metadata' not in entity:
                    entity['metadata'] = {}

                # Track extraction method (unified_pipeline or table_row)
                if 'extraction_method' not in entity.get('metadata', {}):
                    entity['metadata']['extraction_method'] = 'unified_pipeline'

                # Generate stable entity ID if not present
                if 'entity_id' not in entity:
                    entity_id = compute_mdhash_id(entity['entity_name'], prefix=ENTITY_PREFIX)
                    entity['entity_id'] = entity_id

            # Add source_id, relation_id, and linked_entities to relations (production pipeline pattern)
            for relation in extraction.get('relations', []):
                if 'source_id' not in relation:
                    relation['source_id'] = chunk_id
                if 'metadata' not in relation:
                    relation['metadata'] = {}

                # Track extraction method
                if 'extraction_method' not in relation.get('metadata', {}):
                    relation['metadata']['extraction_method'] = 'unified_pipeline'

                # Generate relation ID if not present
                if 'relation_id' not in relation:
                    relation_id = compute_mdhash_id(relation.get('content', ''), prefix=RELATION_PREFIX)
                    relation['relation_id'] = relation_id

                # Extract linked entities from relation content
                # Skip if already populated (e.g., by TableFactExtractor)
                if 'linked_entities' not in relation.get('metadata', {}):
                    linked_entities = []
                    for entity in extraction.get('entities', []):
                        # Simple heuristic: if entity name appears in relation content
                        if entity['entity_name'] in relation.get('content', ''):
                            linked_entities.append(entity['entity_id'])
                    relation['metadata']['linked_entities'] = linked_entities

            all_entities.extend(extraction.get('entities', []))
            all_relations.extend(extraction.get('relations', []))

        return all_entities, all_relations

    async def _validate(
        self,
        entities: List[Dict],
        relations: List[Dict],
        chunks: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """Validate entities and relations (DIRECT use of existing validator)."""
        validation_report = {
            'status': 'PASSED',
            'original_entities': len(entities),
            'original_relations': len(relations),
            'warnings': []
        }

        # NEW: Full numeric validation using NumericValidator.validate_extraction()
        if self.validator and self.features.enable_numeric_validation:
            try:
                logger.info(f"[Validation] Running numeric validation (level: {self.features.validation_strictness})")

                # Reconstruct source document from chunks
                source_document = '\n\n'.join([
                    chunk.get('content', '') for chunk in chunks
                ])

                # Call full validation method
                numeric_result = await self.validator.validate_extraction(
                    source_document=source_document,
                    entities=entities,
                    relations=relations,
                    validation_level=self.features.validation_strictness
                )

                validation_report['numeric_validation'] = numeric_result

                # Update overall status based on numeric validation
                if numeric_result.get('status') == 'FAIL':
                    validation_report['status'] = 'FAILED'
                    validation_report['warnings'].append(
                        f"Numeric validation failed: {numeric_result.get('message', 'Unknown error')}"
                    )
                elif numeric_result.get('status') == 'WARNING':
                    validation_report['warnings'].append(
                        f"Numeric validation warning: {numeric_result.get('message', 'See details')}"
                    )

                logger.info(f"[Validation] Numeric validation: {numeric_result.get('status', 'UNKNOWN')}")

            except Exception as e:
                logger.warning(f"[Validation] Numeric validation failed: {e}")
                validation_report['warnings'].append(f"Numeric validation error: {str(e)}")
                validation_report['numeric_validation'] = {
                    'status': 'ERROR',
                    'message': str(e)
                }

        # Entity quality filtering (if enabled)
        if self.features.enable_entity_validation:
            original_count = len(entities)
            entities = self._filter_low_quality_entities(entities)
            filtered = original_count - len(entities)
            validation_report['filtered_entities'] = filtered
            logger.info(f"[Validation] Filtered {filtered} low-quality entities")

        # Relation quality filtering (if enabled)
        if self.features.enable_relation_validation:
            original_count = len(relations)
            relations = self._filter_incomplete_relations(relations)
            filtered = original_count - len(relations)
            validation_report['filtered_relations'] = filtered
            logger.info(f"[Validation] Filtered {filtered} incomplete relations")

        validation_report['final_entities'] = len(entities)
        validation_report['final_relations'] = len(relations)

        return entities, relations, validation_report

    def _filter_low_quality_entities(self, entities: List[Dict]) -> List[Dict]:
        """Filter low-quality entities (inline logic, no wrapper)."""
        from ..pipeline.features import VALIDATION_THRESHOLDS
        thresholds = VALIDATION_THRESHOLDS.get(
            self.features.validation_strictness,
            VALIDATION_THRESHOLDS["MODERATE"]
        )

        filtered = []
        generic_terms = ['thing', 'stuff', 'entity', 'item', 'object', 'element']

        for entity in entities:
            name = entity.get('entity_name', '').strip()
            if len(name) >= thresholds['entity_name_min_length'] and name.lower() not in generic_terms:
                filtered.append(entity)

        return filtered

    def _filter_incomplete_relations(self, relations: List[Dict]) -> List[Dict]:
        """Filter incomplete relations (inline logic, no wrapper)."""
        from ..pipeline.features import VALIDATION_THRESHOLDS
        thresholds = VALIDATION_THRESHOLDS.get(
            self.features.validation_strictness,
            VALIDATION_THRESHOLDS["MODERATE"]
        )

        filtered = []
        for relation in relations:
            # Relations use 'content' field (not 'description')
            content = relation.get('content', '').strip()
            if len(content) >= thresholds['relation_description_min_length']:
                filtered.append(relation)

        return filtered

    async def _merge_entities(self, entities: List[Dict], relations: List[Dict]) -> List[Dict]:
        """Merge entities (DIRECT use of existing entity linkers)."""
        if not entities:
            return []

        # Use entity linker DIRECTLY
        if self.features.merge_strategy == "fuzzy":
            # ProductionEntityLinker
            merged = await self.entity_linker.link_entities_across_chunks(entities)
        else:
            # SimpleEntityLinker - just do basic dedup
            merged = await self._simple_dedup(entities)

        return merged

    async def _simple_dedup(self, entities: List[Dict]) -> List[Dict]:
        """Simple hash-based deduplication (inline, no wrapper)."""
        seen = {}
        merged = []

        for entity in entities:
            name = entity.get('entity_name', '').strip().lower()
            if name not in seen:
                seen[name] = entity
                merged.append(entity)
            else:
                # Merge weights
                seen[name]['weight'] = seen[name].get('weight', 1.0) + entity.get('weight', 1.0)

        return merged

    def _add_hyper_relations(self, entities: List[Dict], relations: List[Dict]) -> None:
        """
        Add hyper_relation to entities (bidirectional linking).

        Following production_pipeline.py pattern (lines 299-327).
        This creates a reverse mapping from entities to relations.

        IMPORTANT: After entity merging, entity_ids change. We need to remap
        old entity_ids to new merged entity_ids using entity_ids_merged field.
        """
        # Build entity lookup dict AND entity ID remapping
        entity_lookup = {}
        entity_id_remap = {}  # old_id -> new_id mapping

        for entity in entities:
            entity_id = entity.get('entity_id')
            if entity_id:
                entity_lookup[entity_id] = entity

                # If this entity was created by merging, map all old IDs to new ID
                merged_ids = entity.get('entity_ids_merged', [])
                if merged_ids:
                    for old_id in merged_ids:
                        entity_id_remap[old_id] = entity_id

        hyper_relation_added = 0
        for relation in relations:
            relation_id = relation.get('relation_id')
            if not relation_id:
                # Generate relation_id if somehow missing
                from bigrag.utils import compute_mdhash_id
                from bigrag.constants import RELATION_PREFIX
                relation_id = compute_mdhash_id(relation.get('content', ''), prefix=RELATION_PREFIX)
                relation['relation_id'] = relation_id

            linked_entities = relation.get('metadata', {}).get('linked_entities', [])

            for old_entity_id in linked_entities:
                # Remap old entity ID to merged entity ID
                new_entity_id = entity_id_remap.get(old_entity_id, old_entity_id)

                if new_entity_id in entity_lookup:
                    entity_lookup[new_entity_id]['hyper_relation'] = relation_id
                    hyper_relation_added += 1

        logger.info(f"[Pipeline] Added hyper_relation to {hyper_relation_added} entity references")

    def _detect_orphans(self, entities: List[Dict], relations: List[Dict]) -> int:
        """
        Detect orphan entities using hyper_relation field (production pipeline pattern).

        Orphan entities are those without any relation connection.
        """
        # Check hyper_relation instead of name matching (production pipeline: line 322)
        orphans = [e for e in entities if not e.get('hyper_relation')]
        orphan_count = len(orphans)
        orphan_ratio = orphan_count / len(entities) if entities else 0

        if orphan_ratio > 0.1:
            logger.warning(f"[Orphan Detection] {orphan_count} orphans ({orphan_ratio:.1%}) - consider improving extraction")
        else:
            logger.info(f"[Orphan Detection] {orphan_count} orphans ({orphan_ratio:.1%})")

        return orphan_count
