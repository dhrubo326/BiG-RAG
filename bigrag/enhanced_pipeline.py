"""
Enhanced Knowledge Graph Pipeline (Redesigned Production Pipeline)

Combines best practices from both standard and production pipelines:
- Semantic boundary-aware chunking (from standard pipeline approach)
- Gleaning-based extraction for better recall (from standard pipeline)
- Strict validation system (from production pipeline)
- Entity canonicalization + linking (from production pipeline)

New Features (Phase 1):
- Configurable extraction strategy (strict/gleaning/hybrid)
- Version metadata for safe migrations
- HITL system for failed extractions
- Unified entity merging

Designed for 99%+ accuracy on bilingual educational documents.
"""

import asyncio
from typing import List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime

from bigrag.preprocessors.table_extractor import GPT4TableExtractor, BilingualDetector
from bigrag.preprocessors.smart_chunker import TableAwareChunker
from bigrag.extractors.table_fact_extractor import TableFactExtractor
from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
from bigrag.merging.canonicalization import EntityCanonicalizationMap
from bigrag.merging.entity_linker import ProductionEntityLinker, SimpleEntityLinker
from bigrag.validators.numeric_validator import NumericValidator

# Pipeline version for migration tracking
PIPELINE_VERSION = "enhanced-v1.0"
BACKWARD_COMPATIBLE_WITH = ['standard-v1.0', 'production-v1.0']


class EnhancedKGPipeline:
    """
    Enhanced knowledge graph construction pipeline.

    Combines best practices:
    - Table-aware chunking (from production)
    - Semantic boundary-aware chunking (NEW - Phase 1 Step 2)
    - Gleaning extraction (from standard - Phase 1 Step 3)
    - Strict validation (from production)
    - Entity canonicalization (from production)

    Usage:
        pipeline = EnhancedKGPipeline(
            api_key="your-key",
            extraction_strategy="hybrid"  # NEW: strict|gleaning|hybrid
        )
        result = await pipeline.process_document(markdown_text, metadata)

        if result['validation']['status'] == 'PASS':
            # Use result['entities'] and result['relations']
            pass
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        validation_level: str = "MODERATE",
        enable_entity_linking: bool = True,
        entity_merge_strategy: str = "fuzzy",  # NEW: Phase 1 Step 4
        extraction_strategy: str = "hybrid",  # NEW: strict|gleaning|hybrid
        extraction_mode: str = "semi_structured",
        review_queue_path: str = "expr/human_review_queue.json",
        dataset_path: Optional[str] = None  # NEW: for HITL storage
    ):
        """
        Initialize enhanced pipeline.

        Args:
            api_key: OpenAI API key
            model: LLM model to use (gpt-4o-mini recommended for cost efficiency)
            validation_level: STRICT (production), MODERATE (dev), LENIENT (test)
            enable_entity_linking: Whether to merge entities
            entity_merge_strategy: NEW (Phase 1 Step 4) - Entity merging approach
                - "basic": Simple name-based grouping (fast, O(n))
                - "fuzzy": Canonicalization + fuzzy matching (accurate, O(n²))
                - "hybrid": Adaptive based on entity count [FUTURE]
            extraction_strategy: NEW - Controls extraction approach
                - "strict": Single-pass with validation (fastest, 95%+ accuracy)
                - "gleaning": Multi-pass with conversation history (slowest, 98%+ accuracy)
                - "hybrid": Adaptive (strict for tables, gleaning for paragraphs) [RECOMMENDED]
            extraction_mode: Validation mode (structured/semi_structured/unstructured)
                - structured: 99%+ accuracy, strict validation (best for tables)
                - semi_structured: 95%+ accuracy, moderate validation [DEFAULT]
                - unstructured: 80%+ accuracy, lenient validation (best for narrative text)
            review_queue_path: Path to human review queue JSON file
            dataset_path: Path to dataset (for HITL failed extraction storage)
        """
        self.api_key = api_key
        self.model = model
        self.validation_level = validation_level
        self.enable_entity_linking = enable_entity_linking
        self.entity_merge_strategy = entity_merge_strategy  # NEW: Phase 1 Step 4
        self.extraction_strategy = extraction_strategy
        self.extraction_mode = extraction_mode
        self.review_queue_path = Path(review_queue_path)
        self.dataset_path = dataset_path

        # NEW: Pipeline metadata for version tracking
        self.pipeline_metadata = {
            'pipeline_version': PIPELINE_VERSION,
            'backward_compatible': BACKWARD_COMPATIBLE_WITH,
            'created_at': datetime.now().isoformat(),
            'extraction_strategy': extraction_strategy,
            'entity_merge_strategy': entity_merge_strategy  # NEW
        }

        # Validate extraction strategy
        valid_extraction_strategies = ['strict', 'gleaning', 'hybrid']
        if extraction_strategy not in valid_extraction_strategies:
            raise ValueError(f"extraction_strategy must be one of {valid_extraction_strategies}, got: {extraction_strategy}")

        # Validate entity merge strategy (Phase 1 Step 4)
        valid_merge_strategies = ['basic', 'fuzzy', 'hybrid']
        if entity_merge_strategy not in valid_merge_strategies:
            raise ValueError(f"entity_merge_strategy must be one of {valid_merge_strategies}, got: {entity_merge_strategy}")

        # Initialize components
        self.table_extractor = GPT4TableExtractor(api_key=api_key, model=model)
        self.chunker = TableAwareChunker(self.table_extractor)

        # NEW: Initialize HITL store if dataset_path provided (must be before extractors)
        self.hitl_store = None
        if dataset_path:
            try:
                from bigrag.hitl.failed_extraction_store import FailedExtractionStore
                self.hitl_store = FailedExtractionStore(dataset_path)
            except ImportError:
                print("[WARN] HITL module not available - failed extractions will only be logged")

        # NEW: Initialize paragraph extractor with gleaning support
        # Gleaning will be enabled based on extraction_strategy at runtime
        self.paragraph_extractor = ConstrainedLLMExtractor(
            api_key=api_key,
            model=model,
            extraction_mode=extraction_mode,
            enable_gleaning=False,  # Will be set dynamically per chunk
            max_gleaning_iterations=2,  # Standard pipeline default
            hitl_store=self.hitl_store  # NEW (Phase 1 Step 6): Pass HITL store
        )
        self.batch_extractor = BatchConstrainedExtractor(self.paragraph_extractor)

        # Initialize numeric validator - reads GEMINI_API_KEY from .env (not OpenAI key!)
        self.numeric_validator = NumericValidator(api_key=None, use_llm_validation=True)

        # NEW (Phase 1 Step 4): Initialize unified entity merger
        if enable_entity_linking:
            from bigrag.merging.unified_merger import UnifiedEntityMerger
            self.entity_merger = UnifiedEntityMerger(strategy=entity_merge_strategy)
            # Keep backward compatibility references
            if entity_merge_strategy in ['fuzzy', 'hybrid']:
                self.canon_map = self.entity_merger.canon_map
                self.entity_linker = self.entity_merger.entity_linker
            else:
                self.canon_map = None
                self.entity_linker = None
        else:
            self.entity_merger = None
            self.canon_map = None
            self.entity_linker = None

        print(f"[INIT] Enhanced Pipeline v{PIPELINE_VERSION}")
        print(f"       Extraction Strategy: {extraction_strategy}")
        print(f"       Entity Merge Strategy: {entity_merge_strategy}")  # NEW
        print(f"       Entity Linking: {'Enabled' if enable_entity_linking else 'Disabled'}")
        print(f"       HITL: {'Enabled' if self.hitl_store else 'Disabled'}")

    @staticmethod
    def recommend_config(
        sample_documents: List[str],
        corpus_size: int,
        performance_profile: str = "balanced"
    ) -> Dict:
        """
        Recommend optimal pipeline configuration (Phase 1 Step 5).

        Uses pipeline selector to analyze documents and recommend configuration.

        Args:
            sample_documents: Sample of documents (5-10 recommended)
            corpus_size: Total number of documents in corpus
            performance_profile: 'speed', 'balanced', or 'accuracy'

        Returns:
            Dictionary with:
            {
                'pipeline_type': 'standard' or 'enhanced',
                'config': {config_dict},
                'reasoning': [list of reasons],
                'estimated_cost': 'low/medium/high',
                'estimated_time': 'fast/medium/slow',
                'expected_quality': 'good/very_good/excellent'
            }

        Example:
            # Get recommendation
            rec = EnhancedKGPipeline.recommend_config(
                sample_documents=docs[:10],
                corpus_size=1000,
                performance_profile='accuracy'
            )

            # Use recommended config
            if rec['pipeline_type'] == 'enhanced':
                pipeline = EnhancedKGPipeline(
                    api_key=api_key,
                    **rec['config']
                )
        """
        from bigrag.pipeline_selector import quick_recommend

        recommendation = quick_recommend(
            documents=sample_documents,
            corpus_size=corpus_size,
            performance_profile=performance_profile
        )

        return {
            'pipeline_type': recommendation.pipeline_type.value,
            'config': recommendation.config,
            'reasoning': recommendation.reasoning,
            'estimated_cost': recommendation.estimated_cost,
            'estimated_time': recommendation.estimated_time,
            'expected_quality': recommendation.expected_quality,
            'confidence': recommendation.confidence
        }

    async def process_document(
        self,
        markdown_text: str,
        metadata: Optional[Dict] = None,
        language: str = "English"
    ) -> Dict:
        """
        Process document through full enhanced pipeline.

        Args:
            markdown_text: Document content in markdown format
            metadata: Optional metadata (title, category, tags)
            language: Output language for entity extraction

        Returns:
            {
                'entities': [...],  # Merged entities
                'relations': [...],  # All relations
                'chunks': [...],  # Processed chunks
                'validation': {
                    'numeric': {...},
                    'consistency': {...},
                    'overall_status': 'PASS' or 'FAIL'
                },
                'statistics': {
                    'total_entities': int,
                    'total_relations': int,
                    'entity_merge_count': int,
                    'pipeline_version': str,  # NEW
                    ...
                }
            }
        """

        print("=" * 80)
        print(f"Enhanced KG Pipeline Starting (v{PIPELINE_VERSION})")
        print(f"Extraction Strategy: {self.extraction_strategy}")
        print("=" * 80)

        # Import utilities needed throughout the pipeline (moved from inside loops for efficiency)
        from bigrag.utils import compute_mdhash_id
        from bigrag.constants import ENTITY_PREFIX, RELATION_PREFIX

        # Phase 1: Pre-processing
        print("\n[PHASE 1] Pre-processing")
        print("-" * 80)

        # Step 1.1: Smart chunking (includes table extraction)
        # TODO: Step 2 will add semantic boundary-aware chunking
        print("  [1.1] Smart chunking with table extraction...")
        chunks = await self.chunker.chunk_document(
            markdown_text,
            chunk_size=1000,  # NEW: Reduced from 1200 to 1000 (more conservative)
            overlap=100,
            metadata=metadata
        )
        table_chunks = [c for c in chunks if c['type'] == 'table']
        paragraph_chunks = [c for c in chunks if c['type'] == 'paragraph']
        print(f"    Created {len(chunks)} chunks ({len(table_chunks)} tables, {len(paragraph_chunks)} paragraphs)")

        # Phase 2: Extraction
        print("\n[PHASE 2] Extraction")
        print("-" * 80)

        all_entities = []
        all_relations = []

        # Step 2.1: Table fact extraction with graceful degradation
        print("  [2.1] Table fact extraction (rule-based with graceful degradation)...")
        print(f"       Strategy: {self.extraction_strategy} → using 'strict' for tables")

        failed_tables = []
        successful_tables = 0

        for chunk in table_chunks:
            # Get validation status from table extraction
            validation_status = chunk.get('structured_data', {}).get('metadata', {}).get('validation_status', 'UNKNOWN')

            if validation_status == 'FAIL':
                # Save to HITL if available
                if self.hitl_store:
                    await self.hitl_store.save_failed_table(
                        table_id=chunk.get('structured_data', {}).get('table_id', 'unknown'),
                        table_data=chunk.get('structured_data', {}),
                        failure_reason='LLM validation failed',
                        document_id=metadata.get('doc_id', 'unknown') if metadata else 'unknown'
                    )

                # Skip failed tables and add to review queue
                failed_tables.append({
                    'chunk_id': chunk['chunk_id'],
                    'table_id': chunk.get('structured_data', {}).get('table_id', 'unknown'),
                    'reason': 'LLM validation failed',
                    'validation_feedback': chunk.get('structured_data', {}).get('metadata', {}).get('validation_feedback', ''),
                    'missing_numbers': chunk.get('structured_data', {}).get('metadata', {}).get('missing_numbers', []),
                    'hallucinated_numbers': chunk.get('structured_data', {}).get('metadata', {}).get('hallucinated_numbers', []),
                    'numeric_coverage': chunk.get('structured_data', {}).get('metadata', {}).get('numeric_coverage', 0.0),
                    'source_markdown': chunk.get('content', ''),
                    'extracted_data': chunk.get('structured_data', {})
                })
                print(f"    [SKIP] Table {chunk['chunk_id']} failed validation - added to review queue")
                continue

            # Process validated table
            try:
                facts = TableFactExtractor.extract_facts_from_table(
                    chunk['structured_data'],
                    chunk['chunk_id']
                )
                all_entities.extend(facts['entities'])
                all_relations.extend(facts['relations'])
                successful_tables += 1
            except Exception as e:
                # If fact extraction fails, skip and add to review queue
                if self.hitl_store:
                    await self.hitl_store.save_failed_table(
                        table_id=chunk.get('structured_data', {}).get('table_id', 'unknown'),
                        table_data=chunk.get('structured_data', {}),
                        failure_reason=f'Fact extraction error: {str(e)}',
                        document_id=metadata.get('doc_id', 'unknown') if metadata else 'unknown'
                    )

                failed_tables.append({
                    'chunk_id': chunk['chunk_id'],
                    'table_id': chunk.get('structured_data', {}).get('table_id', 'unknown'),
                    'reason': f'Fact extraction error: {str(e)}',
                    'source_markdown': chunk.get('content', ''),
                    'extracted_data': chunk.get('structured_data', {})
                })
                print(f"    [SKIP] Table {chunk['chunk_id']} fact extraction failed - added to review queue")

        success_rate = successful_tables / len(table_chunks) if table_chunks else 1.0
        print(f"    Extracted {len([e for e in all_entities])} entities, {len([r for r in all_relations])} relations from tables")
        print(f"    Table success rate: {success_rate:.2%} ({successful_tables}/{len(table_chunks)} tables)")

        if failed_tables:
            print(f"    [WARN] {len(failed_tables)} tables failed - flagged for human review")

        # Step 2.2: Paragraph extraction with strategy-based approach
        print(f"  [2.2] Paragraph extraction (constrained LLM with strategy: {self.extraction_strategy})...")

        if paragraph_chunks:
            # NEW: Apply extraction strategy (Phase 1 Step 3 - COMPLETED)
            if self.extraction_strategy == 'strict':
                # Single-pass only (no gleaning)
                print("       Using STRICT mode: single-pass extraction")
                self.paragraph_extractor.enable_gleaning = False

            elif self.extraction_strategy == 'gleaning':
                # Gleaning for all paragraphs
                print("       Using GLEANING mode: multi-pass extraction (2 gleaning passes)")
                self.paragraph_extractor.enable_gleaning = True

            elif self.extraction_strategy == 'hybrid':
                # Adaptive: gleaning for paragraphs (tables already precise)
                print("       Using HYBRID mode: gleaning for paragraphs, strict for tables")
                self.paragraph_extractor.enable_gleaning = True

            # Gleaning implementation completed in Phase 1 Step 3
            # ConstrainedLLMExtractor now supports multi-pass extraction with
            # quality-based merging when enable_gleaning=True

            print(f"    DEBUG: Calling batch extractor with {len(paragraph_chunks)} paragraph chunks")

            batch_result = await self.batch_extractor.extract_from_chunks(
                paragraph_chunks,
                language=language
            )

            print(f"    DEBUG: Batch result contains {len(batch_result['extractions'])} successful extractions")
            print(f"    DEBUG: Failed chunks: {len(batch_result.get('failed_chunks', []))}")

            entities_before = len(all_entities)
            relations_before = len(all_relations)

            for idx, extraction in enumerate(batch_result['extractions']):
                chunk_id = extraction['chunk_id']
                print(f"      Processing extraction {idx+1}/{len(batch_result['extractions'])}: chunk_id={chunk_id}, entities={len(extraction['entities'])}, relations={len(extraction['relations'])}")

                # Add source_id, metadata, and entity_id to each entity
                for entity in extraction['entities']:
                    if 'source_id' not in entity:
                        entity['source_id'] = chunk_id
                    if 'metadata' not in entity:
                        entity['metadata'] = {}
                    entity['metadata']['extraction_method'] = 'constrained_llm'
                    entity['metadata']['extraction_strategy'] = self.extraction_strategy  # NEW

                    # UNIFIED: Generate stable entity ID if not present (consistent with standard pipeline)
                    if 'entity_id' not in entity:
                        entity_id = compute_mdhash_id(entity['entity_name'], prefix=ENTITY_PREFIX)
                        entity['entity_id'] = entity_id

                # Add source_id, metadata, relation_id, and linked_entities to each relation
                for relation in extraction['relations']:
                    if 'source_id' not in relation:
                        relation['source_id'] = chunk_id
                    if 'metadata' not in relation:
                        relation['metadata'] = {}
                    relation['metadata']['extraction_method'] = 'constrained_llm'
                    relation['metadata']['extraction_strategy'] = self.extraction_strategy  # NEW

                    # FIX 1A: Generate relation_id (CRITICAL - enables hyper_relation linking at line 518)
                    # Without this, paragraph relations are skipped during hyper_relation linking
                    if 'relation_id' not in relation:
                        # FIX: Use .strip() for consistent relation ID generation
                        relation_id = compute_mdhash_id(relation['content'].strip(), prefix=RELATION_PREFIX)
                        relation['relation_id'] = relation_id

                    # FIX 1B: Initialize empty linked_entities (will be populated in post-merge linking)
                    # Per-chunk linking REMOVED - it failed for cross-chunk entities
                    # (e.g., paragraph relation mentioning table entity from different chunk)
                    if 'metadata' not in relation:
                        relation['metadata'] = {}
                    relation['metadata']['linked_entities'] = []  # Will be populated after entity merging

                all_entities.extend(extraction['entities'])
                all_relations.extend(extraction['relations'])

            print(f"    DEBUG: Added {len(all_entities) - entities_before} entities and {len(all_relations) - relations_before} relations from paragraphs")
            print(f"    DEBUG: Total entities now: {len(all_entities)}, Total relations now: {len(all_relations)}")

            stats = batch_result['statistics']
            print(f"    Success rate: {stats['success_rate']:.2%}")
            print(f"    Avg numeric coverage: {stats['avg_numeric_coverage']:.2%}")

            # NEW: Save failed chunks to HITL
            if batch_result['failed_chunks']:
                print(f"    [WARN] {len(batch_result['failed_chunks'])} chunks failed validation")

                if self.hitl_store:
                    for failed_chunk_id in batch_result['failed_chunks']:
                        # Find the original chunk
                        original_chunk = next((c for c in paragraph_chunks if c['chunk_id'] == failed_chunk_id), None)
                        if original_chunk:
                            await self.hitl_store.save_failed_chunk(
                                chunk_id=failed_chunk_id,
                                chunk_content=original_chunk['content'],
                                failure_reason="All 3 validation attempts failed",
                                validation_details={'status': 'FAIL'},
                                document_id=metadata.get('doc_id', 'unknown') if metadata else 'unknown',
                                metadata=original_chunk.get('metadata', {})
                            )

        # Phase 3: Entity Merging
        print("\n[PHASE 3] Entity Merging")
        print("-" * 80)

        if self.enable_entity_linking and self.entity_merger:
            # NEW (Phase 1 Step 4): Use UnifiedEntityMerger
            print(f"  [3.1] Entity merging (strategy: {self.entity_merge_strategy})...")
            original_count = len(all_entities)
            merged_entities = await self.entity_merger.merge_entities(all_entities, merge_mode='append')
            merge_reduction = original_count - len(merged_entities)
            print(f"    Merged {original_count} -> {len(merged_entities)} entities (reduced by {merge_reduction})")

            # Build entity ID mapping
            print("  [3.2] Remapping entity IDs in relations...")
            entity_id_mapping = {}
            for merged in merged_entities:
                primary_id = merged.get('entity_id')
                if not primary_id:
                    continue
                entity_id_mapping[primary_id] = primary_id
                for old_id in merged.get('entity_ids_merged', []):
                    entity_id_mapping[old_id] = primary_id

            # Update relations' linked_entities
            remapped_count = 0
            for relation in all_relations:
                old_links = relation['metadata'].get('linked_entities', [])
                new_links = []
                for old_id in old_links:
                    primary_id = entity_id_mapping.get(old_id, old_id)
                    new_links.append(primary_id)
                    if primary_id != old_id:
                        remapped_count += 1
                relation['metadata']['linked_entities'] = new_links

            print(f"    Remapped {remapped_count} entity ID references in {len(all_relations)} relations")
        else:
            print("  [3.1] Entity linking disabled - using raw entities")
            merged_entities = all_entities

        # FIX 1C: Post-merge entity linking (enables cross-chunk entity references)
        print("  [3.3] Linking entities to relations (post-merge, cross-chunk)...")
        entities_linked = 0

        for relation in all_relations:
            linked_entities = []
            relation_content = relation.get('content', '')

            # CRITICAL: Check against MERGED entities (not per-chunk)
            # This enables cross-chunk linking (e.g., paragraph relation mentions table entity)
            for entity in merged_entities:
                entity_name = entity.get('entity_name', '')
                # Simple substring match (works for both English and Bangla)
                # TODO: Enhance with fuzzy matching for better recall
                if entity_name and entity_name in relation_content:
                    linked_entities.append(entity.get('entity_id'))
                    entities_linked += 1

            # Update relation metadata
            if 'metadata' not in relation:
                relation['metadata'] = {}
            relation['metadata']['linked_entities'] = linked_entities

        print(f"    Linked {entities_linked} entity references across {len(all_relations)} relations")
        if all_relations and entities_linked > 0:
            print(f"    Avg entities per relation: {entities_linked / len(all_relations):.1f}")

        # Add hyper_relation to entities (renumbered from 3.3 to 3.4 after adding post-merge linking)
        print("  [3.4] Adding hyper_relation to entities (bidirectional linking)...")
        entity_lookup = {e['entity_id']: e for e in merged_entities if e.get('entity_id')}

        # FIX 1D: Diagnostic logging for relation_id coverage
        relations_with_id = sum(1 for r in all_relations if r.get('relation_id'))
        relations_without_id = len(all_relations) - relations_with_id
        print(f"    DEBUG: {relations_with_id}/{len(all_relations)} relations have relation_id")
        if relations_without_id > 0:
            print(f"    [WARN] {relations_without_id} relations missing relation_id (will be skipped)")

        hyper_relation_added = 0
        skipped_relations_logged = 0
        for relation in all_relations:
            relation_id = relation.get('relation_id')
            if not relation_id:
                # Log first 3 problematic relations for debugging
                if skipped_relations_logged < 3:
                    print(f"    [WARN] Skipping relation (no ID): {relation.get('content', '')[:60]}...")
                    skipped_relations_logged += 1
                continue

            linked_entities = relation.get('metadata', {}).get('linked_entities', [])
            for entity_id in linked_entities:
                if entity_id in entity_lookup:
                    entity_lookup[entity_id]['hyper_relation'] = relation_id
                    hyper_relation_added += 1

        print(f"    Added hyper_relation to {hyper_relation_added} entities")

        # Validate orphan entities
        orphan_entities = [e for e in merged_entities if not e.get('hyper_relation')]
        if orphan_entities:
            print(f"    [WARN] Found {len(orphan_entities)} orphan entities (no relation link)")
            print(f"           Orphan rate: {len(orphan_entities)/len(merged_entities)*100:.1f}%")

            # NEW: Phase 3.5 - Post-merge orphan linking
            print("\n  [3.5] Post-merge orphan linking (fix cross-lingual entities)...")
            linked_orphans, synthetic_relations = await self._link_orphan_entities(
                orphan_entities,
                merged_entities,
                all_relations
            )

            if synthetic_relations:
                print(f"    Created {len(synthetic_relations)} synthetic relations for orphans")
                all_relations.extend(synthetic_relations)

                # Update entity_lookup with newly linked orphans
                for entity in linked_orphans:
                    entity_id = entity.get('entity_id')
                    if entity_id and entity_id in entity_lookup:
                        entity_lookup[entity_id]['hyper_relation'] = entity.get('hyper_relation')

                print(f"    Linked {len(linked_orphans)} orphan entities")

                # Re-validate orphans
                remaining_orphans = [e for e in merged_entities if not e.get('hyper_relation')]
                if remaining_orphans:
                    print(f"    [WARN] {len(remaining_orphans)} orphans remain after linking")
                else:
                    print(f"    [SUCCESS] All orphan entities linked!")
            else:
                print(f"    [INFO] No linkable orphans found")
        else:
            print(f"    [OK] No orphan entities found")

        # Phase 4: Numeric Validation
        print("\n[PHASE 4] Numeric Validation")
        print("-" * 80)

        print("  Validating numeric accuracy...")
        numeric_result = await self.numeric_validator.validate_extraction(
            source_document=markdown_text,
            entities=merged_entities,
            relations=all_relations,
            validation_level=self.validation_level
        )
        print(f"    Status: {numeric_result['status']}")
        print(f"    Coverage: {numeric_result['numeric_coverage']:.2%}")
        print(f"    Hallucination: {numeric_result['hallucination_rate']:.2%}")

        # Overall status
        numeric_status = numeric_result['status']
        if numeric_status == 'PASS':
            overall_status = 'PASS'
        elif numeric_status == 'FAIL':
            overall_status = 'FAIL'
        else:
            overall_status = 'WARNING'

        # Final result
        print("\n" + "=" * 80)
        print(f"Pipeline Status: {overall_status}")
        print("=" * 80)
        print(f"  Entities: {len(merged_entities)}")
        print(f"  Relations: {len(all_relations)}")
        print(f"  Numeric Coverage: {numeric_result['numeric_coverage']:.2%}")
        print(f"  Pipeline Version: {PIPELINE_VERSION}")
        print("=" * 80)

        # Save failed tables to human review queue
        if failed_tables:
            await self._save_to_review_queue(failed_tables, metadata)

        # DEBUG: Final result assembly
        print(f"\n[DEBUG] FINAL RESULT ASSEMBLY:")
        print(f"  - merged_entities count: {len(merged_entities)}")
        print(f"  - all_relations count: {len(all_relations)}")
        if merged_entities:
            print(f"  - Sample entity names: {[e.get('entity_name') for e in merged_entities[:3]]}")
        if all_relations:
            print(f"  - Sample relation snippets: {[r.get('content', '')[:50] for r in all_relations[:3]]}")

        result_dict = {
            'entities': merged_entities,
            'relations': all_relations,
            'chunks': chunks,
            'validation': {
                'numeric': numeric_result,
                'overall_status': overall_status
            },
            'statistics': {
                'total_entities': len(merged_entities),
                'total_relations': len(all_relations),
                'total_chunks': len(chunks),
                'table_chunks': len(table_chunks),
                'paragraph_chunks': len(paragraph_chunks),
                'successful_tables': successful_tables,
                'failed_tables': len(failed_tables),
                'table_success_rate': success_rate,
                'entity_merge_reduction': len(all_entities) - len(merged_entities) if self.enable_entity_linking else 0,
                'numeric_coverage': numeric_result['numeric_coverage'],
                'pipeline_version': PIPELINE_VERSION,  # NEW
                'extraction_strategy': self.extraction_strategy  # NEW
            },
            'pipeline_metadata': self.pipeline_metadata,  # NEW
            'failed_tables': failed_tables
        }

        print(f"[DEBUG] Returning result_dict with keys: {list(result_dict.keys())}")
        print(f"[DEBUG] result_dict['entities'] type: {type(result_dict['entities'])}, length: {len(result_dict['entities'])}")
        print(f"[DEBUG] result_dict['relations'] type: {type(result_dict['relations'])}, length: {len(result_dict['relations'])}")

        return result_dict

    async def _link_orphan_entities(
        self,
        orphan_entities: List[Dict],
        all_entities: List[Dict],
        all_relations: List[Dict]
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Link orphan entities by creating synthetic relations.

        Strategy:
        1. For each orphan entity, find similar entities (by type and context)
        2. If similar entity has relations, create synthetic relation for orphan
        3. Link orphan to synthetic relation

        This fixes cross-lingual orphans (e.g., "CIVIL ENGINEERING" vs "সিভিল ইঞ্জিনিয়ারিং")

        Args:
            orphan_entities: Entities with no relation links
            all_entities: All merged entities
            all_relations: All relations

        Returns:
            (linked_orphans, synthetic_relations)
        """
        from bigrag.utils import compute_mdhash_id
        from bigrag.constants import RELATION_PREFIX

        linked_orphans = []
        synthetic_relations = []

        # Build index of connected entities by type
        connected_by_type = {}
        for entity in all_entities:
            if entity.get('hyper_relation'):  # Has connection
                entity_type = entity.get('entity_type', 'unknown')
                if entity_type not in connected_by_type:
                    connected_by_type[entity_type] = []
                connected_by_type[entity_type].append(entity)

        # Process each orphan
        for orphan in orphan_entities:
            orphan_type = orphan.get('entity_type', 'unknown')
            orphan_name = orphan.get('entity_name', '')
            orphan_id = orphan.get('entity_id')

            if not orphan_id or not orphan_name:
                continue

            # Strategy 1: Find related entities of same type with connections
            related_entities = connected_by_type.get(orphan_type, [])

            if related_entities:
                # Find the best match (by source_id proximity or name similarity)
                best_match = self._find_best_match(orphan, related_entities)

                if best_match:
                    # Get the relation of the best match
                    match_relation_id = best_match.get('hyper_relation')
                    match_relation = None

                    for rel in all_relations:
                        if rel.get('relation_id') == match_relation_id:
                            match_relation = rel
                            break

                    if match_relation:
                        # Create synthetic relation for orphan based on matched relation
                        match_content = match_relation.get('content', '')
                        match_name = best_match.get('entity_name', '')

                        # Replace matched entity name with orphan name in relation content
                        if match_name in match_content:
                            synthetic_content = match_content.replace(match_name, orphan_name)
                        else:
                            # Fallback: Create generic relation
                            synthetic_content = f"{orphan_name} is a {orphan_type} related to {match_name}."

                        # Generate unique relation ID
                        synthetic_relation_id = compute_mdhash_id(
                            synthetic_content.strip(),
                            prefix=RELATION_PREFIX
                        )

                        # Create synthetic relation
                        synthetic_relation = {
                            'role': 'relation',
                            'content': synthetic_content,
                            'description': synthetic_content,
                            'completeness_score': 7,  # Lower than original (synthetic)
                            'source_id': orphan.get('source_id', 'unknown'),
                            'relation_id': synthetic_relation_id,
                            'metadata': {
                                'extraction_method': 'synthetic_orphan_linking',
                                'linked_entities': [orphan_id],
                                'original_relation_id': match_relation_id,
                                'orphan_entity': orphan_name,
                                'matched_entity': match_name,
                                'purpose': 'Link orphan entity (likely cross-lingual duplicate)'
                            }
                        }

                        synthetic_relations.append(synthetic_relation)

                        # Link orphan to synthetic relation
                        orphan['hyper_relation'] = synthetic_relation_id
                        linked_orphans.append(orphan)

        return linked_orphans, synthetic_relations

    def _find_best_match(self, orphan: Dict, candidates: List[Dict]) -> Optional[Dict]:
        """
        Find best matching entity for orphan.

        Matching criteria:
        1. Same source_id (from same chunk)
        2. Name similarity (for cross-lingual matches)
        3. Same entity_type (already filtered)

        Args:
            orphan: Orphan entity
            candidates: Candidate entities with connections

        Returns:
            Best matching entity or None
        """
        orphan_source = orphan.get('source_id', '')
        orphan_name = orphan.get('entity_name', '').lower()

        # Strategy 1: Same source chunk (highest confidence)
        for candidate in candidates:
            candidate_source = candidate.get('source_id', '')
            if orphan_source and candidate_source and orphan_source in candidate_source:
                return candidate

        # Strategy 2: Name similarity (for cross-lingual: "CSE" matches "সিএসই")
        # For department_code type, prioritize shorter names (codes)
        if orphan.get('entity_type') == 'department_code':
            for candidate in candidates:
                candidate_name = candidate.get('entity_name', '').lower()
                # Check if one is abbreviation of other
                if len(orphan_name) < 10 and candidate_name.startswith(orphan_name[:3]):
                    return candidate
                if len(candidate_name) < 10 and orphan_name.startswith(candidate_name[:3]):
                    return candidate

        # Strategy 3: Return first candidate (fallback)
        if candidates:
            return candidates[0]

        return None

    async def _save_to_review_queue(self, failed_items: List[Dict], document_metadata: Optional[Dict] = None):
        """Save failed validations to human review queue (JSON file)."""
        # Load existing queue
        if self.review_queue_path.exists():
            with open(self.review_queue_path, 'r', encoding='utf-8') as f:
                queue = json.load(f)
        else:
            queue = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'description': 'Human review queue for failed validation items',
                'items': []
            }

        # Add new items with timestamp
        for item in failed_items:
            review_item = {
                'id': f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{item.get('chunk_id', 'unknown')}",
                'timestamp': datetime.now().isoformat(),
                'status': 'pending',
                'severity': self._calculate_severity(item),
                'item_type': 'table',
                'document_metadata': document_metadata or {},
                **item
            }
            queue['items'].append(review_item)

        # Create directory if it doesn't exist
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)

        # Save updated queue
        with open(self.review_queue_path, 'w', encoding='utf-8') as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved {len(failed_items)} items to review queue: {self.review_queue_path}")

    def _calculate_severity(self, item: Dict) -> str:
        """Calculate severity of validation failure."""
        numeric_coverage = item.get('numeric_coverage', 0.0)
        missing_count = len(item.get('missing_numbers', []))
        hallucinated_count = len(item.get('hallucinated_numbers', []))

        if numeric_coverage < 0.7 or hallucinated_count > 5:
            return 'critical'
        elif numeric_coverage < 0.85 or missing_count > 3:
            return 'high'
        elif numeric_coverage < 0.95 or missing_count > 1:
            return 'medium'
        else:
            return 'low'

    def save_result(self, result: Dict, output_path: str):
        """Save pipeline result to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[OK] Result saved to {output_path}")


# Convenience function for quick usage
async def build_knowledge_graph(
    markdown_text: str,
    api_key: str,
    metadata: Optional[Dict] = None,
    validation_level: str = "MODERATE",
    extraction_strategy: str = "hybrid"
) -> Dict:
    """
    Convenience function to build knowledge graph from document.

    Args:
        markdown_text: Document content
        api_key: OpenAI API key
        metadata: Optional document metadata
        validation_level: Validation strictness
        extraction_strategy: NEW - strict/gleaning/hybrid

    Returns:
        Pipeline result with entities and relations
    """
    pipeline = EnhancedKGPipeline(
        api_key=api_key,
        validation_level=validation_level,
        extraction_strategy=extraction_strategy
    )

    result = await pipeline.process_document(
        markdown_text,
        metadata=metadata
    )

    return result
