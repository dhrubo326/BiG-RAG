"""
HybridExtractor - Robust table + paragraph extraction strategy.

Fixes all 5 table fact extraction issues:
- Issue #4: Explicit enable_table_fact_extraction parameter
- Issue #5: Graceful degradation with try/except
- Issue #6: Validation-aware processing (checks validation_status)
- Issue #7: HITL integration for failed extractions
- Issue #8: Statistics tracking and reporting

This is a FULLY MODULAR implementation, not a thin wrapper.
"""

from bigrag.interfaces.extractor import ExtractorInterface
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HybridExtractor(ExtractorInterface):
    """
    Hybrid extraction: tables via TableFactExtractor + paragraphs via LLM.

    NEW FEATURES (fixes for issues #4-#8):
    - Explicit control via enable_table_fact_extraction parameter
    - Graceful degradation (skip failed tables, continue processing)
    - Validation-aware (checks validation_status before processing)
    - HITL integration (saves failed tables for human review)
    - Statistics tracking (success rate, failure reasons)
    """

    def __init__(
        self,
        api_key: str,
        gleaning_iterations: int = 2,
        concurrency: int = 16,
        enable_validation: bool = True,
        enable_table_fact_extraction: bool = True,  # NEW: Issue #4
        hitl_handler: Optional[any] = None  # NEW: Issue #7
    ):
        self.enable_table_fact_extraction = enable_table_fact_extraction  # NEW: Explicit control

        # Initialize table extractor (only if enabled)
        if self.enable_table_fact_extraction:
            from bigrag.extractors.table_fact_extractor import TableFactExtractor
            self.table_extractor = TableFactExtractor()
        else:
            self.table_extractor = None

        # Initialize paragraph extractor (always enabled)
        from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
        self.paragraph_extractor = ConstrainedLLMExtractor(
            api_key=api_key,
            enable_gleaning=True,
            max_gleaning_iterations=gleaning_iterations,
            enable_numeric_validation=enable_validation
        )
        self.batch_extractor = BatchConstrainedExtractor(self.paragraph_extractor)

        # NEW: HITL integration (Issue #7)
        self.hitl_handler = hitl_handler

    async def extract(self, chunks: List[Dict], language: str = "English") -> Dict:
        """
        Extract using both table-specific and paragraph extraction.

        NEW: Robust error handling, validation checking, HITL integration, statistics tracking.

        Args:
            chunks: List of chunk dicts from chunker
            language: Language for extraction (default: "English")

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'failed_chunks': [...],
                'chunks': [...],  # For numeric validation
                'statistics': {  # NEW: Issue #8
                    'total_tables': int,
                    'successful_tables': int,
                    'failed_tables': int,
                    'table_success_rate': float,
                    'failure_reasons': {...}
                }
            }
        """
        table_chunks = [c for c in chunks if c.get("type") == "table"]
        para_chunks = [c for c in chunks if c.get("type") == "paragraph"]

        # Build chunk index for quick lookup
        chunk_index = {chunk['chunk_id']: chunk for chunk in chunks}
        chunks_with_extractions = []

        # NEW: Statistics tracking (Issue #8)
        stats = {
            'total_tables': len(table_chunks),
            'successful_tables': 0,
            'failed_tables': 0,
            'table_success_rate': 0.0,
            'failure_reasons': {}
        }

        # Extract from tables (with robust error handling)
        table_entities, table_relations = [], []

        if self.enable_table_fact_extraction and table_chunks:
            logger.info(f"[HybridExtractor] Processing {len(table_chunks)} table chunks")

            for chunk in table_chunks:
                chunk_id = chunk['chunk_id']
                chunk_table_entities = []
                chunk_table_relations = []

                # NEW: Validation-aware processing (Issue #6)
                validation_status = chunk.get('structured_data', {}).get('metadata', {}).get('validation_status', 'PASS')

                if validation_status == 'FAIL':
                    # Table already failed validation during chunking
                    stats['failed_tables'] += 1
                    reason = 'Pre-extraction validation failed'
                    stats['failure_reasons'][reason] = stats['failure_reasons'].get(reason, 0) + 1

                    # NEW: Save to HITL with rich validation metadata (Issue #7)
                    if self.hitl_handler:
                        try:
                            # COPIED FROM production_pipeline.py:161-171
                            # Preserve rich validation metadata for failed tables (for human review)
                            await self.hitl_handler.save_failed_table(
                                chunk_id=chunk_id,
                                table_id=chunk.get('structured_data', {}).get('table_id', 'unknown'),
                                reason=reason,
                                validation_feedback=chunk.get('structured_data', {}).get('metadata', {}).get('validation_feedback', ''),
                                missing_numbers=chunk.get('structured_data', {}).get('metadata', {}).get('missing_numbers', []),
                                hallucinated_numbers=chunk.get('structured_data', {}).get('metadata', {}).get('hallucinated_numbers', []),
                                numeric_coverage=chunk.get('structured_data', {}).get('metadata', {}).get('numeric_coverage', 0.0),
                                source_markdown=chunk.get('content', ''),
                                extracted_data=chunk.get('structured_data', {})
                            )
                            logger.warning(f"[HybridExtractor] Table {chunk_id} failed validation - saved to HITL with rich metadata")
                        except Exception as hitl_error:
                            logger.error(f"[HybridExtractor] HITL save failed: {hitl_error}")

                    continue  # Skip failed table (graceful degradation - Issue #5)

                # NEW: Graceful degradation (Issue #5) - try/except around extraction
                try:
                    from bigrag.utils import compute_mdhash_id
                    from bigrag.constants import RELATION_PREFIX, ENTITY_PREFIX

                    result = self.table_extractor.extract_facts_from_table(
                        chunk.get("structured_data", {}),
                        chunk_id
                    )
                    chunk_table_entities = result.get("entities", [])
                    chunk_table_relations = result.get("relations", [])

                    # CRITICAL: Add entity_id to table entities (required for BipartiteGraphBuilder)
                    for entity in chunk_table_entities:
                        if 'entity_id' not in entity:
                            entity_id = compute_mdhash_id(entity.get('entity_name', ''), prefix=ENTITY_PREFIX)
                            entity['entity_id'] = entity_id

                    # CRITICAL: Add relation_id and linked_entities to table relations
                    for relation in chunk_table_relations:
                        if 'relation_id' not in relation:
                            relation_id = compute_mdhash_id(relation.get('content', '').strip(), prefix=RELATION_PREFIX)
                            relation['relation_id'] = relation_id
                        if 'metadata' not in relation:
                            relation['metadata'] = {}
                        if 'linked_entities' not in relation['metadata']:
                            relation['metadata']['linked_entities'] = []

                    table_entities.extend(chunk_table_entities)
                    table_relations.extend(chunk_table_relations)

                    # Build chunk with extractions for numeric validation
                    chunks_with_extractions.append({
                        'chunk_id': chunk_id,
                        'content': chunk.get('content', ''),
                        'entities': chunk_table_entities,
                        'relations': chunk_table_relations
                    })

                    # NEW: Track success (Issue #8)
                    stats['successful_tables'] += 1

                except Exception as e:
                    # NEW: Graceful degradation (Issue #5)
                    stats['failed_tables'] += 1
                    reason = f"Extraction error: {type(e).__name__}"
                    stats['failure_reasons'][reason] = stats['failure_reasons'].get(reason, 0) + 1

                    logger.error(f"[HybridExtractor] Table {chunk_id} extraction failed: {e}", exc_info=True)

                    # NEW: Save to HITL with rich metadata (Issue #7)
                    if self.hitl_handler:
                        try:
                            # For extraction errors, preserve whatever metadata we have
                            await self.hitl_handler.save_failed_table(
                                chunk_id=chunk_id,
                                table_id=chunk.get('structured_data', {}).get('table_id', 'unknown'),
                                reason=reason,
                                validation_feedback=chunk.get('structured_data', {}).get('metadata', {}).get('validation_feedback', ''),
                                missing_numbers=chunk.get('structured_data', {}).get('metadata', {}).get('missing_numbers', []),
                                hallucinated_numbers=chunk.get('structured_data', {}).get('metadata', {}).get('hallucinated_numbers', []),
                                numeric_coverage=chunk.get('structured_data', {}).get('metadata', {}).get('numeric_coverage', 0.0),
                                source_markdown=chunk.get('content', ''),
                                extracted_data=chunk.get('structured_data', {}),
                                error_traceback=str(e)  # Include exception details
                            )
                            logger.info(f"[HybridExtractor] Failed table {chunk_id} saved to HITL queue with error traceback")
                        except Exception as hitl_error:
                            logger.error(f"[HybridExtractor] HITL save failed: {hitl_error}")

                    # Continue processing other tables (graceful degradation)
                    continue

            # NEW: Calculate success rate (Issue #8)
            if stats['total_tables'] > 0:
                stats['table_success_rate'] = stats['successful_tables'] / stats['total_tables']

            # NEW: Log statistics (Issue #8)
            logger.info(
                f"[HybridExtractor] Table extraction complete: "
                f"{stats['successful_tables']}/{stats['total_tables']} successful "
                f"({stats['table_success_rate']:.1%} success rate)"
            )
            if stats['failed_tables'] > 0:
                logger.warning(
                    f"[HybridExtractor] {stats['failed_tables']} tables failed. "
                    f"Reasons: {stats['failure_reasons']}"
                )

        elif not self.enable_table_fact_extraction and table_chunks:
            # Table fact extraction disabled - log warning
            logger.info(f"[HybridExtractor] Skipping {len(table_chunks)} tables (enable_table_fact_extraction=False)")

        # Extract from paragraphs
        para_entities = []
        para_relations = []

        if para_chunks:
            logger.info(f"[HybridExtractor] Processing {len(para_chunks)} paragraph chunks")
            para_result = await self.batch_extractor.extract_from_chunks(para_chunks, language=language)

            # Flatten paragraph extractions
            for extraction in para_result.get('extractions', []):
                chunk_id = extraction.get('chunk_id')
                chunk_para_entities = []
                chunk_para_relations = []

                # Add source_id AND entity_id to each entity
                for entity in extraction.get('entities', []):
                    from bigrag.utils import compute_mdhash_id
                    from bigrag.constants import ENTITY_PREFIX

                    entity['source_id'] = chunk_id

                    if 'entity_id' not in entity:
                        entity_id = compute_mdhash_id(entity.get('entity_name', ''), prefix=ENTITY_PREFIX)
                        entity['entity_id'] = entity_id

                    para_entities.append(entity)
                    chunk_para_entities.append(entity)

                # Add source_id, relation_id, and initialize metadata for each relation
                for relation in extraction.get('relations', []):
                    from bigrag.utils import compute_mdhash_id
                    from bigrag.constants import RELATION_PREFIX

                    relation['source_id'] = chunk_id

                    if 'relation_id' not in relation:
                        relation_id = compute_mdhash_id(relation.get('content', '').strip(), prefix=RELATION_PREFIX)
                        relation['relation_id'] = relation_id

                    if 'metadata' not in relation:
                        relation['metadata'] = {}
                    if 'linked_entities' not in relation['metadata']:
                        relation['metadata']['linked_entities'] = []

                    para_relations.append(relation)
                    chunk_para_relations.append(relation)

                # Build chunk with extractions for numeric validation
                if chunk_id in chunk_index:
                    chunks_with_extractions.append({
                        'chunk_id': chunk_id,
                        'content': chunk_index[chunk_id].get('content', ''),
                        'entities': chunk_para_entities,
                        'relations': chunk_para_relations
                    })

        # Combine results
        return {
            'entities': table_entities + para_entities,
            'relations': table_relations + para_relations,
            'failed_chunks': para_result.get('failed_chunks', []) if para_chunks else [],
            'chunks': chunks_with_extractions,
            'statistics': stats  # NEW: Issue #8
        }
