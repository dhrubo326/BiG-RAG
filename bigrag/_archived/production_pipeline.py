"""
Production Knowledge Graph Pipeline for Educational Domain

Integrates all Week 1-3 components into a unified pipeline:
- Phase 1: Pre-processing (Table extraction, Bilingual detection, Smart chunking)
- Phase 2: Extraction (Table facts, Paragraph facts with validation)
- Phase 3: Entity Merging (Canonicalization, Multi-strategy linking)
- Phase 4: Numeric Validation (Quality control)

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


class ProductionKGPipeline:
    """
    End-to-end production knowledge graph construction pipeline.

    Usage:
        pipeline = ProductionKGPipeline(api_key="your-key")
        result = await pipeline.process_document(markdown_text, metadata)

        if result['validation']['status'] == 'PASS':
            # Use result['entities'] and result['relations']
            pass
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        validation_level: str = "STRICT",
        enable_entity_linking: bool = True,
        extraction_mode: str = "semi_structured",
        review_queue_path: str = "expr/human_review_queue.json"
    ):
        """
        Initialize production pipeline.

        Args:
            api_key: OpenAI API key
            model: LLM model to use (gpt-4o-mini recommended for cost efficiency)
            validation_level: STRICT (production), MODERATE (dev), LENIENT (test)
            enable_entity_linking: Whether to merge entities
            extraction_mode: Extraction validation mode (structured/semi_structured/unstructured)
                - structured: 99%+ accuracy, strict validation (best for tables)
                - semi_structured: 95%+ accuracy, moderate validation [DEFAULT]
                - unstructured: 80%+ accuracy, lenient validation (best for narrative text)
            review_queue_path: Path to human review queue JSON file
        """
        self.api_key = api_key
        self.model = model
        self.validation_level = validation_level
        self.enable_entity_linking = enable_entity_linking
        self.extraction_mode = extraction_mode
        self.review_queue_path = Path(review_queue_path)

        # Initialize components
        self.table_extractor = GPT4TableExtractor(api_key=api_key, model=model)
        self.chunker = TableAwareChunker(self.table_extractor)
        self.paragraph_extractor = ConstrainedLLMExtractor(
            api_key=api_key,
            model=model,
            extraction_mode=extraction_mode
        )
        self.batch_extractor = BatchConstrainedExtractor(self.paragraph_extractor)
        # Initialize numeric validator - reads GEMINI_API_KEY from .env (not OpenAI key!)
        self.numeric_validator = NumericValidator(api_key=None, use_llm_validation=True)

        if enable_entity_linking:
            self.canon_map = EntityCanonicalizationMap()
            self.entity_linker = SimpleEntityLinker(self.canon_map)
        else:
            self.entity_linker = None

    async def process_document(
        self,
        markdown_text: str,
        metadata: Optional[Dict] = None,
        language: str = "English"
    ) -> Dict:
        """
        Process document through full production pipeline.

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
                    ...
                }
            }
        """

        print("=" * 80)
        print("Production KG Pipeline Starting")
        print("=" * 80)

        # Phase 1: Pre-processing
        print("\n[PHASE 1] Pre-processing")
        print("-" * 80)

        # Step 1.1: Smart chunking (includes table extraction)
        print("  [1.1] Smart chunking with table extraction...")
        chunks = await self.chunker.chunk_document(
            markdown_text,
            chunk_size=1200,
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

        failed_tables = []
        successful_tables = 0

        for chunk in table_chunks:
            # Get validation status from table extraction
            validation_status = chunk.get('structured_data', {}).get('metadata', {}).get('validation_status', 'UNKNOWN')

            if validation_status == 'FAIL':
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

        # Step 2.2: Paragraph extraction (LLM with validation)
        print("  [2.2] Paragraph extraction (constrained LLM)...")
        if paragraph_chunks:
            batch_result = await self.batch_extractor.extract_from_chunks(
                paragraph_chunks,
                language=language
            )

            for extraction in batch_result['extractions']:
                chunk_id = extraction['chunk_id']

                # Add source_id, metadata, and entity_id to each entity
                from bigrag.utils import compute_mdhash_id
                from bigrag.constants import ENTITY_PREFIX

                for entity in extraction['entities']:
                    if 'source_id' not in entity:
                        entity['source_id'] = chunk_id
                    if 'metadata' not in entity:
                        entity['metadata'] = {}
                    entity['metadata']['extraction_method'] = 'constrained_llm'

                    # FIX: Map key_score to weight for entity linker compatibility
                    # ConstrainedLLMExtractor returns entities with 'key_score' field,
                    # but entity linker expects 'weight' field. This ensures paragraph
                    # entities have same weight semantics as table entities.
                    if 'key_score' in entity and 'weight' not in entity:
                        entity['weight'] = entity['key_score']
                    elif 'weight' not in entity:
                        # Fallback: default weight if no key_score (should not happen)
                        entity['weight'] = 50.0

                    # UNIFIED: Generate stable entity ID if not present (consistent with standard pipeline)
                    if 'entity_id' not in entity:
                        entity_id = compute_mdhash_id(entity['entity_name'], prefix=ENTITY_PREFIX)  # UNIFIED: Use name, not description
                        entity['entity_id'] = entity_id

                # Add source_id, metadata, and linked_entities to each relation
                for relation in extraction['relations']:
                    if 'source_id' not in relation:
                        relation['source_id'] = chunk_id
                    if 'metadata' not in relation:
                        relation['metadata'] = {}
                    relation['metadata']['extraction_method'] = 'constrained_llm'

                    # Extract linked entities from relation content using entity_id (Option B3)
                    # (entities mentioned in the relation)
                    linked_entities = []
                    for entity in extraction['entities']:
                        # Simple heuristic: if entity name appears in relation content
                        if entity['entity_name'] in relation['content']:
                            # Store entity_id instead of entity_name (survives name changes)
                            linked_entities.append(entity['entity_id'])

                    relation['metadata']['linked_entities'] = linked_entities

                all_entities.extend(extraction['entities'])
                all_relations.extend(extraction['relations'])

            stats = batch_result['statistics']
            print(f"    Success rate: {stats['success_rate']:.2%}")
            print(f"    Avg numeric coverage: {stats['avg_numeric_coverage']:.2%}")

            if batch_result['failed_chunks']:
                print(f"    [WARN] {len(batch_result['failed_chunks'])} chunks failed validation")

        # Phase 3: Entity Merging
        print("\n[PHASE 3] Entity Merging")
        print("-" * 80)

        if self.enable_entity_linking and self.entity_linker:
            print("  [3.1] Entity linking (canonicalization + fuzzy matching)...")
            original_count = len(all_entities)
            merged_entities = await self.entity_linker.link_entities_across_chunks(all_entities)
            merge_reduction = original_count - len(merged_entities)
            print(f"    Merged {original_count} -> {len(merged_entities)} entities (reduced by {merge_reduction})")

            # Option B3: Build entity ID mapping (old_id -> primary_id)
            print("  [3.2] Remapping entity IDs in relations...")
            entity_id_mapping = {}
            for merged in merged_entities:
                primary_id = merged.get('entity_id')
                if not primary_id:
                    continue
                # Map primary ID to itself
                entity_id_mapping[primary_id] = primary_id
                # Map all merged IDs to primary ID
                for old_id in merged.get('entity_ids_merged', []):
                    entity_id_mapping[old_id] = primary_id

            # Update all relations' linked_entities with primary IDs
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

        # Phase 3.3: Add hyper_relation to entities (reverse mapping)
        print("  [3.3] Adding hyper_relation to entities (bidirectional linking)...")

        # Build entity lookup dict for fast access
        entity_lookup = {e['entity_id']: e for e in merged_entities if e.get('entity_id')}

        # For each relation, update its linked entities with hyper_relation
        hyper_relation_added = 0
        for relation in all_relations:
            relation_id = relation.get('relation_id')
            if not relation_id:
                continue

            linked_entities = relation.get('metadata', {}).get('linked_entities', [])

            for entity_id in linked_entities:
                if entity_id in entity_lookup:
                    entity_lookup[entity_id]['hyper_relation'] = relation_id
                    hyper_relation_added += 1

        print(f"    Added hyper_relation to {hyper_relation_added} entities")

        # Validate orphan entities (entities without relation context)
        orphan_entities = [e for e in merged_entities if not e.get('hyper_relation')]
        if orphan_entities:
            print(f"    [WARN] Found {len(orphan_entities)} orphan entities (no relation link)")
            print(f"           Orphan rate: {len(orphan_entities)/len(merged_entities)*100:.1f}%")
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

        # Log missing numbers for debugging
        if numeric_result.get('missing_numbers'):
            missing_count = len(numeric_result['missing_numbers'])
            print(f"    [DEBUG] Missing numbers ({missing_count}): {numeric_result['missing_numbers'][:20]}")  # Show first 20

        # Log hallucinated numbers for debugging
        if numeric_result.get('hallucinated_numbers'):
            halluc_count = len(numeric_result['hallucinated_numbers'])
            print(f"    [DEBUG] Hallucinated numbers ({halluc_count}): {numeric_result['hallucinated_numbers'][:20]}")

        # Overall status - only numeric validation matters
        numeric_status = numeric_result['status']

        if numeric_status == 'PASS':
            overall_status = 'PASS'
        elif numeric_status == 'FAIL':
            overall_status = 'FAIL'
        else:
            # numeric_status == 'WARNING'
            overall_status = 'WARNING'

        # Track quality metrics for WARNING cases
        extraction_quality = {
            'extraction_mode': self.extraction_mode,
            'numeric_status': numeric_status,
            'warning_reasons': []
        }

        if numeric_status == 'WARNING':
            extraction_quality['warning_reasons'].append(
                f"Numeric validation WARNING (coverage: {numeric_result['numeric_coverage']:.2%}, "
                f"hallucination: {numeric_result['hallucination_rate']:.2%})"
            )

        # Final result with visual flagging
        print("\n" + "=" * 80)
        if overall_status == 'WARNING':
            print(f"[WARNING] Pipeline Status: {overall_status}")
            print("=" * 80)
            print(f"  Mode: {self.extraction_mode}")
            print(f"  This extraction succeeded with warnings and may need review.")
            for reason in extraction_quality['warning_reasons']:
                print(f"  - {reason}")
            print("=" * 80)
        else:
            print(f"Pipeline Status: {overall_status}")
            print("=" * 80)

        print(f"  Entities: {len(merged_entities)}")
        print(f"  Relations: {len(all_relations)}")
        print(f"  Numeric Coverage: {numeric_result['numeric_coverage']:.2%}")
        print("=" * 80)

        # Save failed tables to human review queue
        if failed_tables:
            await self._save_to_review_queue(failed_tables, metadata)

        return {
            'entities': merged_entities,
            'relations': all_relations,
            'chunks': chunks,
            'validation': {
                'numeric': numeric_result,
                'overall_status': overall_status,
                'extraction_quality': extraction_quality
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
                'numeric_coverage': numeric_result['numeric_coverage']
            },
            'failed_tables': failed_tables  # For inspection/debugging
        }

    async def _save_to_review_queue(self, failed_items: List[Dict], document_metadata: Optional[Dict] = None):
        """
        Save failed validations to human review queue (JSON file).

        Args:
            failed_items: List of failed tables/chunks
            document_metadata: Document metadata for context
        """
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
                'status': 'pending',  # pending, reviewed, fixed, rejected
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
        """
        Calculate severity of validation failure.

        Returns: 'critical', 'high', 'medium', 'low'
        """
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
        """
        Save pipeline result to JSON file.

        Args:
            result: Pipeline result dict
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[OK] Result saved to {output_path}")


# Convenience function for quick usage
async def build_knowledge_graph(
    markdown_text: str,
    api_key: str,
    metadata: Optional[Dict] = None,
    validation_level: str = "STRICT"
) -> Dict:
    """
    Convenience function to build knowledge graph from document.

    Args:
        markdown_text: Document content
        api_key: OpenAI API key
        metadata: Optional document metadata
        validation_level: Validation strictness

    Returns:
        Pipeline result with entities and relations
    """
    pipeline = ProductionKGPipeline(
        api_key=api_key,
        validation_level=validation_level
    )

    result = await pipeline.process_document(
        markdown_text,
        metadata=metadata
    )

    return result
