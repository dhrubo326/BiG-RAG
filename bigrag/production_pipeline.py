"""
Production Knowledge Graph Pipeline for Educational Domain

Integrates all Week 1-3 components into a unified pipeline:
- Phase 1: Pre-processing (Table extraction, Bilingual detection, Smart chunking)
- Phase 2: Extraction (Table facts, Paragraph facts with validation)
- Phase 3: Entity Merging (Canonicalization, Multi-strategy linking)
- Phase 4: Validation (Numeric accuracy, Cross-chunk consistency)

Designed for 99%+ accuracy on bilingual educational documents.
"""

import asyncio
from typing import List, Dict, Optional
import json

from bigrag.preprocessors.table_extractor import GPT4TableExtractor, BilingualDetector
from bigrag.preprocessors.smart_chunker import TableAwareChunker
from bigrag.extractors.table_fact_extractor import TableFactExtractor
from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
from bigrag.merging.canonicalization import EntityCanonicalizationMap
from bigrag.merging.entity_linker import ProductionEntityLinker, SimpleEntityLinker
from bigrag.validators.numeric_validator import NumericValidator
from bigrag.validators.consistency_validator import ConsistencyValidator


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
        enable_entity_linking: bool = True
    ):
        """
        Initialize production pipeline.

        Args:
            api_key: OpenAI API key
            model: LLM model to use (gpt-4o-mini recommended for cost efficiency)
            validation_level: STRICT (production), MODERATE (dev), LENIENT (test)
            enable_entity_linking: Whether to merge entities
        """
        self.api_key = api_key
        self.model = model
        self.validation_level = validation_level
        self.enable_entity_linking = enable_entity_linking

        # Initialize components
        self.table_extractor = GPT4TableExtractor(api_key=api_key, model=model)
        self.chunker = TableAwareChunker(self.table_extractor)
        self.paragraph_extractor = ConstrainedLLMExtractor(api_key=api_key, model=model)
        self.batch_extractor = BatchConstrainedExtractor(self.paragraph_extractor)
        self.numeric_validator = NumericValidator()
        self.consistency_validator = ConsistencyValidator()

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

        # Step 2.1: Table fact extraction (deterministic)
        print("  [2.1] Table fact extraction (rule-based)...")
        for chunk in table_chunks:
            facts = TableFactExtractor.extract_facts_from_table(
                chunk['structured_data'],
                chunk['chunk_id']
            )
            all_entities.extend(facts['entities'])
            all_relations.extend(facts['relations'])
        print(f"    Extracted {len([e for e in all_entities])} entities, {len([r for r in all_relations])} relations from tables")

        # Step 2.2: Paragraph extraction (LLM with validation)
        print("  [2.2] Paragraph extraction (constrained LLM)...")
        if paragraph_chunks:
            batch_result = await self.batch_extractor.extract_from_chunks(
                paragraph_chunks,
                language=language
            )

            for extraction in batch_result['extractions']:
                chunk_id = extraction['chunk_id']

                # Add source_id and metadata to each entity
                for entity in extraction['entities']:
                    if 'source_id' not in entity:
                        entity['source_id'] = chunk_id
                    if 'metadata' not in entity:
                        entity['metadata'] = {}
                    entity['metadata']['extraction_method'] = 'constrained_llm'

                # Add source_id, metadata, and linked_entities to each relation
                for relation in extraction['relations']:
                    if 'source_id' not in relation:
                        relation['source_id'] = chunk_id
                    if 'metadata' not in relation:
                        relation['metadata'] = {}
                    relation['metadata']['extraction_method'] = 'constrained_llm'

                    # Extract linked entities from relation content
                    # (entities mentioned in the relation)
                    linked_entities = []
                    for entity in extraction['entities']:
                        # Simple heuristic: if entity name appears in relation content
                        if entity['entity_name'] in relation['content']:
                            linked_entities.append(entity['entity_name'])

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
            print(f"    Merged {original_count} → {len(merged_entities)} entities (reduced by {merge_reduction})")
        else:
            print("  [3.1] Entity linking disabled - using raw entities")
            merged_entities = all_entities

        # Phase 4: Validation
        print("\n[PHASE 4] Validation")
        print("-" * 80)

        # Step 4.1: Numeric validation
        print("  [4.1] Numeric accuracy validation...")
        numeric_result = self.numeric_validator.validate_extraction(
            source_document=markdown_text,
            entities=merged_entities,
            relations=all_relations,
            validation_level=self.validation_level
        )
        print(f"    Status: {numeric_result['status']}")
        print(f"    Coverage: {numeric_result['numeric_coverage']:.2%}")
        print(f"    Hallucination: {numeric_result['hallucination_rate']:.2%}")

        # Step 4.2: Consistency validation
        print("  [4.2] Cross-chunk consistency validation...")
        consistency_result = self.consistency_validator.validate_consistency(
            entities=merged_entities,
            relations=all_relations,
            validation_level=self.validation_level
        )
        print(f"    Status: {consistency_result['status']}")
        print(f"    Consistency: {consistency_result['consistency_score']:.2%}")
        print(f"    Issues: {consistency_result['total_issues']}")

        # Overall status
        overall_status = 'PASS' if (
            numeric_result['status'] == 'PASS' and
            consistency_result['status'] == 'PASS'
        ) else 'FAIL'

        # Final result
        print("\n" + "=" * 80)
        print(f"Pipeline Status: {overall_status}")
        print("=" * 80)
        print(f"  Entities: {len(merged_entities)}")
        print(f"  Relations: {len(all_relations)}")
        print(f"  Numeric Coverage: {numeric_result['numeric_coverage']:.2%}")
        print(f"  Consistency: {consistency_result['consistency_score']:.2%}")
        print("=" * 80)

        return {
            'entities': merged_entities,
            'relations': all_relations,
            'chunks': chunks,
            'validation': {
                'numeric': numeric_result,
                'consistency': consistency_result,
                'overall_status': overall_status
            },
            'statistics': {
                'total_entities': len(merged_entities),
                'total_relations': len(all_relations),
                'total_chunks': len(chunks),
                'table_chunks': len(table_chunks),
                'paragraph_chunks': len(paragraph_chunks),
                'entity_merge_reduction': len(all_entities) - len(merged_entities) if self.enable_entity_linking else 0,
                'numeric_coverage': numeric_result['numeric_coverage'],
                'consistency_score': consistency_result['consistency_score']
            }
        }

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
