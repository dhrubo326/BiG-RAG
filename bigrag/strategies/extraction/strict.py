from bigrag.interfaces.extractor import ExtractorInterface
from typing import List, Dict

class StrictExtractor(ExtractorInterface):
    def __init__(self, api_key: str, concurrency: int = 16, enable_validation: bool = True):
        from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
        self.llm_extractor = ConstrainedLLMExtractor(api_key=api_key, enable_gleaning=False, enable_numeric_validation=enable_validation)
        self.batch_extractor = BatchConstrainedExtractor(self.llm_extractor)
    async def extract(self, chunks: List[Dict], language: str = "English") -> Dict:
        """
        Extract entities and relations using strict schema without gleaning.

        Args:
            chunks: List of chunk dicts from chunker
            language: Language for extraction (default: "English")

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'failed_chunks': [...],
                'chunks': [  # NEW: For numeric validation support
                    {
                        'chunk_id': '...',
                        'content': '...',
                        'entities': [...],  # Entities from this chunk
                        'relations': [...]  # Relations from this chunk
                    }
                ]
            }
        """
        result = await self.batch_extractor.extract_from_chunks(chunks, language=language)

        # Build chunk index for quick lookup
        chunk_index = {chunk['chunk_id']: chunk for chunk in chunks}

        # Flatten extractions array into entities and relations
        # IMPORTANT: Add source_id (chunk_id) to each entity/relation
        all_entities = []
        all_relations = []
        chunks_with_extractions = []  # NEW: For numeric validation

        for extraction in result.get('extractions', []):
            chunk_id = extraction.get('chunk_id')
            chunk_entities = []
            chunk_relations = []

            # Add source_id AND entity_id to each entity
            for entity in extraction.get('entities', []):
                from bigrag.utils import compute_mdhash_id
                from bigrag.constants import ENTITY_PREFIX

                entity['source_id'] = chunk_id

                # CRITICAL: Generate entity_id using hash of entity_name (required for BipartiteGraphBuilder)
                if 'entity_id' not in entity:
                    entity_id = compute_mdhash_id(entity.get('entity_name', ''), prefix=ENTITY_PREFIX)
                    entity['entity_id'] = entity_id

                all_entities.append(entity)
                chunk_entities.append(entity)

            # Add source_id, relation_id, and initialize metadata for each relation
            for relation in extraction.get('relations', []):
                from bigrag.utils import compute_mdhash_id
                from bigrag.constants import RELATION_PREFIX

                relation['source_id'] = chunk_id

                # CRITICAL: Generate relation_id (required for hyper_relation linking)
                if 'relation_id' not in relation:
                    relation_id = compute_mdhash_id(relation.get('content', '').strip(), prefix=RELATION_PREFIX)
                    relation['relation_id'] = relation_id

                # CRITICAL: Populate linked_entities from LLM output (NEW - entity-relation linking)
                # LLM now outputs linked_entities array with entity_name values
                if 'metadata' not in relation:
                    relation['metadata'] = {}

                # Extract linked_entities from LLM output (if provided)
                linked_entities_from_llm = relation.get('linked_entities', [])
                if linked_entities_from_llm:
                    # LLM provided entity names - store in metadata
                    relation['metadata']['linked_entities'] = linked_entities_from_llm
                    relation['metadata']['linking_source'] = 'llm_extraction'
                else:
                    # LLM didn't provide links - will be populated in Step 6.5 (post-merge linking)
                    relation['metadata']['linked_entities'] = []
                    relation['metadata']['linking_source'] = 'post_merge_fallback'

                all_relations.append(relation)
                chunk_relations.append(relation)

            # Build chunk with extractions for numeric validation
            if chunk_id in chunk_index:
                chunks_with_extractions.append({
                    'chunk_id': chunk_id,
                    'content': chunk_index[chunk_id].get('content', ''),
                    'entities': chunk_entities,
                    'relations': chunk_relations
                })

        return {
            'entities': all_entities,
            'relations': all_relations,
            'failed_chunks': result.get('failed_chunks', []),
            'chunks': chunks_with_extractions  # NEW: Enable numeric validation
        }