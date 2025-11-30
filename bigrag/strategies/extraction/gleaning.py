from bigrag.interfaces.extractor import ExtractorInterface
from typing import List, Dict

class GleaningExtractor(ExtractorInterface):
    def __init__(self, api_key: str, max_iterations: int = 2, concurrency: int = 16, enable_validation: bool = True):
        from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
        self.llm_extractor = ConstrainedLLMExtractor(api_key=api_key, enable_gleaning=True, max_gleaning_iterations=max_iterations, enable_numeric_validation=enable_validation)
        self.batch_extractor = BatchConstrainedExtractor(self.llm_extractor)
    async def extract(self, chunks: List[Dict]) -> Dict:
        """
        Extract entities and relations using multi-pass gleaning.

        Args:
            chunks: List of chunk dicts from chunker

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'failed_chunks': [...],
                'chunks': [...]  # NEW: For numeric validation support
            }
        """
        result = await self.batch_extractor.extract_from_chunks(chunks)

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

            # Add source_id to each entity
            for entity in extraction.get('entities', []):
                entity['source_id'] = chunk_id
                all_entities.append(entity)
                chunk_entities.append(entity)

            # Add source_id to each relation
            for relation in extraction.get('relations', []):
                relation['source_id'] = chunk_id
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