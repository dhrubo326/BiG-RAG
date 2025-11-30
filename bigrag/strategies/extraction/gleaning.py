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
                'failed_chunks': [...]
            }
        """
        result = await self.batch_extractor.extract_from_chunks(chunks)

        # Flatten extractions array into entities and relations
        # IMPORTANT: Add source_id (chunk_id) to each entity/relation
        all_entities = []
        all_relations = []
        for extraction in result.get('extractions', []):
            chunk_id = extraction.get('chunk_id')

            # Add source_id to each entity
            for entity in extraction.get('entities', []):
                entity['source_id'] = chunk_id
                all_entities.append(entity)

            # Add source_id to each relation
            for relation in extraction.get('relations', []):
                relation['source_id'] = chunk_id
                all_relations.append(relation)

        return {
            'entities': all_entities,
            'relations': all_relations,
            'failed_chunks': result.get('failed_chunks', [])
        }