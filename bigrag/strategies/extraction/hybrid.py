from bigrag.interfaces.extractor import ExtractorInterface
from typing import List, Dict

class HybridExtractor(ExtractorInterface):
    def __init__(self, api_key: str, gleaning_iterations: int = 2, concurrency: int = 16, enable_validation: bool = True):
        from bigrag.extractors.table_fact_extractor import TableFactExtractor
        from bigrag.extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
        self.table_extractor = TableFactExtractor()  # FIX: Instantiate class (was missing parentheses)
        self.paragraph_extractor = ConstrainedLLMExtractor(api_key=api_key, enable_gleaning=True, max_gleaning_iterations=gleaning_iterations, enable_numeric_validation=enable_validation)
        self.batch_extractor = BatchConstrainedExtractor(self.paragraph_extractor)
    async def extract(self, chunks: List[Dict]) -> Dict:
        """
        Extract using both table-specific and paragraph extraction.

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
        table_chunks = [c for c in chunks if c.get("type") == "table"]
        para_chunks = [c for c in chunks if c.get("type") == "paragraph"]

        # Build chunk index for quick lookup
        chunk_index = {chunk['chunk_id']: chunk for chunk in chunks}
        chunks_with_extractions = []  # NEW: For numeric validation

        # Extract from tables
        table_entities, table_relations = [], []
        for chunk in table_chunks:
            chunk_id = chunk['chunk_id']
            chunk_table_entities = []
            chunk_table_relations = []

            try:
                result = self.table_extractor.extract_facts_from_table(
                    chunk.get("structured_data", {}),
                    chunk_id
                )
                chunk_table_entities = result.get("entities", [])
                chunk_table_relations = result.get("relations", [])
                table_entities.extend(chunk_table_entities)
                table_relations.extend(chunk_table_relations)
            except:
                pass

            # Build chunk with extractions for numeric validation
            chunks_with_extractions.append({
                'chunk_id': chunk_id,
                'content': chunk.get('content', ''),
                'entities': chunk_table_entities,
                'relations': chunk_table_relations
            })

        # Extract from paragraphs
        para_result = await self.batch_extractor.extract_from_chunks(para_chunks)

        # Flatten paragraph extractions
        # IMPORTANT: Add source_id (chunk_id) to each entity/relation
        para_entities = []
        para_relations = []
        for extraction in para_result.get('extractions', []):
            chunk_id = extraction.get('chunk_id')
            chunk_para_entities = []
            chunk_para_relations = []

            # Add source_id to each entity
            for entity in extraction.get('entities', []):
                entity['source_id'] = chunk_id
                para_entities.append(entity)
                chunk_para_entities.append(entity)

            # Add source_id to each relation
            for relation in extraction.get('relations', []):
                relation['source_id'] = chunk_id
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

        return {
            'entities': table_entities + para_entities,
            'relations': table_relations + para_relations,
            'failed_chunks': para_result.get('failed_chunks', []),
            'chunks': chunks_with_extractions  # NEW: Enable numeric validation
        }