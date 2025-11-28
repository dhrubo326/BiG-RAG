"""
LLM-based entity/relation extraction module.

Wrapper around ConstrainedLLMExtractor with batch processing support.
"""

from typing import List, Dict, Tuple
from ...extractors.constrained_extractor import ConstrainedLLMExtractor, BatchConstrainedExtractor
from ...utils import logger


class LLMExtractor:
    """
    LLM-based entity and relation extractor.

    Thin wrapper around ConstrainedLLMExtractor with gleaning support.
    Handles batch extraction from multiple chunks.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        enable_gleaning: bool = True,
        max_iterations: int = 2,
        concurrency: int = 16,
        enable_table_facts: bool = False,
        hitl_store=None
    ):
        self.api_key = api_key
        self.model = model
        self.enable_gleaning = enable_gleaning
        self.max_iterations = max_iterations
        self.concurrency = concurrency
        self.enable_table_facts = enable_table_facts
        self.hitl_store = hitl_store

        # Initialize underlying ConstrainedLLMExtractor
        self.base_extractor = ConstrainedLLMExtractor(
            api_key=api_key,
            model=model,
            enable_gleaning=enable_gleaning,
            max_gleaning_iterations=max_iterations,
            hitl_store=hitl_store
        )

        # Initialize batch extractor for parallel processing
        self.batch_extractor = BatchConstrainedExtractor(self.base_extractor)

    async def extract(
        self,
        chunks: List[Dict],
        metadata: Dict = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relations from chunks.

        Args:
            chunks: List of chunk dicts from chunker
            metadata: Optional document metadata

        Returns:
            Tuple of (entities, relations) where:
            - entities: List of entity dicts
            - relations: List of relation dicts
        """
        metadata = metadata or {}

        # Extract text content from chunks
        chunk_texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk.get('content', ''))
            else:
                chunk_texts.append(str(chunk))

        logger.info(f"[LLMExtractor] Processing {len(chunk_texts)} chunks with concurrency={self.concurrency}")

        try:
            # Use batch extractor for parallel processing
            # Method is extract_from_chunks(chunks, concurrency)
            results = await self.batch_extractor.extract_from_chunks(
                chunks=chunk_texts,
                concurrency=self.concurrency
            )

            # Aggregate entities and relations from all chunks
            all_entities = []
            all_relations = []

            for result in results:
                if result and isinstance(result, dict):
                    all_entities.extend(result.get('entities', []))
                    all_relations.extend(result.get('relations', []))

            logger.info(f"[LLMExtractor] Extracted {len(all_entities)} entities, {len(all_relations)} relations")

            return all_entities, all_relations

        except Exception as e:
            logger.error(f"[LLMExtractor] Extraction failed: {e}")
            # Return empty results on failure
            return [], []
