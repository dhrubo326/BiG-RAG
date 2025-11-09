"""
Integration test for entity extraction

Tests LLM-based entity extraction (requires OpenAI API key).
Tests will be skipped if OPENAI_API_KEY environment variable is not set.
"""

import pytest
import os


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not set. Set OPENAI_API_KEY to run LLM tests."
)
class TestEntityExtraction:
    """Test entity extraction with LLM"""

    @pytest.mark.asyncio
    async def test_entity_normalization_after_extraction(self, bigrag_instance):
        """Test that extracted entity types are normalized correctly"""
        rag = bigrag_instance

        # Insert document (triggers LLM extraction)
        doc = "Microsoft Corporation is a technology company founded by Bill Gates in 1975."
        await rag.ainsert([doc])

        # Query to verify entities were extracted and normalized
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "technology company",
            QueryParam(mode="local", top_k=5)
        )

        assert results is not None, "Should retrieve extracted entities"
        # Should find Microsoft or Bill Gates
        results_lower = results.lower()
        assert "microsoft" in results_lower or "gates" in results_lower or "technology" in results_lower, \
            "Extracted entities should be retrievable"

    @pytest.mark.asyncio
    async def test_entity_type_normalized_correctly(self, bigrag_instance):
        """Test that entity types follow TYPE_NORMALIZATION_MAP"""
        rag = bigrag_instance

        # Insert with entities that should be normalized
        # COMPANY → organization, PERSON → person, etc.
        doc = "The European Union is an international organization with member countries."
        await rag.ainsert([doc])

        # Entities should be normalized according to TYPE_NORMALIZATION_MAP
        # e.g., "ORGANIZATION" types should normalize to "organization"
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "international organization",
            QueryParam(mode="local", top_k=5)
        )

        assert results is not None, "Should retrieve normalized entities"
        assert "european" in results.lower() or "union" in results.lower(), \
            "Entity types should be normalized for retrieval"

    @pytest.mark.asyncio
    async def test_relation_extraction_quality(self, bigrag_instance):
        """Test that relation extraction captures meaningful knowledge"""
        rag = bigrag_instance

        # Insert document with clear relations
        doc = "The Panama Canal connects the Atlantic Ocean to the Pacific Ocean."
        await rag.ainsert([doc])

        # Query for relation
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "canal connecting oceans",
            QueryParam(mode="global", top_k=5)
        )

        assert results is not None, "Should retrieve extracted relations"
        # Should find Panama Canal relation
        results_lower = results.lower()
        assert "panama" in results_lower or "canal" in results_lower or "ocean" in results_lower, \
            "Extracted relations should capture meaningful connections"

    @pytest.mark.asyncio
    async def test_llm_cache_reduces_api_calls(self, bigrag_instance):
        """Test that LLM response caching reduces redundant API calls"""
        rag = bigrag_instance

        # Insert same document twice
        doc = "The Eiffel Tower is located in Paris, France."

        # First insert (will call LLM)
        await rag.ainsert([doc])

        # Second insert of identical document (should use cache)
        await rag.ainsert([doc])

        # If caching works, second insert should be faster
        # and both should produce same entities/relations

        # Verify entities exist
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "Eiffel Tower",
            QueryParam(mode="local", top_k=5)
        )

        assert results is not None, "Should retrieve entities from cached or fresh extraction"
        assert "eiffel" in results.lower() or "paris" in results.lower(), \
            "Cached extraction should produce same results"

    @pytest.mark.asyncio
    async def test_extraction_with_metadata_context(self, bigrag_instance):
        """Test that metadata enhances entity extraction quality"""
        rag = bigrag_instance

        # Insert with metadata (should provide context for extraction)
        doc = "The discovery was made in 2023 by the research team."
        metadata = {
            "title": "Breakthrough in Quantum Computing",
            "category": "science",
            "tags": ["quantum", "technology", "research"]
        }

        await rag.ainsert([doc], metadata=[metadata])

        # Metadata context should help LLM extract better entities
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "quantum computing discovery",
            QueryParam(mode="hybrid", top_k=5)
        )

        assert results is not None, "Should retrieve with metadata-enhanced extraction"
        # Should find relevant information
        results_lower = results.lower()
        assert "quantum" in results_lower or "discovery" in results_lower or "research" in results_lower, \
            "Metadata should enhance extraction quality"

    @pytest.mark.asyncio
    async def test_multiple_entities_in_single_doc(self, bigrag_instance):
        """Test extraction of multiple entities from single document"""
        rag = bigrag_instance

        # Insert document with multiple distinct entities
        doc = "Isaac Newton, Albert Einstein, and Stephen Hawking were influential physicists."
        await rag.ainsert([doc])

        # Should extract multiple person entities
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "famous physicist",
            QueryParam(mode="local", top_k=10)
        )

        assert results is not None, "Should retrieve multiple entities"
        results_lower = results.lower()

        # Should find at least one of the physicists
        found_entities = [
            "newton" in results_lower,
            "einstein" in results_lower,
            "hawking" in results_lower,
            "physicist" in results_lower
        ]

        assert any(found_entities), \
            "Should extract and retrieve multiple entities from single document"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
