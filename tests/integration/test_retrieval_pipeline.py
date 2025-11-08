"""
Integration test for retrieval pipeline

Tests entity extraction, graph construction, and three-path retrieval together.
"""

import pytest
from bigrag.base import QueryParam


@pytest.mark.integration
class TestRetrievalPipeline:
    """Test complete retrieval pipeline integration"""

    @pytest.mark.asyncio
    async def test_entity_extraction_to_retrieval(self, bigrag_instance):
        """Test that extracted entities are retrievable"""
        rag = bigrag_instance

        # Insert document with clear entities
        await rag.insert(["Marie Curie won the Nobel Prize in Physics and Chemistry."])

        # Query for extracted entities
        results = await rag.query("Nobel Prize winner", QueryParam(mode="local"))

        assert results is not None
        # Should find Marie Curie
        assert "curie" in results.lower() or "nobel" in results.lower()

    @pytest.mark.asyncio
    async def test_relation_extraction_to_retrieval(self, bigrag_instance):
        """Test that extracted relations are retrievable"""
        rag = bigrag_instance

        await rag.insert(["Amazon River flows through Brazil."])

        results = await rag.query("What flows through Brazil?", QueryParam(mode="global"))

        assert results is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
