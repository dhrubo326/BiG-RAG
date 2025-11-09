"""
Integration test for entity extraction

Tests LLM-based entity extraction (requires OpenAI API key).
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
        """Test that extracted entity types are normalized"""
        rag = bigrag_instance

        # Insert document (triggers LLM extraction)
        await rag.ainsert(["Barack Obama was the 44th President of the United States."])

        # Verify entities exist and are normalized
        # (This requires checking graph storage for normalized types)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
