"""
Edge case tests for BiG-RAG

Tests unusual inputs, error handling, and edge conditions.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))

from test_documents import get_edge_case_documents


@pytest.mark.edge_cases
class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_document(self, bigrag_instance):
        """Test inserting empty document"""
        rag = bigrag_instance

        # Should handle gracefully
        await rag.insert([""])

    @pytest.mark.asyncio
    async def test_very_long_document(self, bigrag_instance):
        """Test inserting very long document"""
        rag = bigrag_instance

        long_doc = " ".join(["word"] * 5000)
        await rag.insert([long_doc])

    @pytest.mark.asyncio
    async def test_special_characters(self, bigrag_instance):
        """Test documents with special characters"""
        rag = bigrag_instance

        special_docs = [
            "@#$%^&*()_+-=[]{}|;:',.<>?/~`",
            "Unicode: 你好世界 مرحبا",
        ]

        await rag.insert(special_docs)

    @pytest.mark.asyncio
    async def test_query_with_no_results(self, bigrag_with_data):
        """Test query that should return no results"""
        rag = bigrag_with_data

        from bigrag.base import QueryParam
        results = await rag.query(
            "NONEXISTENT_QUERY_12345_RANDOM",
            QueryParam(mode="hybrid"),
        )

        # Should return empty or minimal results, not crash
        assert results is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, bigrag_instance):
        """Test deleting document that doesn't exist"""
        rag = bigrag_instance

        # Should handle gracefully
        try:
            await rag.delete_document("nonexistent-doc-id")
        except Exception:
            # May raise exception or return gracefully
            pass

    @pytest.mark.asyncio
    async def test_all_edge_case_documents(self, bigrag_instance):
        """Test all edge case documents from fixtures"""
        rag = bigrag_instance

        edge_docs = get_edge_case_documents()

        # Should handle all without crashing
        for doc in edge_docs:
            try:
                await rag.insert([doc])
            except Exception as e:
                # Document what fails
                pytest.fail(f"Failed to insert edge case document: {doc[:50]}... Error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
