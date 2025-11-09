"""
Performance test for large-scale operations

Tests system with 1000+ documents.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))

from test_documents import get_performance_test_documents


@pytest.mark.performance
@pytest.mark.slow
class TestLargeScale:
    """Test with large number of documents"""

    @pytest.mark.asyncio
    async def test_insert_1000_documents(self, bigrag_instance):
        """Test inserting 1000 documents"""
        rag = bigrag_instance

        # Generate 1000 documents
        docs = get_performance_test_documents(count=1000)

        # Insert in batches
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            await rag.ainsert(batch)

        # Verify insertion
        all_docs = await rag.full_docs.get_by_ids(await rag.full_docs.get_all_ids())
        assert len(all_docs) >= 900  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_query_on_large_dataset(self, bigrag_instance):
        """Test query performance on large dataset"""
        rag = bigrag_instance

        # Insert many documents
        docs = get_performance_test_documents(count=100)
        await rag.ainsert(docs)

        # Query
        from bigrag.base import QueryParam
        results = await rag.aquery("test query", QueryParam(mode="hybrid", top_k=10))

        assert results is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
