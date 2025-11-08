"""
Performance test for concurrent operations

Tests concurrent queries and operations.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))

from test_documents import get_concurrent_test_queries


@pytest.mark.performance
class TestConcurrency:
    """Test concurrent operations"""

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, bigrag_with_data):
        """Test multiple concurrent queries"""
        rag = bigrag_with_data

        queries = get_concurrent_test_queries()[:50]  # 50 queries

        # Run queries concurrently
        from bigrag.base import QueryParam
        tasks = [
            rag.query(q, QueryParam(mode="hybrid"))
            for q in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Most should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > len(queries) * 0.8  # At least 80% success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
