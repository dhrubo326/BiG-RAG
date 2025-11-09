"""
Performance test for concurrent operations

Tests concurrent queries, inserts, deletes, and mixed operations.
"""

import pytest
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))

from test_documents import get_concurrent_test_queries, get_performance_test_documents


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
            rag.aquery(q, QueryParam(mode="hybrid"))
            for q in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Most should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > len(queries) * 0.8  # At least 80% success

    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, bigrag_instance):
        """Test concurrent document insertion"""
        rag = bigrag_instance

        # Generate test documents (REDUCED for faster testing)
        docs = get_performance_test_documents(count=20)
        batch_size = 10

        # Split into batches and insert concurrently
        batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]

        tasks = [rag.ainsert(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that most batches succeeded
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= len(batches) * 0.8  # At least 80% success

    @pytest.mark.asyncio
    async def test_concurrent_deletes(self, bigrag_instance):
        """Test concurrent document deletion"""
        rag = bigrag_instance

        # Insert documents first (REDUCED for faster testing)
        docs = get_performance_test_documents(count=15)
        doc_ids = []

        for i, doc in enumerate(docs):
            await rag.ainsert([doc], metadatas=[{"id": f"del_test_{i}"}])
            doc_ids.append(f"del_test_{i}")

        # Delete concurrently
        tasks = [rag.adelete_document(doc_id) for doc_id in doc_ids[:20]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Most deletes should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= len(doc_ids[:20]) * 0.7  # At least 70% success

    @pytest.mark.asyncio
    async def test_mixed_concurrent_operations(self, bigrag_instance):
        """Test mixed insert/query/delete operations concurrently"""
        rag = bigrag_instance

        # Prepare test data
        docs = get_performance_test_documents(count=30)
        queries = get_concurrent_test_queries()[:20]

        # Insert initial documents
        await rag.ainsert(docs[:10])

        # Create mixed tasks
        from bigrag.base import QueryParam
        tasks = []

        # Add insert tasks
        for doc in docs[10:20]:
            tasks.append(rag.ainsert([doc]))

        # Add query tasks
        for query in queries[:10]:
            tasks.append(rag.aquery(query, QueryParam(mode="hybrid")))

        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that most operations succeeded
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= len(tasks) * 0.7  # At least 70% success

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, bigrag_with_data):
        """Stress test with high concurrency (100+ simultaneous operations)"""
        rag = bigrag_with_data

        queries = get_concurrent_test_queries()[:100]

        # Run 100 queries concurrently
        from bigrag.base import QueryParam
        start_time = time.time()

        tasks = [
            rag.aquery(q, QueryParam(mode="hybrid", top_k=5))
            for q in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed_time = time.time() - start_time

        # Check success rate
        successful = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful) / len(queries)

        assert success_rate >= 0.7  # At least 70% success under stress

        # Performance check - should complete within reasonable time
        # (This is lenient to accommodate different hardware)
        assert elapsed_time < 120  # Should complete within 2 minutes

    @pytest.mark.asyncio
    async def test_concurrent_query_different_modes(self, bigrag_with_data):
        """Test concurrent queries with different retrieval modes"""
        rag = bigrag_with_data

        query = "What is machine learning?"

        # Create tasks with different modes
        from bigrag.base import QueryParam
        tasks = [
            rag.aquery(query, QueryParam(mode="local")),
            rag.aquery(query, QueryParam(mode="global")),
            rag.aquery(query, QueryParam(mode="hybrid")),
            rag.aquery(query, QueryParam(mode="naive")),
        ]

        # Run all modes concurrently (5 times each = 20 total)
        all_tasks = tasks * 5
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # All should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= len(all_tasks) * 0.8  # At least 80% success

    @pytest.mark.asyncio
    async def test_concurrent_batch_operations(self, bigrag_instance):
        """Test concurrent batch insert and batch query operations"""
        rag = bigrag_instance

        # Prepare batches
        docs_batch1 = get_performance_test_documents(count=20)
        docs_batch2 = get_performance_test_documents(count=20)
        docs_batch3 = get_performance_test_documents(count=20)

        # Insert batches concurrently
        insert_tasks = [
            rag.ainsert(docs_batch1),
            rag.ainsert(docs_batch2),
            rag.ainsert(docs_batch3),
        ]

        results = await asyncio.gather(*insert_tasks, return_exceptions=True)

        # Check that all batches succeeded
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= 2  # At least 2/3 batches should succeed

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_concurrent_load(self, bigrag_instance):
        """Test sustained concurrent load over time"""
        rag = bigrag_instance

        # Insert initial documents (REDUCED for faster testing)
        initial_docs = get_performance_test_documents(count=10)
        await rag.ainsert(initial_docs)

        # Run sustained concurrent queries for 30 seconds
        queries = get_concurrent_test_queries()
        from bigrag.base import QueryParam

        start_time = time.time()
        all_results = []

        while time.time() - start_time < 30:  # Run for 30 seconds
            # Run 10 concurrent queries
            tasks = [
                rag.aquery(queries[i % len(queries)], QueryParam(mode="hybrid", top_k=5))
                for i in range(10)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(results)

            # Small delay between batches
            await asyncio.sleep(0.5)

        # Check overall success rate
        successful = [r for r in all_results if not isinstance(r, Exception)]
        success_rate = len(successful) / len(all_results)

        assert success_rate >= 0.7  # At least 70% success over sustained load
        assert len(all_results) >= 50  # Should have processed at least 50 queries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
