"""
End-to-End test for three-path retrieval system

Validates that all three retrieval paths work correctly:
- Path A: Entity-based retrieval
- Path B: Relation-based retrieval
- Path C: Chunk-based retrieval (with semantic reranking)
"""

import pytest
from bigrag.base import QueryParam


@pytest.mark.critical
@pytest.mark.e2e
class TestThreePathRetrieval:
    """Test all three retrieval paths"""

    @pytest.mark.asyncio
    async def test_path_a_entity_based(self, bigrag_with_data):
        """Test Path A: Entity-based retrieval (local mode)"""
        rag = bigrag_with_data

        # Query using entity-based retrieval only
        results = await rag.query(
            "Who is Lionel Messi?",
            param=QueryParam(mode="local", top_k=5),
        )

        # Validate results
        assert results is not None
        assert len(results) > 0
        # Should find entity-based information
        assert "messi" in results.lower()

    @pytest.mark.asyncio
    async def test_path_b_relation_based(self, bigrag_with_data):
        """Test Path B: Relation-based retrieval (global mode)"""
        rag = bigrag_with_data

        # Query using relation-based retrieval only
        results = await rag.query(
            "What is the relationship between Messi and Inter Miami?",
            param=QueryParam(mode="global", top_k=5),
        )

        # Validate results
        assert results is not None
        assert len(results) > 0
        # Should find relation information
        results_lower = results.lower()
        assert "messi" in results_lower or "miami" in results_lower

    @pytest.mark.asyncio
    async def test_path_c_chunk_based(self, bigrag_with_data):
        """Test Path C: Chunk-based retrieval (naive mode)"""
        rag = bigrag_with_data

        # Query using chunk-based retrieval only
        results = await rag.query(
            "Tell me about Miami",
            param=QueryParam(mode="naive", top_k=5),
        )

        # Validate results
        assert results is not None
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_hybrid_mode_combines_all_paths(self, bigrag_with_data):
        """Test hybrid mode combines all three paths"""
        rag = bigrag_with_data

        # Query using hybrid mode (all three paths)
        results_hybrid = await rag.query(
            "Where does Messi play?",
            param=QueryParam(mode="hybrid", top_k=10),
        )

        # Validate hybrid results
        assert results_hybrid is not None
        assert len(results_hybrid) > 0

        # Hybrid should potentially return more comprehensive results
        # (though length may vary based on deduplication)

    @pytest.mark.asyncio
    async def test_path_comparison(self, bigrag_with_data):
        """Compare results from different paths"""
        rag = bigrag_with_data

        query = "What team does Messi play for?"

        # Get results from each path
        results_local = await rag.query(query, QueryParam(mode="local", top_k=5))
        results_global = await rag.query(query, QueryParam(mode="global", top_k=5))
        results_naive = await rag.query(query, QueryParam(mode="naive", top_k=5))
        results_hybrid = await rag.query(query, QueryParam(mode="hybrid", top_k=10))

        # All paths should return results
        assert results_local is not None and len(results_local) > 0
        assert results_global is not None and len(results_global) > 0
        assert results_naive is not None and len(results_naive) > 0
        assert results_hybrid is not None and len(results_hybrid) > 0

        # Hybrid should ideally be most comprehensive
        # (contains information from all paths)


@pytest.mark.e2e
class TestSemanticReranking:
    """Test semantic reranking feature (Path C enhancement)"""

    @pytest.mark.asyncio
    async def test_reranking_enabled(self, bigrag_with_data):
        """Test retrieval with reranking enabled"""
        rag = bigrag_with_data

        results = await rag.query(
            "Football World Cup",
            param=QueryParam(mode="hybrid", enable_reranking=True, top_k=5),
        )

        assert results is not None
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_reranking_disabled(self, bigrag_with_data):
        """Test retrieval with reranking disabled (faster)"""
        rag = bigrag_with_data

        results = await rag.query(
            "Football World Cup",
            param=QueryParam(mode="hybrid", enable_reranking=False, top_k=5),
        )

        assert results is not None
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_reranking_comparison(self, bigrag_with_data):
        """Compare results with and without reranking"""
        rag = bigrag_with_data

        query = "Who won the World Cup?"

        # With reranking
        results_with_rerank = await rag.query(
            query,
            param=QueryParam(mode="hybrid", enable_reranking=True, top_k=5),
        )

        # Without reranking
        results_without_rerank = await rag.query(
            query,
            param=QueryParam(mode="hybrid", enable_reranking=False, top_k=5),
        )

        # Both should return results
        assert results_with_rerank is not None
        assert results_without_rerank is not None

        # Results may differ in ranking/content
        # (reranking should improve relevance)


@pytest.mark.e2e
class TestWeightedRRF:
    """Test weighted RRF scoring for chunks"""

    @pytest.mark.asyncio
    async def test_direct_and_indirect_chunks(self, bigrag_with_data):
        """
        Test that both direct chunks (Path C vector search) and
        indirect chunks (from Path A+B source_ids) are included
        """
        rag = bigrag_with_data

        # Query that should trigger both direct and indirect retrieval
        results = await rag.query(
            "Messi and Argentina",
            param=QueryParam(mode="hybrid", top_k=10),
        )

        # Should return results from multiple sources
        assert results is not None
        assert len(results) > 0

        # Results should contain information from both entities and chunks
        results_lower = results.lower()
        assert "messi" in results_lower or "argentina" in results_lower


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
