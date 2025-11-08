"""
Unit tests for bigrag.reranker module

Tests semantic reranking with cross-encoder.
"""

import pytest


class TestSemanticReranker:
    """Test semantic reranking functionality"""

    @pytest.mark.asyncio
    async def test_reranker_import(self):
        """Test that reranker module can be imported"""
        try:
            from bigrag.reranker import rerank_chunks
            assert rerank_chunks is not None
        except ImportError:
            pytest.skip("Reranker module requires sentence-transformers")

    @pytest.mark.asyncio
    async def test_rerank_chunks_basic(self):
        """Test basic chunk reranking"""
        try:
            from bigrag.reranker import rerank_chunks

            query = "What is machine learning?"
            chunks = [
                ("Machine learning is a subset of artificial intelligence.", ["chunk-1"]),
                ("Python is a programming language.", ["chunk-2"]),
                ("Deep learning uses neural networks.", ["chunk-3"]),
            ]

            # Rerank
            reranked = await rerank_chunks(query, chunks, top_k=2)

            # Should return top 2
            assert len(reranked) <= 2

            # Each result should have required fields
            for result in reranked:
                assert "content" in result
                assert "score" in result
                assert "sources" in result

        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.asyncio
    async def test_rerank_empty_chunks(self):
        """Test reranking with empty chunk list"""
        try:
            from bigrag.reranker import rerank_chunks

            query = "Test query"
            chunks = []

            reranked = await rerank_chunks(query, chunks, top_k=5)

            # Should return empty list
            assert len(reranked) == 0

        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.asyncio
    async def test_rerank_preserves_sources(self):
        """Test that reranking preserves source IDs"""
        try:
            from bigrag.reranker import rerank_chunks

            query = "artificial intelligence"
            chunks = [
                ("AI is transforming technology.", ["source-1", "source-2"]),
                ("Unrelated content.", ["source-3"]),
            ]

            reranked = await rerank_chunks(query, chunks, top_k=5)

            # Check that sources are preserved
            for result in reranked:
                assert isinstance(result["sources"], list)
                assert len(result["sources"]) > 0

        except ImportError:
            pytest.skip("sentence-transformers not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
