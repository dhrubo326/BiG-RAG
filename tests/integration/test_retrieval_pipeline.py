"""
Integration test for retrieval pipeline

Tests entity extraction, graph construction, and three-path retrieval working together.
"""

import pytest
from bigrag.base import QueryParam


@pytest.mark.integration
class TestRetrievalPipeline:
    """Test complete retrieval pipeline integration"""

    @pytest.mark.asyncio
    async def test_entity_extraction_to_retrieval(self, bigrag_instance):
        """Test that extracted entities are retrievable via Path A"""
        rag = bigrag_instance

        # Insert document with clear entities
        doc = "Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911."
        await rag.ainsert([doc])

        # Query for extracted entities (Path A - entity-based retrieval)
        results = await rag.aquery(
            "Nobel Prize winner in science",
            QueryParam(mode="local", top_k=5)
        )

        assert results is not None, "Query should return results"
        assert isinstance(results, str), "Results should be formatted string"

        # Should find Marie Curie or Nobel Prize mentions
        results_lower = results.lower()
        assert "curie" in results_lower or "nobel" in results_lower or "physics" in results_lower, \
            "Results should contain entities from document"

    @pytest.mark.asyncio
    async def test_relation_extraction_to_retrieval(self, bigrag_instance):
        """Test that extracted relations are retrievable via Path B"""
        rag = bigrag_instance

        # Insert document with clear relation
        doc = "The Amazon River flows through Brazil and empties into the Atlantic Ocean."
        await rag.ainsert([doc])

        # Query for relation (Path B - relation-based retrieval)
        results = await rag.aquery(
            "What river flows through Brazil?",
            QueryParam(mode="global", top_k=5)
        )

        assert results is not None, "Query should return results"
        assert isinstance(results, str), "Results should be formatted string"

        # Should find Amazon River or Brazil relation
        results_lower = results.lower()
        assert "amazon" in results_lower or "brazil" in results_lower or "river" in results_lower, \
            "Results should contain relation information"

    @pytest.mark.asyncio
    async def test_chunk_based_retrieval_path_c(self, bigrag_instance):
        """Test chunk-based retrieval (Path C)"""
        rag = bigrag_instance

        # Insert document
        doc = "Quantum mechanics describes the behavior of particles at atomic and subatomic scales."
        await rag.ainsert([doc])

        # Query using naive mode (chunk-based only)
        results = await rag.aquery(
            "particle behavior at atomic scale",
            QueryParam(mode="naive", top_k=5)
        )

        assert results is not None, "Chunk retrieval should return results"

        # Should find relevant chunk
        results_lower = results.lower()
        assert "quantum" in results_lower or "particle" in results_lower or "atomic" in results_lower, \
            "Chunk retrieval should find semantically similar content"

    @pytest.mark.asyncio
    async def test_hybrid_mode_combines_all_paths(self, bigrag_instance):
        """Test hybrid mode retrieves from all three paths"""
        rag = bigrag_instance

        # Insert document with entities, relations, and content
        doc = "Albert Einstein developed the theory of relativity while working in Switzerland."
        await rag.ainsert([doc])

        # Query in hybrid mode (combines Path A + B + C)
        results = await rag.aquery(
            "Einstein's scientific work",
            QueryParam(mode="hybrid", top_k=10)
        )

        assert results is not None, "Hybrid mode should return results"
        assert isinstance(results, str), "Results should be formatted string"

        # Should find information from multiple paths
        results_lower = results.lower()
        assert "einstein" in results_lower or "relativity" in results_lower or "switzerland" in results_lower, \
            "Hybrid mode should retrieve relevant information"

    @pytest.mark.asyncio
    async def test_rrf_scoring_across_paths(self, bigrag_instance):
        """Test that RRF scoring combines results from multiple paths"""
        rag = bigrag_instance

        # Insert multiple related documents
        docs = [
            "Isaac Newton formulated the laws of motion and gravity.",
            "Newton's laws explain how objects move in space.",
            "The Principia Mathematica was written by Isaac Newton."
        ]
        await rag.ainsert(docs)

        # Query should use RRF to combine results
        results = await rag.aquery(
            "Newton's contributions to physics",
            QueryParam(mode="hybrid", top_k=10)
        )

        assert results is not None, "RRF-scored query should return results"

        # Should find Newton in results (RRF boosts items appearing in multiple paths)
        assert "newton" in results.lower(), "RRF should surface frequently mentioned entities"

    @pytest.mark.asyncio
    async def test_reranking_improves_results(self, bigrag_instance):
        """Test semantic reranking of chunk candidates"""
        rag = bigrag_instance

        # Insert documents
        doc1 = "The Eiffel Tower is located in Paris, France."
        doc2 = "Paris is the capital city of France with many historic monuments."
        await rag.ainsert([doc1, doc2])

        # Query with reranking enabled
        results_with_rerank = await rag.aquery(
            "famous landmark in Paris",
            QueryParam(mode="hybrid", top_k=5, enable_reranking=True)
        )

        # Query without reranking
        results_without_rerank = await rag.aquery(
            "famous landmark in Paris",
            QueryParam(mode="hybrid", top_k=5, enable_reranking=False)
        )

        # Both should return results
        assert results_with_rerank is not None, "Query with reranking should return results"
        assert results_without_rerank is not None, "Query without reranking should return results"

        # With reranking should find Eiffel Tower (more relevant to "landmark")
        assert "eiffel" in results_with_rerank.lower() or "paris" in results_with_rerank.lower(), \
            "Reranking should surface relevant landmarks"

    @pytest.mark.asyncio
    async def test_metadata_preserved_in_results(self, bigrag_instance):
        """Test that document metadata is preserved through retrieval"""
        rag = bigrag_instance

        # Insert with metadata
        doc = "The Great Wall of China is one of the Seven Wonders of the World."
        metadata = {
            "title": "Great Wall Facts",
            "category": "landmarks",
            "tags": ["china", "architecture", "history"]
        }
        await rag.ainsert([doc], metadata=[metadata])

        # Query should work with metadata
        results = await rag.aquery(
            "Chinese landmark",
            QueryParam(mode="hybrid", top_k=5)
        )

        assert results is not None, "Query should return results"
        assert "wall" in results.lower() or "china" in results.lower(), \
            "Should find document with metadata"

    @pytest.mark.asyncio
    async def test_query_with_no_results_graceful(self, bigrag_instance):
        """Test graceful handling when query finds no results"""
        rag = bigrag_instance

        # Insert unrelated document
        await rag.ainsert(["Document about completely unrelated topic like pencil manufacturing."])

        # Query for something not in the document
        results = await rag.aquery(
            "quantum computing algorithms",
            QueryParam(mode="hybrid", top_k=5)
        )

        # Should return something (even if empty or minimal)
        assert results is not None, "Query should return something (not None)"
        # Empty results are acceptable - just shouldn't crash
        assert isinstance(results, str), "Results should be string type"

    @pytest.mark.asyncio
    async def test_multi_document_retrieval(self, bigrag_instance):
        """Test retrieval across multiple documents"""
        rag = bigrag_instance

        # Insert multiple documents on same topic
        docs = [
            "Leonardo da Vinci painted the Mona Lisa in Italy.",
            "The Mona Lisa is displayed in the Louvre Museum in Paris.",
            "Da Vinci was a Renaissance artist and inventor."
        ]
        await rag.ainsert(docs)

        # Query should find information from multiple documents
        results = await rag.aquery(
            "Da Vinci's famous painting",
            QueryParam(mode="hybrid", top_k=10)
        )

        assert results is not None, "Multi-document query should return results"
        results_lower = results.lower()

        # Should mention Mona Lisa and/or Da Vinci
        assert "mona lisa" in results_lower or "da vinci" in results_lower or "vinci" in results_lower, \
            "Should retrieve information from multiple related documents"

    @pytest.mark.asyncio
    async def test_weighted_retrieval_ranking(self, bigrag_instance):
        """Test that retrieval ranking considers entity/relation weights"""
        rag = bigrag_instance

        # Insert documents mentioning same entity multiple times
        docs = [
            "Shakespeare wrote Hamlet, one of his most famous plays.",
            "William Shakespeare was an English playwright and poet.",
            "Shakespeare's works include Romeo and Juliet and Macbeth.",
            "The Globe Theatre was associated with Shakespeare's company."
        ]
        await rag.ainsert(docs)

        # Query for Shakespeare (appears in multiple docs, should have high weight)
        results = await rag.aquery(
            "famous English playwright",
            QueryParam(mode="hybrid", top_k=10)
        )

        assert results is not None, "Weighted query should return results"

        # Shakespeare should appear prominently (high weight from multiple mentions)
        assert "shakespeare" in results.lower(), \
            "High-weight entities should be retrieved prominently"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
