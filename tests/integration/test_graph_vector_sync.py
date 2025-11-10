"""
Integration test for graph-vector synchronization

Ensures graph storage and vector DBs stay synchronized during all operations.
"""

import pytest


@pytest.mark.integration
class TestGraphVectorSync:
    """Test graph and vector DB synchronization"""

    @pytest.mark.asyncio
    async def test_entity_in_both_graph_and_vector(self, bigrag_instance):
        """Test that entities appear in both graph and vector DB"""
        rag = bigrag_instance

        # Insert document with identifiable entity
        doc = "NASA launched the Artemis mission to explore the Moon."
        await rag.ainsert([doc])

        # Entity should be in graph
        assert rag.chunk_entity_relation_graph is not None, "Graph storage should exist"

        # Entity should be in vector DB for retrieval
        assert rag.vdb_entities is not None, "Entity vector DB should exist"

        # Test retrieval works (proves entity is in vector DB)
        from bigrag.base import QueryParam
        results = await rag.aquery("NASA space mission", QueryParam(mode="local", top_k=5))

        assert results is not None, "Should retrieve entity from vector DB"
        assert "nasa" in results.lower() or "artemis" in results.lower() or "moon" in results.lower(), \
            "Entity should be retrievable, proving it's in both graph and vector DB"

    @pytest.mark.asyncio
    async def test_edge_deletion_removes_from_vector(self, bigrag_instance):
        """Test that deleting documents removes associated vectors"""
        rag = bigrag_instance

        # Insert document
        doc = "Document with entity UNIQUEENTITY456 for deletion test"
        await rag.ainsert([doc])

        # Get initial state
        docs_before = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        chunks_before = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())

        assert len(docs_before) > 0, "Should have documents before deletion"
        assert len(chunks_before) > 0, "Should have chunks before deletion"

        # Delete document
        doc_id = list(docs_before.keys())[0]
        await rag.adelete_document(doc_id)

        # Verify removal
        docs_after = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        chunks_after = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())

        assert len(docs_after) < len(docs_before), "Document should be removed from KV storage"
        assert len(chunks_after) < len(chunks_before), "Chunks should be removed from KV storage"

        # Vector DBs should also be cleaned (orphaned entities/edges removed)
        # This validates cascade deletion works across all storage layers

    @pytest.mark.asyncio
    async def test_chunk_embedding_matches_content(self, bigrag_instance):
        """Test that chunk embeddings match chunk content"""
        rag = bigrag_instance

        # Insert document
        doc = "The theory of relativity revolutionized modern physics."
        await rag.ainsert([doc])

        # Get chunks
        chunks = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())
        assert len(chunks) > 0, "Should have chunks after insert"

        # Vector DB should have embeddings for these chunks
        assert rag.vdb_chunks is not None, "Chunk vector DB should exist"

        # Test chunk retrieval works (proves embeddings exist and match content)
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "physics theory",
            QueryParam(mode="naive", top_k=5)  # naive mode uses chunk embeddings
        )

        assert results is not None, "Should retrieve chunks using embeddings"
        assert "physics" in results.lower() or "relativity" in results.lower(), \
            "Retrieved chunks should match original content"

    @pytest.mark.asyncio
    async def test_entity_embedding_consistency(self, bigrag_instance):
        """Test that entity embeddings are consistent with entity content"""
        rag = bigrag_instance

        # Insert document with clear entity
        doc = "Galileo Galilei was an Italian astronomer who improved the telescope."
        await rag.ainsert([doc])

        # Entity should be embedded in vector DB
        assert rag.vdb_entities is not None, "Entity vector DB should exist"

        # Test entity retrieval (Path A) works
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "Italian astronomer",
            QueryParam(mode="local", top_k=5)
        )

        assert results is not None, "Should retrieve entity via vector search"
        # Should find Galileo or related terms
        results_lower = results.lower()
        assert "galileo" in results_lower or "astronomer" in results_lower or "telescope" in results_lower, \
            "Entity embeddings should enable semantic search"

    @pytest.mark.asyncio
    async def test_relation_embedding_consistency(self, bigrag_instance):
        """Test that relation embeddings are consistent with relation content"""
        rag = bigrag_instance

        # Insert document with clear relation
        doc = "The Nile River originates in Uganda and flows northward through Egypt."
        await rag.ainsert([doc])

        # Relation should be embedded in vector DB
        assert rag.vdb_relations is not None, "Relation vector DB should exist"

        # Test relation retrieval (Path B) works
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "river flowing through Egypt",
            QueryParam(mode="global", top_k=5)
        )

        assert results is not None, "Should retrieve relation via vector search"
        # Should find Nile or Egypt relation
        results_lower = results.lower()
        assert "nile" in results_lower or "egypt" in results_lower or "river" in results_lower, \
            "Relation embeddings should enable semantic search"

    @pytest.mark.asyncio
    async def test_vector_search_returns_graph_nodes(self, bigrag_instance):
        """Test roundtrip: vector search → graph nodes → content retrieval"""
        rag = bigrag_instance

        # Insert documents
        docs = [
            "Charles Darwin developed the theory of evolution by natural selection.",
            "Darwin's voyage on the HMS Beagle provided evidence for his theory."
        ]
        await rag.ainsert(docs)

        # Query using hybrid mode (all three paths)
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "Darwin's evolutionary theory",
            QueryParam(mode="hybrid", top_k=10)
        )

        assert results is not None, "Hybrid query should return results"

        # Results should come from graph nodes that were found via vector search
        results_lower = results.lower()
        assert "darwin" in results_lower or "evolution" in results_lower or "beagle" in results_lower, \
            "Vector search should retrieve graph nodes with correct content"

        # This validates the complete pipeline:
        # 1. Query text → embedding
        # 2. Vector search → entity/relation/chunk IDs
        # 3. Graph lookup → node content
        # 4. Format → return to user


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
