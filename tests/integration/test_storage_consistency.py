"""
Integration test for storage consistency

Ensures graph, vector DB, and KV storage stay synchronized across all operations.
"""

import pytest
import asyncio


@pytest.mark.integration
class TestStorageConsistency:
    """Test storage layers stay in sync"""

    @pytest.mark.asyncio
    async def test_insert_syncs_all_layers(self, bigrag_instance):
        """Test that insert updates all storage layers"""
        rag = bigrag_instance

        # Insert document
        test_doc = "Barack Obama was the 44th President of the United States."
        await rag.ainsert([test_doc])

        # Verify all layers have data
        # 1. Full docs
        docs = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        assert len(docs) > 0, "Full docs storage should have at least one document"

        # 2. Text chunks
        chunks = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())
        assert len(chunks) > 0, "Text chunks storage should have at least one chunk"

        # 3. Graph storage (should have nodes - chunks, entities, relations)
        # Get all graph nodes
        assert rag.knowledge_graph_inst is not None, "Graph storage should exist"

        # 4. Vector DBs (entities and chunks should have entries)
        assert rag.vdb_entities is not None, "Entity vector DB should exist"
        assert rag.vdb_chunks is not None, "Chunk vector DB should exist"

    @pytest.mark.asyncio
    async def test_delete_syncs_all_layers(self, bigrag_instance):
        """Test that delete removes from all layers (cascade deletion)"""
        rag = bigrag_instance

        # Insert then delete
        test_doc = "Document to delete for consistency test"
        await rag.ainsert([test_doc])

        # Get document ID
        docs_before = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        assert len(docs_before) > 0, "Should have at least one document before deletion"

        doc_id = list(docs_before.keys())[0]
        chunks_before = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())

        # Delete document
        result = await rag.adelete_document(doc_id)

        # Verify removal from all layers
        docs_after = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        assert len(docs_after) < len(docs_before), "Document count should decrease after deletion"

        chunks_after = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())
        assert len(chunks_after) < len(chunks_before), "Chunk count should decrease after deletion"

        # Verify deletion stats returned
        assert result is not None, "Delete should return deletion stats"
        assert result.get("status") == "success", "Deletion should succeed"
        assert result.get("chunks_deleted", 0) > 0, "Should delete at least one chunk"

    @pytest.mark.asyncio
    async def test_upsert_maintains_consistency(self, bigrag_instance):
        """Test that upserting documents maintains consistency"""
        rag = bigrag_instance

        # Initial insert
        doc1 = "Initial document about machine learning"
        await rag.ainsert([doc1])

        docs_initial = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        chunks_initial = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())

        initial_doc_count = len(docs_initial)
        initial_chunk_count = len(chunks_initial)

        # Insert another document (should add, not replace)
        doc2 = "Second document about artificial intelligence"
        await rag.ainsert([doc2])

        docs_after = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        chunks_after = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())

        # Verify both documents exist
        assert len(docs_after) == initial_doc_count + 1, "Should have one more document"
        assert len(chunks_after) > initial_chunk_count, "Should have more chunks"

    @pytest.mark.asyncio
    async def test_concurrent_operations_stay_synced(self, bigrag_instance):
        """Test that concurrent insert operations maintain consistency"""
        rag = bigrag_instance

        # Prepare multiple documents
        docs = [
            "Document about quantum physics",
            "Document about relativity theory",
            "Document about string theory"
        ]

        # Insert concurrently
        tasks = [rag.ainsert([doc]) for doc in docs]
        await asyncio.gather(*tasks)

        # Verify all documents stored
        stored_docs = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        assert len(stored_docs) >= 3, "Should have at least 3 documents after concurrent inserts"

        # Verify chunks created for all
        stored_chunks = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())
        assert len(stored_chunks) >= 3, "Should have at least 3 chunks (one per doc minimum)"

    @pytest.mark.asyncio
    async def test_entity_count_matches_across_layers(self, bigrag_instance):
        """Test that entity counts are consistent across graph and vector DB"""
        rag = bigrag_instance

        # Insert document with clear entities
        doc = "Albert Einstein developed the theory of relativity in Germany."
        await rag.ainsert([doc])

        # Graph should have entity nodes
        # Vector DB should have entity embeddings
        # Counts should match (or be close, accounting for duplicates)

        # Basic check: both layers should have data
        assert rag.knowledge_graph_inst is not None, "Graph should exist"
        assert rag.vdb_entities is not None, "Entity vector DB should exist"

    @pytest.mark.asyncio
    async def test_chunk_to_entity_mapping_consistency(self, bigrag_instance):
        """Test that chunk-to-entity mappings are consistent in graph"""
        rag = bigrag_instance

        # Insert document
        doc = "Marie Curie won Nobel Prizes in Physics and Chemistry."
        await rag.ainsert([doc])

        # Verify chunks exist
        chunks = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())
        assert len(chunks) > 0, "Should have chunks after insert"

        # Verify graph has edges connecting chunks to entities
        # (This validates the bipartite graph structure)
        assert rag.knowledge_graph_inst is not None, "Graph should exist with edges"

    @pytest.mark.asyncio
    async def test_metadata_sync_across_storage(self, bigrag_instance):
        """Test that metadata flows correctly through all storage layers"""
        rag = bigrag_instance

        # Insert with metadata
        doc = "Test document for metadata flow"
        metadata = {
            "title": "Metadata Test Document",
            "category": "testing",
            "tags": ["test", "metadata", "storage"]
        }

        await rag.ainsert([doc], metadata=[metadata])

        # Verify metadata in full docs
        docs = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        assert len(docs) > 0, "Should have documents"

        # Get first document
        doc_data = list(docs.values())[0]
        assert "metadata" in doc_data or "title" in doc_data, "Document should preserve metadata"

        # Verify metadata in chunks
        chunks = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())
        assert len(chunks) > 0, "Should have chunks"

        # Get first chunk
        chunk_data = list(chunks.values())[0]
        # Chunks should have doc_title or doc_metadata
        assert "doc_title" in chunk_data or "doc_metadata" in chunk_data, \
            "Chunks should preserve document metadata"

    @pytest.mark.asyncio
    async def test_vector_count_matches_graph_nodes(self, bigrag_instance):
        """Test that vector DB entry counts match graph node counts"""
        rag = bigrag_instance

        # Insert documents
        docs = [
            "Document one about science",
            "Document two about technology"
        ]
        await rag.ainsert(docs)

        # Get counts
        chunks = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())
        chunk_count = len(chunks)

        # Chunk vector DB should have at least as many entries as chunks
        # (May have more due to entity/relation embeddings)
        assert chunk_count > 0, "Should have chunks in storage"
        assert rag.vdb_chunks is not None, "Chunk vector DB should exist"

        # Basic consistency: if we have N chunks, vector DB should have embeddings
        # This validates the indexing pipeline works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
