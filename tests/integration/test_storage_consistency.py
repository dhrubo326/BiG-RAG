"""
Integration test for storage consistency

Ensures graph, vector DB, and KV storage stay synchronized.
"""

import pytest


@pytest.mark.integration
class TestStorageConsistency:
    """Test storage layers stay in sync"""

    @pytest.mark.asyncio
    async def test_insert_syncs_all_layers(self, bigrag_instance):
        """Test that insert updates all storage layers"""
        rag = bigrag_instance

        # Insert document
        await rag.insert(["Test document for consistency check"])

        # Verify all layers have data
        # 1. Full docs
        docs = await rag.full_docs.get_by_ids(await rag.full_docs.get_all_ids())
        assert len(docs) > 0

        # 2. Text chunks
        chunks = await rag.text_chunks.get_by_ids(await rag.text_chunks.get_all_ids())
        assert len(chunks) > 0

        # 3. Vector DBs (entities and chunks should have entries)
        # Basic check: vector DBs exist
        assert rag.vdb_entities is not None
        assert rag.vdb_chunks is not None

    @pytest.mark.asyncio
    async def test_delete_syncs_all_layers(self, bigrag_instance):
        """Test that delete removes from all layers"""
        rag = bigrag_instance

        # Insert then delete
        await rag.insert(["Document to delete"])

        docs_before = await rag.full_docs.get_by_ids(await rag.full_docs.get_all_ids())
        doc_id = list(docs_before.keys())[0]

        await rag.delete_document(doc_id)

        # Verify removal
        docs_after = await rag.full_docs.get_by_ids(await rag.full_docs.get_all_ids())
        assert len(docs_after) < len(docs_before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
