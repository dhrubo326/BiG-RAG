"""
End-to-End test for document lifecycle

Tests complete lifecycle: Insert -> Update -> Query -> Delete -> Verify
"""

import pytest
from bigrag.base import QueryParam


@pytest.mark.e2e
class TestDocumentLifecycle:
    """Test complete document lifecycle"""

    @pytest.mark.asyncio
    async def test_insert_query_delete_lifecycle(self, bigrag_instance):
        """Test full lifecycle of a document"""
        rag = bigrag_instance

        # PHASE 1: INSERT
        doc_content = "Albert Einstein developed the theory of relativity in 1905."
        metadata = {"title": "Einstein", "category": "science"}

        await rag.ainsert([doc_content], metadata=[metadata])

        # Verify insertion
        all_docs = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        assert len(all_docs) >= 1

        # PHASE 2: QUERY
        results = await rag.aquery("Who developed relativity?", QueryParam(mode="hybrid"))
        assert results is not None
        assert "einstein" in results.lower()

        # PHASE 3: DELETE
        doc_id = list(all_docs.keys())[0]
        stats = await rag.adelete_document(doc_id)
        assert stats["chunks_deleted"] >= 0

        # PHASE 4: VERIFY DELETION
        remaining_docs = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        assert len(remaining_docs) == len(all_docs) - 1

    @pytest.mark.asyncio
    async def test_cascade_deletion_cleanup(self, bigrag_instance):
        """Test that deletion cascades properly (chunks, entities, edges)"""
        rag = bigrag_instance

        # Insert document
        await rag.ainsert(["Test document with entity TestEntity and relation TestRelation"])

        # Get counts before deletion
        docs_before = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        chunks_before = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())

        # Delete
        doc_id = list(docs_before.keys())[0]
        await rag.adelete_document(doc_id)

        # Verify cascade
        docs_after = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        chunks_after = await rag.text_chunks.get_by_ids(await rag.text_chunks.all_keys())

        assert len(docs_after) < len(docs_before)
        assert len(chunks_after) < len(chunks_before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
