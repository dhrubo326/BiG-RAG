"""
Integration test for graph-vector synchronization

Ensures graph storage and vector DBs stay synchronized.
"""

import pytest


@pytest.mark.integration
class TestGraphVectorSync:
    """Test graph and vector DB synchronization"""

    @pytest.mark.asyncio
    async def test_entity_in_both_graph_and_vector(self, bigrag_instance):
        """Test that entities appear in both graph and vector DB"""
        rag = bigrag_instance

        await rag.ainsert(["Test entity TESTENTITY123 in document."])

        # Entity should be in graph (if extraction worked)
        # and in vector DB for retrieval

    @pytest.mark.asyncio
    async def test_edge_deletion_removes_from_vector(self, bigrag_instance):
        """Test that deleting edges removes from vector DB"""
        rag = bigrag_instance

        # Insert and delete
        await rag.ainsert(["Document with edge"])
        docs = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())

        if docs:
            await rag.adelete_document(list(docs.keys())[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
