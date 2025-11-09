"""
End-to-End test for complete BiG-RAG pipeline

Tests the full workflow: Insert -> Query -> Delete using real demo_test dataset.
This test must pass for system to be considered functional.
"""

import pytest
from pathlib import Path
import sys

# Add test fixtures to path
sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))

from test_documents import (
    DEMO_TEST_AVAILABLE,
    DEMO_TEST_EXPR,
    get_complex_multi_hop_documents,
)
from bigrag import BiGRAG
from bigrag.base import QueryParam


@pytest.mark.critical
@pytest.mark.e2e
@pytest.mark.slow
class TestCompleteWorkflow:
    """Test complete BiG-RAG workflow end-to-end"""

    @pytest.mark.asyncio
    async def test_complete_pipeline(self, bigrag_instance):
        """
        Test complete pipeline: insert -> query -> delete

        This is the most critical test for BiG-RAG.
        """
        rag = bigrag_instance

        # Step 1: INSERT documents with metadata
        docs = [
            "Lionel Messi plays for Inter Miami in Major League Soccer.",
            "Messi won the 2022 FIFA World Cup with Argentina.",
            "Inter Miami is based in Miami, Florida.",
        ]

        metadata = [
            {"title": "Messi Career", "category": "sports"},
            {"title": "World Cup 2022", "category": "sports"},
            {"title": "Inter Miami Info", "category": "sports"},
        ]

        await rag.ainsert(docs, metadata=metadata)

        # Step 2: QUERY the knowledge graph
        results = await rag.aquery(
            "Which team does Messi play for?",
            param=QueryParam(mode="hybrid", top_k=5),
        )

        # Validate query results
        assert results is not None
        assert len(results) > 0
        assert isinstance(results, str)  # Returns formatted string
        # Should contain "Inter Miami"
        assert "miami" in results.lower() or "inter" in results.lower()

        # Step 3: VERIFY data persistence
        all_docs = await rag.full_docs.get_by_ids(
            await rag.full_docs.all_keys()
        )
        assert len(all_docs) >= 3

        # Step 4: DELETE a document
        doc_id = list(all_docs.keys())[0]
        deletion_stats = await rag.adelete_document(doc_id)

        # Validate deletion
        assert deletion_stats is not None
        assert "chunks_deleted" in deletion_stats
        assert deletion_stats["chunks_deleted"] >= 0

        # Step 5: VERIFY deletion
        all_docs_after = await rag.full_docs.get_by_ids(
            await rag.full_docs.all_keys()
        )
        assert len(all_docs_after) == len(all_docs) - 1

        # Step 6: QUERY after deletion (should still work)
        results_after = await rag.aquery(
            "What is Inter Miami?",
            param=QueryParam(mode="hybrid"),
        )
        assert results_after is not None


@pytest.mark.e2e
@pytest.mark.slow
class TestWithDemoTestDataset:
    """
    Test using pre-built demo_test dataset (complex football data)

    This uses real, complex data to validate system behavior.
    """

    @pytest.mark.skipif(
        not DEMO_TEST_AVAILABLE,
        reason="demo_test KG not available. Build it first with script_build.py"
    )
    @pytest.mark.asyncio
    async def test_query_demo_test_dataset(self):
        """Test querying demo_test dataset (if available)"""
        # Load existing demo_test KG
        rag = BiGRAG(working_dir=str(DEMO_TEST_EXPR))

        # Query complex multi-hop question
        results = await rag.aquery(
            "Who won the 2022 FIFA World Cup?",
            param=QueryParam(mode="hybrid", top_k=10),
        )

        # Validate results
        assert results is not None
        assert len(results) > 0

        # Should mention Argentina or Messi
        assert "argentina" in results.lower() or "messi" in results.lower()

    @pytest.mark.skipif(
        not DEMO_TEST_AVAILABLE,
        reason="demo_test KG not available"
    )
    @pytest.mark.asyncio
    async def test_complex_multi_hop_query_demo_test(self):
        """Test complex multi-hop reasoning with demo_test"""
        rag = BiGRAG(working_dir=str(DEMO_TEST_EXPR))

        # Multi-hop query requiring chaining facts
        results = await rag.aquery(
            "Which team does the 2022 World Cup winner captain play for?",
            param=QueryParam(mode="hybrid", top_k=10),
        )

        assert results is not None
        # Should mention Messi (captain) and Inter Miami (team)
        results_lower = results.lower()
        # At least one of these should be present
        assert any(keyword in results_lower for keyword in ["messi", "miami", "inter"])


@pytest.mark.e2e
class TestMultipleInsertionsAndQueries:
    """Test multiple rounds of insertions and queries"""

    @pytest.mark.asyncio
    async def test_incremental_insertions(self, bigrag_instance):
        """Test adding documents incrementally"""
        rag = bigrag_instance

        # First batch
        await rag.ainsert(["Document 1 about topic A"])

        # Query
        results1 = await rag.aquery("topic A", QueryParam(mode="hybrid"))
        assert results1 is not None

        # Second batch
        await rag.ainsert(["Document 2 about topic B", "Document 3 about topic C"])

        # Query again
        results2 = await rag.aquery("topic B", QueryParam(mode="hybrid"))
        assert results2 is not None

        # Verify all documents exist
        all_docs = await rag.full_docs.get_by_ids(await rag.full_docs.all_keys())
        assert len(all_docs) >= 3

    @pytest.mark.asyncio
    async def test_varied_queries(self, bigrag_with_data):
        """Test multiple different queries on same dataset"""
        rag = bigrag_with_data

        queries = [
            "Who is Messi?",
            "What team?",
            "World Cup",
            "Miami Florida",
        ]

        for query in queries:
            results = await rag.aquery(query, QueryParam(mode="hybrid"))
            # All queries should return something
            assert results is not None
            assert len(results) > 0


@pytest.mark.e2e
class TestMetadataPreservation:
    """Test that metadata is preserved through the pipeline"""

    @pytest.mark.asyncio
    async def test_metadata_flows_through_pipeline(self, bigrag_instance):
        """Test metadata preservation from insert to query"""
        rag = bigrag_instance

        # Insert with rich metadata
        docs = ["Test document about AI and machine learning"]
        metadata = [
            {
                "title": "AI Research Paper",
                "category": "technology",
                "tags": ["ai", "ml", "research"],
            }
        ]

        await rag.ainsert(docs, metadata=metadata)

        # Retrieve chunks and verify metadata
        all_chunks = await rag.text_chunks.get_by_ids(
            await rag.text_chunks.all_keys()
        )

        # At least one chunk should have metadata
        assert len(all_chunks) > 0

        for chunk_data in all_chunks.values():
            # Verify metadata fields exist
            assert "doc_title" in chunk_data or chunk_data.get("doc_title") is not None
            # Metadata should be preserved
            if chunk_data.get("doc_title"):
                assert chunk_data["doc_title"] == "AI Research Paper"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
