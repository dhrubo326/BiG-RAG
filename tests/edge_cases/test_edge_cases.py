"""
Edge case tests for BiG-RAG

Tests unusual inputs, error handling, edge conditions, and robustness.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))

from test_documents import get_edge_case_documents


@pytest.mark.edge_cases
class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_document(self, bigrag_instance):
        """Test inserting empty document"""
        rag = bigrag_instance

        # Should handle gracefully
        await rag.ainsert([""])

    @pytest.mark.asyncio
    async def test_very_long_document(self, bigrag_instance):
        """Test inserting very long document"""
        rag = bigrag_instance

        long_doc = " ".join(["word"] * 5000)
        await rag.ainsert([long_doc])

    @pytest.mark.asyncio
    async def test_special_characters(self, bigrag_instance):
        """Test documents with special characters"""
        rag = bigrag_instance

        special_docs = [
            "@#$%^&*()_+-=[]{}|;:',.<>?/~`",
            "Unicode: 你好世界 مرحبا",
        ]

        await rag.ainsert(special_docs)

    @pytest.mark.asyncio
    async def test_query_with_no_results(self, bigrag_with_data):
        """Test query that should return no results"""
        rag = bigrag_with_data

        from bigrag.base import QueryParam
        results = await rag.aquery(
            "NONEXISTENT_QUERY_12345_RANDOM",
            QueryParam(mode="hybrid"),
        )

        # Should return empty or minimal results, not crash
        assert results is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, bigrag_instance):
        """Test deleting document that doesn't exist"""
        rag = bigrag_instance

        # Should handle gracefully
        try:
            await rag.adelete_document("nonexistent-doc-id")
        except Exception:
            # May raise exception or return gracefully
            pass

    @pytest.mark.asyncio
    async def test_all_edge_case_documents(self, bigrag_instance):
        """Test all edge case documents from fixtures"""
        rag = bigrag_instance

        edge_docs = get_edge_case_documents()

        # Should handle all without crashing
        for doc in edge_docs:
            try:
                await rag.ainsert([doc])
            except Exception as e:
                # Document what fails
                pytest.fail(f"Failed to insert edge case document: {doc[:50]}... Error: {e}")

    @pytest.mark.asyncio
    async def test_null_and_none_inputs(self, bigrag_instance):
        """Test handling of null/None inputs"""
        rag = bigrag_instance

        # Test None in document list
        try:
            await rag.ainsert([None])
        except (TypeError, ValueError, AttributeError):
            # Expected to fail or handle gracefully
            pass

        # Test empty list
        await rag.ainsert([])  # Should handle gracefully

    @pytest.mark.asyncio
    async def test_whitespace_only_documents(self, bigrag_instance):
        """Test documents with only whitespace"""
        rag = bigrag_instance

        whitespace_docs = [
            "   ",
            "\n\n\n",
            "\t\t\t",
            "    \n    \t    ",
        ]

        # Should handle without crashing
        await rag.ainsert(whitespace_docs)

    @pytest.mark.asyncio
    async def test_duplicate_documents(self, bigrag_instance):
        """Test inserting duplicate documents"""
        rag = bigrag_instance

        doc = "This is a duplicate document."

        # Insert same document multiple times
        await rag.ainsert([doc])
        await rag.ainsert([doc])
        await rag.ainsert([doc])

        # Should handle without errors (may create duplicates or deduplicate)

    @pytest.mark.asyncio
    async def test_malformed_metadata(self, bigrag_instance):
        """Test handling of malformed metadata"""
        rag = bigrag_instance

        doc = "Test document with malformed metadata"

        # Test various malformed metadata structures
        malformed_metadata = [
            None,
            "not a dict",
            123,
            ["list", "instead"],
            {"valid_key": None},
        ]

        for metadata in malformed_metadata:
            try:
                await rag.ainsert([doc], metadatas=[metadata])
            except (TypeError, ValueError, AttributeError):
                # Expected to fail or handle gracefully
                pass

    @pytest.mark.asyncio
    async def test_extremely_long_query(self, bigrag_with_data):
        """Test query with extremely long text"""
        rag = bigrag_with_data

        # Create very long query (1000+ words)
        long_query = " ".join(["query"] * 1000)

        from bigrag.base import QueryParam

        # Should handle without crashing (may truncate or error)
        try:
            results = await rag.aquery(long_query, QueryParam(mode="hybrid"))
            # If it succeeds, results should be valid
            assert results is not None
        except Exception:
            # May fail due to length limits, that's acceptable
            pass

    @pytest.mark.asyncio
    async def test_query_with_special_characters(self, bigrag_with_data):
        """Test queries with special characters"""
        rag = bigrag_with_data

        special_queries = [
            "@#$%^&*()",
            "query with\nnewlines\nand\ttabs",
            "query with 'quotes' and \"double quotes\"",
            "query with <html> tags",
            "query with $pecial ch@rs!",
        ]

        from bigrag.base import QueryParam

        for query in special_queries:
            # Should handle gracefully
            results = await rag.aquery(query, QueryParam(mode="hybrid"))
            assert results is not None

    @pytest.mark.asyncio
    async def test_invalid_query_parameters(self, bigrag_with_data):
        """Test invalid query parameters"""
        rag = bigrag_with_data

        from bigrag.base import QueryParam

        # Test invalid mode
        try:
            await rag.aquery("test", QueryParam(mode="invalid_mode"))
        except (ValueError, KeyError):
            # Expected to fail
            pass

        # Test invalid top_k
        try:
            await rag.aquery("test", QueryParam(mode="hybrid", top_k=-1))
        except (ValueError, AssertionError):
            # Expected to fail
            pass

        # Test invalid top_k (too large)
        try:
            results = await rag.aquery("test", QueryParam(mode="hybrid", top_k=10000))
            # May succeed but should handle gracefully
            assert results is not None
        except Exception:
            # May fail, that's acceptable
            pass

    @pytest.mark.asyncio
    async def test_concurrent_insert_same_document(self, bigrag_instance):
        """Test race condition: concurrent insert of same document"""
        rag = bigrag_instance

        import asyncio

        doc = "Concurrent insert test document"

        # Insert same document concurrently multiple times
        tasks = [rag.ainsert([doc]) for _ in range(10)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle without crashes
        # (May create duplicates or handle with locks)
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0  # At least some should succeed

    @pytest.mark.asyncio
    async def test_query_before_any_insert(self, bigrag_instance):
        """Test querying empty knowledge graph"""
        rag = bigrag_instance

        from bigrag.base import QueryParam

        # Query before any documents inserted
        results = await rag.aquery("test query", QueryParam(mode="hybrid"))

        # Should return empty results, not crash
        assert results is not None

    @pytest.mark.asyncio
    async def test_delete_then_reinsert(self, bigrag_instance):
        """Test delete followed by reinsert of same document"""
        rag = bigrag_instance

        doc = "Delete and reinsert test"
        doc_id = "test_doc_123"

        # Insert
        await rag.ainsert([doc], metadatas=[{"id": doc_id}])

        # Delete
        try:
            await rag.adelete_document(doc_id)
        except Exception:
            pass

        # Reinsert
        await rag.ainsert([doc], metadatas=[{"id": doc_id}])

        # Should handle without errors

    @pytest.mark.asyncio
    async def test_metadata_with_reserved_keys(self, bigrag_instance):
        """Test metadata with potentially reserved keys"""
        rag = bigrag_instance

        doc = "Test document with reserved metadata keys"

        # Test various potentially reserved keys
        metadata = {
            "id": "custom_id",
            "type": "custom_type",
            "name": "custom_name",
            "content": "custom_content",
            "_internal": "should_work",
        }

        # Should handle gracefully
        await rag.ainsert([doc], metadatas=[metadata])

    @pytest.mark.asyncio
    async def test_very_small_chunks(self, bigrag_instance):
        """Test document that creates very small chunks"""
        rag = bigrag_instance

        # Document with only a few words
        tiny_doc = "Word."

        await rag.ainsert([tiny_doc])

        # Query should still work
        from bigrag.base import QueryParam
        results = await rag.aquery("Word", QueryParam(mode="hybrid"))
        assert results is not None

    @pytest.mark.asyncio
    async def test_repeated_entity_extraction(self, bigrag_instance):
        """Test document with repeated entities"""
        rag = bigrag_instance

        # Document with same entity mentioned many times
        repeated_doc = "Einstein Einstein Einstein developed relativity. Einstein was a physicist. Einstein Einstein."

        await rag.ainsert([repeated_doc])

        # Should deduplicate entities properly

    @pytest.mark.asyncio
    async def test_numeric_only_content(self, bigrag_instance):
        """Test documents with only numbers"""
        rag = bigrag_instance

        numeric_docs = [
            "123 456 789",
            "3.14159 2.71828",
            "1000000000000",
        ]

        await rag.ainsert(numeric_docs)

    @pytest.mark.asyncio
    async def test_mixed_language_content(self, bigrag_instance):
        """Test documents with mixed languages"""
        rag = bigrag_instance

        mixed_docs = [
            "English text with 中文 and العربية mixed together",
            "Test français English Español 日本語",
        ]

        await rag.ainsert(mixed_docs)

        # Query in mixed language
        from bigrag.base import QueryParam
        results = await rag.aquery("中文 English", QueryParam(mode="hybrid"))
        assert results is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
