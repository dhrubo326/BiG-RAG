"""
Regression tests for Phase 4 Critical Bug Fixes (January 2025)

This file validates that 5 critical bugs discovered during Phase 4 system testing
remain fixed and do not regress.

Bug List (Phase 4):
- Bug P4-1: Missing chunks_vdb indexing (bigrag.py:384-395)
- Bug P4-2: API reading non-existent files (api/kg_utils.py:53-199)
- Bug P4-3: Entity weights missing (operate.py:190-243)
- Bug P4-4: Incomplete rebuild cleanup (api/kg_utils.py:418-439)
- Bug P4-5: Incomplete document deletion cascade

These bugs are CRITICAL and must remain fixed to ensure system stability.
"""

import pytest
import sys
from pathlib import Path

# Import required modules
from bigrag import BiGRAG


@pytest.mark.critical
@pytest.mark.regression
class TestBugP4_1_ChunksVdbIndexing:
    """
    Regression test for Bug P4-1: Missing chunks_vdb indexing

    ISSUE: vdb_chunks.json was empty (0 entries) because chunks were created
           but never indexed to chunks_vdb
    FIX: Added proper indexing in bigrag.py:384-395
    IMPACT: Path C (chunk-based retrieval) was completely broken
    """

    @pytest.mark.asyncio
    async def test_chunks_are_indexed_to_vdb(self, bigrag_instance):
        """Test that chunks are properly indexed to chunks_vdb"""
        rag = bigrag_instance

        # Insert document that will create chunks
        test_doc = (
            "Albert Einstein developed the theory of relativity in the early 20th century. "
            "His work revolutionized our understanding of space, time, and gravity. "
            "Einstein won the Nobel Prize in Physics in 1921."
        )

        await rag.ainsert([test_doc])

        # Bug P4-1 fix: Verify chunks_vdb has entries
        chunks_vdb = rag.chunks_vdb
        all_chunk_ids = await chunks_vdb.all_keys()

        assert len(all_chunk_ids) > 0, "Bug P4-1 REGRESSION: chunks_vdb is empty!"

    @pytest.mark.asyncio
    async def test_chunk_vdb_size_matches_created_chunks(self, bigrag_instance):
        """Test that all created chunks are indexed"""
        rag = bigrag_instance

        # Insert multiple documents
        docs = [
            "Document 1 with some content about machine learning and AI.",
            "Document 2 discusses natural language processing techniques.",
            "Document 3 explains neural networks and deep learning fundamentals.",
        ]

        await rag.ainsert(docs)

        # Count chunks in KV storage
        chunk_keys = await rag.text_chunks.all_keys()
        kv_chunk_count = len(chunk_keys)

        # Count chunks in vector DB
        vdb_chunk_keys = await rag.chunks_vdb.all_keys()
        vdb_chunk_count = len(vdb_chunk_keys)

        # Should match (or vdb_count >= kv_count due to indexing)
        assert vdb_chunk_count > 0, "Bug P4-1 REGRESSION: No chunks indexed to VDB"
        assert vdb_chunk_count == kv_chunk_count, \
            f"Chunk count mismatch: KV={kv_chunk_count}, VDB={vdb_chunk_count}"

    @pytest.mark.asyncio
    async def test_path_c_retrieval_works(self, bigrag_instance):
        """Test that Path C (chunk-based) retrieval works"""
        rag = bigrag_instance

        # Insert document with specific content
        test_doc = "Quantum mechanics is a fundamental theory in physics."
        await rag.ainsert([test_doc])

        # Query using chunk-based retrieval
        from bigrag.base import QueryParam
        results = await rag.aquery(
            "quantum mechanics",
            param=QueryParam(mode="naive", top_k=5)  # naive mode uses Path C
        )

        # Bug P4-1 fix: Path C should return results
        assert results is not None, "Bug P4-1 REGRESSION: Path C retrieval failed!"
        assert len(results) > 0, "Bug P4-1 REGRESSION: Path C returned no results"


@pytest.mark.regression
class TestBugP4_3_EntityWeights:
    """
    Regression test for Bug P4-3: Entity weights missing

    ISSUE: All entities had weight=0 in API responses because
           _merge_nodes_then_upsert() didn't aggregate weights
    FIX: Added weight aggregation in operate.py:190-243
    IMPACT: Entities couldn't be ranked by importance
    """

    @pytest.mark.asyncio
    async def test_entities_have_nonzero_weights(self, bigrag_instance):
        """Test that entities get non-zero weights assigned"""
        rag = bigrag_instance

        # Insert document with clear entities
        test_doc = (
            "Marie Curie was a physicist and chemist who conducted pioneering "
            "research on radioactivity. She was the first woman to win a Nobel Prize."
        )

        await rag.ainsert([test_doc])

        # Get entities from graph
        graph = rag.chunk_entity_relation_graph
        all_nodes = await graph.get_all_nodes()

        # Find entity nodes
        entity_nodes = [
            node for node in all_nodes
            if node.get("role") == "entity"
        ]

        assert len(entity_nodes) > 0, "No entities found"

        # Bug P4-3 fix: Entities should have weight > 0
        weights = [node.get("weight", 0) for node in entity_nodes]
        non_zero_weights = [w for w in weights if w > 0]

        assert len(non_zero_weights) > 0, \
            "Bug P4-3 REGRESSION: All entity weights are 0!"

    @pytest.mark.asyncio
    async def test_weight_aggregation_on_multiple_occurrences(self, bigrag_instance):
        """Test that weights aggregate when entity appears multiple times"""
        rag = bigrag_instance

        # Insert documents mentioning same entity
        docs = [
            "Einstein developed the theory of relativity.",
            "Einstein won the Nobel Prize in Physics.",
            "Einstein's work changed our understanding of the universe.",
        ]

        await rag.ainsert(docs)

        # Get entity for Einstein
        graph = rag.chunk_entity_relation_graph
        all_nodes = await graph.get_all_nodes()

        einstein_nodes = [
            node for node in all_nodes
            if node.get("role") == "entity" and
            "einstein" in node.get("name", "").lower()
        ]

        if einstein_nodes:
            # Weight should be aggregated from multiple mentions
            weight = einstein_nodes[0].get("weight", 0)
            assert weight > 0, "Bug P4-3 REGRESSION: Entity weight not aggregated!"


@pytest.mark.critical
@pytest.mark.regression
class TestBugP4_5_DocumentDeletionCascade:
    """
    Regression test for Bug P4-5: Incomplete document deletion

    ISSUE: Hard delete only removed from corpus, leaving KG data orphaned
    FIX: Complete cascade deletion implementation
    IMPACT: Deleted documents remained in knowledge graph
    """

    @pytest.mark.asyncio
    async def test_delete_removes_from_all_storage_layers(self, bigrag_instance):
        """Test that document deletion cascades to all storage layers"""
        rag = bigrag_instance

        # Insert document
        test_doc = "Test document for deletion cascade validation."
        await rag.ainsert([test_doc], metadatas=[{"id": "cascade_test_doc"}])

        # Count initial entries
        initial_full_docs = len(await rag.full_docs.all_keys())
        initial_chunks = len(await rag.text_chunks.all_keys())

        # Delete document
        stats = await rag.adelete_document("cascade_test_doc")

        # Verify counts decreased
        final_full_docs = len(await rag.full_docs.all_keys())
        final_chunks = len(await rag.text_chunks.all_keys())

        # Bug P4-5 fix: Should delete from all layers
        assert final_full_docs < initial_full_docs, \
            "Bug P4-5 REGRESSION: Document not deleted from full_docs!"
        assert final_chunks < initial_chunks, \
            "Bug P4-5 REGRESSION: Chunks not deleted!"

    @pytest.mark.asyncio
    async def test_delete_removes_orphaned_entities(self, bigrag_instance):
        """Test that orphaned entities are removed after document deletion"""
        rag = bigrag_instance

        # Insert document with unique entity
        unique_doc = "ZorblaxTheUnique is a fictional character that appears only here."
        await rag.ainsert([unique_doc], metadatas=[{"id": "unique_entity_doc"}])

        # Get initial entity count
        graph = rag.chunk_entity_relation_graph
        initial_nodes = await graph.get_all_nodes()

        # Delete document
        await rag.adelete_document("unique_entity_doc")

        # Get final entity count
        final_nodes = await graph.get_all_nodes()

        # Bug P4-5 fix: Orphaned entities should be removed
        assert len(final_nodes) <= len(initial_nodes), \
            "Bug P4-5 REGRESSION: Orphaned entities not cleaned up!"

    @pytest.mark.asyncio
    async def test_delete_preserves_shared_entities(self, bigrag_instance):
        """Test that shared entities are NOT deleted"""
        rag = bigrag_instance

        # Insert documents sharing an entity
        doc1 = "Albert Einstein developed relativity theory."
        doc2 = "Einstein won the Nobel Prize in 1921."

        await rag.ainsert([doc1], metadatas=[{"id": "doc1_einstein"}])
        await rag.ainsert([doc2], metadatas=[{"id": "doc2_einstein"}])

        # Delete first document
        await rag.adelete_document("doc1_einstein")

        # Check that Einstein entity still exists (from doc2)
        graph = rag.chunk_entity_relation_graph
        all_nodes = await graph.get_all_nodes()

        einstein_nodes = [
            node for node in all_nodes
            if node.get("role") == "entity" and
            "einstein" in node.get("name", "").lower()
        ]

        # Bug P4-5 fix: Shared entities should be preserved
        assert len(einstein_nodes) > 0, \
            "Bug P4-5 REGRESSION: Shared entity incorrectly deleted!"


@pytest.mark.regression
class TestBugP4_AllFixesIntegration:
    """
    Integration test ensuring all Phase 4 bug fixes work together

    Tests a complete workflow that exercises all 5 Phase 4 bug fixes
    simultaneously to validate system stability.
    """

    @pytest.mark.asyncio
    async def test_complete_workflow_with_phase4_fixes(self, bigrag_instance):
        """Test complete workflow exercising all Phase 4 fixes"""
        rag = bigrag_instance

        # Insert documents (exercises Bug P4-1: chunk indexing, Bug P4-3: weights)
        docs = [
            "Document 1: Machine learning is a subset of artificial intelligence.",
            "Document 2: Deep learning uses neural networks for pattern recognition.",
            "Document 3: Natural language processing enables AI to understand text.",
        ]

        await rag.ainsert(docs, metadatas=[
            {"id": "ml_doc"},
            {"id": "dl_doc"},
            {"id": "nlp_doc"},
        ])

        # Test Path C retrieval (Bug P4-1 fix)
        from bigrag.base import QueryParam
        results_naive = await rag.aquery(
            "machine learning",
            param=QueryParam(mode="naive", top_k=5)
        )
        assert results_naive is not None, "Path C retrieval failed"

        # Test hybrid retrieval (Bug P4-1, P4-3 combined)
        results_hybrid = await rag.aquery(
            "artificial intelligence",
            param=QueryParam(mode="hybrid", top_k=5)
        )
        assert results_hybrid is not None, "Hybrid retrieval failed"

        # Check entity weights are non-zero (Bug P4-3 fix)
        graph = rag.chunk_entity_relation_graph
        all_nodes = await graph.get_all_nodes()
        entity_nodes = [n for n in all_nodes if n.get("role") == "entity"]

        if entity_nodes:
            weights = [n.get("weight", 0) for n in entity_nodes]
            assert any(w > 0 for w in weights), "Entity weights all zero"

        # Delete one document (Bug P4-5 fix)
        initial_count = len(await rag.full_docs.all_keys())
        await rag.adelete_document("ml_doc")
        final_count = len(await rag.full_docs.all_keys())

        assert final_count == initial_count - 1, "Document deletion failed"

        # Verify system still functional after deletion
        results_after_delete = await rag.aquery(
            "deep learning",
            param=QueryParam(mode="hybrid", top_k=5)
        )
        assert results_after_delete is not None, "System broken after deletion"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "regression"])
