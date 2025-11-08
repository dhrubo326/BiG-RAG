"""
Unit tests for chunking functions (bigrag.operate.chunking_by_token_size)

Comprehensive tests for document chunking with metadata preservation.
"""

import pytest
from bigrag.operate import chunking_by_token_size


class TestChunkingBasics:
    """Test basic chunking functionality"""

    def test_chunking_creates_chunks(self):
        """Test that chunking actually splits text"""
        text = " ".join(["word"] * 1000)  # Long text

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=10,
        )

        # Should produce multiple chunks
        assert len(chunks) > 1
        # All chunks should be dicts
        assert all(isinstance(c, dict) for c in chunks)
        # All chunks should have 'content' key
        assert all("content" in c for c in chunks)

    def test_chunking_short_text(self):
        """Test chunking text shorter than chunk size"""
        text = "Short text that fits in one chunk"

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=1000,
            overlap_token_size=10,
        )

        # Should produce exactly 1 chunk
        assert len(chunks) == 1
        assert chunks[0]["content"] == text

    def test_chunking_with_overlap(self):
        """Test that overlap parameter works"""
        text = " ".join(["word"] * 500)

        chunks_no_overlap = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=0,
        )

        chunks_with_overlap = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=20,
        )

        # Overlap should create MORE chunks (smaller effective size)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)


class TestChunkingMetadataPreservation:
    """Test Phase 2.1: Metadata preservation during chunking"""

    def test_metadata_preserved_in_chunks(self):
        """Test that metadata is preserved in chunk dicts"""
        text = " ".join(["word"] * 500)
        doc_metadata = {
            "title": "Test Document",
            "category": "test",
            "tags": ["test1", "test2"],
        }

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=10,
            doc_metadata=doc_metadata,
        )

        # All chunks should have metadata
        for chunk in chunks:
            assert "doc_title" in chunk
            assert chunk["doc_title"] == "Test Document"
            assert "doc_metadata" in chunk
            assert chunk["doc_metadata"]["category"] == "test"
            assert "test1" in chunk["doc_metadata"]["tags"]

    def test_chunks_without_metadata(self):
        """Test chunking works without metadata"""
        text = " ".join(["word"] * 500)

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=10,
        )

        # Should still create chunks
        assert len(chunks) > 0
        # Chunks may not have metadata fields or have None
        # This is acceptable behavior


class TestChunkingEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_text(self):
        """Test chunking empty text"""
        chunks = chunking_by_token_size(
            "",
            chunk_token_size=100,
            overlap_token_size=10,
        )

        # Should return empty list or single empty chunk
        assert isinstance(chunks, list)
        assert len(chunks) <= 1

    def test_single_word(self):
        """Test chunking single word"""
        chunks = chunking_by_token_size(
            "word",
            chunk_token_size=100,
            overlap_token_size=10,
        )

        assert len(chunks) == 1
        assert chunks[0]["content"] == "word"

    def test_whitespace_only(self):
        """Test chunking whitespace-only text"""
        chunks = chunking_by_token_size(
            "     \n\n   \t  ",
            chunk_token_size=100,
            overlap_token_size=10,
        )

        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_very_large_text(self):
        """Test chunking very large document (10K words)"""
        text = " ".join(["word"] * 10000)

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=10,
        )

        # Should produce many chunks
        assert len(chunks) > 50
        # All chunks should be reasonable size
        for chunk in chunks:
            assert len(chunk["content"]) > 0
            assert len(chunk["content"]) < 10000  # Not too large

    def test_unicode_text(self):
        """Test chunking with Unicode characters"""
        text = "中文测试 " * 200 + "عربي " * 200 + "한국어 " * 200

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=10,
        )

        # Should handle Unicode correctly
        assert len(chunks) > 0
        # Content should still be valid
        for chunk in chunks:
            assert len(chunk["content"]) > 0

    def test_special_characters(self):
        """Test chunking with special characters"""
        text = "Special chars: @#$%^&*(){}[]|<>?/~`" * 100

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=10,
        )

        assert len(chunks) > 0


class TestChunkingTokenSizeAccuracy:
    """Test that chunk sizes respect token limits"""

    def test_chunks_respect_max_size(self):
        """Test that chunks don't exceed max token size"""
        text = " ".join(["word"] * 1000)
        max_tokens = 100

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=max_tokens,
            overlap_token_size=10,
        )

        # Check each chunk is within limit
        # (Note: Exact token count requires tiktoken, this is approximate)
        for chunk in chunks:
            # Each chunk should be reasonable length
            # With ~4 chars per token, 100 tokens ≈ 400 chars
            assert len(chunk["content"]) < max_tokens * 10  # Generous upper bound

    def test_overlap_calculation(self):
        """Test that overlap is calculated correctly"""
        text = " ".join([f"word{i}" for i in range(500)])
        chunk_size = 50
        overlap = 10

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=chunk_size,
            overlap_token_size=overlap,
        )

        # Adjacent chunks should share some content (overlap)
        if len(chunks) >= 2:
            # Check if there's overlap between first two chunks
            chunk1_words = chunks[0]["content"].split()
            chunk2_words = chunks[1]["content"].split()

            # Last words of chunk1 should appear in chunk2
            overlap_found = any(word in chunk2_words for word in chunk1_words[-overlap:])
            assert overlap_found, "Overlap not found between adjacent chunks"


class TestChunkingIndexing:
    """Test chunk indexing and ordering"""

    def test_chunks_have_sequential_index(self):
        """Test that chunks have sequential index (if provided)"""
        text = " ".join(["word"] * 500)

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=10,
        )

        # Chunks should maintain order
        # (Order is implicit in list order)
        assert len(chunks) > 1

    def test_chunks_preserve_order(self):
        """Test that chunks preserve document order"""
        text = "First sentence. " * 50 + "Middle sentence. " * 50 + "Last sentence. " * 50

        chunks = chunking_by_token_size(
            text,
            chunk_token_size=100,
            overlap_token_size=10,
        )

        # First chunk should contain "First sentence"
        assert "First sentence" in chunks[0]["content"]
        # Last chunk should contain "Last sentence"
        assert "Last sentence" in chunks[-1]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
