"""
Unit tests for bigrag.base module

Tests base classes, schemas, and parameter objects.
"""

import pytest
from bigrag.base import QueryParam, TextChunkSchema


class TestQueryParam:
    """Test QueryParam configuration object"""

    def test_query_param_defaults(self):
        """Test default QueryParam values"""
        param = QueryParam()

        # Check defaults
        assert param.mode == "hybrid"  # Default mode
        assert param.top_k > 0
        assert isinstance(param.enable_reranking, bool)

    def test_query_param_custom_values(self):
        """Test QueryParam with custom values"""
        param = QueryParam(
            mode="local",
            top_k=20,
            enable_reranking=False,
        )

        assert param.mode == "local"
        assert param.top_k == 20
        assert param.enable_reranking is False

    def test_query_param_valid_modes(self):
        """Test all valid query modes"""
        valid_modes = ["local", "global", "hybrid", "naive"]

        for mode in valid_modes:
            param = QueryParam(mode=mode)
            assert param.mode == mode

    def test_query_param_invalid_mode(self):
        """Test invalid query mode raises error"""
        with pytest.raises(ValueError):
            QueryParam(mode="invalid_mode")


class TestTextChunkSchema:
    """Test TextChunkSchema data class"""

    def test_text_chunk_schema_creation(self):
        """Test creating TextChunkSchema instance"""
        chunk = TextChunkSchema(
            chunk_id="chunk-123",
            content="Test chunk content",
            full_doc_id="doc-456",
            title="Test Document",
        )

        assert chunk.chunk_id == "chunk-123"
        assert chunk.content == "Test chunk content"
        assert chunk.full_doc_id == "doc-456"
        assert chunk.title == "Test Document"

    def test_text_chunk_schema_with_metadata(self):
        """Test TextChunkSchema with metadata fields"""
        chunk = TextChunkSchema(
            chunk_id="chunk-123",
            content="Test content",
            full_doc_id="doc-456",
            title="Test Doc",
            category="test_category",
            tags=["tag1", "tag2"],
        )

        assert chunk.category == "test_category"
        assert chunk.tags == ["tag1", "tag2"]

    def test_text_chunk_schema_optional_fields(self):
        """Test TextChunkSchema with optional fields as None"""
        chunk = TextChunkSchema(
            chunk_id="chunk-123",
            content="Test content",
            full_doc_id="doc-456",
        )

        # Optional fields should be None or have defaults
        assert chunk.title is None or isinstance(chunk.title, str)
        assert chunk.category is None or isinstance(chunk.category, str)
        assert chunk.tags is None or isinstance(chunk.tags, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
