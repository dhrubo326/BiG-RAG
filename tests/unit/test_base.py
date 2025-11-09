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
    """Test TextChunkSchema TypedDict"""

    def test_text_chunk_schema_creation(self):
        """Test creating TextChunkSchema instance (dict)"""
        # TextChunkSchema is a TypedDict, so it's created as a dict
        chunk: TextChunkSchema = {
            "tokens": 10,
            "content": "Test chunk content",
            "full_doc_id": "doc-456",
            "chunk_order_index": 0,
            "doc_title": "Test Document",
            "doc_metadata": {},
        }

        # Access using dictionary keys
        assert chunk["tokens"] == 10
        assert chunk["content"] == "Test chunk content"
        assert chunk["full_doc_id"] == "doc-456"
        assert chunk["doc_title"] == "Test Document"

    def test_text_chunk_schema_with_metadata(self):
        """Test TextChunkSchema with metadata fields"""
        # Create with metadata
        chunk: TextChunkSchema = {
            "tokens": 10,
            "content": "Test content",
            "full_doc_id": "doc-456",
            "chunk_order_index": 0,
            "doc_title": "Test Doc",
            "doc_metadata": {
                "category": "test_category",
                "tags": ["tag1", "tag2"],
            },
        }

        assert chunk["doc_metadata"]["category"] == "test_category"
        assert chunk["doc_metadata"]["tags"] == ["tag1", "tag2"]

    def test_text_chunk_schema_optional_fields(self):
        """Test TextChunkSchema with optional fields omitted"""
        # Create minimal chunk (doc_title and doc_metadata are optional with total=False)
        chunk: TextChunkSchema = {
            "tokens": 10,
            "content": "Test content",
            "full_doc_id": "doc-456",
            "chunk_order_index": 0,
        }

        # Required fields should be present
        assert "content" in chunk
        assert "full_doc_id" in chunk
        assert "tokens" in chunk

        # Optional fields may be omitted
        assert "doc_title" not in chunk or isinstance(chunk.get("doc_title"), str)
        assert "doc_metadata" not in chunk or isinstance(chunk.get("doc_metadata"), dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
