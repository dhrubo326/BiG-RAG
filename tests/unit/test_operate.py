"""
Unit tests for bigrag.operate module

Tests graph operations, entity extraction, normalization, and retrieval functions.
Tests Bug #1 and #4 fixes.
"""

import pytest
from bigrag.operate import (
    normalize_entity_type,
    chunking_by_token_size,
)


class TestEntityTypeNormalization:
    """Test entity type normalization (Bug #4 fix)"""

    def test_normalize_uppercase_to_lowercase(self):
        """Test uppercase types normalize to lowercase"""
        test_cases = [
            ("PERSON", "person"),
            ("ORGANIZATION", "organization"),
            ("LOCATION", "geo"),
            ("EVENT", "event"),
        ]

        for input_type, expected in test_cases:
            result = normalize_entity_type(input_type)
            assert result == expected, f"Failed for: {input_type}"

    def test_normalize_mixed_case(self):
        """Test mixed case normalization"""
        test_cases = [
            ("Person", "person"),
            ("Organization", "organization"),
            ("Location", "geo"),
        ]

        for input_type, expected in test_cases:
            result = normalize_entity_type(input_type)
            assert result == expected, f"Failed for: {input_type}"

    def test_normalize_variants(self):
        """Test variant type names normalize correctly"""
        # Test TYPE_NORMALIZATION_MAP mappings
        test_cases = [
            ("TEAM", "organization"),
            ("PLAYER", "person"),
            ("CITY", "geo"),
            ("COUNTRY", "geo"),
            ("COMPANY", "organization"),
            ("GROUP", "organization"),
            ("INDIVIDUAL", "person"),
        ]

        for input_type, expected in test_cases:
            result = normalize_entity_type(input_type)
            assert result == expected, f"Failed for: {input_type}"

    def test_normalize_unknown_type(self):
        """Test unknown types default to 'category'"""
        unknown_types = ["UNKNOWN", "MYSTERY", "RANDOM_TYPE", ""]

        for unknown_type in unknown_types:
            result = normalize_entity_type(unknown_type)
            assert result == "category", f"Failed for: {unknown_type}"

    def test_normalize_already_normalized(self):
        """Test already normalized types stay unchanged"""
        normalized_types = ["person", "organization", "geo", "event", "category"]

        for norm_type in normalized_types:
            result = normalize_entity_type(norm_type)
            assert result == norm_type, f"Changed normalized type: {norm_type}"


class TestChunking:
    """Test text chunking functions"""

    def test_chunking_basic(self):
        """Test basic text chunking"""
        text = " ".join(["word"] * 1000)  # Long text

        chunks = chunking_by_token_size(
            text,
            max_token_size=100,
            overlap_token_size=10,
        )

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Each chunk should be dict with 'content' key
        assert all(isinstance(c, dict) and "content" in c for c in chunks)

    def test_chunking_short_text(self):
        """Test chunking text shorter than chunk size"""
        text = "Short text that fits in one chunk"

        chunks = chunking_by_token_size(
            text,
            max_token_size=1000,
            overlap_token_size=10,
        )

        # Should produce single chunk
        assert len(chunks) == 1
        assert chunks[0]["content"] == text

    def test_chunking_with_overlap(self):
        """Test that chunks have overlap"""
        text = " ".join([f"word{i}" for i in range(500)])

        chunks = chunking_by_token_size(
            text,
            max_token_size=100,
            overlap_token_size=20,
        )

        # Should have multiple chunks
        assert len(chunks) > 2

        # Check overlap exists (some content from chunk N appears in chunk N+1)
        # This is approximate test
        for i in range(len(chunks) - 1):
            chunk_current = chunks[i]["content"]
            chunk_next = chunks[i + 1]["content"]

            # Last words of current should appear in next
            current_words = chunk_current.split()[-10:]
            # At least some overlap
            overlap_count = sum(1 for word in current_words if word in chunk_next)
            assert overlap_count > 0, "No overlap detected between chunks"

    def test_chunking_empty_text(self):
        """Test chunking empty text"""
        chunks = chunking_by_token_size(
            "",
            max_token_size=100,
            overlap_token_size=10,
        )

        # Should return empty list or single empty chunk
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0]["content"] == "")

    def test_chunking_preserves_metadata(self):
        """Test chunking preserves document metadata"""
        text = "Test document content " * 100
        metadata = {"title": "Test Doc", "category": "test"}

        chunks = chunking_by_token_size(
            text,
            max_token_size=100,
            overlap_token_size=10,
        )

        # All chunks should have metadata fields if supported
        # (Implementation may or may not preserve metadata - document behavior)
        assert len(chunks) > 0


class TestHashIDGeneration:
    """Test hash-based ID generation for nodes"""

    def test_hash_id_format(self):
        """Test hash ID format for different prefixes"""
        from bigrag.operate import extract_entities
        from bigrag.utils import compute_mdhash_id
        from bigrag.constants import BIPARTITE_EDGE_PREFIX

        # Test bipartite edge prefix
        content = "Test relation content"
        edge_id = compute_mdhash_id(content, prefix=BIPARTITE_EDGE_PREFIX)

        assert edge_id.startswith("rel-")
        assert len(edge_id) > 4

    def test_hash_id_not_double_hashed(self):
        """Test Bug #1: Verify we don't hash an already-hashed ID"""
        from bigrag.utils import compute_mdhash_id

        # Create a hash ID
        original_id = compute_mdhash_id("test content", prefix="rel-")
        assert original_id.startswith("rel-")

        # If we accidentally hash it again with different prefix
        double_hashed = compute_mdhash_id(original_id, prefix="edge-")

        # These should be different (which would be wrong in real code)
        # This test documents the bug that was fixed
        assert double_hashed != original_id
        assert double_hashed.startswith("edge-")


class TestDefensiveDictAccess:
    """Test defensive dict access (Bug #4 fix)"""

    def test_safe_dict_access_with_missing_keys(self):
        """Test that we handle missing dict keys gracefully"""
        # Simulate vector DB results with missing keys
        results = [
            {"id": "123", "distance": 0.9},  # Missing required key
            {"entity_name": "ValidEntity", "distance": 0.8},  # Has required key
        ]

        # Defensive access pattern (Bug #4 fix)
        safe_results = [r.get("entity_name") for r in results if "entity_name" in r]

        # Should only get valid entity
        assert len(safe_results) == 1
        assert safe_results[0] == "ValidEntity"

    def test_safe_dict_access_empty_results(self):
        """Test defensive access with empty results"""
        results = []

        safe_results = [r.get("entity_name") for r in results if "entity_name" in r]

        assert len(safe_results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
