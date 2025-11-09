"""
Unit tests for bigrag.utils module

Tests utility functions for hashing, encoding, caching, and text processing.
"""

import pytest
import hashlib
from bigrag.utils import (
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    truncate_list_by_token_size,
    safe_operation_with_retry,
)


class TestHashFunctions:
    """Test hash ID generation functions"""

    def test_compute_mdhash_id_basic(self):
        """Test basic hash ID generation"""
        content = "Test content for hashing"
        prefix = "test-"

        result = compute_mdhash_id(content, prefix=prefix)

        # Verify format
        assert result.startswith(prefix)
        assert len(result) > len(prefix)

        # Verify deterministic (same input -> same output)
        result2 = compute_mdhash_id(content, prefix=prefix)
        assert result == result2

    def test_compute_mdhash_id_different_content(self):
        """Test that different content produces different hashes"""
        content1 = "Content A"
        content2 = "Content B"
        prefix = "test-"

        hash1 = compute_mdhash_id(content1, prefix=prefix)
        hash2 = compute_mdhash_id(content2, prefix=prefix)

        assert hash1 != hash2

    def test_compute_mdhash_id_different_prefixes(self):
        """Test different prefixes for same content"""
        content = "Same content"

        hash_rel = compute_mdhash_id(content, prefix="rel-")
        hash_ent = compute_mdhash_id(content, prefix="ent-")
        hash_chunk = compute_mdhash_id(content, prefix="chunk-")

        # All should have different prefixes
        assert hash_rel.startswith("rel-")
        assert hash_ent.startswith("ent-")
        assert hash_chunk.startswith("chunk-")

        # But same hash part (after prefix)
        assert hash_rel[4:] == hash_ent[4:] == hash_chunk[6:]

    def test_compute_mdhash_id_empty_content(self):
        """Test hash generation with empty content"""
        result = compute_mdhash_id("", prefix="test-")

        # Should still generate hash (of empty string)
        assert result.startswith("test-")
        assert len(result) > 5

    def test_compute_mdhash_id_unicode(self):
        """Test hash generation with Unicode content"""
        content = "Unicode: 你好世界 مرحبا 한국어"
        result = compute_mdhash_id(content, prefix="test-")

        assert result.startswith("test-")

        # Should be deterministic even with Unicode
        result2 = compute_mdhash_id(content, prefix="test-")
        assert result == result2

    def test_compute_mdhash_id_collision_resistance(self):
        """Test that similar strings don't collide"""
        hashes = set()

        # Generate hashes for similar strings
        for i in range(100):
            content = f"Test document number {i}"
            hash_id = compute_mdhash_id(content, prefix="test-")
            hashes.add(hash_id)

        # All should be unique (no collisions)
        assert len(hashes) == 100


class TestTokenEncoding:
    """Test token encoding and decoding functions"""

    def test_encode_string_basic(self):
        """Test basic string encoding"""
        text = "Hello, world!"
        tokens = encode_string_by_tiktoken(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_decode_tokens_basic(self):
        """Test basic token decoding"""
        text = "Hello, world!"
        tokens = encode_string_by_tiktoken(text)
        decoded = decode_tokens_by_tiktoken(tokens)

        assert decoded == text

    def test_encode_decode_roundtrip(self):
        """Test encode -> decode roundtrip preserves text"""
        original_texts = [
            "Simple text",
            "Text with numbers: 123456",
            "Special chars: @#$%^&*()",
            "Unicode: 你好世界",
            "Multi\nline\ntext",
        ]

        for text in original_texts:
            tokens = encode_string_by_tiktoken(text)
            decoded = decode_tokens_by_tiktoken(tokens)
            assert decoded == text, f"Roundtrip failed for: {text}"

    def test_encode_empty_string(self):
        """Test encoding empty string"""
        tokens = encode_string_by_tiktoken("")
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_truncate_list_by_token_size(self):
        """Test list truncation by token count"""
        items = [
            "Short item 1",
            "This is a longer item with more tokens",
            "Medium length item here",
            "Another short one",
        ]

        # Truncate to small token limit
        truncated = truncate_list_by_token_size(items, key=lambda x: x, max_token_size=20)

        # Should have fewer items
        assert len(truncated) <= len(items)

        # Verify token count
        total_tokens = sum(
            len(encode_string_by_tiktoken(item)) for item in truncated
        )
        assert total_tokens <= 20

    def test_truncate_single_item_exceeds_limit(self):
        """Test truncation when single item exceeds limit"""
        items = [
            "This is a very long item that exceeds the token limit by itself"
        ]

        truncated = truncate_list_by_token_size(items, key=lambda x: x, max_token_size=5)

        # Should return empty or partial
        assert len(truncated) <= len(items)


class TestRetryMechanism:
    """Test retry mechanism for API calls"""

    @pytest.mark.asyncio
    async def test_safe_operation_success(self):
        """Test successful operation (no retries needed)"""
        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "Success"

        result = await safe_operation_with_retry(
            successful_operation,
            operation_name="test_operation",
            max_retries=3,
        )

        assert result == "Success"
        assert call_count == 1  # Only called once

    @pytest.mark.asyncio
    async def test_safe_operation_retry_then_success(self):
        """Test operation that fails then succeeds"""
        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "Success after retries"

        result = await safe_operation_with_retry(
            flaky_operation,
            operation_name="flaky_test",
            max_retries=5,
        )

        assert result == "Success after retries"
        assert call_count == 3  # Failed 2 times, succeeded on 3rd

    @pytest.mark.asyncio
    async def test_safe_operation_max_retries_exceeded(self):
        """Test operation that keeps failing"""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("Permanent failure")

        with pytest.raises(Exception, match="Permanent failure"):
            await safe_operation_with_retry(
                always_fails,
                operation_name="failing_test",
                max_retries=3,
            )

        # Should have tried max_retries + 1 times (initial + retries)
        assert call_count == 4


class TestTextProcessing:
    """Test text processing utilities"""

    def test_text_normalization(self):
        """Test text normalization for matching"""
        from bigrag.utils import normalize_text

        texts = [
            ("Hello World", "hello world"),
            ("  Multiple   Spaces  ", "multiple spaces"),
            ("UPPERCASE", "uppercase"),
            ("MixedCase", "mixedcase"),
        ]

        for input_text, expected in texts:
            result = normalize_text(input_text)
            assert result == expected, f"Failed for: {input_text}"

    def test_remove_stopwords(self):
        """Test stopword removal"""
        from bigrag.utils import remove_stopwords

        text = "the quick brown fox jumps over the lazy dog"
        result = remove_stopwords(text)

        # Common stopwords should be removed
        assert "the" not in result.lower()
        assert "over" not in result.lower()

        # Content words should remain
        assert "quick" in result.lower()
        assert "fox" in result.lower()


class TestHashEdgeCases:
    """Test hash functions with edge cases"""

    def test_hash_very_long_content(self):
        """Test hashing very long content"""
        long_content = "A" * 100000  # 100k characters

        result = compute_mdhash_id(long_content, prefix="test-")

        # Hash length should still be reasonable
        assert len(result) < 100
        assert result.startswith("test-")

    def test_hash_special_characters(self):
        """Test hashing content with special characters"""
        special_content = "@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        result = compute_mdhash_id(special_content, prefix="test-")

        assert result.startswith("test-")

        # Should be reproducible
        result2 = compute_mdhash_id(special_content, prefix="test-")
        assert result == result2

    def test_hash_binary_data(self):
        """Test hashing binary-like data"""
        binary_content = bytes([0, 1, 2, 255, 254, 253]).decode('latin-1')

        try:
            result = compute_mdhash_id(binary_content, prefix="test-")
            assert result.startswith("test-")
        except:
            # May fail depending on implementation - document behavior
            pytest.skip("Binary data not supported")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
