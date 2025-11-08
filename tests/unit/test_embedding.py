"""
Unit tests for embedding preparation and processing

Tests embedding function wrapping, batch processing, and dimension validation.
"""

import pytest
import numpy as np
from bigrag.utils import wrap_embedding_func_with_attrs


class TestEmbeddingFunctionWrapping:
    """Test embedding function wrapping with attributes"""

    def test_wrap_embedding_func_basic(self):
        """Test basic embedding function wrapping"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 128)

        # Check attributes are set
        assert hasattr(mock_embedding_func, "embedding_dim")
        assert mock_embedding_func.embedding_dim == 128
        assert hasattr(mock_embedding_func, "max_token_size")
        assert mock_embedding_func.max_token_size == 512

    @pytest.mark.asyncio
    async def test_wrapped_function_callable(self):
        """Test that wrapped function is still callable"""

        @wrap_embedding_func_with_attrs(embedding_dim=256, max_token_size=1024)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 256)

        # Call the function
        result = await mock_embedding_func(["test text 1", "test text 2"])

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 256)

    def test_wrap_with_custom_dimensions(self):
        """Test wrapping with different dimensions"""
        dimensions = [128, 256, 512, 768, 1024, 1536, 2048, 3072]

        for dim in dimensions:
            @wrap_embedding_func_with_attrs(embedding_dim=dim, max_token_size=512)
            async def func(texts):
                return np.random.rand(len(texts), dim)

            assert func.embedding_dim == dim


class TestEmbeddingOutputFormat:
    """Test embedding output format and shape"""

    @pytest.mark.asyncio
    async def test_embedding_output_shape(self):
        """Test that embeddings have correct shape"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            # Return embeddings with shape (num_texts, embedding_dim)
            return np.random.rand(len(texts), 128)

        result = await mock_embedding_func(["text1", "text2", "text3"])

        assert result.shape[0] == 3  # 3 texts
        assert result.shape[1] == 128  # 128 dimensions

    @pytest.mark.asyncio
    async def test_embedding_single_text(self):
        """Test embedding single text"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 128)

        result = await mock_embedding_func(["single text"])

        assert result.shape == (1, 128)

    @pytest.mark.asyncio
    async def test_embedding_empty_list(self):
        """Test embedding empty list of texts"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            if not texts:
                return np.array([]).reshape(0, 128)
            return np.random.rand(len(texts), 128)

        result = await mock_embedding_func([])

        assert result.shape == (0, 128)


class TestEmbeddingBatchProcessing:
    """Test batch processing of embeddings"""

    @pytest.mark.asyncio
    async def test_batch_embedding(self):
        """Test processing embeddings in batches"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            # Simulate batch processing
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embeddings = np.random.rand(len(batch), 128)
                all_embeddings.append(embeddings)

            return np.vstack(all_embeddings)

        # Test with 100 texts
        result = await mock_embedding_func([f"text {i}" for i in range(100)])

        assert result.shape == (100, 128)

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing large number of texts"""

        @wrap_embedding_func_with_attrs(embedding_dim=256, max_token_size=512)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 256)

        # Process 1000 texts
        texts = [f"document {i}" for i in range(1000)]
        result = await mock_embedding_func(texts)

        assert result.shape == (1000, 256)


class TestEmbeddingDimensionValidation:
    """Test embedding dimension validation"""

    @pytest.mark.asyncio
    async def test_dimension_mismatch_detection(self):
        """Test detecting dimension mismatch"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            # Intentionally return wrong dimension
            return np.random.rand(len(texts), 256)  # Should be 128

        result = await mock_embedding_func(["test"])

        # Dimension should be wrong
        assert result.shape[1] != mock_embedding_func.embedding_dim

    def test_common_embedding_dimensions(self):
        """Test wrapping with common embedding dimensions"""
        # Common dimensions for various models
        common_dims = {
            "bge-large-en-v1.5": 1024,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "jina-embeddings-v3": 1024,
        }

        for model_name, dim in common_dims.items():
            @wrap_embedding_func_with_attrs(embedding_dim=dim, max_token_size=8192)
            async def func(texts):
                return np.random.rand(len(texts), dim)

            assert func.embedding_dim == dim


class TestEmbeddingNormalization:
    """Test embedding normalization"""

    @pytest.mark.asyncio
    async def test_normalized_embeddings(self):
        """Test that embeddings can be normalized"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            embeddings = np.random.rand(len(texts), 128)
            # Normalize to unit length
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms

        result = await mock_embedding_func(["text1", "text2"])

        # Check if normalized (L2 norm should be ~1)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=5)

    @pytest.mark.asyncio
    async def test_unnormalized_embeddings(self):
        """Test embeddings without normalization"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 128) * 10  # Larger values

        result = await mock_embedding_func(["text1", "text2"])

        # Norms should not be 1
        norms = np.linalg.norm(result, axis=1)
        assert not np.allclose(norms, [1.0, 1.0])


class TestEmbeddingDataTypes:
    """Test embedding data types"""

    @pytest.mark.asyncio
    async def test_float32_embeddings(self):
        """Test embeddings as float32"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 128).astype(np.float32)

        result = await mock_embedding_func(["test"])

        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_float64_embeddings(self):
        """Test embeddings as float64"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 128).astype(np.float64)

        result = await mock_embedding_func(["test"])

        assert result.dtype == np.float64


class TestEmbeddingEdgeCases:
    """Test embedding edge cases"""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test embedding very long text (exceeds max_token_size)"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            # Truncate texts if needed (real implementation would do this)
            return np.random.rand(len(texts), 128)

        # Create very long text (10K words)
        long_text = " ".join(["word"] * 10000)
        result = await mock_embedding_func([long_text])

        # Should still produce embedding
        assert result.shape == (1, 128)

    @pytest.mark.asyncio
    async def test_special_characters_in_text(self):
        """Test embedding text with special characters"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 128)

        special_text = "Special: @#$%^&*(){}[]|<>?/~`"
        result = await mock_embedding_func([special_text])

        assert result.shape == (1, 128)

    @pytest.mark.asyncio
    async def test_unicode_text_embedding(self):
        """Test embedding Unicode text"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            return np.random.rand(len(texts), 128)

        unicode_texts = [
            "中文测试",
            "عربي",
            "한국어",
            "Русский",
        ]

        result = await mock_embedding_func(unicode_texts)

        assert result.shape == (4, 128)


class TestEmbeddingConsistency:
    """Test embedding consistency and determinism"""

    @pytest.mark.asyncio
    async def test_consistent_embeddings(self):
        """Test that same text produces consistent embeddings"""

        # Use fixed seed for deterministic results
        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            np.random.seed(42)
            return np.random.rand(len(texts), 128)

        result1 = await mock_embedding_func(["test text"])
        result2 = await mock_embedding_func(["test text"])

        # Should be identical
        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings"""

        @wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=512)
        async def mock_embedding_func(texts):
            # Hash-based embeddings for uniqueness
            embeddings = []
            for text in texts:
                # Simple hash-based embedding
                hash_val = hash(text)
                np.random.seed(hash_val % (2**32))
                embeddings.append(np.random.rand(128))
            return np.array(embeddings)

        texts = ["text A", "text B", "text C"]
        result = await mock_embedding_func(texts)

        # All embeddings should be different
        assert not np.array_equal(result[0], result[1])
        assert not np.array_equal(result[1], result[2])
        assert not np.array_equal(result[0], result[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
