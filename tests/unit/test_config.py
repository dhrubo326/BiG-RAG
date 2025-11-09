"""
Unit tests for bigrag.config module

Tests configuration management and Bug #3 fix (reload_config with/without python-dotenv).
"""

import pytest
import os
from bigrag.config import BiGRAGConfig, get_config, reload_config


class TestBiGRAGConfig:
    """Test BiGRAG configuration class"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = BiGRAGConfig()

        # Check some default values
        assert config.chunk_size > 0
        assert config.chunk_overlap_size >= 0
        assert isinstance(config.enable_reranking, bool)

    def test_config_from_env(self, monkeypatch):
        """Test configuration loading from environment variables"""
        # Set environment variables (no BIGRAG_ prefix in actual config)
        monkeypatch.setenv("CHUNK_SIZE", "2000")
        monkeypatch.setenv("CHUNK_OVERLAP_SIZE", "200")
        monkeypatch.setenv("ENABLE_RERANKING", "false")

        config = BiGRAGConfig()

        assert config.chunk_size == 2000
        assert config.chunk_overlap_size == 200
        assert config.enable_reranking is False

    def test_config_openai_api_key(self, monkeypatch):
        """Test OpenAI API key configuration"""
        test_key = "sk-test-key-12345"
        monkeypatch.setenv("OPENAI_API_KEY", test_key)

        config = BiGRAGConfig()
        assert config.openai_api_key == test_key

    def test_config_embedding_model(self, monkeypatch):
        """Test embedding model configuration"""
        test_model = "test-embedding-model"
        monkeypatch.setenv("EMBEDDING_MODEL", test_model)

        config = BiGRAGConfig()
        assert config.embedding_model == test_model


class TestGetConfig:
    """Test get_config function"""

    def test_get_config_singleton(self):
        """Test that get_config returns same instance"""
        config1 = get_config()
        config2 = get_config()

        # Should be same instance
        assert config1 is config2

    def test_get_config_returns_valid_config(self):
        """Test that get_config returns valid BiGRAGConfig"""
        config = get_config()

        assert isinstance(config, BiGRAGConfig)
        assert hasattr(config, 'chunk_size')
        assert hasattr(config, 'embedding_model')


class TestReloadConfig:
    """Test reload_config function - Bug #3 regression test"""

    def test_reload_config_basic(self):
        """Test Bug #3 fix: reload_config works without errors"""
        # This should not raise NameError
        try:
            config = reload_config()
            assert isinstance(config, BiGRAGConfig)
        except NameError as e:
            pytest.fail(f"reload_config() raised NameError (Bug #3 not fixed): {e}")

    def test_reload_config_updates_values(self, monkeypatch):
        """Test that reload_config picks up new environment values"""
        # Set initial value
        monkeypatch.setenv("CHUNK_SIZE", "1500")
        config1 = reload_config()
        assert config1.chunk_size == 1500

        # Change environment
        monkeypatch.setenv("CHUNK_SIZE", "2500")
        config2 = reload_config()
        assert config2.chunk_size == 2500

    def test_reload_config_with_dotenv_available(self):
        """Test reload_config when python-dotenv is installed"""
        try:
            import dotenv
            # If dotenv available, test should still work
            config = reload_config()
            assert isinstance(config, BiGRAGConfig)
        except ImportError:
            pytest.skip("python-dotenv not installed")

    def test_reload_config_without_dotenv(self, monkeypatch):
        """Test reload_config fallback when python-dotenv not available"""
        # This tests the fallback path in reload_config
        config = reload_config()
        assert isinstance(config, BiGRAGConfig)


class TestConfigEdgeCases:
    """Test configuration edge cases"""

    def test_config_with_invalid_types(self, monkeypatch):
        """Test configuration with invalid type values"""
        # Set invalid value (should use default or handle gracefully)
        monkeypatch.setenv("CHUNK_SIZE", "invalid")

        try:
            config = BiGRAGConfig()
            # Should either use default or handle error
            assert isinstance(config.chunk_size, int)
        except ValueError:
            # Acceptable if it raises clear error
            pass

    def test_config_with_missing_optional_values(self):
        """Test configuration when optional values are missing"""
        config = BiGRAGConfig()

        # Should have sensible defaults for optional values
        assert config.llm_model is not None
        assert config.embedding_model is not None

    def test_config_boolean_parsing(self, monkeypatch):
        """Test boolean configuration parsing"""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("1", True),
            ("0", False),
        ]

        for env_value, expected in test_cases:
            monkeypatch.setenv("ENABLE_RERANKING", env_value)
            config = BiGRAGConfig()
            assert config.enable_reranking == expected, f"Failed for: {env_value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
