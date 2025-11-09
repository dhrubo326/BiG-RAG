"""
Pytest configuration and shared fixtures for BiG-RAG tests

This file provides:
- Shared test fixtures (BiGRAG instances, test data, etc.)
- Test setup and teardown hooks
- Pytest plugins configuration
"""

import os
import sys
import shutil
import asyncio
import pytest
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import bigrag
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigrag import BiGRAG
from bigrag.base import QueryParam
from bigrag.config import BiGRAGConfig


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

TEST_WORKING_DIR = Path(__file__).parent / "test_output"
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================================
# SESSION FIXTURES (Run once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        "working_dir": str(TEST_WORKING_DIR),
        "fixtures_dir": str(TEST_FIXTURES_DIR),
        "openai_api_key": os.getenv("OPENAI_API_KEY", "test-key"),
        "enable_llm": os.getenv("ENABLE_LLM_TESTS", "false").lower() == "true",
    }


# ============================================================================
# FUNCTION FIXTURES (Run once per test function)
# ============================================================================

@pytest.fixture
async def clean_working_dir(test_config):
    """Clean working directory before and after tests"""
    working_dir = Path(test_config["working_dir"])

    # Clean before test
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    yield working_dir

    # Clean after test
    if working_dir.exists():
        shutil.rmtree(working_dir)


@pytest.fixture
async def bigrag_instance(clean_working_dir, test_config):
    """Create a fresh BiGRAG instance for testing"""
    rag = BiGRAG(
        working_dir=str(clean_working_dir),
        enable_llm_cache=False,
    )

    yield rag

    # Cleanup
    del rag


@pytest.fixture
async def bigrag_with_data(bigrag_instance):
    """BiGRAG instance with pre-inserted test data"""
    # Sample documents
    docs = [
        "Lionel Messi plays for Inter Miami in Major League Soccer.",
        "Messi won the 2022 FIFA World Cup with Argentina national team.",
        "Inter Miami is based in Miami, Florida, United States.",
        "Cristiano Ronaldo plays for Al Nassr in Saudi Pro League.",
        "The FIFA World Cup 2022 was held in Qatar.",
    ]

    metadata = [
        {"title": "Messi Career", "category": "sports", "tags": ["football", "messi"]},
        {"title": "World Cup 2022", "category": "sports", "tags": ["world cup", "argentina"]},
        {"title": "Inter Miami", "category": "sports", "tags": ["mls", "club"]},
        {"title": "Ronaldo Career", "category": "sports", "tags": ["football", "ronaldo"]},
        {"title": "Qatar World Cup", "category": "sports", "tags": ["world cup", "qatar"]},
    ]

    await bigrag_instance.ainsert(docs, metadata=metadata)

    yield bigrag_instance


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for testing"""
    return [
        "Python is a high-level programming language created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein developed the theory of relativity.",
        "Bitcoin is a decentralized digital currency.",
    ]


@pytest.fixture
def sample_metadata() -> List[Dict[str, Any]]:
    """Sample metadata for testing"""
    return [
        {"title": "Python Language", "category": "technology", "tags": ["programming"]},
        {"title": "Machine Learning", "category": "technology", "tags": ["ai", "ml"]},
        {"title": "Eiffel Tower", "category": "geography", "tags": ["landmark"]},
        {"title": "Einstein", "category": "science", "tags": ["physics"]},
        {"title": "Bitcoin", "category": "finance", "tags": ["cryptocurrency"]},
    ]


@pytest.fixture
def sample_queries() -> List[str]:
    """Sample queries for testing"""
    return [
        "Who created Python?",
        "What is machine learning?",
        "Where is the Eiffel Tower?",
        "What did Einstein develop?",
        "What is Bitcoin?",
    ]


@pytest.fixture
def sample_custom_kg():
    """Sample custom knowledge graph data"""
    return {
        "entities": [
            {"name": "TestEntity1", "entity_type": "person", "description": "Test person entity"},
            {"name": "TestEntity2", "entity_type": "organization", "description": "Test org entity"},
        ],
        "relations": [
            {
                "src_id": "TestEntity1",
                "tgt_id": "TestEntity2",
                "description": "TestEntity1 works for TestEntity2",
                "keywords": "works for, employed by",
            }
        ]
    }


# ============================================================================
# API TESTING FIXTURES
# ============================================================================

@pytest.fixture
def api_base_url():
    """Base URL for API testing"""
    return os.getenv("API_BASE_URL", "http://localhost:8001")


@pytest.fixture
async def api_client(api_base_url):
    """Async HTTP client for API testing"""
    import httpx
    async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
        yield client


# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def large_document_set() -> List[str]:
    """Generate large set of documents for stress testing"""
    from faker import Faker
    fake = Faker()

    return [
        f"{fake.name()} is a {fake.job()} working at {fake.company()}. "
        f"They live in {fake.city()}, {fake.country()}. "
        f"{fake.text(max_nb_chars=200)}"
        for _ in range(1000)
    ]


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest before tests run"""
    # Create test output directory
    TEST_WORKING_DIR.mkdir(parents=True, exist_ok=True)
    TEST_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Set environment variables for testing
    os.environ["BIGRAG_ENV"] = "test"
    os.environ["BIGRAG_LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers automatically based on file location
    for item in items:
        # Get relative path
        rel_path = Path(item.fspath).relative_to(Path(__file__).parent)

        # Auto-add markers based on directory
        if "unit" in str(rel_path):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(rel_path):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(rel_path):
            item.add_marker(pytest.mark.e2e)
        elif "regression" in str(rel_path):
            item.add_marker(pytest.mark.regression)
        elif "api" in str(rel_path):
            item.add_marker(pytest.mark.api)
        elif "frontend" in str(rel_path):
            item.add_marker(pytest.mark.frontend)
        elif "performance" in str(rel_path):
            item.add_marker(pytest.mark.performance)


def pytest_runtest_setup(item):
    """Hook before each test runs"""
    # Skip frontend tests if SKIP_FRONTEND is set
    if "frontend" in item.keywords and os.getenv("SKIP_FRONTEND", "false").lower() == "true":
        pytest.skip("Frontend tests disabled")

    # Skip LLM tests if API key not set
    if "llm" in item.keywords and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not set")


def pytest_runtest_teardown(item, nextitem):
    """Hook after each test runs"""
    # Force garbage collection after heavy tests
    if "performance" in item.keywords or "e2e" in item.keywords:
        import gc
        gc.collect()
