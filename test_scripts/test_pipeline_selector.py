"""
Test Suite for Pipeline Selector Helper

Tests all functionality of bigrag/pipeline_selector.py including:
- Document analysis
- Pipeline recommendation logic
- Preset management
- Convenience functions

Part of Phase 1 Step 5: Pipeline Selector Helper
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from bigrag.pipeline_selector import (
    PipelineSelector,
    PipelineType,
    ContentComplexity,
    PerformanceProfile,
    DocumentCharacteristics,
    PipelineRecommendation,
    quick_recommend,
    get_preset_config,
    list_all_presets
)


# Sample Documents for Testing

SAMPLE_GENERAL_DOCS = [
    """
    The history of computing spans many decades. Early computers were large
    machines that filled entire rooms. The invention of the transistor in 1947
    marked a turning point in computer technology.
    """,
    """
    Modern smartphones are powerful computing devices that fit in your pocket.
    They combine communication, computing, and entertainment in a single device.
    The first iPhone was released in 2007.
    """,
    """
    Artificial intelligence is transforming many industries. Machine learning
    algorithms can now recognize images, translate languages, and make predictions
    with impressive accuracy.
    """
]

SAMPLE_EDUCATIONAL_WITH_TABLES = [
    """
    # Chapter 1: Introduction to Database Systems

    A database is an organized collection of data. Database Management Systems (DBMS)
    provide tools to create, maintain, and query databases.

    ## Types of Databases

    | Type | Description | Example |
    |------|-------------|---------|
    | Relational | Uses tables with rows and columns | MySQL, PostgreSQL |
    | NoSQL | Flexible schema, document-based | MongoDB, Cassandra |
    | Graph | Stores relationships between entities | Neo4j, ArangoDB |

    ## Key Concepts

    1. **Schema**: The structure of the database
    2. **Query**: A request for data from the database
    3. **Index**: A data structure for faster lookups
    """,
    """
    # Chapter 2: SQL Basics

    SQL (Structured Query Language) is used to interact with relational databases.

    ## Common Commands

    | Command | Purpose | Example |
    |---------|---------|---------|
    | SELECT | Retrieve data | SELECT * FROM users |
    | INSERT | Add new data | INSERT INTO users VALUES ('Alice') |
    | UPDATE | Modify data | UPDATE users SET name='Bob' |
    | DELETE | Remove data | DELETE FROM users WHERE id=1 |

    ### Exercise 1.1
    Write a query to find all users who registered in 2024.
    """
]

SAMPLE_TECHNICAL_WITH_CODE = [
    """
    # Python Async Programming Guide

    Asynchronous programming allows concurrent execution of tasks.

    ## Basic Async Function

    ```python
    async def fetch_data(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()
    ```

    ## Running Async Code

    ```python
    import asyncio

    async def main():
        result = await fetch_data('https://api.example.com')
        print(result)

    asyncio.run(main())
    ```

    Key concepts:
    - `async def`: Declares an async function
    - `await`: Pauses execution until result is ready
    - `asyncio.run()`: Entry point for async programs
    """,
    """
    # Database Connection Pool

    Connection pooling improves performance by reusing connections.

    ```python
    class ConnectionPool:
        def __init__(self, max_size=10):
            self.max_size = max_size
            self.connections = []

        async def acquire(self):
            if self.connections:
                return self.connections.pop()
            return await self._create_connection()

        async def release(self, conn):
            if len(self.connections) < self.max_size:
                self.connections.append(conn)
            else:
                await conn.close()
    ```
    """
]

SAMPLE_SHORT_DOCS = [
    "Python is a programming language.",
    "JavaScript runs in browsers.",
    "SQL is for databases."
]

SAMPLE_LONG_DOCS = [
    """
    This is a very long document with extensive content. """ + "Lorem ipsum " * 500 + """
    The document continues with detailed explanations, examples, and references.
    """ + "More content here. " * 200
]


# Test Cases

async def test_analyze_simple_documents():
    """Test document analysis with simple documents."""
    print("\n[TEST 1] Analyzing simple documents...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_GENERAL_DOCS)

    print(f"  Average length: {chars.avg_length:.0f} chars")
    print(f"  Has tables: {chars.has_tables}")
    print(f"  Has code: {chars.has_code}")
    print(f"  Structure complexity: {chars.structure_complexity:.2f}")
    print(f"  Content type: {chars.content_type}")
    print(f"  Entity density: {chars.estimated_entity_density:.1f}/1000 chars")

    assert chars.content_type == 'general'
    assert chars.has_tables == False
    assert chars.has_code == False
    assert 0 <= chars.structure_complexity <= 1
    print("  [PASS] Simple document analysis")


async def test_analyze_educational_with_tables():
    """Test document analysis with educational content containing tables."""
    print("\n[TEST 2] Analyzing educational documents with tables...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_EDUCATIONAL_WITH_TABLES)

    print(f"  Average length: {chars.avg_length:.0f} chars")
    print(f"  Has tables: {chars.has_tables}")
    print(f"  Has code: {chars.has_code}")
    print(f"  Has lists: {chars.has_lists}")
    print(f"  Structure complexity: {chars.structure_complexity:.2f}")
    print(f"  Content type: {chars.content_type}")

    assert chars.has_tables == True
    assert chars.content_type == 'educational'
    assert chars.structure_complexity > 0.3  # Should have decent complexity
    print("  [PASS] Educational document with tables analysis")


async def test_analyze_technical_with_code():
    """Test document analysis with technical content containing code."""
    print("\n[TEST 3] Analyzing technical documents with code...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_TECHNICAL_WITH_CODE)

    print(f"  Average length: {chars.avg_length:.0f} chars")
    print(f"  Has tables: {chars.has_tables}")
    print(f"  Has code: {chars.has_code}")
    print(f"  Structure complexity: {chars.structure_complexity:.2f}")
    print(f"  Content type: {chars.content_type}")

    assert chars.has_code == True
    assert chars.content_type == 'technical'
    assert chars.structure_complexity > 0.2
    print("  [PASS] Technical document with code analysis")


async def test_recommend_large_corpus():
    """Test recommendation for large corpus (>10K docs)."""
    print("\n[TEST 4] Recommending for large corpus...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_GENERAL_DOCS)

    recommendation = selector.recommend_pipeline(
        characteristics=chars,
        corpus_size=15000,
        performance_profile=PerformanceProfile.SPEED
    )

    print(f"  Pipeline type: {recommendation.pipeline_type.value}")
    print(f"  Estimated cost: {recommendation.estimated_cost}")
    print(f"  Estimated time: {recommendation.estimated_time}")
    print(f"  Expected quality: {recommendation.expected_quality}")
    print(f"  Confidence: {recommendation.confidence:.2f}")
    print(f"  Reasoning: {recommendation.reasoning[0]}")

    assert recommendation.pipeline_type == PipelineType.STANDARD
    assert recommendation.estimated_time in ['fast', 'medium']
    assert len(recommendation.reasoning) > 0
    print("  [PASS] Large corpus recommendation (standard pipeline)")


async def test_recommend_educational_tables():
    """Test recommendation for educational content with tables."""
    print("\n[TEST 5] Recommending for educational content with tables...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_EDUCATIONAL_WITH_TABLES)

    recommendation = selector.recommend_pipeline(
        characteristics=chars,
        corpus_size=500,
        performance_profile=PerformanceProfile.ACCURACY
    )

    print(f"  Pipeline type: {recommendation.pipeline_type.value}")
    print(f"  Config keys: {list(recommendation.config.keys())}")
    print(f"  Entity merge strategy: {recommendation.config.get('entity_merge_strategy')}")
    print(f"  Expected quality: {recommendation.expected_quality}")
    print(f"  Reasoning: {recommendation.reasoning}")

    assert recommendation.pipeline_type == PipelineType.ENHANCED
    assert 'entity_merge_strategy' in recommendation.config
    assert recommendation.config.get('enable_entity_linking') == True
    print("  [PASS] Educational with tables recommendation (enhanced pipeline)")


async def test_recommend_speed_priority():
    """Test recommendation with speed priority."""
    print("\n[TEST 6] Recommending with speed priority...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_GENERAL_DOCS)

    recommendation = selector.recommend_pipeline(
        characteristics=chars,
        corpus_size=5000,
        performance_profile=PerformanceProfile.SPEED
    )

    print(f"  Pipeline type: {recommendation.pipeline_type.value}")
    print(f"  Estimated time: {recommendation.estimated_time}")
    print(f"  Config: {recommendation.config}")

    assert recommendation.estimated_time == 'fast'
    assert recommendation.pipeline_type == PipelineType.STANDARD
    print("  [PASS] Speed priority recommendation")


async def test_recommend_accuracy_priority():
    """Test recommendation with accuracy priority."""
    print("\n[TEST 7] Recommending with accuracy priority...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_EDUCATIONAL_WITH_TABLES)

    recommendation = selector.recommend_pipeline(
        characteristics=chars,
        corpus_size=800,
        performance_profile=PerformanceProfile.ACCURACY
    )

    print(f"  Pipeline type: {recommendation.pipeline_type.value}")
    print(f"  Expected quality: {recommendation.expected_quality}")
    print(f"  Enable gleaning: {recommendation.config.get('enable_gleaning')}")

    assert recommendation.expected_quality in ['very_good', 'excellent']
    print("  [PASS] Accuracy priority recommendation")


async def test_budget_constraint():
    """Test recommendation with budget constraint."""
    print("\n[TEST 8] Recommending with budget constraint...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_EDUCATIONAL_WITH_TABLES)

    # Without budget constraint - should recommend enhanced
    rec_no_budget = selector.recommend_pipeline(
        characteristics=chars,
        corpus_size=500,
        performance_profile=PerformanceProfile.BALANCED
    )

    # With low budget constraint - should switch to standard
    rec_low_budget = selector.recommend_pipeline(
        characteristics=chars,
        corpus_size=500,
        performance_profile=PerformanceProfile.BALANCED,
        budget_constraint='low'
    )

    print(f"  Without budget: {rec_no_budget.pipeline_type.value}, cost: {rec_no_budget.estimated_cost}")
    print(f"  With low budget: {rec_low_budget.pipeline_type.value}, cost: {rec_low_budget.estimated_cost}")

    # Low budget should prefer cheaper option
    if rec_no_budget.estimated_cost == 'high':
        assert rec_low_budget.estimated_cost in ['low', 'medium']
        print("  [PASS] Budget constraint applied")
    else:
        print("  [PASS] Budget constraint checked (no change needed)")


async def test_get_preset():
    """Test getting specific presets."""
    print("\n[TEST 9] Getting specific presets...")

    selector = PipelineSelector()

    # Get educational preset
    preset = selector.get_preset('educational_tables')

    print(f"  Preset: educational_tables")
    print(f"  Pipeline type: {preset['pipeline_type'].value}")
    print(f"  Use case: {preset['use_case']}")
    print(f"  Config keys: {list(preset['config'].keys())}")

    assert preset['pipeline_type'] == PipelineType.ENHANCED
    assert 'config' in preset
    assert 'use_case' in preset

    # Try invalid preset
    try:
        selector.get_preset('nonexistent_preset')
        assert False, "Should raise KeyError"
    except KeyError as e:
        print(f"  [OK] Invalid preset raises KeyError: {str(e)[:50]}...")

    print("  [PASS] Preset retrieval")


async def test_list_presets():
    """Test listing all presets."""
    print("\n[TEST 10] Listing all presets...")

    selector = PipelineSelector()
    presets = selector.list_presets()

    print(f"  Total presets: {len(presets)}")
    for name, use_case in list(presets.items())[:3]:
        print(f"    - {name}: {use_case}")

    assert len(presets) >= 8  # Should have at least 8 presets
    assert 'educational_tables' in presets
    assert 'fast_general' in presets
    print("  [PASS] List presets")


async def test_compare_presets():
    """Test comparing presets."""
    print("\n[TEST 11] Comparing presets...")

    selector = PipelineSelector()
    comparison = selector.compare_presets([
        'fast_general',
        'educational_tables',
        'large_corpus_fast'
    ])

    print(f"  Comparing {len(comparison)} presets:")
    for name, data in comparison.items():
        print(f"    {name}:")
        print(f"      Pipeline: {data['pipeline_type']}")
        print(f"      Cost: {data['cost']}, Time: {data['time']}, Quality: {data['quality']}")

    assert len(comparison) == 3
    assert all('cost' in data for data in comparison.values())
    print("  [PASS] Compare presets")


async def test_quick_recommend_function():
    """Test quick_recommend convenience function."""
    print("\n[TEST 12] Testing quick_recommend convenience function...")

    recommendation = quick_recommend(
        documents=SAMPLE_GENERAL_DOCS,
        corpus_size=1000,
        performance_profile='balanced',
        sample_size=3
    )

    print(f"  Pipeline type: {recommendation.pipeline_type.value}")
    print(f"  Config: {list(recommendation.config.keys())}")
    print(f"  Reasoning: {recommendation.reasoning[0][:60]}...")

    assert isinstance(recommendation, PipelineRecommendation)
    assert recommendation.pipeline_type in [PipelineType.STANDARD, PipelineType.ENHANCED]
    print("  [PASS] quick_recommend function")


async def test_get_preset_config_function():
    """Test get_preset_config convenience function."""
    print("\n[TEST 13] Testing get_preset_config convenience function...")

    config = get_preset_config('balanced_general')

    print(f"  Config type: {config['pipeline_type'].value}")
    print(f"  Config keys: {list(config['config'].keys())}")
    print(f"  Use case: {config['use_case']}")

    assert 'pipeline_type' in config
    assert 'config' in config
    assert 'use_case' in config
    print("  [PASS] get_preset_config function")


async def test_list_all_presets_function():
    """Test list_all_presets convenience function."""
    print("\n[TEST 14] Testing list_all_presets convenience function...")

    presets = list_all_presets()

    print(f"  Total presets: {len(presets)}")
    print(f"  Sample: {list(presets.keys())[:3]}")

    assert len(presets) >= 8
    assert isinstance(presets, dict)
    print("  [PASS] list_all_presets function")


async def test_empty_documents():
    """Test error handling for empty document list."""
    print("\n[TEST 15] Testing empty document list...")

    selector = PipelineSelector()

    try:
        chars = selector.analyze_documents([])
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"  [OK] Empty list raises ValueError: {e}")
        print("  [PASS] Empty document error handling")


async def test_sample_size_parameter():
    """Test sample_size parameter in analysis."""
    print("\n[TEST 16] Testing sample_size parameter...")

    selector = PipelineSelector()

    # Create 20 documents
    docs = SAMPLE_GENERAL_DOCS * 7  # 21 documents

    # Analyze with sample_size=5
    chars = selector.analyze_documents(docs, sample_size=5)

    print(f"  Total docs: {len(docs)}")
    print(f"  Sample size: 5")
    print(f"  Analysis completed: avg_length={chars.avg_length:.0f}")

    assert chars.avg_length > 0
    print("  [PASS] Sample size parameter")


async def test_recommendation_confidence():
    """Test confidence calculation in recommendations."""
    print("\n[TEST 17] Testing recommendation confidence...")

    selector = PipelineSelector()

    # Clear match: educational with tables
    chars_clear = selector.analyze_documents(SAMPLE_EDUCATIONAL_WITH_TABLES)
    rec_clear = selector.recommend_pipeline(
        chars_clear,
        corpus_size=500,
        performance_profile=PerformanceProfile.ACCURACY
    )

    # Less clear match: simple general docs
    chars_unclear = selector.analyze_documents(SAMPLE_GENERAL_DOCS)
    rec_unclear = selector.recommend_pipeline(
        chars_unclear,
        corpus_size=5000,
        performance_profile=PerformanceProfile.BALANCED
    )

    print(f"  Clear match confidence: {rec_clear.confidence:.2f}")
    print(f"  Less clear match confidence: {rec_unclear.confidence:.2f}")

    assert 0 <= rec_clear.confidence <= 1
    assert 0 <= rec_unclear.confidence <= 1
    print("  [PASS] Confidence calculation")


async def test_all_performance_profiles():
    """Test all performance profiles."""
    print("\n[TEST 18] Testing all performance profiles...")

    selector = PipelineSelector()
    chars = selector.analyze_documents(SAMPLE_GENERAL_DOCS)

    profiles = [
        PerformanceProfile.SPEED,
        PerformanceProfile.BALANCED,
        PerformanceProfile.ACCURACY
    ]

    for profile in profiles:
        rec = selector.recommend_pipeline(chars, corpus_size=1000, performance_profile=profile)
        print(f"  {profile.value}: {rec.pipeline_type.value}, time={rec.estimated_time}, quality={rec.expected_quality}")

        assert rec.pipeline_type in [PipelineType.STANDARD, PipelineType.ENHANCED]

    print("  [PASS] All performance profiles")


async def test_structure_complexity_calculation():
    """Test structure complexity calculation."""
    print("\n[TEST 19] Testing structure complexity calculation...")

    selector = PipelineSelector()

    # Simple docs (low complexity)
    chars_simple = selector.analyze_documents(SAMPLE_SHORT_DOCS)

    # Educational with tables (higher complexity)
    chars_complex = selector.analyze_documents(SAMPLE_EDUCATIONAL_WITH_TABLES)

    # Technical with code (higher complexity)
    chars_technical = selector.analyze_documents(SAMPLE_TECHNICAL_WITH_CODE)

    print(f"  Simple docs complexity: {chars_simple.structure_complexity:.2f}")
    print(f"  Educational+tables complexity: {chars_complex.structure_complexity:.2f}")
    print(f"  Technical+code complexity: {chars_technical.structure_complexity:.2f}")

    # Complex docs should have higher scores
    assert chars_complex.structure_complexity > chars_simple.structure_complexity
    assert chars_technical.structure_complexity > chars_simple.structure_complexity

    print("  [PASS] Structure complexity calculation")


async def test_entity_density_estimation():
    """Test entity density estimation."""
    print("\n[TEST 20] Testing entity density estimation...")

    selector = PipelineSelector()

    chars_general = selector.analyze_documents(SAMPLE_GENERAL_DOCS)
    chars_educational = selector.analyze_documents(SAMPLE_EDUCATIONAL_WITH_TABLES)
    chars_technical = selector.analyze_documents(SAMPLE_TECHNICAL_WITH_CODE)

    print(f"  General entity density: {chars_general.estimated_entity_density:.1f}/1000 chars")
    print(f"  Educational entity density: {chars_educational.estimated_entity_density:.1f}/1000 chars")
    print(f"  Technical entity density: {chars_technical.estimated_entity_density:.1f}/1000 chars")

    # Technical should have highest, general should have lowest
    assert chars_technical.estimated_entity_density >= chars_general.estimated_entity_density
    assert chars_educational.estimated_entity_density >= chars_general.estimated_entity_density

    print("  [PASS] Entity density estimation")


# Main Test Runner

async def run_all_tests():
    """Run all test cases."""
    print("=" * 70)
    print("PIPELINE SELECTOR TEST SUITE")
    print("=" * 70)

    tests = [
        test_analyze_simple_documents,
        test_analyze_educational_with_tables,
        test_analyze_technical_with_code,
        test_recommend_large_corpus,
        test_recommend_educational_tables,
        test_recommend_speed_priority,
        test_recommend_accuracy_priority,
        test_budget_constraint,
        test_get_preset,
        test_list_presets,
        test_compare_presets,
        test_quick_recommend_function,
        test_get_preset_config_function,
        test_list_all_presets_function,
        test_empty_documents,
        test_sample_size_parameter,
        test_recommendation_confidence,
        test_all_performance_profiles,
        test_structure_complexity_calculation,
        test_entity_density_estimation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"FAILED: {failed} tests")
    else:
        print("ALL TESTS PASSED")
    print("=" * 70)

    return passed, failed


if __name__ == '__main__':
    passed, failed = asyncio.run(run_all_tests())
    sys.exit(0 if failed == 0 else 1)
