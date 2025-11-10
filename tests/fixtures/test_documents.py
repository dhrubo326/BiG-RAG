"""
Complex test data generators for BiG-RAG testing

This module provides realistic, complex test data that exposes edge cases
and validates system behavior under real-world conditions.
"""

from pathlib import Path
from typing import List, Dict, Any
import json


# Path to demo_test dataset (pre-built KG with complex football data)
DEMO_TEST_EXPR = Path(__file__).parent.parent.parent / "expr" / "demo_test"
DEMO_TEST_DATASET = Path(__file__).parent.parent.parent / "datasets" / "demo_test"


def get_demo_test_corpus() -> List[Dict[str, str]]:
    """
    Load demo_test corpus (complex football data)

    Returns realistic multi-entity documents with complex relationships
    """
    corpus_file = DEMO_TEST_DATASET / "raw" / "corpus.jsonl"

    if not corpus_file.exists():
        return []

    docs = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    return docs


def get_complex_multi_hop_documents() -> List[str]:
    """
    Documents designed for multi-hop reasoning tests


 Each requires chaining multiple facts
    """
    return [
        # Multi-hop fact chain 1
        "John Smith is the CEO of TechCorp. TechCorp is headquartered in San Francisco, California. "
        "San Francisco is part of the Bay Area region. The Bay Area is known for its technology industry. "
        "TechCorp was founded in 2010 and specializes in artificial intelligence.",

        # Multi-hop fact chain 2
        "Marie Curie was born in Warsaw, Poland in 1867. She moved to Paris to study at the Sorbonne University. "
        "At the Sorbonne, she met Pierre Curie, whom she married in 1895. Together they discovered radium and polonium. "
        "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.",

        # Multi-hop fact chain 3
        "The Amazon River originates in the Andes Mountains of Peru. It flows through Brazil for most of its length. "
        "The Amazon rainforest surrounds the river and is the largest rainforest in the world. "
        "Brazil is the largest country in South America and has Brasilia as its capital.",

        # Multi-hop fact chain 4
        "Albert Einstein developed the theory of relativity while working at the Swiss Patent Office in Bern. "
        "The theory of relativity revolutionized physics and led to the famous equation E=mc². "
        "Einstein later moved to Princeton University in the United States. "
        "Princeton University is located in New Jersey and is an Ivy League institution.",

        # Multi-hop fact chain 5 (overlapping entities)
        "Google was founded by Larry Page and Sergey Brin at Stanford University in 1998. "
        "Larry Page served as CEO of Alphabet Inc., Google's parent company. "
        "Alphabet Inc. is headquartered in Mountain View, California. "
        "Sergey Brin worked on the Google Search algorithm with Larry Page.",
    ]


def get_entity_type_edge_cases() -> List[Dict[str, Any]]:
    """
    Test data for entity type normalization edge cases

    Tests Bug #4 fix (entity type normalization)
    """
    return [
        # Uppercase entity types (should normalize to lowercase)
        {"name": "TestPerson1", "entity_type": "PERSON", "description": "Test person entity"},
        {"name": "TestOrg1", "entity_type": "ORGANIZATION", "description": "Test org entity"},
        {"name": "TestPlace1", "entity_type": "LOCATION", "description": "Test location"},

        # Mixed case entity types
        {"name": "TestPerson2", "entity_type": "Person", "description": "Mixed case person"},
        {"name": "TestOrg2", "entity_type": "Organization", "description": "Mixed case org"},

        # Variants that should normalize
        {"name": "Team1", "entity_type": "TEAM", "description": "Should normalize to organization"},
        {"name": "Player1", "entity_type": "PLAYER", "description": "Should normalize to person"},
        {"name": "City1", "entity_type": "CITY", "description": "Should normalize to geo"},

        # Unknown types (should normalize to 'category')
        {"name": "Unknown1", "entity_type": "UNKNOWN", "description": "Unknown type"},
        {"name": "Mystery1", "entity_type": "MYSTERY_TYPE", "description": "Mystery type"},
    ]


def get_complex_relations() -> List[Dict[str, Any]]:
    """
    Complex n-ary relations for testing relation extraction

    Tests bipartite edge creation and retrieval
    """
    return [
        {
            "src_id": "Lionel Messi",
            "tgt_id": "Inter Miami",
            "description": "Lionel Messi plays for Inter Miami as a forward in Major League Soccer since July 2023",
            "keywords": "plays for, footballer, club, MLS",
        },
        {
            "src_id": "Marie Curie",
            "tgt_id": "Nobel Prize",
            "description": "Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911, making her the first person to win Nobel Prizes in two different sciences",
            "keywords": "won, Nobel Prize, physics, chemistry, scientist",
        },
        {
            "src_id": "Albert Einstein",
            "tgt_id": "Theory of Relativity",
            "description": "Albert Einstein developed the theory of relativity, which includes special relativity (1905) and general relativity (1915), fundamentally changing our understanding of space and time",
            "keywords": "developed, theory, relativity, physics, E=mc2",
        },
        {
            "src_id": "Amazon River",
            "tgt_id": "Brazil",
            "description": "The Amazon River flows through Brazil for approximately 3,000 kilometers, making Brazil the country with the longest portion of the Amazon",
            "keywords": "flows through, river, country, South America",
        },
    ]


def get_metadata_edge_cases() -> List[Dict[str, Any]]:
    """
    Test metadata preservation and edge cases

    Tests Bug #2 fix (metadata preservation)
    """
    return [
        # Normal metadata
        {"title": "Normal Document", "category": "test", "tags": ["tag1", "tag2"]},

        # Missing fields
        {"title": "Only Title"},
        {"category": "Only Category"},
        {"tags": ["Only Tags"]},

        # Empty metadata
        {},

        # Special characters in metadata
        {"title": "Special Chars: @#$%^&*()", "category": "test/category", "tags": ["tag-1", "tag_2"]},

        # Very long metadata
        {"title": "A" * 500, "category": "B" * 100, "tags": ["C" * 50, "D" * 50]},

        # Unicode metadata
        {"title": "Unicode: 你好世界", "category": "中文", "tags": ["日本語", "한국어"]},

        # Nested structures (should handle gracefully)
        {"title": "Nested", "category": "test", "extra_data": {"nested": "value"}},
    ]


def get_challenging_queries() -> List[Dict[str, str]]:
    """
    Challenging queries that test retrieval robustness

    Returns list of dicts with 'query' and 'expected_context' keys
    """
    return [
        # Multi-hop query
        {
            "query": "Where was the CEO of TechCorp's company founded?",
            "expected_context": "San Francisco",
        },

        # Entity disambiguation
        {
            "query": "Which Curie won the Nobel Prize in Chemistry?",
            "expected_context": "Marie Curie",
        },

        # Relation-focused query
        {
            "query": "What is the relationship between Einstein and Princeton?",
            "expected_context": "worked at",
        },

        # Negation query (challenging)
        {
            "query": "Who won Nobel Prizes in fields other than Peace?",
            "expected_context": "Marie Curie",
        },

        # Temporal query
        {
            "query": "What did Einstein develop while at the Swiss Patent Office?",
            "expected_context": "theory of relativity",
        },

        # Superlative query
        {
            "query": "What is the largest rainforest in the world?",
            "expected_context": "Amazon",
        },

        # Count query (challenging)
        {
            "query": "How many Nobel Prizes did Marie Curie win?",
            "expected_context": "two",
        },

        # Vague query (tests robustness)
        {
            "query": "Tell me about that scientist who won prizes",
            "expected_context": "Marie Curie",
        },
    ]


def get_edge_case_documents() -> List[str]:
    """
    Edge case documents that test error handling
    """
    return [
        # Empty document
        "",

        # Single word
        "Test",

        # Very long document (5000+ words)
        " ".join(["word"] * 5000),

        # Special characters only
        "@#$%^&*()_+-=[]{}|;:',.<>?/~`",

        # Unicode characters
        "This is a test document with Unicode: 你好世界 مرحبا 안녕하세요 こんにちは",

        # Only numbers
        "1234567890 9876543210 1111111111",

        # Repeated entities (stress test entity extraction)
        "John John John Smith Smith Smith works at works at TechCorp TechCorp TechCorp.",

        # No punctuation
        "this is a document with no punctuation at all just words and spaces",

        # Excessive punctuation
        "This... is!!! a??? document,,, with!!! excessive!!! punctuation???!!!",

        # Mixed newlines and whitespace
        "This\n\nis\n\n\na\n\n\n\ndocument\n\nwith\n\nmany\n\nnewlines",
    ]


def get_performance_test_documents(count: int = 1000) -> List[str]:
    """
    Generate large number of realistic documents for performance testing

    Args:
        count: Number of documents to generate

    Returns:
        List of synthetic documents with realistic structure
    """
    try:
        from faker import Faker
        fake = Faker()

        documents = []
        for i in range(count):
            # Generate realistic multi-sentence documents
            doc = (
                f"{fake.name()} is a {fake.job()} at {fake.company()} in {fake.city()}, {fake.country()}. "
                f"The company specializes in {fake.bs()} and was founded in {fake.year()}. "
                f"{fake.text(max_nb_chars=200)} "
                f"They recently announced a partnership with {fake.company()} to develop {fake.catch_phrase()}."
            )
            documents.append(doc)

        return documents

    except ImportError:
        # Fallback if Faker not installed
        return [
            f"Document {i}: This is a test document with entity ENTITY{i} and relation REL{i}."
            for i in range(count)
        ]


def get_concurrent_test_queries() -> List[str]:
    """
    Queries for concurrent execution testing

    Returns varied queries that can be run in parallel
    """
    return [
        "Who is Lionel Messi?",
        "What is the capital of France?",
        "Who won the World Cup?",
        "What is machine learning?",
        "Where is the Eiffel Tower?",
        "Who developed the theory of relativity?",
        "What is Bitcoin?",
        "Which team does Messi play for?",
        "What is artificial intelligence?",
        "Where is the Amazon River?",
    ] * 10  # 100 queries total


def get_deletion_test_documents() -> List[Dict[str, Any]]:
    """
    Documents for testing cascade deletion

    Returns documents with known entity/relation counts for validation
    """
    return [
        {
            "content": "Delete Test Doc 1: Alice works at CompanyX in CityY.",
            "metadata": {"title": "Delete Test 1", "category": "test"},
            "expected_entities": ["Alice", "CompanyX", "CityY"],
            "expected_relations": 2,  # Alice-works-CompanyX, CompanyX-located-CityY
        },
        {
            "content": "Delete Test Doc 2: Bob collaborates with Alice on ProjectZ.",
            "metadata": {"title": "Delete Test 2", "category": "test"},
            "expected_entities": ["Bob", "Alice", "ProjectZ"],
            "expected_relations": 1,  # Bob-collaborates-Alice
        },
        {
            "content": "Delete Test Doc 3: CompanyX acquired CompanyZ in 2023.",
            "metadata": {"title": "Delete Test 3", "category": "test"},
            "expected_entities": ["CompanyX", "CompanyZ"],
            "expected_relations": 1,  # CompanyX-acquired-CompanyZ
        },
    ]


def validate_demo_test_kg() -> bool:
    """
    Validate that demo_test KG exists and is complete

    Returns:
        True if KG is valid, False otherwise
    """
    required_files = [
        "graph_chunk_entity_relation.graphml",
        "vdb_entities.json",
        "vdb_relations.json",
        "vdb_chunks.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
    ]

    for filename in required_files:
        if not (DEMO_TEST_EXPR / filename).exists():
            return False

    return True


# Module-level constants
DEMO_TEST_AVAILABLE = validate_demo_test_kg()

if __name__ == "__main__":
    # Test data generation
    print(f"Demo test KG available: {DEMO_TEST_AVAILABLE}")
    print(f"Demo test corpus docs: {len(get_demo_test_corpus())}")
    print(f"Multi-hop documents: {len(get_complex_multi_hop_documents())}")
    print(f"Entity type edge cases: {len(get_entity_type_edge_cases())}")
    print(f"Complex relations: {len(get_complex_relations())}")
    print(f"Challenging queries: {len(get_challenging_queries())}")
    print(f"Edge case documents: {len(get_edge_case_documents())}")
    print(f"Performance test docs (sample): {len(get_performance_test_documents(100))}")
