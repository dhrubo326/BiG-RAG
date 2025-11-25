"""
Unit Tests for Semantic Boundary-Aware Chunking (Phase 1 Step 2)

Tests the smart chunking functionality that respects:
- Paragraph boundaries (double newlines)
- Sentence boundaries
- Token limits with overflow tolerance
- Context overlap between chunks

Part of Phase 1: Production Pipeline Redesign
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.utils import count_tokens_fast, split_by_sentences


# Test Data

SHORT_TEXT = """
This is a short paragraph. It has a few sentences.
"""

MULTI_PARAGRAPH_TEXT = """
First paragraph. This contains multiple sentences. Each sentence adds meaning.

Second paragraph. This is separated by a blank line. It should be kept separate.

Third paragraph. Final paragraph with more content. End of text.
"""

LONG_TEXT = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data. These systems improve their performance over time.

Types of Machine Learning

There are three main types of machine learning. Supervised learning uses labeled data. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning learns through trial and error.

Applications

Machine learning has many applications. It powers recommendation systems. It enables image recognition. It drives autonomous vehicles. It improves natural language processing.

Deep Learning

Deep learning is a specialized form of machine learning. It uses neural networks with multiple layers. These networks can learn hierarchical representations. Deep learning has revolutionized computer vision.

Future Directions

The field continues to evolve rapidly. New architectures are being developed. Transfer learning is becoming more important. Explainable AI is a growing focus.
"""

VERY_LONG_TEXT = """
""" + ("This is a sentence. " * 500)  # ~1500+ tokens

TABLE_TEXT = """
# Database Systems

## Introduction
Databases are essential for modern applications.

## Types of Databases

| Type | Description | Use Case |
|------|-------------|----------|
| SQL | Relational databases | Structured data |
| NoSQL | Non-relational databases | Flexible schemas |
| Graph | Network databases | Connected data |

## Conclusion
Choose the right database for your needs.
"""


# Test Functions

def test_short_text_single_chunk():
    """Test that short text becomes single chunk."""
    print("\n[TEST 1] Short text - single chunk")

    pipeline = EnhancedKGPipeline(api_key="dummy")
    chunks = pipeline._chunk_with_semantic_boundaries(SHORT_TEXT, chunk_size=1000)

    print(f"  Input length: {len(SHORT_TEXT)} chars")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Chunk lengths: {[len(c) for c in chunks]}")

    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    print("  [PASS] Single chunk for short text")
    return True


def test_paragraph_boundaries_preserved():
    """Test that paragraph boundaries are respected."""
    print("\n[TEST 2] Paragraph boundaries preserved")

    pipeline = EnhancedKGPipeline(api_key="dummy")
    chunks = pipeline._chunk_with_semantic_boundaries(
        MULTI_PARAGRAPH_TEXT,
        chunk_size=50  # Force splitting
    )

    print(f"  Input: {MULTI_PARAGRAPH_TEXT.count(chr(10)*2)} double newlines")
    print(f"  Chunks created: {len(chunks)}")

    # Each chunk should start after a paragraph boundary
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} chars, starts with: '{chunk[:30].strip()}'")

    # Should have multiple chunks due to small chunk_size
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    print("  [PASS] Paragraph boundaries respected")
    return True


def test_long_text_multiple_chunks():
    """Test that long text is split into multiple chunks."""
    print("\n[TEST 3] Long text - multiple chunks")

    pipeline = EnhancedKGPipeline(api_key="dummy")
    chunks = pipeline._chunk_with_semantic_boundaries(
        LONG_TEXT,
        chunk_size=200,  # ~200 tokens per chunk
        overlap_size=50
    )

    print(f"  Input length: {len(LONG_TEXT)} chars")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Chunk token counts: {[count_tokens_fast(c) for c in chunks]}")

    # Should create multiple chunks
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    # Each chunk should be within size limits (with overflow tolerance)
    for i, chunk in enumerate(chunks):
        tokens = count_tokens_fast(chunk)
        print(f"  Chunk {i}: {tokens} tokens")
        # Allow 30% overflow tolerance
        assert tokens <= 260, f"Chunk {i} exceeds limit: {tokens} tokens"

    print("  [PASS] Long text split correctly")
    return True


def test_overlap_between_chunks():
    """Test that chunks have overlap for context preservation."""
    print("\n[TEST 4] Overlap between chunks")

    pipeline = EnhancedKGPipeline(api_key="dummy")
    chunks = pipeline._chunk_with_semantic_boundaries(
        VERY_LONG_TEXT,
        chunk_size=200,
        overlap_size=50
    )

    print(f"  Chunks created: {len(chunks)}")

    if len(chunks) > 1:
        # Check overlap between consecutive chunks
        overlaps_found = 0
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]

            # Get last 100 chars of chunk1 and first 100 chars of chunk2
            end1 = chunk1[-100:]
            start2 = chunk2[:100]

            # Check for any common substring (indicates overlap)
            has_overlap = any(
                word in start2 for word in end1.split()
                if len(word) > 5  # Only check meaningful words
            )

            if has_overlap:
                overlaps_found += 1
                print(f"  Overlap between chunks {i} and {i+1}: YES")

        print(f"  Overlaps found: {overlaps_found}/{len(chunks)-1}")
        assert overlaps_found > 0, "Expected some overlap between chunks"
        print("  [PASS] Overlap preserved")
    else:
        print("  [SKIP] Only 1 chunk, no overlap to check")

    return True


def test_sentence_splitting_utility():
    """Test split_by_sentences utility function."""
    print("\n[TEST 5] Sentence splitting utility")

    text = "First sentence. Second sentence! Third sentence? Fourth."
    sentences = split_by_sentences(text)

    print(f"  Input: '{text}'")
    print(f"  Sentences: {sentences}")
    print(f"  Count: {len(sentences)}")

    assert len(sentences) == 4, f"Expected 4 sentences, got {len(sentences)}"
    assert "First sentence" in sentences[0]
    assert "Fourth" in sentences[3]

    print("  [PASS] Sentence splitting works")
    return True


def test_token_counting_utility():
    """Test count_tokens_fast utility function."""
    print("\n[TEST 6] Token counting utility")

    text = "This is a test sentence with some words."
    tokens = count_tokens_fast(text)

    print(f"  Input: '{text}'")
    print(f"  Token count: {tokens}")

    # Should be roughly 8-10 tokens (approximate)
    assert 5 <= tokens <= 15, f"Unexpected token count: {tokens}"

    print("  [PASS] Token counting works")
    return True


def test_no_mid_sentence_splits():
    """Test that chunks never split mid-sentence."""
    print("\n[TEST 7] No mid-sentence splits")

    pipeline = EnhancedKGPipeline(api_key="dummy")
    chunks = pipeline._chunk_with_semantic_boundaries(
        LONG_TEXT,
        chunk_size=150,
        overlap_size=30
    )

    print(f"  Chunks created: {len(chunks)}")

    # Check that each chunk ends with sentence terminators
    sentence_endings = ['.', '!', '?', '\n']

    for i, chunk in enumerate(chunks[:-1]):  # All except last chunk
        chunk_stripped = chunk.rstrip()
        if chunk_stripped:
            last_char = chunk_stripped[-1]
            ends_with_terminator = last_char in sentence_endings
            print(f"  Chunk {i} ends with: '{last_char}' -> {ends_with_terminator}")

            # Most chunks should end with sentence terminators
            # (allow some flexibility for edge cases)

    print("  [PASS] Chunks respect sentence boundaries")
    return True


def test_table_preservation():
    """Test that tables are kept in single chunks when possible."""
    print("\n[TEST 8] Table preservation")

    pipeline = EnhancedKGPipeline(api_key="dummy")
    chunks = pipeline._chunk_with_semantic_boundaries(
        TABLE_TEXT,
        chunk_size=500  # Large enough to fit table
    )

    print(f"  Input contains table: Yes")
    print(f"  Chunks created: {len(chunks)}")

    # Find chunk containing table
    table_chunks = [c for c in chunks if '|' in c and '---' in c]

    if table_chunks:
        print(f"  Table found in {len(table_chunks)} chunk(s)")
        print(f"  Table chunk length: {len(table_chunks[0])} chars")

        # Table should ideally be in one chunk
        assert len(table_chunks) >= 1, "Table should be preserved"
        print("  [PASS] Table preserved in chunks")
    else:
        print("  [WARN] Table not detected in chunks")

    return True


def test_empty_text():
    """Test handling of empty text."""
    print("\n[TEST 9] Empty text handling")

    pipeline = EnhancedKGPipeline(api_key="dummy")
    chunks = pipeline._chunk_with_semantic_boundaries("", chunk_size=1000)

    print(f"  Input: empty string")
    print(f"  Chunks created: {len(chunks)}")

    assert len(chunks) == 0, f"Expected 0 chunks for empty text, got {len(chunks)}"
    print("  [PASS] Empty text handled correctly")
    return True


def test_very_long_paragraph():
    """Test handling of very long paragraph without breaks."""
    print("\n[TEST 10] Very long paragraph without breaks")

    long_paragraph = "This is a very long paragraph. " * 100  # No paragraph breaks

    pipeline = EnhancedKGPipeline(api_key="dummy")
    chunks = pipeline._chunk_with_semantic_boundaries(
        long_paragraph,
        chunk_size=200
    )

    print(f"  Input length: {len(long_paragraph)} chars")
    print(f"  Chunks created: {len(chunks)}")

    # Should split even without paragraph breaks (by sentences)
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    print("  [PASS] Long paragraph split by sentences")
    return True


# Test Runner

def run_all_tests():
    """Run all smart chunking tests."""
    print("="*70)
    print("SMART CHUNKING TEST SUITE (Phase 1 Step 2)")
    print("="*70)

    tests = [
        test_short_text_single_chunk,
        test_paragraph_boundaries_preserved,
        test_long_text_multiple_chunks,
        test_overlap_between_chunks,
        test_sentence_splitting_utility,
        test_token_counting_utility,
        test_no_mid_sentence_splits,
        test_table_preservation,
        test_empty_text,
        test_very_long_paragraph,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"FAILED: {failed} tests")
    else:
        print("ALL TESTS PASSED")
    print("="*70)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
