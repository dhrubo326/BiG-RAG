"""
Test script to verify relation text extraction from bipartite_edge node IDs

This script tests the _extract_relation_text() function that was added to backend/server.py
"""

import html


def _extract_relation_text(node_id: str) -> str:
    """
    Extract clean relation text from bipartite_edge node IDs.

    Example:
        Input: '<bipartite_edge>"Netflix is an American streaming service"'
        Output: 'Netflix is an American streaming service'
    """
    # Remove <bipartite_edge> prefix if present
    text = node_id
    if text.startswith("<bipartite_edge>"):
        text = text[len("<bipartite_edge>"):]

    # Decode HTML entities (e.g., &quot; -> ")
    text = html.unescape(text)

    # Remove surrounding quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    return text.strip()


def test_relation_extraction():
    """Test various formats of relation node IDs"""

    test_cases = [
        # Format 1: Full GraphML format with XML entities
        (
            '<bipartite_edge>&quot;Netflix is an American subscription video on-demand streaming service.&quot;',
            'Netflix is an American subscription video on-demand streaming service.'
        ),
        # Format 2: With <bipartite_edge> but regular quotes
        (
            '<bipartite_edge>"Python is a high-level programming language"',
            'Python is a high-level programming language'
        ),
        # Format 3: With single quotes
        (
            "<bipartite_edge>'Albert Einstein was a theoretical physicist'",
            'Albert Einstein was a theoretical physicist'
        ),
        # Format 4: No prefix (should still work)
        (
            '"This is a relation description"',
            'This is a relation description'
        ),
        # Format 5: Multiple HTML entities
        (
            '&quot;Netflix &amp; Amazon are streaming services&quot;',
            'Netflix & Amazon are streaming services'
        ),
        # Format 6: Empty text
        (
            '<bipartite_edge>""',
            ''
        ),
    ]

    print("Testing relation text extraction...")
    print("=" * 80)

    all_passed = True
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = _extract_relation_text(input_text)
        passed = result == expected
        all_passed = all_passed and passed

        status = "[OK]" if passed else "[FAIL]"
        print(f"\nTest {i}: {status}")
        print(f"  Input:    {input_text[:60]}{'...' if len(input_text) > 60 else ''}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")

        if not passed:
            print(f"  ERROR: Mismatch!")

    print("\n" + "=" * 80)
    if all_passed:
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed!")
        return 1


def test_real_graphml_data():
    """Test with actual node IDs from demo_test GraphML"""

    print("\n\nTesting with real demo_test data...")
    print("=" * 80)

    # Actual node IDs from expr/demo_test/graph_chunk_entity_relation.graphml
    real_node_ids = [
        '<bipartite_edge>&quot;Netflix is an American subscription video on-demand streaming service.&quot;',
        '&quot;NETFLIX&quot;',  # This is an entity, not a relation
        '<bipartite_edge>&quot;Python is a high-level, general-purpose programming language.&quot;',
    ]

    expected_outputs = [
        'Netflix is an American subscription video on-demand streaming service.',
        'NETFLIX',  # Entity should just remove quotes
        'Python is a high-level, general-purpose programming language.',
    ]

    for node_id, expected in zip(real_node_ids, expected_outputs):
        result = _extract_relation_text(node_id)
        is_relation = node_id.startswith('<bipartite_edge>')

        print(f"\nNode type: {'RELATION' if is_relation else 'ENTITY'}")
        print(f"  Raw ID:   {node_id[:60]}...")
        print(f"  Cleaned:  {result}")
        print(f"  Expected: {expected}")
        print(f"  Match:    {'[OK]' if result == expected else '[FAIL]'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    # Run unit tests
    exit_code = test_relation_extraction()

    # Run real data tests
    test_real_graphml_data()

    sys.exit(exit_code)
