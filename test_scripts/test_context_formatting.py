from bigrag.operate import _format_knowledge_as_structured

def test_empty_knowledge():
    result = _format_knowledge_as_structured([])
    assert result == "No relevant knowledge found."
    print("[OK] Empty knowledge test passed")

def test_entity_formatting():
    knowledge = [{
        "<knowledge>": "ENTITY: Albert Einstein (person) - Physicist",
        "<coherence>": 0.95,
        "<source_ids>": ["chunk-001", "chunk-003"],
        "<type>": "entity"
    }]
    result = _format_knowledge_as_structured(knowledge)
    assert "### Knowledge Graph - Entities" in result
    assert "Relevance Score: 0.95" in result
    assert "Sources: chunk-001, chunk-003" in result
    print("[OK] Entity formatting test passed")

def test_chunk_with_metadata():
    knowledge = [{
        "<knowledge>": "Einstein was born in Germany in 1879.",
        "<coherence>": 0.92,
        "<source_ids>": ["chunk-001"],
        "<type>": "chunk",
        "<metadata>": {
            "category": "Biography",
            "title": "Einstein's Early Life",
            "tags": ["Physics", "History"]
        }
    }]
    result = _format_knowledge_as_structured(knowledge)
    assert "### Document Chunks" in result
    assert "Category=Biography" in result
    assert "Title=Einstein's Early Life" in result
    assert "Tags=Physics,History" in result
    print("[OK] Chunk metadata test passed")

def test_mixed_types():
    knowledge = [
        {"<knowledge>": "ENTITY: Einstein", "<type>": "entity", "<coherence>": 0.95, "<source_ids>": ["c1"]},
        {"<knowledge>": "Einstein won Nobel Prize", "<type>": "relation", "<coherence>": 0.90, "<source_ids>": ["c2"]},
        {"<knowledge>": "In 1921, Einstein received...", "<type>": "chunk", "<coherence>": 0.88, "<source_ids>": ["c3"]}
    ]
    result = _format_knowledge_as_structured(knowledge)
    assert "### Knowledge Graph - Entities" in result
    assert "### Knowledge Graph - Relations" in result
    assert "### Document Chunks" in result
    print("[OK] Mixed types test passed")

if __name__ == "__main__":
    test_empty_knowledge()
    test_entity_formatting()
    test_chunk_with_metadata()
    test_mixed_types()
    print("\n[SUCCESS] All tests passed!")
