# Entity Type Validation & Normalization Plan

**Document Version**: 1.0
**Date**: 2025-01-08
**Status**: Implementation Pending
**Priority**: MEDIUM (Improves consistency, non-breaking change)

---

## Executive Summary

**Current Problem**: No validation exists for entity types extracted by LLM. The system:
- Accepts any type the LLM produces
- Creates inconsistency between configured types and actual types
- Produces uppercase types (`"PERSON"`) instead of lowercase (`"person"`)
- Allows non-standard types like `"STATISTIC"`, `"CONCEPT"` not in configuration

**Solution**: Add type normalization and validation layer after LLM extraction.

**Impact**:
- Improved type consistency across graphs
- Better type-based filtering in queries
- Clearer entity type distribution analytics
- No performance impact (simple string mapping)

**Breaking Change**: No, existing graphs continue to work.

---

## Problem Analysis

### Current Behavior

**Configuration** ([bigrag/prompt.py:11](d:\BiG-RAG\bigrag\prompt.py#L11)):
```python
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]
```

**LLM Output** (actual types in demo_test graph):
```
"PERSON"       # Should be: "person"
"TEAM"         # Should be: "organization"
"LEAGUE"       # Should be: "organization"
"STATISTIC"    # Should be: "event" or "category"
"CONCEPT"      # Should be: "category"
"EVENT"        # Correct format, but uppercase
"GEO"          # Correct, but uppercase
```

**Current Code** ([bigrag/operate.py:121](d:\BiG-RAG\bigrag\operate.py#L121)):
```python
entity_type = clean_str(record_attributes[2].upper())  # Just uppercases, no validation
```

**Problems**:
1. No mapping from LLM output to configured types
2. No fallback for unknown types
3. Inconsistent casing (uppercase in graph, lowercase in config)
4. No logging of unknown types for debugging

---

## Proposed Solution

### Architecture

```
LLM Extraction
    ↓
Extracted Type: "TEAM"
    ↓
normalize_entity_type()
    ↓
Check mapping: "TEAM" → "organization"
    ↓
Validate against allowed types
    ↓
Return: "organization"
```

### Type Mapping Strategy

Create a comprehensive mapping from LLM outputs to standard types:

```python
TYPE_NORMALIZATION_MAP = {
    # Person types
    "PERSON": "person",
    "PEOPLE": "person",
    "INDIVIDUAL": "person",
    "PLAYER": "person",
    "ATHLETE": "person",
    "COACH": "person",
    "MANAGER": "person",

    # Organization types
    "ORGANIZATION": "organization",
    "TEAM": "organization",
    "CLUB": "organization",
    "LEAGUE": "organization",
    "COMPANY": "organization",
    "INSTITUTION": "organization",
    "GOVERNMENT": "organization",

    # Geo types
    "GEO": "geo",
    "LOCATION": "geo",
    "PLACE": "geo",
    "CITY": "geo",
    "COUNTRY": "geo",
    "REGION": "geo",
    "STADIUM": "geo",

    # Event types
    "EVENT": "event",
    "TOURNAMENT": "event",
    "MATCH": "event",
    "GAME": "event",
    "COMPETITION": "event",
    "CEREMONY": "event",

    # Category/Concept types
    "CATEGORY": "category",
    "CONCEPT": "category",
    "TOPIC": "category",
    "STATISTIC": "category",
    "METRIC": "category",
    "ROLE": "category",
    "OBJECT": "category",
}
```

---

## Implementation

### Step 1: Add Normalization Function

**File**: `bigrag/operate.py` (add after imports)

```python
# Type normalization configuration
TYPE_NORMALIZATION_MAP = {
    # Person types
    "PERSON": "person",
    "PEOPLE": "person",
    "INDIVIDUAL": "person",
    "PLAYER": "person",
    "ATHLETE": "person",
    "COACH": "person",
    "MANAGER": "person",
    "AUTHOR": "person",
    "SCIENTIST": "person",

    # Organization types
    "ORGANIZATION": "organization",
    "TEAM": "organization",
    "CLUB": "organization",
    "LEAGUE": "organization",
    "COMPANY": "organization",
    "INSTITUTION": "organization",
    "GOVERNMENT": "organization",
    "ASSOCIATION": "organization",
    "FEDERATION": "organization",

    # Geo types
    "GEO": "geo",
    "LOCATION": "geo",
    "PLACE": "geo",
    "CITY": "geo",
    "COUNTRY": "geo",
    "REGION": "geo",
    "STADIUM": "geo",
    "VENUE": "geo",
    "CONTINENT": "geo",

    # Event types
    "EVENT": "event",
    "TOURNAMENT": "event",
    "MATCH": "event",
    "GAME": "event",
    "COMPETITION": "event",
    "CEREMONY": "event",
    "CHAMPIONSHIP": "event",
    "SEASON": "event",

    # Category/Concept types
    "CATEGORY": "category",
    "CONCEPT": "category",
    "TOPIC": "category",
    "STATISTIC": "category",
    "METRIC": "category",
    "ROLE": "category",
    "OBJECT": "category",
    "POSITION": "category",
    "ACTION": "category",
}

def normalize_entity_type(extracted_type: str, allowed_types: list = None) -> str:
    """
    Normalize and validate entity type from LLM extraction.

    Args:
        extracted_type: Raw type string from LLM (e.g., "TEAM", "Person", "geo")
        allowed_types: List of allowed types. If None, uses DEFAULT_ENTITY_TYPES

    Returns:
        Normalized type string (lowercase, validated)

    Examples:
        >>> normalize_entity_type("TEAM")
        "organization"
        >>> normalize_entity_type("Person")
        "person"
        >>> normalize_entity_type("UNKNOWN_TYPE")
        "category"  # Fallback
    """
    if allowed_types is None:
        from .prompt import PROMPTS
        allowed_types = PROMPTS.get("DEFAULT_ENTITY_TYPES", ["organization", "person", "geo", "event", "category"])

    # Normalize to uppercase for lookup
    normalized_upper = extracted_type.strip().upper()

    # Check mapping
    if normalized_upper in TYPE_NORMALIZATION_MAP:
        mapped_type = TYPE_NORMALIZATION_MAP[normalized_upper]

        # Verify mapped type is in allowed list
        if mapped_type in allowed_types:
            return mapped_type
        else:
            logger.warning(f"Mapped type '{mapped_type}' not in allowed types: {allowed_types}. Using 'category' fallback.")
            return "category"

    # Check if already a valid type (case-insensitive)
    for allowed_type in allowed_types:
        if normalized_upper == allowed_type.upper():
            return allowed_type

    # Unknown type - log warning and use fallback
    logger.warning(f"Unknown entity type: '{extracted_type}'. Fallback to 'category'. Consider adding to TYPE_NORMALIZATION_MAP.")
    return "category"
```

---

### Step 2: Update Entity Extraction

**File**: `bigrag/operate.py`
**Function**: `_pack_single_entity()`
**Lines**: ~120-130

**Current**:
```python
def _pack_single_entity(record_attributes: list[str], chunk_key: str):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1])
    entity_type = clean_str(record_attributes[2].upper())  # ← Just uppercase, no validation
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,  # ← Could be anything
        description=entity_description,
        weight=weight,
        source_id=entity_source_id,
    )
```

**Proposed**:
```python
def _pack_single_entity(record_attributes: list[str], chunk_key: str):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    entity_name = clean_str(record_attributes[1])
    raw_entity_type = clean_str(record_attributes[2])
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    entity_source_id = chunk_key

    # NEW: Normalize and validate entity type
    entity_type = normalize_entity_type(raw_entity_type)

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,  # ← Now validated
        description=entity_description,
        weight=weight,
        source_id=entity_source_id,
    )
```

---

### Step 3: Add Configuration Option

**File**: `bigrag/bigrag.py`
**Class**: `BiGRAG.__init__()`

Add parameter to allow custom type mappings:

```python
@dataclass
class BiGRAG:
    # ... existing parameters ...

    entity_types: list = None
    # Custom entity types to use instead of defaults
    # Default: None (uses PROMPTS["DEFAULT_ENTITY_TYPES"])
    # Example: ["product", "technology", "disease", "drug"]

    custom_type_mapping: dict = None
    # Custom type normalization mapping
    # Default: None (uses built-in TYPE_NORMALIZATION_MAP)
    # Example: {"BRAND": "product", "FRAMEWORK": "technology"}
```

**Implementation**:
```python
def __post_init__(self):
    # ... existing initialization ...

    # Store entity types in global_config
    if self.entity_types is not None:
        self.global_config["entity_types"] = self.entity_types
    else:
        from .prompt import PROMPTS
        self.global_config["entity_types"] = PROMPTS.get("DEFAULT_ENTITY_TYPES")

    # Merge custom type mapping if provided
    if self.custom_type_mapping is not None:
        from .operate import TYPE_NORMALIZATION_MAP
        TYPE_NORMALIZATION_MAP.update(self.custom_type_mapping)
```

---

## Testing

### Unit Tests

**File**: `test_scripts/test_entity_type_normalization.py` (new)

```python
import pytest
from bigrag.operate import normalize_entity_type

def test_normalize_standard_types():
    """Test normalization of standard types"""
    assert normalize_entity_type("PERSON") == "person"
    assert normalize_entity_type("TEAM") == "organization"
    assert normalize_entity_type("LEAGUE") == "organization"
    assert normalize_entity_type("GEO") == "geo"
    assert normalize_entity_type("EVENT") == "event"
    assert normalize_entity_type("CONCEPT") == "category"
    assert normalize_entity_type("STATISTIC") == "category"

def test_normalize_case_insensitive():
    """Test case-insensitive normalization"""
    assert normalize_entity_type("person") == "person"
    assert normalize_entity_type("Person") == "person"
    assert normalize_entity_type("PERSON") == "person"
    assert normalize_entity_type("PeRsOn") == "person"

def test_normalize_unknown_type():
    """Test fallback for unknown types"""
    assert normalize_entity_type("UNKNOWN_TYPE") == "category"
    assert normalize_entity_type("RANDOM") == "category"

def test_normalize_with_custom_allowed_types():
    """Test normalization with custom allowed types"""
    custom_allowed = ["product", "technology"]

    # Should fail to map "PERSON" to standard type and fallback
    result = normalize_entity_type("PERSON", allowed_types=custom_allowed)
    assert result == "category"  # Fallback since "person" not in custom_allowed

def test_normalize_already_valid():
    """Test types that are already valid"""
    assert normalize_entity_type("organization") == "organization"
    assert normalize_entity_type("person") == "person"
    assert normalize_entity_type("geo") == "geo"
```

---

### Integration Tests

**File**: `test_scripts/test_graph_with_type_validation.py` (new)

```python
import pytest
import tempfile
import shutil
import networkx as nx
from bigrag import BiGRAG
from bigrag.llm import gpt_4o_mini_complete, openai_embedding

@pytest.mark.integration
def test_entity_types_normalized():
    """Test that extracted entity types are normalized"""
    temp_dir = tempfile.mkdtemp()

    try:
        rag = BiGRAG(
            working_dir=temp_dir,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=openai_embedding()
        )

        documents = [
            {
                "content": "Lionel Messi plays for Inter Miami. The team competes in MLS.",
                "title": "Football News"
            }
        ]

        rag.insert(documents)

        # Load graph
        import os
        graph_path = os.path.join(temp_dir, "graph_chunk_entity_relation.graphml")
        graph = nx.read_graphml(graph_path)

        # Check entity types
        entity_nodes = [
            (n, d) for n, d in graph.nodes(data=True)
            if d.get('role') == 'entity'
        ]

        allowed_types = ["organization", "person", "geo", "event", "category"]

        for node_id, node_data in entity_nodes:
            entity_type = node_data.get('entity_type', 'unknown')

            # Verify type is in allowed list
            assert entity_type in allowed_types, f"Entity '{node_id}' has invalid type: {entity_type}"

            # Verify lowercase
            assert entity_type.islower(), f"Entity '{node_id}' type not lowercase: {entity_type}"

        print(f"✅ All {len(entity_nodes)} entities have valid, normalized types")

    finally:
        shutil.rmtree(temp_dir)
```

---

## Migration Guide

### For Existing Graphs

**No action required!** This is a non-breaking change:
- Existing graphs with old types (uppercase, non-standard) will continue to work
- New graphs will have normalized types
- Both can coexist

**Optional**: Rebuild for consistency
```bash
# Rebuild to get normalized types in existing graphs
python script_build.py --data_source YOUR_DATASET
```

---

### For Custom Entity Types

If you use custom entity types (e.g., medical domain):

**Before**:
```python
# No way to ensure LLM uses your types
rag = BiGRAG(...)
# LLM might extract "SYMPTOM", "DISEASE", "MEDICINE", etc.
```

**After**:
```python
# Define custom types and mapping
rag = BiGRAG(
    entity_types=["disease", "symptom", "treatment", "drug"],
    custom_type_mapping={
        "ILLNESS": "disease",
        "CONDITION": "disease",
        "MEDICINE": "drug",
        "MEDICATION": "drug",
        "THERAPY": "treatment",
        "PROCEDURE": "treatment",
    }
)
```

---

## Benefits

1. **Consistency**: All graphs use same type vocabulary
2. **Queryability**: Type-based filtering works reliably
3. **Analytics**: Accurate entity type distribution metrics
4. **Extensibility**: Easy to add new type mappings for domain-specific use
5. **Debugging**: Warnings logged for unknown types

---

## Future Enhancements

### Phase 2: LLM Prompt Improvement

Update prompt examples to show desired types:

**File**: `bigrag/prompt.py:47-159`

**Current examples use**:
- `"person"`, `"role"`, `"concept"`, `"location"`, `"event"`, `"object"`, `"action"`

**Should use**:
- `"person"`, `"organization"`, `"geo"`, `"event"`, `"category"`

**Implementation**: Update all 3 examples to use consistent types.

---

### Phase 3: Type Hierarchy

Support hierarchical types:

```python
TYPE_HIERARCHY = {
    "person": {
        "subtypes": ["player", "coach", "manager"],
        "parent": None
    },
    "organization": {
        "subtypes": ["team", "league", "company"],
        "parent": None
    }
}

def normalize_entity_type(extracted_type, allow_subtypes=False):
    # If allow_subtypes, keep "player" instead of mapping to "person"
    pass
```

---

## Appendix

### Type Distribution Analysis

**Script**: `test_scripts/analyze_entity_types.py` (new)

```python
import networkx as nx
from collections import Counter

def analyze_entity_types(graphml_path):
    """Analyze entity type distribution in graph"""
    graph = nx.read_graphml(graphml_path)

    entity_nodes = [
        d for n, d in graph.nodes(data=True)
        if d.get('role') == 'entity'
    ]

    types = [node.get('entity_type', 'unknown') for node in entity_nodes]
    type_counts = Counter(types)

    print(f"Entity Type Distribution ({len(entity_nodes)} total entities):")
    for type_name, count in type_counts.most_common():
        percentage = (count / len(entity_nodes)) * 100
        print(f"  {type_name:20s}: {count:5d} ({percentage:5.1f}%)")

if __name__ == "__main__":
    analyze_entity_types("expr/demo_test/graph_chunk_entity_relation.graphml")
```

**Sample Output**:
```
Entity Type Distribution (125 total entities):
  person              :    65 ( 52.0%)
  organization        :    35 ( 28.0%)
  geo                 :    15 ( 12.0%)
  event               :     8 (  6.4%)
  category            :     2 (  1.6%)
```

---

## Document End

**Last Updated**: 2025-01-08
**Implementation Status**: Pending user approval

**Checklist**:
- [ ] Add `normalize_entity_type()` function
- [ ] Update `_pack_single_entity()`
- [ ] Add configuration options
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update documentation
- [ ] Add type analysis script
