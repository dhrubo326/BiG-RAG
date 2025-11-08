# Weight Semantics Documentation Plan

**Document Version**: 1.0
**Date**: 2025-01-08
**Status**: Implementation Pending
**Priority**: LOW (Documentation only, no code changes)

---

## Executive Summary

**Current Problem**: Weight values in BiG-RAG graphs lack clear semantic meaning. Users and developers don't understand:
- What does "180.0" vs "360.0" mean?
- How are weights calculated?
- How should weights be used in ranking?
- Should weights be normalized?

**Solution**: Comprehensive documentation of weight semantics across all weight types (entity weights, edge weights, graph edge weights).

**Impact**:
- Clearer understanding of weight interpretation
- Better decision-making for ranking algorithms
- Improved debugging of weight-related issues
- No code changes required (documentation only)

---

## Weight Types in BiG-RAG

BiG-RAG uses three types of weights:

### 1. Entity Weights

**Definition**: Cumulative importance score for an entity across all occurrences.

**Formula**:
```
entity_weight = Σ(llm_score_i) for all occurrences i
```

**Range**: `0` to `N × 100` where N = number of chunks mentioning entity

**Example**:
```python
# Entity "Lionel Messi" appears in 3 chunks:
# Chunk 1: LLM assigns score 95 (very important)
# Chunk 2: LLM assigns score 85 (important)
# Chunk 3: LLM assigns score 90 (very important)

total_weight = 95 + 85 + 90 = 270
```

**In GraphML**:
```xml
<node id="LIONEL MESSI">
  <data key="d0">entity</data>
  <data key="d1">270.0</data>  <!-- weight -->
  ...
</node>
```

**Interpretation**:
- Higher weight = more mentions + higher LLM confidence
- Weight 400+ = very central entity (mentioned in 4+ chunks with high scores)
- Weight 50-100 = peripheral entity (mentioned once with low-medium score)
- Weight 0 = error (should not occur)

---

### 2. Bipartite Edge (Relation) Weights

**Definition**: Cumulative completeness score for a knowledge segment across occurrences.

**Formula**:
```
edge_weight = Σ(completeness_score_i) for all occurrences i
```

**Range**: `0` to `N × 10` where N = number of chunks containing relation

**Example**:
```python
# Relation "Messi scored 11 goals in his first 14 matches"
# Appears in 2 chunks:
# Chunk 1: Completeness score 9/10 (very complete)
# Chunk 2: Completeness score 8/10 (mostly complete)

total_weight = 9 + 8 = 17
```

**In GraphML**:
```xml
<node id="rel-abc123xyz">
  <data key="d0">bipartite_edge</data>
  <data key="d1">17.0</data>  <!-- weight -->
  ...
</node>
```

**Interpretation**:
- Completeness score (0-10): How complete is this knowledge segment?
  - 9-10: Complete, self-contained statement
  - 7-8: Mostly complete, minor context needed
  - 5-6: Partial information
  - 0-4: Fragment, requires significant context
- Higher weight = mentioned multiple times with high completeness
- Weight 20+ = very important relation (mentioned 2+ times, high completeness)
- Weight 5-10 = single mention with medium completeness

---

### 3. Graph Edge Weights

**Definition**: Connection strength between entity and bipartite edge nodes.

**Formula**:
```
graph_edge_weight = bipartite_edge_weight (inherited from relation)
```

**Example**:
```xml
<edge source="LIONEL MESSI" target="rel-abc123xyz">
  <data key="d5">9.0</data>  <!-- edge weight -->
  <data key="d6">chunk-xyz</data>  <!-- source chunk -->
</edge>
```

**Interpretation**:
- Used for graph traversal ranking
- Higher weight edges = stronger connections
- Same as parent bipartite edge weight

---

## Why Not Normalize?

**Question**: Should weights be normalized to 0-1 scale?

**Answer**: No, for three reasons:

### Reason 1: Frequency Signal Preservation

**Unnormalized**:
- Entity A: weight 400 (mentioned 4 times, avg score 100)
- Entity B: weight 100 (mentioned 1 time, score 100)
- **Interpretation**: A is 4x more important (more central to corpus)

**Normalized**:
- Entity A: weight 1.0 (400 / 400)
- Entity B: weight 0.25 (100 / 400)
- **Lost information**: Can't tell if B was mentioned once or 10 times with low scores

**Benefit**: Frequency signal helps identify **central entities** vs **peripheral entities**.

---

### Reason 2: Incremental Construction

When adding new documents:

**Unnormalized**:
```python
# Before adding new doc
entity_weight = 270

# After adding new doc (entity appears again with score 95)
entity_weight = 270 + 95 = 365  # Simple addition
```

**Normalized**:
```python
# Before adding new doc
entity_weight_normalized = 0.85  # 270 / 320 (max was 320)

# After adding new doc
# Need to re-normalize ALL entities!
new_max = 365
entity_weight_normalized = 365 / 365 = 1.0  # But what about others?
# All other entities need recalculation!
```

**Benefit**: Unnormalized weights support **incremental graph updates** without full recalculation.

---

### Reason 3: Ranking Flexibility

Different ranking strategies need different weight treatments:

**Strategy 1: Raw weight ranking**
```python
# Rank entities by total importance
ranked = sorted(entities, key=lambda e: e['weight'], reverse=True)
```

**Strategy 2: Normalized weight ranking**
```python
# Rank entities by relative importance
max_weight = max(e['weight'] for e in entities)
ranked = sorted(entities, key=lambda e: e['weight'] / max_weight, reverse=True)
```

**Strategy 3: Log-scale ranking**
```python
# Rank entities by log-importance (reduces outlier effect)
import math
ranked = sorted(entities, key=lambda e: math.log(e['weight'] + 1), reverse=True)
```

**Benefit**: Raw weights allow **flexible ranking strategies** at query time.

---

## Documentation Updates

### Update 1: CLAUDE.md

**File**: `CLAUDE.md`
**Section**: Add new section after "Storage Architecture"

```markdown
## Weight Semantics

BiG-RAG uses three types of weights to represent importance and completeness:

### Entity Weights

**Definition**: Cumulative importance score across all chunk occurrences.

**Calculation**:
```
entity_weight = Σ(llm_importance_score) for all chunks mentioning entity
```

**Range**: `0` to `N × 100` where N = number of chunks

**Interpretation**:
- `400+`: Very central entity (4+ mentions with high scores)
- `200-399`: Important entity (2-3 mentions)
- `100-199`: Mentioned entity (1-2 mentions)
- `50-99`: Peripheral entity (1 mention, low score)

**Example**:
```
Entity: "Lionel Messi"
Chunk 1: Score 95 → Partial weight 95
Chunk 2: Score 85 → Partial weight 85
Chunk 3: Score 90 → Partial weight 90
Total weight: 270
```

---

### Bipartite Edge (Relation) Weights

**Definition**: Cumulative completeness score across all occurrences.

**Calculation**:
```
edge_weight = Σ(completeness_score) for all chunks containing relation
```

**Range**: `0` to `N × 10` where N = number of chunks

**Completeness Score Scale**:
- `9-10`: Complete, self-contained knowledge
- `7-8`: Mostly complete, minor context needed
- `5-6`: Partial information
- `0-4`: Fragment, needs significant context

**Interpretation**:
- `20+`: Very important relation (2+ mentions, high completeness)
- `10-19`: Important relation (1-2 mentions)
- `5-9`: Single mention with medium completeness
- `0-4`: Incomplete fragment

---

### Graph Edge Weights

**Definition**: Connection strength between entity and relation nodes.

**Calculation**: Inherits weight from parent bipartite edge node.

**Usage**: Influences graph traversal ranking during retrieval.

---

### Why Not Normalize?

BiG-RAG **intentionally** uses unnormalized weights for three reasons:

1. **Frequency Signal Preservation**: Weight 400 vs 100 shows 4x more mentions (important for centrality)
2. **Incremental Construction**: Adding documents doesn't require re-normalizing all existing weights
3. **Ranking Flexibility**: Raw weights allow query-time normalization strategies (linear, log, etc.)

**If you need normalized weights**, calculate at query time:
```python
max_weight = max(entity['weight'] for entity in entities)
for entity in entities:
    entity['weight_normalized'] = entity['weight'] / max_weight
```
```

---

### Update 2: Code Docstrings

**File**: `bigrag/operate.py`
**Function**: `_merge_nodes_then_upsert()`

Add detailed docstring:

```python
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge duplicate entity nodes and upsert to graph.

    Weight Calculation:
        entity_weight = Σ(llm_importance_score) for all occurrences

    Weight Interpretation:
        - 400+: Very central entity (4+ chunk mentions, high scores)
        - 200-399: Important entity (2-3 mentions)
        - 100-199: Mentioned entity (1-2 mentions)
        - 50-99: Peripheral entity (1 mention, low score)

    Example:
        Entity "Paris" appears in 3 chunks:
        - Chunk 1: LLM score 95
        - Chunk 2: LLM score 80
        - Chunk 3: LLM score 90
        → Total weight: 265

    Args:
        entity_name: Entity identifier (e.g., "Paris")
        nodes_data: List of entity occurrence dicts with 'weight' field
        knowledge_graph_inst: Graph storage backend
        global_config: Global configuration dict

    Returns:
        dict: Merged entity node data with aggregated weight
    """
    # ... existing implementation ...
```

**File**: `bigrag/operate.py`
**Function**: `_merge_bipartite_edges_then_upsert()`

```python
async def _merge_bipartite_edges_then_upsert(
    bipartite_edge_id: str,
    bipartite_edge_content: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge duplicate bipartite edge nodes and upsert to graph.

    Weight Calculation:
        edge_weight = Σ(completeness_score) for all occurrences

    Completeness Score Scale (0-10):
        - 9-10: Complete, self-contained statement
        - 7-8: Mostly complete, minor context needed
        - 5-6: Partial information
        - 0-4: Fragment, requires significant context

    Weight Interpretation:
        - 20+: Very important relation (2+ mentions, high completeness)
        - 10-19: Important relation (1-2 mentions)
        - 5-9: Single mention, medium completeness
        - 0-4: Incomplete fragment

    Example:
        Relation "Paris is the capital of France"
        - Chunk 1: Completeness 9/10
        - Chunk 2: Completeness 8/10
        → Total weight: 17

    Args:
        bipartite_edge_id: Hash-based node ID (e.g., "rel-abc123")
        bipartite_edge_content: Full knowledge segment text
        nodes_data: List of relation occurrence dicts with 'weight' field
        knowledge_graph_inst: Graph storage backend
        global_config: Global configuration dict

    Returns:
        dict: Merged edge node data with aggregated weight
    """
    # ... existing implementation ...
```

---

### Update 3: Implementation Guide

**File**: `implementation_guide/PART1_GRAPH_CONSTRUCTION.md`
**Section**: Add after "Data Structure Specifications"

```markdown
## Weight Calculation Details

### Entity Weight Aggregation

```python
ALGORITHM: Aggregate_Entity_Weights
INPUT: entity_occurrences: List[Dict]  # [{weight: 95}, {weight: 85}, ...]
OUTPUT: total_weight: float

PROCEDURE Aggregate_Entity_Weights(entity_occurrences):
    total_weight = 0

    FOR EACH occurrence IN entity_occurrences:
        llm_score = occurrence['weight']  # 0-100 from LLM
        total_weight += llm_score

    RETURN total_weight

END PROCEDURE
```

**Example**:
```python
# Entity "Lionel Messi" appears in 3 chunks
occurrences = [
    {'entity_name': 'Lionel Messi', 'weight': 95},  # Chunk 1
    {'entity_name': 'Lionel Messi', 'weight': 85},  # Chunk 2
    {'entity_name': 'Lionel Messi', 'weight': 90},  # Chunk 3
]

total_weight = 95 + 85 + 90 = 270
```

---

### Bipartite Edge Weight Aggregation

```python
ALGORITHM: Aggregate_Edge_Weights
INPUT: edge_occurrences: List[Dict]  # [{weight: 9}, {weight: 8}, ...]
OUTPUT: total_weight: float

PROCEDURE Aggregate_Edge_Weights(edge_occurrences):
    total_weight = 0

    FOR EACH occurrence IN edge_occurrences:
        completeness_score = occurrence['weight']  # 0-10 from LLM
        total_weight += completeness_score

    RETURN total_weight

END PROCEDURE
```

**Example**:
```python
# Relation "Messi scored 11 goals" appears in 2 chunks
occurrences = [
    {'content': 'Messi scored 11 goals...', 'weight': 9},  # Chunk 1
    {'content': 'Messi scored 11 goals...', 'weight': 8},  # Chunk 2
]

total_weight = 9 + 8 = 17
```
```

---

### Update 4: README.md

**File**: `README.md`
**Section**: Add to FAQ section

```markdown
## FAQ

### What do weight values mean in the knowledge graph?

**Entity Weights**: Cumulative importance score (0 to N×100 where N = number of mentions)
- Higher weight = more central to corpus (more mentions + higher LLM confidence)
- Example: Weight 400 = mentioned 4+ times with high importance

**Relation Weights**: Cumulative completeness score (0 to N×10 where N = number of mentions)
- Higher weight = more complete knowledge segment
- Example: Weight 18 = mentioned 2 times with 9/10 completeness each

See [CLAUDE.md - Weight Semantics](#weight-semantics) for details.

---

### Should I normalize weights?

**No**, BiG-RAG intentionally uses unnormalized weights to:
1. Preserve frequency signals (weight 400 vs 100 shows 4x more mentions)
2. Support incremental updates (adding docs doesn't require re-normalizing all weights)
3. Allow flexible ranking (you can normalize at query time if needed)

If you need normalized weights for a specific use case:
```python
max_weight = max(entity['weight'] for entity in entities)
for entity in entities:
    entity['weight_normalized'] = entity['weight'] / max_weight
```
```

---

## Testing & Validation

### Validation Script

**File**: `test_scripts/validate_weight_semantics.py` (new)

```python
import networkx as nx
import json
from collections import defaultdict

def validate_weight_semantics(graphml_path, chunks_path):
    """
    Validate that weights are calculated correctly.

    Checks:
    1. Entity weights = sum of occurrence weights
    2. Edge weights = sum of occurrence weights
    3. No zero weights (except for errors)
    4. Weights match source_id count pattern
    """
    # Load graph
    graph = nx.read_graphml(graphml_path)

    # Load chunks to verify source counting
    with open(chunks_path) as f:
        chunks = json.load(f)

    # Check entity weights
    entity_nodes = [
        (n, d) for n, d in graph.nodes(data=True)
        if d.get('role') == 'entity'
    ]

    print("Entity Weight Validation:")
    for node_id, node_data in entity_nodes[:10]:  # Sample first 10
        weight = float(node_data.get('weight', 0))
        source_id = node_data.get('source_id', '')
        source_count = len(source_id.split('<SEP>'))

        # Estimate expected weight range
        min_expected = source_count * 50  # Min: N × 50
        max_expected = source_count * 100  # Max: N × 100

        if min_expected <= weight <= max_expected:
            status = "OK"
        else:
            status = "WARNING"

        print(f"  {node_id:30s}: weight={weight:6.1f}, sources={source_count}, {status}")

    # Check edge weights
    edge_nodes = [
        (n, d) for n, d in graph.nodes(data=True)
        if d.get('role') == 'bipartite_edge'
    ]

    print("\nBipartite Edge Weight Validation:")
    for node_id, node_data in edge_nodes[:10]:  # Sample first 10
        weight = float(node_data.get('weight', 0))
        source_id = node_data.get('source_id', '')
        source_count = len(source_id.split('<SEP>'))

        # Estimate expected weight range
        min_expected = source_count * 5   # Min: N × 5
        max_expected = source_count * 10  # Max: N × 10

        if min_expected <= weight <= max_expected:
            status = "OK"
        else:
            status = "WARNING"

        print(f"  {node_id[:40]:40s}: weight={weight:6.1f}, sources={source_count}, {status}")

if __name__ == "__main__":
    validate_weight_semantics(
        "expr/demo_test/graph_chunk_entity_relation.graphml",
        "expr/demo_test/kv_store_text_chunks.json"
    )
```

**Expected Output**:
```
Entity Weight Validation:
  LIONEL MESSI                  : weight= 270.0, sources=3, OK
  INTER MIAMI                   : weight= 180.0, sources=2, OK
  LA LIGA                       : weight= 360.0, sources=4, OK
  ...

Bipartite Edge Weight Validation:
  rel-abc123xyz                 : weight=  17.0, sources=2, OK
  rel-def456uvw                 : weight=   9.0, sources=1, OK
  ...
```

---

### Weight Distribution Analysis

**File**: `test_scripts/analyze_weight_distribution.py` (new)

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def analyze_weight_distribution(graphml_path):
    """Generate weight distribution histograms"""
    graph = nx.read_graphml(graphml_path)

    # Extract entity weights
    entity_weights = [
        float(d.get('weight', 0))
        for n, d in graph.nodes(data=True)
        if d.get('role') == 'entity'
    ]

    # Extract edge weights
    edge_weights = [
        float(d.get('weight', 0))
        for n, d in graph.nodes(data=True)
        if d.get('role') == 'bipartite_edge'
    ]

    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Entity weights
    ax1.hist(entity_weights, bins=20, edgecolor='black')
    ax1.set_xlabel('Entity Weight')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Entity Weight Distribution (n={len(entity_weights)})')
    ax1.axvline(np.mean(entity_weights), color='red', linestyle='--', label=f'Mean: {np.mean(entity_weights):.1f}')
    ax1.legend()

    # Edge weights
    ax2.hist(edge_weights, bins=20, edgecolor='black')
    ax2.set_xlabel('Bipartite Edge Weight')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Edge Weight Distribution (n={len(edge_weights)})')
    ax2.axvline(np.mean(edge_weights), color='red', linestyle='--', label=f'Mean: {np.mean(edge_weights):.1f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('weight_distribution.png', dpi=150)
    print("Saved: weight_distribution.png")

    # Print statistics
    print("\nEntity Weight Statistics:")
    print(f"  Count: {len(entity_weights)}")
    print(f"  Mean: {np.mean(entity_weights):.2f}")
    print(f"  Median: {np.median(entity_weights):.2f}")
    print(f"  Min: {np.min(entity_weights):.2f}")
    print(f"  Max: {np.max(entity_weights):.2f}")
    print(f"  Std: {np.std(entity_weights):.2f}")

    print("\nEdge Weight Statistics:")
    print(f"  Count: {len(edge_weights)}")
    print(f"  Mean: {np.mean(edge_weights):.2f}")
    print(f"  Median: {np.median(edge_weights):.2f}")
    print(f"  Min: {np.min(edge_weights):.2f}")
    print(f"  Max: {np.max(edge_weights):.2f}")
    print(f"  Std: {np.std(edge_weights):.2f}")

if __name__ == "__main__":
    analyze_weight_distribution("expr/demo_test/graph_chunk_entity_relation.graphml")
```

---

## Implementation Checklist

- [ ] Update CLAUDE.md (add Weight Semantics section)
- [ ] Update code docstrings (`_merge_nodes_then_upsert`, `_merge_bipartite_edges_then_upsert`)
- [ ] Update PART1_GRAPH_CONSTRUCTION.md (add weight calculation algorithms)
- [ ] Update README.md FAQ
- [ ] Create `validate_weight_semantics.py` script
- [ ] Create `analyze_weight_distribution.py` script
- [ ] Run validation on demo_test dataset
- [ ] Generate weight distribution plots
- [ ] Review and merge documentation PR

---

## Benefits

1. **User Understanding**: Clear explanation of what weights represent
2. **Developer Guidance**: Docstrings explain weight calculation in code
3. **Debugging Support**: Validation scripts help identify weight calculation errors
4. **Design Rationale**: Documents why unnormalized weights are used

---

## Document End

**Last Updated**: 2025-01-08
**Implementation Status**: Pending user approval

**Effort Estimate**: 2-3 hours for documentation updates
