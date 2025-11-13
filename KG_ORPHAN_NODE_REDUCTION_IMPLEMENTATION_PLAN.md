# BiG-RAG Knowledge Graph Orphan Node Reduction - Implementation Plan

**Version**: 1.1
**Date**: 2025-01-13
**Last Updated**: 2025-01-13 (Added Step 1.5: Metadata Formatting Enhancement)
**Status**: Ready for Implementation
**Estimated Time**: 2-3 days
**Target**: Reduce orphan relation rate from 22.5% to <5%

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Analysis](#problem-analysis)
3. [Solution Architecture](#solution-architecture)
4. [Implementation Guide](#implementation-guide)
5. [Testing Strategy](#testing-strategy)
6. [Rollback Plan](#rollback-plan)
7. [Success Criteria](#success-criteria)
8. [Troubleshooting](#troubleshooting)

---

## Executive Summary

### Current State (Football Dataset)
- **Total nodes**: 177 (97 entities + 80 relations)
- **Orphan relations**: 18/80 (22.5%) ← **CRITICAL ISSUE**
- **Edges**: 108
- **Average connectivity**: 1.35 edges/relation

### Root Cause
LLM outputs consecutive `("relation"...)` records without extracting `("entity"...)` records, creating orphan relation nodes that are unreachable during retrieval.

### Solution Summary
**6 Critical Improvements** inspired by LightRAG analysis:
1. **Metadata formatting enhancement** (bracket-style formatting for clearer boundaries)
2. **Enforce entity extraction after each relation** (60% orphan reduction)
3. **Enhanced input sanitization** (prevent parsing errors)
4. **Smart gleaning merge** (improve entity quality)
5. **Delimiter corruption handling** (recover from LLM format errors)
6. **Post-extraction validation** (early orphan detection)

### Target State (After Implementation)
- **Orphan relations**: <4/80 (<5%)
- **Edges**: 280-320
- **Average connectivity**: 3.0-3.5 edges/relation
- **Retrieval recall**: +30-40%
- **Retrieval precision**: +15-25%

---

## Problem Analysis

### Evidence from Football Dataset

**Orphan Relation Example 1:**
```xml
<node id="rel-79a06bae5060c9ecba31015810f133a4">
  <data key="d0">relation</data>
  <data key="d1">"Lionel Messi, widely regarded as one of the greatest football
                  players of all time, continues to make headlines in Major
                  League Soccer with Inter Miami."</data>
  <data key="d2">19.0</data>  <!-- High weight! -->
</node>
```
**Expected entities**: LIONEL MESSI, INTER MIAMI, MAJOR LEAGUE SOCCER
**Actual entities extracted**: **NONE** (orphan!)
**Impact**: Rich knowledge unreachable during retrieval

**Orphan Relation Example 2:**
```xml
<node id="rel-6ff9c06b77fc30f64384c2db61816bc2">
  <data key="d1">"The Argentine superstar joined the club in July 2023
                  and has transformed the team's fortunes dramatically."</data>
</node>
```
**Expected entities**: LIONEL MESSI (inferred), INTER MIAMI, JULY 2023
**Actual entities extracted**: **NONE**

### LLM Output Pattern (What's Actually Happening)

**Incorrect Sequence (Current):**
```python
("relation", "Messi plays for Inter Miami in MLS", 9)
("relation", "Argentina won the 2022 World Cup", 9)  # ← No entities extracted!
("relation", "Haaland broke Premier League record", 8)  # ← Still no entities!
```

**Correct Sequence (Target):**
```python
("relation", "Messi plays for Inter Miami in MLS", 9)
("entity", "LIONEL MESSI", "person", "Professional footballer", 95)  # ← MUST extract
("entity", "INTER MIAMI", "organization", "MLS club", 85)
("entity", "MAJOR LEAGUE SOCCER", "organization", "Soccer league", 80)
("relation", "Argentina won the 2022 World Cup", 9)  # ← Only after entities
("entity", "ARGENTINA", "organization", "National team", 90)
("entity", "2022 FIFA WORLD CUP", "event", "Football tournament", 85)
```

### Why LightRAG Doesn't Have This Problem

**LightRAG Architecture:**
- Traditional graph: Entity → Entity edges
- Prompt enforces: "Output all entities first, then all relationships"
- Relations are edges (not nodes), so orphan relations = missing edge (still has entity nodes)

**BiG-RAG Architecture:**
- Bipartite graph: Relation nodes ↔ Entity nodes
- Relations are first-class nodes (retrievable independently)
- **Orphan relation node = completely unreachable knowledge**

**Critical Difference:**
```
LightRAG: Missing edge → Entities still retrievable (degraded)
BiG-RAG: Orphan relation node → Knowledge completely lost (critical)
```

---

## Solution Architecture

### Phase 1: Prompt Improvements (Week 1)

**Improvement 1: Enforce Sequential Extraction** ⭐⭐⭐⭐⭐
- **File**: `bigrag/prompt.py`
- **Impact**: 60% orphan reduction
- **Approach**: Add explicit sequencing rules to LLM prompt

**Improvement 2: Enhanced Sanitization** ⭐⭐⭐⭐
- **File**: `bigrag/utils.py` (new functions), `bigrag/operate.py` (use functions)
- **Impact**: Prevent parsing errors from malformed LLM output
- **Approach**: Field-specific text cleaning with validation

**Improvement 3: Delimiter Corruption Handling** ⭐⭐⭐
- **File**: `bigrag/utils.py`
- **Impact**: Recover from LLM format errors (`<>` instead of `<|>`)
- **Approach**: Pattern matching and correction

### Phase 2: Quality Improvements (Week 2)

**Improvement 4: Smart Gleaning Merge** ⭐⭐⭐⭐
- **File**: `bigrag/operate.py`
- **Impact**: Keep better entity descriptions
- **Approach**: Compare description quality, keep best version

**Improvement 5: Post-Extraction Validation** ⭐⭐⭐
- **File**: `bigrag/operate.py`
- **Impact**: Early detection of orphan relations
- **Approach**: Validate entity-relation links before storage

### Phase 3: Testing & Verification (Week 2-3)

**Test Script: Orphan Detection** ⭐⭐⭐⭐⭐
- **File**: `test_scripts/test_orphan_detection.py` (new)
- **Impact**: Measure improvement
- **Approach**: Parse GraphML, count orphans, compare before/after

---

## Implementation Guide

### Pre-Implementation Checklist

**Before starting, complete these steps:**

1. **Backup current code:**
   ```bash
   cd d:/BiG-RAG
   git add .
   git commit -m "Backup before orphan reduction implementation"
   git branch backup-pre-orphan-fix
   ```

2. **Establish baseline:**
   ```bash
   # Create baseline test script (see Section 5.1)
   python test_scripts/test_orphan_detection.py football

   # Record results:
   # Baseline: 18 orphan relations out of 80 (22.5%)
   ```

3. **Verify LLM context window:**
   ```bash
   # Check your LLM model
   # GPT-4o-mini: 128K tokens ✅
   # GPT-3.5-turbo: 16K tokens ⚠️ (prompt may be too long)
   ```

4. **Create working branch:**
   ```bash
   git checkout -b feature/orphan-node-reduction
   ```

---

### Implementation Step 1: Baseline Test Script

**Create file:** `test_scripts/test_orphan_detection.py`

```python
"""
BiG-RAG Knowledge Graph Orphan Node Detection Test

Analyzes a GraphML file to detect orphan nodes (nodes with no edges).
Provides before/after comparison for improvement tracking.

Usage:
    python test_orphan_detection.py football
    python test_orphan_detection.py football --compare expr/football_backup/graph_chunk_entity_relation.graphml
"""

import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
from typing import Dict, List, Tuple


def parse_graphml_structure(graph_path: Path) -> Dict:
    """
    Parse GraphML file and extract node/edge statistics.

    Returns:
        dict: {
            "entity_nodes": list[str],
            "relation_nodes": list[str],
            "edges": list[tuple],
            "orphan_entities": list[str],
            "orphan_relations": list[str],
            "orphan_relation_details": list[dict]
        }
    """
    tree = ET.parse(graph_path)
    root = tree.getroot()
    ns = {'gml': 'http://graphml.graphdrawing.org/xmlns'}

    # Collect nodes by type
    entity_nodes = []
    relation_nodes = []
    relation_node_details = {}  # id -> {content, weight}

    for node in root.findall('.//gml:node', ns):
        node_id = node.get('id')
        role = None
        content = None
        weight = None

        for data in node.findall('gml:data', ns):
            key = data.get('key')
            if key == 'd0':  # role
                role = data.text
            elif key == 'd1':  # content (for relations)
                content = data.text
            elif key == 'd2':  # weight
                try:
                    weight = float(data.text)
                except (ValueError, TypeError):
                    weight = 0.0

        if role == 'entity':
            entity_nodes.append(node_id)
        elif role == 'relation':
            relation_nodes.append(node_id)
            relation_node_details[node_id] = {
                'content': content or '',
                'weight': weight or 0.0
            }

    # Collect edges
    edges = []
    connected_nodes = set()
    for edge in root.findall('.//gml:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        edges.append((source, target))
        connected_nodes.add(source)
        connected_nodes.add(target)

    # Find orphans
    all_nodes = set(entity_nodes + relation_nodes)
    orphan_nodes = all_nodes - connected_nodes

    orphan_entities = [n for n in orphan_nodes if n in entity_nodes]
    orphan_relations = [n for n in orphan_nodes if n in relation_nodes]

    # Collect orphan relation details
    orphan_relation_details = []
    for orphan_id in orphan_relations:
        if orphan_id in relation_node_details:
            orphan_relation_details.append({
                'id': orphan_id,
                'content': relation_node_details[orphan_id]['content'],
                'weight': relation_node_details[orphan_id]['weight']
            })

    return {
        'entity_nodes': entity_nodes,
        'relation_nodes': relation_nodes,
        'edges': edges,
        'orphan_entities': orphan_entities,
        'orphan_relations': orphan_relations,
        'orphan_relation_details': orphan_relation_details,
        'all_nodes': len(all_nodes)
    }


def print_orphan_report(stats: Dict, dataset_name: str, label: str = "CURRENT"):
    """Print formatted orphan analysis report"""

    total_nodes = stats['all_nodes']
    entity_count = len(stats['entity_nodes'])
    relation_count = len(stats['relation_nodes'])
    edge_count = len(stats['edges'])

    orphan_count = len(stats['orphan_entities']) + len(stats['orphan_relations'])
    orphan_entity_count = len(stats['orphan_entities'])
    orphan_relation_count = len(stats['orphan_relations'])

    orphan_rate = orphan_count / total_nodes if total_nodes > 0 else 0
    orphan_relation_rate = orphan_relation_count / relation_count if relation_count > 0 else 0

    avg_degree = (edge_count * 2) / total_nodes if total_nodes > 0 else 0
    avg_edges_per_relation = edge_count / relation_count if relation_count > 0 else 0

    print(f"\n{'='*80}")
    print(f"ORPHAN NODE ANALYSIS: {dataset_name} ({label})")
    print(f"{'='*80}")
    print(f"\n📊 Graph Statistics:")
    print(f"   Total Nodes:        {total_nodes}")
    print(f"   ├─ Entity Nodes:    {entity_count}")
    print(f"   └─ Relation Nodes:  {relation_count}")
    print(f"   Total Edges:        {edge_count}")
    print(f"   Avg Degree:         {avg_degree:.2f}")
    print(f"   Avg Edges/Relation: {avg_edges_per_relation:.2f}")

    print(f"\n🔍 Orphan Analysis:")
    print(f"   Total Orphans:      {orphan_count} ({orphan_rate:.1%})")
    print(f"   ├─ Orphan Entities: {orphan_entity_count}")
    print(f"   └─ Orphan Relations: {orphan_relation_count} ({orphan_relation_rate:.1%})")

    # Quality assessment
    print(f"\n📈 Quality Assessment:")
    if orphan_relation_rate < 0.05:
        status = "✅ EXCELLENT (Production Ready)"
    elif orphan_relation_rate < 0.10:
        status = "⚠️  GOOD (Minor Issues)"
    elif orphan_relation_rate < 0.20:
        status = "⚠️  NEEDS IMPROVEMENT"
    else:
        status = "❌ CRITICAL ISSUES"

    print(f"   Orphan Relation Rate: {orphan_relation_rate:.1%}")
    print(f"   Status: {status}")
    print(f"   Target: <5% for production quality")

    # Show sample orphan relations
    if stats['orphan_relation_details']:
        print(f"\n{'='*80}")
        print(f"SAMPLE ORPHAN RELATIONS (First 5):")
        print(f"{'='*80}")

        for i, detail in enumerate(stats['orphan_relation_details'][:5], 1):
            content = detail['content']
            weight = detail['weight']
            # Truncate long content
            display_content = content[:100] + "..." if len(content) > 100 else content
            print(f"\n{i}. ID: {detail['id']}")
            print(f"   Weight: {weight}")
            print(f"   Content: {display_content}")


def print_comparison_report(before_stats: Dict, after_stats: Dict, dataset_name: str):
    """Print before/after comparison report"""

    before_orphan_rate = (
        len(before_stats['orphan_relations']) / len(before_stats['relation_nodes'])
        if before_stats['relation_nodes'] else 0
    )
    after_orphan_rate = (
        len(after_stats['orphan_relations']) / len(after_stats['relation_nodes'])
        if after_stats['relation_nodes'] else 0
    )

    improvement = before_orphan_rate - after_orphan_rate
    improvement_pct = (improvement / before_orphan_rate * 100) if before_orphan_rate > 0 else 0

    before_avg_edges = (
        len(before_stats['edges']) / len(before_stats['relation_nodes'])
        if before_stats['relation_nodes'] else 0
    )
    after_avg_edges = (
        len(after_stats['edges']) / len(after_stats['relation_nodes'])
        if after_stats['relation_nodes'] else 0
    )

    print(f"\n{'='*80}")
    print(f"BEFORE/AFTER COMPARISON: {dataset_name}")
    print(f"{'='*80}")

    print(f"\n📊 Orphan Relation Rate:")
    print(f"   Before: {before_orphan_rate:.1%} ({len(before_stats['orphan_relations'])}/{len(before_stats['relation_nodes'])})")
    print(f"   After:  {after_orphan_rate:.1%} ({len(after_stats['orphan_relations'])}/{len(after_stats['relation_nodes'])})")
    print(f"   Change: {improvement:+.1%} ({improvement_pct:+.0f}%)")

    print(f"\n📈 Graph Connectivity:")
    print(f"   Edges Before: {len(before_stats['edges'])}")
    print(f"   Edges After:  {len(after_stats['edges'])}")
    print(f"   Change: {len(after_stats['edges']) - len(before_stats['edges']):+d}")

    print(f"\n🔗 Avg Edges per Relation:")
    print(f"   Before: {before_avg_edges:.2f}")
    print(f"   After:  {after_avg_edges:.2f}")
    print(f"   Change: {after_avg_edges - before_avg_edges:+.2f}")

    # Overall verdict
    print(f"\n{'='*80}")
    print(f"OVERALL VERDICT:")
    print(f"{'='*80}")

    if improvement > 0.15:
        verdict = "✅ EXCELLENT IMPROVEMENT (>15% reduction)"
    elif improvement > 0.10:
        verdict = "✅ GOOD IMPROVEMENT (10-15% reduction)"
    elif improvement > 0.05:
        verdict = "⚠️  MODERATE IMPROVEMENT (5-10% reduction)"
    elif improvement > 0:
        verdict = "⚠️  MINOR IMPROVEMENT (<5% reduction)"
    else:
        verdict = "❌ NO IMPROVEMENT or REGRESSION"

    print(f"   {verdict}")

    if after_orphan_rate < 0.05:
        print(f"   ✅ Target achieved: Orphan rate now <5%")
    else:
        print(f"   ⚠️  Target not yet achieved: Orphan rate still {after_orphan_rate:.1%} (target: <5%)")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python test_orphan_detection.py <dataset_name> [--compare <before_graphml_path>]")
        print("Example: python test_orphan_detection.py football")
        print("Example: python test_orphan_detection.py football --compare expr/football_backup/graph_chunk_entity_relation.graphml")
        sys.exit(1)

    dataset_name = sys.argv[1]
    current_graph_path = Path(f"expr/{dataset_name}/graph_chunk_entity_relation.graphml")

    if not current_graph_path.exists():
        print(f"ERROR: Graph not found at {current_graph_path}")
        sys.exit(1)

    # Parse current graph
    print(f"Parsing current graph: {current_graph_path}")
    current_stats = parse_graphml_structure(current_graph_path)

    # Check for comparison mode
    if len(sys.argv) >= 4 and sys.argv[2] == '--compare':
        before_graph_path = Path(sys.argv[3])
        if not before_graph_path.exists():
            print(f"ERROR: Before graph not found at {before_graph_path}")
            sys.exit(1)

        print(f"Parsing before graph: {before_graph_path}")
        before_stats = parse_graphml_structure(before_graph_path)

        # Print before report
        print_orphan_report(before_stats, dataset_name, label="BEFORE")

        # Print after report
        print_orphan_report(current_stats, dataset_name, label="AFTER")

        # Print comparison
        print_comparison_report(before_stats, current_stats, dataset_name)
    else:
        # Print single report
        print_orphan_report(current_stats, dataset_name, label="CURRENT")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
```

**Test the baseline script:**
```bash
cd d:/BiG-RAG
python test_scripts/test_orphan_detection.py football

# Expected output:
# ================================================================================
# ORPHAN NODE ANALYSIS: football (CURRENT)
# ================================================================================
#
# 📊 Graph Statistics:
#    Total Nodes:        177
#    ├─ Entity Nodes:    97
#    └─ Relation Nodes:  80
#    Total Edges:        108
#    Avg Degree:         1.22
#    Avg Edges/Relation: 1.35
#
# 🔍 Orphan Analysis:
#    Total Orphans:      18 (10.2%)
#    ├─ Orphan Entities: 0
#    └─ Orphan Relations: 18 (22.5%)
#
# 📈 Quality Assessment:
#    Orphan Relation Rate: 22.5%
#    Status: ❌ CRITICAL ISSUES
#    Target: <5% for production quality
```

**Save baseline for comparison:**
```bash
# Backup current graph
mkdir -p expr/football_backup
cp expr/football/graph_chunk_entity_relation.graphml expr/football_backup/

# Record baseline metrics
echo "Baseline (2025-01-13): 18 orphan relations out of 80 (22.5%)" > test_scripts/baseline_football.txt
```

---

### Implementation Step 1.5: Metadata Formatting Enhancement

**File to modify:** `bigrag/operate.py`

**Location:** Lines 567-588 (metadata context enrichment section)

**Purpose:** Adopt bracket-style formatting (like LightRAG) to provide clearer semantic boundaries between metadata and content, preventing LLM from confusing metadata with extractable entities.

**Current Implementation Analysis:**

BiG-RAG already uses metadata during extraction (added in Phase 2.1):
```python
# Current format (lines 567-588):
doc_title = chunk_dp.get("doc_title", "")
doc_metadata = chunk_dp.get("doc_metadata", {})

context_parts = []
if doc_title:
    context_parts.append(f"Document Title: {doc_title}")
if doc_metadata:
    metadata_str = ", ".join(
        f"{k}: {v}" for k, v in doc_metadata.items()
        if k != "title" and v
    )
    if metadata_str:
        context_parts.append(f"Document Context: {metadata_str}")

# Combine context with content
if context_parts:
    enriched_content = "\n".join(context_parts) + "\n\n" + content
else:
    enriched_content = content
```

**Current Output Format:**
```
Document Title: football and footballer related news
Document Context: category: organisation, tags: epl, laliga

[chunk content here...]
```

**Issue with Current Format:**
- Plain text format lacks clear boundaries
- LLM might extract "Document Title" as an entity
- No explicit instruction that metadata is contextual, not extractable

**LightRAG's Approach (for comparison):**
```python
# LightRAG uses brackets (lightrag/operate.py:2066-2087):
content_parts = []
if "doc_summary" in chunk_dp:
    content_parts.append(f"[Document Context: {chunk_dp['doc_summary']}]")
if "doc_metadata" in chunk_dp:
    metadata_str = ", ".join(f"{k}: {v}" for k, v in chunk_dp["doc_metadata"].items())
    content_parts.append(f"[Metadata: {metadata_str}]")
content_parts.append(chunk_dp["content"])
content = "\n".join(content_parts)
```

**Code Change:**

Replace lines 567-588 in `bigrag/operate.py`:

```python
# Extract metadata for context enhancement (Phase 2.1 improvement)
doc_title = chunk_dp.get("doc_title", "")
doc_metadata = chunk_dp.get("doc_metadata", {})

# Build context-enriched input text with bracket-style formatting
# (Phase 3.1: Adopted from LightRAG to prevent metadata confusion)
context_parts = []

if doc_title:
    context_parts.append(f"Title: {doc_title}")

if doc_metadata:
    metadata_str = ", ".join(
        f"{k}: {v}" for k, v in doc_metadata.items()
        if k != "title" and v  # Skip empty values and title (already included)
    )
    if metadata_str:
        context_parts.append(f"Metadata: {metadata_str}")

# Combine with bracket markers for clear semantic boundaries
if context_parts:
    metadata_block = "\n".join(context_parts)
    enriched_content = (
        f"[DOCUMENT CONTEXT]\n"
        f"{metadata_block}\n\n"
        f"[CHUNK CONTENT]\n"
        f"{content}"
    )
else:
    # No metadata available, use content as-is
    enriched_content = content
```

**New Output Format:**
```
[DOCUMENT CONTEXT]
Title: football and footballer related news
Metadata: category: organisation, tags: epl, laliga

[CHUNK CONTENT]
[chunk content here...]
```

**Benefits:**

1. **Clear Semantic Boundaries**: Brackets explicitly mark metadata vs content sections
2. **Prevent LLM Confusion**: LLM won't extract "Document Title" or "DOCUMENT CONTEXT" as entities
3. **Proven Pattern**: LightRAG uses this format successfully
4. **Low Risk**: Cosmetic change to string formatting (no logic changes)
5. **Aligns with Prompt**: Brackets match extraction prompt's delimiter style (`<|>`, `##`)

**Expected Impact:**

- **Orphan Reduction**: 5-10% reduction (combined with other improvements)
- **Entity Extraction Quality**: Fewer false entities from metadata
- **Consistency**: Better alignment between prompt instructions and input format

**Testing:**

After implementation, verify with football dataset:
```bash
# Rebuild graph with new formatting
python script_build.py --data_source football

# Check orphan count
python test_scripts/test_orphan_detection.py football

# Expected: Some reduction in orphan relations (target: 18 → 16-17)
```

**Estimated Time:** 15 minutes

**Risk Level:** Very Low (cosmetic change, no logic modification)

---

### Implementation Step 2: Enhanced Input Sanitization

**File to modify:** `bigrag/utils.py`

**Location:** Add after line ~200 (after existing utility functions)

```python
# ============================================================================
# Text Sanitization for LLM Extraction Output
# Added: 2025-01-13 for orphan node reduction
# ============================================================================

import re
from typing import Literal

def sanitize_extracted_text(
    text: str,
    field_type: Literal["entity_name", "entity_type", "description", "relation", "general"] = "general"
) -> str:
    """
    Sanitize LLM-extracted text with field-specific rules.

    Purpose: Clean malformed LLM output to prevent parsing errors and
             ensure consistent entity/relation data quality.

    Args:
        text: Raw text from LLM output
        field_type: Type of field being sanitized
            - "entity_name": Strict cleaning for entity identifiers
            - "entity_type": Very strict (lowercase, no spaces)
            - "description": Allow most characters, remove control chars
            - "relation": Relation content cleaning
            - "general": Basic cleaning

    Returns:
        Cleaned text string (may be empty if input is invalid)

    Examples:
        >>> sanitize_extracted_text('"  LIONEL MESSI  "', "entity_name")
        'LIONEL MESSI'

        >>> sanitize_extracted_text('  person  ', "entity_type")
        'person'

        >>> sanitize_extracted_text('" He is a player "', "description")
        'He is a player'
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Remove outer quotes and whitespace
    text = text.strip()

    # Remove outer quotes (single or double, but not inner quotes)
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()

    # Step 2: Normalize whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)

    # Step 3: Remove control characters (always)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # Step 4: Field-specific cleaning
    if field_type == "entity_name":
        # Entity names: Remove ALL inner quotes
        text = text.replace('"', '').replace("'", '')

        # Check for delimiter corruption (entity names should never contain delimiters)
        forbidden_patterns = ['<|>', '<SEP>', '##', '<GRAPH_FIELD_SEP>', '||', '<>', '|>']
        for pattern in forbidden_patterns:
            if pattern in text:
                logger.warning(f"Entity name contains reserved delimiter '{pattern}': '{text}'")
                return ""

        # Entity name cannot be empty after cleaning
        if not text.strip():
            return ""

    elif field_type == "entity_type":
        # Entity type must be single word, lowercase, no spaces
        text = text.replace(" ", "").lower()

        # Reject if contains invalid characters
        invalid_chars = ["'", '"', "(", ")", "<", ">", "|", "/", "\\", ",", ";", "!", "?"]
        if any(char in text for char in invalid_chars):
            logger.warning(f"Invalid entity type contains special characters: '{text}'")
            return ""

        # Type cannot be empty
        if not text.strip():
            return ""

    elif field_type == "description":
        # Descriptions: Allow most characters, just remove control chars (already done in Step 3)
        # Remove any remaining double control characters
        text = re.sub(r'\s+', ' ', text)

        # Trim to reasonable length if extremely long
        MAX_DESC_LENGTH = 2000
        if len(text) > MAX_DESC_LENGTH:
            logger.warning(f"Description truncated from {len(text)} to {MAX_DESC_LENGTH} chars")
            text = text[:MAX_DESC_LENGTH] + "..."

    elif field_type == "relation":
        # Relation content: Similar to description
        text = re.sub(r'\s+', ' ', text)

        # Trim if extremely long
        MAX_RELATION_LENGTH = 1000
        if len(text) > MAX_RELATION_LENGTH:
            logger.warning(f"Relation content truncated from {len(text)} to {MAX_RELATION_LENGTH} chars")
            text = text[:MAX_RELATION_LENGTH] + "..."

    # Final validation: Return empty string if only whitespace remains
    return text.strip()


def fix_delimiter_corruption(record: str, tuple_delimiter: str = "<|>") -> str:
    """
    Fix common LLM delimiter corruption patterns.

    Purpose: LLM sometimes outputs variations of the tuple delimiter
             (e.g., <> instead of <|>, || instead of <|>). This function
             corrects these patterns to enable proper parsing.

    Args:
        record: Raw LLM output record (single line)
        tuple_delimiter: Expected delimiter (default: "<|>" for BiG-RAG)

    Returns:
        Record with corrected delimiters

    Examples:
        >>> fix_delimiter_corruption('entity<>MESSI<>person', '<|>')
        'entity<|>MESSI<|>person'

        >>> fix_delimiter_corruption('relation||content||score', '<|>')
        'relation<|>content<|>score'

    Common Corruption Patterns:
        - <> instead of <|>
        - || instead of <|>
        - <| or |> (incomplete)
        - < | > (with spaces)
        - <#> instead of <|#|> (if delimiter has core character)
    """
    if not record:
        return record

    # Extract core delimiter character if present
    # For BiG-RAG: tuple_delimiter = "<|>" → core = "|"
    # For LightRAG: tuple_delimiter = "<|#|>" → core = "#"
    if len(tuple_delimiter) >= 3 and tuple_delimiter.startswith('<') and tuple_delimiter.endswith('>'):
        core = tuple_delimiter[2:-2] if len(tuple_delimiter) > 4 else tuple_delimiter[1:-1]
    else:
        core = "|"  # Default fallback

    # Define corruption patterns (ordered by likelihood)
    corrupted_patterns = [
        # Missing pipes
        f"<{core}>",      # <#> instead of <|#|>
        "<>",             # Empty brackets

        # Missing brackets
        f"|{core}|",      # |#| instead of <|#|>
        "||",             # Double pipes

        # Partial patterns
        f"<|{core}>",     # <|#> instead of <|#|>
        f"<{core}|>",     # <#|> instead of <|#|>
        "<|",             # Opening only
        "|>",             # Closing only

        # With spaces
        "< >",            # Brackets with space
        "| |",            # Pipes with space
        f"< {core} >",    # Brackets with spaces around core
        f"< | {core} | >", # Full pattern with spaces
        f"<| {core} |>",  # Spaces inside
    ]

    # Apply corrections
    for pattern in corrupted_patterns:
        if pattern in record:
            record = record.replace(pattern, tuple_delimiter)
            logger.debug(f"Fixed delimiter corruption: '{pattern}' → '{tuple_delimiter}'")

    return record


def description_quality_score(description: str) -> float:
    """
    Calculate quality score for entity/relation descriptions.

    Purpose: During gleaning merge, compare quality of descriptions
             to keep the better version (not just longer version).

    Args:
        description: Entity or relation description text

    Returns:
        Quality score (float, higher = better quality)

    Scoring Factors:
        - Base: Length (more detail assumed better)
        - +10: Ends with proper sentence (period)
        - -50%: Very short (<20 chars)
        - +20: Contains specific keywords (who, which, known for, etc.)
        - +10: Contains numbers/dates (specific facts)

    Examples:
        >>> description_quality_score("Messi is a player")
        17  # Short, no period, no keywords

        >>> description_quality_score("Lionel Messi is a professional footballer known for winning 8 Ballon d'Or awards.")
        132  # Long (92) + period (+10) + keywords (+20) + numbers (+10)
    """
    if not description:
        return 0.0

    score = len(description)  # Base score: length

    # Bonus: Complete sentence (ends with period)
    if description.rstrip().endswith('.'):
        score += 10

    # Penalty: Very short descriptions (likely incomplete)
    if len(description) < 20:
        score *= 0.5

    # Bonus: Contains specific keywords (indicates detailed description)
    quality_keywords = [
        'who', 'which', 'where', 'when', 'professional',
        'known for', 'famous for', 'specialist', 'expert',
        'won', 'achieved', 'played', 'founded', 'established'
    ]
    keyword_matches = sum(1 for kw in quality_keywords if kw.lower() in description.lower())
    score += keyword_matches * 20

    # Bonus: Contains numbers/dates (indicates specific facts)
    has_numbers = bool(re.search(r'\d+', description))
    if has_numbers:
        score += 10

    # Bonus: Mentions multiple entities (rich context)
    # Heuristic: Count capitalized words (potential entity mentions)
    capitalized_words = len([w for w in description.split() if w and w[0].isupper()])
    if capitalized_words >= 3:
        score += 15

    return score
```

**Update imports in `bigrag/operate.py`:**

```python
# File: bigrag/operate.py
# Location: Line ~10 (imports section)

# BEFORE (existing imports):
from .utils import (
    logger,
    compute_mdhash_id,
    clean_str,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    # ... other imports
)

# AFTER (add new imports):
from .utils import (
    logger,
    compute_mdhash_id,
    clean_str,
    sanitize_extracted_text,         # ← NEW
    fix_delimiter_corruption,        # ← NEW
    description_quality_score,       # ← NEW
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    # ... other imports
)
```

---

### Implementation Step 3: Update Extraction Handlers

**File to modify:** `bigrag/operate.py`

**Part A: Update `_handle_single_entity_extraction`**

**Location:** Line ~261 (function definition)

```python
# BEFORE (existing code):
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())


# AFTER (with validation and sanitization):
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    """
    Extract and validate a single entity from LLM output.

    Validation:
    - Exactly 5 fields (not <5, not >5)
    - Must have relation context (prevent orphan entities)
    - Sanitize all fields
    - Validate entity name/type/description

    Returns:
        dict or None: Entity data if valid, None if invalid
    """

    # Validate field count (EXACT, not >=)
    if len(record_attributes) != 5:
        if len(record_attributes) > 1 and '"entity"' in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: Entity has {len(record_attributes)}/5 fields "
                f"(expected exactly 5). Entity: {record_attributes[1] if len(record_attributes) > 1 else 'N/A'}"
            )
        return None

    # Validate first field is "entity"
    if record_attributes[0] != '"entity"':
        return None

    # Validate relation context exists (prevent orphan entities)
    if not now_hyper_relation or now_hyper_relation == "":
        logger.warning(
            f"{chunk_key}: Entity extracted without relation context. "
            f"This may indicate prompt sequencing issue. Entity: {record_attributes[1]}"
        )
        return None

    # Sanitize entity name
    entity_name_raw = record_attributes[1]
    entity_name = sanitize_extracted_text(entity_name_raw, "entity_name")

    if not entity_name:
        logger.warning(
            f"{chunk_key}: Entity name became empty after sanitization. "
            f"Raw: '{entity_name_raw}'"
        )
        return None

    # Apply BiG-RAG convention: UPPERCASE entity names
    entity_name = entity_name.upper()

    # Sanitize entity type
    entity_type_raw = record_attributes[2]
    entity_type = sanitize_extracted_text(entity_type_raw, "entity_type")

    if not entity_type:
        logger.warning(
            f"{chunk_key}: Entity type invalid for entity '{entity_name}'. "
            f"Raw: '{entity_type_raw}'"
        )
        return None

    # Normalize entity type (e.g., "PERSON" → "person", "Organization" → "organization")
    entity_type = normalize_entity_type(entity_type)

    # Sanitize description
    description_raw = record_attributes[3]
    description = sanitize_extracted_text(description_raw, "description")

    if not description:
        logger.warning(
            f"{chunk_key}: Description empty for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    # Parse weight (key score)
    try:
        weight = float(record_attributes[4])
        # Validate reasonable range
        if weight < 0 or weight > 100:
            logger.warning(
                f"{chunk_key}: Entity weight out of range (0-100): {weight} for '{entity_name}'. "
                f"Using clamped value."
            )
            weight = max(0, min(100, weight))
    except (ValueError, IndexError) as e:
        logger.warning(
            f"{chunk_key}: Invalid weight for entity '{entity_name}': {record_attributes[4]}. "
            f"Using default 50.0. Error: {e}"
        )
        weight = 50.0  # Default fallback

    # Return validated entity data
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=chunk_key,
        weight=weight,
        hyper_relation=now_hyper_relation,  # Link to parent relation
    )
```

**Part B: Update `_handle_single_hyperrelation_extraction`**

**Location:** Line ~291 (function definition)

```python
# BEFORE (existing code):
async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str
):
    if len(record_attributes) < 3 or record_attributes[0] != '"relation"':
        return None
    knowledge_fragment = clean_str(record_attributes[1])


# AFTER (with validation and sanitization):
async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str
):
    """
    Extract and validate a single relation from LLM output.

    Validation:
    - Exactly 3 fields (not <3, not >3)
    - Sanitize content
    - Validate completeness score range

    Returns:
        dict or None: Relation data if valid, None if invalid
    """

    # Validate field count (EXACT, not >=)
    if len(record_attributes) != 3:
        if len(record_attributes) > 1 and '"relation"' in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: Relation has {len(record_attributes)}/3 fields (expected exactly 3)"
            )
        return None

    # Validate first field is "relation"
    if record_attributes[0] != '"relation"':
        return None

    # Sanitize knowledge fragment (relation content)
    knowledge_fragment_raw = record_attributes[1]
    knowledge_fragment = sanitize_extracted_text(knowledge_fragment_raw, "relation")

    if not knowledge_fragment:
        logger.warning(
            f"{chunk_key}: Relation content became empty after sanitization. "
            f"Raw: '{knowledge_fragment_raw[:50]}...'"
        )
        return None

    # Parse completeness score
    try:
        weight = float(record_attributes[2])
        # Validate reasonable range
        if weight < 0 or weight > 10:
            logger.warning(
                f"{chunk_key}: Relation completeness score out of range (0-10): {weight}. "
                f"Using clamped value."
            )
            weight = max(0, min(10, weight))
    except (ValueError, IndexError) as e:
        logger.warning(
            f"{chunk_key}: Invalid completeness score: {record_attributes[2]}. "
            f"Using default 5.0. Error: {e}"
        )
        weight = 5.0  # Default fallback

    # Generate hash-based ID for relation node
    edge_id = compute_mdhash_id(knowledge_fragment, prefix="rel-")

    # Return validated relation data
    return dict(
        hyper_relation=edge_id,                # Relation node ID (hash-based)
        hyper_relation_content=knowledge_fragment,  # Actual content
        weight=weight,                         # Completeness score
        source_id=chunk_key,                   # Which chunk this came from
    )
```

**Part C: Update extraction result parsing (fix corrupted delimiters)**

**Location:** Line ~590 (in `_process_single_content` function, after LLM call)

```python
# File: bigrag/operate.py
# Function: _process_single_content
# Location: After line ~597 (after LLM extraction call)

# BEFORE (existing code):
async with llm_semaphore:
    final_result = await use_llm_func(hint_prompt)
history = pack_user_ass_to_openai_messages(hint_prompt, final_result)


# AFTER (add delimiter corruption fix):
async with llm_semaphore:
    final_result = await use_llm_func(hint_prompt)

# Fix corrupted delimiters BEFORE parsing
# (LLM sometimes outputs <> instead of <|>, || instead of <|>, etc.)
final_result = fix_delimiter_corruption(
    final_result,
    context_base["tuple_delimiter"]
)

history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
```

**Part D: Update gleaning result processing**

**Location:** Line ~600 (in gleaning loop)

```python
# File: bigrag/operate.py
# Function: _process_single_content
# Location: In gleaning loop (after line ~600)

# BEFORE (existing code):
for now_glean_index in range(entity_extract_max_gleaning):
    async with llm_semaphore:
        glean_result = await use_llm_func(continue_prompt, history_messages=history)

    history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
    final_result += glean_result  # ← Just concatenates!


# AFTER (smart merge with quality comparison):
for now_glean_index in range(entity_extract_max_gleaning):
    async with llm_semaphore:
        glean_result = await use_llm_func(continue_prompt, history_messages=history)

    # Fix corrupted delimiters in gleaning result
    glean_result = fix_delimiter_corruption(
        glean_result,
        context_base["tuple_delimiter"]
    )

    history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

    # Parse gleaning results into separate structures
    glean_records = split_string_by_multi_markers(
        glean_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )

    maybe_glean_nodes = defaultdict(list)
    maybe_glean_edges = defaultdict(list)
    now_hyper_relation = ""

    # Parse each gleaning record
    for record in glean_records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )

        # Try parsing as relation
        if_relation = await _handle_single_hyperrelation_extraction(
            record_attributes, chunk_key
        )
        if if_relation is not None:
            relation_id = if_relation["hyper_relation"]
            maybe_glean_edges[relation_id].append(if_relation)
            now_hyper_relation = relation_id

        # Try parsing as entity
        if_entities = await _handle_single_entity_extraction(
            record_attributes, chunk_key, now_hyper_relation
        )
        if if_entities is not None:
            entity_name = if_entities["entity_name"]
            maybe_glean_nodes[entity_name].append(if_entities)

    # SMART MERGE: Compare quality and keep better version
    # Merge entities
    for entity_name, glean_entity_list in maybe_glean_nodes.items():
        if entity_name in maybe_nodes:
            # Entity already exists - compare descriptions
            original_desc = maybe_nodes[entity_name][0].get("description", "")
            glean_desc = glean_entity_list[0].get("description", "")

            # Use quality scoring (considers length, keywords, completeness)
            original_quality = description_quality_score(original_desc)
            glean_quality = description_quality_score(glean_desc)

            if glean_quality > original_quality:
                logger.debug(
                    f"{chunk_key}: Gleaning improved entity '{entity_name}': "
                    f"quality {original_quality:.0f} → {glean_quality:.0f}"
                )
                maybe_nodes[entity_name] = glean_entity_list
            else:
                # Keep original (better quality)
                logger.debug(
                    f"{chunk_key}: Keeping original entity '{entity_name}' "
                    f"(better quality: {original_quality:.0f} vs {glean_quality:.0f})"
                )
        else:
            # New entity from gleaning
            logger.debug(f"{chunk_key}: Gleaning found new entity: '{entity_name}'")
            maybe_nodes[entity_name] = glean_entity_list

    # Merge relations
    for relation_id, glean_relation_list in maybe_glean_edges.items():
        if relation_id in maybe_edges:
            # Relation already exists - compare content quality
            original_content = maybe_edges[relation_id][0].get("hyper_relation_content", "")
            glean_content = glean_relation_list[0].get("hyper_relation_content", "")

            original_quality = description_quality_score(original_content)
            glean_quality = description_quality_score(glean_content)

            if glean_quality > original_quality:
                logger.debug(
                    f"{chunk_key}: Gleaning improved relation {relation_id[:16]}..."
                )
                maybe_edges[relation_id] = glean_relation_list
        else:
            # New relation from gleaning
            logger.debug(f"{chunk_key}: Gleaning found new relation: {relation_id[:16]}...")
            maybe_edges[relation_id] = glean_relation_list
```

---

### Implementation Step 4: Post-Extraction Validation

**File to modify:** `bigrag/operate.py`

**Location:** Add after line ~650 (after all extraction/gleaning, before return)

```python
# File: bigrag/operate.py
# Function: _process_single_content
# Location: Add new function before _process_single_content

def validate_extraction_results(
    maybe_nodes: dict,
    maybe_edges: dict,
    chunk_key: str
) -> dict:
    """
    Validate extraction results and detect potential orphan relations.

    Purpose: Catch orphan relations early (at extraction time) before they're
             stored in the graph, allowing for corrective action or warnings.

    Args:
        maybe_nodes: dict[entity_name, list[entity_data]]
        maybe_edges: dict[relation_id, list[relation_data]]
        chunk_key: Chunk identifier for logging

    Returns:
        dict: {
            "total_entities": int,
            "total_relations": int,
            "orphan_relations": list[str],  # Relation IDs without entities
            "warnings": list[str]
        }
    """
    validation_report = {
        "total_entities": len(maybe_nodes),
        "total_relations": len(maybe_edges),
        "orphan_relations": [],
        "warnings": []
    }

    # Check each relation for linked entities
    for relation_id, relation_list in maybe_edges.items():
        relation_content = relation_list[0].get("hyper_relation_content", "")

        # Find entities that reference this relation
        linked_entities = []
        for entity_name, entity_list in maybe_nodes.items():
            entity_relation = entity_list[0].get("hyper_relation", "")
            if entity_relation == relation_id:
                linked_entities.append(entity_name)

        if len(linked_entities) == 0:
            # Orphan relation detected!
            validation_report["orphan_relations"].append(relation_id)

            # Log warning with truncated content
            display_content = relation_content[:80] + "..." if len(relation_content) > 80 else relation_content
            warning = (
                f"ORPHAN RELATION (no entities): '{display_content}'"
            )
            validation_report["warnings"].append(warning)
            logger.warning(f"{chunk_key}: {warning}")

    # Log summary
    if validation_report["orphan_relations"]:
        orphan_count = len(validation_report["orphan_relations"])
        total_count = validation_report["total_relations"]
        orphan_rate = orphan_count / total_count if total_count > 0 else 0

        logger.warning(
            f"{chunk_key}: Found {orphan_count} orphan relations "
            f"out of {total_count} total relations ({orphan_rate:.1%})"
        )

    return validation_report


# Then update _process_single_content to call validation:
# Location: After line ~658 (after all parsing, before return)

# BEFORE (existing code):
        # ... parsing logic ...
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)


# AFTER (add validation before return):
        # ... parsing logic ...

        # Validate extraction results (detect orphans early)
        validation_report = validate_extraction_results(
            maybe_nodes,
            maybe_edges,
            chunk_key
        )

        # If orphan rate is very high, log error
        if validation_report["total_relations"] > 0:
            orphan_rate = (
                len(validation_report["orphan_relations"]) /
                validation_report["total_relations"]
            )

            # Threshold: 10% (not 30% as in original plan)
            if orphan_rate > 0.10:
                logger.error(
                    f"{chunk_key}: HIGH ORPHAN RATE ({orphan_rate:.1%})! "
                    f"Expected <5%, found {len(validation_report['orphan_relations'])}/"
                    f"{validation_report['total_relations']} orphans. "
                    f"LLM may not be following extraction rules."
                )

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
```

---

### Implementation Step 5: Prompt Improvements

**File to modify:** `bigrag/prompt.py`

**Location:** Replace `PROMPTS["entity_extraction"]` (line ~20)

**CRITICAL NOTE**: Test prompt length with your LLM first!

```python
# File: bigrag/prompt.py
# Location: Line ~20

# BACKUP ORIGINAL FIRST:
# PROMPTS["entity_extraction_BACKUP_20250113"] = PROMPTS["entity_extraction"]

# NEW PROMPT (with explicit sequencing):
PROMPTS["entity_extraction"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and knowledge segments from text documents.

---Critical Extraction Rule---

⚠️ MANDATORY SEQUENCING: After EVERY ("relation"...) output, you MUST immediately extract
ALL entities mentioned in that relation BEFORE outputting the next ("relation"...).

Failure to follow this rule creates ORPHAN RELATIONS that become unreachable during retrieval!

---Instructions---

1. **Knowledge Segment Extraction (Relations):**
   * Divide the text into complete, self-contained knowledge segments
   * Each segment should capture a coherent piece of information
   * Assign a completeness score (0-10) based on how complete the information is
   * Format: ("relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. **Entity Extraction (MUST FOLLOW EACH RELATION):**
   * **IMMEDIATELY after each ("relation"...), extract ALL entities mentioned in that relation**
   * **DO NOT output another ("relation"...) until all entities from previous relation are extracted**
   * For each entity, extract:
     - entity_name: Use UPPERCASE (e.g., "LIONEL MESSI", not "Lionel Messi")
     - entity_type: Must be one of: {entity_types}. If none fit, use "category"
     - entity_description: Comprehensive description of attributes and activities
     - key_score (0-100): Importance of the entity in the text
   * Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. **Sequencing Rules (CRITICAL):**

   ✅ CORRECT Pattern (Interleaved: relation → entities → relation → entities):

   ("relation"{tuple_delimiter}"Lionel Messi plays for Inter Miami in Major League Soccer"{tuple_delimiter}9){record_delimiter}
   ("entity"{tuple_delimiter}"LIONEL MESSI"{tuple_delimiter}"person"{tuple_delimiter}"Professional footballer"{tuple_delimiter}95){record_delimiter}
   ("entity"{tuple_delimiter}"INTER MIAMI"{tuple_delimiter}"organization"{tuple_delimiter}"MLS soccer club"{tuple_delimiter}85){record_delimiter}
   ("entity"{tuple_delimiter}"MAJOR LEAGUE SOCCER"{tuple_delimiter}"organization"{tuple_delimiter}"American soccer league"{tuple_delimiter}80){record_delimiter}
   ("relation"{tuple_delimiter}"Argentina won the 2022 FIFA World Cup in Qatar"{tuple_delimiter}9){record_delimiter}
   ("entity"{tuple_delimiter}"ARGENTINA"{tuple_delimiter}"organization"{tuple_delimiter}"National football team"{tuple_delimiter}90){record_delimiter}
   ("entity"{tuple_delimiter}"2022 FIFA WORLD CUP"{tuple_delimiter}"event"{tuple_delimiter}"Football tournament"{tuple_delimiter}85){record_delimiter}
   ("entity"{tuple_delimiter}"QATAR"{tuple_delimiter}"geo"{tuple_delimiter}"Host country"{tuple_delimiter}80){record_delimiter}

   ❌ INCORRECT Pattern (Consecutive relations - CREATES ORPHANS):

   ("relation"{tuple_delimiter}"Lionel Messi plays for Inter Miami in Major League Soccer"{tuple_delimiter}9){record_delimiter}
   ("relation"{tuple_delimiter}"Argentina won the 2022 FIFA World Cup in Qatar"{tuple_delimiter}9){record_delimiter}
   ^^ WRONG! Missing entities for both relations ^^

4. **Multi-Entity Relations (N-ary Decomposition):**
   * If a relation mentions 3+ entities, extract the relation ONCE
   * Then extract EACH entity individually
   * Example: "Messi, Suarez, and Alba played together at Barcelona"

     ("relation"{tuple_delimiter}"Messi, Suarez, and Alba played together at Barcelona"{tuple_delimiter}8){record_delimiter}
     ("entity"{tuple_delimiter}"LIONEL MESSI"{tuple_delimiter}"person"{tuple_delimiter}"Professional footballer"{tuple_delimiter}95){record_delimiter}
     ("entity"{tuple_delimiter}"LUIS SUAREZ"{tuple_delimiter}"person"{tuple_delimiter}"Professional footballer"{tuple_delimiter}90){record_delimiter}
     ("entity"{tuple_delimiter}"JORDI ALBA"{tuple_delimiter}"person"{tuple_delimiter}"Professional footballer"{tuple_delimiter}85){record_delimiter}
     ("entity"{tuple_delimiter}"BARCELONA"{tuple_delimiter}"organization"{tuple_delimiter}"Spanish football club"{tuple_delimiter}90){record_delimiter}

5. **Formatting Rules:**
   * Use **{record_delimiter}** as the delimiter between records
   * Ensure each record ends with ){record_delimiter}
   * Do NOT add extra delimiters or newlines between records
   * Output all records in a single continuous list
   * Entity names MUST be UPPERCASE for consistency

6. **Completion:**
   * When finished, output {completion_delimiter}

---Examples---
{examples}

---Real Data---
Text: {input_text}

---Output---
"""

# Update examples to match new format:
PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order. Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us." The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands.
################
Output:
("relation"{tuple_delimiter}"Alex clenched his jaw, feeling frustration against Taylor's authoritarian certainty."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"ALEX"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person experiencing frustration"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"TAYLOR"{tuple_delimiter}"person"{tuple_delimiter}"Taylor displays authoritarian certainty"{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, rebelling against Cruz's vision of control."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"ALEX"{tuple_delimiter}"person"{tuple_delimiter}"Alex is committed to discovery"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"JORDAN"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares commitment to discovery with Alex"{tuple_delimiter}88){record_delimiter}
("entity"{tuple_delimiter}"CRUZ"{tuple_delimiter}"person"{tuple_delimiter}"Cruz has a narrowing vision of control and order"{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Taylor paused beside Jordan and observed the device with reverence."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"TAYLOR"{tuple_delimiter}"person"{tuple_delimiter}"Taylor showed reverence for the device"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"JORDAN"{tuple_delimiter}"person"{tuple_delimiter}"Jordan was beside Taylor"{tuple_delimiter}88){record_delimiter}
("entity"{tuple_delimiter}"DEVICE"{tuple_delimiter}"object"{tuple_delimiter}"The device was observed by Taylor with reverence"{tuple_delimiter}80){record_delimiter}
("relation"{tuple_delimiter}"Taylor said the tech could change the game for everyone if understood."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"TAYLOR"{tuple_delimiter}"person"{tuple_delimiter}"Taylor spoke about the tech's potential"{tuple_delimiter}85){record_delimiter}
{completion_delimiter}
#############################""",

    """Example 2:

Text:
Manchester City secured their third consecutive Premier League title in the 2022-23 season under manager Pep Guardiola. The team's success has been built on the brilliance of players like Erling Haaland, Kevin De Bruyne, and Phil Foden. Haaland, who joined from Borussia Dortmund, broke the Premier League single-season scoring record with 36 goals.
################
Output:
("relation"{tuple_delimiter}"Manchester City secured their third consecutive Premier League title in 2022-23 under Pep Guardiola."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"MANCHESTER CITY"{tuple_delimiter}"organization"{tuple_delimiter}"Manchester City is a football club that won three consecutive Premier League titles"{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"PREMIER LEAGUE"{tuple_delimiter}"organization"{tuple_delimiter}"The Premier League is the top English football division"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"PEP GUARDIOLA"{tuple_delimiter}"person"{tuple_delimiter}"Pep Guardiola is the manager of Manchester City"{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"Manchester City's success is built on players like Erling Haaland, Kevin De Bruyne, and Phil Foden."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"MANCHESTER CITY"{tuple_delimiter}"organization"{tuple_delimiter}"Manchester City has brilliant players"{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"ERLING HAALAND"{tuple_delimiter}"person"{tuple_delimiter}"Erling Haaland is a key player for Manchester City"{tuple_delimiter}92){record_delimiter}
("entity"{tuple_delimiter}"KEVIN DE BRUYNE"{tuple_delimiter}"person"{tuple_delimiter}"Kevin De Bruyne is a key player for Manchester City"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"PHIL FODEN"{tuple_delimiter}"person"{tuple_delimiter}"Phil Foden is a key player for Manchester City"{tuple_delimiter}88){record_delimiter}
("relation"{tuple_delimiter}"Erling Haaland joined from Borussia Dortmund and broke the Premier League scoring record with 36 goals."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"ERLING HAALAND"{tuple_delimiter}"person"{tuple_delimiter}"Haaland broke the Premier League scoring record with 36 goals"{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"BORUSSIA DORTMUND"{tuple_delimiter}"organization"{tuple_delimiter}"Borussia Dortmund is the club Haaland joined Manchester City from"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"PREMIER LEAGUE"{tuple_delimiter}"organization"{tuple_delimiter}"The Premier League has a single-season scoring record"{tuple_delimiter}80){record_delimiter}
{completion_delimiter}
#############################""",
]
```

**Test prompt length:**
```python
# Quick test script to check prompt length
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
prompt = PROMPTS["entity_extraction"].format(
    tuple_delimiter="<|>",
    record_delimiter="##",
    completion_delimiter="<|COMPLETE|>",
    entity_types=", ".join(["person", "organization", "geo", "event", "category"]),
    examples="\n".join(PROMPTS["entity_extraction_examples"]),
    input_text="Sample text" * 100  # Simulate chunk
)

token_count = len(enc.encode(prompt))
print(f"Prompt tokens: {token_count}")
print(f"GPT-4o-mini limit: 128000 tokens")
print(f"Safe: {token_count < 8000}")  # Leave room for response

# Expected: ~2500-3500 tokens (safe for GPT-4o-mini)
```

---

### Implementation Step 6: Rebuild and Test

**Step 6.1: Performance Monitoring**

Add timing code to measure impact:

```python
# File: bigrag/operate.py
# Function: extract_entities
# Location: At start and end of function

import time

async def extract_entities(...):
    # Start timing
    start_time = time.time()

    # ... existing extraction logic ...

    # End timing
    elapsed = time.time() - start_time
    logger.info(
        f"[PERFORMANCE] Extraction completed for {len(ordered_chunks)} chunks "
        f"in {elapsed:.2f}s ({elapsed/len(ordered_chunks):.2f}s per chunk)"
    )

    return chunk_results
```

**Step 6.2: Rebuild Football Graph**

```bash
cd d:/BiG-RAG

# Rebuild with new improvements
python script_build.py --data_source football

# This will take 2-4 hours depending on corpus size
# Monitor progress:
tail -f build.log  # If running in background with nohup

# Expected output changes:
# OLD: "Extracted 80 entities, 80 relations" with many warnings
# NEW: "Extracted 120+ entities, 80 relations" with fewer orphan warnings
```

**Step 6.3: Compare Results**

```bash
# Run test with comparison
python test_scripts/test_orphan_detection.py football --compare expr/football_backup/graph_chunk_entity_relation.graphml

# Expected output:
# ================================================================================
# BEFORE/AFTER COMPARISON: football
# ================================================================================
#
# 📊 Orphan Relation Rate:
#    Before: 22.5% (18/80)
#    After:  4.2% (3/72)     ← Target achieved!
#    Change: -18.3% (-81.3%)
#
# 📈 Graph Connectivity:
#    Edges Before: 108
#    Edges After:  285        ← 164% increase!
#    Change: +177
#
# 🔗 Avg Edges per Relation:
#    Before: 1.35
#    After:  3.96             ← 193% increase!
#    Change: +2.61
#
# ================================================================================
# OVERALL VERDICT:
# ================================================================================
#    ✅ EXCELLENT IMPROVEMENT (>15% reduction)
#    ✅ Target achieved: Orphan rate now <5%
```

**Step 6.4: Verify Messi-Argentina Connection**

```bash
# Check if orphan relations are now connected
python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('expr/football/graph_chunk_entity_relation.graphml')
root = tree.getroot()
ns = {'gml': 'http://graphml.graphdrawing.org/xmlns'}

# Find Messi-related edges
messi_edges = []
for edge in root.findall('.//gml:edge', ns):
    if 'MESSI' in edge.get('source') or 'MESSI' in edge.get('target'):
        messi_edges.append((edge.get('source'), edge.get('target')))

print(f'Messi has {len(messi_edges)} edges:')
for src, tgt in messi_edges[:10]:
    print(f'  {src} ↔ {tgt}')
"

# Expected output:
# Messi has 12 edges:  (up from 2!)
#   rel-abc123... ↔ "LIONEL MESSI"
#   rel-def456... ↔ "LIONEL MESSI"
#   rel-ghi789... ↔ "LIONEL MESSI"  ← Including Argentina relations!
```

---

## Testing Strategy

### Phase 1: Baseline Establishment (Day 1, Morning)

**1.1: Run baseline test**
```bash
python test_scripts/test_orphan_detection.py football
```

**1.2: Record metrics**
```bash
# Save to file for later comparison
python test_scripts/test_orphan_detection.py football > test_results/baseline_20250113.txt
```

**1.3: Backup current graph**
```bash
mkdir -p expr/football_backup
cp expr/football/graph_chunk_entity_relation.graphml expr/football_backup/
cp expr/football/vdb_*.json expr/football_backup/
cp expr/football/kv_store_*.json expr/football_backup/
```

### Phase 2: Implementation (Day 1, Afternoon - Day 2)

**2.1: Implement in order**
1. Step 1.5: Metadata formatting enhancement (operate.py) - 15 minutes
2. Step 2: Enhanced sanitization (utils.py) - 1 hour
3. Step 3: Update extraction handlers (operate.py) - 2 hours
4. Step 4: Post-extraction validation (operate.py) - 1 hour
5. Step 5: Prompt improvements (prompt.py) - 30 minutes
6. Test each step with unit tests

**2.2: Unit tests**
```bash
# Test sanitization
python -c "
from bigrag.utils import sanitize_extracted_text
assert sanitize_extracted_text('  MESSI  ', 'entity_name') == 'MESSI'
assert sanitize_extracted_text('  person  ', 'entity_type') == 'person'
print('✅ Sanitization tests passed')
"

# Test delimiter corruption fix
python -c "
from bigrag.utils import fix_delimiter_corruption
assert fix_delimiter_corruption('entity<>name<>type', '<|>') == 'entity<|>name<|>type'
assert fix_delimiter_corruption('relation||content||score', '<|>') == 'relation<|>content<|>score'
print('✅ Delimiter fix tests passed')
"

# Test quality scoring
python -c "
from bigrag.utils import description_quality_score
score1 = description_quality_score('Short')
score2 = description_quality_score('Lionel Messi is a professional footballer known for winning 8 Ballon d\'Or awards.')
assert score2 > score1
print('✅ Quality scoring tests passed')
"
```

### Phase 3: Integration Testing (Day 2-3)

**3.1: Test on small dataset first**
```bash
# Create mini test corpus (10 documents only)
head -10 datasets/football/raw/corpus.jsonl > datasets/football_mini/raw/corpus.jsonl

# Build mini graph
python script_build.py --data_source football_mini

# Check results
python test_scripts/test_orphan_detection.py football_mini

# Expected: Very low orphan rate on small dataset
```

**3.2: Test on full football dataset**
```bash
# Full rebuild (2-4 hours)
python script_build.py --data_source football

# Compare with baseline
python test_scripts/test_orphan_detection.py football --compare expr/football_backup/graph_chunk_entity_relation.graphml
```

**3.3: Performance check**
```bash
# Check processing time increased by <20%
# OLD: ~2.5 hours for football corpus
# NEW: ~3.0 hours acceptable (20% increase)
# NEW: >3.5 hours (>40% increase) → need optimization
```

### Phase 4: Regression Testing (Day 3)

**4.1: Test on multiple datasets**
```bash
# Test on different dataset types
python script_build.py --data_source 2WikiMultiHopQA
python test_scripts/test_orphan_detection.py 2WikiMultiHopQA

# Verify improvement generalizes across datasets
```

**4.2: Retrieval quality test**
```bash
# Query tests (manual verification)
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What did Messi achieve in 2023?"]}'

# Expected: Richer context, more entities, more facts
# Before: 2-3 context items
# After: 5-7 context items with better coverage
```

---

## Rollback Plan

### Scenario 1: Prompt Changes Make Things Worse

**Symptoms:**
- Orphan rate increases instead of decreases
- LLM produces malformed output
- Extraction errors increase

**Rollback Steps:**
```bash
# Option A: Git revert
git checkout bigrag/prompt.py

# Option B: Use backup
cp bigrag/prompt_backup.py bigrag/prompt.py

# Rebuild with old prompt
python script_build.py --data_source football

# Verify rollback
python test_scripts/test_orphan_detection.py football
```

### Scenario 2: Sanitization Too Strict (Losing Valid Data)

**Symptoms:**
- Fewer entities extracted than before
- Many warnings: "Entity name became empty after sanitization"
- Total nodes decreased instead of increased

**Fix:**
```python
# Relax sanitization rules in bigrag/utils.py
# Change from:
if any(char in text for char in ["'", '"', "(", ")", ...]):
    return ""  # Reject

# To:
if any(char in text for char in ["'", '"', "(", ")", ...]):
    logger.warning(f"Entity name contains special characters: {text}")
    # Continue processing instead of rejecting
```

### Scenario 3: Performance Degradation >40%

**Symptoms:**
- Processing time increased by >40%
- Graph building takes >4 hours

**Fix:**
```python
# Disable validation in production mode
# In bigrag/operate.py:
ENABLE_VALIDATION = False  # Set to False for production

if ENABLE_VALIDATION:
    validation_report = validate_extraction_results(...)
```

### Scenario 4: Complete Failure (Graph Not Building)

**Symptoms:**
- script_build.py crashes
- Many errors in logs
- No graph files produced

**Emergency Rollback:**
```bash
# Full revert to previous version
git reset --hard backup-pre-orphan-fix

# Rebuild with old code
python script_build.py --data_source football

# Verify system working
python test_scripts/test_orphan_detection.py football
```

---

## Success Criteria

### Primary Metrics (Must Achieve)

| Metric | Before | Target | Acceptable Range |
|--------|--------|--------|------------------|
| **Orphan Relation Rate** | 22.5% | <5% | 3-7% |
| **Total Edges** | 108 | 280-320 | 250-350 |
| **Avg Edges/Relation** | 1.35 | 3.0-3.5 | 2.5-4.0 |
| **Processing Time** | 2.5h | <3.0h | 2.5-3.5h |

### Secondary Metrics (Nice to Have)

| Metric | Before | Target |
|--------|--------|--------|
| **Retrieval Recall** | Baseline | +30-40% |
| **Retrieval Precision** | Baseline | +15-25% |
| **Query Response Quality** | 3-4 items | 5-7 items |
| **Messi-Argentina Connection** | Missing | Present |

### Validation Checklist

**✅ Must Pass:**
- [ ] Orphan relation rate <7%
- [ ] No decrease in total entities
- [ ] Total edges increased by >150%
- [ ] Processing time increased by <50%
- [ ] No crashes during graph building
- [ ] All extraction handlers working correctly
- [ ] Validation warnings appear in logs

**⚠️ Should Pass:**
- [ ] Orphan relation rate <5%
- [ ] Messi-Argentina connection exists
- [ ] Query quality improved
- [ ] No false positive orphan warnings

**🎯 Ideal:**
- [ ] Orphan relation rate <3%
- [ ] Total edges 300+
- [ ] Avg edges/relation >3.5
- [ ] Processing time <3 hours

---

## Troubleshooting

### Issue 1: "Orphan rate only decreased to 15%"

**Diagnosis:**
- LLM not following prompt instructions
- Prompt too long/complex
- Model capacity insufficient

**Solutions:**
1. **Simplify prompt:**
   ```python
   # Remove verbose examples, keep only rules
   # Reduce from 3 examples to 1 example
   ```

2. **Test with different LLM:**
   ```python
   # Try GPT-4 instead of GPT-4o-mini
   # Try Claude 3.5 Sonnet
   ```

3. **Add post-processing fallback:**
   ```python
   # Use NER to extract entities from orphan relations
   import spacy
   nlp = spacy.load("en_core_web_sm")
   # ... entity extraction logic
   ```

### Issue 2: "Sanitization rejecting valid entities"

**Diagnosis:**
- Rules too strict
- Valid entity names contain special characters

**Solutions:**
1. **Check logs for patterns:**
   ```bash
   grep "became empty after sanitization" build.log | head -20
   # Look for common patterns
   ```

2. **Relax specific rules:**
   ```python
   # If many "O'Neill" entities rejected:
   # Allow apostrophes in entity names
   if field_type == "entity_name":
       # Don't remove apostrophes
       text = text.replace('"', '')  # Only remove double quotes
   ```

### Issue 3: "Performance degradation >40%"

**Diagnosis:**
- Validation taking too long
- Quality scoring expensive
- Delimiter fixing overhead

**Solutions:**
1. **Profile bottlenecks:**
   ```python
   import cProfile
   cProfile.run('validate_extraction_results(...)')
   ```

2. **Optimize hot paths:**
   ```python
   # Cache regex compilations
   import functools

   @functools.lru_cache(maxsize=128)
   def compiled_delimiter_pattern(delimiter):
       return re.compile(re.escape(delimiter))
   ```

3. **Disable non-critical validation:**
   ```python
   # Skip quality scoring in production
   ENABLE_QUALITY_SCORING = False

   if ENABLE_QUALITY_SCORING:
       quality = description_quality_score(desc)
   else:
       quality = len(desc)  # Simple length comparison
   ```

### Issue 4: "New bugs introduced"

**Diagnosis:**
- Import errors
- Function signature changes
- Missing dependencies

**Solutions:**
1. **Check imports:**
   ```bash
   python -c "from bigrag.utils import sanitize_extracted_text"
   # Should not error
   ```

2. **Run existing tests:**
   ```bash
   pytest test_scripts/  # If you have pytest
   # Or manually test critical paths
   ```

3. **Incremental rollback:**
   ```bash
   # Revert only problematic changes
   git checkout bigrag/utils.py  # Keep other changes
   ```

### Issue 5: "Prompt token limit exceeded"

**Diagnosis:**
- Prompt + chunk content > model limit
- Examples too verbose

**Solutions:**
1. **Shorten examples:**
   ```python
   # Use 1 example instead of 3
   PROMPTS["entity_extraction_examples"] = [
       PROMPTS["entity_extraction_examples"][0]  # Keep only first example
   ]
   ```

2. **Move examples to separate section:**
   ```python
   # Reference examples by ID instead of embedding
   "See Examples in prompt_examples.txt"
   ```

3. **Use model with larger context:**
   ```python
   # Switch from GPT-4o-mini (128K) to GPT-4-turbo (128K) or Claude 3.5 (200K)
   ```

---

## Implementation Timeline

### Day 1: Setup & Core Implementation (8 hours)

**Morning (4 hours):**
- [ ] 30min: Backup code, establish baseline
- [ ] 15min: Implement Step 1.5 (metadata formatting)
- [ ] 1h: Implement Step 2 (sanitization functions)
- [ ] 1h: Implement Step 3 Part A-B (entity/relation handlers)
- [ ] 45min: Unit tests for sanitization
- [ ] 30min: Break

**Afternoon (4 hours):**
- [ ] 1h: Implement Step 3 Part C-D (delimiter fix, gleaning)
- [ ] 1h: Implement Step 4 (validation)
- [ ] 1h: Implement Step 5 (prompt improvements)
- [ ] 1h: Integration tests

### Day 2: Testing & Refinement (8 hours)

**Morning (4 hours):**
- [ ] 30min: Create mini test dataset
- [ ] 1h: Test on mini dataset, verify improvements
- [ ] 2h: Full rebuild football dataset
- [ ] 30min: Compare results

**Afternoon (4 hours):**
- [ ] 1h: Analyze results, fix issues if needed
- [ ] 2h: Test on additional dataset (2WikiMultiHopQA)
- [ ] 1h: Document changes, update metrics

### Day 3: Validation & Deployment (4 hours)

**Morning (4 hours):**
- [ ] 1h: Run retrieval quality tests
- [ ] 1h: Performance benchmarking
- [ ] 1h: Code review, final adjustments
- [ ] 1h: Documentation, commit changes

**Total: 20 hours over 3 days**

---

## Post-Implementation

### Documentation Updates

**Files to update:**
1. `CHANGELOG.md`: Add entry for orphan reduction improvements
2. `README.md`: Update graph quality metrics
3. `docs/technical/KG_CONSTRUCTION.md`: Document new validation process
4. `docs/updates/ORPHAN_REDUCTION_2025.md`: Detailed implementation notes

### Monitoring

**Add to production monitoring:**
```python
# In bigrag/operate.py - log metrics after each build
logger.info(f"[METRICS] Orphan rate: {orphan_rate:.1%}")
logger.info(f"[METRICS] Total edges: {total_edges}")
logger.info(f"[METRICS] Avg edges/relation: {avg_edges:.2f}")

# Export to metrics file
with open("expr/{dataset}/metrics.json", "w") as f:
    json.dump({
        "orphan_rate": orphan_rate,
        "total_edges": total_edges,
        "avg_edges_per_relation": avg_edges,
        "timestamp": datetime.now().isoformat()
    }, f)
```

### Next Steps

**Future Improvements (Post-Implementation):**
1. **Map-reduce summarization** (if entities have 20+ mentions)
2. **NER fallback** (catch missed entities automatically)
3. **Two-pass extraction** (extract relations first, then entities explicitly)
4. **Fine-tune LLM** on entity extraction task
5. **Graph validation dashboard** (visualize orphan rates over time)

---

## Conclusion

This implementation plan provides a comprehensive, step-by-step guide to reduce BiG-RAG's orphan node rate from 22.5% to <5%. The approach combines:

1. **Prompt improvements** (enforce sequential extraction)
2. **Input sanitization** (handle malformed LLM output)
3. **Quality-based gleaning** (keep best entity descriptions)
4. **Early validation** (detect orphans before storage)
5. **Comprehensive testing** (baseline → implement → verify → deploy)

The plan includes exact file locations, code snippets, rollback strategies, and troubleshooting guides to ensure successful implementation in any session.

**Key Success Factors:**
- Test baseline first (critical for comparison)
- Implement incrementally (easier to debug)
- Monitor performance (catch regressions early)
- Have rollback ready (safety net)

**Expected Outcome:**
- Orphan rate: 22.5% → <5% (80%+ reduction)
- Graph connectivity: 108 edges → 280-320 edges (160%+ increase)
- Retrieval quality: +30-40% recall, +15-25% precision

Good luck with implementation! 🚀
