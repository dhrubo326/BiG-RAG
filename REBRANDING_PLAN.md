# BiG-RAG Rebranding Plan

**Project**: Rebrand Graph-R1 → BiG-RAG
**Date**: 2025-10-24
**Status**: In Progress

## Overview

This document outlines the complete rebranding strategy for transforming the Graph-R1 codebase to BiG-RAG. The rebranding involves:

1. **Identifier changes**: `graphr1` → `bigrag`
2. **Terminology changes**: `hypergraph`/`hyperedge`/`hypernode` → `bipartite_graph`/`bipartite_edge`/`bipartite_node`
3. **Class renaming**: `GraphR1` → `BiGRAG`
4. **No logic changes** - only renaming and documentation updates

## Rationale

The Graph-R1 paper refers to a "hypergraph" structure, but the actual implementation uses a **bipartite graph construction**. This rebranding aligns the terminology with the actual implementation and provides clearer, more accurate naming.

## Rebranding Checklist

### 1. Repository / Top-level
- [x] Rename top-level folder `graphr1` → `bigrag` (already done by user)
- [ ] Update README.md
- [ ] Update all documentation files

### 2. Code Identifiers

#### Python Module Files
- [x] Rename: `bigrag/graphr1.py` → `bigrag/bigrag.py` ✅ COMPLETED
- [x] Update class name: `GraphR1` → `BiGRAG` ✅ COMPLETED
- [x] Update all imports: `from .graphr1 import BiGRAG` → `from .bigrag import BiGRAG` ✅ COMPLETED

#### Key Identifiers to Change
- [ ] Class: `GraphR1` → `BiGRAG`
- [ ] Working directory default: `graphr1_cache_*` → `bigrag_cache_*`
- [ ] Log file: `graphr1.log` → `bigrag.log`
- [ ] Conda env references: `graphr1` → `bigrag`

### 3. Terminology Changes

#### Hypergraph → Bipartite Graph
- [ ] `hyperedge` → `bipartite_edge`
- [ ] `hyperedges` → `bipartite_edges`
- [ ] `hyperedges_vdb` → `bipartite_edges_vdb`
- [ ] `hyperedge_name` → `bipartite_edge_name`
- [ ] `hyperedge_match` → `bipartite_edge_match`
- [ ] `hyperedge_vdb` → `bipartite_edge_vdb`
- [ ] String markers: `"<hyperedge>"` → `"<bipartite_edge>"`
- [ ] Comments: "hyperedge" → "bipartite edge"
- [ ] `hypergraph` → `bipartite_graph`
- [ ] `HyperGraph` → `BipartiteGraph`

### 4. Files to Update

#### Python Files (Code)
- [ ] `bigrag/__init__.py`
- [ ] `bigrag/graphr1.py` (rename to `bigrag_core.py`)
- [ ] `bigrag/base.py`
- [ ] `bigrag/operate.py`
- [ ] `bigrag/utils.py`
- [ ] `bigrag/kg/chroma_impl.py`
- [ ] `bigrag/kg/milvus_impl.py`
- [ ] `bigrag/kg/mongo_impl.py`
- [ ] `bigrag/kg/neo4j_impl.py`
- [ ] `bigrag/kg/oracle_impl.py`
- [ ] `bigrag/kg/tidb_impl.py`
- [ ] `script_api.py`
- [ ] `script_build.py`
- [ ] `docs/Helper_code/api_server.py`
- [ ] `docs/Helper_code/build_knowledge_graph.py`

#### Documentation Files
- [ ] `README.md`
- [ ] `CLAUDE.md`
- [ ] `docs/Graph-R1_full_paper.md` (keep as reference, add note about rebranding)
- [ ] `docs/BiG-RAG_Full_Paper.md`
- [ ] `docs/SETUP_AND_TESTING_GUIDE.md`
- [ ] `docs/BIPARTITE_VALIDATION_AND_RELEVANCE_FILTERING_COMPARISON.md`
- [ ] `docs/DEEP_DIVE_INDEXING_PIPELINES.md`
- [ ] `docs/EDUCATIONAL_DEEP_DIVE_RETRIEVAL_ARCHITECTURES.md`
- [ ] `docs/GRAPH_R1_VS_BIGRAG_DEEP_DIVE.md`
- [ ] `docs/Helper_code/README.md`

#### Configuration Files
- [ ] `setup.py` (if contains graphr1 references)
- [ ] `requirements_graphrag_only.txt` (filename and content)

### 5. Specific Replacements

#### In Code
```python
# OLD → NEW
GraphR1 → BiGRAG
graphr1 → bigrag
graphr1_cache → bigrag_cache
graphr1.log → bigrag.log
hyperedges_vdb → bipartite_edges_vdb
hyperedge_vdb → bipartite_edge_vdb
hyperedge_name → bipartite_edge_name
hyperedge_match → bipartite_edge_match
"<hyperedge>" → "<bipartite_edge>"
.replace("<hyperedge>","") → .replace("<bipartite_edge>","")
all_hyperedges_data → all_bipartite_edges_data
_merge_hyperedges_then_upsert → _merge_bipartite_edges_then_upsert
```

#### In Documentation
```
Graph-R1 → BiG-RAG (in most places, except citations)
knowledge hypergraph → bipartite knowledge graph
HyperGraph → BipartiteGraph
hypergraph → bipartite graph
hyperedge → bipartite edge
hypernode → bipartite node
```

### 6. Files NOT to Change
- [ ] `docs/Graph-R1_full_paper.md` - Keep as historical reference with note
- [ ] BibTeX citations - Keep original paper name
- [ ] External URLs and links to Graph-R1 paper
- [ ] Acknowledgement section mentioning Graph-R1's origin

### 7. Verification Steps
- [ ] All imports resolve correctly
- [ ] No broken relative imports
- [ ] String consistency (all hyperedge→bipartite_edge)
- [ ] Documentation coherence
- [ ] Conda/venv instructions updated
- [ ] File naming conventions followed

## Implementation Order

1. **Phase 1: Core Code Files** (Python modules in `bigrag/`)
   - Rename `graphr1.py` → `bigrag_core.py`
   - Update class `GraphR1` → `BiGRAG`
   - Replace hypergraph terminology
   - Update `__init__.py`

2. **Phase 2: Dependent Code Files**
   - Update `script_*.py` files
   - Update `docs/Helper_code/*.py` files
   - Update kg implementation files

3. **Phase 3: Documentation**
   - Update `README.md`
   - Update all docs in `docs/` folder
   - Create CHANGELOG entry

4. **Phase 4: Configuration**
   - Update any config files
   - Update installation instructions

5. **Phase 5: Final Verification**
   - Check all imports
   - Verify string consistency
   - Test import statements

## Notes

- **No logic changes**: All changes are purely cosmetic (renaming, rebranding)
- **Production-ready**: Code must remain functionally identical
- **Bipartite graph is accurate**: The implementation uses bipartite graph structure, not true hypergraph
- **Legal rights**: User has confirmed legal right to rebrand

## Success Criteria

✓ All `graphr1` identifiers replaced with `bigrag`
✓ All `hyperedge/hypergraph` terms replaced with `bipartite_edge/bipartite_graph`
✓ All imports functional
✓ Documentation coherent and accurate
✓ No logic changes
✓ Historical references to Graph-R1 paper preserved where appropriate
