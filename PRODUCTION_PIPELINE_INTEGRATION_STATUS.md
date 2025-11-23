# ProductionKGPipeline Integration Status

**Date**: January 23, 2025
**Goal**: Make ProductionKGPipeline a drop-in replacement for standard extraction in BiGRAG.ainsert()
**Status**: 40% Complete (Config + Branch Logic Done, Core Implementation Pending)

---

## ✅ Completed (Phase 1-2)

### Phase 1: Config Flags Added
**File**: `bigrag/bigrag.py` (lines 168-174)

```python
# Production KG Pipeline (NEW - opt-in for higher accuracy)
use_production_pipeline: bool = False  # Default: False (backward compatible)
production_pipeline_config: dict = field(default_factory=lambda: {
    "validation_level": "MODERATE",  # STRICT (99%) | MODERATE (95%) | LENIENT (80%)
    "enable_entity_linking": True,
    "extraction_mode": "semi_structured"  # structured | semi_structured | unstructured
})
```

**Result**: Users can now enable production pipeline via:
```python
rag = BiGRAG(
    working_dir="expr/educational_kg",
    use_production_pipeline=True  # Enable new pipeline
)
```

---

### Phase 2: Branch Logic in ainsert()
**File**: `bigrag/bigrag.py` (lines 384-440)

**Changes**:
- Added if/else branch to choose between production vs standard pipeline
- Standard pipeline remains **100% unchanged** (backward compatible)
- Production pipeline calls `_process_document_with_production_pipeline()` (TO BE IMPLEMENTED)

**Code Structure**:
```python
if self.use_production_pipeline:
    logger.info("[Production Pipeline] Using enhanced extraction...")
    for doc_key, doc in new_docs.items():
        await self._process_document_with_production_pipeline(
            doc_key,
            doc["content"],
            doc.get("metadata", {})
        )
else:
    # EXISTING: Standard pipeline (unchanged)
    # ... chunking → extract_entities → upsert ...
```

---

## ⏳ Pending (Phase 3-5)

### Phase 3: Implement Core Methods (CRITICAL - 400+ lines)

Two methods need to be added to `bigrag/bigrag.py`:

#### Method 1: `_process_document_with_production_pipeline()`

**Location**: Add after `ainsert()` method (around line 490)

**Purpose**: Process document with ProductionKGPipeline and insert to BiGRAG storage

**Implementation**: See [PRODUCTION_PIPELINE_METHOD_IMPLEMENTATION.md](PRODUCTION_PIPELINE_METHOD_IMPLEMENTATION.md) for complete code (400+ lines)

**Key Steps**:
1. Initialize ProductionKGPipeline with API key
2. Run pipeline on FULL document (before chunking)
3. Check validation status (PASS/WARNING/FAIL)
4. Map ProductionPipeline chunks → BiGRAG chunk IDs
5. Remap source_ids in entities/relations
6. Insert entities using `_merge_nodes_then_upsert()` (CORRECT signature)
7. Insert relations using `_merge_relations_then_upsert()` (CORRECT signature)
8. Index to vector DBs (entities, relations, chunks)
9. Store chunks and full document to KV storage

**Fallback Logic**:
- If no API key → fallback to standard extraction
- If validation FAIL → fallback to standard extraction
- If exception → fallback to standard extraction

---

#### Method 2: `_process_document_standard()` (Fallback)

**Location**: Add after `_process_document_with_production_pipeline()`

**Purpose**: Fallback to standard extraction for single document

**Implementation**: Extract existing chunking + extract_entities code from ainsert() else branch

**Why Needed**: Avoid code duplication when production pipeline falls back

---

### Phase 4: Testing

**Test 1: Backward Compatibility** (CRITICAL)
```python
# Should work exactly as before
rag = BiGRAG(working_dir="expr/demo_test")  # use_production_pipeline=False (default)
await rag.ainsert(documents, metadata)

# Verify: Graph files created same as before
# Verify: Query results same as before
```

**Test 2: Production Pipeline** (NEW)
```python
# Enable production pipeline
rag = BiGRAG(
    working_dir="expr/educational_kg",
    use_production_pipeline=True
)
await rag.ainsert(kuet_document, metadata)

# Verify: Graph files created
# Verify: Validation report shows PASS/WARNING
# Verify: Query results improved quality
```

**Test 3: Fallback Logic**
```python
# Test fallback when no API key
import os
os.environ.pop("OPENAI_API_KEY", None)

rag = BiGRAG(use_production_pipeline=True)  # Should fallback
await rag.ainsert(doc, metadata)

# Verify: Falls back to standard extraction (no crash)
```

---

### Phase 5: script_build.py Integration

**File**: `script_build.py`

**Changes Needed**:
```python
# Add CLI argument
parser.add_argument(
    "--production",
    action="store_true",
    help="Use production pipeline (table-aware, 95%+ validation, higher cost)"
)

# Pass to BiGRAG
rag = BiGRAG(
    working_dir=args.output,
    use_production_pipeline=args.production,  # NEW
    enable_llm_cache=True
)
```

**Usage**:
```bash
# Standard extraction (fast, cheap)
python script_build.py --data_source KUET

# Production extraction (slow, expensive, accurate)
python script_build.py --data_source KUET --production
```

---

## 🎯 Current State

### What Works
✅ Config flags in BiGRAG dataclass
✅ Branch logic in ainsert() (production vs standard)
✅ Standard pipeline **completely unchanged** (backward compatible)
✅ Rollback mechanism (default is standard pipeline)

### What's Missing
❌ `_process_document_with_production_pipeline()` implementation (400+ lines)
❌ `_process_document_standard()` fallback method
❌ Testing (backward compat, production mode, fallback)
❌ script_build.py --production flag

---

## 📊 Estimated Effort to Complete

| Task | Lines of Code | Time Estimate |
|------|--------------|---------------|
| Implement `_process_document_with_production_pipeline()` | ~400 | 2-3 hours |
| Implement `_process_document_standard()` | ~50 | 15 min |
| Test backward compatibility | - | 30 min |
| Test production mode | - | 30 min |
| Test fallback logic | - | 15 min |
| Add --production flag to script_build.py | ~10 | 10 min |
| **TOTAL** | **~460** | **4-5 hours** |

---

## 🚨 Critical Implementation Notes

### 1. Function Signatures (VERIFIED CORRECT)

```python
# operate.py lines 595-660
async def _merge_nodes_then_upsert(
    entity_name: str,  # ✅ Individual entity name
    nodes_data: list[dict],  # ✅ List of entity data dicts
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    # Merges multiple occurrences of same entity
    # Aggregates weights, source_ids, descriptions
    # Returns: entity_data dict with 'entity_name' key

# operate.py lines 535-592
async def _merge_relations_then_upsert(
    relation_name: str,  # ✅ Hash ID of relation
    nodes_data: list[dict],  # ✅ List of relation data dicts
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    # Merges multiple occurrences of same relation
    # Aggregates weights, source_ids
    # Returns: relation_data dict with 'relation_name' and 'relation_content' keys
```

### 2. ProductionPipeline Output Format

```python
result = await pipeline.process_document(doc_content, metadata)

# result structure:
{
    'entities': [
        {
            'entity_name': str,
            'entity_type': str,
            'description': str,
            'weight': float,
            'source_id': str,  # ProductionPipeline chunk ID
            'metadata': dict
        },
        ...
    ],
    'relations': [
        {
            'content': str,
            'completeness_score': float (0-10),
            'source_id': str,  # ProductionPipeline chunk ID
            'metadata': {
                'linked_entities': List[str],  # Entity names in this relation
                ...
            }
        },
        ...
    ],
    'chunks': [
        {
            'content': str,
            'tokens': List[int],
            'chunk_order_index': int,
            'chunk_id': str,  # ProductionPipeline chunk ID
            'type': 'table' | 'paragraph',
            'structured_data': dict (for tables)
        },
        ...
    ],
    'validation': {
        'overall_status': 'PASS' | 'WARNING' | 'FAIL',
        'numeric': {...},
        'consistency': {...}
    }
}
```

### 3. BiGRAG Storage Format (MUST MATCH)

**Entity nodes**:
```python
{
    "entity_type": str,
    "description": str,
    "weight": float,
    "source_id": str  # BiGRAG chunk ID (not ProductionPipeline chunk ID!)
}
```

**Relation nodes**:
```python
{
    "hyper_relation_content": str,  # Actual content
    "weight": float (0-10),
    "source_id": str  # BiGRAG chunk ID
}
```

**Chunks** (KV storage):
```python
{
    "content": str,
    "tokens": List[int],
    "chunk_order_index": int,
    "full_doc_id": str,
    "doc_title": str,
    "doc_metadata": dict
}
```

### 4. Chunk ID Mapping (CRITICAL)

**Problem**: ProductionPipeline creates its own chunk IDs (from table-aware chunking), but BiGRAG storage expects BiGRAG chunk IDs.

**Solution**: Create mapping dict during chunk processing:
```python
production_chunk_to_bigrag_id = {}

for prod_chunk in result['chunks']:
    # Create BiGRAG chunk ID (hash of content)
    bigrag_chunk_id = compute_mdhash_id(prod_chunk['content'], prefix='chunk-')

    # Map ProductionPipeline chunk ID → BiGRAG chunk ID
    prod_chunk_id = prod_chunk.get('chunk_id') or prod_chunk.get('source_id')
    production_chunk_to_bigrag_id[prod_chunk_id] = bigrag_chunk_id

# Then remap all source_ids in entities/relations
for entity in entities:
    old_source_id = entity.get('source_id')
    if old_source_id in production_chunk_to_bigrag_id:
        entity['source_id'] = production_chunk_to_bigrag_id[old_source_id]
```

---

## 🔄 Rollback Plan (Zero Risk)

**If production pipeline doesn't improve quality**:
1. User simply doesn't set `use_production_pipeline=True` (default is False)
2. No code changes needed - standard pipeline unchanged
3. Can delete `_process_document_with_production_pipeline()` method if desired

**To switch existing subgraphs back**:
```python
# Rebuild with standard pipeline
rag = BiGRAG(
    working_dir="expr/demo_test",
    use_production_pipeline=False  # Explicitly use old extraction
)
await rag.ainsert(documents, metadata)
```

---

## 📝 Next Steps for Developer

1. **Read**: `PRODUCTION_PIPELINE_METHOD_IMPLEMENTATION.md` (contains complete 400-line implementation)
2. **Copy**: Implementation code to `bigrag/bigrag.py` after line 490
3. **Test**: Run backward compatibility test (should pass without changes)
4. **Test**: Run production mode test with KUET document
5. **Add**: `--production` flag to `script_build.py`
6. **Compare**: Build same document with both pipelines and compare quality

---

## 🎯 Success Criteria

- [ ] Backward compatibility: Standard pipeline produces **identical** graphs
- [ ] Production mode: Validation reports show PASS or WARNING (not FAIL)
- [ ] Fallback: Graceful fallback when API key missing or validation fails
- [ ] No crashes: All error paths handled
- [ ] Documentation: Update README with --production flag usage

---

**Questions?** See detailed implementation in `PRODUCTION_PIPELINE_METHOD_IMPLEMENTATION.md`
