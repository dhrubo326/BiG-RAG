# Testing Checklist for EnhancedPipeline + PipelineFeatures

**Target Branch**: `feature/production-kg-educational`
**Goal**: Verify PipelineFeatures integration works correctly
**Time**: 30-45 minutes

---

## Prerequisites

Before testing, ensure:

```bash
# 1. On correct branch
git checkout feature/production-kg-educational
git pull origin feature/production-kg-educational

# 2. Changes applied
# - CHANGES_NEEDED.md changes applied to enhanced_pipeline.py
# - ENDPOINT_GUIDE.md endpoint created
# - Router registered in server.py

# 3. Environment ready
# venv activated (Windows)
venv\Scripts\activate

# OpenAI API key set
echo %OPENAI_API_KEY%  # Should not be empty
```

---

## Test Suite Overview

| Test Level | Tests | Time | Critical |
|-----------|-------|------|----------|
| Unit Tests | 4 tests | 5 min | YES |
| Integration Tests | 3 tests | 10 min | YES |
| Graph Quality Tests | 5 checks | 15 min | YES |
| Query Tests | 3 queries | 10 min | NO |

**Total**: 15 tests, ~40 minutes

---

## Unit Tests: PipelineFeatures Interface

### Test 1.1: Legacy API Compatibility

**Purpose**: Verify old code still works without PipelineFeatures.

**Code**:
```python
# test_scripts/test_enhanced_pipeline_legacy.py
from bigrag.enhanced_pipeline import EnhancedKGPipeline

pipeline = EnhancedKGPipeline(
    api_key="your-key",
    validation_level="MODERATE",
    enable_entity_linking=True,
    entity_merge_strategy="fuzzy"
)

print("[OK] Legacy API works")
```

**Run**:
```bash
cd test_scripts
python test_enhanced_pipeline_legacy.py
```

**Expected Output**:
```
[OK] Legacy API works
```

**If Fails**: Check EnhancedPipeline.__init__ - ensure legacy mode (else block) is correct.

---

### Test 1.2: PipelineFeatures API (Standard Preset)

**Purpose**: Verify PipelineFeatures.from_preset() works.

**Code**:
```python
# test_scripts/test_enhanced_pipeline_features.py
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures
import os

features = PipelineFeatures.from_preset("standard", openai_api_key=os.getenv('OPENAI_API_KEY'))
pipeline = EnhancedKGPipeline(features=features)

assert pipeline.features is not None, "Features should be stored"
assert pipeline.api_key == os.getenv('OPENAI_API_KEY'), "API key should match"
assert pipeline.enable_entity_linking == True, "Standard should have entity linking"
assert pipeline.entity_merge_strategy == "basic", "Standard should use basic merge"

print("[OK] Standard preset works")
```

**Run**:
```bash
python test_enhanced_pipeline_features.py
```

**Expected Output**:
```
[OK] Standard preset works
```

**If Fails**: Check PipelineFeatures mapping in EnhancedPipeline.__init__ (if features: block).

---

### Test 1.3: PipelineFeatures API (Quality Preset)

**Purpose**: Verify quality preset enables advanced features.

**Code**:
```python
# test_scripts/test_enhanced_pipeline_quality.py
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures
import os

features = PipelineFeatures.from_preset("quality", openai_api_key=os.getenv('OPENAI_API_KEY'))
pipeline = EnhancedKGPipeline(features=features)

assert pipeline.features.enable_table_detection == True, "Quality should have table detection"
assert pipeline.features.enable_gleaning == True, "Quality should have gleaning"
assert pipeline.features.enable_entity_validation == True, "Quality should have validation"
assert pipeline.entity_merge_strategy == "fuzzy", "Quality should use fuzzy merge"
assert pipeline.extraction_strategy == "gleaning", "Should map to gleaning strategy"

print("[OK] Quality preset works")
```

**Run**:
```bash
python test_enhanced_pipeline_quality.py
```

**Expected Output**:
```
[OK] Quality preset works
```

**If Fails**: Check extraction_strategy mapping logic in EnhancedPipeline.__init__.

---

### Test 1.4: Custom Feature Mix

**Purpose**: Verify custom feature combinations work.

**Code**:
```python
# test_scripts/test_enhanced_pipeline_custom.py
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures
import os

features = PipelineFeatures(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    enable_table_detection=True,
    enable_gleaning=False,
    enable_entity_validation=True,
    enable_entity_merging=True,
    merge_strategy="fuzzy",
    chunk_size=1500,
    chunk_overlap=150
)

pipeline = EnhancedKGPipeline(features=features)

assert pipeline.features.enable_table_detection == True
assert pipeline.features.enable_gleaning == False
assert pipeline.features.chunk_size == 1500
assert pipeline.entity_merge_strategy == "fuzzy"

print("[OK] Custom features work")
```

**Run**:
```bash
python test_enhanced_pipeline_custom.py
```

**Expected Output**:
```
[OK] Custom features work
```

**If Fails**: Check that all 15 feature flags are mapped correctly.

---

## Integration Tests: Full Document Processing

### Test 2.1: Process Document (Standard Preset)

**Purpose**: Verify full pipeline execution with standard preset.

**Code**:
```python
# test_scripts/test_full_pipeline_standard.py
import asyncio
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures
import os

async def test_standard():
    features = PipelineFeatures.from_preset("standard", openai_api_key=os.getenv('OPENAI_API_KEY'))
    pipeline = EnhancedKGPipeline(features=features)

    result = await pipeline.process_document(
        "KUET offers 1065 seats across 18 departments.",
        metadata={"title": "Test Doc"}
    )

    entities = result.get('entities', [])
    relations = result.get('relations', [])
    chunks = result.get('chunks', [])

    print(f"[RESULT] Entities: {len(entities)}, Relations: {len(relations)}, Chunks: {len(chunks)}")

    assert len(entities) > 0, "Should extract at least 1 entity"
    assert len(relations) > 0, "Should extract at least 1 relation"
    assert len(chunks) > 0, "Should create at least 1 chunk"

    # Check entity linking
    has_linked = any('linked_entities' in r.get('metadata', {}) for r in relations)
    assert has_linked, "Relations should have linked_entities"

    print("[OK] Standard pipeline works")

asyncio.run(test_standard())
```

**Run**:
```bash
python test_full_pipeline_standard.py
```

**Expected Output**:
```
[RESULT] Entities: 3-5, Relations: 2-3, Chunks: 1
[OK] Standard pipeline works
```

**If Fails**:
- Check chunker call (should use features.chunk_size/overlap)
- Check entity linking happens POST-merge (line 523-540)

---

### Test 2.2: Process Document (Quality Preset)

**Purpose**: Verify quality preset extracts more entities/relations.

**Code**:
```python
# test_scripts/test_full_pipeline_quality.py
import asyncio
from bigrag.enhanced_pipeline import EnhancedKGPipeline
from bigrag.pipeline.features import PipelineFeatures
import os

async def test_quality():
    features = PipelineFeatures.from_preset("quality", openai_api_key=os.getenv('OPENAI_API_KEY'))
    pipeline = EnhancedKGPipeline(features=features)

    result = await pipeline.process_document(
        "KUET offers 1065 seats across 18 departments including CSE, EEE, and ME.",
        metadata={"title": "Test Doc"}
    )

    entities = result.get('entities', [])
    relations = result.get('relations', [])

    print(f"[RESULT] Entities: {len(entities)}, Relations: {len(relations)}")

    # Quality should extract more (gleaning + validation)
    assert len(entities) >= 5, "Quality should extract 5+ entities"
    assert len(relations) >= 2, "Quality should extract 2+ relations"

    # Check orphan count
    orphan_count = sum(1 for r in relations if not r.get('metadata', {}).get('linked_entities'))
    orphan_rate = orphan_count / len(relations) if relations else 0

    print(f"[ORPHAN] {orphan_count}/{len(relations)} ({orphan_rate*100:.1f}%)")
    assert orphan_rate < 0.20, "Orphan rate should be <20%"

    print("[OK] Quality pipeline works")

asyncio.run(test_quality())
```

**Run**:
```bash
python test_full_pipeline_quality.py
```

**Expected Output**:
```
[RESULT] Entities: 5-8, Relations: 3-5
[ORPHAN] 0-1/5 (0-20%)
[OK] Quality pipeline works
```

**If Fails**:
- High orphan rate → Entity linking timing issue (check line 523-540)
- Low entity count → Gleaning not working (check extraction_strategy mapping)

---

### Test 2.3: Backend Endpoint Integration

**Purpose**: Verify `/indexing/index-document` endpoint works end-to-end.

**Setup**:
```bash
# Terminal 1: Start server
cd backend
python server.py --data_source test_endpoint
```

**Test**:
```bash
# Terminal 2: Create test file
echo "KUET offers 1065 seats across 18 departments." > test.md

# Call endpoint (basic indexing)
curl -X POST "http://localhost:8001/indexing/index-document" \
  -F "file=@test.md" \
  -F "data_source=test_endpoint" \
  -F "title=Test Document"
```

**Expected Response**:
```json
{
  "success": true,
  "message": "Document queued",
  "dataset_name": "test_endpoint",
  "document_id": "doc-abc123...",
  "job_id": "job-xyz789...",
  "filename": "test.md",
  "title": "Test Document",
  "content_preview": "KUET offers 1065 seats...",
  "content_length": 48,
  "status": "processing",
  "upload_date": "2025-01-28T...",
  "features_enabled": {
    "table_extraction": false,
    "dynamic_chunking": false,
    "gleaning": false,
    "merge_strategy": "basic"
  },
  "estimated_time": "30s",
  "estimated_cost": "$0.050"
}
```

**Verification**:
```bash
# Check graph created
ls expr/test_endpoint/graph_chunk_entity_relation.graphml
# Should exist

# Check vector DBs
ls expr/test_endpoint/vdb_*.json
# Should have 3 files (entities, relations, chunks)
```

**If Fails**:
- 500 error → Check backend logs (api.log)
- No graph created → Check BiGRAG.ainsert() is called
- Empty graph → Check pipeline results are passed correctly

---

## Graph Quality Tests: KUET Document

### Test 3.1: Build Full KUET Graph (Quality Preset)

**Purpose**: Verify graph quality with real document.

**Setup**:
```bash
# Create test file
cp datasets/SingleTopic/raw/KUET_Admission_info.md test_kuet.md
```

**Index**:
```bash
curl -X POST "http://localhost:8001/indexing/index-document" \
  -F "file=@test_kuet.md" \
  -F "data_source=kuet_quality_test" \
  -F "title=KUET Admission Info" \
  -F "need_table_extraction=true" \
  -F "need_gleaning=true" \
  -F "need_table_fact_extraction=true" \
  -F "need_numeric_validation=true" \
  -F "need_semantic_validation=true" \
  -F "merge_strategy=fuzzy" \
  -F "enable_orphan_linking=true" \
  -F "process_async=false"
```

**Wait**: ~2-3 minutes (quality preset is slower).

---

### Test 3.2: Verify Graph Size

**Check**:
```bash
# Count lines in GraphML
wc -l expr/kuet_quality_test/graph_chunk_entity_relation.graphml
# Or on Windows:
type expr\kuet_quality_test\graph_chunk_entity_relation.graphml | find /c /v ""
```

**Expected**: 1800-2200 lines

**Acceptable Range**: 1500-2500 lines

**If <1500**: Something is broken (entity linking issue).

**If >2500**: Possible duplicate nodes (check merging logic).

---

### Test 3.3: Verify Node Counts

**Check**:
```bash
# Count entity nodes
grep '<node id="entity-' expr/kuet_quality_test/graph_chunk_entity_relation.graphml | wc -l
# Windows: find /c "entity-" expr\kuet_quality_test\graph_chunk_entity_relation.graphml

# Count relation nodes
grep '<node id="rel-' expr/kuet_quality_test/graph_chunk_entity_relation.graphml | wc -l
# Windows: find /c "rel-" expr\kuet_quality_test\graph_chunk_entity_relation.graphml
```

**Expected**:
- Entities: 80-100
- Relations: 60-90

**Acceptable Range**:
- Entities: 60-120
- Relations: 40-100

**If Too Low**: Extraction not working (check gleaning is enabled).

**If Too High**: No merging happening (check merge_strategy).

---

### Test 3.4: Verify Edge Count

**Check**:
```bash
# Count edges
grep '<edge source="rel-' expr/kuet_quality_test/graph_chunk_entity_relation.graphml | wc -l
# Windows: find /c "<edge source=" expr\kuet_quality_test\graph_chunk_entity_relation.graphml
```

**Expected**: 40-60 edges

**Acceptable Range**: 30-80 edges

**Critical Check**:
```bash
# Calculate edge/relation ratio
# Edges should be ~0.5-1.0x relations count
# Example: 50 edges / 70 relations = 0.71 (GOOD)
#          5 edges / 70 relations = 0.07 (BAD - linking broken!)
```

**If <20 edges**: Entity linking is broken (check POST-merge timing).

**If edges > relations**: Duplicate edges (bug in graph builder).

---

### Test 3.5: Verify Orphan Rate

**Check**:
```bash
# Search for orphan warnings in logs
grep "ORPHAN RELATION" backend/api.log
```

**Expected**: 2-3 orphan warnings (5-10% orphan rate)

**Acceptable**: <10 orphan warnings (<20% orphan rate)

**Critical**: >20 orphan warnings (>50% orphan rate) = BROKEN

**Analysis**:
```bash
# Count orphan relations
ORPHAN_COUNT=$(grep -c "ORPHAN RELATION" backend/api.log)

# Count total relations
RELATION_COUNT=$(grep -c '<node id="rel-' expr/kuet_quality_test/graph_chunk_entity_relation.graphml)

# Calculate rate
ORPHAN_RATE=$((ORPHAN_COUNT * 100 / RELATION_COUNT))

echo "Orphan Rate: $ORPHAN_RATE%"
# Should be <20%
```

**If >50%**: Entity linking timing bug (check line 523-540 in enhanced_pipeline.py).

---

## Query Tests: Verify Retrieval Works

### Test 4.1: Basic Query

**Purpose**: Verify graph is queryable.

**Query**:
```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["How many seats does KUET offer?"],
    "dataset": "kuet_quality_test"
  }'
```

**Expected Response**:
```json
{
  "results": [
    {
      "content": "KUET offers 1065 seats...",
      "score": 0.85,
      "source": "chunk-abc123..."
    },
    ...
  ]
}
```

**If No Results**: Graph not loaded (restart server with correct dataset).

**If Low Scores (<0.5)**: Embeddings not matching (check vdb_entities.json is populated).

---

### Test 4.2: Entity-Based Query

**Purpose**: Verify entity retrieval (Path A).

**Query**:
```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["CSE department"],
    "dataset": "kuet_quality_test",
    "mode": "local"
  }'
```

**Expected**: Results mentioning "CSE" or "Computer Science".

**If No Results**: Entity extraction failed OR vdb_entities.json empty.

---

### Test 4.3: Relation-Based Query

**Purpose**: Verify relation retrieval (Path B).

**Query**:
```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["admission requirements"],
    "dataset": "kuet_quality_test",
    "mode": "global"
  }'
```

**Expected**: Results with admission-related relations.

**If No Results**: Relation extraction failed OR vdb_relations.json empty.

---

## Summary Checklist

### Before Declaring Success

- [ ] All 4 unit tests pass
- [ ] All 3 integration tests pass
- [ ] Graph size: 1800-2200 lines
- [ ] Entity count: 80-100
- [ ] Relation count: 60-90
- [ ] Edge count: 40-60
- [ ] Orphan rate: <20%
- [ ] Basic query returns results
- [ ] Entity query returns results
- [ ] Relation query returns results

### If Any Test Fails

**Failure Category → Action**:

| Failure | Likely Cause | Fix Location |
|---------|-------------|--------------|
| Unit test 1.1 fails | Legacy mode broken | EnhancedPipeline.__init__ else block |
| Unit test 1.2-1.4 fails | Feature mapping broken | EnhancedPipeline.__init__ if features block |
| Integration test orphan rate >50% | Entity linking timing | EnhancedPipeline line 523-540 |
| Graph size <1500 lines | No edges created | Entity linking not happening |
| Edge count <20 | Entity linking broken | POST-merge timing issue |
| Query returns no results | Graph not loaded | Restart server with correct dataset |

---

## Final Verification Command

```bash
# Run all tests at once
cd test_scripts
python test_enhanced_pipeline_legacy.py && \
python test_enhanced_pipeline_features.py && \
python test_enhanced_pipeline_quality.py && \
python test_enhanced_pipeline_custom.py && \
python test_full_pipeline_standard.py && \
python test_full_pipeline_quality.py

# If all pass:
echo "[SUCCESS] All tests passed! EnhancedPipeline + PipelineFeatures is working."

# If any fail:
echo "[FAIL] Check failure category above and apply fixes."
```

---

## Next Steps After Testing

1. ✅ All tests pass → Commit changes
2. ✅ Graph quality good → Delete base_pipeline.py (obsolete)
3. ✅ Endpoint works → Update frontend to use new endpoint
4. ✅ Production ready → Rebuild all datasets with quality preset

---

**Total Time**: ~40 minutes
**Critical Tests**: 10/15 (unit + integration + graph quality)
**Optional Tests**: 5/15 (query tests - nice to have)
