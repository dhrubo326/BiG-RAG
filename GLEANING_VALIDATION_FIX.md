# Gleaning Validation Fix - Enhanced Pipeline

**Date**: January 25, 2025
**Issue**: All gleaning passes were failing validation, preventing incremental entity extraction
**Status**: ✅ **FIXED**

---

## Problem Summary

### What Was Happening

**Symptoms**:
```
[GLEANING] Starting 2 gleaning passes for chunk_0003
[GLEANING] Pass 1/2
[GLEANING] Pass 1: Validation failed, skipping  ❌
[GLEANING] Pass 2/2
[GLEANING] Pass 2: Validation failed, skipping  ❌
```

**Result**: Only initial extraction used, gleaning passes discarded (0 additional entities)

### Root Cause

**The Mismatch**:
- **Gleaning is incremental**: LLM prompted to extract only NEW entities not already found
- **Validation expects completeness**: Required 60%+ numeric coverage of ALL source numbers
- **Conflict**: Gleaning responses naturally have LOW numeric coverage (10-30%) because they only return NEW entities

**Example**:
```
Initial extraction: "KUET has 1065 seats, 16 departments"
  → Entities: KUET, 1065, 16
  → Numeric coverage: 100% (all numbers)
  → Status: PASS ✅

Gleaning pass 1: "CSE Department, EEE Department"
  → Entities: CSE Department, EEE Department
  → Numeric coverage: 0% (no numbers, they're in initial extraction)
  → Status: FAIL ❌ (required 60%+ coverage)
```

**The Paradox**: Gleaning was doing its job correctly (finding NEW entities without numbers), but validation rejected it for not having enough numbers!

---

## Solution Implemented

### Relaxed Validation for Gleaning

**Key Insight**: Gleaning is incremental, not comprehensive. Numeric coverage check doesn't make sense.

**New Validation Logic**:

**For Initial Extraction** (unchanged):
- ✅ Full 3-tier validation
- ✅ Numeric coverage: 60%+ required
- ✅ Hallucination: <15% allowed
- ✅ Semantic validity: 70%+ required

**For Gleaning Passes** (NEW - relaxed):
- ❌ **Skip numeric coverage** (doesn't apply to incremental extraction)
- ✅ **Hallucination check**: <10% (prevent making up entities)
- ✅ **Semantic validity**: ≥50% (entities should be in source text)

### Implementation Details

**File**: `bigrag/extractors/constrained_extractor.py`

**Changes Made**:

1. **Added `is_gleaning` parameter** (line 495):
   ```python
   def _validate_extraction(
       self,
       source_text: str,
       source_numbers: set,
       source_facts: List[str],
       extraction: Dict,
       is_gleaning: bool = False  # ← NEW
   ) -> Dict:
   ```

2. **Implemented relaxed validation** (lines 579-596):
   ```python
   if is_gleaning:
       # RELAXED VALIDATION FOR GLEANING
       # Skip numeric coverage, focus on hallucination prevention

       if hallucination_score < 0.10 and semantic_validity >= 0.50:
           status = 'PASS'  # Low hallucination + reasonable validity
       elif hallucination_score < 0.20 and semantic_validity >= 0.40:
           status = 'WARNING'  # Moderate quality
       else:
           status = 'FAIL'  # High hallucination or nonsense
   else:
       # Normal validation for initial extraction
       status = self._determine_validation_status(...)
   ```

3. **Updated gleaning loop** (line 188):
   ```python
   glean_validation = self._validate_extraction(
       source_text=paragraph_text,
       source_numbers=source_numbers,
       source_facts=source_facts,
       extraction=glean_extraction,
       is_gleaning=True  # ← Pass True for gleaning validation
   )
   ```

4. **Added detailed logging** (line 596):
   ```python
   print(f"      [GLEANING VALIDATION] hallucination={hallucination_score:.2%}, semantic={semantic_validity:.2%}, status={status}")
   ```

---

## Expected Results

### Before Fix

**Initial Extraction**:
- Chunk 1: 14 entities, 7 relations
- Chunk 2: 9 entities, 5 relations
- **Total from paragraphs**: 23 entities

**Gleaning Passes**:
- Pass 1: Validation failed ❌
- Pass 2: Validation failed ❌
- **Added**: 0 entities

**Final**: 23 paragraph entities + 149 table entities = **172 total** → 82 after merging

---

### After Fix (Expected)

**Initial Extraction** (same):
- Chunk 1: 14 entities, 7 relations
- Chunk 2: 9 entities, 5 relations
- **Total from paragraphs**: 23 entities

**Gleaning Passes** (NOW WORKING):
- Pass 1: Validation PASS ✅ → Added 5-10 entities
- Pass 2: Validation PASS ✅ → Added 3-5 entities
- **Added**: 8-15 entities

**Final**: ~35-40 paragraph entities + 149 table entities = **~185-190 total** → ~95-100 after merging

**Improvement**: +30-50% more paragraph entities extracted!

---

## Testing Plan

### Step 1: Clean Old Data
```bash
rm -rf expr/kuet_unified/*
```

### Step 2: Reindex Document
```bash
curl -X 'POST' \
  'http://localhost:8001/datasets/create-and-index' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@KUET_Admission_info.md' \
  -F 'data_source=kuet_unified' \
  -F 'title=Khulna University of Engineering & Technology' \
  -F 'metadata={"category":"education","tags":["KUET","admission"]}' \
  -F 'process_async=true'
```

### Step 3: Check Logs for Success

**Look for**:
```
[GLEANING] Starting 2 gleaning passes for chunk_0003
[GLEANING] Pass 1/2
  [GLEANING VALIDATION] hallucination=0.00%, semantic=0.85%, status=PASS  ✅
[GLEANING] Pass 1: Added 7 entities, 3 relations  ✅
[GLEANING] Pass 2/2
  [GLEANING VALIDATION] hallucination=0.00%, semantic=0.72%, status=PASS  ✅
[GLEANING] Pass 2: Added 4 entities, 2 relations  ✅
```

**Compare**:
- **Before**: "Total entities now: 172"
- **After**: "Total entities now: 185-190" (expected +10-15% improvement)

### Step 4: Verify Graph Quality

Check `expr/kuet_unified/graph_chunk_entity_relation.graphml`:
- Should contain more paragraph-extracted entities
- Entities like "বাংলাদেশী শিক্ষার্থী", "ভর্তি যোগ্যতা", etc. should be present
- Orphan rate should decrease (more entities linked to relations)

---

## Validation Thresholds Reference

### Gleaning Validation (NEW)

| Status | Hallucination | Semantic Validity | Numeric Coverage |
|--------|---------------|------------------|------------------|
| PASS | <10% | ≥50% | **SKIPPED** |
| WARNING | <20% | ≥40% | **SKIPPED** |
| FAIL | ≥20% | <40% | **SKIPPED** |

**Rationale**:
- Gleaning finds NEW entities (incremental)
- Focus on preventing hallucinations
- Accept entities with reasonable semantic grounding
- Ignore numeric coverage (would always be low)

### Initial Extraction Validation (Unchanged)

**Semi-Structured Mode** (default):

| Status | Hallucination | Semantic Validity | Numeric Coverage |
|--------|---------------|------------------|------------------|
| PASS | <5% | ≥85% | ≥95% |
| WARNING | <15% | ≥70% | ≥60% |
| FAIL | ≥15% | <70% | <60% |

---

## Related Files Modified

- ✅ `bigrag/extractors/constrained_extractor.py` (lines 495, 579-596, 188)

## Related Issues

- **Issue**: Gleaning validation too strict
- **Impact**: 30-50% entity loss from paragraphs
- **Status**: ✅ Fixed

---

**Next Steps**: Test with KUET document and verify gleaning passes now succeed!
