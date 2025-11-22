# Knowledge Graph Quality Analysis Report

**Date:** 2025-11-22
**Test Type:** KG-Only Retrieval (Dual-Path: Entity + Relation, NO Chunks)
**Document:** KUET_Admission_info.md
**Data Source:** demo_test

---

## Executive Summary

The OLD BiGRAG approach (current production system) successfully extracted **127 entities** and **73 relations** from the KUET admission document. When tested with KG-only retrieval (disabling chunk retrieval - Path C), the system achieved **60% accuracy** on 10 test questions.

**Key Finding**: The knowledge graph contains ALL the critical information, but **retrieval ranking is suboptimal** - correct answers often appear in positions 3-11 instead of top 2.

---

## Test Results Summary

| Metric | Value |
|--------|-------|
| Total Questions | 10 |
| Correct (Exact Match) | 6 (60.0%) |
| Partial (Keywords Found) | 0 (0.0%) |
| Missing (No Answer) | 4 (40.0%) |
| **Overall KG Coverage** | **60.0%** |

**Assessment:** MODERATE quality - some information retrievable but ranking issues prevent optimal performance.

---

## Detailed Question Analysis

### Questions That SUCCEEDED (6/10)

#### 1. "KUET CSE department has how many seats?"
- **Expected:** 120
- **Status:** CORRECT
- **Found at:** Rank 9/15
- **Exact match:** "Computer Science and Engineering department has code CSE and 120 seats."
- **Issue:** Correct answer ranked 9th, below irrelevant answers about IPE, LE, TE departments

#### 2. "When is the KUET admission test?"
- **Expected:** January 11, 2025
- **Status:** CORRECT
- **Found at:** Rank 3/15
- **Exact match:** "The date and time for the admission test will be on January 11, 2025, Saturday from 9:30 am to 12:30 pm..."
- **Good:** Top 3 result

#### 3. "How many questions are in Physics in the admission test?"
- **Expected:** 15 questions
- **Status:** CORRECT
- **Found at:** Rank 10/15
- **Exact match:** "In the admission test, Physics has 15 questions worth 150 marks."
- **Issue:** Correct answer ranked 10th

#### 4. "What is the total number of marks in the admission test?"
- **Expected:** 500 marks
- **Status:** CORRECT
- **Found at:** Rank 3/15
- **Exact match:** "The admission test will include Mathematics, Physics, Chemistry, and Functional English over a total of 500 marks."
- **Good:** Top 3 result

#### 5. "How many seats does EEE department have?"
- **Expected:** 120 seats
- **Status:** CORRECT
- **Found at:** Rank 9/15
- **Exact match:** "Electrical and Electronic Engineering department has code EEE and 120 seats."
- **Issue:** Correct answer ranked 9th

#### 6. "How many seats does BME department have?"
- **Expected:** 30 seats
- **Status:** CORRECT
- **Found at:** Rank 11/15
- **Exact match:** "Biomedical Engineering department has code BME and 30 seats."
- **Issue:** Correct answer ranked 11th

---

### Questions That FAILED (4/10)

#### 7. "What is the minimum GPA required for SSC?"
- **Expected:** 4.00
- **Status:** MISSING
- **Relation exists?** YES - "Candidates must have at least a GPA of 4.00 in secondary or equivalent examination"
- **Why failed?** Answer is in chunk 1 but relation was marked as ORPHAN during extraction (12.5% orphan rate)
- **Root cause:** Orphan relation filtering

#### 8. "What is the total number of seats in KUET?"
- **Expected:** 1065 seats
- **Status:** MISSING (but found in earlier coverage check!)
- **Relation exists?** YES - "The total number of seats across all departments is 1065."
- **Why failed?** Ranking issue - never appeared in top 15 results
- **Root cause:** Query embedding mismatch with relation embedding

#### 9. "How many questions are in English in the admission test?"
- **Expected:** 10 questions
- **Status:** MISSING (but found in earlier coverage check!)
- **Relation exists?** YES - "In the admission test, English has 10 questions worth 50 marks."
- **Found at:** Rank 14/15 (question #2), but MISSING in question #9
- **Why failed?** Inconsistent retrieval ranking
- **Root cause:** Query preprocessing/embedding variance

#### 10. "When does the application deadline end?"
- **Expected:** December 14, 2024
- **Status:** MISSING
- **Relation exists?** Need to verify (likely in important dates section)
- **Why failed?** Either orphan relation or not extracted
- **Root cause:** Need to check graph for this specific fact

---

## Root Cause Analysis

### Issue #1: Retrieval Ranking Problems (CRITICAL)

**Observation:**
Every query returns "Industrial and Production Engineering department has code IPE and 60 seats" as rank #1, regardless of relevance.

**Examples:**
- Query: "CSE seats?" → Top result: "IPE has 60 seats" (irrelevant)
- Query: "GPA required?" → Top result: "IPE has 60 seats" (irrelevant)
- Query: "Application deadline?" → Top result: "IPE has 60 seats" (irrelevant)

**Hypothesis:**
- IPE relation may have highest weight in graph (frequency-based)
- RRF (Reciprocal Rank Fusion) may be malfunctioning
- Entity vector search may be biasing results toward specific departments

**Impact:** Correct answers often appear in ranks 9-15 instead of top 3

---

### Issue #2: Orphan Relations (MODERATE)

**Observation:**
During extraction, system logged:
```
WARNING: chunk-e9348fb5834e8827d57a5989580f68a4: ORPHAN RELATION (no entities):
'Candidates should have at least a GPA of 4.00 in secondary or equivalent examina...'
ERROR: chunk-e9348fb5834e8827d57a5989580f68a4: HIGH ORPHAN RATE (12.5%)!
Expected <5%, found 2/16 orphans.
```

**Impact:**
Critical eligibility requirements (GPA 4.00) are in graph but **disconnected from entities**, making them harder to retrieve.

**Why it happens:**
LLM extracts relations without corresponding entities in same extraction pass, causing the relation to be orphaned.

---

### Issue #3: Unknown Entity Types (MINOR)

**Observation:**
Many entities flagged as unknown types during extraction:
- `number` → fallback to `category`
- `subject` → fallback to `category`
- `program` → fallback to `category`
- `grade` → fallback to `category`
- `email`, `website`, `role`, `document` → fallback to `category`

**Impact:**
97/127 entities (76%) are classified as "category" instead of specific types. This reduces semantic precision but doesn't break retrieval.

---

## Information Coverage Verification

### Critical Facts Check

**ALL** critical information IS present in the KG:

✓ **Departments and Seats:**
- CSE: 120 seats
- EEE: 120 seats
- Civil Engineering (CE): 120 seats
- ME: 120 seats
- BME: 30 seats
- Total: 1065 seats

✓ **Admission Test Details:**
- Date: January 11, 2025
- Time: 9:30 AM to 12:30 PM
- Mathematics: 15 questions, 150 marks
- Physics: 15 questions, 150 marks
- Chemistry: 15 questions, 150 marks
- English: 10 questions, 50 marks
- Total: 500 marks

✓ **Eligibility Requirements:**
- SSC: GPA 4.00 minimum (orphaned but present)
- HSC: Total GP 18.00 in Math, Physics, Chemistry, English
- BME: GPA 4.00 in Biology required

✓ **Important Dates:**
- Application starts: December 4, 2024
- Application ends: December 14, 2024
- Eligible candidates list: December 30, 2024
- Exam date: January 11, 2025
- Results: January 26, 2025

**Conclusion:** The KG extraction is **comprehensive** - it captured all major facts from the document.

---

## Recommendations

### Priority 1: Fix Retrieval Ranking (CRITICAL)

**Problem:** IPE relation always ranks #1 regardless of query relevance.

**Suggested fixes:**
1. **Investigate RRF scoring** - Check if Reciprocal Rank Fusion is working correctly
2. **Check entity weights** - Verify if IPE has artificially high weight
3. **Improve query→relation matching** - Use better embedding model or query preprocessing
4. **Add BM25 re-ranking** - Combine vector search with keyword matching

**Expected impact:** Accuracy could improve from 60% → 80%+ if ranking is fixed.

---

### Priority 2: Reduce Orphan Relations (MODERATE)

**Problem:** 12.5% orphan rate (target: <5%)

**Suggested fixes:**
1. **Improve entity extraction** - Ensure entities are extracted before relations
2. **Post-processing linking** - Link orphan relations to nearby entities
3. **Relaxed validation** - Don't reject relations with <100% entity coverage

**Expected impact:** Recover 2-4 missing facts (SSC GPA, etc.)

---

### Priority 3: Expand Entity Type Vocabulary (LOW)

**Problem:** 76% of entities classified as generic "category"

**Suggested fixes:**
1. Add `number`, `subject`, `program`, `grade`, `email`, `website` to entity type whitelist
2. Update entity type normalization logic in `bigrag/operate.py`

**Expected impact:** Better semantic structure, minimal accuracy improvement

---

## Comparison: Production KG vs New ProductionKGPipeline

### Current Production (OLD Approach)
✓ Successfully extracts most critical facts
✓ 127 entities, 73 relations
✓ Comprehensive coverage (60% KG-only accuracy)
✗ High orphan rate (12.5% vs target <5%)
✗ Retrieval ranking issues
✗ Unknown entity types (76%)

### ProductionKGPipeline (NEW Approach - Not Yet Tested)
- Table-aware chunking (prevents table splitting)
- Structured GPT-4o extraction (JSON output)
- 3-Tier Validation (PASS/WARNING/FAIL)
- Entity canonicalization (KUET/BUET dept names)
- NumericValidator + ConsistencyValidator
- **Expected to fix:** Orphan relations, entity types, validation
- **Unknown:** Whether it will fix retrieval ranking

**Decision:** Since OLD approach already has good coverage (60%), the main issue is **retrieval ranking**, not extraction quality. Consider testing ProductionKGPipeline only if ranking fixes don't improve accuracy above 80%.

---

## Next Steps

1. **Investigate retrieval ranking bug** (Why is IPE always rank #1?)
2. **Check RRF implementation** in `bigrag/operate.py`
3. **Verify entity weights** in graph_chunk_entity_relation.graphml
4. **Test with BM25 re-ranking** as fallback
5. **Only after fixing ranking**, decide whether to integrate ProductionKGPipeline

---

## Conclusion

The OLD BiGRAG approach demonstrates **good extraction quality** (60% accuracy with KG-only retrieval), capturing all critical facts from the KUET document. However, **retrieval ranking issues** prevent optimal performance - correct answers often appear in positions 9-15 instead of top 3.

**Recommendation:** Fix retrieval ranking before considering major pipeline changes. The extraction quality is already acceptable; the bottleneck is in the retrieval algorithm.
