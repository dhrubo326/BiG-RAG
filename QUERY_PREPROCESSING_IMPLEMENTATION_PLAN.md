# Query Preprocessing Implementation Plan

**Version:** 1.2
**Date:** January 2025
**Status:** Awaiting Approval

**Revision History:**
- v1.2: Added practical implementation details:
  - Added section 4.0.1: Verify required imports (json, os, PROMPTS)
  - Added insertion points for all code additions (lines specified)
  - Added OLD vs NEW code comparison for modifications (section 4.3)
  - Added verification commands for each step (grep checks)
  - Added line 1404 reranking typo fix (section 4.4)
  - Added test execution commands with expected output (section 5.3)
  - Updated checklist with verification steps for each phase
  - Added feature flag testing commands (section 4.5)
- v1.1: Fixed 4 critical issues from first review:
  - Added Phase 0: Prerequisite typo fixes (5 locations)
  - Fixed language access: `global_config.get("default_language", PROMPTS["DEFAULT_LANGUAGE"])`
  - Fixed typo in section 4.4: `hl_keywords` not `hl_keywrds`
  - Clarified dependencies: `DEFAULT_LANGUAGE` from constants.py via PROMPTS
- v1.0: Initial plan

---

## Overview

### Problem Statement

BiG-RAG currently passes raw user queries directly to all three retrieval paths without preprocessing. This causes:

1. **Typos and grammar errors** reduce embedding quality
2. **Mixed language queries** (Banglish, code-switched) fail to retrieve correctly
3. **Question form** mismatches with declarative knowledge segments (Path B & C)
4. **Vague queries** ("who is messi") lack context for semantic matching

**Result:** Suboptimal retrieval accuracy across all three paths.

### Solution Approach

Add query preprocessing layer that:
1. **Normalizes** user queries (fix typos, grammar, translate to default language)
2. **Generates two forms:** question form for entity search, statement form for knowledge/chunk search
3. **Optimizes each path** with appropriate query type

### Proposed Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────────┐
│  User Query: "who is messi"                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Query Preprocessing (Single LLM Call)                      │
│                                                              │
│  Input:  "who is messi"                                     │
│  Output:                                                     │
│    - normalized_query: "Who is Lionel Messi?"               │
│    - statement_query: "Lionel Messi is an Argentine..."     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Three-Path Retrieval (Parallel)                            │
│                                                              │
│  Path A (Entity):    uses normalized_query                  │
│    → "Who is Lionel Messi?"                                 │
│    → Query entity vector DB                                 │
│    → Find: "LIONEL MESSI" entity                            │
│                                                              │
│  Path B (Relation):  uses statement_query                   │
│    → "Lionel Messi is an Argentine footballer..."           │
│    → Query relation vector DB                               │
│    → Find: knowledge segments about Messi                   │
│                                                              │
│  Path C (Chunk):     uses statement_query                   │
│    → "Lionel Messi is an Argentine footballer..."           │
│    → Query chunk vector DB                                  │
│    → Find: document chunks about Messi                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  RRF Fusion & Return Top-10 Context Items                   │
│    - 5 structured (Path A + B)                              │
│    - 5 chunks (Path C)                                      │
└─────────────────────────────────────────────────────────────┘
```

### Why Path-Optimized Queries?

**Path A (Entity):** Searches entity names
- Best input: Normalized question ("Who is Lionel Messi?")
- Entity names are often proper nouns that appear in questions

**Path B (Relation):** Searches knowledge segments (declarative statements)
- Best input: Statement form ("Messi is a footballer...")
- Knowledge segments are declarative, not questions

**Path C (Chunk):** Searches document text (articles, books)
- Best input: Statement form (documents contain statements, not questions)
- Better semantic alignment with document content

### Implementation Strategy

**Single LLM Call:** Generate both forms in one request
- Lower latency than separate calls
- Consistent processing
- Temperature = 0.0 for deterministic results

**Graceful Degradation:** If LLM fails, use raw query
- System continues working
- No breaking changes

**Caching (Optional - Can be added later):** Store results by query hash
- Identical queries return instantly
- Reduces cost for repeated queries
- Initial implementation can skip this for simplicity

---

## 1. Objective

Add query preprocessing to BiG-RAG's retrieval pipeline to improve accuracy across three retrieval paths.

**Current State:** Raw user query passed directly to all three paths
**Target State:** Preprocessed queries optimized for each path's semantic space

---

## 2. Architecture Decision: Path-Optimized Queries

### Analysis of RRF Consistency Concern

**External Concern:** Different queries per path hurts RRF consistency.

**Analysis:** BiG-RAG's RRF implementation (`bigrag/operate.py:1339-1453`):
```python
# Path A + B: Combined with RRF
for i, (k, source_ids) in enumerate(knowledge_list_1):  # Path A
    know_score[k] += 1/(i+1)
for i, (k, source_ids) in enumerate(knowledge_list_2):  # Path B
    know_score[k] += 1/(i+1)

# Path C: Separate weighted RRF
chunk_knowledge = [...]

# Final: CONCATENATION (not cross-path fusion)
knowledge = structured_knowledge[:5] + chunk_knowledge[:5]
```

**Result:** Path A+B fused together, Path C separate. No cross-path RRF exists. Different queries per path do NOT hurt consistency.

**Decision:** Use path-optimized queries:
- **Path A (Entity):** Normalized question (better for entity names + PRF)
- **Path B (Relation):** Statement form (better for knowledge segments)
- **Path C (Chunk):** Statement form (better for document text)

---

## 3. Implementation Approach

### 3.1 Query Preprocessing Function

**What:** Single LLM call to generate two query forms
**Why:** Simpler than entity extraction (PRF already discovers entities), lower cost
**Output:** `(normalized_query, statement_query)` tuple

**Example:**
```python
Input:  "who is messi"
Output: ("Who is Lionel Messi?",
         "Lionel Messi is an Argentine footballer who plays as a forward...")
```

---

## 4. Code Changes

### 4.0.1 Verify Required Imports

**File:** `bigrag/operate.py`

**Check these imports exist at the top of the file:**
```python
import json  # Should be at line 2 or near top (verify)
import os    # Should exist (verify, add if missing)
from .prompt import GRAPH_FIELD_SEP, PROMPTS  # Should be at line 35 (verify)
```

**Action:** If `os` is not imported, add it to the import section at the top of the file.

**Verification:**
```bash
# Check if imports exist
grep "^import json" bigrag/operate.py
grep "^import os" bigrag/operate.py
grep "from .prompt import" bigrag/operate.py
```

---

### 4.0.2 Prerequisite Typo Fixes (MUST DO FIRST)

**File:** `bigrag/operate.py`

**Problem:** Existing code has typos in variable names that will break the implementation.

**Lines to fix:**
```python
# Line 1298: Variable assignment
ll_kewwords, hl_keywrds = query[0], query[1]
# Change to:
ll_keywords, hl_keywords = query[0], query[1]
```

**Impact:** 5 locations need fixing:
1. Line 1298: `ll_kewwords, hl_keywrds = query[0], query[1]`
2. Line 1302: `_get_node_data(ll_kewwords, ...)`
3. Line 1311: `_get_edge_data(hl_keywrds, ...)`
4. Line 1331: `_get_chunk_data(ll_kewwords, ...)`
5. Line 1404: `rerank_chunks(query=ll_kewwords, ...)`

**Action:**
```bash
# Find and replace in operate.py
ll_kewwords → ll_keywords  (3 occurrences)
hl_keywrds  → hl_keywords   (2 occurrences)
```

**Why this matters:**
- Without fixing these first, the new code won't integrate properly
- These typos exist in production code right now
- Must be fixed regardless of whether preprocessing is implemented

**Verification:**
```bash
# After fixing, these should return 0 results
grep "kewwords" bigrag/operate.py
grep "keywrds" bigrag/operate.py

# Run existing tests to ensure nothing broke
cd test_scripts
python test_improvements.py  # Should pass all tests
```

---

### 4.1 Add Prompt Template

**File:** `bigrag/prompt.py`

**Insertion Point:** After line ~270 (after `PROMPTS["entity_extraction"]` definition ends)

**Add this new prompt:**

```python
PROMPTS["query_preprocessing"] = """\
---Role---
You are a query preprocessor for a multilingual knowledge graph retrieval system.

---Goal---
Given a user query, produce two forms:
1. **normalized_query**: Clean question form (fix typos, grammar, translate to {language})
2. **statement_query**: Declarative statement with expanded context

---Output Format---
Return ONLY valid JSON (no markdown):
{{
  "normalized_query": "cleaned question",
  "statement_query": "declarative statement with context"
}}

---Examples---

Example 1 (English):
Input: "who is messi"
Output:
{{
  "normalized_query": "Who is Lionel Messi?",
  "statement_query": "Lionel Messi is an Argentine professional footballer who plays as a forward and captains Inter Miami and Argentina."
}}

Example 2 (Technical):
Input: "newtons 2nd law"
Output:
{{
  "normalized_query": "What is Newton's second law of motion?",
  "statement_query": "Newton's second law states that force equals mass times acceleration (F=ma), describing the relationship between force, mass, and acceleration."
}}

Example 3 (Bangla):
Input: "নিউটনের সূত্র কি"
Output:
{{
  "normalized_query": "নিউটনের গতির সূত্র কী?",
  "statement_query": "নিউটনের গতির সূত্র পদার্থবিজ্ঞানের তিনটি মৌলিক সূত্র যা বস্তুর গতি এবং বলের সম্পর্ক বর্ণনা করে।"
}}

Example 4 (Typo):
Input: "whn was einstien born"
Output:
{{
  "normalized_query": "When was Albert Einstein born?",
  "statement_query": "Albert Einstein was born on March 14, 1879, in Ulm, Germany."
}}

---User Query---
{query}

---Important---
- Output ONLY JSON (no ``` markers)
- Both queries in {language}
- Preserve technical terms and proper nouns
"""
```

**Verification:**
```bash
# Should return 1 match
grep "query_preprocessing" bigrag/prompt.py
```

---

### 4.2 Add Preprocessing Function

**File:** `bigrag/operate.py`

**Insertion Point:** After line ~1205, before `_format_knowledge_as_string()` function

**Location:** Insert between `_insert_entities()` and `_format_knowledge_as_string()` functions.

**Why here:** Keeps query-related functions together, before the main `kg_query()` entry point.

**Add this new function:**

```python
async def preprocess_query(
    query: str,
    language: str,
    llm_func: callable,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[str, str]:
    """
    Preprocess query to generate normalized and statement forms.

    Returns:
        (normalized_query, statement_query)
    """
    # OPTIONAL: Check cache (can be implemented later)
    # if hashing_kv is not None:
    #     args_hash = compute_mdhash_id(query + language, prefix="query_preprocess-")
    #     cached_result = await hashing_kv.get_by_id(args_hash)
    #     if cached_result is not None:
    #         cached_data = json.loads(cached_result.get("return_response", "{}"))
    #         return (
    #             cached_data.get("normalized_query", query),
    #             cached_data.get("statement_query", query)
    #         )

    # Build prompt
    prompt = PROMPTS["query_preprocessing"].format(query=query, language=language)

    # Call LLM
    try:
        response = await llm_func(
            prompt,
            max_tokens=512,
            temperature=0.0,
        )

        # Parse JSON (handle potential markdown wrapping)
        response_text = response.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        result = json.loads(response_text)
        normalized_query = result.get("normalized_query", query)
        statement_query = result.get("statement_query", query)

        # OPTIONAL: Save to cache (can be implemented later)
        # if hashing_kv is not None:
        #     await hashing_kv.upsert({
        #         args_hash: {
        #             "query": query,
        #             "language": language,
        #             "return_response": json.dumps(result),
        #         }
        #     })

        logger.info(f"[Query Preprocess] Normalized: {normalized_query[:50]}...")
        logger.info(f"[Query Preprocess] Statement: {statement_query[:50]}...")

        return normalized_query, statement_query

    except Exception as e:
        logger.warning(f"[Query Preprocess] Failed: {e}. Using raw query.")
        return query, query  # Graceful degradation
```

**Verification:**
```bash
# Should return 1 match
grep "async def preprocess_query" bigrag/operate.py
```

---

### 4.3 Integrate with kg_query

**File:** `bigrag/operate.py` (modify function)

**Location:** `async def kg_query(...)` at line 1245

**Note:** `PROMPTS` is already imported in the file (line 35):
```python
from .prompt import GRAPH_FIELD_SEP, PROMPTS  # Already exists
```

**OLD CODE (lines 1257-1259 - AFTER typos are fixed):**
```python
    ll_keywords = query
    hl_keywords = query
    keywords = [ll_keywords, hl_keywords]
```

**NEW CODE (replace lines 1257-1259 with):**
```python
    # NEW: Preprocess query
    # Use default_language from config, fallback to PROMPTS dict
    language = global_config.get("default_language", PROMPTS["DEFAULT_LANGUAGE"])
    llm_func = global_config["llm_model_func"]

    normalized_query, statement_query = await preprocess_query(
        query=query,
        language=language,
        llm_func=llm_func,
        global_config=global_config,
        hashing_kv=hashing_kv,
    )

    # Path A: Use normalized query (entity names + PRF)
    ll_keywords = normalized_query

    # Path B & C: Use statement query (knowledge segments + chunks)
    hl_keywords = statement_query

    keywords = [ll_keywords, hl_keywords]
```

**What changed:**
- Deleted: Direct assignment `ll_keywords = query` and `hl_keywords = query`
- Added: Preprocessing step with two query forms

**Verification:**
```bash
# Should return 1 match
grep "await preprocess_query" bigrag/operate.py
```

### 4.4 Fix Path C Query Usage

**File:** `bigrag/operate.py`
**Location:** `async def _build_query_context(...)` around line 1331

**Current (Bug):**
```python
knowledge_list_3 = await _get_chunk_data(
    ll_kewwords,  # Wrong: uses normalized query (also has typo)
    vdb_chunks,
    text_chunks_db,
    entity_source_ids,
    edge_source_ids,
    query_param,
)
```

**Fixed:**
```python
knowledge_list_3 = await _get_chunk_data(
    hl_keywords,  # Correct: uses statement query (typo fixed)
    vdb_chunks,
    text_chunks_db,
    entity_source_ids,
    edge_source_ids,
    query_param,
)
```

**Note:** This assumes typos from section 4.0.2 are already fixed. If not, the variable will still be `ll_kewwords`/`hl_keywrds` but should be changed to `hl_keywords`.

**Additional Fix: Line 1404 (Reranking Section)**

**File:** `bigrag/operate.py`
**Location:** Inside `_build_query_context()` function, reranking section

**OLD CODE:**
```python
reranked = await rerank_chunks(
    query=ll_kewwords,  # ❌ Typo (or ll_keywords if Phase 0 done)
    ...
)
```

**NEW CODE:**
```python
reranked = await rerank_chunks(
    query=ll_keywords,  # ✅ Should use ll_keywords (not change to hl_keywords)
    ...
)
```

**Important:** Line 1404 keeps using `ll_keywords` (not `hl_keywords`) because reranking should use the normalized query, same as the direct vector search.

**Verification:**
```bash
# Line 1331 should use hl_keywords
grep -n "await _get_chunk_data" bigrag/operate.py | grep "hl_keywords"

# Line 1404 should use ll_keywords
grep -n "query=ll_keywords" bigrag/operate.py | grep "rerank_chunks"
```

---

### 4.5 Add Feature Flag (Optional)

**File:** `bigrag/operate.py` (in `kg_query` function)

**Prerequisites:** Ensure `import os` is at the top of the file (should already exist from section 4.0.1).

**Location:** Add at the very beginning of `kg_query()` function (after function signature, before preprocessing call).

**Code:**
```python
async def kg_query(...):
    # Feature flag for easy rollback
    ENABLE_PREPROCESSING = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"

    if ENABLE_PREPROCESSING:
        # Preprocess query
        language = global_config.get("default_language", PROMPTS["DEFAULT_LANGUAGE"])
        llm_func = global_config["llm_model_func"]
        normalized_query, statement_query = await preprocess_query(
            query=query,
            language=language,
            llm_func=llm_func,
            global_config=global_config,
            hashing_kv=hashing_kv,
        )
    else:
        # Fallback: use raw query for both paths
        normalized_query = query
        statement_query = query

    ll_keywords = normalized_query
    hl_keywords = statement_query
    # ... rest of function
```

**Usage:** Set `ENABLE_QUERY_PREPROCESSING=false` in `.env` to disable.

**Verification:**
```bash
# Test with flag enabled (default)
python backend/server.py --data_source SingleTopic

# Test with flag disabled
export ENABLE_QUERY_PREPROCESSING=false
python backend/server.py --data_source SingleTopic
# (On Windows: set ENABLE_QUERY_PREPROCESSING=false)
```

---

## 5. Testing Plan

### 5.1 Unit Tests

**File:** `test_scripts/test_query_preprocessing.py`

```python
import asyncio
from bigrag.operate import preprocess_query
from bigrag.llm import gpt_4o_mini_complete

async def test_english():
    n, s = await preprocess_query("who is messi", "English", gpt_4o_mini_complete, {})
    assert "?" in n
    assert len(s) > len(n)
    print(f"[OK] English: {n} | {s[:50]}...")

async def test_bangla():
    n, s = await preprocess_query("নিউটনের সূত্র কি", "Bangla", gpt_4o_mini_complete, {})
    assert "নিউটন" in n
    print(f"[OK] Bangla: {n} | {s[:50]}...")

async def test_typo():
    n, s = await preprocess_query("whn was einstien born", "English", gpt_4o_mini_complete, {})
    assert "einstein" in n.lower() and "when" in n.lower()
    print(f"[OK] Typo: {n} | {s[:50]}...")

if __name__ == "__main__":
    asyncio.run(test_english())
    asyncio.run(test_bangla())
    asyncio.run(test_typo())
```

### 5.2 Integration Test

**File:** `test_scripts/test_query_preprocessing_integration.py`

```python
import asyncio
from bigrag import BiGRAG
from bigrag.base import QueryParam

async def test_full_pipeline():
    rag = BiGRAG(working_dir="./expr/demo_test", enable_llm_cache=True)

    results = await rag.aquery(
        "who is messi",
        param=QueryParam(mode="hybrid", top_k=60, only_need_context=True)
    )

    print(f"[OK] Retrieved {len(results)} items")
    for item in results[:3]:
        print(f"  - {item['<knowledge>'][:80]}...")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
```

---

### 5.3 Running Tests

**Unit Tests:**
```bash
cd test_scripts
python test_query_preprocessing.py
```

**Expected Output:**
```
[OK] English: Who is Lionel Messi? | Lionel Messi is an Argentine professional...
[OK] Bangla: নিউটনের গতির সূত্র কী? | নিউটনের গতির সূত্র পদার্থবিজ্ঞানের...
[OK] Typo: When was Albert Einstein born? | Albert Einstein was born on March 14, 1879...
```

**Integration Test:**
```bash
cd test_scripts
python test_query_preprocessing_integration.py
```

**Expected Output:**
```
[OK] Retrieved 10 items
  - ENTITY: LIONEL MESSI - Argentine footballer who plays...
  - Messi won the 2022 FIFA World Cup with Argentina...
  - Inter Miami signed Messi in 2023...
```

---

## 6. Implementation Checklist

### Phase 0: Prerequisite Fixes (MUST DO FIRST)
- [ ] **Verify imports** (section 4.0.1):
  - [ ] Check `import json` exists
  - [ ] Check `import os` exists (add if missing)
  - [ ] Check `from .prompt import PROMPTS` exists
- [ ] **Fix typos** in `bigrag/operate.py` (section 4.0.2):
  - [ ] Line 1298: `ll_kewwords, hl_keywrds` → `ll_keywords, hl_keywords`
  - [ ] Line 1302: `ll_kewwords` → `ll_keywords`
  - [ ] Line 1311: `hl_keywrds` → `hl_keywords`
  - [ ] Line 1331: `ll_kewwords` → `ll_keywords`
  - [ ] Line 1404: `ll_kewwords` → `ll_keywords`
  - [ ] **Verify:** `grep "kewwords" bigrag/operate.py` should return 0 results
  - [ ] **Verify:** `grep "keywrds" bigrag/operate.py` should return 0 results
  - [ ] **Verify:** Run `cd test_scripts && python test_improvements.py` (should pass)

### Phase 1: Core Implementation
- [ ] **Add prompt** to `bigrag/prompt.py` (section 4.1):
  - [ ] Add `PROMPTS["query_preprocessing"]` after line ~270
  - [ ] **Verify:** `grep "query_preprocessing" bigrag/prompt.py` returns 1 match
- [ ] **Add preprocessing function** to `bigrag/operate.py` (section 4.2):
  - [ ] Add `preprocess_query()` after line ~1205
  - [ ] **Verify:** `grep "async def preprocess_query" bigrag/operate.py` returns 1 match
- [ ] **Integrate with kg_query** (section 4.3):
  - [ ] Replace lines 1257-1259 with preprocessing call
  - [ ] Use correct language: `global_config.get("default_language", PROMPTS["DEFAULT_LANGUAGE"])`
  - [ ] **Verify:** `grep "await preprocess_query" bigrag/operate.py` returns 1 match
- [ ] **Fix Path C query** (section 4.4):
  - [ ] Line 1331: Change to `hl_keywords`
  - [ ] **Verify:** Line 1331 contains `hl_keywords,`
  - [ ] **Verify:** Line 1404 still uses `ll_keywords` (not changed)
- [ ] **(Optional) Add feature flag** (section 4.5):
  - [ ] Add feature flag at start of `kg_query()`
  - [ ] Test with flag enabled/disabled

### Phase 2: Testing
- [ ] Write unit tests (`test_scripts/test_query_preprocessing.py`)
- [ ] Write integration test (`test_scripts/test_query_preprocessing_integration.py`)
- [ ] Test English queries (simple, technical, multi-hop)
- [ ] Test Bangla queries
- [ ] Test typo correction and grammar fixing
- [ ] (Optional) Verify caching works if implemented

### Phase 3: Validation
- [ ] Run on 2WikiMultiHopQA dataset (sample 50-100 queries)
- [ ] Compare retrieval quality before/after (manual spot check)
- [ ] Measure latency impact (with/without caching)
- [ ] Test multilingual support (Bangla, English)
- [ ] Check edge cases (empty query, very long query, special characters)

### Phase 4: Documentation
- [ ] Update `CLAUDE.md` with query preprocessing section
- [ ] Add example to README.md
- [ ] Document rollback procedure (feature flag)

---

## 7. Rollback Plan

If preprocessing causes issues:

**Immediate Disable:**
```bash
# In .env
ENABLE_QUERY_PREPROCESSING=false
```

**Per-Query Disable (future enhancement):**
```python
# Add to QueryParam class
@dataclass
class QueryParam:
    enable_query_preprocessing: bool = True

# Usage
results = await rag.aquery("query", param=QueryParam(enable_query_preprocessing=False))
```

---

## 8. File Summary

| File | Change Type | Description | Section |
|------|-------------|-------------|---------|
| `bigrag/operate.py` | Verify | **Phase 0:** Verify imports (json, os, PROMPTS) | 4.0.1 |
| `bigrag/operate.py` | Fix | **Phase 0:** Fix typos `ll_kewwords` → `ll_keywords`, `hl_keywrds` → `hl_keywords` (5 locations) | 4.0.2 |
| `bigrag/prompt.py` | Add | New prompt template `PROMPTS["query_preprocessing"]` (after line ~270) | 4.1 |
| `bigrag/operate.py` | Add | New function `preprocess_query()` (after line ~1205) | 4.2 |
| `bigrag/operate.py` | Modify | Update `kg_query()` to call preprocessing (replace lines 1257-1259) | 4.3 |
| `bigrag/operate.py` | Fix | Change Path C query from `ll_keywords` to `hl_keywords` (line 1331) | 4.4 |
| `bigrag/operate.py` | Fix | Fix reranking query (line 1404 - keep as `ll_keywords`) | 4.4 |
| `bigrag/operate.py` | Add (Optional) | Feature flag for rollback | 4.5 |
| `test_scripts/test_query_preprocessing.py` | Add | Unit tests | 5.1 |
| `test_scripts/test_query_preprocessing_integration.py` | Add | Integration test | 5.2 |

---

## 9. Dependencies

**No new dependencies required.** Uses existing:
- `DEFAULT_LANGUAGE` (from `bigrag/constants.py`, exposed via `PROMPTS["DEFAULT_LANGUAGE"]`)
- `default_language` (from `global_config`, optionally configurable via `.env` as `DEFAULT_LANGUAGE`)
- `llm_model_func` (from `global_config`)
- `hashing_kv` (existing cache storage, optional)
- OpenAI API (already in use)

**Note:** The code uses `global_config.get("default_language", PROMPTS["DEFAULT_LANGUAGE"])` which:
1. First tries to get user-configured language from `global_config["default_language"]`
2. Falls back to hardcoded `PROMPTS["DEFAULT_LANGUAGE"]` (which is "English" from constants.py)

---

## 10. Latency & Cost Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Added latency | 200-300ms | Single LLM call |
| Added latency (with cache) | 0ms | Instant cache hit (if caching implemented) |
| Prompt tokens | ~800 | Template + examples |
| Response tokens | ~100 | JSON output |
| Cost per query | ~$0.0003 | GPT-4o-mini pricing |

**Note:** Caching is optional and can be added later to optimize for repeated queries.

---

**Status:** Ready for implementation
**Estimated Time:** 4-6 hours implementation + 2-3 hours testing
**Next Step:** Awaiting approval to begin

---

## Important Notes for Developers

1. **Line 1404 stays as `ll_keywords`:** The reranking query should use the normalized query (same as direct vector search), NOT the statement query. Only line 1331 changes to `hl_keywords`.

2. **Phase 0 MUST be done first:** Typo fixes are prerequisites. Without them, variable names won't match and implementation will fail.

3. **Verification is critical:** Run grep commands after each step to ensure changes were applied correctly.

4. **Test execution order:** Run unit tests first, then integration tests, then validation on real dataset.
