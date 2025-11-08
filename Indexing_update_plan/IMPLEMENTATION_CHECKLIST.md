# BiG-RAG Improvement Implementation Checklist
**Quick Reference for Developers**

**Based on**: [UNIFIED_IMPROVEMENT_PLAN.md](UNIFIED_IMPROVEMENT_PLAN.md)
**Start Date**: TBD
**Target Completion**: 4-5 days

---

## Priority 1: Must-Have Fixes (3 days)

### Day 1-2: A1 - Hash-Based Node IDs

#### Code Changes
- [ ] **bigrag/operate.py:151** - Generate hash ID
  ```python
  from .utils import compute_mdhash_id
  edge_id = compute_mdhash_id(knowledge_fragment, prefix="rel-")
  return dict(
      hyper_relation=edge_id,
      hyper_relation_content=knowledge_fragment,
      ...
  )
  ```

- [ ] **bigrag/operate.py:157-188** - Update `_merge_bipartite_edges_then_upsert()`
  ```python
  async def _merge_bipartite_edges_then_upsert(
      bipartite_edge_id: str,      # ← Now hash
      bipartite_edge_content: str,  # ← NEW param
      nodes_data: list[dict],
      ...
  ):
      node_data = {
          "content": bipartite_edge_content,  # ← Store content
          "role": "bipartite_edge",
          ...
      }
  ```

- [ ] **bigrag/operate.py:~400** - Update callers
  ```python
  edge_id = group_data[0]["hyper_relation"]
  edge_content = group_data[0]["hyper_relation_content"]
  await _merge_bipartite_edges_then_upsert(edge_id, edge_content, ...)
  ```

- [ ] **bigrag/operate.py:518-520** - Remove redundant hashing in VDB
  ```python
  data_for_vdb = {
      dp["bipartite_edge_name"]: {  # Already hash now
          "content": dp["bipartite_edge_content"],
          ...
      }
  }
  ```

- [ ] **bigrag/storage.py:246-249** - Skip uppercase for hash IDs
  ```python
  if node.startswith("rel-") or node.startswith("ent-") or node.startswith("chunk-"):
      node_mapping[node] = node  # Keep as-is
  else:
      node_mapping[node] = html.unescape(node.upper().strip())
  ```

#### Testing
- [ ] Unit test: Hash ID generation
- [ ] Unit test: Hash ID consistency (same content → same ID)
- [ ] Integration test: Full graph construction
- [ ] Validation: File size reduction (expect 30-40%)
- [ ] Validation: Performance benchmark (expect 5-10x)

#### Documentation
- [ ] Update CLAUDE.md (breaking change note)
- [ ] Write migration guide
- [ ] Create version check script
- [ ] Update README.md

**Estimated Time**: 16 hours (2 days)

---

### Day 3: A2 + B3 - Type Validation + Retry Wrapper

#### A2: Entity Type Validation

**Code Changes**:
- [ ] **bigrag/operate.py** - Add type normalization
  ```python
  TYPE_NORMALIZATION_MAP = {
      "TEAM": "organization",
      "LEAGUE": "organization",
      "PLAYER": "person",
      "STATISTIC": "category",
      "CONCEPT": "category",
      # ... full mapping
  }

  def normalize_entity_type(extracted_type: str, allowed_types: list = None) -> str:
      normalized = extracted_type.strip().upper()
      if normalized in TYPE_NORMALIZATION_MAP:
          return TYPE_NORMALIZATION_MAP[normalized]
      # Check if already valid
      for allowed in allowed_types:
          if normalized == allowed.upper():
              return allowed
      logger.warning(f"Unknown type: {extracted_type}, using 'category'")
      return "category"
  ```

- [ ] **bigrag/operate.py:~120-130** - Update `_pack_single_entity()`
  ```python
  raw_entity_type = clean_str(record_attributes[2])
  entity_type = normalize_entity_type(raw_entity_type)  # ← NEW
  ```

**Testing**:
- [ ] Unit test: Type normalization
- [ ] Unit test: Unknown type fallback
- [ ] Integration test: Entity types are consistent

**Estimated Time**: 4 hours

---

#### B3: Retry Wrapper

**Code Changes**:
- [ ] **bigrag/utils.py** - Add retry wrapper
  ```python
  import asyncio
  import logging

  logger = logging.getLogger(__name__)

  async def safe_operation_with_retry(
      operation: Callable,
      operation_name: str,
      context: str = "",
      max_retries: int = 3,
      retry_delay: float = 0.2,
  ) -> Any:
      for attempt in range(max_retries):
          try:
              return await operation()
          except Exception as e:
              if attempt >= max_retries - 1:
                  raise Exception(f"{operation_name} failed after {max_retries} attempts: {e}")
              wait = retry_delay * (2 ** attempt)
              logger.warning(f"{operation_name} attempt {attempt+1} failed, retry in {wait:.1f}s")
              await asyncio.sleep(wait)
  ```

- [ ] **bigrag/operate.py** - Wrap VDB operations
  ```python
  from .utils import safe_operation_with_retry

  await safe_operation_with_retry(
      lambda: vdb_entities.upsert(data_for_vdb),
      "VDB upsert entities",
      context=f"{len(data_for_vdb)} entities",
      max_retries=global_config.get("api_retry_attempts", 3),
  )
  ```

**Testing**:
- [ ] Unit test: Retry on failure
- [ ] Unit test: Exponential backoff
- [ ] Integration test: VDB operations retry

**Estimated Time**: 4 hours

**Day 3 Total**: 8 hours

---

## Priority 2: Quality Improvements (2 days)

### Day 4: B1 + B2 - Prompts + Constants

#### B1: Improve Prompts

**Code Changes**:
- [ ] **bigrag/prompt.py:13-79** - Restructure prompt
  ```python
  PROMPTS["entity_extraction"] = """---Role---
  You are a Knowledge Graph Specialist responsible for extracting entities and knowledge segments.

  ---Instructions---
  1. **Knowledge Segment Extraction:**
     * Divide text into complete knowledge segments
     * Format: ("bipartite_edge"{tuple_delimiter}<segment>{tuple_delimiter}<completeness_score>)

  2. **Entity Extraction:**
     * Entity types MUST be one of: {entity_types}
     * If no type applies, classify as "Other"
     * Format: ("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<description>{tuple_delimiter}<score>)

  ---Examples---
  [3 diverse examples]

  ---Real Data---
  {input_text}
  """
  ```

- [ ] **bigrag/prompt.py:47-79** - Add 2 more examples
  - Example 2: Financial/business domain
  - Example 3: Sports/events domain

**Testing**:
- [ ] Extraction quality test (compare before/after)
- [ ] Format error rate (should decrease)

**Estimated Time**: 4 hours

---

#### B2: Create Constants File

**Code Changes**:
- [ ] **Create bigrag/constants.py**
  ```python
  """Code-level constants for BiG-RAG extraction and indexing."""

  # Extraction Settings
  DEFAULT_CHUNK_SIZE = 1200
  DEFAULT_CHUNK_OVERLAP = 100
  DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]
  DEFAULT_MAX_ENTITY_TOKENS = 6000
  DEFAULT_MAX_RELATION_TOKENS = 8000

  # Graph Settings
  GRAPH_FIELD_SEP = "<SEP>"
  BIPARTITE_EDGE_PREFIX = "rel-"
  ENTITY_PREFIX = "ent-"
  CHUNK_PREFIX = "chunk-"

  # Retrieval Settings
  DEFAULT_TOP_K_ENTITIES = 60
  DEFAULT_TOP_K_RELATIONS = 60
  DEFAULT_TOP_K_CHUNKS = 10
  ```

- [ ] **bigrag/operate.py** - Import from constants
  ```python
  from .constants import (
      DEFAULT_CHUNK_SIZE,
      GRAPH_FIELD_SEP,
      BIPARTITE_EDGE_PREFIX,
  )
  ```

- [ ] **bigrag/prompt.py** - Import from constants
  ```python
  from .constants import DEFAULT_ENTITY_TYPES
  ```

**Testing**:
- [ ] Import verification
- [ ] No hardcoded values remain

**Estimated Time**: 3 hours

**Day 4 Total**: 7 hours

---

### Day 5: B4 + A3 - Logging + Documentation

#### B4: Rotating File Handler

**Code Changes**:
- [ ] **bigrag/utils.py** - Add logging setup
  ```python
  import logging
  import logging.handlers
  from pathlib import Path

  def setup_bigrag_logger(
      logger_name: str = "bigrag",
      level: str = "INFO",
      log_dir: str = None,
  ):
      logger = logging.getLogger(logger_name)
      logger.setLevel(level)
      logger.handlers = []

      # Console handler
      console = logging.StreamHandler()
      console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
      logger.addHandler(console)

      # Rotating file handler
      if log_dir:
          Path(log_dir).mkdir(parents=True, exist_ok=True)
          file_handler = logging.handlers.RotatingFileHandler(
              filename=f"{log_dir}/bigrag.log",
              maxBytes=10 * 1024 * 1024,  # 10MB
              backupCount=5,
          )
          file_handler.setFormatter(
              logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
          )
          logger.addHandler(file_handler)

      return logger
  ```

- [ ] **bigrag/bigrag.py** - Use setup function
  ```python
  from .utils import setup_bigrag_logger
  from .config import config

  logger = setup_bigrag_logger(
      level=config.log_level,
      log_dir=config.log_dir
  )
  ```

**Testing**:
- [ ] Logs written to file
- [ ] Rotation at 10MB
- [ ] 5 backup files maintained

**Estimated Time**: 2 hours

---

#### A3: Weight Documentation

**Documentation Updates**:
- [ ] **CLAUDE.md** - Add Weight Semantics section
  ```markdown
  ## Weight Semantics

  ### Entity Weights
  - Calculation: Σ(llm_importance_score) across all occurrences
  - Range: 0 to N×100 (N = number of chunks)
  - Interpretation: 400+ (very central), 200-399 (important), ...

  ### Relation Weights
  - Calculation: Σ(completeness_score) across all occurrences
  - Range: 0 to N×10 (N = number of chunks)
  - Interpretation: 20+ (very important), 10-19 (important), ...
  ```

- [ ] **bigrag/operate.py** - Add docstrings
  - `_merge_nodes_then_upsert()` - Explain entity weight calculation
  - `_merge_bipartite_edges_then_upsert()` - Explain relation weight calculation

- [ ] **README.md** - Add FAQ
  ```markdown
  ### What do weight values mean?
  Entity weights: Sum of importance scores (0-100) × occurrences
  Relation weights: Sum of completeness scores (0-10) × occurrences
  ```

- [ ] **Create test_scripts/validate_weight_semantics.py**
  - Validate weight = sum of occurrence weights
  - Check weight ranges match source count

**Estimated Time**: 3 hours

**Day 5 Total**: 5 hours

---

## Testing Checklist

### After Each Change
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] No regressions

### Before Final Commit
- [ ] All tests pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

### Validation
- [ ] GraphML file size reduced (A1)
- [ ] Query performance improved (A1)
- [ ] Entity types consistent (A2)
- [ ] Retry mechanism works (B3)
- [ ] Prompts improve extraction (B1)
- [ ] Constants centralized (B2)
- [ ] Logs rotate (B4)
- [ ] Weight docs clear (A3)

---

## Quick Command Reference

### Run Tests
```bash
cd test_scripts
python test_improvements.py
```

### Rebuild Graph (after A1)
```bash
python script_build.py --data_source demo_test
```

### Check File Size Reduction (A1)
```bash
python test_scripts/validate_file_size_reduction.py
```

### Validate Entity Types (A2)
```bash
python test_scripts/analyze_entity_types.py
```

### Validate Weights (A3)
```bash
python test_scripts/validate_weight_semantics.py
```

---

## Progress Tracking

### Priority 1 (Must-Have)
- [ ] Day 1-2: Hash-based IDs (A1)
- [ ] Day 3: Type validation (A2)
- [ ] Day 3: Retry wrapper (B3)

### Priority 2 (Should-Have)
- [ ] Day 4: Improved prompts (B1)
- [ ] Day 4: Constants file (B2)
- [ ] Day 5: Rotating logs (B4)
- [ ] Day 5: Weight documentation (A3)

### Optional (Nice-to-Have)
- [ ] Semaphore control (B5) - Only if needed
- [ ] Map-reduce (B6) - Only if needed

---

## Completion Criteria

### All Priority 1 Done
- ✅ GraphML files 30-40% smaller
- ✅ Queries 5-10x faster
- ✅ Entity types validated
- ✅ VDB operations retry on failure

### All Priority 2 Done
- ✅ Prompts have 3 examples
- ✅ Constants centralized
- ✅ Logs rotate at 10MB
- ✅ Weight semantics documented

**Total Estimated Time**: 38-42 hours (4-5 days)

---

## Emergency Rollback

### If A1 (Hash IDs) Fails
```bash
git revert <commit-hash>
# Use backup graphs
python -m bigrag --version  # Check version
```

### If Other Changes Fail
All other changes are non-breaking, just revert the commit.

---

**Last Updated**: 2025-01-08
**Status**: Ready to Start
**Next**: Begin with A1 (Hash IDs) for maximum impact
