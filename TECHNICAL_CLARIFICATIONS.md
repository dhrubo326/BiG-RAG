# Technical Clarifications for Enhanced Pipeline Redesign

**Date:** November 25, 2025
**Status:** Final clarifications before implementation

This document addresses critical ambiguities identified in the Enhanced Pipeline redesign plan. All implementation teams must read this before starting work.

---

## Issue 1: Semantic Chunking Accumulation Logic

### The Ambiguity

**Original spec:** "keep paragraphs under 1300 tokens intact (30% overflow tolerance)"

**Question:** When accumulating multiple small paragraphs, does 1300 apply per-paragraph or per-chunk?

**Scenario:**
```
Para 1: 400 tokens
Para 2: 350 tokens
Para 3: 420 tokens
Total so far: 1170 tokens (< 1300, OK?)

Now add Para 4: 300 tokens
New total: 1470 tokens (> 1300, exceeds tolerance!)
```

### Clarified Behavior

**Target chunk size:** 1000 tokens
**Tolerance limit:** 1300 tokens (30% overflow)
**Accumulation rule:** **Stop BEFORE exceeding 1300**

**Decision Tree:**

```python
def should_flush_chunk(current_tokens, next_para_tokens, chunk_size=1000, tolerance=1.3):
    """
    Decide whether to flush current chunk before adding next paragraph.

    Rules:
    1. If current_tokens + next_para_tokens <= chunk_size (1000):
       → Keep accumulating (no flush)

    2. If current_tokens + next_para_tokens > chunk_size AND <= chunk_size * tolerance (1300):
       → Keep accumulating IF this improves context coherence
       → Otherwise flush

    3. If current_tokens + next_para_tokens > chunk_size * tolerance (1300):
       → MUST flush (hard limit)
    """

    new_total = current_tokens + next_para_tokens

    if new_total <= chunk_size:
        return False  # No flush, keep accumulating
    elif new_total <= chunk_size * tolerance:
        # In overflow zone - use heuristic
        # If current chunk is already > 1000, flush to avoid growing too large
        if current_tokens >= chunk_size:
            return True  # Flush
        else:
            return False  # Allow overflow for context coherence
    else:
        return True  # Hard limit exceeded, must flush
```

**Example Application:**

```
Scenario: Para1=400, Para2=350, Para3=420, Para4=300

Step 1: Start empty
  - Add Para1 (400) → total 400 < 1000 → keep

Step 2: Consider Para2 (350)
  - New total: 400 + 350 = 750 < 1000 → keep

Step 3: Consider Para3 (420)
  - New total: 750 + 420 = 1170
  - 1170 > 1000 but < 1300 (tolerance)
  - Current chunk (750) < 1000 → ALLOW overflow → keep
  - Chunk now: [Para1, Para2, Para3] = 1170 tokens ✅

Step 4: Consider Para4 (300)
  - New total: 1170 + 300 = 1470 > 1300 (EXCEEDS tolerance)
  - MUST flush → Create Chunk1 [Para1, Para2, Para3] = 1170 tokens
  - Start Chunk2 with overlap + Para4
```

**Clarification:** 1300 tokens applies to **accumulated chunk total**, not individual paragraphs. Single large paragraphs up to 1300 are kept intact; those exceeding 1300 are sentence-split.

---

## Issue 2: Gleaning Validation Flow

### The Ambiguity

**Original spec:** "after initial extraction passes validation, if gleaning is enabled..."

**Question:** What if initial extraction FAILS validation?

### Clarified Behavior

**Two-Stage Extraction Flow:**

```
STAGE 1: Initial Extraction with Validation Retry
├─ Attempt 1: Extract + Validate
│  ├─ PASS → Go to STAGE 2 (gleaning)
│  └─ FAIL → Retry
├─ Attempt 2: Extract + Validate (same prompt)
│  ├─ PASS → Go to STAGE 2 (gleaning)
│  └─ FAIL → Retry
└─ Attempt 3: Extract + Validate (same prompt)
   ├─ PASS → Go to STAGE 2 (gleaning)
   └─ FAIL → REJECT chunk (return None)

STAGE 2: Gleaning (only if STAGE 1 passed)
├─ Pass 1: Extract with history + Validate
│  ├─ PASS/WARNING → Merge with initial
│  └─ FAIL → Skip this pass, continue
└─ Pass 2: Extract with history + Validate
   ├─ PASS/WARNING → Merge with accumulated
   └─ FAIL → Skip this pass
Final: Return merged extraction
```

**Key Decision:** Gleaning is a **refinement step**, not error recovery. Only run if initial extraction succeeds.

**Mode-Specific Behavior:**

| Mode | Initial Retry | Gleaning Passes | Total LLM Calls |
|------|---------------|-----------------|-----------------|
| `strict` | Up to 3x | 0 | 1-3 |
| `gleaning` | Up to 3x | 2 | 2-5 |
| `hybrid` | Tables: 1-3x<br>Paragraphs: 1-3x + 2 | Tables: 0<br>Paragraphs: 2 | Mixed |

**Example:**

```python
# Scenario: Initial extraction fails twice, passes on attempt 3

async def extract_from_paragraph(text, enable_gleaning=True):
    # STAGE 1: Initial extraction with retry
    for attempt in range(1, 4):  # Max 3 attempts
        extraction = await llm_extract(text)
        validation = validate_extraction(extraction)

        if validation['status'] in ['PASS', 'WARNING']:
            # Initial extraction succeeded
            break
    else:
        # All 3 attempts failed
        return None  # Reject chunk

    # STAGE 2: Gleaning (only if stage 1 succeeded)
    if not enable_gleaning:
        return extraction  # Return initial result

    merged = extraction
    history = [initial_prompt, extraction_response]

    for gleaning_pass in range(2):  # 2 gleaning passes
        glean_result = await llm_extract_with_history(text, history)
        glean_validation = validate_extraction(glean_result)

        if glean_validation['status'] in ['PASS', 'WARNING']:
            merged = merge_by_quality(merged, glean_result)
        # If validation fails, skip this pass but continue to next

    return merged
```

**Clarification:** Initial extraction retries up to 3x with same prompt (error recovery). If any attempt passes, gleaning runs for refinement. Gleaning failures are non-fatal (logged but don't reject chunk).

---

## Issue 3: Overlap Handling for Edge Cases

### The Ambiguity

**Original spec:** "200 tokens overlap (100 before + 100 after)"

**Question:** What about first/last chunks? Single-paragraph documents?

### Clarified Behavior

**Overlap Rules:**

| Chunk Position | Overlap Before | Overlap After | Total Overlap |
|----------------|----------------|---------------|---------------|
| **First chunk** | 0 (no previous) | 100 tokens | 100 tokens |
| **Middle chunks** | 100 tokens | 100 tokens | 200 tokens |
| **Last chunk** | 100 tokens | 0 (no next) | 100 tokens |
| **Single chunk** (doc < 1300 tokens) | 0 | 0 | 0 tokens |

**Implementation:**

```python
def add_overlap(chunks: List[str]) -> List[str]:
    """Add overlap to chunks, handling edge cases."""

    if len(chunks) == 1:
        # Single chunk - no overlap needed
        return chunks

    overlapped_chunks = []

    for i, chunk in enumerate(chunks):
        is_first = (i == 0)
        is_last = (i == len(chunks) - 1)

        # Build overlapped chunk
        parts = []

        # Add overlap from previous chunk (if not first)
        if not is_first:
            prev_overlap = get_last_n_sentences(chunks[i-1], target_tokens=100)
            parts.append(prev_overlap)

        # Add current chunk
        parts.append(chunk)

        # Add overlap from next chunk (if not last)
        if not is_last:
            next_overlap = get_first_n_sentences(chunks[i+1], target_tokens=100)
            parts.append(next_overlap)

        overlapped_chunks.append('\n\n'.join(parts))

    return overlapped_chunks
```

**Example:**

```
Document: Para1 (800) + Para2 (600) = 1400 tokens
Split into 2 chunks

Chunk 1 (first):
  [Para1: 800 tokens] + [Overlap: first 100 tokens of Para2]
  Total: ~900 tokens

Chunk 2 (last):
  [Overlap: last 100 tokens of Para1] + [Rest of Para2: 500 tokens]
  Total: ~600 tokens
```

**Clarification:** Overlap is asymmetric at document boundaries. Single-paragraph documents under 1300 tokens stored as-is without artificial overlap.

---

## Issue 4: Quality Score Tiebreaker

### The Ambiguity

**Original spec:** "keep the version with higher description_quality_score()"

**Question:** What if scores are exactly equal?

### Clarified Behavior

**Tiebreaking Hierarchy:**

```python
def select_better_description(desc1, desc2):
    """
    Select better description using multi-level tiebreaking.

    Priority:
    1. Higher quality score (primary)
    2. Longer description (secondary)
    3. Keep first seen (tertiary - stable sort)
    """

    score1 = description_quality_score(desc1)
    score2 = description_quality_score(desc2)

    # Level 1: Quality score
    if score1 > score2:
        return desc1
    elif score2 > score1:
        return desc2

    # Level 2: Length (tie on quality)
    if len(desc1) > len(desc2):
        return desc1
    elif len(desc2) > len(desc1):
        return desc2

    # Level 3: Stable sort (tie on length)
    return desc1  # Keep first seen (initial extraction)
```

**Example:**

```
Scenario: Two descriptions with same quality score (42)

Initial extraction:  "CSE হল একটি বিভাগ।" (length: 18 chars, quality: 42)
Gleaning pass 1:     "CSE একটি বিভাগের নাম।" (length: 20 chars, quality: 42)

Decision: Keep gleaning (longer: 20 > 18)
```

**Clarification:** Tiebreaker is deterministic (quality → length → first-seen). Prevents random selection and ensures reproducibility.

---

## Q1: Entity Merging Across Gleaning Passes

### The Question

**Scenario:** Same entity found in initial extraction AND gleaning pass with different scores.

```
Initial:  Entity("CSE", key_score=85, description="CSE একটি বিভাগ।")
Gleaning: Entity("CSE", key_score=95, description="কম্পিউটার সায়েন্স...")
```

**Question:** Sum key_scores or keep highest?

### Clarified Behavior

**Merging Rule:** **SUM key_scores across passes** (reflects cumulative importance)

**Rationale:** key_score represents "importance" in source text. If entity is mentioned in initial extraction (score 85) and re-emphasized in gleaning (score 95), total importance is 180, not 95.

**Implementation:**

```python
def merge_entity_across_passes(base_entity, glean_entity):
    """
    Merge same entity from different extraction passes.

    Rules:
    - Description: Keep better quality (tiebreak by length)
    - Key score: SUM across passes
    - Other attributes: Prefer gleaning (more recent)
    """

    # Select better description
    base_desc = base_entity['description']
    glean_desc = glean_entity['description']
    better_desc = select_better_description(base_desc, glean_desc)

    # Sum key scores
    total_key_score = base_entity['key_score'] + glean_entity['key_score']

    return {
        'entity_name': base_entity['entity_name'],
        'description': better_desc,
        'key_score': total_key_score,  # SUM
        'entity_type': glean_entity.get('entity_type', base_entity['entity_type']),
        'passes_found': [base_entity['pass_id'], glean_entity['pass_id']]
    }
```

**Example:**

```
Para: "কুয়েটে CSE সবচেয়ে জনপ্রিয়। CSE বিভাগে ১২০টি আসন।"

Initial extraction:
  Entity("CSE", score=85, desc="CSE একটি বিভাগ")

Gleaning pass 1:
  Entity("CSE", score=92, desc="কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগ")

Merged result:
  Entity("CSE", score=177, desc="কম্পিউটার সায়েন্স...")
  (Sum: 85+92=177, Description: gleaning is better quality)
```

**Clarification:** SUM key_scores to capture entity prominence across all extraction passes. This aligns with weight aggregation in standard pipeline.

---

## Q2: Gleaning Context Window Management

### The Question

**Scenario:** Long document + conversation history may exceed model context limit.

```
Chunk: 1000 tokens
Initial prompt + examples: 500 tokens
Initial response: 800 tokens
Gleaning prompt: 300 tokens
Pass 1 response: 700 tokens

Pass 2 total: 1000 + 500 + 800 + 300 + 700 = 3300 tokens
```

**Question:** How to handle context overflow?

### Clarified Behavior

**Context Budget Management:**

```python
# Model context limits
GPT_4O_MINI_CONTEXT = 8192 tokens
CONTEXT_SAFETY_MARGIN = 2000  # Reserve for response

MAX_CONVERSATION_TOKENS = GPT_4O_MINI_CONTEXT - CONTEXT_SAFETY_MARGIN  # 6192

def build_gleaning_context(source_text, initial_extraction, previous_passes):
    """
    Build conversation history for gleaning, with intelligent truncation.

    Priority (keep in order):
    1. Source text (MUST keep - required for extraction)
    2. Latest extraction result (most recent context)
    3. Initial extraction result (baseline)
    4. Gleaning prompt template (no examples)
    5. Older gleaning passes (drop if needed)
    """

    messages = []
    token_count = 0

    # 1. Source text (ALWAYS include)
    source_tokens = count_tokens_fast(source_text)
    if source_tokens > 2000:
        # If source is huge, we have bigger problems - log warning
        logger.warning(f"Source text is {source_tokens} tokens - may exceed context")
    token_count += source_tokens

    # 2. Initial extraction prompt (WITHOUT examples to save tokens)
    prompt_template = create_extraction_prompt_template(
        include_examples=False  # KEY: Skip examples in gleaning
    )
    prompt_tokens = count_tokens_fast(prompt_template)
    messages.append({"role": "user", "content": prompt_template.format(text=source_text)})
    token_count += prompt_tokens

    # 3. Initial extraction response
    initial_response_tokens = count_tokens_fast(json.dumps(initial_extraction))
    messages.append({"role": "assistant", "content": json.dumps(initial_extraction)})
    token_count += initial_response_tokens

    # 4. Previous gleaning passes (add if space available)
    for pass_idx, (gleaning_prompt, gleaning_response) in enumerate(previous_passes):
        pass_tokens = count_tokens_fast(gleaning_prompt) + count_tokens_fast(gleaning_response)

        if token_count + pass_tokens > MAX_CONVERSATION_TOKENS:
            logger.warning(f"Dropping gleaning pass {pass_idx} to fit context window")
            break

        messages.append({"role": "user", "content": gleaning_prompt})
        messages.append({"role": "assistant", "content": gleaning_response})
        token_count += pass_tokens

    logger.info(f"Gleaning context: {token_count} tokens ({len(messages)} messages)")
    return messages
```

**Key Optimizations:**

1. **Skip examples in gleaning prompts** (saves ~400 tokens)
2. **Drop oldest gleaning passes first** if overflow
3. **Always keep source text + initial extraction** (required context)
4. **Reserve 2000 tokens for response** (safety margin)

**Clarification:** Context overflow handled by intelligent truncation. Extraction examples only in initial prompt. Oldest gleaning passes dropped first if needed.

---

## Q3: Hybrid Mode Content Type Detection

### The Question

**Scenario:** User passes raw chunks without `content_type` field.

```python
raw_chunks = [{"content": "Some text...", "chunk_id": "chunk_0"}]  # Missing content_type!
```

**Question:** How does hybrid mode route these chunks?

### Clarified Behavior

**Content Type Detection Hierarchy:**

```python
def determine_extraction_strategy(chunk: dict, global_strategy: str) -> str:
    """
    Determine extraction strategy for a chunk.

    Priority:
    1. Explicit chunk content_type (if set by TableAwareChunker)
    2. Heuristic detection (fallback)
    3. Global strategy default
    """

    if global_strategy in ['strict', 'gleaning']:
        # Non-hybrid mode - use global strategy
        return global_strategy

    # Hybrid mode - need content type detection

    # Priority 1: Explicit content_type (from TableAwareChunker)
    if 'content_type' in chunk:
        content_type = chunk['content_type']
        if content_type == 'table':
            return 'strict'
        elif content_type == 'paragraph':
            return 'gleaning'

    # Priority 2: Heuristic detection (fallback for raw chunks)
    content = chunk.get('content', '')

    # Check for markdown table syntax
    if detect_markdown_table(content):
        logger.info(f"Chunk {chunk.get('chunk_id', 'unknown')}: Detected table (heuristic) → strict mode")
        return 'strict'

    # Check for high numeric density (suggests structured data)
    numeric_density = count_numbers(content) / len(content.split())
    if numeric_density > 0.3:  # 30%+ of words are numbers
        logger.info(f"Chunk {chunk.get('chunk_id', 'unknown')}: High numeric density ({numeric_density:.1%}) → strict mode")
        return 'strict'

    # Priority 3: Default to gleaning (conservative choice for hybrid)
    logger.info(f"Chunk {chunk.get('chunk_id', 'unknown')}: No clear type → gleaning mode (default)")
    return 'gleaning'

def detect_markdown_table(text: str) -> bool:
    """Detect markdown table syntax."""
    # Markdown table has:
    # 1. Lines with | delimiters
    # 2. Header separator line with |---|---|
    lines = text.split('\n')

    has_pipe_lines = any('|' in line for line in lines)
    has_separator = any(re.match(r'\|[\s\-:]+\|', line) for line in lines)

    return has_pipe_lines and has_separator
```

**Example:**

```python
# Case 1: Explicit content_type (from TableAwareChunker)
chunk1 = {"content": "...", "content_type": "table"}
→ Uses strict mode ✅

# Case 2: Heuristic detection (markdown table)
chunk2 = {"content": "| Col1 | Col2 |\n|---|---|\n| A | B |"}
→ detect_markdown_table() returns True → strict mode ✅

# Case 3: High numeric density
chunk3 = {"content": "১২০ আসন, ৬০ আসন, ৩০ আসন"}
→ numeric_density = 3/5 = 60% > 30% → strict mode ✅

# Case 4: Default fallback (narrative text)
chunk4 = {"content": "This is a normal paragraph with no special structure"}
→ No markers detected → gleaning mode (safe default) ✅
```

**Clarification:** Hybrid mode requires content_type for optimal routing. If missing, uses heuristic fallback (markdown table syntax, numeric density). Default to gleaning (conservative - higher accuracy, slower).

---

## Additional Recommendations

### Recommendation 1: Add Chunk Boundary Visualization

**Purpose:** Debug why semantic chunking fails/succeeds.

```python
def visualize_chunk_boundaries(chunks: List[dict], output_file: str = "chunks_debug.txt"):
    """
    Save human-readable visualization of chunk boundaries.

    Format:
    ===== CHUNK 0 (tokens: 1170, type: paragraph) =====
    Content preview (first 200 chars)...
    [Overlap from previous: 0 tokens]
    [Overlap to next: 100 tokens]
    ====================================================
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n{'='*80}\n")
            f.write(f"CHUNK {i} (tokens: {chunk.get('tokens', 'N/A')}, type: {chunk.get('content_type', 'N/A')})\n")
            f.write(f"{'='*80}\n\n")

            content = chunk['content']
            preview = content[:200] + ('...' if len(content) > 200 else '')
            f.write(preview)

            f.write(f"\n\n[Overlap before: {chunk.get('overlap_before_tokens', 0)} tokens]")
            f.write(f"\n[Overlap after: {chunk.get('overlap_after_tokens', 0)} tokens]")
            f.write(f"\n{'='*80}\n")
```

**Usage:** Call after chunking in test suite to verify semantic boundaries.

---

### Recommendation 2: Add Extraction Metrics Logging

**Purpose:** Quantify gleaning effectiveness.

```python
def log_extraction_metrics(chunk_id: str, initial_result: dict, gleaning_results: List[dict]):
    """
    Log structured metrics for analysis.

    Saves to: logs/extraction_metrics.jsonl
    """
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "chunk_id": chunk_id,
        "extraction_strategy": "gleaning" if gleaning_results else "strict",
        "initial": {
            "entity_count": len(initial_result.get('entities', [])),
            "relation_count": len(initial_result.get('relations', [])),
            "validation_status": initial_result.get('validation', {}).get('status', 'UNKNOWN')
        },
        "gleaning_passes": []
    }

    for pass_idx, glean_result in enumerate(gleaning_results):
        metrics["gleaning_passes"].append({
            "pass_number": pass_idx + 1,
            "new_entities": len(glean_result.get('new_entities', [])),
            "improved_entities": len(glean_result.get('improved_entities', [])),
            "validation_status": glean_result.get('validation', {}).get('status', 'UNKNOWN')
        })

    # Append to JSONL file (one JSON object per line)
    with open("logs/extraction_metrics.jsonl", 'a', encoding='utf-8') as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
```

**Analysis:**

```bash
# Count gleaning improvements
cat logs/extraction_metrics.jsonl | jq '.gleaning_passes[].new_entities' | awk '{sum+=$1} END {print "Total new entities from gleaning:", sum}'

# Average improvement per chunk
cat logs/extraction_metrics.jsonl | jq '[.gleaning_passes[].new_entities] | add' | awk '{sum+=$1; count++} END {print "Avg per chunk:", sum/count}'
```

---

### Recommendation 3: Add Gleaning Budget Parameter

**Purpose:** Give users cost/quality control.

```python
# In enhanced_pipeline_config
enhanced_pipeline_config = {
    "extraction_strategy": "hybrid",
    "gleaning_budget": "moderate"  # 'light' | 'moderate' | 'thorough'
}

# Mapping
GLEANING_BUDGET_MAP = {
    'light': 1,      # 1 gleaning pass (total 2 LLM calls)
    'moderate': 2,   # 2 gleaning passes (total 3 LLM calls) [DEFAULT]
    'thorough': 3    # 3 gleaning passes (total 4 LLM calls)
}
```

---

## Critical Path Checklist

Before starting implementation, ensure ALL these points are addressed:

- [x] **Issue 1 resolved:** Chunk accumulation stops before exceeding 1300 tokens
- [x] **Issue 2 resolved:** Gleaning only runs after initial extraction succeeds
- [x] **Issue 3 resolved:** Overlap asymmetric at boundaries, none for single chunks
- [x] **Issue 4 resolved:** Tiebreaker: quality → length → first-seen
- [x] **Q1 resolved:** Sum key_scores across gleaning passes
- [x] **Q2 resolved:** Skip examples in gleaning, truncate oldest passes first
- [x] **Q3 resolved:** Hybrid mode uses heuristics if content_type missing

---

**All clarifications are FINAL and should be used as implementation spec.**
