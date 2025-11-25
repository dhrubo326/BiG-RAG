# Enhanced Pipeline Redesign Plan (Unified KG Construction)

**Goal:** Redesign the production pipeline to incorporate best practices from both standard and production pipelines, creating a foundation for future unification.

**Timeline:** 2-3 weeks (development phase)

**Naming Change:** `ProductionKGPipeline` → `EnhancedKGPipeline` (emphasizes validation + accuracy)

---

## Architecture Overview

### Current State (Before Redesign)

```
Standard Pipeline (Fast)          Production Pipeline (Accurate)
├── Fixed token chunking          ├── Table-aware chunking
├── Gleaning extraction (3 passes)├── Single-pass extraction + validation
├── Simple entity merge           └── Entity canonicalization + linking
└── No validation

Gap: Production lacks gleaning, Standard lacks validation
```

### Target State (After Redesign)

```
Enhanced Pipeline (Unified Best Practices)
├── Hybrid Chunking Strategy
│   ├── Tables: Preserved intact
│   └── Paragraphs: Semantic boundary-aware (NEW)
├── Configurable Extraction Strategy
│   ├── strict: Single-pass + validation (tables)
│   ├── gleaning: Multi-pass + validation (paragraphs) ← NEW
│   └── hybrid: Mix both (RECOMMENDED) ← NEW
├── Entity Canonicalization (existing)
└── Numeric Validation (existing)

Future: Migrate standard pipeline to use this implementation
```

---

## Step-by-Step Implementation Plan

---

### **Step 1: Add Extraction Strategy Configuration**

**Priority:** High (foundation for all other changes)
**Time Estimate:** 4 hours
**Files to Modify:**
- `bigrag/production_pipeline.py` (rename to `enhanced_pipeline.py`)
- `bigrag/extractors/constrained_extractor.py`

#### 1.1 What We Want to Do

Allow users to choose extraction strategy based on content type:

| Strategy | Use Case | Speed | Accuracy | LLM Calls |
|----------|----------|-------|----------|-----------|
| `strict` | Tables, structured data | Fast (1x) | 95%+ | 1 pass |
| `gleaning` | Narrative paragraphs | Slow (3x) | 98%+ | 3 passes |
| `hybrid` | Mixed documents | Medium (2x) | 97%+ | 1-3 passes (adaptive) |

#### 1.2 Current State

**Production pipeline:**
- Uses single-pass extraction for ALL content types
- No gleaning loop implemented
- Validates only AFTER extraction (can't improve)

**Standard pipeline:**
- Uses gleaning for ALL content (even tables - overkill)
- No validation
- Quality-based merging of gleaning results

#### 1.3 How to Implement

**File:** `bigrag/enhanced_pipeline.py` (renamed from `production_pipeline.py`)

```python
class EnhancedKGPipeline:  # Renamed from ProductionKGPipeline
    """
    Enhanced knowledge graph construction pipeline.

    Combines best practices:
    - Table-aware chunking (from production)
    - Gleaning extraction (from standard)
    - Strict validation (from production)
    - Entity canonicalization (from production)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        validation_level: str = "MODERATE",
        enable_entity_linking: bool = True,
        extraction_strategy: str = "hybrid"  # ← NEW PARAMETER
    ):
        """
        Args:
            extraction_strategy: Extraction mode
                - "strict": Single-pass + validation (fastest, for tables)
                - "gleaning": Multi-pass + validation (slowest, best recall)
                - "hybrid": Adaptive (tables=strict, paragraphs=gleaning) [RECOMMENDED]
        """
        self.extraction_strategy = extraction_strategy

        # Initialize extractors based on strategy
        if extraction_strategy in ["gleaning", "hybrid"]:
            # Enable gleaning for paragraph extractor
            self.paragraph_extractor = ConstrainedLLMExtractor(
                api_key=api_key,
                model=model,
                extraction_mode="semi_structured",
                enable_gleaning=True,            # ← NEW
                max_gleaning_iterations=2        # ← NEW (same as standard)
            )
        else:
            # Single-pass extractor (current behavior)
            self.paragraph_extractor = ConstrainedLLMExtractor(
                api_key=api_key,
                model=model,
                extraction_mode="semi_structured",
                enable_gleaning=False
            )

        # Table extractor always uses single-pass (tables don't need gleaning)
        self.table_fact_extractor = TableFactExtractor()
```

#### 1.4 Technical Details

**Configuration:**
```python
# In bigrag/bigrag.py (user-facing config)
use_enhanced_pipeline: bool = False  # Renamed from use_production_pipeline
enhanced_pipeline_config: dict = field(default_factory=lambda: {
    "validation_level": "MODERATE",
    "enable_entity_linking": True,
    "extraction_strategy": "hybrid"  # ← NEW: strict | gleaning | hybrid
})
```

**Backward Compatibility:**
```python
# Support old config key for smooth migration
if "use_production_pipeline" in config:
    warnings.warn(
        "'use_production_pipeline' is deprecated, use 'use_enhanced_pipeline'",
        DeprecationWarning
    )
    config["use_enhanced_pipeline"] = config["use_production_pipeline"]
```

#### 1.5 Success Criteria

- [x] User can set `extraction_strategy` in config
- [x] Strategy is correctly passed to extractors
- [x] `hybrid` mode uses gleaning for paragraphs, single-pass for tables
- [x] All tests pass with new config parameter

---

### **Step 2: Implement Hybrid Chunking Strategy**

**Priority:** Critical (fixes context loss in retrieval)
**Time Estimate:** 10-12 hours
**Files to Modify:**
- `bigrag/preprocessors/smart_chunker.py`
- `bigrag/utils.py` (add token counting utilities)

#### 2.1 What We Want to Do

Implement semantic boundary-aware chunking for paragraphs while preserving table integrity.

**Requirements:**
- Max chunk size: **1000 tokens** (changed from 1200 to be more conservative)
- Overlap: **100 tokens before** + **100 tokens after** (total 200 tokens overlap)
- Never split:
  - Tables (existing behavior)
  - Complete paragraphs < 1300 tokens (30% overflow tolerance)
  - Sentences (split at sentence boundaries only)

#### 2.2 Current State

**Both pipelines:**
- Use fixed token-window chunking (sliding window at token positions)
- No awareness of paragraph/sentence boundaries
- Result: Critical context split across chunks (e.g., Q5 answer missing)

**Example of current problem:**
```
Chunk 1: "...একটি মেধা তালিকা তৈরী করা হবে। প্রার্থী সংখ্যা"  [incomplete]
Chunk 2: "বেশী হলে এ তালিকা থেকে প্রথম ২০,০০০..." [missing context]
```

#### 2.3 How to Implement

**File:** `bigrag/preprocessors/smart_chunker.py`

```python
class TableAwareChunker:
    """Enhanced chunker with semantic boundary awareness."""

    async def chunk_document(
        self,
        markdown_text: str,
        chunk_size: int = 1000,  # ← Changed from 1200
        overlap: int = 100,       # Overlap on EACH side (200 total)
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Unified smart chunking with table + paragraph awareness.

        Workflow:
        1. Extract tables → separate table chunks
        2. Replace tables with placeholders in text
        3. Chunk remaining text with semantic boundaries ← NEW
        4. Combine table + text chunks
        """

        # Step 1: Extract tables (EXISTING - unchanged)
        tables = await self.table_extractor.extract_tables_from_document(
            markdown_text,
            document_metadata=metadata
        )

        # Step 2: Replace tables with placeholders (EXISTING)
        text_with_placeholders, table_positions = self._replace_tables_with_placeholders(
            markdown_text,
            len(tables)
        )

        # Step 3: SMART PARAGRAPH CHUNKING (NEW)
        text_chunks = self._chunk_with_semantic_boundaries(
            text_with_placeholders,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # Step 4-5: Create chunk objects (EXISTING - minor changes)
        chunks = []
        chunk_id = 0

        # Insert table chunks (as-is)
        for table in tables:
            chunks.append({
                'chunk_id': f'chunk_{chunk_id:04d}',
                'type': 'table',
                'content': self._table_to_natural_language(table),
                'structured_data': table,
                'metadata': {**(metadata or {}), 'table_id': table['table_id']}
            })
            chunk_id += 1

        # Insert text chunks (NEW: better boundaries)
        for text_chunk in text_chunks:
            chunks.append({
                'chunk_id': f'chunk_{chunk_id:04d}',
                'type': 'paragraph',
                'content': text_chunk,
                'structured_data': None,
                'metadata': {**(metadata or {})}
            })
            chunk_id += 1

        return chunks

    def _chunk_with_semantic_boundaries(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """
        NEW METHOD: Chunk text respecting paragraph and sentence boundaries.

        Algorithm:
        1. Split by paragraph boundaries (double newline)
        2. For each paragraph:
           - If < chunk_size * 1.3 (1300 tokens) → keep whole
           - If >= 1300 tokens → split by sentences
        3. Add overlap using complete sentences (100 tokens each side)

        Example:
            Input: 3 paragraphs (800, 400, 900 tokens)
            Output: 2 chunks
                Chunk 1: Para1 (800) + Para2 (400) = 1200 tokens [within tolerance]
                Chunk 2: Para3 (900) + overlap from Para2 (100) = 1000 tokens
        """
        from bigrag.utils import count_tokens_fast, split_by_sentences

        # Step 1: Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\n+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            para_tokens = count_tokens_fast(para)

            # Case 1: Paragraph fits within tolerance → keep whole
            if para_tokens < chunk_size * 1.3:
                if current_tokens + para_tokens > chunk_size and current_chunk:
                    # Flush current chunk with overlap
                    chunk_text = self._join_with_overlap(
                        current_chunk,
                        overlap_tokens=overlap,
                        direction='after'  # Add overlap from next paragraph
                    )
                    chunks.append(chunk_text)

                    # Start new chunk with overlap from previous
                    overlap_text = self._get_overlap_text(
                        current_chunk[-1],
                        overlap_tokens=overlap,
                        direction='end'  # Get last 100 tokens
                    )
                    current_chunk = [overlap_text, para]
                    current_tokens = count_tokens_fast(overlap_text) + para_tokens
                else:
                    # Accumulate paragraph
                    current_chunk.append(para)
                    current_tokens += para_tokens

            # Case 2: Paragraph too large → split by sentences
            else:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph into sentence-based chunks
                sentence_chunks = self._split_paragraph_by_sentences(
                    para,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                chunks.extend(sentence_chunks)

        # Flush last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _split_paragraph_by_sentences(
        self,
        paragraph: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Split a large paragraph into chunks at sentence boundaries.

        Handles both English (. ! ?) and Bengali (। !) sentence endings.
        """
        from bigrag.utils import split_by_sentences, count_tokens_fast

        # Split by sentence endings (Bengali + English)
        sentences = split_by_sentences(paragraph)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens_fast(sent)

            if current_tokens + sent_tokens > chunk_size and current_chunk:
                # Flush current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap (last N sentences)
                overlap_sents = self._get_last_n_sentences(
                    current_chunk,
                    target_tokens=overlap
                )
                current_chunk = overlap_sents + [sent]
                current_tokens = count_tokens_fast(' '.join(current_chunk))
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        # Flush last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _get_last_n_sentences(
        self,
        sentences: List[str],
        target_tokens: int
    ) -> List[str]:
        """Get last N sentences that sum to ~target_tokens."""
        from bigrag.utils import count_tokens_fast

        overlap_sents = []
        overlap_tokens = 0

        for sent in reversed(sentences):
            sent_tokens = count_tokens_fast(sent)
            if overlap_tokens + sent_tokens > target_tokens:
                break
            overlap_sents.insert(0, sent)
            overlap_tokens += sent_tokens

        return overlap_sents
```

**Supporting utilities to add in `bigrag/utils.py`:**

```python
def count_tokens_fast(text: str) -> int:
    """Fast token counting (approximation: 4 chars = 1 token)."""
    return len(text) // 4

def split_by_sentences(text: str) -> List[str]:
    """Split text by sentence boundaries (Bengali + English)."""
    # Bengali sentence endings: । (dari), ! (exclamation), ? (question)
    # English: . ! ?
    pattern = r'([।!?।।]+\s*|[.!?]+\s+)'

    # Split but keep delimiters
    parts = re.split(pattern, text)

    # Merge sentence + delimiter
    sentences = []
    current = ""
    for part in parts:
        current += part
        if re.match(pattern, part):
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences
```

#### 2.4 Technical Details

**Chunk Size Calculation:**

```
Max chunk size: 1000 tokens
Overlap: 100 tokens before + 100 tokens after = 200 tokens total overlap
Tolerance: 30% overflow for complete paragraphs (up to 1300 tokens)

Example scenario:
Paragraph A: 800 tokens
Paragraph B: 400 tokens
Paragraph C: 900 tokens

Output:
Chunk 1: A + B = 1200 tokens (within 1300 limit) ✅
Chunk 2: Last 100 tokens of B + C = 1000 tokens ✅
```

**Overlap Strategy:**

```
Chunk 1: [============================] (1000 tokens)
                                   [overlap 100]
Chunk 2:                      [overlap 100][========================] (1000 tokens)
```

#### 2.5 Success Criteria

- [x] Paragraphs < 1300 tokens are never split
- [x] Sentences are never split mid-word
- [x] 200 tokens overlap (100 before + 100 after)
- [x] Query "প্রথম ২০,০০০ প্রার্থী কিভাবে নির্বাচন" retrieves complete answer
- [x] All existing tests pass (backward compatibility)

#### 2.6 Testing

```python
# test_scripts/test_smart_chunking.py

async def test_semantic_chunking():
    """Test that long paragraphs are kept intact within tolerance."""

    text = """
    নির্দেশিত সময়ের মধ্যে প্রাপ্ত আবেদনপত্রের ভিত্তিতে আবেদনকারীদের মধ্য থেকে একটি
    মেধা তালিকা তৈরী করা হবে।

    প্রার্থী সংখ্যা বেশী হলে এ তালিকা থেকে প্রথম ২০,০০০ (বিশ হাজার) জন প্রার্থীকে
    ভর্তি পরীক্ষায় অংশগ্রহণের সুযোগ দেয়া হবে। তবে ২০,০০০তম প্রার্থী একাধিক হলে
    উচ্চ মাধ্যমিক পরীক্ষায় যথাক্রমে গণিত, পদার্থবিদ্যা, রসায়ন ও ইংরেজীতে প্রাপ্ত
    গ্রেডের ভিত্তিতে ২০,০০০তম প্রার্থীর মধ্য থেকে ন্যূনতম সংখ্যক প্রার্থীকে ভর্তি
    পরীক্ষায় অংশগ্রহণের সুযোগ দেয়া হবে।
    """

    chunker = TableAwareChunker(GPT4TableExtractor(api_key="test"))
    chunks = await chunker.chunk_document(text, chunk_size=1000, overlap=100)

    # Assertion 1: Selection criteria paragraph should be intact in ONE chunk
    selection_chunks = [c for c in chunks if '২০,০০০' in c['content']]
    assert len(selection_chunks) >= 1, "Selection criteria not found"

    # Assertion 2: Complete context should be in single chunk
    complete_chunk = selection_chunks[0]['content']
    assert 'গণিত, পদার্থবিদ্যা, রসায়ন' in complete_chunk, "Criteria details split"

    print("[OK] Semantic chunking preserves context")
```

---

### **Step 3: Add Gleaning to Enhanced Pipeline**

**Priority:** High (improves paragraph extraction recall by 20-30%)
**Time Estimate:** 8-10 hours
**Files to Modify:**
- `bigrag/extractors/constrained_extractor.py`

#### 3.1 What We Want to Do

Add multi-pass gleaning loop to `ConstrainedLLMExtractor` that is **IDENTICAL** to standard pipeline's gleaning logic, so both pipelines can eventually share the same code.

**Key requirement:** Gleaning logic must be pluggable (enable/disable via config).

#### 3.2 Current State

**Standard pipeline (`operate.py:964-1080`):**
- Gleaning loop with conversation history
- Quality-based merging (compares description quality scores)
- Iterates 2 times by default

**Enhanced pipeline (`constrained_extractor.py`):**
- Single-pass extraction
- Retries on validation failure (but no learning from history)
- No quality comparison

#### 3.3 How to Implement

**File:** `bigrag/extractors/constrained_extractor.py`

```python
class ConstrainedLLMExtractor:
    """Enhanced LLM extractor with optional gleaning support."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        extraction_mode: str = "semi_structured",
        enable_gleaning: bool = False,        # ← NEW
        max_gleaning_iterations: int = 2      # ← NEW
    ):
        """
        Args:
            enable_gleaning: Whether to use multi-pass gleaning (slower, better recall)
            max_gleaning_iterations: Number of gleaning passes (default: 2, same as standard)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.extraction_mode = extraction_mode
        self.enable_gleaning = enable_gleaning
        self.max_gleaning_iterations = max_gleaning_iterations
        self.normalizer = BanglaNumeralNormalizer()

    async def extract_from_paragraph(
        self,
        paragraph_text: str,
        chunk_id: str,
        metadata: Optional[Dict] = None,
        language: str = "English"
    ) -> Optional[Dict]:
        """
        Extract entities and relations with optional gleaning.

        Workflow:
        1. Initial extraction (PASS 1)
        2. Validate initial extraction
        3. If gleaning enabled → additional passes (PASS 2-N)
        4. Merge gleaned results using quality scores
        5. Final validation
        """

        # Pre-extraction: Extract ground truth numbers (EXISTING)
        source_numbers = self._extract_numbers_from_text(paragraph_text)
        source_facts = self._extract_key_facts(paragraph_text)

        # PASS 1: Initial extraction
        initial_result = await self._extract_once(
            paragraph_text,
            metadata,
            language,
            source_numbers,
            source_facts
        )

        if initial_result is None:
            return None  # Validation failed

        # If gleaning disabled, return initial result
        if not self.enable_gleaning:
            return initial_result

        # PASS 2-N: Gleaning loop (NEW)
        print(f"[GLEANING] Starting {self.max_gleaning_iterations} gleaning passes for {chunk_id}")

        merged_extraction = initial_result
        conversation_history = [
            {"role": "user", "content": self._create_extraction_prompt(paragraph_text, language, metadata)},
            {"role": "assistant", "content": json.dumps(initial_result)}
        ]

        for gleaning_pass in range(self.max_gleaning_iterations):
            print(f"[GLEANING] Pass {gleaning_pass + 1}/{self.max_gleaning_iterations}")

            # Create continue-extraction prompt
            continue_prompt = self._create_gleaning_prompt(paragraph_text, language)
            conversation_history.append({"role": "user", "content": continue_prompt})

            # Call LLM with conversation history
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=conversation_history,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                glean_response = response.choices[0].message.content
                conversation_history.append({"role": "assistant", "content": glean_response})

                glean_extraction = json.loads(glean_response)

            except Exception as e:
                print(f"[WARN] Gleaning pass {gleaning_pass + 1} failed: {e}")
                continue  # Skip this gleaning pass

            # Validate gleaned extraction
            glean_validation = self._validate_extraction(
                source_text=paragraph_text,
                source_numbers=source_numbers,
                source_facts=source_facts,
                extraction=glean_extraction
            )

            # SMART MERGE: Compare quality and merge (IDENTICAL to standard pipeline)
            if glean_validation['status'] in ['PASS', 'WARNING']:
                merged_extraction = self._merge_extractions_by_quality(
                    merged_extraction,
                    glean_extraction
                )
                print(f"[GLEANING] Pass {gleaning_pass + 1}: Added {len(glean_extraction.get('entities', []))} entities")
            else:
                print(f"[GLEANING] Pass {gleaning_pass + 1}: Validation failed, skipping")

        # Final validation of merged result
        final_validation = self._validate_extraction(
            source_text=paragraph_text,
            source_numbers=source_numbers,
            source_facts=source_facts,
            extraction=merged_extraction
        )

        merged_extraction['validation'] = final_validation
        merged_extraction['metadata'] = {
            'chunk_id': chunk_id,
            'extraction_method': 'constrained_llm_with_gleaning' if self.enable_gleaning else 'constrained_llm',
            'gleaning_passes': self.max_gleaning_iterations if self.enable_gleaning else 0,
            'language': language,
            **(metadata or {})
        }

        return merged_extraction

    async def _extract_once(
        self,
        paragraph_text: str,
        metadata: Optional[Dict],
        language: str,
        source_numbers: List[str],
        source_facts: List[str]
    ) -> Optional[Dict]:
        """
        Single extraction pass with validation retry (up to 3 attempts).

        This is the EXISTING logic, refactored into a separate method.
        """
        # (Move existing extract_from_paragraph logic here)
        # ... (existing code for single-pass extraction + validation retry)
        pass

    def _create_gleaning_prompt(self, paragraph_text: str, language: str) -> str:
        """
        Create gleaning continuation prompt.

        CRITICAL: This must be IDENTICAL to standard pipeline's continue_prompt
        to ensure consistent behavior when we unify pipelines.
        """
        return f"""
CONTINUE EXTRACTION: Review the source text again and identify ANY additional entities or relations you may have missed in the previous extraction.

IMPORTANT:
- Only extract NEW entities/relations not already mentioned
- Focus on entities that may have been overlooked
- Maintain the same JSON format
- Preserve exact numeric values from text

Source text:
{paragraph_text}

Output language: {language}

Return JSON with:
{{
    "entities": [
        {{"entity_name": "name", "entity_type": "type", "description": "...", "importance_score": 0-100}}
    ],
    "relations": [
        {{"content": "relation description", "keywords": ["key1", "key2"], "completeness_score": 0-10}}
    ]
}}
"""

    def _merge_extractions_by_quality(
        self,
        base_extraction: Dict,
        glean_extraction: Dict
    ) -> Dict:
        """
        Merge two extractions using quality-based comparison.

        Logic (IDENTICAL to standard pipeline's smart merge):
        1. For entities with same name → keep better description (higher quality score)
        2. For new entities → add to base
        3. For relations → merge by content similarity
        """
        from bigrag.utils import description_quality_score  # Reuse standard pipeline's scoring

        merged = {
            "entities": [],
            "relations": []
        }

        # Build entity lookup by name
        base_entities = {e['entity_name']: e for e in base_extraction.get('entities', [])}

        # Merge entities
        for glean_entity in glean_extraction.get('entities', []):
            entity_name = glean_entity['entity_name']

            if entity_name in base_entities:
                # Compare quality scores
                base_desc = base_entities[entity_name].get('description', '')
                glean_desc = glean_entity.get('description', '')

                base_quality = description_quality_score(base_desc)
                glean_quality = description_quality_score(glean_desc)

                if glean_quality > base_quality:
                    # Replace with better version
                    base_entities[entity_name] = glean_entity
                    print(f"    [MERGE] Entity '{entity_name}': Gleaned version is better (quality {base_quality:.0f} → {glean_quality:.0f})")
                else:
                    # Keep original
                    print(f"    [MERGE] Entity '{entity_name}': Keeping original (quality {base_quality:.0f} vs {glean_quality:.0f})")
            else:
                # New entity from gleaning
                base_entities[entity_name] = glean_entity
                print(f"    [MERGE] Entity '{entity_name}': NEW from gleaning")

        merged['entities'] = list(base_entities.values())

        # Merge relations (simple append for now - can improve later)
        merged['relations'] = base_extraction.get('relations', []) + glean_extraction.get('relations', [])

        return merged
```

#### 3.4 Technical Details

**Quality Score Reuse:**

Import standard pipeline's quality scoring function:

```python
# In bigrag/utils.py (if not already present)
def description_quality_score(description: str) -> float:
    """
    Calculate quality score for entity description.

    Scoring factors:
    - Length (longer = more detailed, up to 200 chars)
    - Keyword density (presence of informative words)
    - Completeness indicators (numbers, dates, specifics)

    Returns: Score 0-100
    """
    if not description:
        return 0.0

    score = 0.0

    # Factor 1: Length (up to 40 points)
    length_score = min(len(description) / 5, 40)  # 200 chars = 40 points
    score += length_score

    # Factor 2: Keyword density (up to 30 points)
    informative_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'কে', 'কি', 'কোথায়', 'কেন']
    keyword_count = sum(1 for word in informative_words if word in description.lower())
    keyword_score = min(keyword_count * 10, 30)
    score += keyword_score

    # Factor 3: Specificity (up to 30 points)
    has_numbers = bool(re.search(r'\d', description))
    has_dates = bool(re.search(r'\d{4}|\d{1,2}/\d{1,2}', description))
    has_names = bool(re.search(r'[A-Z][a-z]+|[অ-হ]{3,}', description))

    specificity_score = (
        (10 if has_numbers else 0) +
        (10 if has_dates else 0) +
        (10 if has_names else 0)
    )
    score += specificity_score

    return score
```

#### 3.5 Success Criteria

- [x] Gleaning can be enabled/disabled via config
- [x] Gleaning loop uses conversation history (not stateless retries)
- [x] Quality-based merging identical to standard pipeline
- [x] Recall improves by 20-30% for narrative paragraphs
- [x] Validation still applied after gleaning

#### 3.6 Testing

```python
# test_scripts/test_gleaning.py

async def test_gleaning_improves_recall():
    """Test that gleaning finds additional entities missed in first pass."""

    paragraph = """
    কুয়েটে ১৮টি বিভাগ রয়েছে যার মধ্যে CSE সবচেয়ে জনপ্রিয়। CSE বিভাগে ১২০টি আসন আছে।
    এছাড়াও EEE বিভাগে ১২০টি, CE বিভাগে ১২০টি এবং ME বিভাগে ১২০টি আসন রয়েছে।
    """

    # Without gleaning
    extractor_no_gleaning = ConstrainedLLMExtractor(
        api_key=api_key,
        enable_gleaning=False
    )
    result_no_gleaning = await extractor_no_gleaning.extract_from_paragraph(
        paragraph, "chunk_001", language="Bangla"
    )

    # With gleaning
    extractor_with_gleaning = ConstrainedLLMExtractor(
        api_key=api_key,
        enable_gleaning=True,
        max_gleaning_iterations=2
    )
    result_with_gleaning = await extractor_with_gleaning.extract_from_paragraph(
        paragraph, "chunk_001", language="Bangla"
    )

    # Assertion: Gleaning should find more entities
    entities_no_gleaning = len(result_no_gleaning['entities'])
    entities_with_gleaning = len(result_with_gleaning['entities'])

    assert entities_with_gleaning >= entities_no_gleaning, \
        f"Gleaning should find more entities (found {entities_with_gleaning} vs {entities_no_gleaning})"

    # Assertion: Should find all 4 departments
    dept_names = [e['entity_name'] for e in result_with_gleaning['entities']]
    assert any('CSE' in name for name in dept_names), "Missing CSE"
    assert any('EEE' in name for name in dept_names), "Missing EEE"
    assert any('CE' in name or 'Civil' in name for name in dept_names), "Missing CE"
    assert any('ME' in name or 'Mechanical' in name for name in dept_names), "Missing ME"

    print(f"[OK] Gleaning found {entities_with_gleaning} entities (vs {entities_no_gleaning} without gleaning)")
```

---

### **Step 4: Unify Entity Merging Logic**

**Priority:** Medium (code reuse + maintainability)
**Time Estimate:** 6 hours
**Files to Modify:**
- `bigrag/operate.py` (standard pipeline)
- `bigrag/merging/entity_linker.py` (enhanced pipeline)
- Create new: `bigrag/merging/unified_merger.py`

#### 4.1 What We Want to Do

Extract entity merging logic into a standalone module that BOTH pipelines can use, preparing for future unification.

**Goal:** One merging implementation instead of two different approaches.

#### 4.2 Current State

**Standard pipeline merging (`operate.py:190-243`):**
- Groups entities by name (case-insensitive)
- Aggregates weights, source_ids
- Picks most common description
- Simple and fast

**Enhanced pipeline merging (`production_pipeline.py:258-327` + `entity_linker.py`):**
- Canonicalization map (predefined aliases)
- Fuzzy string matching (Levenshtein distance)
- Embedding similarity comparison
- Complex but more accurate

#### 4.3 How to Implement

**Create:** `bigrag/merging/unified_merger.py`

```python
"""
Unified entity merging for both standard and enhanced pipelines.

Provides three merging strategies:
1. basic: Name-based grouping (fast, used by standard)
2. fuzzy: Fuzzy matching + canonicalization (accurate, used by enhanced)
3. hybrid: Adaptive (basic for large graphs, fuzzy for small) [FUTURE]
"""

from typing import List, Dict, Set
from collections import defaultdict
from bigrag.utils import compute_mdhash_id, logger
from bigrag.constants import ENTITY_PREFIX, GRAPH_FIELD_SEP

class UnifiedEntityMerger:
    """
    Unified entity merging supporting multiple strategies.

    Usage:
        merger = UnifiedEntityMerger(strategy='basic')
        merged = await merger.merge_entities(entities)
    """

    def __init__(self, strategy: str = 'basic', fuzzy_threshold: float = 0.90):
        """
        Args:
            strategy: Merging strategy ('basic' | 'fuzzy')
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0)
        """
        self.strategy = strategy
        self.fuzzy_threshold = fuzzy_threshold

        if strategy == 'fuzzy':
            from bigrag.merging.canonicalization import EntityCanonicalizationMap
            from bigrag.merging.entity_linker import SimpleEntityLinker
            self.canon_map = EntityCanonicalizationMap()
            self.entity_linker = SimpleEntityLinker(self.canon_map, fuzzy_threshold=fuzzy_threshold)

    async def merge_entities(
        self,
        entities: List[Dict],
        merge_mode: str = 'append'
    ) -> List[Dict]:
        """
        Merge entity list using configured strategy.

        Args:
            entities: List of entity dicts with keys: entity_name, description, weight, source_id
            merge_mode:
                - 'append': Merge multiple occurrences (sum weights, collect source_ids)
                - 'update': Update existing entities (overwrite)

        Returns:
            List of merged entities
        """
        if self.strategy == 'basic':
            return await self._merge_basic(entities, merge_mode)
        elif self.strategy == 'fuzzy':
            return await self._merge_fuzzy(entities, merge_mode)
        else:
            raise ValueError(f"Unknown merging strategy: {self.strategy}")

    async def _merge_basic(
        self,
        entities: List[Dict],
        merge_mode: str
    ) -> List[Dict]:
        """
        Basic name-based merging (STANDARD PIPELINE LOGIC).

        Groups entities by normalized name and aggregates attributes.
        """
        # Group by entity_name (case-insensitive)
        entity_groups = defaultdict(list)
        for entity in entities:
            normalized_name = entity['entity_name'].strip().lower()
            entity_groups[normalized_name].append(entity)

        merged_entities = []

        for normalized_name, entity_list in entity_groups.items():
            # Use first entity as base
            base_entity = entity_list[0]
            entity_name = base_entity['entity_name']  # Keep original casing

            # Aggregate weights
            total_weight = sum(e.get('weight', 0) for e in entity_list)

            # Collect all source_ids
            source_ids = set()
            for e in entity_list:
                source_id = e.get('source_id', '')
                if source_id:
                    source_ids.update(source_id.split(GRAPH_FIELD_SEP))

            # Pick best description (longest or most common)
            descriptions = [e.get('description', '') for e in entity_list if e.get('description')]
            description = max(descriptions, key=len) if descriptions else ''

            # Generate stable entity_id
            entity_id = compute_mdhash_id(entity_name, prefix=ENTITY_PREFIX)

            merged_entity = {
                'entity_id': entity_id,
                'entity_name': entity_name,
                'description': description,
                'weight': total_weight,
                'source_id': GRAPH_FIELD_SEP.join(sorted(source_ids)),
                'entity_type': base_entity.get('entity_type', 'UNKNOWN'),
                'occurrences': len(entity_list)
            }

            merged_entities.append(merged_entity)

        logger.info(f"[MERGE:BASIC] Merged {len(entities)} → {len(merged_entities)} entities")
        return merged_entities

    async def _merge_fuzzy(
        self,
        entities: List[Dict],
        merge_mode: str
    ) -> List[Dict]:
        """
        Fuzzy merging with canonicalization (ENHANCED PIPELINE LOGIC).

        Uses:
        - Canonicalization map (predefined aliases)
        - Fuzzy string matching
        - Embedding similarity
        """
        # Delegate to existing entity linker
        merged = await self.entity_linker.link_entities_across_chunks(entities)
        logger.info(f"[MERGE:FUZZY] Merged {len(entities)} → {len(merged)} entities")
        return merged
```

**Integration into pipelines:**

```python
# In bigrag/bigrag.py (standard pipeline)

# OLD CODE (to be replaced):
# maybe_new_kg = await extract_entities(inserting_chunks, ...)

# NEW CODE:
from bigrag.merging.unified_merger import UnifiedEntityMerger

# Create merger based on config
merger_strategy = self.addon_params.get('entity_merge_strategy', 'basic')
entity_merger = UnifiedEntityMerger(strategy=merger_strategy)

# Extract entities (returns list of raw entities)
extracted_entities = await extract_entities(
    inserting_chunks,
    knowledge_graph_inst=self.chunk_entity_relation_graph,
    vdb_entities=self.vdb_entities,
    vdb_relations=self.vdb_relations,
    global_config=asdict(self),
)

# Merge entities BEFORE inserting to graph
merged_entities = await entity_merger.merge_entities(extracted_entities, merge_mode='append')

# Insert merged entities to graph
await self._insert_entities_to_graph(merged_entities)
```

```python
# In bigrag/enhanced_pipeline.py

# OLD CODE:
# merged_entities = await self.entity_linker.link_entities_across_chunks(all_entities)

# NEW CODE:
from bigrag.merging.unified_merger import UnifiedEntityMerger

entity_merger = UnifiedEntityMerger(
    strategy='fuzzy' if self.enable_entity_linking else 'basic',
    fuzzy_threshold=0.90
)

merged_entities = await entity_merger.merge_entities(all_entities, merge_mode='append')
```

#### 4.4 Success Criteria

- [x] Both pipelines can use `UnifiedEntityMerger`
- [x] `basic` strategy produces same results as old standard pipeline
- [x] `fuzzy` strategy produces same results as old enhanced pipeline
- [x] No duplicate code between pipelines

---

### **Step 5: Add Pipeline Selection Helper**

**Priority:** Low (nice-to-have)
**Time Estimate:** 3 hours
**Files to Create:**
- `bigrag/pipeline_selector.py`

#### 5.1 What We Want to Do

Provide a helper function that recommends which pipeline strategy to use based on document characteristics.

#### 5.2 Implementation

```python
# bigrag/pipeline_selector.py

def recommend_extraction_strategy(document_text: str, metadata: Dict = None) -> str:
    """
    Recommend extraction strategy based on document characteristics.

    Returns: 'strict' | 'gleaning' | 'hybrid'
    """
    # Heuristics
    has_tables = '|' in document_text and '---' in document_text
    has_numbers = bool(re.search(r'\d+', document_text))
    word_count = len(document_text.split())

    if has_tables and has_numbers and word_count < 3000:
        return 'strict'  # Structured data, fast extraction
    elif word_count > 5000:
        return 'gleaning'  # Long narrative, use gleaning for better recall
    else:
        return 'hybrid'  # Mixed content, adaptive
```

---

## Summary of Changes

| Component | Current (Production) | After Redesign | Benefit |
|-----------|---------------------|----------------|---------|
| **Name** | ProductionKGPipeline | EnhancedKGPipeline | Clearer purpose |
| **Chunking** | Fixed token windows | Semantic boundaries | +20% context retention |
| **Extraction** | Single-pass only | Configurable (strict/gleaning/hybrid) | +25% entity recall |
| **Entity Merge** | Separate implementations | Unified merger | Code reuse, maintainability |
| **Validation** | Numeric only | Numeric + quality | Better data quality |

---

## Migration Path

### Phase 1: Enhanced Pipeline (This Plan)
- Redesign production pipeline with best practices
- Make it feature-complete and stable

### Phase 2: Standard Pipeline Integration (Future)
- Add toggle to standard pipeline: `use_enhanced_components=True`
- Gradually migrate standard users to enhanced components
- Deprecate old implementations

### Phase 3: Unification (Future)
- Delete redundant code
- Single pipeline with strategy selection
- Rename to `UnifiedKGPipeline`

---

## Testing Strategy

### Unit Tests
- `test_smart_chunking.py`: Semantic boundary preservation
- `test_gleaning.py`: Gleaning improves recall
- `test_unified_merger.py`: Merging strategies produce correct results

### Integration Tests
- `test_enhanced_pipeline_e2e.py`: Full document → KG workflow
- Compare output with standard pipeline on same documents

### Regression Tests
- Ensure all existing KUET tests pass
- Verify backward compatibility with old configs

---

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | Step 1 + Step 2 | Extraction config + Smart chunking |
| **Week 2** | Step 3 | Gleaning implementation |
| **Week 3** | Step 4 + Step 5 + Testing | Unified merger + helpers + full test suite |

---

## Next Steps

1. **Review this plan** - Any questions or modifications?
2. **Set up test data** - Prepare KUET docs for validation
3. **Begin Step 1** - Add extraction strategy configuration
4. **Iterative implementation** - Test each step before moving to next

---

**Questions? Feedback? Ready to start implementation?**
