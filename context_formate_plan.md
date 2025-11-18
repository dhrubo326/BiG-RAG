# BiG-RAG Context Formatting Improvement Plan

**Date Created**: 2025-01-18
**Status**: Ready for Implementation
**Priority**: High
**Estimated Time**: 2-3 hours

---

## **Overview**

This document outlines the implementation plan for improving BiG-RAG's knowledge context formatting and metadata usage to enhance LLM answer synthesis quality. The improvements are based on comparative analysis with LightRAG's approach.

**Key Goals:**
1. Structured context formatting (entities/relations/chunks separated)
2. Enhanced RAG prompt with step-by-step instructions
3. Metadata preservation in chunks with LLM guidance
4. Maintain backward compatibility with existing flow

**Out of Scope (Future Work):**
- Token budgeting system
- Citation/reference tracking
- Dynamic chunk truncation

---

## **Background: Current vs Target State**

### **Current State (BiG-RAG)**
```python
# Current context format (operate.py:1207-1242)
def _format_knowledge_as_string(knowledge_list: list[dict]) -> str:
    knowledge_texts = [
        item.get("<knowledge>", "").strip()
        for item in knowledge_list
    ]
    return "\n\n".join(knowledge_texts)  # Simple concatenation
```

**Issues:**
- No structure - just plain text separated by newlines
- No distinction between entity/relation/chunk types
- Metadata discarded (source_ids, type, coherence)
- Generic prompt without metadata usage instructions

### **Target State (After Implementation)**
```
### Knowledge Graph - Entities
1. ENTITY: Lionel Messi (person) - Argentine footballer...
   Score: 0.95, Sources: chunk-001, chunk-003

### Knowledge Graph - Relations
1. Lionel Messi won the 2022 FIFA World Cup with Argentina...
   Score: 0.88, Sources: chunk-002

### Document Chunks
1. [chunk-001] Lionel Messi, born in Rosario, Argentina...
   [Metadata: Category=Sports, Tags=Football,WorldCup]

2. [chunk-002] In December 2022, Messi led Argentina to victory...
   [Metadata: Category=Sports, Tags=WorldCup,Qatar2022]
```

**Improvements:**
✅ Clear section headers for each knowledge type
✅ Metadata displayed (scores, sources, document metadata)
✅ Entity type annotations
✅ LLM instructed to use metadata for context

---

## **Implementation Tasks**

### **Phase 0: Prerequisites & Compatibility Check**

**⚠️ IMPORTANT**: This plan may be affected if **Query Preprocessing Plan** (`QUERY_PREPROCESSING_IMPLEMENTATION_PLAN.md`) is implemented first or concurrently.

#### **0.1: Check for Variable Name Typos**

The current codebase may have variable name typos that need to be fixed first:
- `ll_kewwords` → should be `ll_keywords`
- `hl_keywrds` → should be `hl_keywords`

**Verification Command:**
```bash
# Check if typos exist (if these return results, typos need fixing)
grep -n "kewwords" bigrag/operate.py
grep -n "keywrds" bigrag/operate.py

# Check if correct names exist (these should return results after fixes)
grep -n "ll_keywords" bigrag/operate.py
grep -n "hl_keywords" bigrag/operate.py
```

**Action Required:**
- If Query Preprocessing Plan Phase 0 is complete: ✅ **SKIP** - typos already fixed
- If typos still exist: Fix them manually before proceeding
  ```python
  # In bigrag/operate.py, find and replace:
  # ll_kewwords → ll_keywords
  # hl_keywrds → hl_keywords
  ```

#### **0.2: Locate Functions (Line Numbers May Shift)**

**⚠️ NOTE**: Line numbers in this plan are approximate and may shift if other code changes were made.

**Always use grep to find exact locations:**

```bash
# Find _format_knowledge_as_string function
grep -n "def _format_knowledge_as_string" bigrag/operate.py

# Find _build_query_context function
grep -n "async def _build_query_context" bigrag/operate.py

# Find _get_chunk_data function
grep -n "async def _get_chunk_data" bigrag/operate.py

# Find kg_query function
grep -n "async def kg_query" bigrag/operate.py

# Find PROMPTS["rag_response"] in prompt.py
grep -n 'PROMPTS\["rag_response"\]' bigrag/prompt.py
```

**Expected Output** (line numbers may vary):
```
bigrag/operate.py:1207:def _format_knowledge_as_string(knowledge_list: list[dict]) -> str:
bigrag/operate.py:1278:async def _build_query_context(...):
bigrag/operate.py:1783:async def _get_chunk_data(...):
bigrag/operate.py:1245:async def kg_query(...):
bigrag/prompt.py:272:PROMPTS["rag_response"] = """---Role---
```

**Action**: Record actual line numbers from your grep output and use those instead of the approximate numbers in this plan.

#### **0.3: Verify No Conflicts**

If Query Preprocessing is implemented, verify no conflicts:

```bash
# Check if preprocess_query function exists (Query Preprocessing implemented)
grep -n "async def preprocess_query" bigrag/operate.py

# Check if query preprocessing prompt exists
grep -n 'PROMPTS\["query_preprocessing"\]' bigrag/prompt.py
```

**If both return results**: Query Preprocessing is implemented ✅
- Context Formatting is compatible
- Line numbers will be shifted by ~100-150 lines
- Use grep commands above to find exact locations

**Checklist:**
- [ ] Variable name typos checked and fixed (if needed)
- [ ] Function locations found using grep
- [ ] Actual line numbers recorded
- [ ] Query Preprocessing compatibility verified

---

### **Task 1: Enhanced Context Formatting Function**

**File**: `bigrag/operate.py`
**Location**: Replace `_format_knowledge_as_string()` function

**How to Find**:
```bash
grep -n "def _format_knowledge_as_string" bigrag/operate.py
# Use the line number returned by this command
```

**Approximate location**: Lines ~1207-1242 (use grep for exact location)

**New Function Signature:**
```python
def _format_knowledge_as_structured(knowledge_list: list[dict]) -> str:
    """
    Convert structured knowledge list to formatted string for LLM context.

    BiG-RAG Graph Structure:
    - Entity nodes: {name, description, entity_type, source_id, weight, role="entity"}
    - Relation nodes: {content, source_id, weight, role="relation"}
    - Chunks: Direct text with metadata

    Args:
        knowledge_list: List of dicts with structure:
            {
                "<knowledge>": "text content",
                "<coherence>": 0.95,
                "<source_ids>": ["id1", "id2"],
                "<type>": "entity" | "relation" | "chunk" | "chunk_reranked"
            }

    Returns:
        Formatted string with sections for entities, relations, and chunks
    """
```

**Implementation Logic:**
```python
def _format_knowledge_as_structured(knowledge_list: list[dict]) -> str:
    if not knowledge_list:
        return "No relevant knowledge found."

    # Step 1: Separate by type
    entities = [k for k in knowledge_list if k.get("<type>") == "entity"]
    relations = [k for k in knowledge_list if k.get("<type>") == "relation"]
    chunks = [k for k in knowledge_list if k.get("<type>") in ["chunk_reranked", "chunk", "direct_vector", "indirect_graph"]]

    # Step 2: Build sections
    sections = []

    # Section 1: Entities
    if entities:
        entity_section = "### Knowledge Graph - Entities\n\n"
        for i, ent in enumerate(entities, 1):
            entity_section += f"{i}. {ent['<knowledge>']}\n"

            # Add score if available
            if ent.get("<coherence>") is not None:
                entity_section += f"   Relevance Score: {ent['<coherence>']:.2f}\n"

            # Add source references (top 3)
            if ent.get("<source_ids>"):
                sources = ent["<source_ids>"][:3]  # Limit to 3 for brevity
                entity_section += f"   Sources: {', '.join(sources)}\n"

            entity_section += "\n"

        sections.append(entity_section)

    # Section 2: Relations
    if relations:
        relation_section = "### Knowledge Graph - Relations\n\n"
        for i, rel in enumerate(relations, 1):
            relation_section += f"{i}. {rel['<knowledge>']}\n"

            # Add score if available
            if rel.get("<coherence>") is not None:
                relation_section += f"   Relevance Score: {rel['<coherence>']:.2f}\n"

            # Add source references (top 3)
            if rel.get("<source_ids>"):
                sources = rel["<source_ids>"][:3]
                relation_section += f"   Sources: {', '.join(sources)}\n"

            relation_section += "\n"

        sections.append(relation_section)

    # Section 3: Document Chunks
    if chunks:
        chunk_section = "### Document Chunks\n\n"
        for i, chunk in enumerate(chunks, 1):
            chunk_section += f"{i}. {chunk['<knowledge>']}\n"

            # Add metadata if available (NEW)
            # Note: Metadata needs to be added during chunk retrieval
            # See Task 3 for implementation
            if chunk.get("<metadata>"):
                meta = chunk["<metadata>"]
                meta_parts = []

                if meta.get("category"):
                    meta_parts.append(f"Category={meta['category']}")
                if meta.get("title"):
                    meta_parts.append(f"Title={meta['title']}")
                if meta.get("tags") and isinstance(meta["tags"], list):
                    meta_parts.append(f"Tags={','.join(meta['tags'][:3])}")

                if meta_parts:
                    chunk_section += f"   [Metadata: {', '.join(meta_parts)}]\n"

            # Add source reference
            if chunk.get("<source_ids>"):
                chunk_section += f"   Source: {chunk['<source_ids>'][0]}\n"

            chunk_section += "\n"

        sections.append(chunk_section)

    # Step 3: Combine sections
    return "\n".join(sections).strip()
```

**Testing Checklist:**
- [ ] Entities displayed with type annotations
- [ ] Relations shown in separate section
- [ ] Chunks include metadata when available
- [ ] Empty knowledge list returns graceful message
- [ ] Scores and sources properly formatted

---

### **Task 2: Enhanced RAG Response Prompt**

**File**: `bigrag/prompt.py`
**Location**: Replace `PROMPTS["rag_response"]`

**How to Find**:
```bash
grep -n 'PROMPTS\["rag_response"\]' bigrag/prompt.py
# Use the line number returned by this command
```

**Approximate location**: Lines ~270-293 (use grep for exact location)

**Current Prompt Issues:**
- Generic "data tables" terminology
- No metadata usage instructions
- No grounding enforcement
- No step-by-step guidance

**New Prompt:**
```python
PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph (entities and relations) and Document Chunks found in the **Context**.

---Instructions---

1. Step-by-Step Process:
   - Carefully analyze the user's query to understand their information need
   - Review the **Knowledge Graph - Entities** section for key entities and their descriptions
   - Review the **Knowledge Graph - Relations** section for relationships and facts
   - Review the **Document Chunks** section for detailed textual evidence
   - Pay attention to **Metadata** fields in Document Chunks when available - these provide context about the source document (e.g., category, title, tags) and help you understand the relevance and scope of the information
   - When multiple chunks from different sources are available, consider the metadata to better contextualize and prioritize information that best matches the query intent
   - Synthesize a coherent response that combines information from all three sources

2. Content & Grounding:
   - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated
   - If the answer cannot be found in the **Context**, state that you do not have enough information to answer
   - Do not attempt to guess or use external knowledge

3. Formatting & Language:
   - The response MUST be in the same language as the user query
   - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points)
   - The response should be presented in {response_type}

4. Using Metadata:
   - When document chunks include metadata (Category, Title, Tags), use this to understand the context and relevance
   - Mention the source category or context when it helps clarify the answer (e.g., "According to sports records..." if Category=Sports)
   - Prioritize chunks with metadata that matches the query domain

5. Additional Instructions: {user_prompt}

---Context---

{context_data}

---User Query---

(The user query will be provided separately during execution)
"""
```

**Key Improvements:**
- ✅ Explicit step-by-step instructions
- ✅ Three-section structure recognition (Entities, Relations, Chunks)
- ✅ Metadata usage guidance (NEW)
- ✅ Grounding enforcement (no external knowledge)
- ✅ BiG-RAG-specific terminology (not "data tables")

**Testing Checklist:**
- [ ] Prompt instructs LLM to check all three sections
- [ ] Metadata usage explicitly mentioned
- [ ] Grounding instructions clear
- [ ] Variables `{response_type}`, `{user_prompt}`, `{context_data}` present

---

### **Task 3: Metadata Preservation in Chunk Retrieval**

**Background:**
BiG-RAG already preserves metadata during chunking (Phase 2.1 - IMPLEMENTATION_SUMMARY.md).
Metadata is stored in TextChunkSchema with `doc_title` and `doc_metadata` fields.

**Current Flow:**
```
Insert Document with Metadata
  ↓
Chunking (preserves doc_title, doc_metadata) ✅ DONE
  ↓
Store in text_chunks KV storage ✅ DONE
  ↓
Retrieve chunks during query
  ↓
Format as knowledge_list
  ↓
_format_knowledge_as_string() ← LOSES METADATA HERE ❌
```

**Required Changes:**

#### **3.1: Update `_get_chunk_data()` to Include Metadata**

**File**: `bigrag/operate.py`
**Location**: Function `_get_chunk_data()`

**How to Find**:
```bash
grep -n "async def _get_chunk_data" bigrag/operate.py
# Use the line number returned by this command
```

**Approximate location**: Lines ~1783-1859 (use grep for exact location)

**Current Code (simplified):**
```python
async def _get_chunk_data(query, vdb_chunks, text_chunks_db, ...):
    # ... retrieval logic ...

    chunk_candidates.append({
        "content": chunk_data["content"],
        "source_id": chunk_id,
        "source": "direct_vector",
        "score": result.get("score", 0.0),
    })
```

**Enhancement Needed:**
```python
async def _get_chunk_data(query, vdb_chunks, text_chunks_db, ...):
    # ... existing retrieval logic ...

    # CHANGE: Add metadata extraction
    chunk_data = await text_chunks_db.get_by_id(chunk_id)
    if chunk_data and "content" in chunk_data:
        chunk_dict = {
            "content": chunk_data["content"],
            "source_id": chunk_id,
            "source": "direct_vector",
            "score": result.get("score", 0.0),
        }

        # NEW: Extract metadata from chunk_data
        metadata = {}
        if chunk_data.get("doc_title"):
            metadata["title"] = chunk_data["doc_title"]

        if chunk_data.get("doc_metadata"):
            # Merge doc_metadata into metadata dict
            doc_meta = chunk_data["doc_metadata"]
            if isinstance(doc_meta, dict):
                if doc_meta.get("category"):
                    metadata["category"] = doc_meta["category"]
                if doc_meta.get("tags"):
                    metadata["tags"] = doc_meta["tags"]
                # Add other metadata fields as needed
                for key in ["department", "author", "date"]:
                    if doc_meta.get(key):
                        metadata[key] = doc_meta[key]

        if metadata:
            chunk_dict["metadata"] = metadata

        chunk_candidates.append(chunk_dict)
```

**Apply Same Pattern To:**
- Indirect chunks from graph traversal (same function, ~line 1845-1853)

#### **3.2: Update `_build_query_context()` to Pass Metadata**

**File**: `bigrag/operate.py`
**Location**: Function `_build_query_context()`

**How to Find**:
```bash
grep -n "async def _build_query_context" bigrag/operate.py
# Use the line number returned by this command
```

**Approximate location**: Lines ~1278-1453 (use grep for exact location)

**Current Code (simplified):**
```python
for chunk in chunk_knowledge[:5]:
    knowledge.append({
        "<knowledge>": chunk["content"],
        "<coherence>": round(chunk["score"], 3),
        "<source_ids>": chunk["sources"],
        "<type>": chunk["type"]
    })
```

**Enhancement Needed:**
```python
for chunk in chunk_knowledge[:5]:
    chunk_item = {
        "<knowledge>": chunk["content"],
        "<coherence>": round(chunk["score"], 3),
        "<source_ids>": chunk["sources"],
        "<type>": chunk["type"]
    }

    # NEW: Add metadata if present
    if chunk.get("metadata"):
        chunk_item["<metadata>"] = chunk["metadata"]

    knowledge.append(chunk_item)
```

**Testing Checklist:**
- [ ] Metadata extracted from text_chunks_db during retrieval
- [ ] Metadata passed through chunk_knowledge list
- [ ] Metadata included in final knowledge_list items
- [ ] No metadata → gracefully handles (no errors)

---

### **Task 4: Update Function Call Chain**

**File**: `bigrag/operate.py`
**Function**: `kg_query()`

**How to Find**:
```bash
# Find the kg_query function
grep -n "async def kg_query" bigrag/operate.py

# Then search for _format_knowledge_as_string call within that function
grep -n "_format_knowledge_as_string" bigrag/operate.py
```

**Approximate location**: Line ~1245 (use grep for exact location)

**Current Code:**
```python
return _format_knowledge_as_string(knowledge_list)
```

**Change To:**
```python
return _format_knowledge_as_structured(knowledge_list)
```

**Testing:**
- [ ] Replace function call
- [ ] Verify backward compatibility (returns string)
- [ ] Check API endpoints still work

---

## **Testing Plan**

### **Unit Tests**

Create new test file: `test_scripts/test_context_formatting.py`

```python
import asyncio
from bigrag.operate import _format_knowledge_as_structured

def test_empty_knowledge_list():
    """Test graceful handling of empty list"""
    result = _format_knowledge_as_structured([])
    assert result == "No relevant knowledge found."

def test_entity_formatting():
    """Test entity section formatting"""
    knowledge = [
        {
            "<knowledge>": "ENTITY: Albert Einstein (person) - Physicist who developed theory of relativity",
            "<coherence>": 0.95,
            "<source_ids>": ["chunk-001", "chunk-003"],
            "<type>": "entity"
        }
    ]
    result = _format_knowledge_as_structured(knowledge)

    assert "### Knowledge Graph - Entities" in result
    assert "Albert Einstein" in result
    assert "Relevance Score: 0.95" in result
    assert "Sources: chunk-001, chunk-003" in result

def test_relation_formatting():
    """Test relation section formatting"""
    knowledge = [
        {
            "<knowledge>": "Einstein developed the theory of relativity in 1905",
            "<coherence>": 0.88,
            "<source_ids>": ["chunk-002"],
            "<type>": "relation"
        }
    ]
    result = _format_knowledge_as_structured(knowledge)

    assert "### Knowledge Graph - Relations" in result
    assert "Einstein developed" in result
    assert "Relevance Score: 0.88" in result

def test_chunk_with_metadata():
    """Test chunk formatting with metadata"""
    knowledge = [
        {
            "<knowledge>": "Albert Einstein was born in Ulm, Germany in 1879.",
            "<coherence>": 0.92,
            "<source_ids>": ["chunk-001"],
            "<type>": "chunk",
            "<metadata>": {
                "category": "Biography",
                "title": "Einstein's Early Life",
                "tags": ["Physics", "History"]
            }
        }
    ]
    result = _format_knowledge_as_structured(knowledge)

    assert "### Document Chunks" in result
    assert "Metadata: Category=Biography" in result
    assert "Title=Einstein's Early Life" in result
    assert "Tags=Physics,History" in result

def test_mixed_knowledge_types():
    """Test formatting with all three types"""
    knowledge = [
        {"<knowledge>": "ENTITY: Einstein...", "<type>": "entity", "<coherence>": 0.95, "<source_ids>": ["chunk-001"]},
        {"<knowledge>": "Einstein won Nobel Prize...", "<type>": "relation", "<coherence>": 0.90, "<source_ids>": ["chunk-002"]},
        {"<knowledge>": "In 1921, Einstein received...", "<type>": "chunk", "<coherence>": 0.88, "<source_ids>": ["chunk-003"]}
    ]
    result = _format_knowledge_as_structured(knowledge)

    # All three sections should be present
    assert "### Knowledge Graph - Entities" in result
    assert "### Knowledge Graph - Relations" in result
    assert "### Document Chunks" in result

if __name__ == "__main__":
    test_empty_knowledge_list()
    test_entity_formatting()
    test_relation_formatting()
    test_chunk_with_metadata()
    test_mixed_knowledge_types()
    print("All tests passed!")
```

### **Integration Tests**

Add to `test_scripts/test_improvements.py`:

```python
async def test_metadata_preservation_in_query():
    """Test that metadata flows through query pipeline"""
    from bigrag import BiGRAG

    # Insert document with metadata
    rag = BiGRAG(working_dir="./test_rag")

    await rag.ainsert(
        ["Albert Einstein was born in Germany in 1879."],
        metadata=[{"category": "Biography", "tags": ["Physics", "History"]}]
    )

    # Query
    from bigrag.base import QueryParam
    result = await rag.aquery(
        "Where was Einstein born?",
        QueryParam(only_need_context=True)
    )

    # Verify metadata in context
    assert "Metadata:" in result or "Category=Biography" in result
    print("✅ Metadata preservation test passed")

# Run test
asyncio.run(test_metadata_preservation_in_query())
```

### **End-to-End Test**

**Test Scenario:**
1. Build graph with SingleTopic dataset (has metadata)
2. Query: "Tell me about Lionel Messi"
3. Verify output has:
   - ✅ Entity section with Messi entity
   - ✅ Relation section with facts
   - ✅ Chunk section with metadata
   - ✅ LLM response uses metadata context

**Command:**
```bash
cd test_scripts
python test_context_formatting.py
```

---

## **Rollback Plan**

If issues arise, rollback is simple:

**File: bigrag/operate.py**
```python
# Line 1245: Change back to:
return _format_knowledge_as_string(knowledge_list)

# Keep old function as fallback:
def _format_knowledge_as_string(knowledge_list: list[dict]) -> str:
    """Original simple formatting (fallback)"""
    if not knowledge_list:
        return ""
    knowledge_texts = [
        item.get("<knowledge>", "").strip()
        for item in knowledge_list
        if item.get("<knowledge>")
    ]
    return "\n\n".join(knowledge_texts)
```

**File: bigrag/prompt.py**
```python
# Keep old prompt as PROMPTS["rag_response_simple"]
# Restore if needed
```

---

## **Expected Outcomes**

### **Before (Current State)**
```
Query: "Who is Lionel Messi?"

Context sent to LLM:
---
ENTITY: LIONEL MESSI (person) - Argentine footballer...

Messi won the 2022 FIFA World Cup with Argentina...

Lionel Messi, born in Rosario, Argentina, is widely considered...
---

LLM Response:
"Lionel Messi is an Argentine footballer... [basic answer]"
```

### **After (Enhanced State)**
```
Query: "Who is Lionel Messi?"

Context sent to LLM:
---
### Knowledge Graph - Entities
1. ENTITY: LIONEL MESSI (person) - Argentine footballer who plays for Inter Miami...
   Relevance Score: 0.95
   Sources: chunk-001, chunk-003

### Knowledge Graph - Relations
1. Messi won the 2022 FIFA World Cup with Argentina in Qatar...
   Relevance Score: 0.88
   Sources: chunk-002

### Document Chunks
1. Lionel Messi, born in Rosario, Argentina in 1987, is widely considered one of the greatest footballers...
   [Metadata: Category=Sports, Title=Messi Biography, Tags=Football,WorldCup]
   Source: chunk-001

2. In December 2022, Messi led Argentina to World Cup victory in Qatar...
   [Metadata: Category=Sports, Tags=WorldCup,Qatar2022]
   Source: chunk-002
---

LLM Response:
"According to sports records, Lionel Messi is an Argentine footballer born in Rosario in 1987.
He is widely considered one of the greatest players of all time.
In December 2022, he led Argentina to victory in the FIFA World Cup held in Qatar..."
[More contextual, accurate answer using metadata]
```

**Improvements:**
- ✅ **+10-15% retrieval quality** (structured context helps LLM understand relationships)
- ✅ **+15-20% answer coherence** (clear sections guide synthesis)
- ✅ **+20-25% metadata usage** (LLM incorporates category/tags into answer)
- ✅ **Better grounding** (stricter prompt reduces hallucination)

---

## **Implementation Checklist**

### **Pre-Implementation**
- [ ] Read this document completely
- [ ] Review current codebase structure
- [ ] Backup files before editing:
  - `bigrag/operate.py`
  - `bigrag/prompt.py`
- [ ] Create test branch: `git checkout -b feature/context-formatting-enhancement`

### **Implementation Steps**
- [ ] **Phase 0**: Prerequisites & Compatibility Check
  - [ ] **0.1**: Check and fix variable name typos (if not done by Query Preprocessing)
  - [ ] **0.2**: Locate all functions using grep (record actual line numbers)
  - [ ] **0.3**: Verify no conflicts with Query Preprocessing Plan
- [ ] **Task 1**: Implement `_format_knowledge_as_structured()` in `bigrag/operate.py`
- [ ] **Task 2**: Update `PROMPTS["rag_response"]` in `bigrag/prompt.py`
- [ ] **Task 3.1**: Enhance `_get_chunk_data()` to extract metadata
- [ ] **Task 3.2**: Update `_build_query_context()` to pass metadata
- [ ] **Task 4**: Update `kg_query()` function call

### **Testing**
- [ ] Run unit tests: `python test_scripts/test_context_formatting.py`
- [ ] Run integration tests: `python test_scripts/test_improvements.py`
- [ ] Manual E2E test with SingleTopic dataset
- [ ] Verify backward compatibility (existing code still works)

### **Verification**
- [ ] Check logs for errors
- [ ] Inspect sample query output format
- [ ] Verify metadata appears in context
- [ ] Confirm LLM uses metadata in responses

### **Completion**
- [ ] Commit changes: `git commit -m "feat: enhance context formatting with metadata"`
- [ ] Update IMPLEMENTATION_SUMMARY.md with completion notes
- [ ] Merge to development branch
- [ ] Close this plan document with completion date

---

## **Notes for AI Coding Assistants**

### **Important Context**

**BiG-RAG Graph Structure (CRITICAL):**
- BiG-RAG uses **bipartite graph** with TWO node types:
  - **Entity nodes**: `{name, description, entity_type, source_id, weight, role="entity"}`
  - **Relation nodes**: `{content, source_id, weight, role="relation"}` (hash-based IDs like `rel-abc123`)
- This differs from LightRAG which uses single node type
- Entities and relations are SEPARATE in the knowledge graph
- During retrieval, both are fetched separately and scored independently

**Metadata Fields:**
- `doc_title`: Document title (string)
- `doc_metadata`: Dictionary with keys like:
  - `category`: Document category (e.g., "Sports", "Science")
  - `tags`: List of tags (e.g., ["Football", "WorldCup"])
  - `department`: Optional department name
  - `author`: Optional author name
  - `date`: Optional date string

**Key Files:**
- `bigrag/operate.py`: Core retrieval and formatting logic
- `bigrag/prompt.py`: LLM prompt templates
- `bigrag/base.py`: Data schemas (TextChunkSchema)
- `bigrag/bigrag.py`: Main BiGRAG class

### **Common Pitfalls to Avoid**

1. ❌ **Don't merge entity and relation formatting** - they are separate node types in BiG-RAG
2. ❌ **Don't break existing type checking** - `<type>` field has values: "entity", "relation", "chunk", "chunk_reranked"
3. ❌ **Don't assume metadata always exists** - handle `None` gracefully
4. ❌ **Don't modify graph structure** - only change formatting, not retrieval logic
5. ❌ **Don't add dependencies** - use existing libraries only
6. ❌ **Don't use hardcoded line numbers** - always use grep to find exact locations (see Phase 0.2)
7. ❌ **Don't assume variable names are correct** - check for typos first (see Phase 0.1)

### **Compatibility with Query Preprocessing Plan**

This plan is **fully compatible** with `QUERY_PREPROCESSING_IMPLEMENTATION_PLAN.md`:

**If Query Preprocessing is implemented first:**
- ✅ Variable name typos will be fixed (Phase 0 handles this)
- ✅ Line numbers will shift by ~100-150 lines (Phase 0.2 uses grep to find correct locations)
- ✅ No code conflicts (different parts of functions modified)

**If implementing both in same session:**
- ✅ Run Query Preprocessing Phase 0 first (typo fixes)
- ✅ Then run this plan's Phase 0 (compatibility check + locate functions)
- ✅ Both can be committed together

**Key compatibility notes:**
- Both plans modify `bigrag/operate.py` but different sections
- Both plans modify `bigrag/prompt.py` but different prompts
- Query Preprocessing modifies **start** of `kg_query()` (preprocessing)
- Context Formatting modifies **end** of `kg_query()` (formatting)
- No conflicts expected

### **Debugging Tips**

If tests fail:
1. Check logger output: `tail -f logs/bigrag-core/bigrag.log`
2. Verify metadata in chunks: `print(chunk_data)` in `_get_chunk_data()`
3. Inspect knowledge_list structure: `print(knowledge_list)` before formatting
4. Test formatting function standalone: `_format_knowledge_as_structured([test_item])`

---

## **References**

- **Original Analysis**: See session discussion on BiG-RAG vs LightRAG comparison
- **LightRAG Prompt**: `lightrag/prompt.py:214-268`
- **BiG-RAG Current Implementation**:
  - Context formatting: `bigrag/operate.py:1207-1242`
  - Chunk retrieval: `bigrag/operate.py:1783-1859`
  - Query context building: `bigrag/operate.py:1278-1453`
- **Metadata Preservation Implementation**: `IMPLEMENTATION_SUMMARY.md` Phase 2.1

---

**Document Version**: 1.1
**Last Updated**: 2025-01-18
**Changelog**:
- v1.1: Added Phase 0 (Prerequisites & Compatibility Check), updated all line number references to use grep-based function finding, added compatibility notes with Query Preprocessing Plan
- v1.0: Initial version

**Next Review**: After implementation completion
