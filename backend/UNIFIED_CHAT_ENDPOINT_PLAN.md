# Unified Chat Endpoint Implementation Plan

**Version:** 2.0 (Revised)
**Date:** 2025-01-26
**Goal:** Create `/api/unified/chat` endpoint aligned with existing `/api/unified/query` architecture

---

## Overview

Create a comprehensive chat endpoint that:
1. **Aligns with unified architecture** - Uses same patterns as `/api/unified/query`
2. **Flexible output modes** - Context only, answer only, or both
3. **Auto-routing by default** - LLM-based subgraph selection
4. **Force single-mode** - Optional `force_subgraphs` parameter
5. **Enhanced pipeline aware** - Works with all graph types

---

## Architecture Alignment

### Existing Unified Endpoints (Keep as Reference)

```python
# /api/unified/query - Returns context only
POST /api/unified/query
{
    "query": "...",
    "force_subgraphs": ["kuet_test"],  # Optional
    "top_k": 10,
    "enable_reranking": true
}

# /api/unified/ask - Simple wrapper around query
POST /api/unified/ask
{
    "question": "...",
    "top_k": 5
}
```

### New Unified Chat Endpoint (Add)

```python
# /api/unified/chat - Returns answer, context, or both
POST /api/unified/chat
{
    "messages": [...],
    "output_mode": "answer_with_context",  # NEW
    "force_subgraphs": ["kuet_test"],      # Optional (like /query)
    "top_k": 10,
    "enable_reranking": true,
    # ... other params
}
```

**Key Insight:** `/api/unified/chat` = `/api/unified/query` + LLM synthesis + flexible output modes

---

## Implementation Plan

### Phase 1: Request/Response Models (Week 1)

#### 1.1 Request Model

**File:** `backend/api/routes/unified.py` (add to existing file)

```python
class UnifiedChatRequest(BaseModel):
    """
    Unified chat request - extends UnifiedQueryRequest with LLM parameters.

    Inherits auto-routing behavior from /api/unified/query:
    - By default: LLM routes to relevant subgraph(s)
    - With force_subgraphs: Uses specific subgraph(s) (single-mode behavior)
    """

    # ============================================================================
    # Message Handling (OpenAI-compatible)
    # ============================================================================
    messages: List[Message] = Field(
        ...,
        description="Chat messages in OpenAI format: [{role: 'user', content: '...'}]"
    )

    # ============================================================================
    # Retrieval Configuration (Same as UnifiedQueryRequest)
    # ============================================================================
    use_rag: bool = Field(
        True,
        description="Enable knowledge graph retrieval (false = LLM-only mode)"
    )

    force_subgraphs: Optional[List[str]] = Field(
        None,
        description="Force specific subgraph(s) - bypasses auto-routing (single-mode behavior)"
    )

    # ============================================================================
    # Retrieval Parameters (Same as UnifiedQueryRequest)
    # ============================================================================
    mode: str = Field(
        "hybrid",
        description="Retrieval mode: hybrid (default) | local | global | naive"
    )

    top_k: int = Field(
        60,
        ge=1,
        le=100,
        description="Number of items to retrieve from vector DBs"
    )

    num_kg_in_context: int = Field(
        15,
        ge=1,
        le=50,
        description="Number of KG relations in final context"
    )

    num_chunks_in_context: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of text chunks in final context"
    )

    enable_reranking: bool = Field(
        True,
        description="Enable semantic reranking for chunks"
    )

    # ============================================================================
    # LLM Parameters (NEW)
    # ============================================================================
    model: str = Field(
        "gpt-4o-mini",
        description="LLM model name"
    )

    llm_provider: Optional[str] = Field(
        None,
        description="LLM provider: openai | huggingface | anthropic (auto-detect if None)"
    )

    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 1.0 = creative)"
    )

    max_tokens: int = Field(
        4096,
        ge=1,
        le=16384,
        description="Maximum tokens in generated answer"
    )

    # ============================================================================
    # Output Configuration (NEW - KEY FEATURE)
    # ============================================================================
    output_mode: str = Field(
        "answer_with_context",
        description="Output mode: context_only | answer_only | answer_with_context"
    )

    include_metadata: bool = Field(
        True,
        description="Include routing and execution metadata (same as /api/unified/query)"
    )

    include_retrieval_metrics: bool = Field(
        False,
        description="Include detailed retrieval performance metrics"
    )

    # ============================================================================
    # Language
    # ============================================================================
    language: Optional[str] = Field(
        None,
        description="Response language override (defaults to DEFAULT_LANGUAGE from .env)"
    )


    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "How many seats in KUET CSE?"}
                ],
                "output_mode": "answer_with_context",
                "use_rag": True,
                "top_k": 10,
                "enable_reranking": True,
                "model": "gpt-4o-mini",
                "temperature": 0.7
            }
        }
```

#### 1.2 Response Model

```python
class UnifiedChatResponse(BaseModel):
    """
    Unified chat response - extends UnifiedQueryResponse with answer field.
    """

    # ============================================================================
    # Answer (if output_mode includes answer)
    # ============================================================================
    answer: Optional[str] = Field(
        None,
        description="Synthesized answer from LLM (None if output_mode=context_only)"
    )

    # ============================================================================
    # Context (if output_mode includes context)
    # ============================================================================
    contexts: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Retrieved context items (None if output_mode=answer_only)"
    )

    num_contexts: Optional[int] = Field(
        None,
        description="Number of context items retrieved"
    )

    # ============================================================================
    # Routing Metadata (Same as UnifiedQueryResponse)
    # ============================================================================
    routing: Optional[Dict[str, Any]] = Field(
        None,
        description="Subgraph routing decision (if include_metadata=True)"
    )

    # ============================================================================
    # Performance Metrics
    # ============================================================================
    retrieval_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Retrieval performance metrics (if include_retrieval_metrics=True)"
    )

    llm_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="LLM performance metrics (tokens, latency)"
    )

    execution_time_ms: float = Field(
        ...,
        description="Total execution time in milliseconds"
    )

    # ============================================================================
    # Request Echo
    # ============================================================================
    output_mode: str = Field(
        ...,
        description="Output mode used for this response"
    )


    class Config:
        json_schema_extra = {
            "example": {
                "answer": "KUET CSE department has 120 seats for undergraduate admission.",
                "contexts": [
                    {
                        "type": "chunk",
                        "content": "KUET CSE has 120 seats...",
                        "score": 0.95,
                        "metadata": {"title": "KUET Admission Guide"}
                    }
                ],
                "num_contexts": 5,
                "routing": {
                    "subgraphs": ["kuet_test"],
                    "confidence": 0.98,
                    "reasoning": "Query asks about KUET admission..."
                },
                "llm_metrics": {
                    "model": "gpt-4o-mini",
                    "prompt_tokens": 320,
                    "completion_tokens": 45,
                    "total_tokens": 365,
                    "latency_ms": 850
                },
                "execution_time_ms": 1245,
                "output_mode": "answer_with_context"
            }
        }
```

---

### Phase 2: Core Endpoint Implementation (Week 1)

#### 2.1 Main Endpoint

**File:** `backend/api/routes/unified.py` (add to existing file)

```python
@router.post("/chat", summary="Unified chat with RAG + LLM synthesis")
async def unified_chat(request: UnifiedChatRequest) -> UnifiedChatResponse:
    """
    Unified chat endpoint with automatic subgraph routing and LLM synthesis.

    This endpoint extends /api/unified/query with answer generation:
    - Inherits auto-routing behavior (LLM selects relevant subgraph(s))
    - Supports single-mode via force_subgraphs parameter
    - Flexible output: context-only, answer-only, or both

    **Routing Modes:**

    1. **Auto-routing (default):**
       ```json
       {
         "messages": [{"role": "user", "content": "KUET CSE seats?"}],
         "output_mode": "answer_with_context"
       }
       ```
       → LLM routes to relevant subgraph (e.g., "kuet_test")

    2. **Force specific subgraph (single-mode):**
       ```json
       {
         "messages": [...],
         "force_subgraphs": ["kuet_test"],
         "output_mode": "answer_with_context"
       }
       ```
       → Uses only specified subgraph(s), bypasses routing

    **Output Modes:**

    - `context_only`: Returns retrieved contexts (no LLM synthesis)
      → Same as /api/unified/query but with chat message format

    - `answer_only`: Returns synthesized answer (no context details)
      → Lightweight response for simple Q&A

    - `answer_with_context`: Returns both answer and contexts (default)
      → Full RAG pipeline with transparency

    **Example Requests:**

    ```bash
    # Full RAG pipeline (auto-routing)
    curl -X POST "http://localhost:8001/api/unified/chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "How many seats in KUET CSE?"}],
        "output_mode": "answer_with_context",
        "top_k": 10
      }'

    # Force specific subgraph (single-mode behavior)
    curl -X POST "http://localhost:8001/api/unified/chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "KUET CSE seats?"}],
        "force_subgraphs": ["kuet_test"],
        "output_mode": "answer_only"
      }'

    # Context-only (no LLM synthesis, same as /api/unified/query)
    curl -X POST "http://localhost:8001/api/unified/chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "KUET CSE seats?"}],
        "output_mode": "context_only"
      }'

    # LLM-only (no retrieval)
    curl -X POST "http://localhost:8001/api/unified/chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "use_rag": false,
        "output_mode": "answer_only"
      }'
    ```

    **Returns:** UnifiedChatResponse with answer, contexts, and metadata
    """
    executor = dependencies.get_unified_executor()

    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Unified mode not enabled. Start server with --unified flag."
        )

    start_time = time.time()

    try:
        # ====================================================================
        # Step 1: Extract user query from messages
        # ====================================================================
        user_query = extract_user_query(request.messages)
        system_prompt = extract_system_prompt(request.messages)
        history = extract_history(request.messages)

        # ====================================================================
        # Step 2: Retrieval (if use_rag=True)
        # ====================================================================
        contexts = None
        routing_info = None
        retrieval_metrics = None

        if request.use_rag and request.output_mode != "answer_only":
            # Uses same retrieval logic as /api/unified/query
            retrieval_start = time.time()

            query_param = QueryParam(
                mode=request.mode,
                only_need_context=True,  # Always return structured contexts
                top_k=request.top_k,
                num_kg_in_context=request.num_kg_in_context,
                num_chunks_in_context=request.num_chunks_in_context,
                enable_reranking=request.enable_reranking,
                language=request.language
            )

            result = await executor.query(
                query=user_query,
                query_param=query_param,
                force_subgraphs=request.force_subgraphs,  # Single-mode support
                include_metadata=request.include_metadata
            )

            contexts = result['results']
            routing_info = result.get('routing')

            if request.include_retrieval_metrics:
                retrieval_metrics = {
                    'retrieval_time_ms': (time.time() - retrieval_start) * 1000,
                    'num_results': len(contexts),
                    'subgraphs_queried': routing_info.get('subgraphs', []) if routing_info else []
                }

        # ====================================================================
        # Step 3: Answer Generation (if output_mode includes answer)
        # ====================================================================
        answer = None
        llm_metrics = None

        if request.output_mode in ["answer_only", "answer_with_context"]:
            llm_start = time.time()

            # Get LLM manager
            from ..core.dependencies import get_llm_manager
            llm_manager = get_llm_manager()

            # Format context for LLM
            if contexts and request.use_rag:
                context_str = format_contexts_for_llm(
                    contexts,
                    language=request.language or "English"
                )

                # Build RAG system prompt
                if not system_prompt:
                    system_prompt = build_rag_system_prompt(request.language or "English")

                # Prepend context to user query
                augmented_query = f"""Based on the following context from the knowledge graph:

{context_str}

---

Question: {user_query}

Please provide a comprehensive answer in **{request.language or 'English'}** by synthesizing information from the entities, relations, and document chunks above."""
            else:
                augmented_query = user_query
                if not system_prompt and request.language and request.language != "English":
                    system_prompt = f"You are a helpful AI assistant. Always respond in {request.language} language."

            # Call LLM
            answer = await llm_manager.complete(
                prompt=augmented_query,
                provider=request.llm_provider,
                model=request.model,
                system_prompt=system_prompt,
                history_messages=history,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            # Calculate metrics
            llm_metrics = {
                'model': request.model,
                'prompt_tokens': count_tokens(augmented_query + (system_prompt or ""), request.model),
                'completion_tokens': count_tokens(answer, request.model),
                'latency_ms': (time.time() - llm_start) * 1000
            }
            llm_metrics['total_tokens'] = llm_metrics['prompt_tokens'] + llm_metrics['completion_tokens']

        # ====================================================================
        # Step 4: Format Response
        # ====================================================================
        execution_time_ms = (time.time() - start_time) * 1000

        response = UnifiedChatResponse(
            answer=answer if request.output_mode in ["answer_only", "answer_with_context"] else None,
            contexts=contexts if request.output_mode in ["context_only", "answer_with_context"] else None,
            num_contexts=len(contexts) if contexts else None,
            routing=routing_info if request.include_metadata else None,
            retrieval_metrics=retrieval_metrics,
            llm_metrics=llm_metrics,
            execution_time_ms=execution_time_ms,
            output_mode=request.output_mode
        )

        return response

    except Exception as e:
        logger.error(f"[Unified Chat] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )
```

---

### Phase 3: Helper Functions (Week 1)

#### 3.1 Message Extraction Utilities

```python
def extract_user_query(messages: List[Dict]) -> str:
    """Extract user query from OpenAI-format messages."""
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            return msg.get('content', '')
    raise HTTPException(400, "No user message found in messages array")


def extract_system_prompt(messages: List[Dict]) -> Optional[str]:
    """Extract system prompt from messages."""
    for msg in messages:
        if msg.get('role') == 'system':
            return msg.get('content')
    return None


def extract_history(messages: List[Dict]) -> List[Dict]:
    """Extract conversation history (assistant messages)."""
    history = []
    for msg in messages:
        if msg.get('role') == 'assistant':
            history.append({
                'role': 'assistant',
                'content': msg.get('content', '')
            })
    return history
```

#### 3.2 Context Formatting

```python
def format_contexts_for_llm(
    contexts: List[Dict],
    language: str = "English"
) -> str:
    """
    Format retrieved contexts for LLM consumption.

    Groups contexts by type (entities, relations, chunks) and formats
    with metadata for better LLM understanding.
    """
    entities = []
    relations = []
    chunks = []

    for ctx in contexts:
        ctx_type = ctx.get('<type>', 'unknown')
        content = ctx.get('<knowledge>', str(ctx))
        score = ctx.get('<coherence>', 0.0)

        if ctx_type == 'entity':
            entities.append(f"- {content} (relevance: {score:.2f})")
        elif ctx_type == 'relation':
            relations.append(f"- {content} (relevance: {score:.2f})")
        elif ctx_type == 'chunk':
            metadata = ctx.get('<metadata>', {})
            meta_str = ""
            if metadata:
                meta_parts = []
                if metadata.get('title'):
                    meta_parts.append(f"Title: {metadata['title']}")
                if metadata.get('category'):
                    meta_parts.append(f"Category: {metadata['category']}")
                if metadata.get('tags'):
                    meta_parts.append(f"Tags: {', '.join(metadata['tags'])}")
                if meta_parts:
                    meta_str = f" [{', '.join(meta_parts)}]"

            chunks.append(f"- {content}{meta_str} (relevance: {score:.2f})")

    sections = []

    if entities:
        sections.append("### Knowledge Graph - Entities\n" + "\n".join(entities))

    if relations:
        sections.append("### Knowledge Graph - Relations\n" + "\n".join(relations))

    if chunks:
        sections.append("### Document Chunks\n" + "\n".join(chunks))

    if not sections:
        return "No relevant knowledge found."

    return "\n\n".join(sections)


def build_rag_system_prompt(language: str = "English") -> str:
    """Build RAG-optimized system prompt."""
    return f"""You are an expert AI assistant specializing in synthesizing information from a knowledge graph.

Instructions:
- Analyze the provided context which contains three types of information:
  1. Knowledge Graph Entities (key concepts and their descriptions)
  2. Knowledge Graph Relations (facts and relationships between concepts)
  3. Document Chunks (detailed textual evidence with metadata)
- Pay attention to metadata fields (Category, Title, Tags) which provide context about the source
- Synthesize a comprehensive, well-structured answer using information from all three sources
- Be accurate and cite relevant information when appropriate
- If the context doesn't fully answer the question, acknowledge what you know and what's uncertain
- **IMPORTANT: You MUST respond in {language} language.** Match the language of the user's question."""


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken."""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return len(text) // 4  # Rough estimate
```

---

### Phase 4: Update Existing Endpoints (Week 2)

#### 4.1 Deprecate Old Chat Endpoint

**File:** `backend/api/routes/llm.py`

```python
@router.post("/completions")
async def chat_completions_deprecated(request: ChatCompletionRequest):
    """
    ⚠️ DEPRECATED: Use POST /api/unified/chat instead.

    This endpoint will be removed in v2.0 (July 2025).

    Migration:
        OLD: POST /chat/completions
        NEW: POST /api/unified/chat with output_mode="answer_only"
    """
    import warnings
    warnings.warn(
        "POST /chat/completions is deprecated. Use POST /api/unified/chat instead.",
        DeprecationWarning
    )

    # Could redirect to new endpoint or keep existing implementation
    # with deprecation warning in response headers
    response = await chat_completions(request, rag, llm_manager, embedding_manager)
    response.headers["X-Deprecated"] = "true"
    response.headers["X-Deprecated-Replacement"] = "/api/unified/chat"
    return response
```

#### 4.2 Update /api/unified/ask Documentation

**File:** `backend/api/routes/unified.py`

```python
@router.post("/ask", summary="Simple unified ask (context-only)")
async def unified_ask(request: AskRequest) -> Dict:
    """
    Simple search endpoint - returns context only.

    💡 **TIP:** For answers with LLM synthesis, use /api/unified/chat instead.

    This endpoint is a lightweight wrapper around /api/unified/query.
    For full RAG pipeline with answer generation, use:

    ```bash
    POST /api/unified/chat
    {
      "messages": [{"role": "user", "content": "Your question"}],
      "output_mode": "answer_with_context"
    }
    ```
    """
    # ... existing implementation
```

---

### Phase 5: OpenAPI Documentation (Week 2)

#### 5.1 Update Endpoint Tags

```python
router = APIRouter(prefix="/api/unified", tags=["Unified Subgraph"])

# Group endpoints logically
@router.post("/query", tags=["Unified Subgraph - Retrieval"])
@router.post("/ask", tags=["Unified Subgraph - Retrieval"])
@router.post("/chat", tags=["Unified Subgraph - Chat"])  # NEW
@router.post("/route", tags=["Unified Subgraph - Routing"])
```

#### 5.2 Add Examples to OpenAPI Schema

```python
class UnifiedChatRequest(BaseModel):
    # ... fields ...

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Full RAG (auto-routing)",
                    "value": {
                        "messages": [{"role": "user", "content": "KUET CSE seats?"}],
                        "output_mode": "answer_with_context",
                        "top_k": 10
                    }
                },
                {
                    "summary": "Force specific subgraph",
                    "value": {
                        "messages": [{"role": "user", "content": "KUET CSE seats?"}],
                        "force_subgraphs": ["kuet_test"],
                        "output_mode": "answer_only"
                    }
                },
                {
                    "summary": "Context only (no LLM)",
                    "value": {
                        "messages": [{"role": "user", "content": "KUET CSE seats?"}],
                        "output_mode": "context_only"
                    }
                },
                {
                    "summary": "LLM only (no retrieval)",
                    "value": {
                        "messages": [{"role": "user", "content": "What is 2+2?"}],
                        "use_rag": False,
                        "output_mode": "answer_only"
                    }
                }
            ]
        }
```

---

## API Usage Examples

### Example 1: Full RAG Pipeline (Auto-Routing)

**Most common use case - let LLM choose subgraph**

```bash
curl -X POST "http://localhost:8001/api/unified/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "How many seats in KUET CSE department?"}
    ],
    "output_mode": "answer_with_context",
    "top_k": 10,
    "enable_reranking": true
  }'
```

**Response:**
```json
{
  "answer": "KUET CSE department has 120 seats for undergraduate admission.",
  "contexts": [
    {
      "<type>": "chunk",
      "<knowledge>": "KUET CSE has 120 seats...",
      "<coherence>": 0.95,
      "<metadata>": {"title": "KUET Admission Guide", "category": "education"}
    },
    {
      "<type>": "entity",
      "<knowledge>": "KUET: Khulna University of Engineering & Technology...",
      "<coherence>": 0.89
    }
  ],
  "num_contexts": 5,
  "routing": {
    "subgraphs": ["kuet_test"],
    "confidence": 0.98,
    "reasoning": "Query asks about KUET admission information"
  },
  "llm_metrics": {
    "model": "gpt-4o-mini",
    "prompt_tokens": 320,
    "completion_tokens": 45,
    "total_tokens": 365,
    "latency_ms": 850
  },
  "execution_time_ms": 1245,
  "output_mode": "answer_with_context"
}
```

---

### Example 2: Force Specific Subgraph (Single-Mode)

**Use when you know which subgraph to query**

```bash
curl -X POST "http://localhost:8001/api/unified/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "KUET CSE admission requirements?"}
    ],
    "force_subgraphs": ["kuet_test"],
    "output_mode": "answer_only",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "KUET CSE admission requires...",
  "contexts": null,
  "num_contexts": null,
  "routing": {
    "subgraphs": ["kuet_test"],
    "mode": "forced"
  },
  "llm_metrics": {
    "model": "gpt-4o-mini",
    "prompt_tokens": 280,
    "completion_tokens": 65,
    "total_tokens": 345,
    "latency_ms": 920
  },
  "execution_time_ms": 1050,
  "output_mode": "answer_only"
}
```

---

### Example 3: Context Only (No LLM Synthesis)

**Same as /api/unified/query but with message format**

```bash
curl -X POST "http://localhost:8001/api/unified/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "KUET CSE seats?"}
    ],
    "output_mode": "context_only",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": null,
  "contexts": [
    {"<type>": "chunk", "<knowledge>": "...", "<coherence>": 0.95},
    {"<type>": "entity", "<knowledge>": "...", "<coherence>": 0.89}
  ],
  "num_contexts": 5,
  "routing": {
    "subgraphs": ["kuet_test"],
    "confidence": 0.98
  },
  "llm_metrics": null,
  "execution_time_ms": 320,
  "output_mode": "context_only"
}
```

---

### Example 4: LLM Only (No Retrieval)

**Pure LLM without knowledge graph**

```bash
curl -X POST "http://localhost:8001/api/unified/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ],
    "use_rag": false,
    "output_mode": "answer_only"
  }'
```

**Response:**
```json
{
  "answer": "2 + 2 equals 4.",
  "contexts": null,
  "num_contexts": null,
  "routing": null,
  "llm_metrics": {
    "model": "gpt-4o-mini",
    "prompt_tokens": 15,
    "completion_tokens": 8,
    "total_tokens": 23,
    "latency_ms": 450
  },
  "execution_time_ms": 455,
  "output_mode": "answer_only"
}
```

---

### Example 5: Multi-Turn Conversation

**Chat history support**

```bash
curl -X POST "http://localhost:8001/api/unified/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "How many seats in KUET CSE?"},
      {"role": "assistant", "content": "KUET CSE has 120 seats."},
      {"role": "user", "content": "What about EEE?"}
    ],
    "output_mode": "answer_only",
    "force_subgraphs": ["kuet_test"]
  }'
```

---

## Endpoint Comparison Table

| Endpoint | Returns | Routing | Use Case |
|----------|---------|---------|----------|
| `POST /api/unified/query` | Context only | Auto or forced | Retrieval-only workflows |
| `POST /api/unified/ask` | Context only | Auto | Simple retrieval wrapper |
| `POST /api/unified/chat` | Flexible | Auto or forced | **Comprehensive chat interface** |
| `POST /chat/completions` | Answer only | Single-mode | ⚠️ Deprecated (OpenAI compat) |

---

## Migration Guide

### For Frontend Developers

#### Before (Multiple Endpoints)

```typescript
// Get context
const contextRes = await fetch('/api/unified/query', {
  method: 'POST',
  body: JSON.stringify({ query: "KUET CSE seats?", top_k: 5 })
});
const { results: contexts } = await contextRes.json();

// Get answer (separate call)
const answerRes = await fetch('/chat/completions', {
  method: 'POST',
  body: JSON.stringify({
    messages: [{ role: 'user', content: "KUET CSE seats?" }],
    use_rag: true
  })
});
const { choices } = await answerRes.json();
const answer = choices[0].message.content;
```

#### After (Single Endpoint)

```typescript
// Get both in one call
const res = await fetch('/api/unified/chat', {
  method: 'POST',
  body: JSON.stringify({
    messages: [{ role: 'user', content: "KUET CSE seats?" }],
    output_mode: 'answer_with_context',
    top_k: 5
  })
});

const { answer, contexts, routing } = await res.json();
```

---

## Implementation Checklist

### Week 1: Core Implementation
- [ ] Add `UnifiedChatRequest` model to `unified.py`
- [ ] Add `UnifiedChatResponse` model to `unified.py`
- [ ] Implement `/api/unified/chat` endpoint
- [ ] Add message extraction utilities
- [ ] Add context formatting utilities
- [ ] Add token counting utility
- [ ] Test all output modes
- [ ] Test auto-routing vs forced subgraphs

### Week 2: Documentation & Deprecation
- [ ] Update OpenAPI docs with examples
- [ ] Add deprecation warnings to `/chat/completions`
- [ ] Update `/api/unified/ask` documentation
- [ ] Create migration guide
- [ ] Update frontend examples
- [ ] Add integration tests

### Week 3: Testing
- [ ] Test with enhanced pipeline graphs
- [ ] Test with standard pipeline graphs
- [ ] Test mixed subgraph queries
- [ ] Test error handling
- [ ] Load testing
- [ ] Frontend integration testing

---

## Benefits

1. ✅ **Unified Architecture** - Aligns with `/api/unified/query` pattern
2. ✅ **Single Endpoint** - One endpoint for all chat needs
3. ✅ **Flexible Output** - Context, answer, or both
4. ✅ **Auto-Routing** - LLM selects relevant subgraph(s)
5. ✅ **Single-Mode Support** - Force specific subgraph(s) via `force_subgraphs`
6. ✅ **Enhanced Pipeline Aware** - Works with all graph types
7. ✅ **OpenAI-Compatible Messages** - Standard message format
8. ✅ **Performance Metrics** - Detailed retrieval and LLM metrics
9. ✅ **Language Support** - Per-query language override
10. ✅ **Future-Proof** - Easy to add streaming, multi-modal, etc.

---

## Questions for Review

1. ✅ **Endpoint path**: `/api/unified/chat` - Confirmed
2. ✅ **Auto-routing**: Default behavior - Confirmed
3. ✅ **Single-mode**: Via `force_subgraphs` parameter - Confirmed
4. ✅ **Alignment**: Follows `/api/unified/query` pattern - Confirmed
5. **Output modes**: Keep 3 modes or add `openai_compatible` as 4th?
6. **Streaming**: Defer to Phase 2 or implement now?
7. **Deprecation**: Mark `/chat/completions` as deprecated or keep for OpenAI compatibility?

---

## Ready to Proceed?

This plan creates a unified, flexible chat endpoint that:
- Extends the existing `/api/unified/query` pattern
- Supports all use cases (auto-routing, forced subgraphs, context-only, answer-only)
- Maintains backward compatibility
- Provides clear migration path

Please review and confirm before implementation.
