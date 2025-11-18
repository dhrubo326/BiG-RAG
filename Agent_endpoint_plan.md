Agent-Like Endpoint Implementation Plan
I'll create a comprehensive plan for building a multi-hop reasoning agent for BiG-RAG. This goes beyond simple retrieval to enable dynamic, iterative query planning and execution.
🎯 Core Concept: ReAct-Style Agent for BiG-RAG
Your idea is excellent! This is essentially a ReAct agent (Reasoning + Acting) pattern applied to knowledge retrieval:
┌─────────────────────────────────────────────────────────────┐
│  Agent Loop: Think → Plan → Act → Observe → Repeat          │
└─────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
    Planning Phase      Execution Phase      Synthesis Phase
📋 Implementation Plan
Phase 1: Core Agent Architecture
1.1 Agent Execution Engine (backend/agent/executor.py)
Purpose: Main loop that orchestrates the agent's reasoning cycle Key Components:
AgentExecutor: Manages the think-act-observe loop
State Management: Tracks query history, intermediate results, contexts
Safety Guards: Max iterations (default 3), timeout per step (30s), infinite loop detection
Features:
class AgentExecutor:
    async def execute(
        self,
        question: str,
        max_iterations: int = 3,
        agent_model: str = "gpt-4o",
        language: str = "auto",
        enable_parallel: bool = True,
        stream: bool = False
    ) -> AgentResponse
    
    # Key methods:
    - _plan_queries()           # LLM decides what to search
    - _execute_queries()        # Run BiG-RAG searches (parallel if possible)
    - _synthesize_answer()      # Final answer generation
    - _should_continue()        # Decide if more iterations needed
1.2 State Management (backend/agent/state.py)
Purpose: Track execution state across iterations Structure:
@dataclass
class AgentState:
    question: str
    current_iteration: int
    max_iterations: int
    
    # Reasoning trace
    thoughts: List[str]                    # LLM reasoning per step
    actions: List[Action]                  # Queries executed
    observations: List[Observation]        # Retrieved contexts
    
    # Intermediate results
    variables: Dict[str, Any]              # Store findings (e.g., {"winner": "Argentina"})
    all_contexts: List[Context]            # All retrieved contexts
    
    # Metadata
    total_tokens: int
    execution_time_ms: float
    model_used: str
1.3 Query Planner (backend/agent/planner.py)
Purpose: Use LLM to decompose complex queries into retrieval steps Two Strategies: A. Sequential Planning (like your Scenario 1):
async def plan_sequential(question: str, state: AgentState) -> List[QueryPlan]:
    """
    Generate a chain of dependent queries.
    Example: "2022 winner" → "captain" → "current club"
    """
B. Parallel Planning (like your Scenario 2):
async def plan_parallel(question: str, state: AgentState) -> List[List[QueryPlan]]:
    """
    Generate multiple independent query groups.
    Example: ["Bangladesh history", "Bangladesh economy"] || ["India history", "India economy"]
    """
LLM Prompt Template (in backend/prompts/agent_prompts.py):
AGENT_PLANNER_PROMPT = """
You are a query planning agent for a knowledge retrieval system.

Given a question, break it down into 1-3 specific search queries.

Question: {question}

Previous iterations:
{iteration_history}

Current knowledge gathered:
{current_variables}

Output a JSON plan:
{{
  "strategy": "sequential" | "parallel",
  "queries": [
    {{"query": "...", "language": "English", "reason": "..."}},
    ...
  ],
  "needs_more_iterations": true | false,
  "confidence": 0.0-1.0
}}

Guidelines:
- Use previous knowledge to refine queries
- Be specific and targeted
- Specify language for each query
- Set needs_more_iterations=false if current info is sufficient
"""
1.4 Tool System (backend/agent/tools.py)
Available Tools:
class AgentTools:
    @staticmethod
    async def search_bigrag(
        query: str,
        language: str = "English",
        top_k: int = 5
    ) -> List[Context]:
        """Primary tool: Query BiG-RAG retrieval system"""
    
    @staticmethod
    def store_variable(key: str, value: Any, state: AgentState):
        """Store intermediate result (e.g., "winner" → "Argentina")"""
    
    @staticmethod
    def get_variable(key: str, state: AgentState) -> Optional[Any]:
        """Retrieve stored variable"""
    
    @staticmethod
    async def summarize_contexts(
        contexts: List[Context],
        focus: str,
        model: str = "gpt-4o-mini"
    ) -> str:
        """Compress contexts to prevent context window overflow"""
Phase 2: API Design
2.1 Endpoint Definition (backend/api/agent.py)
@router.post("/agent/query", response_model=AgentResponse)
async def agent_query(request: AgentRequest):
    """
    Multi-hop reasoning agent endpoint.
    
    Uses LLM to iteratively plan and execute BiG-RAG queries
    until sufficient evidence is gathered.
    """
2.2 Request/Response Models (backend/api/agent_models.py)
class AgentRequest(BaseModel):
    question: str
    language: str = "auto"                  # Auto-detect or specify
    max_iterations: int = Field(3, ge=1, le=5)  # Hard limit: 1-5
    agent_model: str = "gpt-4o"            # Thinking model
    enable_parallel: bool = True            # Parallel query execution
    stream: bool = False                    # Stream reasoning trace
    top_k_per_query: int = 5               # Contexts per retrieval
    
    # Advanced options
    enable_variable_storage: bool = True
    enable_context_compression: bool = False
    confidence_threshold: float = 0.8      # Stop if confidence >= threshold

class AgentResponse(BaseModel):
    answer: str                            # Final synthesized answer
    reasoning_trace: List[ReasoningStep]   # Full execution trace
    total_iterations: int
    contexts_used: List[Context]           # All contexts retrieved
    metadata: AgentMetadata

class ReasoningStep(BaseModel):
    step: int
    thought: str                           # LLM reasoning
    planned_queries: List[PlannedQuery]
    executed_actions: List[ExecutedAction]
    observations: List[Observation]
    variables_stored: Dict[str, Any]       # Intermediate results
    confidence: float                      # LLM confidence (0-1)
    execution_time_ms: float

class PlannedQuery(BaseModel):
    query: str
    language: str
    reason: str                            # Why this query is needed

class ExecutedAction(BaseModel):
    action_type: str = "search_bigrag"
    query: str
    language: str
    num_results: int
    execution_time_ms: float

class Observation(BaseModel):
    query: str
    contexts: List[Context]
    summary: Optional[str]                 # Compressed version

class AgentMetadata(BaseModel):
    model_used: str
    total_tokens: int
    total_cost_usd: float                  # Estimate
    execution_time_ms: float
    queries_executed: int
    stopped_reason: str                    # "max_iterations" | "high_confidence" | "complete"
Phase 3: Enhanced Features (My Improvements)
3.1 Query Deduplication
Before executing a query, check if it's semantically similar to a previous query
Use embedding similarity (threshold: 0.95)
Reuse previous results if duplicate detected
3.2 Context Relevance Filtering
After each retrieval, use LLM to score context relevance
Discard contexts with relevance < 0.3
Prevents context pollution in subsequent steps
async def filter_relevant_contexts(
    contexts: List[Context],
    question: str,
    current_focus: str,
    model: str = "gpt-4o-mini"
) -> List[Context]:
    """
    LLM scores each context for relevance.
    Only keep contexts with score >= 0.3
    """
3.3 Adaptive Iteration Count
Instead of fixed max_iterations, use LLM confidence scoring
Stop early if confidence >= threshold (default 0.8)
Continue up to max_iterations if confidence is low
3.4 Streaming Support
@router.post("/agent/query/stream")
async def agent_query_stream(request: AgentRequest):
    """
    Stream reasoning trace in real-time using SSE (Server-Sent Events).
    
    User sees:
    - "Step 1: Searching for 2022 World Cup winner..."
    - "Found: Argentina won the 2022 World Cup"
    - "Step 2: Searching for Argentina captain..."
    - etc.
    """
    
    async def event_generator():
        async for event in agent_executor.execute_stream(request):
            yield f"data: {event.json()}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
3.5 Query Optimization
Analyze planned queries to identify parallelization opportunities
Example:
# Sequential plan: ["Bangladesh GDP", "India GDP", "Bangladesh population", "India population"]
# Optimized: [["Bangladesh GDP", "Bangladesh population"], ["India GDP", "India population"]]
# Execute each group in parallel
3.6 Self-Correction Mechanism
# In synthesis phase, ask LLM:
"Based on the evidence gathered, do any previous conclusions seem incorrect?
If yes, plan corrective queries."

# Example:
# Step 1: Concluded "Winner: France" (wrong)
# Step 3: LLM detects contradiction in new context
# → Issue corrective query: "Who actually won 2022 World Cup?"
Phase 4: Prompt Engineering
4.1 System Prompts (backend/prompts/agent_prompts.py)
Main Agent Prompt:
AGENT_SYSTEM_PROMPT = """
You are an intelligent research assistant with access to a knowledge retrieval system (BiG-RAG).

Your goal: Answer the user's question by iteratively planning and executing targeted searches.

Capabilities:
1. search_bigrag(query: str, language: str) → List[Context]
   - Retrieves relevant knowledge from the graph database
   - Supports multilingual queries

2. store_variable(key: str, value: Any)
   - Store important intermediate findings

3. get_variable(key: str) → Any
   - Retrieve previously stored information

Process for each iteration:
1. THINK: Analyze the question and current knowledge
2. PLAN: Decide what additional information is needed
3. ACT: Execute 1-3 targeted BiG-RAG searches
4. OBSERVE: Review retrieved contexts
5. DECIDE: Determine if you have enough information to answer

Stop conditions:
- You have high confidence in the answer (>= 0.8)
- You've reached max iterations ({max_iterations})
- Additional searches yield no new information

Output format (JSON):
{{
  "thought": "Current reasoning...",
  "queries": [
    {{"query": "...", "language": "...", "reason": "..."}},
    ...
  ],
  "variables_to_store": {{"key": "value", ...}},
  "confidence": 0.0-1.0,
  "needs_more_iterations": true | false,
  "partial_answer": "..." (if available)
}}

Guidelines:
- Be specific in queries (avoid vague terms)
- Use intermediate results to refine subsequent queries
- Cite contexts explicitly in your answer
- If context is insufficient, acknowledge it
"""
Synthesis Prompt:
AGENT_SYNTHESIS_PROMPT = """
You have completed {num_iterations} iterations of research.

Original question: {question}

Evidence gathered:
{all_contexts}

Intermediate findings:
{variables}

Task: Synthesize a comprehensive, accurate answer.

Requirements:
1. Answer the question directly
2. Cite specific contexts used (e.g., "According to Context 3...")
3. If information is incomplete, state what's missing
4. Highlight any contradictions found
5. Confidence score (0-1)

Output format (JSON):
{{
  "answer": "...",
  "confidence": 0.0-1.0,
  "contexts_cited": [3, 7, 12],
  "limitations": "..." (if any),
  "reasoning": "How you arrived at this answer"
}}
"""
Phase 5: Implementation Steps
backend/
├── agent/
│   ├── __init__.py
│   ├── executor.py          # ✅ NEW: AgentExecutor class
│   ├── planner.py           # ✅ NEW: Query planning logic
│   ├── state.py             # ✅ NEW: AgentState management
│   └── tools.py             # ✅ NEW: search_bigrag, store_variable, etc.
│
├── api/
│   ├── agent.py             # ✅ NEW: /agent/query endpoint
│   └── agent_models.py      # ✅ NEW: Pydantic models
│
├── prompts/
│   └── agent_prompts.py     # ✅ NEW: Agent system prompts
│
└── server.py                # ✅ UPDATE: Register agent routes
Step-by-step implementation order:
Step 1: Create data models (agent_models.py) ✅
Step 2: Implement state management (state.py) ✅
Step 3: Build tool system (tools.py) ✅
Step 4: Write agent prompts (agent_prompts.py) ✅
Step 5: Implement query planner (planner.py) ✅
Step 6: Build executor engine (executor.py) ✅
Step 7: Create API endpoint (agent.py) ✅
Step 8: Register routes in server.py ✅
Step 9: Add tests (test_scripts/test_agent.py) ✅
Step 10: Documentation (docs/technical/AGENT_ENDPOINT_GUIDE.md) ✅
Phase 6: Testing Strategy
6.1 Unit Tests (test_scripts/test_agent.py)
# Test cases:
1. Single-hop query (should use 1 iteration)
2. Sequential multi-hop (your Scenario 1)
3. Parallel multi-hop (your Scenario 2)
4. Max iteration limit enforcement
5. Confidence-based early stopping
6. Query deduplication
7. Variable storage/retrieval
8. Context compression
9. Streaming output
10. Error handling (no results, timeout, etc.)
6.2 Integration Tests
# End-to-end tests:
1. Test with SingleTopic dataset
2. Test with multilingual queries (English + Bangla)
3. Test parallel execution performance
4. Test streaming endpoint
5. Test with actual LLM (GPT-4o)
Phase 7: Configuration
7.1 Environment Variables (.env)
# Agent Configuration
AGENT_DEFAULT_MODEL=gpt-4o                # Thinking model
AGENT_MAX_ITERATIONS=3                    # Hard limit
AGENT_ENABLE_STREAMING=true               # Enable streaming
AGENT_CONFIDENCE_THRESHOLD=0.8            # Early stopping threshold
AGENT_ENABLE_CONTEXT_COMPRESSION=false    # Auto-compress contexts
AGENT_QUERY_DEDUP_THRESHOLD=0.95          # Similarity threshold for dedup
AGENT_TIMEOUT_PER_STEP_SEC=30             # Timeout per iteration
7.2 Feature Flags
# In agent_models.py
class AgentConfig(BaseModel):
    enable_parallel_execution: bool = True
    enable_variable_storage: bool = True
    enable_query_deduplication: bool = True
    enable_context_filtering: bool = True
    enable_self_correction: bool = False    # Experimental
    enable_adaptive_iterations: bool = True
Phase 8: Example Execution Flow
Scenario 1: Sequential Multi-Hop
// Request
POST /agent/query
{
  "question": "I want to know the 2022 World Cup winner's captain's current club.",
  "max_iterations": 3,
  "agent_model": "gpt-4o"
}

// Response
{
  "answer": "The 2022 FIFA World Cup was won by Argentina, captained by Lionel Messi, who currently plays for Inter Miami CF in Major League Soccer (MLS).",
  "reasoning_trace": [
    {
      "step": 1,
      "thought": "I need to first identify the winner of the 2022 FIFA World Cup.",
      "planned_queries": [
        {"query": "2022 FIFA World Cup winner", "language": "English", "reason": "Identify the winning team"}
      ],
      "executed_actions": [
        {"action_type": "search_bigrag", "query": "2022 FIFA World Cup winner", "num_results": 5}
      ],
      "observations": [
        {"query": "2022 FIFA World Cup winner", "contexts": [...], "summary": "Argentina won the 2022 World Cup"}
      ],
      "variables_stored": {"world_cup_winner": "Argentina"},
      "confidence": 0.5
    },
    {
      "step": 2,
      "thought": "Now I know Argentina won. I need to find who was their captain.",
      "planned_queries": [
        {"query": "Argentina 2022 World Cup captain", "language": "English", "reason": "Identify the team captain"}
      ],
      "executed_actions": [
        {"action_type": "search_bigrag", "query": "Argentina 2022 World Cup captain", "num_results": 5}
      ],
      "observations": [
        {"query": "Argentina 2022 World Cup captain", "contexts": [...], "summary": "Lionel Messi was the captain"}
      ],
      "variables_stored": {"captain": "Lionel Messi"},
      "confidence": 0.7
    },
    {
      "step": 3,
      "thought": "I know the captain is Lionel Messi. Now I need to find his current club.",
      "planned_queries": [
        {"query": "Lionel Messi current club 2024", "language": "English", "reason": "Find his current team"}
      ],
      "executed_actions": [
        {"action_type": "search_bigrag", "query": "Lionel Messi current club 2024", "num_results": 5}
      ],
      "observations": [
        {"query": "Lionel Messi current club 2024", "contexts": [...], "summary": "Messi plays for Inter Miami CF"}
      ],
      "variables_stored": {"current_club": "Inter Miami CF"},
      "confidence": 0.95
    }
  ],
  "total_iterations": 3,
  "contexts_used": [...],  // All 15 contexts from 3 queries
  "metadata": {
    "model_used": "gpt-4o",
    "total_tokens": 4567,
    "total_cost_usd": 0.23,
    "execution_time_ms": 3200,
    "queries_executed": 3,
    "stopped_reason": "high_confidence"
  }
}
Scenario 2: Parallel Multi-Hop
// Request
POST /agent/query
{
  "question": "Compare Bangladesh vs India in terms of population and GDP.",
  "max_iterations": 2,
  "enable_parallel": true
}

// Response
{
  "answer": "Bangladesh has a population of approximately 170 million with a GDP of $460 billion (2023), while India has a population of approximately 1.4 billion with a GDP of $3.7 trillion (2023). India has roughly 8x the population and 8x the GDP of Bangladesh.",
  "reasoning_trace": [
    {
      "step": 1,
      "thought": "I need information about both countries. I can query them in parallel.",
      "planned_queries": [
        {"query": "Bangladesh population 2023", "language": "English"},
        {"query": "Bangladesh GDP 2023", "language": "English"},
        {"query": "India population 2023", "language": "English"},
        {"query": "India GDP 2023", "language": "English"}
      ],
      "executed_actions": [
        // All 4 queries executed in parallel using asyncio.gather()
        {"action_type": "search_bigrag", "query": "Bangladesh population 2023", "num_results": 5},
        {"action_type": "search_bigrag", "query": "Bangladesh GDP 2023", "num_results": 5},
        {"action_type": "search_bigrag", "query": "India population 2023", "num_results": 5},
        {"action_type": "search_bigrag", "query": "India GDP 2023", "num_results": 5}
      ],
      "observations": [...],
      "variables_stored": {
        "bangladesh_population": "170 million",
        "bangladesh_gdp": "$460 billion",
        "india_population": "1.4 billion",
        "india_gdp": "$3.7 trillion"
      },
      "confidence": 0.9
    }
  ],
  "total_iterations": 1,
  "metadata": {
    "queries_executed": 4,
    "stopped_reason": "high_confidence"
  }
}
🚀 Advantages of This Design
Flexible: Handles both sequential and parallel query patterns
Safe: Hard limits prevent infinite loops
Transparent: Full reasoning trace for debugging
Efficient: Parallel execution where possible
Accurate: Uses powerful thinking models (GPT-4o, Claude)
Extensible: Easy to add new tools beyond search_bigrag
Cost-Aware: Tracks token usage and estimated cost
User-Friendly: Streaming option for real-time feedback
📊 Performance Considerations
Aspect	Expected Performance
Single-hop query	~500-800ms (1 BiG-RAG call + synthesis)
3-hop sequential	~2-3 seconds (3 sequential calls + planning overhead)
4 parallel queries	~800-1200ms (parallel execution)
Streaming latency	First token in ~200ms, continuous updates
Token usage	~3000-8000 tokens per request (depends on iterations)
⚠️ Potential Challenges & Solutions
Challenge	Solution
LLM hallucinates intermediate results	Force citation of exact text from contexts
Context window overflow	Implement context compression after each step
Expensive API costs	Add cost estimation and warnings (though not a constraint)
Query loops	Track query history, detect semantic duplicates
Slow execution	Implement aggressive parallelization and caching
Complex question ambiguity	Ask clarifying questions or return multiple interpretations