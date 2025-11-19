"""
Simplified Agent Executor - 2-call-per-iteration design.

This executor implements the simplified agent flow:
1. Plan next action (1 call) → decide "answer" OR "query"
2. If "query": Execute BiG-RAG (0 calls) → Extract and assess (1 call)

Target: 2 calls per iteration × 3 max = 4-6 total calls (vs 19 in complex agent)

Key improvements:
- Uses variable_X to accumulate facts (no lossy extraction)
- Plans 1 query at a time (sequential multi-hop)
- No context pruning (uses all 20 contexts)
- No query preprocessing (enable_query_preprocessing=False)
- Early exit when sufficient information gathered
"""

import time
from datetime import datetime
from typing import Optional

from api.agent_models import (
    AgentRequest,
    AgentResponse,
    ReasoningStep,
    ExecutedAction,
    Observation,
    AgentMetadata,
    PlannedQuery
)
from agent.state import AgentState
from agent.planner import QueryPlanner
from agent.tools import AgentTools


class SimplifiedAgentExecutor:
    """
    Simplified agent executor with 2-call-per-iteration pattern.

    Flow per iteration:
    1. PLAN: Decide if can answer OR what to search next (1 LLM call)
    2. ACT: Execute BiG-RAG query with enable_query_preprocessing=False (0 LLM calls)
    3. EXTRACT & ASSESS: Extract facts to variable_X + assess sufficiency (1 LLM call)
    4. DECIDE: Continue if insufficient, else generate final answer
    """

    def __init__(self, bigrag_instance, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize simplified executor.

        Args:
            bigrag_instance: BiGRAG instance for retrieval
            model: LLM model for reasoning (gpt-4o or gpt-4o-mini)
            api_key: OpenAI API key
        """
        self.bigrag = bigrag_instance
        self.tools = AgentTools(bigrag_instance)
        self.planner = QueryPlanner(model=model, api_key=api_key)
        self.model = model

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute simplified agent.

        Args:
            request: Agent request

        Returns:
            Agent response with answer and reasoning trace
        """
        print(f"\n[AGENT_SIMPLIFIED] Starting execution")
        print(f"[AGENT_SIMPLIFIED] Question: {request.question}")
        print(f"[AGENT_SIMPLIFIED] Max iterations: {request.max_iterations}")
        print(f"[AGENT_SIMPLIFIED] Model: {request.agent_model}")

        # Initialize state with variable_X
        state = AgentState(
            question=request.question,
            max_iterations=request.max_iterations,
            start_time=datetime.now(),
            model_used=request.agent_model
        )

        final_answer = None
        final_confidence = 0.0
        stop_reason = "unknown"

        # Main reasoning loop
        for iteration in range(request.max_iterations):
            step_start_time = time.time()
            print(f"\n[AGENT_SIMPLIFIED] === Iteration {iteration + 1}/{request.max_iterations} ===")

            # =====================================================================
            # PHASE 1: PLAN NEXT ACTION (1 LLM call)
            # Decide: Can we answer now OR do we need more information?
            # =====================================================================
            print(f"[AGENT_SIMPLIFIED] Planning next action...")
            plan = await self.planner.plan_next_action_simplified(
                question=request.question,
                state=state
            )

            # Check if LLM decided to answer
            if plan.get("action") == "answer":
                print(f"[AGENT_SIMPLIFIED] Decision: ANSWER (confidence: {plan.get('confidence', 0.0):.2f})")
                final_answer = plan.get("answer", "Unable to generate answer")
                final_confidence = plan.get("confidence", 0.5)
                stop_reason = "sufficient_information"

                # Create final reasoning step
                reasoning_step = ReasoningStep(
                    step=iteration + 1,
                    thought=plan.get("reasoning", "Decided to answer based on accumulated knowledge"),
                    planned_queries=[],
                    executed_actions=[],
                    observations=[],
                    variables_stored={},
                    confidence=final_confidence,
                    execution_time_ms=(time.time() - step_start_time) * 1000
                )
                state.add_reasoning_step(reasoning_step)
                break

            # LLM decided to query
            query = plan.get("query", request.question)
            query_language = plan.get("query_language", "English")
            query_purpose = plan.get("query_purpose", "Gather information")

            print(f"[AGENT_SIMPLIFIED] Decision: QUERY")
            print(f"  - Query: {query}")
            print(f"  - Language: {query_language}")
            print(f"  - Purpose: {query_purpose}")

            # =====================================================================
            # PHASE 2: EXECUTE BiG-RAG (0 LLM calls - local operation)
            # Key: enable_query_preprocessing=False to avoid extra API call
            # =====================================================================
            print(f"[AGENT_SIMPLIFIED] Executing BiG-RAG query...")

            # Import QueryParam to set enable_query_preprocessing
            from bigrag.base import QueryParam

            # Execute query with preprocessing DISABLED
            contexts, action = await self.tools.search_bigrag_with_params(
                query=query,
                language=query_language,
                query_param=QueryParam(
                    top_k=request.top_k_per_query,
                    enable_reranking=request.enable_reranking,
                    enable_query_preprocessing=False  # KEY: Disable preprocessing!
                ),
                num_kg_in_context=request.num_kg_in_context,
                num_chunks_in_context=request.num_chunks_in_context,
                state=state
            )

            print(f"  - Retrieved: {len(contexts)} contexts")
            print(f"  - No context pruning (using all contexts)")

            # =====================================================================
            # PHASE 3: EXTRACT & ASSESS (1 LLM call)
            # Extract facts into variable_X and assess if we have enough
            # =====================================================================
            print(f"[AGENT_SIMPLIFIED] Extracting facts and assessing sufficiency...")
            assessment = await self.planner.extract_and_assess(
                question=request.question,
                query_executed=query,
                contexts=contexts,
                state=state
            )

            # Create observation with ALL contexts (no pruning)
            observation = Observation(
                query=query,
                contexts=contexts,  # All 20 contexts kept!
                summary=None
            )

            # Extract newly added facts for this step
            # facts_extracted is a list of keys, we need to get their values from variable_X
            facts_extracted_keys = assessment.get("facts_extracted", [])
            variables_stored_this_step = {}
            for key in facts_extracted_keys:
                if key in state.variable_X and key != "metadata":
                    variables_stored_this_step[key] = state.variable_X[key]

            # Create reasoning step
            reasoning_step = ReasoningStep(
                step=iteration + 1,
                thought=plan.get("reasoning", "Planning query"),
                planned_queries=[PlannedQuery(
                    query=query,
                    language=query_language,
                    reason=query_purpose
                )],
                executed_actions=[action],
                observations=[observation],
                variables_stored=variables_stored_this_step,  # Now a dict!
                confidence=assessment.get("confidence", 0.5),
                execution_time_ms=(time.time() - step_start_time) * 1000
            )

            # Update state
            state.add_reasoning_step(reasoning_step)
            state.increment_iteration()

            # Add summary for debugging
            summary = f"Iteration {iteration + 1}: Searched '{query}', extracted {len(assessment.get('facts_extracted', []))} facts"
            state.add_iteration_summary(summary)

            # Store all contexts for final response
            state.all_contexts.extend(contexts)

            print(f"  - Facts extracted: {assessment.get('facts_extracted', [])}")
            print(f"  - Is sufficient: {assessment.get('is_sufficient', False)}")
            print(f"  - Confidence: {assessment.get('confidence', 0.0):.2f}")
            print(f"  - variable_X size: {len(state.variable_X)} keys")

            # =====================================================================
            # PHASE 4: DECIDE - Continue or stop?
            # =====================================================================
            if assessment.get("is_sufficient", False):
                print(f"[AGENT_SIMPLIFIED] Sufficient information gathered!")
                # Next iteration will trigger final answer
                continue

            # Check if max iterations reached
            if iteration + 1 >= request.max_iterations:
                print(f"[AGENT_SIMPLIFIED] Max iterations reached")
                stop_reason = "max_iterations"
                # Will generate answer from variable_X below
                break

        # =====================================================================
        # FINAL: Generate answer if not already generated
        # =====================================================================
        if final_answer is None:
            print(f"\n[AGENT_SIMPLIFIED] Generating final answer from variable_X...")

            # Use plan_next_action one more time to force answer
            final_plan = await self.planner.plan_next_action_simplified(
                question=request.question,
                state=state
            )

            if final_plan.get("action") == "answer":
                final_answer = final_plan.get("answer", "Unable to answer based on gathered information")
                final_confidence = final_plan.get("confidence", 0.5)
            else:
                # Fallback: generate answer from variable_X
                import json
                variable_x_str = json.dumps(state.variable_X, indent=2)
                final_answer = f"Based on the research, here is what was found:\n\n{variable_x_str}"
                final_confidence = 0.4

        # Calculate execution time
        total_execution_time = state.get_execution_time_ms()

        # Build metadata
        metadata = AgentMetadata(
            model_used=request.agent_model,
            total_tokens=state.total_tokens,
            total_cost_usd=self._estimate_cost(state.total_tokens, request.agent_model),
            execution_time_ms=total_execution_time,
            queries_executed=state.queries_executed,
            stopped_reason=stop_reason
        )

        # Build response with variable_X for debugging
        response = AgentResponse(
            answer=final_answer,
            reasoning_trace=state.reasoning_steps,
            total_iterations=state.current_iteration,
            contexts_used=state.all_contexts[:30],  # Limit to 30 for response size
            metadata=metadata,
            limitations=None,
            confidence=final_confidence,
            variable_X=state.variable_X  # Include for debugging
        )

        print(f"\n[AGENT_SIMPLIFIED] Execution complete!")
        print(f"  - Total iterations: {state.current_iteration}")
        print(f"  - Total LLM calls: ~{state.current_iteration * 2} (vs 19 in complex agent)")
        print(f"  - Total tokens: {state.total_tokens}")
        print(f"  - Total time: {total_execution_time:.0f}ms")
        print(f"  - Stop reason: {stop_reason}")
        print(f"  - variable_X keys: {len(state.variable_X)}")

        return response

    @staticmethod
    def _estimate_cost(total_tokens: int, model: str) -> float:
        """
        Estimate API cost based on token usage.

        Args:
            total_tokens: Total tokens used
            model: Model name

        Returns:
            Estimated cost in USD
        """
        pricing = {
            "gpt-4o": 0.000005,  # $5 per 1M tokens (average)
            "gpt-4o-mini": 0.0000002,  # $0.2 per 1M tokens
            "gpt-4": 0.00003,  # $30 per 1M tokens
        }

        rate = pricing.get(model, 0.000005)
        return total_tokens * rate
