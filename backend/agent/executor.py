"""
Agent executor - orchestrates the think-act-observe loop.

This is the main engine that runs the multi-hop reasoning agent.

Enhanced with:
- LLM-based variable extraction
- Context pruning (keep 2-3 best per iteration)
- Iteration summaries
- Source-aware extraction (chunks vs KG)
- Multilingual search support
"""

import time
from datetime import datetime
from typing import Optional

from api.agent_models import (
    AgentRequest,
    AgentResponse,
    ReasoningStep,
    Observation,
    AgentMetadata
)
from agent.state import AgentState
from agent.planner import QueryPlanner
from agent.tools import AgentTools
from agent.extraction import ContextExtractor
from agent.summarization import IterationSummarizer


class AgentExecutor:
    """
    Executes the agent's reasoning loop.

    Enhanced process:
    1. PLAN: Use LLM to decide what queries to execute
    2. ACT: Execute queries using BiG-RAG (with multilingual support)
    3. EXTRACT: Use LLM to extract specific facts from contexts
    4. PRUNE: Keep only 2-3 most relevant contexts per iteration
    5. SUMMARIZE: Create concise summary of findings
    6. DECIDE: Assess if more iterations needed
    7. SYNTHESIZE: Generate final answer from summaries
    """

    def __init__(self, bigrag_instance, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize executor.

        Args:
            bigrag_instance: BiGRAG instance for retrieval
            model: LLM model for reasoning
            api_key: OpenAI API key
        """
        self.bigrag = bigrag_instance
        self.tools = AgentTools(bigrag_instance)
        self.planner = QueryPlanner(model=model, api_key=api_key)
        self.extractor = ContextExtractor(self.planner.client, model="gpt-4o-mini")
        self.summarizer = IterationSummarizer(self.planner.client, model="gpt-4o-mini")
        self.model = model

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute agent for a request.

        Args:
            request: Agent request

        Returns:
            Agent response with answer and reasoning trace
        """
        print(f"\n[AGENT] Starting execution for question: {request.question}")
        print(f"[AGENT] Max iterations: {request.max_iterations}")
        print(f"[AGENT] Model: {request.agent_model}")

        # Initialize state
        state = AgentState(
            question=request.question,
            max_iterations=request.max_iterations,
            start_time=datetime.now(),
            model_used=request.agent_model
        )

        # Main reasoning loop
        while state.should_continue(request.confidence_threshold):
            step_start_time = time.time()

            print(f"\n[AGENT] === Iteration {state.current_iteration + 1}/{state.max_iterations} ===")

            # PHASE 1: PLAN
            print(f"[AGENT] Planning queries...")
            plan = await self.planner.plan(request.question, state)

            print(f"[AGENT] Thought: {plan['thought']}")
            print(f"[AGENT] Strategy: {plan['strategy']}")
            print(f"[AGENT] Queries planned: {len(plan['queries'])}")
            for q in plan["queries"]:
                print(f"  - {q.query} ({q.language}): {q.reason}")

            # Check if LLM thinks we're done
            if not plan["needs_more_iterations"] and plan["confidence"] >= request.confidence_threshold:
                print(f"[AGENT] LLM indicates completion (confidence: {plan['confidence']:.2f})")
                break

            # PHASE 2: ACT (Execute queries with multilingual support)
            print(f"[AGENT] Executing queries...")
            observations = []
            all_actions = []

            for planned_query in plan["queries"]:
                # Detect if multilingual search would help
                languages_to_search = [planned_query.language]

                # If query language is Bangla/Banglish, also search in English
                if planned_query.language in ["Bangla", "Bengali"]:
                    languages_to_search.append("English")
                    print(f"[AGENT] Multilingual search: {languages_to_search}")

                # Execute search (multilingual if applicable)
                if len(languages_to_search) > 1:
                    contexts, actions = await self.tools.search_bigrag_multilingual(
                        query=planned_query.query,
                        languages=languages_to_search,
                        top_k=request.top_k_per_query,
                        state=state
                    )
                    all_actions.extend(actions)
                else:
                    contexts, action = await self.tools.search_bigrag(
                        query=planned_query.query,
                        language=planned_query.language,
                        top_k=request.top_k_per_query,
                        state=state
                    )
                    all_actions.append(action)

                print(f"  - {planned_query.query}: {len(contexts)} results")

                # PHASE 3: PRUNE (Keep only 2-3 most relevant contexts)
                print(f"[AGENT] Pruning contexts...")
                pruned_contexts, scores = await self.extractor.prune_contexts(
                    contexts=contexts,
                    query=planned_query.query,
                    original_question=request.question,
                    keep_top_n=3  # Keep only top 3 per query
                )

                # Store pruned contexts
                state.add_pruned_contexts(pruned_contexts)

                # Create observation with pruned contexts
                obs = Observation(
                    query=planned_query.query,
                    contexts=pruned_contexts,  # Only the best 2-3!
                    summary=None
                )
                observations.append(obs)

            # PHASE 4: EXTRACT (LLM-based variable extraction)
            print(f"[AGENT] Extracting variables...")
            variables_stored = {}

            if request.enable_variable_storage and plan["variables_to_store"]:
                for var_name, var_desc in plan["variables_to_store"].items():
                    # Collect all contexts from this iteration
                    all_iteration_contexts = []
                    for obs in observations:
                        all_iteration_contexts.extend(obs.contexts)

                    if all_iteration_contexts:
                        # Use LLM to extract specific fact
                        extraction_result = await self.extractor.extract_variable(
                            variable_name=var_name,
                            variable_description=var_desc,
                            contexts=all_iteration_contexts,
                            question=request.question
                        )

                        value = extraction_result.get("value", "NOT_FOUND")
                        confidence = extraction_result.get("confidence", 0.0)

                        variables_stored[var_name] = value
                        state.store_variable(var_name, value)

                        print(f"  - Extracted {var_name}: {value} (confidence: {confidence:.2f})")

            # PHASE 5: EXTRACT METADATA FACTS
            print(f"[AGENT] Extracting metadata facts...")
            all_iteration_contexts = []
            for obs in observations:
                all_iteration_contexts.extend(obs.contexts)

            metadata_facts = await self.extractor.extract_metadata_facts(
                contexts=all_iteration_contexts,
                question=request.question
            )

            if metadata_facts:
                state.add_metadata_facts(metadata_facts)
                print(f"  - Entities from metadata: {metadata_facts.get('entities_from_metadata', [])}")
                print(f"  - Facts from titles: {metadata_facts.get('facts_from_titles', {})}")

            # PHASE 6: SUMMARIZE ITERATION
            print(f"[AGENT] Summarizing iteration...")
            iteration_summary = await self.summarizer.summarize_iteration(
                step_number=state.current_iteration + 1,
                question=request.question,
                observations=observations,
                variables_stored=variables_stored,
                metadata_facts=metadata_facts
            )

            state.add_iteration_summary(iteration_summary)
            print(f"  - Summary: {iteration_summary}")

            # Create reasoning step
            step_execution_time = (time.time() - step_start_time) * 1000
            reasoning_step = ReasoningStep(
                step=state.current_iteration + 1,
                thought=plan["thought"],
                planned_queries=plan["queries"],
                executed_actions=all_actions if plan["strategy"] == "parallel" else all_actions,
                observations=observations,
                variables_stored=variables_stored,
                confidence=plan["confidence"],
                execution_time_ms=step_execution_time
            )

            # Add step to state
            state.add_reasoning_step(reasoning_step)
            state.increment_iteration()

            print(f"[AGENT] Step completed in {step_execution_time:.0f}ms")
            print(f"[AGENT] Step confidence: {plan['confidence']:.2f}")
            print(f"[AGENT] Pruned contexts kept: {len(state.pruned_contexts)}")

            # PHASE 7: ASSESS SUFFICIENCY
            print(f"[AGENT] Assessing information sufficiency...")
            sufficiency = await self.summarizer.assess_information_sufficiency(
                question=request.question,
                iteration_summaries=state.iteration_summaries,
                variables_collected=state.variables,
                max_iterations=request.max_iterations,
                current_iteration=state.current_iteration
            )

            print(f"  - Sufficient: {sufficiency.get('is_sufficient', False)}")
            print(f"  - Should continue: {sufficiency.get('should_continue', True)}")
            print(f"  - Reasoning: {sufficiency.get('reasoning', 'N/A')}")

            # Override state's should_continue if sufficiency says stop
            if not sufficiency.get("should_continue", True):
                print(f"[AGENT] Sufficiency check indicates we should stop")
                break

        # PHASE 8: SYNTHESIZE
        print(f"\n[AGENT] Synthesizing final answer...")
        synthesis = await self.planner.synthesize_answer(request.question, state)

        print(f"[AGENT] Answer confidence: {synthesis['confidence']:.2f}")
        print(f"[AGENT] Answer: {synthesis['answer'][:200]}...")

        # Determine stop reason
        if state.current_iteration >= state.max_iterations:
            stop_reason = "max_iterations"
        elif state.get_latest_confidence() >= request.confidence_threshold:
            stop_reason = "high_confidence"
        else:
            stop_reason = "complete"

        # Calculate total execution time
        total_execution_time = state.get_execution_time_ms()

        # Build metadata
        metadata = AgentMetadata(
            model_used=request.agent_model,
            total_tokens=state.total_tokens,
            total_cost_usd=self._estimate_cost(state.total_tokens, request.agent_model),
            execution_time_ms=total_execution_time,
            queries_executed=len(state.executed_queries),
            stopped_reason=stop_reason
        )

        # Use PRUNED contexts in response (only the best 2-3 per iteration)
        # This prevents huge JSON and keeps only relevant contexts
        contexts_to_return = state.pruned_contexts

        print(f"[AGENT] Returning {len(contexts_to_return)} pruned contexts (from {len(state.all_contexts)} total)")

        # Build response
        response = AgentResponse(
            answer=synthesis["answer"],
            reasoning_trace=state.reasoning_steps,
            total_iterations=state.current_iteration,
            contexts_used=contexts_to_return,  # Pruned contexts only!
            metadata=metadata,
            limitations=synthesis.get("limitations"),
            confidence=synthesis["confidence"]
        )

        print(f"\n[AGENT] Execution complete!")
        print(f"[AGENT] Total iterations: {state.current_iteration}")
        print(f"[AGENT] Total queries: {len(state.executed_queries)}")
        print(f"[AGENT] Total contexts retrieved: {len(state.all_contexts)}")
        print(f"[AGENT] Pruned contexts kept: {len(state.pruned_contexts)}")
        print(f"[AGENT] Iteration summaries: {len(state.iteration_summaries)}")
        print(f"[AGENT] Total tokens: {state.total_tokens}")
        print(f"[AGENT] Total time: {total_execution_time:.0f}ms")
        print(f"[AGENT] Stop reason: {stop_reason}")

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
        # Rough pricing estimates (as of 2024)
        pricing = {
            "gpt-4o": 0.000005,  # $5 per 1M tokens (average of input/output)
            "gpt-4o-mini": 0.0000002,  # $0.2 per 1M tokens
            "gpt-4": 0.00003,  # $30 per 1M tokens
        }

        rate = pricing.get(model, 0.000005)  # Default to gpt-4o pricing
        return total_tokens * rate
