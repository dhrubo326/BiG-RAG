"""
Query planning for the agent.

Uses LLM to decide what queries to execute and how to execute them.
"""

import json
import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from api.agent_models import PlannedQuery
from agent.state import AgentState
from prompts.agent_prompts import AGENT_PLANNER_PROMPT


class QueryPlanner:
    """
    Uses LLM to plan queries for the agent.

    The planner decides:
    - What queries to execute
    - What language to use
    - Whether to run queries in parallel or sequential
    - What variables to store
    - Confidence in current knowledge
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize query planner.

        Args:
            model: OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def plan(
        self,
        question: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Plan queries for the next iteration.

        Args:
            question: User's question
            state: Current agent state

        Returns:
            Dict with:
                - thought: Reasoning
                - strategy: "sequential" or "parallel"
                - queries: List of PlannedQuery
                - variables_to_store: Dict of variables
                - confidence: float
                - needs_more_iterations: bool
        """
        # Build prompt
        prompt = AGENT_PLANNER_PROMPT.format(
            question=question,
            current_iteration=state.current_iteration + 1,
            max_iterations=state.max_iterations,
            iteration_history=state.get_iteration_history(),
            current_variables=state.get_current_knowledge()
        )

        try:
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query planning assistant. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            plan = json.loads(content)

            # Track token usage
            state.total_tokens += response.usage.total_tokens

            # Convert queries to PlannedQuery objects
            planned_queries = []
            for q in plan.get("queries", []):
                planned_queries.append(PlannedQuery(
                    query=q.get("query", ""),
                    language=q.get("language", "English"),
                    reason=q.get("reason", "")
                ))

            return {
                "thought": plan.get("thought", "Planning queries..."),
                "strategy": plan.get("strategy", "parallel"),
                "queries": planned_queries,
                "variables_to_store": plan.get("variables_to_store", {}),
                "confidence": plan.get("confidence", 0.5),
                "needs_more_iterations": plan.get("needs_more_iterations", True)
            }

        except json.JSONDecodeError as e:
            print(f"[PLANNER] Error parsing LLM response: {e}")
            print(f"[PLANNER] Response: {content}")

            # Fallback plan
            return {
                "thought": "Error parsing plan, using fallback",
                "strategy": "sequential",
                "queries": [PlannedQuery(
                    query=question,
                    language="English",
                    reason="Fallback query"
                )],
                "variables_to_store": {},
                "confidence": 0.3,
                "needs_more_iterations": True
            }

        except Exception as e:
            print(f"[PLANNER] Error in planning: {e}")
            import traceback
            traceback.print_exc()

            # Fallback plan
            return {
                "thought": f"Error in planning: {str(e)}",
                "strategy": "sequential",
                "queries": [PlannedQuery(
                    query=question,
                    language="English",
                    reason="Fallback query due to error"
                )],
                "variables_to_store": {},
                "confidence": 0.2,
                "needs_more_iterations": True
            }

    async def synthesize_answer(
        self,
        question: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Synthesize final answer from gathered evidence.

        Args:
            question: User's question
            state: Agent state with all contexts

        Returns:
            Dict with:
                - answer: Final answer
                - confidence: float
                - contexts_cited: List of indices
                - limitations: Optional[str]
                - reasoning: str
        """
        from prompts.agent_prompts import AGENT_SYNTHESIS_PROMPT
        from agent.summarization import IterationSummarizer

        # Use iteration summaries instead of raw contexts
        # This is much more efficient and focused
        summarizer = IterationSummarizer(self.client, model="gpt-4o-mini")

        formatted = summarizer.format_for_synthesis(
            iteration_summaries=state.iteration_summaries,
            variables_collected=state.variables,
            top_contexts=state.pruned_contexts,  # Use pruned contexts, not all
            max_contexts=10
        )

        # Build prompt
        prompt = AGENT_SYNTHESIS_PROMPT.format(
            num_iterations=state.current_iteration,
            question=question,
            iteration_summaries=formatted["summaries"],
            variables=formatted["variables"],
            top_contexts=formatted["top_contexts"]
        )

        try:
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at synthesizing information. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for synthesis
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            synthesis = json.loads(content)

            # Track token usage
            state.total_tokens += response.usage.total_tokens

            return {
                "answer": synthesis.get("answer", "Unable to generate answer."),
                "confidence": synthesis.get("confidence", 0.5),
                "contexts_cited": synthesis.get("contexts_cited", []),
                "limitations": synthesis.get("limitations"),
                "reasoning": synthesis.get("reasoning", "")
            }

        except json.JSONDecodeError as e:
            print(f"[SYNTHESIS] Error parsing LLM response: {e}")
            print(f"[SYNTHESIS] Response: {content}")

            # Fallback synthesis
            return {
                "answer": f"Based on the research conducted, I was unable to synthesize a proper answer due to a technical error. Please try again.",
                "confidence": 0.1,
                "contexts_cited": [],
                "limitations": "Error in synthesis process",
                "reasoning": "Fallback due to JSON parsing error"
            }

        except Exception as e:
            print(f"[SYNTHESIS] Error in synthesis: {e}")
            import traceback
            traceback.print_exc()

            return {
                "answer": f"An error occurred while synthesizing the answer: {str(e)}",
                "confidence": 0.0,
                "contexts_cited": [],
                "limitations": "Technical error in synthesis",
                "reasoning": "Error in synthesis process"
            }

    # =========================================================================
    # SIMPLIFIED AGENT METHODS (2-call-per-iteration design)
    # =========================================================================

    async def plan_next_action_simplified(
        self,
        question: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        SIMPLIFIED: Decide to ANSWER or plan ONE query.

        This replaces the complex multi-query planning with a simple decision:
        - If we have enough info in variable_X → ANSWER
        - If we need more info → plan ONE specific QUERY

        Args:
            question: User's question
            state: Current agent state (with variable_X)

        Returns:
            Dict with:
                - action: "answer" or "query"
                - reasoning: Explanation
                - confidence: float

                If action == "query":
                - query: str
                - query_language: str
                - query_purpose: str

                If action == "answer":
                - answer: str
                - answer_sources: List[str]
        """
        from prompts.agent_prompts import SIMPLIFIED_PLAN_NEXT_ACTION_PROMPT
        import json

        # Format variable_X for prompt
        variable_x_str = json.dumps(state.variable_X, indent=2) if state.variable_X else "{}"

        # Format action history
        action_history = []
        for step in state.reasoning_steps:
            for action in step.executed_actions:
                action_history.append(f"- Searched: {action.query} ({action.language})")
        action_history_str = "\n".join(action_history) if action_history else "No previous actions"

        # Build prompt
        prompt = SIMPLIFIED_PLAN_NEXT_ACTION_PROMPT.format(
            question=question,
            current_iteration=state.current_iteration + 1,
            max_iterations=state.max_iterations,
            variable_x=variable_x_str,
            action_history=action_history_str
        )

        try:
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research planning assistant. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temp for more focused decisions
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            plan = json.loads(content)

            # Track token usage
            state.total_tokens += response.usage.total_tokens

            print(f"[PLAN_SIMPLIFIED] Action: {plan.get('action')}")
            print(f"[PLAN_SIMPLIFIED] Reasoning: {plan.get('reasoning')}")
            print(f"[PLAN_SIMPLIFIED] Confidence: {plan.get('confidence', 0.0):.2f}")

            return plan

        except json.JSONDecodeError as e:
            print(f"[PLAN_SIMPLIFIED] Error parsing LLM response: {e}")
            print(f"[PLAN_SIMPLIFIED] Response: {content}")

            # Fallback: try one more query
            return {
                "action": "query",
                "reasoning": "Error in planning, trying fallback query",
                "confidence": 0.3,
                "query": question,
                "query_language": "English",
                "query_purpose": "Fallback query due to parsing error"
            }

        except Exception as e:
            print(f"[PLAN_SIMPLIFIED] Error in planning: {e}")
            import traceback
            traceback.print_exc()

            # Fallback
            return {
                "action": "query",
                "reasoning": f"Error in planning: {str(e)}",
                "confidence": 0.2,
                "query": question,
                "query_language": "English",
                "query_purpose": "Fallback query due to error"
            }

    async def extract_and_assess(
        self,
        question: str,
        query_executed: str,
        contexts: list,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        SIMPLIFIED: Extract facts from contexts + assess sufficiency.

        This combines extraction and sufficiency assessment into ONE LLM call.

        Args:
            question: User's question
            query_executed: The query that was just executed
            contexts: Retrieved contexts from BiG-RAG
            state: Current agent state (with variable_X)

        Returns:
            Dict with:
                - updated_variable_X: Updated variable_X dict
                - facts_extracted: List of fact keys added/updated
                - is_sufficient: bool
                - missing_info: List[str]
                - next_query_suggestion: Optional[str]
                - confidence: float
                - reasoning: str
        """
        from prompts.agent_prompts import SIMPLIFIED_EXTRACT_AND_ASSESS_PROMPT
        import json

        # Format variable_X
        variable_x_str = json.dumps(state.variable_X, indent=2) if state.variable_X else "{}"

        # Format contexts (limit to 20 for token efficiency)
        context_lines = []
        for idx, ctx in enumerate(contexts[:20]):
            source_type = ctx.metadata.get("type", "unknown") if hasattr(ctx, 'metadata') else "unknown"
            text = ctx.text if hasattr(ctx, 'text') else str(ctx)

            # Add metadata for chunks
            metadata_str = ""
            if hasattr(ctx, 'metadata') and ctx.metadata:
                title = ctx.metadata.get("title", "")
                if title:
                    metadata_str = f" [Title: {title}]"

            context_lines.append(f"[{idx}] ({source_type}){metadata_str}\n{text[:500]}")

        contexts_str = "\n\n".join(context_lines)

        # Build prompt
        prompt = SIMPLIFIED_EXTRACT_AND_ASSESS_PROMPT.format(
            question=question,
            query=query_executed,
            variable_x=variable_x_str,
            contexts=contexts_str
        )

        try:
            # Call LLM
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for cost efficiency
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fact extraction and assessment assistant. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temp for factual extraction
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)

            # Track token usage
            state.total_tokens += response.usage.total_tokens

            # Update state's variable_X
            state.variable_X = result.get("updated_variable_X", state.variable_X)

            print(f"[EXTRACT_ASSESS] Facts extracted: {result.get('facts_extracted', [])}")
            print(f"[EXTRACT_ASSESS] Is sufficient: {result.get('is_sufficient', False)}")
            print(f"[EXTRACT_ASSESS] Confidence: {result.get('confidence', 0.0):.2f}")

            return result

        except json.JSONDecodeError as e:
            print(f"[EXTRACT_ASSESS] Error parsing LLM response: {e}")
            print(f"[EXTRACT_ASSESS] Response: {content}")

            # Fallback: mark as insufficient, suggest continuing
            return {
                "updated_variable_X": state.variable_X,
                "facts_extracted": [],
                "is_sufficient": False,
                "missing_info": ["Error in extraction"],
                "next_query_suggestion": None,
                "confidence": 0.3,
                "reasoning": "Error parsing extraction results"
            }

        except Exception as e:
            print(f"[EXTRACT_ASSESS] Error in extraction: {e}")
            import traceback
            traceback.print_exc()

            # Fallback
            return {
                "updated_variable_X": state.variable_X,
                "facts_extracted": [],
                "is_sufficient": False,
                "missing_info": ["Technical error"],
                "next_query_suggestion": None,
                "confidence": 0.2,
                "reasoning": f"Error in extraction: {str(e)}"
            }
