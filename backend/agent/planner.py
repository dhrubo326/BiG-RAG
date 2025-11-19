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
