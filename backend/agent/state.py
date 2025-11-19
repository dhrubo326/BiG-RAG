"""
Agent state management.

Tracks execution state across reasoning iterations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from api.agent_models import ReasoningStep, PlannedQuery, ExecutedAction, Observation, ContextItem


@dataclass
class AgentState:
    """
    Simplified agent state for efficient multi-hop reasoning.

    Uses variable_X to accumulate important knowledge across iterations,
    avoiding lossy extraction and pruning.
    """

    # Core execution parameters
    question: str
    current_iteration: int = 0
    max_iterations: int = 3

    # ═══════════════════════════════════════════════════════════
    # VARIABLE X: Accumulated knowledge (the core state)
    # ═══════════════════════════════════════════════════════════
    variable_X: Dict[str, Any] = field(default_factory=dict)
    # Structure: {"facts": {...}, "sources": [...], "confidence": float}

    # Reasoning trace (for debugging and response)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    actions: List[ExecutedAction] = field(default_factory=list)

    # Iteration summaries (for debugging - shown to user)
    iteration_summaries: List[str] = field(default_factory=list)

    # All contexts seen (kept for final response, not used in reasoning)
    all_contexts: List[ContextItem] = field(default_factory=list)

    # Execution metadata
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    start_time: Optional[datetime] = None
    model_used: str = "gpt-4o"
    queries_executed: int = 0

    # Confidence tracking
    step_confidences: List[float] = field(default_factory=list)

    def add_reasoning_step(self, step: ReasoningStep):
        """Add a completed reasoning step to the trace."""
        self.reasoning_steps.append(step)
        self.thoughts.append(step.thought)
        self.actions.extend(step.executed_actions)
        self.step_confidences.append(step.confidence)

        # Track executed queries (for stats)
        for action in step.executed_actions:
            self.queries_executed += 1

        # Note: variable_X is updated separately in extract_and_assess()
        # step.variables_stored contains the facts extracted in this step

        # Accumulate contexts from observations
        for obs in step.observations:
            self.all_contexts.extend(obs.contexts)

    def has_executed_similar_query(self, query: str, threshold: float = 0.9) -> bool:
        """
        Check if a similar query has been executed.

        Note: Simplified agent allows duplicate queries for sequential multi-hop.
        This method is kept for compatibility but returns False.
        """
        # In simplified agent, we allow duplicate queries since each iteration
        # uses results from previous iteration to refine the query
        return False

    def get_variable(self, key: str) -> Optional[Any]:
        """Retrieve a stored variable from variable_X."""
        return self.variable_X.get(key)

    def store_variable(self, key: str, value: Any):
        """Store an intermediate result in variable_X."""
        self.variable_X[key] = value

    def get_execution_time_ms(self) -> float:
        """Calculate total execution time in milliseconds."""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds() * 1000

    def get_average_confidence(self) -> float:
        """Calculate average confidence across all steps."""
        if not self.step_confidences:
            return 0.0
        return sum(self.step_confidences) / len(self.step_confidences)

    def get_latest_confidence(self) -> float:
        """Get confidence from the most recent step."""
        if not self.step_confidences:
            return 0.0
        return self.step_confidences[-1]

    def increment_iteration(self):
        """Move to the next iteration."""
        self.current_iteration += 1

    def should_continue(self, confidence_threshold: float = 0.8) -> bool:
        """
        Determine if agent should continue iterating.

        Returns False if:
        - Max iterations reached
        - Confidence threshold exceeded
        """
        # Check max iterations
        if self.current_iteration >= self.max_iterations:
            return False

        # Check confidence threshold (if we have at least one step)
        if self.step_confidences and self.get_latest_confidence() >= confidence_threshold:
            return False

        return True

    def get_iteration_history(self) -> str:
        """
        Format iteration history for LLM context.

        Returns a human-readable summary of previous steps.
        """
        if not self.reasoning_steps:
            return "No previous iterations."

        history = []
        for i, step in enumerate(self.reasoning_steps, 1):
            history.append(f"--- Iteration {i} ---")
            history.append(f"Thought: {step.thought}")
            history.append(f"Queries: {', '.join(q.query for q in step.planned_queries)}")
            history.append(f"Confidence: {step.confidence:.2f}")
            if step.variables_stored:
                history.append(f"Variables stored: {list(step.variables_stored.keys())}")
            history.append("")

        return "\n".join(history)

    def get_current_knowledge(self) -> str:
        """
        Format current variable_X as a knowledge summary.

        Returns a human-readable summary of stored knowledge.
        """
        if not self.variable_X:
            return "No knowledge gathered yet."

        knowledge = []
        for key, value in self.variable_X.items():
            if key == "metadata":
                continue  # Skip metadata for brevity
            knowledge.append(f"- {key}: {value}")

        return "\n".join(knowledge)

    def add_iteration_summary(self, summary: str):
        """Add a summary for this iteration."""
        self.iteration_summaries.append(summary)

    def add_metadata_facts(self, facts: Dict[str, Any]):
        """
        Add extracted metadata facts (legacy method, not used in simplified agent).

        Simplified agent stores all facts in variable_X instead.
        """
        # No-op: simplified agent uses variable_X for all knowledge
        pass

    def add_pruned_contexts(self, contexts: List[ContextItem]):
        """
        Add pruned contexts (legacy method, not used in simplified agent).

        Simplified agent keeps ALL contexts without pruning.
        """
        # No-op: simplified agent doesn't prune contexts
        pass
