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
    Tracks the state of agent execution across iterations.

    This includes the reasoning trace, intermediate results,
    and metadata about execution.
    """

    # Core execution parameters
    question: str
    current_iteration: int = 0
    max_iterations: int = 3

    # Reasoning trace
    thoughts: List[str] = field(default_factory=list)
    actions: List[ExecutedAction] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)

    # Intermediate results storage
    variables: Dict[str, Any] = field(default_factory=dict)
    all_contexts: List[ContextItem] = field(default_factory=list)
    pruned_contexts: List[ContextItem] = field(default_factory=list)  # Only keep 2-3 best per iteration

    # Iteration summaries (concise, not raw contexts)
    iteration_summaries: List[str] = field(default_factory=list)

    # Metadata facts extracted from chunks
    metadata_facts: List[Dict[str, Any]] = field(default_factory=list)

    # Execution metadata
    total_tokens: int = 0
    start_time: Optional[datetime] = None
    model_used: str = "gpt-4o"

    # Query deduplication
    executed_queries: List[str] = field(default_factory=list)

    # Confidence tracking
    step_confidences: List[float] = field(default_factory=list)

    def add_reasoning_step(self, step: ReasoningStep):
        """Add a completed reasoning step to the trace."""
        self.reasoning_steps.append(step)
        self.thoughts.append(step.thought)
        self.actions.extend(step.executed_actions)
        self.observations.extend(step.observations)
        self.step_confidences.append(step.confidence)

        # Track executed queries for deduplication
        for action in step.executed_actions:
            self.executed_queries.append(action.query.lower().strip())

        # Store variables
        self.variables.update(step.variables_stored)

        # Accumulate contexts
        for obs in step.observations:
            self.all_contexts.extend(obs.contexts)

    def has_executed_similar_query(self, query: str, threshold: float = 0.9) -> bool:
        """
        Check if a similar query has been executed.

        For now, uses simple string matching.
        TODO: Use embedding similarity for better detection.
        """
        query_normalized = query.lower().strip()

        # Exact match check
        if query_normalized in self.executed_queries:
            return True

        # Simple substring check (can be improved with embeddings)
        for executed in self.executed_queries:
            if query_normalized in executed or executed in query_normalized:
                # Check length similarity to avoid false positives
                len_ratio = len(query_normalized) / len(executed)
                if 0.8 <= len_ratio <= 1.2:
                    return True

        return False

    def get_variable(self, key: str) -> Optional[Any]:
        """Retrieve a stored variable."""
        return self.variables.get(key)

    def store_variable(self, key: str, value: Any):
        """Store an intermediate result."""
        self.variables[key] = value

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
        Format current variables as a knowledge summary.

        Returns a human-readable summary of stored variables.
        """
        if not self.variables:
            return "No knowledge gathered yet."

        knowledge = []
        for key, value in self.variables.items():
            knowledge.append(f"- {key}: {value}")

        return "\n".join(knowledge)

    def add_iteration_summary(self, summary: str):
        """Add a summary for this iteration."""
        self.iteration_summaries.append(summary)

    def add_metadata_facts(self, facts: Dict[str, Any]):
        """Add extracted metadata facts."""
        if facts:
            self.metadata_facts.append(facts)

    def add_pruned_contexts(self, contexts: List[ContextItem]):
        """Add pruned contexts (only the best 2-3 per iteration)."""
        self.pruned_contexts.extend(contexts)
