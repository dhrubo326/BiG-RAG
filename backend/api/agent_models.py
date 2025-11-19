"""
Pydantic models for the Agent API endpoint.

This module defines request/response models for the multi-hop reasoning agent.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class PlannedQuery(BaseModel):
    """A query planned by the agent for execution."""
    query: str = Field(..., description="The search query text")
    language: str = Field(default="English", description="Language for this query")
    reason: str = Field(..., description="Why this query is needed")


class ExecutedAction(BaseModel):
    """An action executed by the agent."""
    action_type: str = Field(default="search_bigrag", description="Type of action")
    query: str = Field(..., description="Query executed")
    language: str = Field(..., description="Language used")
    num_results: int = Field(..., description="Number of results retrieved")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")


class ContextItem(BaseModel):
    """A single context item from retrieval."""
    text: str = Field(..., description="Context text")
    source: Optional[str] = Field(None, description="Source document ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    relevance_score: Optional[float] = Field(None, description="Relevance score")


class Observation(BaseModel):
    """Observations from executing actions."""
    query: str = Field(..., description="Query that was executed")
    contexts: List[ContextItem] = Field(default_factory=list, description="Retrieved contexts")
    summary: Optional[str] = Field(None, description="Compressed summary of contexts")


class ReasoningStep(BaseModel):
    """A single step in the agent's reasoning trace."""
    step: int = Field(..., description="Step number (1-indexed)")
    thought: str = Field(..., description="Agent's reasoning for this step")
    planned_queries: List[PlannedQuery] = Field(default_factory=list, description="Queries planned")
    executed_actions: List[ExecutedAction] = Field(default_factory=list, description="Actions executed")
    observations: List[Observation] = Field(default_factory=list, description="Observations from actions")
    variables_stored: Dict[str, Any] = Field(default_factory=dict, description="Intermediate results stored")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this step")
    execution_time_ms: float = Field(..., description="Execution time for this step")


class AgentMetadata(BaseModel):
    """Metadata about agent execution."""
    model_used: str = Field(..., description="LLM model used for reasoning")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    execution_time_ms: float = Field(..., description="Total execution time")
    queries_executed: int = Field(..., description="Number of BiG-RAG queries executed")
    stopped_reason: str = Field(
        ...,
        description="Why execution stopped: 'max_iterations' | 'high_confidence' | 'complete'"
    )


class AgentRequest(BaseModel):
    """Request model for agent endpoint."""
    question: str = Field(..., description="User's question")
    language: str = Field(default="auto", description="Language preference (auto-detect or specify)")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Maximum reasoning iterations")
    agent_model: str = Field(default="gpt-4o", description="LLM model for reasoning")
    enable_parallel: bool = Field(default=True, description="Enable parallel query execution")

    # BiG-RAG retrieval parameters (aligned with /ask and /chat/completions)
    top_k_per_query: int = Field(default=60, ge=10, le=100, description="Items to retrieve from vector DBs per query")
    num_kg_in_context: int = Field(default=15, ge=1, le=30, description="KG relations in final context per query")
    num_chunks_in_context: int = Field(default=5, ge=0, le=20, description="Text chunks in final context per query")
    enable_reranking: bool = Field(default=False, description="Enable semantic reranking (requires sentence-transformers)")

    # Advanced options
    enable_variable_storage: bool = Field(default=True, description="Enable intermediate result storage")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Early stopping threshold")
    data_source: Optional[str] = Field(None, description="Dataset to query (overrides default)")


class AgentResponse(BaseModel):
    """Response model for agent endpoint."""
    answer: str = Field(..., description="Final synthesized answer")
    reasoning_trace: List[ReasoningStep] = Field(default_factory=list, description="Full execution trace")
    total_iterations: int = Field(..., description="Number of iterations executed")
    contexts_used: List[ContextItem] = Field(default_factory=list, description="All contexts retrieved")
    metadata: AgentMetadata = Field(..., description="Execution metadata")

    # Optional fields
    limitations: Optional[str] = Field(None, description="Any limitations or caveats")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in answer")
