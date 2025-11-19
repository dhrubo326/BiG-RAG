"""
Prompt templates for BiG-RAG.

Contains prompt templates for:
- Agent query planning
- Answer synthesis
- Context relevance filtering
"""

from prompts.agent_prompts import (
    AGENT_PLANNER_PROMPT,
    AGENT_SYNTHESIS_PROMPT,
    CONTEXT_RELEVANCE_PROMPT,
    VARIABLE_EXTRACTION_PROMPT
)

__all__ = [
    "AGENT_PLANNER_PROMPT",
    "AGENT_SYNTHESIS_PROMPT",
    "CONTEXT_RELEVANCE_PROMPT",
    "VARIABLE_EXTRACTION_PROMPT"
]
