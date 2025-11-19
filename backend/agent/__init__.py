"""
Agent module for multi-hop reasoning.

Enhanced with:
- LLM-based variable extraction
- Context pruning (keep only relevant info)
- Iteration summaries (avoid context overload)
- Source-aware extraction (chunks vs KG)
- Multilingual search support
"""

from agent.executor import AgentExecutor
from agent.planner import QueryPlanner
from agent.state import AgentState
from agent.tools import AgentTools
from agent.extraction import ContextExtractor
from agent.summarization import IterationSummarizer

__all__ = [
    "AgentExecutor",
    "QueryPlanner",
    "AgentState",
    "AgentTools",
    "ContextExtractor",
    "IterationSummarizer"
]
