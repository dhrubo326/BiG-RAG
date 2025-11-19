"""
Agent API endpoint.

Provides multi-hop reasoning capabilities for BiG-RAG.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import os

from api.agent_models import AgentRequest, AgentResponse
from agent.executor import AgentExecutor


# Create router
router = APIRouter(prefix="/agent", tags=["agent"])

# Global executor instance (will be initialized on startup)
_executor: Optional[AgentExecutor] = None


def get_executor() -> AgentExecutor:
    """Dependency to get executor instance."""
    if _executor is None:
        raise HTTPException(
            status_code=500,
            detail="Agent executor not initialized. Make sure BiG-RAG is loaded."
        )
    return _executor


def initialize_agent(bigrag_instance, model: str = "gpt-4o", api_key: Optional[str] = None):
    """
    Initialize the agent executor.

    This should be called on server startup.

    Args:
        bigrag_instance: BiGRAG instance
        model: LLM model to use
        api_key: OpenAI API key (defaults to env var)
    """
    global _executor

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[AGENT] WARNING: OPENAI_API_KEY not found. Agent endpoint will not work.")
        return

    _executor = AgentExecutor(
        bigrag_instance=bigrag_instance,
        model=model,
        api_key=api_key
    )

    print(f"[AGENT] Initialized with model: {model}")


@router.post("/query", response_model=AgentResponse)
async def agent_query(
    request: AgentRequest,
    executor: AgentExecutor = Depends(get_executor)
):
    """
    Multi-hop reasoning agent endpoint.

    This endpoint uses an LLM to iteratively plan and execute BiG-RAG queries
    until sufficient evidence is gathered to answer the question.

    Process:
    1. Plan: LLM decides what queries to execute
    2. Act: Execute queries using BiG-RAG
    3. Observe: Collect results
    4. Repeat: Continue until confident or max iterations reached
    5. Synthesize: Generate final answer

    Args:
        request: Agent request with question and parameters

    Returns:
        Agent response with answer and reasoning trace

    Example:
        ```
        POST /agent/query
        {
            "question": "Who is the captain of the 2022 World Cup winner?",
            "max_iterations": 3,
            "agent_model": "gpt-4o"
        }
        ```
    """
    try:
        # Execute agent
        response = await executor.execute(request)
        return response

    except Exception as e:
        import traceback
        print(f"[AGENT] Error in agent_query: {e}")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Error executing agent: {str(e)}"
        )


@router.get("/health")
async def agent_health():
    """
    Check if agent is ready.

    Returns:
        Status information
    """
    if _executor is None:
        return {
            "status": "not_ready",
            "message": "Agent executor not initialized",
            "ready": False
        }

    return {
        "status": "ready",
        "message": "Agent is ready",
        "ready": True,
        "model": _executor.model
    }


@router.get("/info")
async def agent_info():
    """
    Get information about the agent.

    Returns:
        Agent configuration and capabilities
    """
    return {
        "name": "BiG-RAG Multi-Hop Reasoning Agent",
        "version": "1.0.0",
        "description": "Iterative query planning and execution for complex questions",
        "capabilities": [
            "Multi-hop reasoning",
            "Dynamic query generation",
            "Parallel query execution",
            "Multilingual support",
            "Intermediate result storage",
            "Confidence-based early stopping"
        ],
        "supported_languages": [
            "English", "Bangla", "Hindi", "Arabic", "Chinese",
            "Spanish", "French", "German", "Japanese", "Korean"
        ],
        "max_iterations": 5,
        "default_model": _executor.model if _executor else "gpt-4o"
    }
