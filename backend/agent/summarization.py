"""
Iteration summarization for agent.

Provides:
- Per-iteration summary (what was learned)
- Progressive knowledge accumulation
- Metadata-aware summarization
"""

import json
from typing import List, Dict, Any
from openai import AsyncOpenAI

from api.agent_models import Observation, ContextItem


class IterationSummarizer:
    """
    Summarizes findings from each agent iteration.

    Instead of carrying 30+ contexts through iterations,
    we carry concise summaries of key findings.
    """

    def __init__(self, openai_client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model

    async def summarize_iteration(
        self,
        step_number: int,
        question: str,
        observations: List[Observation],
        variables_stored: Dict[str, Any],
        metadata_facts: Dict[str, Any]
    ) -> str:
        """
        Summarize key findings from this iteration.

        Args:
            step_number: Iteration number
            question: Original question
            observations: Retrieved observations
            variables_stored: Variables extracted this iteration
            metadata_facts: Facts extracted from metadata

        Returns:
            Concise summary (2-4 sentences)
        """
        # Collect query info
        queries_executed = [obs.query for obs in observations]

        # Collect contexts by source type
        chunk_contexts = []
        kg_contexts = []  # entities + relations

        for obs in observations:
            for ctx in obs.contexts:
                source_type = ctx.metadata.get("type", "unknown")
                if source_type == "chunk":
                    chunk_contexts.append(ctx)
                elif source_type in ["entity", "relation"]:
                    kg_contexts.append(ctx)

        # Format contexts (brief)
        chunks_str = "\n".join([f"- {ctx.text[:200]}" for ctx in chunk_contexts[:5]])
        kg_str = "\n".join([f"- {ctx.text[:150]}" for ctx in kg_contexts[:5]])

        prompt = f"""Summarize the key findings from this research iteration.

ORIGINAL QUESTION: {question}
ITERATION: {step_number}

QUERIES EXECUTED:
{', '.join(queries_executed)}

CONTEXTS FROM VECTOR CHUNKS (with metadata):
{chunks_str if chunks_str else "None"}

CONTEXTS FROM KNOWLEDGE GRAPH:
{kg_str if kg_str else "None"}

VARIABLES EXTRACTED:
{json.dumps(variables_stored, indent=2) if variables_stored else "None"}

METADATA FACTS:
{json.dumps(metadata_facts, indent=2) if metadata_facts else "None"}

Provide a 2-4 sentence summary covering:
1. What specific facts were found (names, numbers, dates)
2. What source types were most useful (chunks vs KG)
3. What key information is still missing (if any)

Be concise and factual. Focus on WHAT was learned, not HOW.

Example:
"Found that KUET has 130 seats in CSE department (from chunk metadata and KG relations).
Document titles confirm this is for 2024 admission.
Still need information about EEE seat count at KUET."
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research summarizer. Be concise and specific."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=250
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            print(f"[SUMMARIZATION] Error summarizing iteration: {e}")
            # Fallback: basic summary
            return f"Iteration {step_number}: Searched for {', '.join(queries_executed)}. Found {len(variables_stored)} variables."

    async def assess_information_sufficiency(
        self,
        question: str,
        iteration_summaries: List[str],
        variables_collected: Dict[str, Any],
        max_iterations: int,
        current_iteration: int
    ) -> Dict[str, Any]:
        """
        Assess if we have enough information to answer the question.

        Helps agent decide whether to continue or stop.

        Args:
            question: Original question
            iteration_summaries: Summaries from all iterations so far
            variables_collected: All variables extracted
            max_iterations: Maximum allowed iterations
            current_iteration: Current iteration number

        Returns:
            {
                "is_sufficient": bool,
                "confidence": 0.0-1.0,
                "missing_info": List[str],
                "should_continue": bool,
                "reasoning": str
            }
        """
        summaries_str = "\n\n".join([f"Iteration {i+1}: {s}" for i, s in enumerate(iteration_summaries)])

        prompt = f"""Assess if we have sufficient information to answer the question.

QUESTION: {question}

INFORMATION GATHERED SO FAR ({current_iteration}/{max_iterations} iterations):
{summaries_str}

VARIABLES EXTRACTED:
{json.dumps(variables_collected, indent=2)}

Evaluate:
1. Can this question be answered with the information gathered?
2. What critical information is still missing (if any)?
3. Would additional searches likely find new useful information?

OUTPUT FORMAT (must be valid JSON):
{{
  "is_sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_info": ["list of missing facts"],
  "should_continue": true/false,
  "reasoning": "brief explanation"
}}

DECISION LOGIC:
- is_sufficient=true if all key facts are found
- should_continue=false if:
  * is_sufficient=true AND confidence >= 0.7
  * OR current_iteration >= max_iterations
  * OR additional searches unlikely to help
- should_continue=true if:
  * Missing critical facts
  * AND current_iteration < max_iterations
  * AND searches might find new info
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an information sufficiency evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"[SUMMARIZATION] Error assessing sufficiency: {e}")
            # Fallback: continue if not at max
            return {
                "is_sufficient": False,
                "confidence": 0.5,
                "missing_info": ["unknown"],
                "should_continue": current_iteration < max_iterations,
                "reasoning": f"Error in assessment: {e}"
            }

    def format_for_synthesis(
        self,
        iteration_summaries: List[str],
        variables_collected: Dict[str, Any],
        top_contexts: List[ContextItem],
        max_contexts: int = 10
    ) -> Dict[str, str]:
        """
        Format accumulated knowledge for final synthesis.

        Instead of sending 30+ raw contexts, send:
        - Iteration summaries (concise)
        - Extracted variables (structured)
        - Top 10 raw contexts (for citation)

        Args:
            iteration_summaries: All iteration summaries
            variables_collected: All extracted variables
            top_contexts: Top contexts for reference
            max_contexts: Max contexts to include

        Returns:
            {
                "summaries": formatted summaries,
                "variables": formatted variables,
                "top_contexts": formatted contexts
            }
        """
        # Format summaries
        summaries_formatted = "\n\n".join([
            f"**Iteration {i+1}**:\n{summary}"
            for i, summary in enumerate(iteration_summaries)
        ])

        # Format variables
        variables_formatted = json.dumps(variables_collected, indent=2)

        # Format top contexts (for citation)
        contexts_formatted = []
        for idx, ctx in enumerate(top_contexts[:max_contexts]):
            source_type = ctx.metadata.get("type", "unknown")
            source = ctx.source or f"context_{idx}"

            # Include metadata if available
            metadata_str = ""
            if source_type == "chunk" and ctx.metadata:
                title = ctx.metadata.get("title", "")
                if title:
                    metadata_str = f" [Title: {title}]"

            contexts_formatted.append(
                f"[{idx}] ({source_type}){metadata_str}\n"
                f"Source: {source}\n"
                f"{ctx.text[:400]}"
            )

        contexts_str = "\n\n".join(contexts_formatted)

        return {
            "summaries": summaries_formatted,
            "variables": variables_formatted,
            "top_contexts": contexts_str
        }
