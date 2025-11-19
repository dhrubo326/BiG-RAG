"""
Context extraction and pruning for agent.

Provides:
- LLM-based variable extraction (replace stub)
- Context relevance scoring
- Intelligent context pruning (keep 2-3 best, discard noise)
- Source-aware extraction (vector chunks vs KG)
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI

from api.agent_models import ContextItem


class ContextExtractor:
    """
    Handles intelligent extraction and pruning of contexts.

    Key features:
    - Extract specific facts (not paragraphs)
    - Score context relevance
    - Prune noise between iterations
    - Source-aware handling (chunks vs KG)
    """

    def __init__(self, openai_client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model

    async def extract_variable(
        self,
        variable_name: str,
        variable_description: str,
        contexts: List[ContextItem],
        question: str
    ) -> Dict[str, Any]:
        """
        Extract specific fact from contexts using LLM.

        Returns SPECIFIC values (names, numbers, dates), NOT paragraphs.

        Args:
            variable_name: Name of variable to extract
            variable_description: Description of what to extract
            contexts: Retrieved contexts
            question: Original question

        Returns:
            {
                "value": "extracted value",
                "confidence": 0.0-1.0,
                "source_index": int,
                "source_type": "chunk" | "entity" | "relation"
            }
        """
        if not contexts:
            return {
                "value": "NOT_FOUND",
                "confidence": 0.0,
                "source_index": -1,
                "source_type": "none"
            }

        # Format contexts with source type info
        context_lines = []
        for idx, ctx in enumerate(contexts[:10]):  # Limit to top 10
            source_type = ctx.metadata.get("type", "unknown")
            context_lines.append(f"[{idx}] ({source_type}) {ctx.text[:300]}")

        contexts_str = "\n\n".join(context_lines)

        prompt = f"""Extract the specific information requested from the contexts.

ORIGINAL QUESTION: {question}

VARIABLE TO EXTRACT: {variable_name}
DESCRIPTION: {variable_description}

CONTEXTS (with source types):
{contexts_str}

EXTRACTION RULES:
- Extract ONLY the specific value requested
- For entity names: Return just the name (e.g., "KUET", "Argentina", "Lionel Messi")
- For numbers: Return just the number with units (e.g., "120 seats", "1.4 billion", "130")
- For dates: Return just the date (e.g., "2022", "December 18, 2022")
- If the information is not found: Return "NOT_FOUND"
- Do NOT return full sentences or paragraphs
- Prefer information from "chunk" sources (they have document context)
- If found in multiple sources, choose the most specific one

OUTPUT FORMAT (must be valid JSON):
{{
  "value": "the extracted specific value",
  "confidence": 0.0 to 1.0,
  "source_index": index of context used (0-9),
  "reasoning": "brief explanation of why this value was chosen"
}}

Example:
Question: "Which university has 120 seats in CSE?"
Variable: university_name
Contexts: [0] "KUET has 130 seats in CSE and 120 in EEE..."
Output: {{"value": "KUET", "confidence": 0.9, "source_index": 0, "reasoning": "KUET mentioned with seat numbers"}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise information extractor. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temp for factual extraction
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Add source type
            if 0 <= result.get("source_index", -1) < len(contexts):
                source_ctx = contexts[result["source_index"]]
                result["source_type"] = source_ctx.metadata.get("type", "unknown")
            else:
                result["source_type"] = "unknown"

            return result

        except Exception as e:
            print(f"[EXTRACTION] Error extracting variable: {e}")
            return {
                "value": "ERROR",
                "confidence": 0.0,
                "source_index": -1,
                "source_type": "error",
                "reasoning": str(e)
            }

    async def score_contexts(
        self,
        query: str,
        contexts: List[ContextItem],
        original_question: str
    ) -> List[Dict[str, Any]]:
        """
        Score each context for relevance to the query.

        Returns list of:
        {
            "context_index": int,
            "relevance_score": 0.0-1.0,
            "is_useful": bool,
            "reason": str,
            "source_type": "chunk" | "entity" | "relation"
        }
        """
        if not contexts:
            return []

        # Format contexts
        context_lines = []
        for idx, ctx in enumerate(contexts):
            source_type = ctx.metadata.get("type", "unknown")
            # Include metadata for chunks
            metadata_str = ""
            if source_type == "chunk" and ctx.metadata:
                title = ctx.metadata.get("title", "")
                tags = ctx.metadata.get("tags", [])
                if title:
                    metadata_str = f" [Title: {title}]"
                if tags:
                    metadata_str += f" [Tags: {', '.join(tags[:3])}]"

            context_lines.append(f"[{idx}] ({source_type}){metadata_str}\n{ctx.text[:400]}")

        contexts_str = "\n\n".join(context_lines)

        prompt = f"""Score each context for relevance to answering the query.

ORIGINAL QUESTION: {original_question}
SEARCH QUERY: {query}

CONTEXTS:
{contexts_str}

For each context, evaluate:
1. Does it contain information relevant to the query?
2. Is the information specific and useful (not generic)?
3. For chunks: Does the metadata (title/tags) add missing context?

SCORING RULES:
- 0.9-1.0: Directly answers the query with specific facts
- 0.7-0.8: Contains relevant information, may need other contexts
- 0.5-0.6: Somewhat relevant, provides background
- 0.3-0.4: Tangentially related
- 0.0-0.2: Not relevant

SOURCE TYPE GUIDANCE:
- "chunk" contexts: Look for specific facts in text AND metadata
- "entity" contexts: Look for entity descriptions
- "relation" contexts: Look for relationships between entities

OUTPUT FORMAT (must be valid JSON):
{{
  "scores": [
    {{
      "context_index": 0,
      "relevance_score": 0.0-1.0,
      "is_useful": true/false,
      "reason": "brief explanation"
    }},
    ...
  ]
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a context relevance evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            scores = result.get("scores", [])

            # Add source type to each score
            for score in scores:
                idx = score.get("context_index", -1)
                if 0 <= idx < len(contexts):
                    score["source_type"] = contexts[idx].metadata.get("type", "unknown")

            return scores

        except Exception as e:
            print(f"[EXTRACTION] Error scoring contexts: {e}")
            # Fallback: return all contexts with medium score
            return [
                {
                    "context_index": idx,
                    "relevance_score": 0.5,
                    "is_useful": True,
                    "reason": "Error in scoring, keeping by default",
                    "source_type": ctx.metadata.get("type", "unknown")
                }
                for idx, ctx in enumerate(contexts)
            ]

    async def prune_contexts(
        self,
        contexts: List[ContextItem],
        query: str,
        original_question: str,
        keep_top_n: int = 3
    ) -> Tuple[List[ContextItem], List[Dict[str, Any]]]:
        """
        Prune contexts to keep only the most relevant ones.

        From 10 contexts, keep only 2-3 best. Discard noise.

        Args:
            contexts: All retrieved contexts
            query: Query executed
            original_question: Original user question
            keep_top_n: Number of contexts to keep (default 3)

        Returns:
            (pruned_contexts, scores)
        """
        print(f"[PRUNING] Pruning {len(contexts)} contexts to top {keep_top_n}...")

        # Score all contexts
        scores = await self.score_contexts(query, contexts, original_question)

        # Sort by relevance score
        scores_sorted = sorted(scores, key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Keep top N
        top_indices = [s["context_index"] for s in scores_sorted[:keep_top_n]]
        pruned = [contexts[i] for i in top_indices if i < len(contexts)]

        print(f"[PRUNING] Kept contexts {top_indices} with scores {[s['relevance_score'] for s in scores_sorted[:keep_top_n]]}")

        return pruned, scores_sorted[:keep_top_n]

    async def extract_metadata_facts(
        self,
        contexts: List[ContextItem],
        question: str
    ) -> Dict[str, Any]:
        """
        Extract facts from chunk metadata (title, tags).

        Useful when text mentions "120 seats" but doesn't specify university,
        but metadata has title="KUET Admission Info".

        Args:
            contexts: Contexts to extract from (only chunks have metadata)
            question: Original question

        Returns:
            {
                "entities_from_metadata": ["KUET", "CSE"],
                "facts_from_titles": {"university": "KUET", ...},
                "relevant_tags": ["education", "university"]
            }
        """
        # Collect metadata from chunk contexts
        chunk_metadata = []
        for ctx in contexts:
            if ctx.metadata.get("type") == "chunk":
                title = ctx.metadata.get("title", "")
                tags = ctx.metadata.get("tags", [])
                category = ctx.metadata.get("category", "")

                if title or tags or category:
                    chunk_metadata.append({
                        "title": title,
                        "tags": tags,
                        "category": category,
                        "text_snippet": ctx.text[:200]
                    })

        if not chunk_metadata:
            return {
                "entities_from_metadata": [],
                "facts_from_titles": {},
                "relevant_tags": []
            }

        prompt = f"""Extract useful facts from document metadata.

QUESTION: {question}

DOCUMENT METADATA (from chunks):
{json.dumps(chunk_metadata, indent=2)}

Extract:
1. Entity names mentioned in titles (universities, people, places)
2. Key facts that can be inferred from titles/tags
3. Relevant tags that provide context

OUTPUT FORMAT (must be valid JSON):
{{
  "entities_from_metadata": ["list of entity names found in titles"],
  "facts_from_titles": {{"key": "value pairs of facts"}},
  "relevant_tags": ["list of relevant tags"]
}}

Example:
Title: "KUET Admission 2024 - CSE Department Seats"
Output: {{
  "entities_from_metadata": ["KUET", "CSE"],
  "facts_from_titles": {{"university": "KUET", "department": "CSE", "year": "2024"}},
  "relevant_tags": ["admission", "engineering"]
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a metadata fact extractor. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[EXTRACTION] Error extracting metadata facts: {e}")
            return {
                "entities_from_metadata": [],
                "facts_from_titles": {},
                "relevant_tags": []
            }
