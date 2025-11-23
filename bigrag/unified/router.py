"""
Subgraph Router - LLM-based query routing to relevant subgraphs.

Uses LLM to analyze query and subgraph metadata to determine which
subgraph(s) are most relevant for a given query.
"""

import json
import logging
from typing import Dict, List, Callable, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class SubgraphRouter:
    """Routes queries to relevant subgraphs using LLM analysis."""

    def __init__(
        self,
        registry_path: str = "expr/subgraph_registry.json",
        llm_func: Optional[Callable] = None
    ):
        """
        Initialize router.

        Args:
            registry_path: Path to subgraph_registry.json
            llm_func: Async LLM completion function (e.g., gpt_4o_mini_complete)
        """
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
        self.llm_func = llm_func

        if not self.llm_func:
            logger.warning("No LLM function provided - router will use fallback logic")

    def _load_registry(self) -> Dict:
        """Load and validate subgraph registry."""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Subgraph registry not found at {self.registry_path}"
            )

        with open(self.registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)

        # Validate registry structure
        if 'subgraphs' not in registry:
            raise ValueError("Registry missing 'subgraphs' key")

        # Filter to enabled subgraphs only
        enabled_subgraphs = {
            name: config
            for name, config in registry['subgraphs'].items()
            if config.get('enabled', True)
        }

        registry['subgraphs'] = enabled_subgraphs
        logger.info(
            f"Loaded registry with {len(enabled_subgraphs)} enabled subgraphs: "
            f"{list(enabled_subgraphs.keys())}"
        )

        return registry

    def _build_routing_prompt(self, query: str) -> str:
        """Build LLM prompt for routing decision."""
        subgraph_info = []
        for name, config in self.registry['subgraphs'].items():
            info = f"""
Subgraph: {name}
Description: {config['description']}
Aliases: {', '.join(config['aliases'])}
Topics: {', '.join(config['topics'][:10])}  # Limit to first 10 topics
"""
            subgraph_info.append(info.strip())

        subgraph_list = '\n\n'.join(subgraph_info)

        prompt = f"""You are a query router for a multi-subgraph knowledge base system.
Your task is to analyze the user's query and determine which subgraph(s) contain relevant information.

Available Subgraphs:
{subgraph_list}

User Query: "{query}"

Instructions:
1. Analyze the query to identify the main topic/domain
2. Match the query against subgraph descriptions, aliases, and topics
3. Select 1-3 most relevant subgraphs (prefer fewer if possible)
4. Provide confidence score (0.0-1.0) and brief reasoning

Response Format (JSON):
{{
  "subgraphs": ["subgraph_name1", "subgraph_name2"],
  "reasoning": "Brief explanation of why these subgraphs were selected",
  "confidence": 0.95
}}

Important:
- If query clearly matches ONE subgraph, only select that one
- If query is ambiguous or cross-domain, select 2-3 subgraphs
- If no clear match, select the fallback subgraph: "{self.registry.get('routing_config', {}).get('fallback_subgraph', 'demo_test')}"
- Respond ONLY with valid JSON, no additional text
"""
        return prompt

    def _parse_routing_response(self, llm_response: str) -> Dict:
        """Parse LLM routing response into structured format."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = llm_response.strip()
            if response_text.startswith('```'):
                # Remove markdown code blocks
                lines = response_text.split('\n')
                response_text = '\n'.join(
                    line for line in lines
                    if not line.strip().startswith('```')
                )

            routing_decision = json.loads(response_text)

            # Validate structure
            if 'subgraphs' not in routing_decision:
                raise ValueError("Missing 'subgraphs' key in routing response")

            # Ensure subgraphs is a list
            if isinstance(routing_decision['subgraphs'], str):
                routing_decision['subgraphs'] = [routing_decision['subgraphs']]

            # Validate selected subgraphs exist
            valid_subgraphs = []
            for sg in routing_decision['subgraphs']:
                if sg in self.registry['subgraphs']:
                    valid_subgraphs.append(sg)
                else:
                    logger.warning(f"Invalid subgraph '{sg}' - ignoring")

            routing_decision['subgraphs'] = valid_subgraphs

            # Add defaults
            routing_decision.setdefault('reasoning', 'No reasoning provided')
            routing_decision.setdefault('confidence', 0.5)

            return routing_decision

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse routing response as JSON: {e}")
            logger.debug(f"Raw LLM response: {llm_response}")
            return self._fallback_routing()

        except Exception as e:
            logger.error(f"Error parsing routing response: {e}")
            return self._fallback_routing()

    def _fallback_routing(self) -> Dict:
        """Return fallback routing decision when LLM routing fails."""
        fallback_sg = self.registry.get('routing_config', {}).get(
            'fallback_subgraph', 'demo_test'
        )

        # Ensure fallback exists in registry
        if fallback_sg not in self.registry['subgraphs']:
            # Use first available subgraph
            fallback_sg = list(self.registry['subgraphs'].keys())[0]

        return {
            'subgraphs': [fallback_sg],
            'reasoning': 'Fallback routing - LLM routing failed or not configured',
            'confidence': 0.3
        }

    async def route(self, query: str, force_subgraphs: Optional[List[str]] = None) -> Dict:
        """
        Route query to relevant subgraph(s).

        Args:
            query: User query string
            force_subgraphs: Optional list of subgraphs to force (bypass routing)

        Returns:
            Dict with keys:
                - subgraphs: List[str] - Selected subgraph names
                - reasoning: str - Explanation of routing decision
                - confidence: float - Confidence score (0.0-1.0)
        """
        # If force_subgraphs provided, skip LLM routing
        if force_subgraphs:
            valid_subgraphs = [
                sg for sg in force_subgraphs
                if sg in self.registry['subgraphs']
            ]
            if not valid_subgraphs:
                logger.warning(
                    f"None of forced subgraphs {force_subgraphs} are valid - "
                    "using fallback"
                )
                return self._fallback_routing()

            return {
                'subgraphs': valid_subgraphs,
                'reasoning': 'Forced subgraph selection (bypass routing)',
                'confidence': 1.0
            }

        # Use LLM routing if available
        if self.llm_func:
            try:
                prompt = self._build_routing_prompt(query)
                llm_response = await self.llm_func(
                    prompt,
                    max_tokens=300,
                    temperature=0.0
                )
                routing_decision = self._parse_routing_response(llm_response)

                logger.info(
                    f"Routed query to subgraphs: {routing_decision['subgraphs']} "
                    f"(confidence: {routing_decision['confidence']:.2f})"
                )

                return routing_decision

            except Exception as e:
                logger.error(f"LLM routing failed: {e}")
                return self._fallback_routing()

        # No LLM - use fallback
        return self._fallback_routing()

    def get_subgraph_info(self, subgraph_name: str) -> Optional[Dict]:
        """Get metadata for a specific subgraph."""
        return self.registry['subgraphs'].get(subgraph_name)

    def list_subgraphs(self) -> List[str]:
        """List all enabled subgraph names."""
        return list(self.registry['subgraphs'].keys())

    def reload_registry(self):
        """Reload registry from disk (useful if registry updated)."""
        self.registry = self._load_registry()
        logger.info("Registry reloaded")
