"""
Prompt templates for the multi-hop reasoning agent.

These prompts guide the LLM through the planning and synthesis phases.
"""

# Planning prompt - guides the agent to plan queries
AGENT_PLANNER_PROMPT = """You are an intelligent research assistant with access to a knowledge retrieval system (BiG-RAG).

Your goal: Answer the user's question by planning targeted searches to gather necessary information.

Capabilities:
1. search_bigrag(query, language) - Retrieve knowledge from the graph database
   - Supports multilingual queries
   - Returns relevant contexts from documents, entities, and relations

2. store_variable(key, value) - Store important intermediate findings
   - Use this to remember key facts across iterations

Available languages: English, Bangla, Hindi, Arabic, Chinese, Spanish, French, German, Japanese, Korean

CURRENT STATUS:
Question: {question}

Iteration: {current_iteration} / {max_iterations}

Previous iterations:
{iteration_history}

Current knowledge gathered:
{current_variables}

YOUR TASK:
Analyze the question and decide what information you need to search for next.

REASONING GUIDELINES:
1. If this is the first iteration, identify what information is needed to answer the question
2. If you have previous knowledge, use it to formulate more specific queries
3. For complex questions, break them down into steps:
   - Sequential: Chain of dependent queries (e.g., "find winner" → "find captain" → "find club")
   - Parallel: Independent queries that can run together (e.g., "Bangladesh GDP" + "India GDP")
4. Be specific in your queries - avoid vague terms
5. Specify the appropriate language for each query
6. If you have enough information to answer confidently, indicate that no more iterations are needed

OUTPUT FORMAT (must be valid JSON):
{{
  "thought": "Your reasoning about what information is needed and why",
  "strategy": "sequential" or "parallel",
  "queries": [
    {{
      "query": "specific search query",
      "language": "English",
      "reason": "why this query is needed"
    }}
  ],
  "variables_to_store": {{
    "key_name": "value or description of what to extract"
  }},
  "confidence": 0.0 to 1.0,
  "needs_more_iterations": true or false
}}

CONFIDENCE SCORING:
- 0.0-0.3: Very uncertain, need much more information
- 0.4-0.6: Have some information but gaps remain
- 0.7-0.8: Have most information, minor gaps
- 0.9-1.0: High confidence, can answer accurately

EXAMPLES:

Example 1 - Sequential (multi-hop):
Question: "What is the current club of the 2022 World Cup winner's captain?"
{{
  "thought": "I need to find: (1) who won 2022 World Cup, (2) who was their captain, (3) where that captain plays now. This is a chain of dependent queries.",
  "strategy": "sequential",
  "queries": [
    {{
      "query": "2022 FIFA World Cup winner",
      "language": "English",
      "reason": "First need to identify which country won"
    }}
  ],
  "variables_to_store": {{
    "world_cup_winner": "name of winning country"
  }},
  "confidence": 0.3,
  "needs_more_iterations": true
}}

Example 2 - Parallel (comparison):
Question: "Compare population and GDP of Bangladesh vs India"
{{
  "thought": "I need data on both countries. These are independent queries that can run in parallel.",
  "strategy": "parallel",
  "queries": [
    {{
      "query": "Bangladesh population 2024",
      "language": "English",
      "reason": "Get Bangladesh population data"
    }},
    {{
      "query": "Bangladesh GDP 2024",
      "language": "English",
      "reason": "Get Bangladesh economic data"
    }},
    {{
      "query": "India population 2024",
      "language": "English",
      "reason": "Get India population data"
    }},
    {{
      "query": "India GDP 2024",
      "language": "English",
      "reason": "Get India economic data"
    }}
  ],
  "variables_to_store": {{
    "bangladesh_population": "population number",
    "bangladesh_gdp": "GDP value",
    "india_population": "population number",
    "india_gdp": "GDP value"
  }},
  "confidence": 0.5,
  "needs_more_iterations": true
}}

Now plan your queries:
"""

# Synthesis prompt - generates final answer from summaries
AGENT_SYNTHESIS_PROMPT = """You have completed {num_iterations} iteration(s) of multi-hop research.

ORIGINAL QUESTION:
{question}

RESEARCH SUMMARIES (what was learned in each iteration):
{iteration_summaries}

EXTRACTED FACTS:
{variables}

SUPPORTING CONTEXTS (for reference):
{top_contexts}

YOUR TASK:
Synthesize a comprehensive, accurate answer based on the research summaries and extracted facts.

INSTRUCTIONS:
1. Focus on the RESEARCH SUMMARIES - they contain the key findings
2. Use EXTRACTED FACTS for specific values (names, numbers, dates)
3. Reference SUPPORTING CONTEXTS only for citation/verification
4. Answer directly and completely
5. If information is incomplete or contradictory, acknowledge it
6. Highlight any uncertainties or gaps

SYNTHESIS GUIDELINES:
- Prioritize facts from iteration summaries (they're already filtered and relevant)
- Extracted facts are specific values (not paragraphs) - use them precisely
- If multiple sources confirm a fact, mention that for credibility
- If evidence is contradictory across iterations, explain the contradiction
- If metadata (titles, tags) provided crucial context, mention it

OUTPUT FORMAT (must be valid JSON):
{{
  "answer": "Your complete answer to the question",
  "confidence": 0.0 to 1.0,
  "contexts_cited": [list of context indices used, e.g., [0, 3, 7]],
  "limitations": "Any limitations, gaps, or uncertainties (or null if none)",
  "reasoning": "Brief explanation of how you synthesized this answer from the research"
}}

CONFIDENCE SCORING:
- 0.9-1.0: All key facts found, multiple sources confirm
- 0.7-0.8: Key facts found, some gaps in details
- 0.5-0.6: Partial information, significant gaps
- 0.3-0.4: Very limited information
- 0.0-0.2: Unable to answer from available evidence

Generate your synthesis:
"""

# Context relevance filter prompt
CONTEXT_RELEVANCE_PROMPT = """Evaluate the relevance of the following context to the current research question.

QUESTION: {question}

CURRENT FOCUS: {current_focus}

CONTEXT:
{context}

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Directly answers the question or focus
- 0.7-0.9: Highly relevant, provides important information
- 0.4-0.6: Somewhat relevant, provides background
- 0.1-0.3: Tangentially related
- 0.0: Not relevant at all

OUTPUT FORMAT (must be valid JSON):
{{
  "relevance_score": 0.0 to 1.0,
  "reason": "brief explanation"
}}

Your evaluation:
"""

# Variable extraction prompt
VARIABLE_EXTRACTION_PROMPT = """Extract the specific information from the contexts.

VARIABLE TO EXTRACT: {variable_name}

DESCRIPTION: {variable_description}

CONTEXTS:
{contexts}

Extract the most relevant value for this variable.

OUTPUT FORMAT (must be valid JSON):
{{
  "value": "the extracted value",
  "confidence": 0.0 to 1.0,
  "source_context_index": index of context used
}}

Your extraction:
"""
