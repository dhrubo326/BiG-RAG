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

# =========================================================================
# SIMPLIFIED AGENT PROMPTS (2-call-per-iteration design)
# =========================================================================

# Simplified: Plan next action (answer OR search)
SIMPLIFIED_PLAN_NEXT_ACTION_PROMPT = """You are a research assistant with access to a knowledge retrieval system.

QUESTION: {question}

ITERATION: {current_iteration} / {max_iterations}

ACCUMULATED KNOWLEDGE (variable_X):
{variable_x}

PREVIOUS ACTIONS:
{action_history}

YOUR TASK: Decide what to do next.

OPTIONS:
1. ANSWER: You have enough information to answer the question confidently
2. QUERY: You need more information - plan ONE specific search query

DECISION CRITERIA:
- Choose ANSWER if:
  * variable_X contains all facts needed to answer the question
  * Confidence is high (>0.8) based on accumulated knowledge
  * You've exhausted reasonable search paths

- Choose QUERY if:
  * Missing critical facts to answer the question
  * Have partial information but need more specifics
  * Can formulate a specific query to fill the gap

IMPORTANT:
- Plan ONE query at a time (sequential approach for multi-hop)
- Be specific - use knowledge from variable_X to refine your query
- Choose appropriate language for the query

OUTPUT FORMAT (must be valid JSON):
{{
  "action": "answer" or "query",
  "reasoning": "explain your decision",
  "confidence": 0.0 to 1.0,

  // IF action == "query":
  "query": "specific search query",
  "query_language": "English/Bangla/etc",
  "query_purpose": "what this query will help establish",

  // IF action == "answer":
  "answer": "your complete answer to the question",
  "answer_sources": ["list of key facts from variable_X used"]
}}

EXAMPLE 1 - Need more info:
Question: "Who is the captain of the 2022 World Cup winner?"
variable_X: {{}}
Output:
{{
  "action": "query",
  "reasoning": "Need to first find out which country won the 2022 World Cup",
  "confidence": 0.0,
  "query": "2022 FIFA World Cup winner country",
  "query_language": "English",
  "query_purpose": "Identify the winning country"
}}

EXAMPLE 2 - Have partial, need more:
Question: "Who is the captain of the 2022 World Cup winner?"
variable_X: {{"world_cup_winner": "Argentina", "tournament_date": "December 2022"}}
Output:
{{
  "action": "query",
  "reasoning": "Know Argentina won, now need to find their captain",
  "confidence": 0.5,
  "query": "Argentina national team captain 2022 World Cup",
  "query_language": "English",
  "query_purpose": "Find the captain of Argentina's winning team"
}}

EXAMPLE 3 - Can answer:
Question: "Who is the captain of the 2022 World Cup winner?"
variable_X: {{"world_cup_winner": "Argentina", "argentina_captain": "Lionel Messi", "confidence": 0.95}}
Output:
{{
  "action": "answer",
  "reasoning": "Have both facts: Argentina won, Messi was captain. High confidence from multiple sources.",
  "confidence": 0.95,
  "answer": "Lionel Messi is the captain of Argentina, who won the 2022 FIFA World Cup.",
  "answer_sources": ["world_cup_winner: Argentina", "argentina_captain: Lionel Messi"]
}}

Now make your decision:
"""

# Simplified: Extract facts and assess sufficiency
SIMPLIFIED_EXTRACT_AND_ASSESS_PROMPT = """You are extracting facts from search results to update accumulated knowledge.

QUESTION: {question}

SEARCH QUERY EXECUTED: {query}

CURRENT KNOWLEDGE (variable_X):
{variable_x}

SEARCH RESULTS (from BiG-RAG):
{contexts}

YOUR TASKS:
1. Extract IMPORTANT facts from the search results
2. Update variable_X with new knowledge (preserve old facts, add new ones)
3. Assess if we now have sufficient information to answer the question

EXTRACTION GUIDELINES:
- Extract SPECIFIC facts: names, numbers, dates, relationships
- Do NOT extract full paragraphs or generic background
- Focus on facts that help answer the question
- Include source information for each fact
- Keep facts concise (1-2 sentences max per fact)

VARIABLE_X STRUCTURE:
{{
  "fact_key_1": {{"value": "...", "source": "context index", "confidence": 0.0-1.0}},
  "fact_key_2": {{"value": "...", "source": "context index", "confidence": 0.0-1.0}},
  ...
  "metadata": {{
    "entities_found": ["list of key entities"],
    "last_query": "what was just searched"
  }}
}}

SUFFICIENCY ASSESSMENT:
- is_sufficient: true if variable_X contains all facts needed to answer
- missing_info: list of what's still unknown
- next_query_suggestion: if not sufficient, suggest what to search next

OUTPUT FORMAT (must be valid JSON):
{{
  "updated_variable_X": {{...updated variable_X structure...}},
  "facts_extracted": ["list of fact keys added/updated"],
  "is_sufficient": true or false,
  "missing_info": ["list of what's still unknown"],
  "next_query_suggestion": "suggested next query (if not sufficient)",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation of sufficiency assessment"
}}

EXAMPLE:
Question: "Who is the captain of the 2022 World Cup winner?"
Query: "2022 FIFA World Cup winner country"
Current variable_X: {{}}
Contexts: [
  [0] "Argentina won the 2022 FIFA World Cup...",
  [1] "The tournament was held in Qatar from November to December 2022...",
  [2] "Lionel Messi led Argentina to victory..."
]

Output:
{{
  "updated_variable_X": {{
    "world_cup_winner": {{"value": "Argentina", "source": 0, "confidence": 0.95}},
    "tournament_year": {{"value": "2022", "source": 1, "confidence": 0.95}},
    "tournament_location": {{"value": "Qatar", "source": 1, "confidence": 0.9}},
    "partial_captain_info": {{"value": "Messi mentioned as leader", "source": 2, "confidence": 0.7}},
    "metadata": {{
      "entities_found": ["Argentina", "Lionel Messi", "Qatar"],
      "last_query": "2022 FIFA World Cup winner country"
    }}
  }},
  "facts_extracted": ["world_cup_winner", "tournament_year", "tournament_location", "partial_captain_info"],
  "is_sufficient": false,
  "missing_info": ["Need explicit confirmation that Messi was the captain"],
  "next_query_suggestion": "Argentina national team captain 2022 World Cup Messi",
  "confidence": 0.6,
  "reasoning": "Found the winning country (Argentina) with high confidence. Messi is mentioned as a leader but need explicit confirmation he was the captain."
}}

Now extract and assess:
"""
