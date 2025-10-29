# Part 7: Extension System

**Deep-Dive Documentation for BiG-RAG Framework**

---

## 1. Conceptual Overview

### What Problem Does This Solve?

**Problem:** Different use cases need different components:
- **Custom Storage**: Integrate with existing databases (PostgreSQL, Elasticsearch)
- **New Tools**: Add domain-specific retrieval (SQL, web search, calculators)
- **Custom Extraction**: Industry-specific entity types (medical, legal, financial)
- **LLM Flexibility**: Use proprietary models or local deployments
- **Embedding Choice**: Optimize for domain or language
- **Reward Functions**: Task-specific optimization metrics

**BiG-RAG Solution:** Plugin architecture with extension points:
- **Storage Plugins**: Abstract base classes for graph, vector, KV storage
- **Tool Plugins**: Extensible tool system via ToolEnv
- **Extraction Plugins**: Customizable prompts and parsers
- **LLM Plugins**: Provider abstraction layer
- **Embedding Plugins**: Swappable embedding functions
- **Reward Plugins**: Custom reward computation

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTENSION ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  BiGRAG Core                                                 │
│    ├─ Abstract Interfaces                                   │
│    │   ├─ BaseGraphStorage                                  │
│    │   ├─ BaseVectorStorage                                 │
│    │   ├─ BaseKVStorage                                     │
│    │   ├─ ToolBase                                          │
│    │   └─ RewardBase                                        │
│    │                                                         │
│    ├─ Extension Points                                      │
│    │   ├─ llm_model_func (LLM provider)                     │
│    │   ├─ embedding_func (embedding model)                  │
│    │   ├─ entity_types (extraction schema)                  │
│    │   ├─ entity_relationship_prompt (extraction logic)     │
│    │   └─ reward_func (RL optimization)                     │
│    │                                                         │
│    └─ Plugin Registry                                       │
│        ├─ lazy_external_import() (storage)                  │
│        ├─ ToolEnv.register_tool() (tools)                   │
│        └─ RewardManager.register() (rewards)                │
│                                                              │
│  User Extensions                                            │
│    ├─ custom_storage.py                                     │
│    ├─ custom_tools.py                                       │
│    ├─ custom_extraction.py                                  │
│    ├─ custom_llm.py                                         │
│    └─ custom_reward.py                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Implementation Details

### Storage Plugin System

**File:** `bigrag/base.py`, `bigrag/bigrag.py`

**Registration Pattern:**
```python
# bigrag/bigrag.py
def lazy_external_import(cls_name: str):
    """
    Lazy load storage backends to avoid requiring all dependencies

    Args:
        cls_name: Storage class name (e.g., "Neo4JStorage")

    Returns:
        Storage class or None if not found
    """
    # Graph storage
    if cls_name == "Neo4JStorage":
        from bigrag.kg.graph_impl.neo4j_impl import Neo4JStorage
        return Neo4JStorage
    elif cls_name == "OracleGraphStorage":
        from bigrag.kg.graph_impl.oracle_impl import OracleGraphStorage
        return OracleGraphStorage

    # Vector storage
    elif cls_name == "MilvusVectorDBStorage":
        from bigrag.kg.vectordb_impl.milvus_impl import MilvusVectorDBStorage
        return MilvusVectorDBStorage
    elif cls_name == "ChromaVectorDBStorage":
        from bigrag.kg.vectordb_impl.chroma_impl import ChromaVectorDBStorage
        return ChromaVectorDBStorage

    # KV storage
    elif cls_name == "MongoKVStorage":
        from bigrag.kg.kv_impl.mongo_impl import MongoKVStorage
        return MongoKVStorage

    # Return None if not found (will use default)
    return None
```

**Usage Pattern:**
```python
# bigrag/bigrag.py - BiGRAG.__init__()
def __init__(self, graph_storage: str = "NetworkXStorage", ...):
    # Load class via lazy import
    GraphStorage = lazy_external_import(graph_storage)

    if GraphStorage is None:
        # Fallback to default
        from bigrag.storage import NetworkXStorage
        GraphStorage = NetworkXStorage

    # Instantiate with config
    self.graph_storage = GraphStorage(
        namespace="graph_chunk_entity_relation",
        working_dir=self.working_dir,
        **storage_config
    )
```

### Tool Plugin System

**File:** `agent/tool/tool_env.py`, `agent/tool/search.py`

**Tool Registration:**
```python
# agent/tool/tool_env.py
class ToolEnv:
    def __init__(self, env: str, tool_config: dict):
        self.env = env
        self.tools = {}
        self._register_tools(tool_config)

    def _register_tools(self, config: dict):
        """Register available tools based on env type"""
        if self.env == "search":
            from agent.tool.search import SearchTool
            self.tools["search"] = SearchTool(
                api_url=config.get("api_url", "http://localhost:8001")
            )

        elif self.env == "custom":
            # Load custom tools from config
            for tool_name, tool_cls in config.get("tools", {}).items():
                self.tools[tool_name] = tool_cls(config.get(tool_name, {}))

    def execute(self, tool_name: str, query: str) -> str:
        """Execute tool by name"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(query)
```

**Tool Interface:**
```python
# agent/tool/base.py
class ToolBase(ABC):
    @abstractmethod
    def execute(self, query: str, **kwargs) -> str:
        """
        Execute tool with query

        Args:
            query: User query string
            **kwargs: Additional parameters

        Returns:
            Tool response as string
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM prompt"""
        pass
```

### LLM Provider System

**File:** `bigrag/llm.py`

**Provider Pattern:**
```python
# bigrag/llm.py
async def openai_complete(prompt: str, **kwargs) -> str:
    """OpenAI completion"""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=kwargs.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=kwargs.get("temperature", 0.0)
    )
    return response.choices[0].message.content

async def anthropic_complete(prompt: str, **kwargs) -> str:
    """Anthropic completion"""
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=kwargs.get("model", "claude-3-5-sonnet-20241022"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=kwargs.get("max_tokens", 4096)
    )
    return response.content[0].text

# Usage: Pass function to BiGRAG
rag = BiGRAG(llm_model_func=openai_complete)
# or
rag = BiGRAG(llm_model_func=anthropic_complete)
```

### Embedding System

**File:** `bigrag/operate.py`

**Embedding Function Pattern:**
```python
# bigrag/operate.py
async def default_embedding_func(texts: list[str]) -> np.ndarray:
    """
    Default embedding function using FlagEmbedding

    Args:
        texts: List of text strings

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    embeddings = embedding_model.encode(texts)
    return np.array(embeddings)

# Custom embedding example
async def custom_embedding_func(texts: list[str]) -> np.ndarray:
    """Custom embedding with your model"""
    # Your implementation
    embeddings = my_model.encode(texts)
    return np.array(embeddings)

# Usage
rag = BiGRAG(embedding_func=custom_embedding_func)
```

### Extraction Customization

**File:** `bigrag/prompt.py`, `bigrag/operate.py`

**Custom Prompts:**
```python
# bigrag/prompt.py
ENTITY_RELATIONSHIP_PROMPT_MEDICAL = """
You are a medical knowledge extractor. Extract entities and relationships from the text.

Entity Types:
- Disease: Medical conditions
- Symptom: Clinical symptoms
- Treatment: Therapeutic interventions
- Medication: Pharmaceutical drugs
- Anatomy: Body parts

Relationship Types:
- causes: Disease -> Symptom
- treats: Treatment -> Disease
- affects: Disease -> Anatomy

Text: {input_text}

Output format:
("entity", "entity_type", "description")
("source_entity", "relationship_type", "target_entity", "description")
"""

# Usage
rag = BiGRAG(
    entity_types=["Disease", "Symptom", "Treatment", "Medication", "Anatomy"],
    entity_relationship_prompt=ENTITY_RELATIONSHIP_PROMPT_MEDICAL
)
```

### Reward System

**File:** `verl/utils/reward_score/base.py`

**Custom Reward Function:**
```python
# verl/utils/reward_score/custom_reward.py
async def custom_reward_func(
    responses: list[str],
    ground_truths: list[str],
    **kwargs
) -> list[float]:
    """
    Compute custom rewards for RL training

    Args:
        responses: Model-generated responses
        ground_truths: Reference answers

    Returns:
        List of reward scores (one per response)
    """
    rewards = []

    for response, truth in zip(responses, ground_truths):
        # Extract answer from tags
        answer = extract_answer(response)

        # Custom scoring logic
        reward = 0.0

        # Format reward (has proper tags)
        if "<answer>" in response and "</answer>" in response:
            reward += 0.5

        # Content reward (semantic similarity)
        similarity = compute_similarity(answer, truth)
        reward += similarity * 0.5

        rewards.append(reward)

    return rewards
```

---

## 3. Configuration Reference

### Storage Backend Configuration

```python
# Default storage (development)
rag = BiGRAG(
    working_dir="./expr/my_kg",
    graph_storage="NetworkXStorage",      # In-memory graph
    vector_storage="NanoVectorDBStorage", # FAISS-based
    kv_storage="JsonKVStorage"            # JSON files
)

# Production storage
rag = BiGRAG(
    working_dir="./expr/prod",
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorage",
    kv_storage="MongoKVStorage",
    storage_config={
        # Neo4J config
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",

        # Milvus config
        "milvus_host": "localhost",
        "milvus_port": 19530,
        "milvus_collection": "bigrag_vectors",

        # MongoDB config
        "mongo_uri": "mongodb://localhost:27017",
        "mongo_db": "bigrag",
        "mongo_collection": "entities"
    }
)
```

### Tool Configuration

```python
# Default search tool
tool_config = {
    "env": "search",
    "api_url": "http://localhost:8001",
    "max_turns": 5,
    "query_start_tag": "<query>",
    "query_end_tag": "</query>",
    "knowledge_start_tag": "<knowledge>",
    "knowledge_end_tag": "</knowledge>"
}

# Custom tools
tool_config = {
    "env": "custom",
    "tools": {
        "search": SearchTool,
        "calculator": CalculatorTool,
        "sql": SQLTool
    },
    "search": {"api_url": "http://localhost:8001"},
    "calculator": {"precision": 6},
    "sql": {"connection_string": "postgresql://..."}
}

# In training script
tool_env = ToolEnv(**tool_config)
```

### LLM Provider Configuration

```python
# Environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."

# Code configuration
from bigrag.llm import openai_complete, anthropic_complete, gemini_complete

# Single provider
rag = BiGRAG(llm_model_func=openai_complete)

# Multi-provider with load balancing
from bigrag.llm import MultiModel, Model

models = [
    Model(func=openai_complete, name="gpt-4o-mini"),
    Model(func=anthropic_complete, name="claude-sonnet"),
    Model(func=gemini_complete, name="gemini-flash")
]

multi_model = MultiModel(models)
rag = BiGRAG(llm_model_func=multi_model.llm_model_func)
```

### Embedding Configuration

```python
# Default FlagEmbedding
rag = BiGRAG()  # Uses bge-large-en-v1.5

# Custom embedding model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

async def custom_embed(texts: list[str]) -> np.ndarray:
    return model.encode(texts)

rag = BiGRAG(embedding_func=custom_embed)

# OpenAI embeddings
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def openai_embed(texts: list[str]) -> np.ndarray:
    response = await client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([e.embedding for e in response.data])

rag = BiGRAG(embedding_func=openai_embed)
```

---

## 4. Usage Examples

**⚠️ IMPORTANT: Example Status Disclaimer**

The examples in this section are **illustrative templates** that demonstrate the extension patterns and API design. They are not currently implemented in the BiG-RAG codebase.

**What this means:**
- ✅ **Extension Pattern**: The approach shown (inheriting from base classes, implementing required methods) is correct and tested
- ✅ **API Design**: The method signatures and interfaces shown match the actual base classes in `bigrag/base.py`
- ✅ **Use as Templates**: You can copy these examples and implement them for real use
- ❌ **Not Pre-Built**: These specific implementations (PostgreSQL storage, SQL tool, legal extraction, code generation reward) don't exist in `bigrag/kg/` or `agent/tool/tools/`

**Status of Built-in Extensions:**
| Extension | Status | Location |
|-----------|--------|----------|
| **NetworkXStorage** | ✅ Implemented | `bigrag/storage.py:178-318` |
| **NanoVectorDBStorage** | ✅ Implemented | `bigrag/storage.py:67-175` |
| **JsonKVStorage** | ✅ Implemented | `bigrag/storage.py:26-64` |
| **Neo4JStorage** | ✅ Implemented | `bigrag/kg/neo4j_impl.py` |
| **OracleStorage** | ✅ Implemented | `bigrag/kg/oracle_impl.py` |
| **MilvusStorage** | ✅ Implemented | `bigrag/kg/milvus_impl.py` |
| **SearchTool** | ✅ Implemented | `agent/tool/tools/search_tool.py` |
| **PostgreSQLStorage** | ⚠️ Template Example | Not implemented |
| **SQLTool** | ⚠️ Template Example | Not implemented |
| **Legal Extraction** | ⚠️ Template Example | Not implemented |
| **Code Generation Reward** | ⚠️ Template Example | Not implemented |

**How to Use These Examples:**
1. Copy the template code
2. Install required dependencies (`asyncpg`, `sqlparse`, etc.)
3. Implement any missing methods marked with `# TODO`
4. Test thoroughly before production use
5. Contribute back to BiG-RAG if you build something useful!

### Example 1: Adding PostgreSQL Vector Storage (Template)

**File:** `custom_extensions/postgres_storage.py`

```python
import asyncpg
import numpy as np
from bigrag.base import BaseVectorStorage

class PostgresVectorStorage(BaseVectorStorage):
    """PostgreSQL with pgvector extension"""

    def __init__(self, namespace: str, working_dir: str, **config):
        self.namespace = namespace
        self.connection_string = config.get(
            "connection_string",
            "postgresql://user:pass@localhost/bigrag"
        )
        self.pool = None

    async def _ensure_connection(self):
        """Initialize connection pool"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.connection_string)

            # Create table with vector extension
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;

                    CREATE TABLE IF NOT EXISTS {namespace} (
                        id TEXT PRIMARY KEY,
                        embedding vector(1536),
                        content TEXT,
                        metadata JSONB
                    );

                    CREATE INDEX IF NOT EXISTS {namespace}_embedding_idx
                    ON {namespace}
                    USING ivfflat (embedding vector_cosine_ops);
                """.format(namespace=self.namespace))

    async def query(self, query: str, top_k: int) -> list[dict]:
        """Query by vector similarity"""
        await self._ensure_connection()

        # Embed query
        query_embedding = await self.global_config["embedding_func"]([query])
        query_vector = query_embedding[0].tolist()

        # Search
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM {namespace}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """.format(namespace=self.namespace), query_vector, top_k)

        # Format results
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": row["metadata"],
                "similarity": row["similarity"]
            }
            for row in rows
        ]

    async def upsert(self, data: dict[str, dict]):
        """Insert or update vectors"""
        await self._ensure_connection()

        async with self.pool.acquire() as conn:
            for id, item in data.items():
                await conn.execute("""
                    INSERT INTO {namespace} (id, embedding, content, metadata)
                    VALUES ($1, $2::vector, $3, $4)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata
                """.format(namespace=self.namespace),
                    id,
                    item["__vector__"],
                    item.get("content", ""),
                    item.get("metadata", {})
                )

    async def index_done_callback(self):
        """Finalize indexing"""
        # Analyze table for query optimization
        async with self.pool.acquire() as conn:
            await conn.execute(f"ANALYZE {self.namespace}")

# Register in bigrag/bigrag.py
def lazy_external_import(cls_name: str):
    if cls_name == "PostgresVectorStorage":
        from custom_extensions.postgres_storage import PostgresVectorStorage
        return PostgresVectorStorage
    # ... rest of imports

# Usage
rag = BiGRAG(
    vector_storage="PostgresVectorStorage",
    storage_config={
        "connection_string": "postgresql://user:pass@localhost/bigrag"
    }
)
```

### Example 2: Adding SQL Query Tool (Template)

**File:** `custom_extensions/sql_tool.py`

```python
import sqlalchemy
from agent.tool.base import ToolBase

class SQLTool(ToolBase):
    """Execute SQL queries on database"""

    def __init__(self, config: dict):
        self.engine = sqlalchemy.create_engine(
            config.get("connection_string")
        )
        self.allowed_operations = config.get(
            "allowed_operations",
            ["SELECT"]  # Security: only allow reads
        )

    @property
    def name(self) -> str:
        return "sql"

    @property
    def description(self) -> str:
        return (
            "Execute SQL queries on the database. "
            "Use for structured data retrieval. "
            "Example: <query>SELECT * FROM users WHERE age > 30</query>"
        )

    def execute(self, query: str, **kwargs) -> str:
        """Execute SQL query safely"""
        # Security check
        query_upper = query.strip().upper()
        if not any(query_upper.startswith(op) for op in self.allowed_operations):
            return f"Error: Only {self.allowed_operations} operations allowed"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(query))
                rows = result.fetchall()

                # Format as markdown table
                if not rows:
                    return "No results found"

                # Header
                columns = result.keys()
                output = "| " + " | ".join(columns) + " |\n"
                output += "| " + " | ".join(["---"] * len(columns)) + " |\n"

                # Rows (limit to 10)
                for row in rows[:10]:
                    output += "| " + " | ".join(str(v) for v in row) + " |\n"

                if len(rows) > 10:
                    output += f"\n... and {len(rows) - 10} more rows"

                return output

        except Exception as e:
            return f"SQL Error: {str(e)}"

# Register in training config
tool_config = {
    "env": "custom",
    "tools": {
        "search": SearchTool,
        "sql": SQLTool
    },
    "sql": {
        "connection_string": "sqlite:///data/company.db",
        "allowed_operations": ["SELECT"]
    }
}

# Update tool environment
from agent.tool.tool_env import ToolEnv

class MultiToolEnv(ToolEnv):
    def execute(self, tool_name: str, query: str) -> str:
        """Execute tool with automatic selection"""
        # Detect tool from query format
        if "<sql>" in query:
            return self.tools["sql"].execute(query)
        else:
            return self.tools["search"].execute(query)

# Usage in training
tool_env = MultiToolEnv(**tool_config)
```

### Example 3: Custom Entity Extraction for Legal Documents (Template)

**File:** `custom_extensions/legal_extraction.py`

```python
from bigrag import BiGRAG

# Define legal entity types
LEGAL_ENTITY_TYPES = [
    "Case",           # Legal cases
    "Statute",        # Laws and statutes
    "Party",          # Parties involved
    "Court",          # Courts
    "Judge",          # Judges
    "Attorney",       # Lawyers
    "Date",           # Important dates
    "Precedent"       # Legal precedents
]

# Custom extraction prompt
LEGAL_EXTRACTION_PROMPT = """
You are a legal document analyzer. Extract entities and relationships from the legal text.

Entity Types:
- Case: Legal cases (e.g., "Brown v. Board of Education")
- Statute: Laws, codes, regulations (e.g., "Title VII", "42 U.S.C. § 1983")
- Party: Plaintiffs, defendants, appellants
- Court: Courts involved (e.g., "Supreme Court", "9th Circuit")
- Judge: Judges mentioned
- Attorney: Lawyers representing parties
- Date: Important dates (filing, ruling, etc.)
- Precedent: Cited precedents

Relationship Types:
- cites: Case -> Precedent
- violates: Party -> Statute
- presides: Judge -> Case
- represents: Attorney -> Party
- rules_on: Court -> Case

Input Text:
{input_text}

Output format (one per line):
("entity", "entity_type", "description")
("source_entity", "relationship_type", "target_entity", "description")

Begin extraction:
"""

# Custom parser for legal citations
def parse_legal_entities(response: str) -> list[tuple]:
    """Parse extraction response with legal-specific rules"""
    entities = []

    for line in response.split('\n'):
        line = line.strip()

        # Skip empty lines
        if not line or line.startswith('#'):
            continue

        try:
            # Parse tuple format
            data = eval(line)

            # Entity format: (name, type, description)
            if len(data) == 3:
                name, entity_type, desc = data

                # Normalize case citations
                if entity_type == "Case":
                    name = normalize_case_citation(name)

                # Normalize statute citations
                elif entity_type == "Statute":
                    name = normalize_statute_citation(name)

                entities.append((name, entity_type, desc))

            # Relation format: (src, type, tgt, desc)
            elif len(data) == 4:
                entities.append(data)

        except:
            continue

    return entities

def normalize_case_citation(citation: str) -> str:
    """Normalize case name format"""
    # Remove trailing periods
    citation = citation.strip(' .')

    # Standardize "v." vs "vs" vs "versus"
    import re
    citation = re.sub(r'\s+(v|vs|versus)\.?\s+', ' v. ', citation, flags=re.IGNORECASE)

    return citation

def normalize_statute_citation(statute: str) -> str:
    """Normalize statute citation format"""
    # Remove spaces around § symbol
    statute = statute.replace(' § ', ' § ')

    # Standardize U.S.C. format
    import re
    statute = re.sub(r'(\d+)\s*USC\s*§?\s*(\d+)', r'\1 U.S.C. § \2', statute)

    return statute

# Build legal knowledge graph
rag = BiGRAG(
    working_dir="./expr/legal_kg",
    entity_types=LEGAL_ENTITY_TYPES,
    entity_relationship_prompt=LEGAL_EXTRACTION_PROMPT,
    entity_parser=parse_legal_entities
)

# Insert legal documents
legal_documents = [
    {
        "id": "case_001",
        "contents": "In Brown v. Board of Education, 347 U.S. 483 (1954), the Supreme Court..."
    },
    # ... more documents
]

await rag.ainsert(legal_documents)

# Query legal knowledge
context = await rag.aquery(
    "What precedents did the court cite in Brown v. Board of Education?",
    QueryParam(mode="hybrid", top_k=20)
)
```

### Example 4: Custom Reward Function for Code Generation (Template)

**File:** `custom_extensions/code_reward.py`

```python
import ast
import subprocess
import tempfile
from typing import List

async def code_generation_reward(
    responses: List[str],
    ground_truths: List[str],
    test_cases: List[dict],
    **kwargs
) -> List[float]:
    """
    Custom reward for code generation tasks

    Evaluates:
    1. Syntax validity (0.2)
    2. Test case pass rate (0.6)
    3. Code style (0.2)

    Args:
        responses: Generated code
        ground_truths: Reference implementations
        test_cases: Test cases with inputs/outputs

    Returns:
        Reward scores (0.0 to 1.0)
    """
    rewards = []

    for response, truth, tests in zip(responses, ground_truths, test_cases):
        reward = 0.0

        # Extract code from answer tags
        code = extract_code(response)

        # 1. Syntax validity (0.2)
        if is_valid_syntax(code):
            reward += 0.2
        else:
            # Invalid syntax = no further evaluation
            rewards.append(reward)
            continue

        # 2. Test case evaluation (0.6)
        test_results = run_test_cases(code, tests)
        pass_rate = sum(test_results) / len(test_results)
        reward += pass_rate * 0.6

        # 3. Code style (0.2)
        style_score = evaluate_style(code)
        reward += style_score * 0.2

        rewards.append(reward)

    return rewards

def extract_code(response: str) -> str:
    """Extract code from response"""
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    elif "<answer>" in response:
        code = response.split("<answer>")[1].split("</answer>")[0]
    else:
        code = response

    return code.strip()

def is_valid_syntax(code: str) -> bool:
    """Check Python syntax"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def run_test_cases(code: str, test_cases: List[dict]) -> List[bool]:
    """Run test cases and return pass/fail for each"""
    results = []

    for test in test_cases:
        try:
            # Create temp file with code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.write('\n\n')
                f.write(f"# Test case\n")
                f.write(f"input_data = {test['input']}\n")
                f.write(f"expected = {test['output']}\n")
                f.write(f"result = solution(input_data)\n")
                f.write(f"assert result == expected, f'Expected {{expected}}, got {{result}}'\n")
                temp_file = f.name

            # Run test
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                timeout=5
            )

            # Check if passed
            results.append(result.returncode == 0)

        except Exception:
            results.append(False)

        finally:
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)

    return results

def evaluate_style(code: str) -> float:
    """Evaluate code style (0.0 to 1.0)"""
    score = 1.0

    # Check for docstrings
    if '"""' not in code and "'''" not in code:
        score -= 0.3

    # Check for type hints
    if '->' not in code:
        score -= 0.2

    # Check line length (< 100 chars)
    for line in code.split('\n'):
        if len(line) > 100:
            score -= 0.1
            break

    # Check for comments
    if '#' not in code:
        score -= 0.2

    return max(0.0, score)

# Use in training config
reward_config = {
    "reward_func": code_generation_reward,
    "test_cases": load_test_cases("data/code_tests.json")
}
```

---

## 5. Troubleshooting

### Issue: Custom Storage Backend Not Loading

**Error:**
```
AttributeError: module 'custom_extensions.postgres_storage' has no attribute 'PostgresVectorStorage'
```

**Solutions:**

```python
# 1. Check import path in lazy_external_import()
def lazy_external_import(cls_name: str):
    if cls_name == "PostgresVectorStorage":
        # Ensure correct path
        from custom_extensions.postgres_storage import PostgresVectorStorage
        return PostgresVectorStorage

# 2. Ensure custom_extensions is a package
# custom_extensions/__init__.py should exist

# 3. Add to PYTHONPATH
import sys
sys.path.append('./custom_extensions')

# 4. Verify class inheritance
class PostgresVectorStorage(BaseVectorStorage):  # Must inherit
    pass

# 5. Check for circular imports
# Avoid importing BiGRAG inside storage class
```

### Issue: Custom Tool Not Executing

**Problem:** Tool registered but not called during generation

**Solutions:**

```python
# 1. Verify tool tags match config
tool_config = {
    "query_start_tag": "<query>",
    "query_end_tag": "</query>"
}

# Model MUST generate exact tags
# "<query>search for X</query>" ✓
# "<search>search for X</search>" ✗

# 2. Check tool registration
tool_env = ToolEnv(env="custom", tool_config=tool_config)
print(tool_env.tools)  # Should show your tool

# 3. Verify tool name in execute()
class CustomTool(ToolBase):
    @property
    def name(self) -> str:
        return "custom"  # Must match key in tools dict

# 4. Add tool description to prompt
system_prompt = f"""
Available tools:
{tool_env.get_tool_descriptions()}

To use a tool:
<query>your query here</query>
"""
```

### Issue: Custom Embedding Dimension Mismatch

**Error:**
```
ValueError: Input vector dimension 384 does not match index dimension 1536
```

**Solutions:**

```python
# 1. Rebuild FAISS indices with new dimension
import shutil
import os

# Delete old indices
for file in ["index_entity.bin", "index_bipartite_edge.bin", "index.bin"]:
    if os.path.exists(f"expr/my_kg/{file}"):
        os.remove(f"expr/my_kg/{file}")

# Rebuild with new embedding
rag = BiGRAG(
    working_dir="./expr/my_kg",
    embedding_func=custom_embed_384d  # 384-dim embeddings
)
await rag.ainsert(documents)  # Rebuilds indices

# 2. Use embedding dimension parameter
rag = BiGRAG(
    embedding_dim=384,  # Must match your embeddings
    embedding_func=custom_embed_384d
)

# 3. Verify embedding function output
embeddings = await custom_embed_384d(["test"])
print(embeddings.shape)  # Should be (1, 384)
```

### Issue: Custom Extraction Returns No Entities

**Problem:** LLM extraction returns empty results

**Solutions:**

```python
# 1. Test prompt with single document
from bigrag.operate import extract_entities

test_doc = "Paris is the capital of France."
entities = await extract_entities(
    [test_doc],
    entity_types=["City", "Country"],
    prompt=CUSTOM_PROMPT
)
print(entities)  # Debug output

# 2. Check prompt format
# Prompt MUST include {input_text} placeholder
prompt = """
Extract entities from:
{input_text}

Output format:
("entity", "type", "description")
"""

# 3. Add examples to prompt (few-shot)
prompt = """
Extract entities from text.

Example:
Input: "Paris is the capital of France."
Output:
("Paris", "City", "Capital of France")
("France", "Country", "European nation")

Now extract from:
{input_text}
"""

# 4. Check LLM response directly
from bigrag.llm import openai_complete

response = await openai_complete(prompt)
print(response)  # See what LLM outputs

# 5. Adjust entity_types to match domain
entity_types = ["Person", "Location", "Organization"]  # Generic
# vs
entity_types = ["Disease", "Symptom", "Drug"]  # Domain-specific
```

---

## 6. API Reference

### Storage Plugin Interface

```python
class BaseGraphStorage(ABC):
    """Abstract base class for graph storage backends"""

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if node exists"""
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> dict:
        """Retrieve node data"""
        pass

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict):
        """Insert or update node"""
        pass

    @abstractmethod
    async def upsert_edge(self, src: str, tgt: str, edge_data: dict):
        """Insert or update edge"""
        pass

    @abstractmethod
    async def get_node_edges(self, node_id: str) -> list[tuple]:
        """Get all edges for a node"""
        pass

    @abstractmethod
    async def index_done_callback(self):
        """Called after indexing completes"""
        pass

class BaseVectorStorage(ABC):
    """Abstract base class for vector storage backends"""

    @abstractmethod
    async def query(self, query: str, top_k: int) -> list[dict]:
        """
        Query by vector similarity

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of results with similarity scores
        """
        pass

    @abstractmethod
    async def upsert(self, data: dict[str, dict]):
        """
        Insert or update vectors

        Args:
            data: {id: {content, __vector__, ...}}
        """
        pass

    @abstractmethod
    async def index_done_callback(self):
        """Called after indexing completes"""
        pass

class BaseKVStorage(ABC):
    """Abstract base class for key-value storage backends"""

    @abstractmethod
    async def get_by_id(self, id: str) -> dict:
        """Get single item by ID"""
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str], fields: set[str] = None) -> list[dict]:
        """Get multiple items by IDs"""
        pass

    @abstractmethod
    async def filter_keys(self, data: list[str]) -> set[str]:
        """Filter to existing keys"""
        pass

    @abstractmethod
    async def upsert(self, data: dict[str, dict]):
        """Insert or update items"""
        pass

    @abstractmethod
    async def index_done_callback(self):
        """Called after indexing completes"""
        pass
```

### Tool Plugin Interface

```python
class ToolBase(ABC):
    """Abstract base class for tools"""

    @abstractmethod
    def execute(self, query: str, **kwargs) -> str:
        """
        Execute tool with query

        Args:
            query: User query string
            **kwargs: Additional parameters

        Returns:
            Tool response as string

        Raises:
            ToolExecutionError: If execution fails
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM prompt"""
        pass

    def format_response(self, result: Any) -> str:
        """
        Format tool result for LLM consumption

        Args:
            result: Raw tool output

        Returns:
            Formatted string
        """
        return str(result)
```

### Extension Registration

```python
def lazy_external_import(cls_name: str) -> type | None:
    """
    Lazy load storage backend class

    Args:
        cls_name: Class name (e.g., "Neo4JStorage")

    Returns:
        Storage class or None if not found

    Example:
        def lazy_external_import(cls_name: str):
            if cls_name == "PostgresVectorStorage":
                from custom_extensions.postgres_storage import PostgresVectorStorage
                return PostgresVectorStorage
            return None
    """
    pass

def register_tool(tool: ToolBase):
    """
    Register custom tool

    Args:
        tool: Tool instance implementing ToolBase

    Example:
        class MyTool(ToolBase):
            def execute(self, query: str) -> str:
                return "result"

            @property
            def name(self) -> str:
                return "my_tool"

        register_tool(MyTool())
    """
    pass
```

### Embedding Function Signature

```python
async def embedding_func(texts: list[str]) -> np.ndarray:
    """
    Embed text strings to vectors

    Args:
        texts: List of text strings

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
        - Must be 2D array
        - dtype should be float32 or float64
        - All vectors must have same dimension

    Example:
        async def my_embedding_func(texts: list[str]) -> np.ndarray:
            embeddings = model.encode(texts)
            return np.array(embeddings, dtype=np.float32)
    """
    pass
```

### LLM Function Signature

```python
async def llm_model_func(
    prompt: str | list[dict],
    **kwargs
) -> str:
    """
    LLM completion function

    Args:
        prompt: Either:
            - str: Single prompt string
            - list[dict]: Chat messages [{"role": "user", "content": "..."}]
        **kwargs: Optional parameters
            - temperature: float (default 0.0)
            - max_tokens: int (default 4096)
            - model: str (model name)

    Returns:
        Completion text as string

    Example:
        async def my_llm_func(prompt: str, **kwargs) -> str:
            response = await client.chat.completions.create(
                model=kwargs.get("model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.0)
            )
            return response.choices[0].message.content
    """
    pass
```

---

## 7. Performance Analysis

### Extension Overhead

| Extension Type | Overhead | Notes |
|----------------|----------|-------|
| Storage Backend | < 5ms | Async I/O hides latency |
| Tool Execution | 10-100ms | Depends on tool complexity |
| Custom Embedding | Variable | Depends on model size |
| Custom LLM | Variable | Depends on provider/model |
| Custom Reward | < 10ms | Pure Python computation |

### Storage Backend Comparison

| Backend | Write Throughput | Read Latency | Scalability | Setup Complexity |
|---------|-----------------|--------------|-------------|------------------|
| NetworkX | 10K/s | 0.1ms | 100K nodes | Low |
| Neo4J | 5K/s | 2ms | 10M nodes | Medium |
| PostgreSQL | 8K/s | 3ms | 100M nodes | Medium |
| MongoDB | 20K/s | 3ms | Unlimited | Medium |
| Oracle | 15K/s | 4ms | Unlimited | High |

### Tool Execution Benchmarks

```python
# Benchmark tool execution
import time
import asyncio

async def benchmark_tool(tool: ToolBase, queries: list[str]):
    """Benchmark tool performance"""

    # Warmup
    for _ in range(10):
        await tool.execute(queries[0])

    # Benchmark
    start = time.time()
    results = []

    for query in queries:
        result = await tool.execute(query)
        results.append(result)

    end = time.time()

    # Report
    avg_latency = (end - start) / len(queries)
    throughput = len(queries) / (end - start)

    print(f"Tool: {tool.name}")
    print(f"  Avg Latency: {avg_latency*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} QPS")

    return results

# Run benchmark
queries = ["test query"] * 100
results = await benchmark_tool(SearchTool(), queries)
```

### Memory Usage

```python
import tracemalloc
import gc

def profile_extension_memory(rag: BiGRAG, documents: list[dict]):
    """Profile memory usage of custom extensions"""

    # Start profiling
    tracemalloc.start()
    gc.collect()

    # Baseline
    snapshot_before = tracemalloc.take_snapshot()

    # Insert documents (tests storage backend)
    await rag.ainsert(documents)

    # After insert
    snapshot_after = tracemalloc.take_snapshot()

    # Compute diff
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

    print("Memory usage by extension:")
    for stat in top_stats[:10]:
        print(stat)

    # Stop profiling
    tracemalloc.stop()
```

---

## 8. Testing Guide

### Testing Storage Backends

```python
import pytest
import asyncio
from bigrag.base import BaseVectorStorage

class TestStorageBackend:
    """Test suite for custom storage backends"""

    @pytest.fixture
    async def storage(self):
        """Create storage instance"""
        from custom_extensions.postgres_storage import PostgresVectorStorage

        storage = PostgresVectorStorage(
            namespace="test",
            working_dir="./test_data",
            connection_string="postgresql://test:test@localhost/test"
        )

        yield storage

        # Cleanup
        await storage._cleanup_test_data()

    @pytest.mark.asyncio
    async def test_upsert(self, storage: BaseVectorStorage):
        """Test data insertion"""
        data = {
            "test_1": {
                "content": "Test content",
                "__vector__": [0.1] * 1536,
                "metadata": {"source": "test"}
            }
        }

        await storage.upsert(data)

        # Verify
        result = await storage.query("Test", top_k=1)
        assert len(result) == 1
        assert result[0]["content"] == "Test content"

    @pytest.mark.asyncio
    async def test_query(self, storage: BaseVectorStorage):
        """Test query functionality"""
        # Insert test data
        test_data = {
            f"test_{i}": {
                "content": f"Content {i}",
                "__vector__": [i * 0.01] * 1536
            }
            for i in range(100)
        }
        await storage.upsert(test_data)

        # Query
        results = await storage.query("Content 50", top_k=10)

        # Verify
        assert len(results) == 10
        assert all("content" in r for r in results)
        assert all("similarity" in r for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_access(self, storage: BaseVectorStorage):
        """Test concurrent operations"""
        # Create concurrent tasks
        tasks = [
            storage.upsert({f"test_{i}": {"content": f"Content {i}", "__vector__": [i * 0.01] * 1536}})
            for i in range(100)
        ]

        # Execute concurrently
        await asyncio.gather(*tasks)

        # Verify all inserted
        results = await storage.query("Content", top_k=100)
        assert len(results) == 100
```

### Testing Tools

```python
import pytest
from agent.tool.base import ToolBase

class TestCustomTool:
    """Test suite for custom tools"""

    @pytest.fixture
    def tool(self):
        """Create tool instance"""
        from custom_extensions.sql_tool import SQLTool

        return SQLTool({
            "connection_string": "sqlite:///:memory:",
            "allowed_operations": ["SELECT"]
        })

    def test_tool_properties(self, tool: ToolBase):
        """Test tool metadata"""
        assert tool.name == "sql"
        assert len(tool.description) > 0
        assert "SQL" in tool.description

    def test_execute_valid_query(self, tool: ToolBase):
        """Test valid SQL execution"""
        # Setup test table
        tool.engine.execute("CREATE TABLE users (id INT, name TEXT)")
        tool.engine.execute("INSERT INTO users VALUES (1, 'Alice')")

        # Execute query
        result = tool.execute("SELECT * FROM users")

        # Verify
        assert "Alice" in result
        assert "users" in result.lower()

    def test_execute_invalid_operation(self, tool: ToolBase):
        """Test security restrictions"""
        result = tool.execute("DROP TABLE users")

        assert "Error" in result
        assert "allowed" in result.lower()

    def test_execute_error_handling(self, tool: ToolBase):
        """Test error cases"""
        result = tool.execute("SELECT * FROM nonexistent_table")

        assert "Error" in result
```

### Testing Custom Extractors

```python
import pytest
from bigrag import BiGRAG
from bigrag.operate import extract_entities

class TestCustomExtractor:
    """Test suite for custom entity extraction"""

    @pytest.fixture
    def rag(self):
        """Create BiGRAG with custom extraction"""
        from custom_extensions.legal_extraction import (
            LEGAL_ENTITY_TYPES,
            LEGAL_EXTRACTION_PROMPT
        )

        return BiGRAG(
            working_dir="./test_legal",
            entity_types=LEGAL_ENTITY_TYPES,
            entity_relationship_prompt=LEGAL_EXTRACTION_PROMPT
        )

    @pytest.mark.asyncio
    async def test_extract_case_entities(self, rag: BiGRAG):
        """Test case entity extraction"""
        text = "In Brown v. Board of Education, the Supreme Court ruled..."

        entities = await extract_entities([text], rag.entity_types, rag.entity_relationship_prompt)

        # Verify case extracted
        assert any(e[0] == "Brown v. Board of Education" and e[1] == "Case" for e in entities)

        # Verify court extracted
        assert any(e[0] == "Supreme Court" and e[1] == "Court" for e in entities)

    @pytest.mark.asyncio
    async def test_normalize_citations(self, rag: BiGRAG):
        """Test citation normalization"""
        from custom_extensions.legal_extraction import normalize_case_citation

        # Test various formats
        assert normalize_case_citation("Brown v Board") == "Brown v. Board"
        assert normalize_case_citation("Smith vs Jones") == "Smith v. Jones"
        assert normalize_case_citation("Doe versus Roe") == "Doe v. Roe"

    @pytest.mark.asyncio
    async def test_full_pipeline(self, rag: BiGRAG):
        """Test full extraction and graph building"""
        documents = [
            {
                "id": "case_001",
                "contents": "Brown v. Board of Education, 347 U.S. 483 (1954)..."
            }
        ]

        await rag.ainsert(documents)

        # Query
        context = await rag.aquery("Brown v. Board of Education", QueryParam(mode="hybrid"))

        assert len(context) > 0
        assert "Brown" in context
```

### Integration Testing

```python
import pytest
from bigrag import BiGRAG
from agent.tool.tool_env import ToolEnv

class TestExtensionIntegration:
    """Test integration between extensions"""

    @pytest.fixture
    async def full_system(self):
        """Setup full system with extensions"""
        # Custom storage
        rag = BiGRAG(
            working_dir="./test_integration",
            graph_storage="PostgresGraphStorage",
            vector_storage="PostgresVectorStorage"
        )

        # Insert test data
        documents = [
            {"id": f"doc_{i}", "contents": f"Test content {i}"}
            for i in range(100)
        ]
        await rag.ainsert(documents)

        # Custom tools
        tool_env = ToolEnv(
            env="custom",
            tool_config={
                "tools": {"search": SearchTool, "sql": SQLTool},
                "search": {"api_url": "http://localhost:8001"},
                "sql": {"connection_string": "sqlite:///test.db"}
            }
        )

        return rag, tool_env

    @pytest.mark.asyncio
    async def test_end_to_end(self, full_system):
        """Test complete retrieval pipeline"""
        rag, tool_env = full_system

        # Execute search tool
        search_result = tool_env.execute("search", "test query")

        # Verify results
        assert len(search_result) > 0

        # Query RAG directly
        context = await rag.aquery("test query")

        assert len(context) > 0
```

---

## Summary

**Key Takeaways:**
1. **Pluggable architecture** enables customization without modifying core code
2. **Abstract base classes** define clear extension interfaces
3. **Lazy loading** allows optional dependencies
4. **Tool system** is fully extensible for new retrieval sources
5. **LLM and embedding** functions are swappable
6. **Storage backends** can be replaced for different deployment scenarios
7. **Custom extractors** adapt to domain-specific needs
8. **Reward functions** customize RL training objectives

**Extension Points:**
- Storage: Graph, vector, KV backends
- Tools: Search, calculators, databases, APIs
- Extraction: Entity types, prompts, parsers
- Models: LLM providers, embedding models
- Rewards: Custom scoring functions

**Best Practices:**
- Always inherit from base classes
- Implement all abstract methods
- Use async/await for I/O operations
- Add comprehensive error handling
- Write tests for custom components
- Document configuration options
- Profile performance impact
