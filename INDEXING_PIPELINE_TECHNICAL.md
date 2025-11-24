# BiG-RAG Indexing Pipeline - Technical Documentation

**Target Audience**: Developers, AI coding assistants, contributors
**Purpose**: Technical reference for understanding and modifying the production indexing pipeline
**Last Updated**: November 24, 2025

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Phases](#pipeline-phases)
3. [Code Organization](#code-organization)
4. [Data Flow](#data-flow)
5. [Key Classes and Methods](#key-classes-and-methods)
6. [Configuration](#configuration)
7. [Storage Layer](#storage-layer)
8. [Validation System](#validation-system)
9. [Error Handling](#error-handling)
10. [Extension Points](#extension-points)

---

## Architecture Overview

### High-Level Flow

```
Raw Document → Chunking → Extraction → Entity Linking → Validation → Graph Building → Vector Indexing
```

### Core Components

| Component | Purpose | Primary File |
|-----------|---------|--------------|
| **Pipeline Orchestrator** | Coordinates all phases | `bigrag/production_pipeline.py` |
| **Chunker** | Splits documents into processable units | `bigrag/text_splitter.py`, `bigrag/table_extractor.py` |
| **Extractors** | Extract entities and relations | `bigrag/extractors/constrained_extractor.py` |
| **Entity Linker** | Merges duplicate entities | `bigrag/merging/entity_linker.py` |
| **Validators** | Quality checks | `bigrag/validators/numeric_validator.py`, `bigrag/validators/consistency_validator.py` |
| **Graph Builder** | Constructs bipartite graph | `bigrag/builders/bipartite_graph_builder.py` |
| **Storage Manager** | Persists graph and vectors | `bigrag/storage.py` |

---

## Pipeline Phases

### Phase 1: Pre-processing

**File**: `bigrag/production_pipeline.py` → `_chunk_document()`
**Purpose**: Split document into tables and paragraphs

#### Implementation

```python
# Entry point
async def _chunk_document(self, document: str, metadata: dict) -> List[dict]:
    """
    Smart chunking with table detection.

    Returns:
        List[dict]: [
            {
                'chunk_id': 'chunk_0000',
                'content_type': 'table' | 'paragraph',
                'content': '...',
                'metadata': {...}
            }
        ]
    """
    return await self.chunker.chunk_text_with_tables(document, metadata)
```

#### Key Logic

**Table Detection**: `bigrag/table_extractor.py`
```python
class GPT4TableExtractor:
    async def extract_tables(self, markdown: str) -> List[dict]:
        """
        Uses GPT-4o to detect and extract tables from markdown.

        Detection logic:
        - Looks for markdown table syntax (|---|---|)
        - Extracts headers and rows
        - Preserves column structure
        """
```

**Text Chunking**: `bigrag/text_splitter.py`
```python
class TableAwareChunker:
    def chunk_text_with_tables(self, text: str) -> List[dict]:
        """
        Splits text while keeping tables intact.

        Strategy:
        1. Extract all tables as complete units
        2. Split remaining paragraphs (max 1200 tokens, 100 overlap)
        3. Return combined list with content_type labels
        """
```

---

### Phase 2: Extraction

**File**: `bigrag/production_pipeline.py` → `_extract_from_chunks()`
**Purpose**: Extract entities and relations from each chunk

#### Entry Point

```python
async def _extract_from_chunks(self, chunks: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Extracts entities and relations from chunks.

    Returns:
        Tuple[List[entities], List[relations]]
    """
    all_entities = []
    all_relations = []

    for chunk in chunks:
        if chunk['content_type'] == 'table':
            # Table extraction (rule-based)
            entities, relations = await self._extract_from_table(chunk)
        else:
            # Paragraph extraction (LLM-based)
            entities, relations = await self._extract_from_paragraph(chunk)

        all_entities.extend(entities)
        all_relations.extend(relations)

    return all_entities, all_relations
```

#### Table Extraction (Rule-Based)

**File**: `bigrag/extractors/table_extractor.py`

```python
async def _extract_from_table(self, chunk: dict) -> Tuple[List[dict], List[dict]]:
    """
    Rule-based extraction for structured tables.

    Logic:
    1. Parse table structure (headers + rows)
    2. For each row:
       - Create entity for each cell value
       - Create relation connecting row values
    3. No LLM needed (deterministic)

    Example:
        Table: | Department | Code | Seats |
               | CSE        | CSE  | 120   |

        Entities: [
            {'entity_name': 'CSE', 'entity_type': 'department'},
            {'entity_name': '120', 'entity_type': 'number'}
        ]
        Relations: [
            {'subject': 'CSE', 'predicate': 'has_seats', 'object': '120'}
        ]
    """
    table_data = self.table_extractor.parse_table(chunk['content'])
    entities = []
    relations = []

    headers = table_data['headers']
    for row in table_data['rows']:
        for i, cell_value in enumerate(row):
            # Create entity for cell
            entities.append({
                'entity_name': cell_value,
                'entity_type': self._infer_type(headers[i]),
                'source_id': chunk['chunk_id'],
                'key_score': 100  # High confidence for tables
            })

        # Create relation connecting row values
        relations.append({
            'subject': row[0],
            'predicate': f"has_{headers[1]}",
            'object': row[1],
            'source_id': chunk['chunk_id'],
            'completeness_score': 10
        })

    return entities, relations
```

#### Paragraph Extraction (LLM-Based)

**File**: `bigrag/extractors/constrained_extractor.py`

```python
class ConstrainedLLMExtractor:
    async def extract(self, chunk: dict) -> dict:
        """
        LLM-based extraction with validation retry loop.

        Process:
        1. Build extraction prompt with examples
        2. Call LLM (GPT-4o-mini)
        3. Parse JSON response
        4. Validate extraction (numeric coverage, semantic validity)
        5. If validation fails, retry (max 3 attempts)
        6. If all attempts fail, return None (chunk rejected)

        Returns:
            {
                'entities': [...],
                'relations': [...],
                'validation': {
                    'status': 'PASS' | 'WARNING' | 'FAIL',
                    'numeric_coverage': 0.0-1.0,
                    'attempts': 1-3
                }
            }
        """
        for attempt in range(1, 4):
            # Step 1: Extract
            prompt = self._build_extraction_prompt(chunk)
            llm_response = await self._call_llm(prompt)
            extraction = json.loads(llm_response)

            # Step 2: Validate
            validation = self._validate_extraction(
                source_text=chunk['content'],
                extraction=extraction
            )

            # Step 3: Check status
            if validation['status'] in ['PASS', 'WARNING']:
                return extraction  # Accept

            if attempt == 3:
                return None  # Reject after 3 attempts

        return None
```

**Extraction Prompt Template**: `bigrag/prompt.py`

```python
EXTRACTION_PROMPT = """
Extract entities and relations from the following text.

TEXT:
{source_text}

INSTRUCTIONS:
1. Extract ALL named entities (persons, places, departments, dates, numbers)
2. Extract relations connecting entities (X relates_to Y)
3. Preserve ALL numbers exactly as they appear (including Bangla numerals)
4. Output valid JSON only

OUTPUT FORMAT:
{{
  "entities": [
    {{
      "entity_name": "CSE",
      "entity_type": "department",
      "key_score": 95
    }}
  ],
  "relations": [
    {{
      "subject": "CSE",
      "predicate": "has_seats",
      "object": "120",
      "completeness_score": 10
    }}
  ]
}}
"""
```

---

### Phase 3: Entity Linking

**File**: `bigrag/merging/entity_linker.py`
**Purpose**: Merge duplicate entities across chunks

#### Implementation

```python
class SimpleEntityLinker:
    async def link_entities_across_chunks(self, entities: List[dict]) -> List[dict]:
        """
        Multi-stage entity merging for multilingual documents.

        Stages:
        1. Canonicalization - Apply domain-specific rules (KUET → Khulna University)
        2. Exact alias matching - Group identical names
        3. Fuzzy matching - Merge similar names (edit distance < 0.15)
        4. (Optional) Embedding similarity - Merge semantically similar
        5. (Optional) LLM verification - Confirm uncertain merges

        Returns:
            List[dict]: Merged entities with updated metadata
        """
        # Stage 1: Canonicalization
        entities = self._apply_canonicalization(entities)

        # Stage 2: Exact alias grouping
        groups = self._group_by_exact_alias(entities)

        # Stage 3: Fuzzy matching
        groups = self._fuzzy_merge_groups(groups)

        # Stage 4: Create final merged entities
        merged_entities = []
        for group in groups:
            merged_entity = self._merge_entity_group(group)
            merged_entities.append(merged_entity)

        return merged_entities
```

#### Fuzzy Matching Logic

```python
def _fuzzy_merge_groups(self, groups: List[List[dict]]) -> List[List[dict]]:
    """
    Merge groups with similar entity names.

    Algorithm:
    - Compare all group pairs
    - If similarity > 0.85, merge groups
    - Uses SequenceMatcher for string comparison

    Handles:
    - Typos: "Computer Science" vs "Computer Sceince"
    - Abbreviations: "CSE" vs "CS Engineering" (similarity > 0.85)
    - Transliterations: Partial matches across scripts
    """
    from difflib import SequenceMatcher

    merged = False
    while not merged:
        merged = True
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                # Compare canonical names
                name1 = self._get_canonical_name(groups[i])
                name2 = self._get_canonical_name(groups[j])

                similarity = SequenceMatcher(None, name1, name2).ratio()

                if similarity > 0.85:
                    # Merge group j into group i
                    groups[i].extend(groups[j])
                    groups.pop(j)
                    merged = False
                    break
            if not merged:
                break

    return groups
```

#### Merge Strategy

```python
def _merge_entity_group(self, entities: List[dict]) -> dict:
    """
    Combine multiple entity instances into one merged entity.

    Merge strategy:
    - Name: Use most common variant
    - Type: Use majority vote
    - Weight: Sum all key_scores (reflects frequency)
    - Source IDs: Combine all source chunks
    - Description: Concatenate all descriptions

    Example:
        Input: [
            {'entity_name': 'CSE', 'key_score': 90, 'source_id': 'chunk_0'},
            {'entity_name': 'Computer Science', 'key_score': 95, 'source_id': 'chunk_1'}
        ]

        Output: {
            'entity_name': 'Computer Science',  # Most common
            'entity_type': 'department',
            'weight': 185.0,  # 90 + 95
            'source_id': 'chunk_0,chunk_1',
            'aliases': ['CSE', 'Computer Science']
        }
    """
    # Choose canonical name (most frequent)
    name_counts = Counter([e['entity_name'] for e in entities])
    canonical_name = name_counts.most_common(1)[0][0]

    # Sum weights
    total_weight = sum(e.get('key_score', 0) for e in entities)

    # Combine source IDs
    source_ids = ','.join(set(e['source_id'] for e in entities))

    return {
        'entity_name': canonical_name,
        'entity_type': entities[0]['entity_type'],
        'weight': float(total_weight),
        'source_id': source_ids,
        'aliases': list(set(e['entity_name'] for e in entities))
    }
```

---

### Phase 4: Validation

**File**: `bigrag/production_pipeline.py` → `_validate_extraction()`
**Purpose**: Quality checks before graph building

#### Two Validation Types

##### 4.1 Numeric Validation (Gemini 2.5 Pro)

**File**: `bigrag/validators/numeric_validator.py`

```python
class NumericValidator:
    def __init__(self, api_key: str = None):
        """
        Initialize with Gemini 2.5 Pro for multilingual numeric validation.

        Uses google-genai SDK (not google-generativeai).
        Reads GEMINI_API_KEY from .env file.
        """
        from google import genai
        from dotenv import load_dotenv
        import os

        load_dotenv()
        self.gemini_api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=self.gemini_api_key)
        self.model_name = 'gemini-2.0-flash-exp'

    async def validate_extraction(
        self,
        source_document: str,
        entities: List[dict],
        relations: List[dict],
        validation_level: str = "MODERATE"
    ) -> dict:
        """
        Validate numeric accuracy using Gemini as judge.

        Process:
        1. Build KG text from entities and relations
        2. Ask Gemini to compare source vs KG
        3. Gemini returns coverage percentage and missing numbers
        4. Map coverage to PASS/WARNING/FAIL based on level

        Thresholds:
        - STRICT: 95%+ PASS, 90-95% WARNING, <90% FAIL
        - MODERATE: 90%+ PASS, 85-90% WARNING, <85% FAIL
        - LENIENT: 80%+ PASS, 75-80% WARNING, <75% FAIL

        Returns:
            {
                'status': 'PASS' | 'WARNING' | 'FAIL',
                'numeric_coverage': 0.0-1.0,
                'missing_numbers': [...],
                'hallucinated_numbers': [...],
                'gemini_feedback': '...'
            }
        """
        kg_text = self._build_kg_text(entities, relations)

        prompt = f"""Compare numbers in SOURCE vs EXTRACTED KG.

SOURCE:
{source_document}

EXTRACTED KG:
{kg_text}

Treat Bangla and English numbers as SAME (১২০ = 120).
Output JSON: {{"coverage_percent": ..., "missing_numbers": [...]}}
"""

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        result = json.loads(response.text)
        coverage = result['coverage_percent'] / 100.0

        # Determine status based on threshold
        if validation_level == "MODERATE":
            if coverage >= 0.90:
                status = "PASS"
            elif coverage >= 0.85:
                status = "WARNING"
            else:
                status = "FAIL"

        return {
            'status': status,
            'numeric_coverage': coverage,
            'missing_numbers': result.get('missing_numbers', []),
            'hallucinated_numbers': result.get('hallucinated_numbers', [])
        }
```

##### 4.2 Consistency Validation (Non-Blocking)

**File**: `bigrag/validators/consistency_validator.py`

```python
class ConsistencyValidator:
    def validate_consistency(
        self,
        entities: List[dict],
        relations: List[dict],
        validation_level: str = "STRICT"
    ) -> dict:
        """
        Detect cross-chunk naming conflicts.

        Checks:
        1. Entity conflicts - Same name, different attributes
        2. Numeric conflicts - Same subject, different numbers
        3. Relation contradictions - Conflicting facts
        4. Reference integrity - Relations reference existing entities

        NOTE: For multilingual documents, this validator generates
        many false positives (e.g., "CSE" vs "Computer Science").

        Therefore, consistency validation is NON-BLOCKING.
        Status FAIL → Overall status WARNING (not FAIL).

        Returns:
            {
                'status': 'PASS' | 'FAIL',
                'consistency_score': 0.0-1.0,
                'total_issues': int,
                'entity_conflicts': [...],
                'numeric_conflicts': [...],
                'relation_contradictions': [...]
            }
        """
        # Build entity registry
        entity_registry = self._build_entity_registry(entities)

        # Run checks
        entity_conflicts = self._check_entity_consistency(entity_registry)
        numeric_conflicts = self._check_numeric_consistency(entities, relations)
        relation_contradictions = self._check_relation_contradictions(relations)

        # Calculate score
        total_issues = len(entity_conflicts) + len(numeric_conflicts) + len(relation_contradictions)
        total_checks = len(entities) + len(relations)
        consistency_score = 1.0 - (total_issues / max(total_checks, 1))

        status = "PASS" if consistency_score >= 0.8 else "FAIL"

        return {
            'status': status,
            'consistency_score': consistency_score,
            'total_issues': total_issues,
            'entity_conflicts': entity_conflicts,
            'numeric_conflicts': numeric_conflicts,
            'relation_contradictions': relation_contradictions
        }
```

**Consistency Handling in Pipeline**: `bigrag/production_pipeline.py`

```python
# Line 310-320
if numeric_status == 'FAIL':
    # Only block on numeric validation failure
    overall_status = 'FAIL'
elif consistency_status == 'FAIL':
    # Consistency failure -> WARNING (not blocking)
    # Reason: Entity linking already handles multilingual merging
    overall_status = 'WARNING'
    print("[WARNING] Consistency validation failed but not blocking pipeline")
else:
    overall_status = 'PASS'
```

---

### Phase 5: Graph Building

**File**: `bigrag/builders/bipartite_graph_builder.py`
**Purpose**: Construct bipartite graph from validated entities and relations

#### Graph Structure

```
Bipartite Graph:
- Partition 1: Document chunks (text nodes)
- Partition 2: Semantic nodes (entities + relations)
- Edges: Connect chunks to entities/relations they contain

Node types:
- role="chunk": Text chunk node
- role="entity": Entity node
- role="relation": Relation node (yes, relations are nodes!)
```

#### Implementation

```python
class BipartiteGraphBuilder:
    async def build_graph(
        self,
        entities: List[dict],
        relations: List[dict],
        chunks: List[dict]
    ) -> dict:
        """
        Build bipartite graph with three node types.

        Process:
        1. Create entity nodes (V_E)
        2. Create relation nodes (V_R)
        3. Create bipartite edges:
           - chunk → entity (if entity mentioned in chunk)
           - chunk → relation (if relation extracted from chunk)

        Returns:
            {
                'graph': NetworkX graph object,
                'entity_count': int,
                'relation_count': int,
                'edge_count': int,
                'orphan_relations': int
            }
        """
        # Step 1: Create entity nodes
        entity_nodes_created = await self._create_entity_nodes(entities)

        # Step 2: Create relation nodes
        relation_nodes_created = await self._create_relation_nodes(relations)

        # Step 3: Create bipartite edges
        edges_created, orphan_count = await self._create_relations(
            entities, relations, chunks
        )

        return {
            'graph': self.graph,
            'entity_count': entity_nodes_created,
            'relation_count': relation_nodes_created,
            'edge_count': edges_created,
            'orphan_relations': orphan_count
        }
```

#### Node Creation

```python
async def _create_entity_nodes(self, entities: List[dict]) -> int:
    """
    Create entity nodes with full attributes.

    Node attributes:
    - role: 'entity' (partition label)
    - entity_type: 'person' | 'place' | 'concept' | etc.
    - description: Text description
    - weight: Aggregated importance score
    - source_id: Comma-separated chunk IDs
    - extraction_quality: 'PASS' | 'WARNING' | 'FAIL'

    Node ID: The entity name itself (e.g., "CSE", "Computer Science")

    Note: Entity name is stored as node ID, NOT as 'name' attribute.
    This is standard NetworkX practice.
    """
    for entity in entities:
        entity_name = entity['entity_name']

        node_data = {
            'role': 'entity',
            'entity_type': entity.get('entity_type', 'concept'),
            'description': entity.get('description', ''),
            'weight': float(entity.get('weight', 0)),
            'source_id': entity.get('source_id', ''),
            'extraction_quality': entity.get('extraction_quality', 'PASS')
        }

        await self.graph.upsert_node(entity_name, node_data=node_data)

    return len(entities)
```

#### Edge Creation Logic

```python
async def _create_relations(
    self,
    entities: List[dict],
    relations: List[dict],
    chunks: List[dict]
) -> Tuple[int, int]:
    """
    Create edges connecting chunks to entities/relations.

    Edge creation logic:
    1. For each entity:
       - Get source_id (which chunk it came from)
       - Create edge: chunk_node → entity_node

    2. For each relation:
       - Get source_id
       - Create edge: chunk_node → relation_node
       - Also create edges to linked entities (if they exist)

    Orphan detection:
    - Relation with no linked entities = orphan
    - Orphan relations are still added to graph (may contain useful info)

    Returns:
        Tuple[total_edges, orphan_count]
    """
    edges_created = 0
    orphan_count = 0

    # Create chunk → entity edges
    for entity in entities:
        chunk_id = entity['source_id']
        entity_name = entity['entity_name']

        await self.graph.add_edge(chunk_id, entity_name)
        edges_created += 1

    # Create chunk → relation edges
    for relation in relations:
        chunk_id = relation['source_id']
        relation_name = relation.get('name', f"{relation['subject']}_{relation['predicate']}_{relation['object']}")

        # Edge: chunk → relation
        await self.graph.add_edge(chunk_id, relation_name)
        edges_created += 1

        # Edge: relation → linked entities
        linked_entities = relation.get('linked_entities', [])
        has_valid_links = False

        for entity_name in linked_entities:
            if await self.graph.has_node(entity_name):
                await self.graph.add_edge(relation_name, entity_name)
                edges_created += 1
                has_valid_links = True

        if not has_valid_links:
            orphan_count += 1

    return edges_created, orphan_count
```

---

### Phase 6: Vector Indexing

**File**: `bigrag/bigrag.py` → `_index_vectors()`
**Purpose**: Generate embeddings for three-path retrieval

#### Three Retrieval Paths

```
Path A (Entity-based): Query → Entities → Connected chunks
Path B (Relation-based): Query → Relations → Connected chunks
Path C (Chunk-based): Query → Chunks directly
```

#### Implementation

```python
async def _index_vectors(self, entities: List[dict], relations: List[dict], chunks: List[dict]):
    """
    Generate embeddings for all three paths.

    Process:
    1. Path A: Embed entity names + descriptions
    2. Path B: Embed relation triplets (subject-predicate-object)
    3. Path C: Embed text chunks

    Embedding model: FlagEmbedding (bge-large-en-v1.5)
    Dimension: 1024 or 3072 (depends on model)

    Storage:
    - vdb_entities.json (Path A)
    - vdb_relations.json (Path B)
    - vdb_chunks.json (Path C)
    """
    # Path A: Entity embeddings
    entity_texts = [
        f"{e['entity_name']}: {e.get('description', '')}"
        for e in entities
    ]
    entity_embeddings = await self.embedding_model.embed(entity_texts)
    await self.entities_vdb.upsert(entity_texts, entity_embeddings)

    # Path B: Relation embeddings
    relation_texts = [
        f"{r['subject']} {r['predicate']} {r['object']}"
        for r in relations
    ]
    relation_embeddings = await self.embedding_model.embed(relation_texts)
    await self.relations_vdb.upsert(relation_texts, relation_embeddings)

    # Path C: Chunk embeddings
    chunk_texts = [c['content'] for c in chunks]
    chunk_embeddings = await self.embedding_model.embed(chunk_texts)
    await self.chunks_vdb.upsert(chunk_texts, chunk_embeddings)
```

---

## Code Organization

### Directory Structure

```
bigrag/
├── bigrag.py                    # Main BiGRAG class (entry point)
├── production_pipeline.py       # Pipeline orchestrator
│
├── extractors/
│   ├── constrained_extractor.py # LLM-based paragraph extraction
│   └── table_extractor.py       # Rule-based table extraction
│
├── merging/
│   ├── entity_linker.py         # Entity linking and merging
│   └── canonicalization.py      # Domain-specific canonicalization rules
│
├── validators/
│   ├── numeric_validator.py     # Gemini-based numeric validation
│   └── consistency_validator.py # Cross-chunk consistency checks
│
├── builders/
│   └── bipartite_graph_builder.py # Graph construction
│
├── storage.py                   # Storage abstraction layer
├── base.py                      # Abstract base classes
├── text_splitter.py             # Chunking logic
├── table_extractor.py           # Table detection
├── prompt.py                    # LLM prompt templates
├── bangla_utils.py              # Bangla numeral normalization
└── utils.py                     # Utility functions
```

---

## Data Flow

### Input Format

```python
# Document with metadata
document = {
    'content': '...',  # Raw text (markdown supported)
    'metadata': {
        'title': 'KUET Admission 2024-25',
        'category': 'education',
        'language': 'bangla',
        'tags': ['admission', 'university']
    }
}
```

### Intermediate Representations

#### After Chunking
```python
chunks = [
    {
        'chunk_id': 'chunk_0000',
        'content_type': 'table',
        'content': '| Department | Seats |\n|------------|-------|\n| CSE | 120 |',
        'metadata': {'title': 'KUET Admission 2024-25'}
    },
    {
        'chunk_id': 'chunk_0001',
        'content_type': 'paragraph',
        'content': 'CSE department requires minimum GPA 4.50...',
        'metadata': {'title': 'KUET Admission 2024-25'}
    }
]
```

#### After Extraction
```python
entities = [
    {
        'entity_name': 'CSE',
        'entity_type': 'department',
        'key_score': 95,
        'source_id': 'chunk_0000',
        'description': 'Computer Science and Engineering department'
    },
    {
        'entity_name': '120',
        'entity_type': 'number',
        'key_score': 100,
        'source_id': 'chunk_0000',
        'description': 'Number of seats in CSE'
    }
]

relations = [
    {
        'subject': 'CSE',
        'predicate': 'has_seats',
        'object': '120',
        'source_id': 'chunk_0000',
        'completeness_score': 10,
        'linked_entities': ['CSE', '120']
    }
]
```

#### After Entity Linking
```python
merged_entities = [
    {
        'entity_name': 'Computer Science and Engineering',  # Canonical name
        'entity_type': 'department',
        'weight': 285.0,  # Sum of all key_scores (95 + 95 + 95)
        'source_id': 'chunk_0000,chunk_0001,chunk_0002',
        'aliases': ['CSE', 'Computer Science', 'Computer Science and Engineering']
    }
]
```

### Output Format

#### GraphML File Structure
```xml
<graphml>
  <key id="d0" for="node" attr.name="role" />
  <key id="d1" for="node" attr.name="entity_type" />
  <key id="d2" for="node" attr.name="description" />
  <key id="d3" for="node" attr.name="weight" />

  <!-- Entity node -->
  <node id="&quot;Computer Science and Engineering&quot;">
    <data key="d0">entity</data>
    <data key="d1">department</data>
    <data key="d2">Computer Science and Engineering department</data>
    <data key="d3">285.0</data>
  </node>

  <!-- Relation node -->
  <node id="&quot;CSE_has_seats_120&quot;">
    <data key="d0">relation</data>
    <data key="d2">CSE has 120 seats</data>
    <data key="d3">10.0</data>
  </node>

  <!-- Chunk node -->
  <node id="chunk_0000">
    <data key="d0">chunk</data>
  </node>

  <!-- Edges -->
  <edge source="chunk_0000" target="&quot;Computer Science and Engineering&quot;" />
  <edge source="chunk_0000" target="&quot;CSE_has_seats_120&quot;" />
</graphml>
```

---

## Key Classes and Methods

### BiGRAG (Main Entry Point)

**File**: `bigrag/bigrag.py`

```python
class BiGRAG:
    async def ainsert(
        self,
        documents: List[str],
        metadata: List[dict] = None,
        use_production_pipeline: bool = True
    ):
        """
        Main insertion method.

        Args:
            documents: List of document texts
            metadata: List of metadata dicts (one per document)
            use_production_pipeline: If True, use ProductionKGPipeline

        Process:
        1. For each document:
           a. Process with production pipeline OR standard pipeline
           b. Get entities, relations, chunks
           c. Build graph
           d. Index vectors
        2. Save to storage (GraphML, vector DBs, KV stores)
        """
```

### ProductionKGPipeline

**File**: `bigrag/production_pipeline.py`

```python
class ProductionKGPipeline:
    async def process_document(self, document: str, metadata: dict) -> dict:
        """
        Full pipeline execution.

        Returns:
            {
                'entities': List[dict],
                'relations': List[dict],
                'chunks': List[dict],
                'validation': {
                    'overall_status': 'PASS' | 'WARNING' | 'FAIL',
                    'numeric': {...},
                    'consistency': {...}
                },
                'metadata': {
                    'extraction_mode': 'semi_structured',
                    'entity_linking_enabled': True,
                    'total_chunks': 7,
                    'successful_chunks': 7
                }
            }
        """
```

---

## Configuration

### Pipeline Configuration

**File**: `bigrag/production_pipeline.py` → `__init__()`

```python
def __init__(
    self,
    api_key: str,
    model: str = "gpt-4o-mini",
    validation_level: str = "MODERATE",
    enable_entity_linking: bool = True,
    extraction_mode: str = "semi_structured"
):
    """
    Args:
        api_key: OpenAI API key (for extraction)
        model: LLM model for paragraph extraction
        validation_level: "STRICT" | "MODERATE" | "LENIENT"
        enable_entity_linking: Whether to merge duplicate entities
        extraction_mode: "structured" | "semi_structured" | "unstructured"
    """
```

### Validation Thresholds

**File**: `bigrag/validators/numeric_validator.py`

```python
# Numeric validation thresholds
THRESHOLDS = {
    "STRICT": {
        "PASS": 0.95,
        "WARNING": 0.90,
        "FAIL": 0.00
    },
    "MODERATE": {
        "PASS": 0.90,
        "WARNING": 0.85,
        "FAIL": 0.00
    },
    "LENIENT": {
        "PASS": 0.80,
        "WARNING": 0.75,
        "FAIL": 0.00
    }
}
```

**File**: `bigrag/extractors/constrained_extractor.py`

```python
# Per-chunk extraction thresholds
EXTRACTION_THRESHOLDS = {
    "structured": {
        "PASS": {"numeric": 1.00, "hallucination": 0.00, "semantic": 0.90},
        "WARNING": {"numeric": 0.95, "hallucination": 0.05, "semantic": 0.85}
    },
    "semi_structured": {
        "PASS": {"numeric": 0.95, "hallucination": 0.05, "semantic": 0.85},
        "WARNING": {"numeric": 0.60, "hallucination": 0.15, "semantic": 0.70}  # Lowered for paragraphs
    },
    "unstructured": {
        "PASS": {"numeric": 0.80, "hallucination": 0.15, "semantic": 0.70},
        "WARNING": {"numeric": 0.70, "hallucination": 0.20, "semantic": 0.60}
    }
}
```

---

## Storage Layer

### Storage Abstraction

**File**: `bigrag/base.py`

```python
class BaseGraphStorage:
    """Abstract base class for graph storage."""
    async def upsert_node(self, node_id: str, node_data: dict): ...
    async def add_edge(self, source: str, target: str): ...
    async def has_node(self, node_id: str) -> bool: ...
```

### Default Implementation (NetworkX)

**File**: `bigrag/storage.py`

```python
class NetworkXStorage(BaseGraphStorage):
    def __init__(self):
        self._graph = nx.Graph()

    async def upsert_node(self, node_id: str, node_data: dict):
        """
        Add or update node.

        NetworkX behavior:
        - If node exists: Updates/adds attributes (doesn't replace)
        - If node doesn't exist: Creates new node
        """
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id].update(node_data)
        else:
            self._graph.add_node(node_id, **node_data)

    async def save(self, filepath: str):
        """Save graph to GraphML format."""
        nx.write_graphml(self._graph, filepath)
```

### Vector Storage (NanoVectorDB)

**File**: `bigrag/storage.py`

```python
class NanoVectorDBStorage(BaseVectorStorage):
    async def upsert(self, ids: List[str], vectors: List[List[float]]):
        """
        Insert vectors into index.

        Format:
        {
            "id": "CSE: Computer Science and Engineering",
            "vector": [0.123, 0.456, ...],  # 1024 or 3072 dimensions
            "metadata": {...}
        }
        """
        for id, vector in zip(ids, vectors):
            await self._index.upsert(id, vector)

    async def search(self, query_vector: List[float], top_k: int = 10):
        """
        Similarity search.

        Returns:
            List[Tuple[id, score]]: Top-k results with cosine similarity scores
        """
        return await self._index.search(query_vector, top_k)
```

---

## Validation System

### Per-Chunk Validation (During Extraction)

**File**: `bigrag/extractors/constrained_extractor.py` → `_validate_extraction()`

```python
def _validate_extraction(
    self,
    source_text: str,
    source_numbers: set,
    source_facts: List[str],
    extraction: dict
) -> dict:
    """
    Triple-constraint validation for each chunk.

    Checks:
    1. Numeric coverage: All numbers from source in extraction
    2. No hallucination: No numbers in extraction not in source
    3. Semantic validity: Entity names mentioned in source

    Uses:
    - Regex for number extraction (fast, deterministic)
    - BanglaNumeralNormalizer for Bangla ↔ English conversion
    - Fuzzy string matching for entity validation

    Returns:
        {
            'status': 'PASS' | 'WARNING' | 'FAIL',
            'numeric_coverage': 0.0-1.0,
            'hallucination_score': 0.0-1.0,
            'semantic_validity': 0.0-1.0,
            'missing_numbers': List[str],
            'hallucinated_numbers': List[str],
            'hallucinated_entities': List[str]
        }
    """
```

### Overall Validation (After Extraction)

**File**: `bigrag/production_pipeline.py` → `_validate_extraction()`

```python
async def _validate_extraction(
    self,
    source_document: str,
    all_entities: List[dict],
    all_relations: List[dict]
) -> dict:
    """
    Overall quality validation.

    Two validators:
    1. Numeric Validator (Gemini) - LLM-based semantic comparison
    2. Consistency Validator - Rule-based conflict detection

    Status mapping:
    - Numeric FAIL → Overall FAIL (blocks pipeline)
    - Consistency FAIL → Overall WARNING (non-blocking)

    Returns:
        {
            'overall_status': 'PASS' | 'WARNING' | 'FAIL',
            'numeric': {
                'status': '...',
                'numeric_coverage': ...,
                'missing_numbers': [...]
            },
            'consistency': {
                'status': '...',
                'consistency_score': ...,
                'total_issues': ...
            }
        }
    """
```

---

## Error Handling

### Retry Logic

**File**: `bigrag/extractors/constrained_extractor.py`

```python
async def extract(self, chunk: dict) -> dict:
    """
    Extraction with validation retry loop.

    Retry strategy:
    - Max 3 attempts per chunk
    - If validation fails, retry extraction
    - If all attempts fail, return None (chunk rejected)

    Why retry?
    - LLM may miss numbers in first attempt
    - Different random seed may improve extraction
    - Validation guides LLM to focus on missed elements
    """
    for attempt in range(1, 4):
        extraction = await self._extract_with_llm(chunk)
        validation = self._validate_extraction(chunk, extraction)

        if validation['status'] in ['PASS', 'WARNING']:
            return extraction

        if attempt < 3:
            print(f"[RETRY] Attempt {attempt} failed, retrying...")

    return None  # All attempts failed
```

### Graceful Degradation

**File**: `bigrag/production_pipeline.py`

```python
async def _extract_from_chunks(self, chunks: List[dict]) -> Tuple[List, List]:
    """
    Graceful degradation: Skip failed chunks, continue with successful ones.

    Old behavior: Reject entire document if one chunk fails
    New behavior: Process all chunks, skip failures

    Success tracking:
    - Count successful vs total chunks
    - Log success rate (e.g., "6/7 chunks extracted successfully")
    - If >50% chunks fail, log warning but still proceed
    """
    successful_entities = []
    successful_relations = []
    failed_chunks = []

    for chunk in chunks:
        try:
            entities, relations = await self._extract_from_chunk(chunk)
            if entities is not None:
                successful_entities.extend(entities)
                successful_relations.extend(relations)
            else:
                failed_chunks.append(chunk['chunk_id'])
        except Exception as e:
            print(f"[ERROR] Chunk {chunk['chunk_id']} failed: {e}")
            failed_chunks.append(chunk['chunk_id'])

    success_rate = (len(chunks) - len(failed_chunks)) / len(chunks)
    print(f"Success rate: {success_rate:.1%} ({len(chunks)-len(failed_chunks)}/{len(chunks)})")

    return successful_entities, successful_relations
```

---

## Extension Points

### Adding New Extraction Modes

**File**: `bigrag/extractors/constrained_extractor.py`

```python
# To add new extraction mode (e.g., "highly_unstructured"):
# 1. Add thresholds in _determine_validation_status()

elif self.extraction_mode == "highly_unstructured":
    if (numeric_coverage >= 0.60 and
        hallucination_score < 0.25 and
        semantic_validity >= 0.50):
        return 'PASS'
    elif (numeric_coverage >= 0.50 and
          hallucination_score < 0.30 and
          semantic_validity >= 0.40):
        return 'WARNING'
    else:
        return 'FAIL'

# 2. Update ProductionKGPipeline.__init__() to accept new mode
# 3. Document in CLAUDE.md under extraction modes
```

### Adding New Validators

**File**: Create `bigrag/validators/your_validator.py`

```python
class YourCustomValidator:
    def validate(self, entities: List[dict], relations: List[dict]) -> dict:
        """
        Custom validation logic.

        Returns:
            {
                'status': 'PASS' | 'WARNING' | 'FAIL',
                'score': 0.0-1.0,
                'issues': List[str],
                'recommendations': List[str]
            }
        """
        # Your validation logic here
        pass

# Register in production_pipeline.py:
self.your_validator = YourCustomValidator()

# Call in _validate_extraction():
your_result = await self.your_validator.validate(entities, relations)
```

### Adding New Storage Backends

**File**: Create `bigrag/storage_backends/your_backend.py`

```python
from bigrag.base import BaseGraphStorage

class YourGraphStorage(BaseGraphStorage):
    async def upsert_node(self, node_id: str, node_data: dict):
        # Implement for your storage (Neo4j, MongoDB, etc.)
        pass

    async def add_edge(self, source: str, target: str):
        # Implement edge creation
        pass

    async def save(self, filepath: str):
        # Implement persistence
        pass

# Use in BiGRAG:
from bigrag.storage_backends.your_backend import YourGraphStorage

rag = BiGRAG(
    graph_storage=YourGraphStorage(),
    # ... other params
)
```

---

## Common Issues and Solutions

### Issue 1: Entity Names Show "N/A" in Test Output

**Symptom**: Test script displays `N/A` instead of entity names
**Root Cause**: Entity names stored as node IDs (not as 'name' attribute)
**Solution**: Read from `node_id` instead of `node_data.get('name')`

```python
# WRONG:
name = entity_data.get('name', 'N/A')

# CORRECT:
name = entity_id.strip('"')  # Remove GraphML quotes
```

### Issue 2: Consistency Validation Always Fails for Multilingual Docs

**Symptom**: 200+ consistency issues for Bangla+English documents
**Root Cause**: Validator sees "CSE" and "Computer Science" as different entities
**Solution**: Make consistency validation non-blocking (already implemented)

```python
# In production_pipeline.py
if consistency_status == 'FAIL':
    overall_status = 'WARNING'  # Not 'FAIL'
```

### Issue 3: Per-Chunk Extraction Fails at 65% Coverage

**Symptom**: Paragraphs rejected during extraction
**Root Cause**: Per-chunk threshold too strict (90%)
**Solution**: Lower WARNING threshold to 60% for semi_structured mode (already implemented)

### Issue 4: Gemini API Key Error

**Symptom**: "400 API key not valid"
**Root Cause**:
1. Wrong SDK (`google-generativeai` instead of `google-genai`)
2. API key has quotes in `.env` file
3. `.env` not loaded

**Solution**:
```python
# 1. Use correct SDK
from google import genai  # Not: import google.generativeai

# 2. Remove quotes in .env
GEMINI_API_KEY=AIzaSy...  # Not: GEMINI_API_KEY="AIzaSy..."

# 3. Load .env explicitly
from dotenv import load_dotenv
load_dotenv()
```

---

## Performance Considerations

### Chunking Performance

- **Table extraction**: O(n) where n = document length
- **Text chunking**: O(n) with overlapping windows
- **Optimization**: Cache table detection results

### Extraction Performance

- **Table extraction**: Fast (rule-based, no LLM)
- **Paragraph extraction**: Slow (LLM call per chunk)
- **Optimization**: Batch multiple chunks in one LLM call (not yet implemented)

### Entity Linking Performance

- **Fuzzy matching**: O(n²) where n = number of entity groups
- **Optimization**: Use embedding similarity instead of string comparison

### Vector Indexing Performance

- **Embedding generation**: O(n) where n = number of texts
- **Batch size**: 100 texts per batch (default)
- **Optimization**: Use GPU for embedding generation

---

## API Reference

### Main Methods

#### BiGRAG.ainsert()
```python
await rag.ainsert(
    documents=["Document text..."],
    metadata=[{"title": "...", "language": "bangla"}],
    use_production_pipeline=True
)
```

#### BiGRAG.aquery()
```python
results = await rag.aquery(
    query="CSE তে কত আসন আছে?",
    param=QueryParam(
        mode="hybrid",  # "local" | "global" | "hybrid" | "naive"
        top_k=10,
        enable_reranking=True
    )
)
```

#### BiGRAG.adelete_document()
```python
await rag.adelete_document("doc-abc123")
```

---

## Testing

### Unit Tests

**Location**: `test_scripts/`

```bash
# Test numeric validation
cd test_scripts
python test_numeric_validator.py

# Test entity linking
python test_entity_linker.py

# Test full pipeline
python test_bangla_production_diagnosis.py
```

### Integration Tests

```bash
# Build small test corpus
python script_build.py --data_source SingleTopic

# Query test
cd backend
python server.py --data_source SingleTopic
curl -X POST http://localhost:8001/search \
  -d '{"queries": ["test query"]}'
```

---

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Intermediate Results

```python
# In production_pipeline.py, add print statements:
print(f"[DEBUG] Chunks created: {len(chunks)}")
print(f"[DEBUG] Entities before linking: {len(entities)}")
print(f"[DEBUG] Entities after linking: {len(merged_entities)}")
```

### Dump GraphML for Inspection

```python
import networkx as nx
G = nx.read_graphml("path/to/graph.graphml")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
for node, data in list(G.nodes(data=True))[:5]:
    print(f"{node}: {data}")
```

---

## Future Improvements

### High Priority

1. **Batch paragraph extraction** - Combine multiple chunks in one LLM call
2. **Parallel processing** - Process chunks concurrently
3. **Caching** - Cache LLM responses to avoid repeated calls

### Medium Priority

4. **Embedding similarity for entity linking** - Replace fuzzy string matching
5. **LLM verification for uncertain merges** - Confirm low-confidence entity links
6. **Transform consistency validator** - Convert to alias extractor for query expansion

### Low Priority

7. **Support more storage backends** - Neo4j, MongoDB, etc.
8. **Multi-language prompt templates** - Native Bangla prompts for better extraction
9. **Adaptive thresholds** - Learn optimal thresholds per document type

---

## References

### Key Files for Contribution

| Task | Primary File | Supporting Files |
|------|-------------|------------------|
| **Add extraction mode** | `bigrag/extractors/constrained_extractor.py` | `bigrag/production_pipeline.py` |
| **Improve entity linking** | `bigrag/merging/entity_linker.py` | `bigrag/merging/canonicalization.py` |
| **Add validator** | `bigrag/validators/your_validator.py` | `bigrag/production_pipeline.py` |
| **Change thresholds** | `bigrag/validators/numeric_validator.py`, `bigrag/extractors/constrained_extractor.py` | - |
| **Add storage backend** | `bigrag/storage_backends/your_backend.py` | `bigrag/base.py` |

### External Dependencies

- **LLMs**: OpenAI GPT-4o/GPT-4o-mini (extraction), Google Gemini 2.5 Pro (validation)
- **Embeddings**: FlagEmbedding (bge-large-en-v1.5)
- **Graph**: NetworkX
- **Vector DB**: NanoVectorDB (default), optional: Milvus, ChromaDB
- **Utilities**: python-dotenv, regex, difflib

---

## Contact and Contribution

- **GitHub**: [BiG-RAG Repository](https://github.com/yourusername/BiG-RAG)
- **Issues**: Report bugs or request features via GitHub Issues
- **Pull Requests**: Follow coding style and add tests for new features

---

**End of Technical Documentation**
