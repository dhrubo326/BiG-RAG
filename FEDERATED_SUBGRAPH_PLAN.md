# Federated Subgraph Architecture for BiG-RAG
**Target Accuracy:** 99%+
**Domain:** Academic admission information (multi-university)
**Approach:** Institute-specific subgraphs + Agentic routing
**Last Updated:** 2025-01-22
 
---

## Executive Summary

### Current Problem
- **50 disconnected components** in single-document graph
- **Entity-relation disconnection**: Entities from same chunk not properly linked to their relations
- **Cross-institute confusion**: KUET CSE mixed with RUET CSE in same graph
- **Fixed-size chunking**: Breaks semantic units (tables, paragraphs)

### Proposed Solution: Federated Subgraph Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    FEDERATED ARCHITECTURE                       │
└────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   MASTER MAP        │
                    │   (LLM-readable)    │
                    │                     │
                    │ KUET: CSE, EEE...   │
                    │ BUET: ARCH, CE...   │
                    │ RUET: ME, IPE...    │
                    │ DU: Physics, CS...  │
                    └──────────┬──────────┘
                               │
                     Agentic Router (LLM)
                               │
        ┌──────────────┬───────┴───────┬──────────────┐
        │              │               │              │
   ┌────▼────┐   ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
   │ KUET KG │   │ BUET KG │    │ RUET KG │   │  DU KG  │
   │ (dense) │   │ (dense) │    │ (dense) │   │ (dense) │
   └─────────┘   └─────────┘    └─────────┘   └─────────┘

    Each subgraph:
    - Self-contained (no cross-contamination)
    - Highly connected (same-chunk entities linked to relations)
    - Context-aware chunks (semantic boundaries)
```

**Key Benefits:**
1. ✅ **No cross-institute confusion**: KUET CSE ≠ RUET CSE (separate graphs)
2. ✅ **Dense subgraphs**: Within-chunk entity-relation linking ensures connectivity
3. ✅ **Hallucination reduction**: Limited search space per query
4. ✅ **Parallel retrieval**: Query multiple subgraphs simultaneously for complex questions
5. ✅ **Incremental updates**: Add new universities without rebuilding entire graph
6. ✅ **Preserves bipartite structure**: Entities ↔ Relations ONLY (no entity-entity edges)

---

## Phase 1: Within-Chunk Entity-Relation Linking (CRITICAL FIX)

### Problem Analysis

**Current Issue:**
```python
# Current extraction creates:
Chunk_001: "CSE department has 120 seats"
  → Entity: CSE (source_id: chunk_001)
  → Entity: 120 (source_id: chunk_001)
  → Relation: "CSE has 120 seats" (source_id: chunk_001)

# BUT in graph:
#   CSE node ↔ ? (no edge created!)
#   120 node ↔ ? (no edge created!)
#   rel-abc ↔ ? (no edge created!)
#
# Result: 3 isolated nodes instead of connected component
```

**Root Cause:** `_merge_edges_then_upsert()` expects edges but current extraction only creates nodes.

### Solution: Explicit Within-Chunk Edge Creation

**Implementation Location:** `bigrag/operate.py` - modify `_process_single_content()`

```python
# MODIFIED: bigrag/operate.py line ~750-900

async def _process_single_content(
    chunk_key_dp: tuple[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
) -> tuple[dict, dict]:
    """
    Extract entities and relations WITH within-chunk edge creation.

    NEW BEHAVIOR:
    - Extract entities and relations (existing)
    - CREATE EDGES: relation ↔ entity for ALL entities in same chunk
    - Ensures every entity connects to at least one relation
    """

    chunk_key, chunk_dp = chunk_key_dp
    content = chunk_dp["content"]

    # Get metadata (title, category, etc.)
    doc_title = chunk_dp.get("doc_title", "")
    doc_metadata = chunk_dp.get("doc_metadata", {})

    # Build context-enriched input for LLM
    enriched_content = (
        f"[DOCUMENT CONTEXT]\n"
        f"Title: {doc_title}\n"
        f"Metadata: {json.dumps(doc_metadata, ensure_ascii=False)}\n\n"
        f"[CHUNK CONTENT]\n"
        f"{content}"
    )

    # STEP 1: Extract entities and relations (existing logic)
    llm_output = await llm_func(
        PROMPTS["entity_extraction"].format(
            input_text=enriched_content,
            language=language
        )
    )

    # Parse output (existing parsing logic)
    maybe_nodes, maybe_edges_initial = _parse_llm_output(llm_output, chunk_key)

    # STEP 2: NEW - Ensure within-chunk connectivity
    maybe_edges_enhanced = _ensure_within_chunk_connectivity(
        maybe_nodes,
        maybe_edges_initial,
        chunk_key
    )

    return maybe_nodes, maybe_edges_enhanced


def _ensure_within_chunk_connectivity(
    entities: dict,      # {entity_name: [entity_data, ...]}
    relations: dict,     # {relation_id: [relation_data, ...]}
    chunk_id: str
) -> dict:
    """
    Create edges connecting ALL entities to their chunk's relations.

    CRITICAL FIX for disconnected components.

    Strategy:
    1. For each relation in chunk:
       - Extract entity mentions from relation content
       - Create edge: relation ↔ entity for each mention
    2. If entity has NO edges after step 1:
       - Create edge to FIRST relation in chunk (fallback)

    Returns:
        Enhanced relations dict with edge information
    """
    enhanced_relations = relations.copy()
    entity_edge_count = {name: 0 for name in entities.keys()}

    # STEP 1: Link entities mentioned in relations
    for relation_id, relation_data_list in relations.items():
        for relation_data in relation_data_list:
            relation_content = relation_data.get("hyper_relation_content", "")

            # Find which entities are mentioned in this relation
            mentioned_entities = []
            for entity_name in entities.keys():
                # Case-insensitive check
                if entity_name.lower() in relation_content.lower():
                    mentioned_entities.append(entity_name)
                    entity_edge_count[entity_name] += 1

            # Store linked entities in relation metadata
            relation_data["linked_entities"] = mentioned_entities

            logger.debug(
                f"{chunk_id}: Relation '{relation_content[:40]}...' "
                f"linked to {len(mentioned_entities)} entities: {mentioned_entities}"
            )

    # STEP 2: Fallback - link orphan entities to first relation
    orphan_entities = [name for name, count in entity_edge_count.items() if count == 0]

    if orphan_entities and relations:
        # Get first relation
        first_relation_id = list(relations.keys())[0]
        first_relation_data = relations[first_relation_id][0]

        # Add orphan entities to this relation
        if "linked_entities" not in first_relation_data:
            first_relation_data["linked_entities"] = []

        first_relation_data["linked_entities"].extend(orphan_entities)

        logger.warning(
            f"{chunk_id}: Linked {len(orphan_entities)} orphan entities "
            f"to first relation (fallback): {orphan_entities}"
        )

    # STEP 3: Log connectivity stats
    total_entities = len(entities)
    connected_entities = sum(1 for count in entity_edge_count.values() if count > 0)

    logger.info(
        f"{chunk_id}: Within-chunk connectivity: "
        f"{connected_entities}/{total_entities} entities connected "
        f"({connected_entities/total_entities*100:.1f}%)"
    )

    return enhanced_relations
```

**Expected Impact:**
- Before: 50 components (3-5 nodes each)
- After: 1-3 components (90%+ of nodes connected)

---

## Phase 2: Context-Aware Chunking (Semantic Boundaries)

### Current Problem: Fixed-Size Chunking Breaks Semantic Units

```
Current (1200 tokens, 100 overlap):
Chunk_001: "...CSE department offers 120 seats. The department focuses on..."
                                              ↑ SPLIT HERE (arbitrary)
Chunk_002: "...software engineering and AI. Admission requirements include..."

Problem:
- "120 seats" separated from "CSE"
- Context lost across chunk boundary
```

### Solution: Content-Aware Chunking

**Strategy:**
1. **Detect semantic boundaries**: Paragraphs, tables, sections
2. **Chunk at boundaries**: Never split mid-paragraph or mid-table
3. **Adaptive chunk size**: 800-1500 tokens (flexible based on content)

**Implementation:**

```python
# NEW FILE: bigrag/preprocessors/semantic_chunker.py

from typing import List, Dict
import re

class SemanticChunker:
    """
    Context-aware chunking that respects semantic boundaries.

    Chunk boundaries:
    - Markdown headers (##, ###)
    - Paragraph breaks (double newline)
    - Table boundaries (before/after table)
    - List boundaries (before/after bulleted/numbered lists)
    """

    @staticmethod
    def chunk_by_semantic_units(
        markdown_text: str,
        min_chunk_size: int = 800,
        max_chunk_size: int = 1500,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Chunk document at semantic boundaries.

        Algorithm:
        1. Split by headers (## Section)
        2. Within sections, split by paragraphs
        3. Keep tables as standalone chunks
        4. Merge small chunks until min_chunk_size reached

        Returns:
            [
                {
                    'chunk_id': 'chunk_001',
                    'content': 'Full paragraph or section...',
                    'type': 'paragraph' | 'table' | 'section',
                    'metadata': {...}
                }
            ]
        """

        # Step 1: Extract tables FIRST (from Graph_indexing_plan.md Phase 1.1)
        tables = SemanticChunker._extract_table_blocks(markdown_text)

        # Step 2: Replace tables with placeholders
        text_without_tables = markdown_text
        for i, table in enumerate(tables):
            placeholder = f"<<<TABLE_{i:03d}>>>"
            text_without_tables = text_without_tables.replace(
                table['raw_text'],
                placeholder,
                1
            )

        # Step 3: Split by semantic units
        semantic_units = SemanticChunker._split_by_semantic_boundaries(
            text_without_tables
        )

        # Step 4: Merge small units and restore tables
        chunks = SemanticChunker._merge_and_restore_tables(
            semantic_units,
            tables,
            min_chunk_size,
            max_chunk_size,
            metadata
        )

        return chunks

    @staticmethod
    def _split_by_semantic_boundaries(text: str) -> List[Dict]:
        """
        Split text at semantic boundaries.

        Priority:
        1. Headers (##, ###) - highest priority
        2. Double newlines (paragraph breaks)
        3. Table placeholders
        4. List boundaries
        """
        units = []

        # Split by headers first
        header_pattern = r'^(#{1,3}\s+.+)$'
        sections = re.split(header_pattern, text, flags=re.MULTILINE)

        current_section_header = None
        for i, section in enumerate(sections):
            if re.match(header_pattern, section):
                current_section_header = section
            elif section.strip():
                # Split section by paragraphs
                paragraphs = re.split(r'\n\n+', section)

                for para in paragraphs:
                    if para.strip():
                        units.append({
                            'content': para.strip(),
                            'type': 'paragraph',
                            'section_header': current_section_header
                        })

        return units

    @staticmethod
    def _merge_and_restore_tables(
        units: List[Dict],
        tables: List[Dict],
        min_size: int,
        max_size: int,
        metadata: Dict
    ) -> List[Dict]:
        """
        Merge small units and restore table content.
        """
        from bigrag.utils import count_tokens

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0

        for unit in units:
            # Check if unit contains table placeholder
            table_match = re.search(r'<<<TABLE_(\d+)>>>', unit['content'])

            if table_match:
                # Flush current chunk before table
                if current_chunk:
                    chunks.append(
                        SemanticChunker._create_chunk(
                            current_chunk,
                            f'chunk_{chunk_id:04d}',
                            metadata
                        )
                    )
                    chunk_id += 1
                    current_chunk = []
                    current_size = 0

                # Create table chunk
                table_idx = int(table_match.group(1))
                table_data = tables[table_idx]

                chunks.append({
                    'chunk_id': f'chunk_{chunk_id:04d}',
                    'content': table_data['content'],
                    'type': 'table',
                    'metadata': {
                        **(metadata or {}),
                        'table_index': table_idx
                    }
                })
                chunk_id += 1
            else:
                # Regular paragraph unit
                unit_size = count_tokens(unit['content'])

                # Check if adding this unit exceeds max_size
                if current_size + unit_size > max_size and current_chunk:
                    # Flush current chunk
                    chunks.append(
                        SemanticChunker._create_chunk(
                            current_chunk,
                            f'chunk_{chunk_id:04d}',
                            metadata
                        )
                    )
                    chunk_id += 1
                    current_chunk = []
                    current_size = 0

                # Add unit to current chunk
                current_chunk.append(unit)
                current_size += unit_size

                # Flush if reached min_size and at semantic boundary
                if current_size >= min_size:
                    chunks.append(
                        SemanticChunker._create_chunk(
                            current_chunk,
                            f'chunk_{chunk_id:04d}',
                            metadata
                        )
                    )
                    chunk_id += 1
                    current_chunk = []
                    current_size = 0

        # Flush remaining
        if current_chunk:
            chunks.append(
                SemanticChunker._create_chunk(
                    current_chunk,
                    f'chunk_{chunk_id:04d}',
                    metadata
                )
            )

        return chunks

    @staticmethod
    def _create_chunk(units: List[Dict], chunk_id: str, metadata: Dict) -> Dict:
        """Create final chunk from semantic units."""
        content = "\n\n".join(u['content'] for u in units)

        return {
            'chunk_id': chunk_id,
            'content': content,
            'type': 'paragraph',
            'metadata': metadata or {}
        }

    @staticmethod
    def _extract_table_blocks(markdown_text: str) -> List[Dict]:
        """Extract markdown tables."""
        table_pattern = r'(\|[^\n]+\|(?:\n\|[^\n]+\|)+)'
        tables = []

        for match in re.finditer(table_pattern, markdown_text):
            raw_text = match.group(1)

            # Convert to natural language (simplified)
            # Full implementation should use GPT-4o (from Graph_indexing_plan.md)
            tables.append({
                'raw_text': raw_text,
                'content': raw_text,  # Placeholder - should convert to NL
                'type': 'table'
            })

        return tables
```

**Expected Impact:**
- Before: ~10 chunks with broken context
- After: ~6-8 chunks with complete semantic units
- Better entity-relation co-occurrence (entities stay with their describing text)

---

## Phase 3: Entity Canonicalization (NO Entity-Entity Edges)

**IMPORTANT:** We keep bipartite structure intact. NO direct entity→entity edges.
Only canonicalize entity names to merge duplicates.

### Strategy: Simple Name-Based Canonicalization

**Within-Graph Entity Canonicalization:**
```python
# NEW FILE: bigrag/merging/simple_canonicalization.py

class SimpleEntityCanonicalizer:
    """
    Lightweight entity canonicalization for single-institute graphs.

    No LLM needed - just exact matching + department codes.
    """

    def __init__(self, institute_name: str):
        self.institute = institute_name
        self.canonical_map = self._load_institute_mappings()

    def _load_institute_mappings(self) -> Dict[str, str]:
        """
        Load pre-defined mappings for specific institute.

        Example for KUET:
        {
            'CSE': 'COMPUTER SCIENCE AND ENGINEERING',
            'কম্পিউটার সায়েন্স': 'COMPUTER SCIENCE AND ENGINEERING',
            'Computer Science': 'COMPUTER SCIENCE AND ENGINEERING',
            ...
        }
        """
        mappings = {
            'KUET': {
                'CSE': 'COMPUTER SCIENCE AND ENGINEERING',
                'কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং': 'COMPUTER SCIENCE AND ENGINEERING',
                'EEE': 'ELECTRICAL AND ELECTRONIC ENGINEERING',
                'ইলেক্ট্রিক্যাল এন্ড ইলেক্ট্রনিক': 'ELECTRICAL AND ELECTRONIC ENGINEERING',
                # ... all 16 KUET departments
            },
            'BUET': {
                'ARCH': 'ARCHITECTURE',
                'স্থাপত্য': 'ARCHITECTURE',
                # ... all BUET departments
            },
            # Add RUET, DU, etc.
        }

        return mappings.get(self.institute, {})

    def canonicalize(self, entity_name: str) -> str:
        """Return canonical form or original if no mapping."""
        # Try exact match
        if entity_name in self.canonical_map:
            return self.canonical_map[entity_name]

        # Try case-insensitive
        for variant, canonical in self.canonical_map.items():
            if variant.lower() == entity_name.lower():
                return canonical

        # No mapping - return original
        return entity_name
```

**Why NO Entity-Entity Edges:**
- Preserves bipartite structure (entities ↔ relations only)
- Existing retrieval logic works without modification
- Entity connections happen through shared relations (natural multi-hop)
- Simpler implementation, no breaking changes

**How Canonicalization Helps:**
```
Before canonicalization:
- Entity: "CSE" (from chunk_001)
- Entity: "Computer Science and Engineering" (from chunk_002)
- Entity: "কম্পিউটার সায়েন্স" (from chunk_003)
→ 3 separate nodes (disconnected)

After canonicalization:
- Entity: "COMPUTER SCIENCE AND ENGINEERING" (merged)
  - source_id: chunk_001, chunk_002, chunk_003
  - weight: sum of all occurrences
→ 1 node (connected to all chunks' relations)
```

**Expected Impact:**
- Reduces duplicate entities by 40-60%
- Increases graph connectivity through entity merging
- Multi-hop reasoning works through relation nodes (bipartite traversal)

---

## Phase 4: Federated Subgraph Architecture

### 4.1 Subgraph Building (Per-Institute)

**Directory Structure:**
```
expr/
├── federated/
│   ├── master_map.json          # Institute → entities mapping
│   ├── KUET/
│   │   ├── graph_chunk_entity_relation.graphml
│   │   ├── vdb_entities.json
│   │   ├── vdb_relations.json
│   │   ├── vdb_chunks.json
│   │   └── kv_store_text_chunks.json
│   ├── BUET/
│   │   └── ... (same structure)
│   ├── RUET/
│   │   └── ...
│   └── DU/
│       └── ...
```

**Master Map Format:**
```json
{
  "version": "1.0",
  "last_updated": "2025-01-22T10:30:00Z",
  "institutes": {
    "KUET": {
      "full_name": "Khulna University of Engineering and Technology",
      "full_name_bn": "খুলনা প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয়",
      "aliases": ["KUET", "খুলনা বিশ্ববিদ্যালয়"],
      "departments": [
        "Computer Science and Engineering",
        "Electrical and Electronic Engineering",
        "Civil Engineering",
        "... (all 16 departments)"
      ],
      "entity_count": 450,
      "relation_count": 380,
      "chunk_count": 12,
      "topics": ["admission", "departments", "seats", "eligibility", "exams"],
      "subgraph_path": "expr/federated/KUET"
    },
    "BUET": {
      "full_name": "Bangladesh University of Engineering and Technology",
      "full_name_bn": "বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়",
      "aliases": ["BUET", "বুয়েট"],
      "departments": [
        "Architecture",
        "Civil Engineering",
        "... (all BUET departments)"
      ],
      "entity_count": 520,
      "relation_count": 445,
      "chunk_count": 15,
      "topics": ["admission", "departments", "seats", "eligibility", "fees"],
      "subgraph_path": "expr/federated/BUET"
    }
  }
}
```

**Subgraph Builder:**
```python
# NEW FILE: bigrag/federated/subgraph_builder.py

from typing import List, Dict
from pathlib import Path

class SubgraphBuilder:
    """
    Build institute-specific knowledge graphs.

    Each institute gets its own isolated graph.
    """

    def __init__(self, federated_root: str = "expr/federated"):
        self.federated_root = Path(federated_root)
        self.federated_root.mkdir(parents=True, exist_ok=True)

    async def build_subgraph_for_institute(
        self,
        institute_name: str,      # "KUET", "BUET", etc.
        documents: List[str],     # Markdown documents for this institute
        metadata: Dict = None
    ):
        """
        Build isolated knowledge graph for single institute.

        Steps:
        1. Create institute directory
        2. Chunk documents (semantic chunking)
        3. Extract entities + relations (with within-chunk linking)
        4. Apply entity canonicalization (institute-specific)
        5. Create cross-chunk co-occurrence edges
        6. Build indices (VDB + graph)
        7. Update master map
        """

        # Create institute directory
        institute_dir = self.federated_root / institute_name
        institute_dir.mkdir(exist_ok=True)

        logger.info(f"Building subgraph for {institute_name}...")

        # Step 1: Semantic chunking
        from bigrag.preprocessors.semantic_chunker import SemanticChunker

        all_chunks = []
        for doc in documents:
            chunks = SemanticChunker.chunk_by_semantic_units(
                doc,
                metadata={
                    'institute': institute_name,
                    **(metadata or {})
                }
            )
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} semantic chunks for {institute_name}")

        # Step 2: Extract entities + relations (existing BiGRAG logic)
        from bigrag import BiGRAG

        rag = BiGRAG(
            working_dir=str(institute_dir),
            enable_llm_cache=True
        )

        # Insert chunks (triggers extraction)
        await rag.ainsert(
            [chunk['content'] for chunk in all_chunks],
            metadata=[chunk['metadata'] for chunk in all_chunks]
        )

        # Step 3: Entity canonicalization
        from bigrag.merging.simple_canonicalization import SimpleEntityCanonicalizer

        canonicalizer = SimpleEntityCanonicalizer(institute_name)
        await self._apply_canonicalization(rag, canonicalizer)

        # Step 4: REMOVED - No entity-entity edges (preserves bipartite structure)
        # Canonicalization already applied in Step 3

        # Step 5: Update master map
        await self._update_master_map(institute_name, rag)

        logger.info(f"Subgraph for {institute_name} built successfully")
        logger.info(f"  - Entities: {await rag.chunk_entity_relation_graph.get_node_count()}")
        logger.info(f"  - Relations: {await rag.chunk_entity_relation_graph.get_edge_count()}")
        logger.info(f"  - Chunks: {len(all_chunks)}")

    async def _apply_canonicalization(
        self,
        rag: 'BiGRAG',
        canonicalizer: SimpleEntityCanonicalizer
    ):
        """Apply entity canonicalization to graph."""
        all_entities = await rag.chunk_entity_relation_graph.get_all_nodes(role="entity")

        for entity in all_entities:
            canonical_name = canonicalizer.canonicalize(entity['entity_name'])

            if canonical_name != entity['entity_name']:
                # Merge entity into canonical form
                logger.info(
                    f"Canonicalizing: {entity['entity_name']} → {canonical_name}"
                )
                # (Merging logic - reuse existing _merge_nodes_then_upsert)

    async def _update_master_map(self, institute_name: str, rag: 'BiGRAG'):
        """Update master map with subgraph metadata."""
        import json

        master_map_path = self.federated_root / "master_map.json"

        # Load existing map
        if master_map_path.exists():
            with open(master_map_path, 'r', encoding='utf-8') as f:
                master_map = json.load(f)
        else:
            master_map = {
                "version": "1.0",
                "institutes": {}
            }

        # Get stats from subgraph
        entity_count = await rag.chunk_entity_relation_graph.get_node_count()
        # ... (extract other stats)

        # Update map
        master_map["institutes"][institute_name] = {
            "full_name": institute_name,  # TODO: Get from metadata
            "entity_count": entity_count,
            # ... (other metadata)
            "subgraph_path": f"expr/federated/{institute_name}"
        }
        master_map["last_updated"] = datetime.now().isoformat()

        # Save
        with open(master_map_path, 'w', encoding='utf-8') as f:
            json.dump(master_map, f, ensure_ascii=False, indent=2)
```

---

### 4.2 Agentic Router (LLM-based Query Routing)

```python
# NEW FILE: bigrag/federated/agentic_router.py

from typing import List, Dict
import json

class AgenticRouter:
    """
    LLM-based router to select relevant subgraphs for queries.

    Uses master map to make informed routing decisions.
    """

    def __init__(
        self,
        master_map_path: str,
        llm_func,
        enable_parallel: bool = True
    ):
        self.master_map = self._load_master_map(master_map_path)
        self.llm_func = llm_func
        self.enable_parallel = enable_parallel

    def _load_master_map(self, path: str) -> Dict:
        """Load master map."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def route_query(self, query: str, language: str = "auto") -> Dict:
        """
        Determine which subgraph(s) to query.

        Returns:
            {
                'institutes': ['KUET', 'BUET'],  # Subgraphs to query
                'query_type': 'single' | 'comparative',
                'reasoning': 'Query mentions KUET and BUET departments'
            }
        """

        # Create routing prompt
        routing_prompt = self._create_routing_prompt(query)

        # Ask LLM
        llm_response = await self.llm_func(
            routing_prompt,
            max_tokens=200,
            temperature=0.0
        )

        # Parse response
        routing_decision = self._parse_routing_decision(llm_response)

        logger.info(
            f"[Router] Query: '{query[:50]}...' → "
            f"Institutes: {routing_decision['institutes']} "
            f"(reasoning: {routing_decision['reasoning']})"
        )

        return routing_decision

    def _create_routing_prompt(self, query: str) -> str:
        """
        Create prompt for LLM router.

        Includes master map context for informed decision.
        """
        # Extract institute info from master map
        institutes_info = []
        for name, info in self.master_map['institutes'].items():
            institutes_info.append(
                f"- {name} ({info['full_name']}): "
                f"Departments: {', '.join(info['departments'][:3])}..."
            )

        institutes_context = "\n".join(institutes_info)

        return f"""You are a query routing agent for a federated knowledge graph system.

AVAILABLE INSTITUTES:
{institutes_context}

USER QUERY:
{query}

TASK: Determine which institute(s) are relevant to this query.

RULES:
1. Single institute query: Return ONE institute if query is specific (e.g., "KUET CSE seats")
2. Comparative query: Return MULTIPLE institutes if comparing (e.g., "Compare KUET and BUET CSE")
3. General query: Return ALL institutes if query is broad (e.g., "engineering admission in Bangladesh")

OUTPUT FORMAT (JSON only):
{{
  "institutes": ["KUET"],  // or ["KUET", "BUET"] for multi
  "query_type": "single" | "comparative" | "general",
  "reasoning": "Brief explanation"
}}

OUTPUT:"""

    def _parse_routing_decision(self, llm_response: str) -> Dict:
        """Parse LLM routing decision."""
        try:
            decision = json.loads(llm_response)

            # Validate institutes exist in master map
            valid_institutes = [
                inst for inst in decision['institutes']
                if inst in self.master_map['institutes']
            ]

            if not valid_institutes:
                # Fallback: search all
                logger.warning("No valid institutes in routing decision, falling back to all")
                valid_institutes = list(self.master_map['institutes'].keys())

            return {
                'institutes': valid_institutes,
                'query_type': decision.get('query_type', 'single'),
                'reasoning': decision.get('reasoning', 'LLM routing decision')
            }

        except json.JSONDecodeError:
            # Fallback: search all institutes
            logger.error(f"Failed to parse routing decision: {llm_response}")
            return {
                'institutes': list(self.master_map['institutes'].keys()),
                'query_type': 'general',
                'reasoning': 'Parser fallback - searching all institutes'
            }
```

---

### 4.3 Federated Query Executor

```python
# NEW FILE: bigrag/federated/federated_query.py

import asyncio
from typing import List, Dict

class FederatedQueryExecutor:
    """
    Execute queries across multiple subgraphs (parallel or sequential).

    Combines results from multiple institutes.
    """

    def __init__(
        self,
        federated_root: str,
        router: AgenticRouter,
        enable_parallel: bool = True
    ):
        self.federated_root = Path(federated_root)
        self.router = router
        self.enable_parallel = enable_parallel
        self.loaded_subgraphs = {}  # Cache loaded BiGRAG instances

    async def query(
        self,
        query: str,
        language: str = "auto",
        query_param: QueryParam = None
    ) -> List[Dict]:
        """
        Execute federated query.

        Steps:
        1. Route query to relevant subgraphs
        2. Query selected subgraphs (parallel if enabled)
        3. Aggregate and rank results
        4. Return combined context
        """

        # Step 1: Route query
        routing_decision = await self.router.route_query(query, language)
        selected_institutes = routing_decision['institutes']

        logger.info(
            f"[Federated Query] Querying {len(selected_institutes)} institutes: "
            f"{selected_institutes}"
        )

        # Step 2: Load subgraphs (lazy loading with cache)
        subgraphs = []
        for institute in selected_institutes:
            subgraph = await self._load_subgraph(institute)
            subgraphs.append((institute, subgraph))

        # Step 3: Query subgraphs (parallel or sequential)
        if self.enable_parallel and len(subgraphs) > 1:
            # Parallel execution
            results_per_institute = await asyncio.gather(*[
                self._query_single_subgraph(
                    institute,
                    subgraph,
                    query,
                    query_param
                )
                for institute, subgraph in subgraphs
            ])
        else:
            # Sequential execution
            results_per_institute = []
            for institute, subgraph in subgraphs:
                result = await self._query_single_subgraph(
                    institute,
                    subgraph,
                    query,
                    query_param
                )
                results_per_institute.append(result)

        # Step 4: Aggregate results
        aggregated_results = self._aggregate_results(
            results_per_institute,
            selected_institutes,
            routing_decision['query_type']
        )

        return aggregated_results

    async def _load_subgraph(self, institute_name: str) -> 'BiGRAG':
        """Load subgraph (with caching)."""
        if institute_name in self.loaded_subgraphs:
            return self.loaded_subgraphs[institute_name]

        from bigrag import BiGRAG

        subgraph_dir = self.federated_root / institute_name

        rag = BiGRAG(
            working_dir=str(subgraph_dir),
            enable_llm_cache=True
        )

        # Cache for future queries
        self.loaded_subgraphs[institute_name] = rag

        logger.info(f"Loaded subgraph: {institute_name}")
        return rag

    async def _query_single_subgraph(
        self,
        institute_name: str,
        subgraph: 'BiGRAG',
        query: str,
        query_param: QueryParam
    ) -> Dict:
        """Query single subgraph."""
        logger.info(f"[{institute_name}] Querying subgraph...")

        # Query subgraph (existing BiGRAG query logic)
        results = await subgraph.aquery(
            query,
            param=query_param or QueryParam()
        )

        # Add institute metadata to results
        for item in results:
            item['institute'] = institute_name
            item['institute_full_name'] = self.router.master_map['institutes'][institute_name]['full_name']

        logger.info(f"[{institute_name}] Retrieved {len(results)} context items")

        return {
            'institute': institute_name,
            'results': results
        }

    def _aggregate_results(
        self,
        results_per_institute: List[Dict],
        institutes: List[str],
        query_type: str
    ) -> List[Dict]:
        """
        Aggregate results from multiple subgraphs.

        Strategy:
        - Single query: Return top results from primary institute
        - Comparative query: Interleave results from all institutes
        - General query: Merge and re-rank by relevance
        """

        if query_type == 'single':
            # Return results from first institute (most relevant)
            primary = results_per_institute[0]
            return primary['results']

        elif query_type == 'comparative':
            # Interleave results from all institutes
            aggregated = []
            max_results = max(len(r['results']) for r in results_per_institute)

            for i in range(max_results):
                for inst_results in results_per_institute:
                    if i < len(inst_results['results']):
                        aggregated.append(inst_results['results'][i])

            return aggregated

        else:  # general
            # Merge all and re-rank by coherence score
            all_results = []
            for inst_results in results_per_institute:
                all_results.extend(inst_results['results'])

            # Sort by coherence (existing BiGRAG scoring)
            all_results.sort(
                key=lambda x: x.get('<coherence>', 0),
                reverse=True
            )

            return all_results
```

---

## Phase 5: Language Override & Query Preprocessing

### Per-Query Language Parameter

**Already implemented** (from API_UPDATES_2025.md), just need to ensure it works with federated system.

**Integration:**
```python
# MODIFIED: bigrag/federated/federated_query.py

async def query(
    self,
    query: str,
    language: str = "auto",  # Support per-query language override
    query_param: QueryParam = None
) -> List[Dict]:
    """
    Execute federated query with language override.

    Language precedence:
    1. Per-query language parameter (highest)
    2. Global DEFAULT_LANGUAGE from .env
    3. Auto-detection (langdetect)
    """

    # Pass language to each subgraph
    for institute, subgraph in subgraphs:
        result = await self._query_single_subgraph(
            institute,
            subgraph,
            query,
            query_param,
            language=language  # NEW: pass language parameter
        )
```

---

## Implementation Plan (4 Weeks)

### Week 1: Core Fixes
- [ ] **Day 1-2**: Implement within-chunk entity-relation linking
  - Modify `_process_single_content()` in `bigrag/operate.py`
  - Add `_ensure_within_chunk_connectivity()` function
  - Test on KUET document (target: 90%+ connectivity)

- [ ] **Day 3-4**: Implement semantic chunking
  - Create `bigrag/preprocessors/semantic_chunker.py`
  - Test on KUET document (verify no broken tables/paragraphs)

- [ ] **Day 5**: Integration testing
  - Build single KUET graph with both fixes
  - Measure component count (target: <5 components)

### Week 2: Entity Canonicalization (NO Entity-Entity Edges)
- [ ] **Day 6-7**: Create canonicalization maps
  - Create `bigrag/merging/simple_canonicalization.py`
  - Add KUET, BUET, RUET department mappings

- [ ] **Day 8-9**: Implement canonicalization merge logic
  - Merge duplicate entities during graph building
  - Test on KUET (verify entity merging works)

- [ ] **Day 10**: Integration testing
  - Rebuild KUET graph with all fixes
  - Measure connectivity + accuracy (target: 85%+ accuracy)
  - Verify bipartite structure intact (entities ↔ relations ONLY)

### Week 3: Federated Architecture
- [ ] **Day 11-12**: Subgraph builder
  - Create `bigrag/federated/subgraph_builder.py`
  - Build KUET, BUET, RUET subgraphs

- [ ] **Day 13-14**: Master map + router
  - Generate master map
  - Implement `AgenticRouter` with LLM routing

- [ ] **Day 15**: Federated query executor
  - Implement `FederatedQueryExecutor`
  - Test parallel querying

### Week 4: Testing & Optimization
- [ ] **Day 16-17**: Comprehensive QA testing
  - 50 single-institute queries (target: 95%+ accuracy)
  - 20 comparative queries (target: 90%+ accuracy)

- [ ] **Day 18-19**: Performance optimization
  - Measure query latency (target: <300ms P99)
  - Implement subgraph caching

- [ ] **Day 20**: Documentation & deployment
  - Update API docs
  - Deploy to production

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **Graph connectivity** | 50 components | <5 components | NetworkX component analysis |
| **Within-chunk connectivity** | ~20% | >90% | Entity-relation edge ratio |
| **Cross-institute confusion** | High (mixed KUET/BUET) | 0% | Manual review of 100 queries |
| **Query accuracy (single)** | 60% | >95% | EM on single-institute test set |
| **Query accuracy (comparative)** | N/A | >90% | EM on comparative test set |
| **Query latency (P99)** | ~150ms | <300ms | Federated query with 2 subgraphs |
| **Hallucination rate** | Medium | <1% | Manual review of 100 generated answers |

---

## API Changes

### New Endpoints

**1. Build Federated Subgraph**
```python
POST /api/v1/federated/build_subgraph
{
  "institute_name": "KUET",
  "documents": ["path/to/kuet_admission.md"],
  "metadata": {"category": "admission", "year": "2024-2025"}
}
```

**2. Federated Query**
```python
POST /api/v1/federated/query
{
  "query": "কুয়েটে CSE তে কতটি আসন আছে?",
  "language": "Bangla",  // Optional override
  "enable_routing": true,  // Default: true
  "force_institutes": []   // Optional: override router
}

Response:
{
  "query": "কুয়েটে CSE তে কতটি আসন আছে?",
  "routing": {
    "institutes": ["KUET"],
    "reasoning": "Query specifically mentions KUET"
  },
  "results": [
    {
      "content": "KUET CSE department has 120 seats",
      "institute": "KUET",
      "type": "relation",
      "coherence": 0.95
    }
  ]
}
```

**3. Get Master Map**
```python
GET /api/v1/federated/map

Response:
{
  "institutes": {
    "KUET": {"departments": [...], "entity_count": 450},
    "BUET": {"departments": [...], "entity_count": 520}
  }
}
```

---

## Questions & Clarifications Needed

### Question 1: Cross-Institute Entity Handling
**Scenario:** KUET has "CSE" and BUET has "CSE". Should they:
- **Option A**: Keep separate (recommended) - KUET.CSE vs BUET.CSE
- **Option B**: Share entity with institute metadata

**My recommendation:** Option A - Complete isolation. No shared entities between subgraphs.

### Question 2: Router Fallback Strategy
**Scenario:** Router returns empty institute list or invalid institute.

**Current approach:** Query ALL subgraphs (safe but slower)

**Alternative:** Ask user to clarify which institute?

**My recommendation:** Query all (safe default), add confidence threshold to warn user.

### Question 3: Comparative Query Result Ranking
**Scenario:** User asks "Compare KUET and BUET CSE seats"

**Should results be:**
- **Option A**: Grouped by institute (KUET results first, then BUET)
- **Option B**: Interleaved by relevance (alternate KUET/BUET)
- **Option C**: Ranked globally by coherence score (mixed)

**My recommendation:** Option A for comparative queries (easier to read), Option C for general queries.

### Question 4: Subgraph Update Strategy
**Scenario:** New KUET document added (e.g., 2025-2026 admission info)

**Should we:**
- **Option A**: Rebuild entire KUET subgraph (slow but safe)
- **Option B**: Incremental update (fast but complex)

**My recommendation:** Option A initially, add Option B in future iteration.

---

## Migration Path from Current System

### Step 1: Test on Single Institute (KUET)
1. Apply all fixes (within-chunk linking, semantic chunking, co-occurrence)
2. Build KUET subgraph
3. Measure improvement vs current system
4. If successful (>85% accuracy), proceed

### Step 2: Add Second Institute (BUET)
1. Build BUET subgraph
2. Create master map
3. Test router on KUET vs BUET queries
4. Measure routing accuracy (target: >95%)

### Step 3: Full Deployment (All Institutes)
1. Build RUET, DU, CUET subgraphs
2. Deploy federated system
3. Monitor hallucination rate
4. Iterate based on feedback

---

## Estimated Costs (LLM Usage)

### Per-Institute Subgraph Build
| Component | Model | Calls | Cost |
|-----------|-------|-------|------|
| Entity extraction | GPT-4o-mini | 10 chunks × 1 | $0.10 |
| Router decisions | GPT-4o-mini | 1 per query | $0.001 |
| **Total per institute** | | | **~$0.10** |

### Per Query (Federated)
| Component | Cost |
|-----------|------|
| Router LLM call | $0.001 |
| Subgraph query (FAISS) | $0 |
| **Total per query** | **$0.001** |

**Very affordable** - Even 10,000 queries/month = $10

---

## Summary: Why This Approach is Better

| Aspect | Previous (Hierarchical Hub) | New (Federated Subgraph) |
|--------|---------------------------|------------------------|
| **Complexity** | High (4-layer hierarchy) | Medium (isolated subgraphs) |
| **Cross-contamination** | Possible (shared graph) | Impossible (isolated graphs) |
| **Hallucination risk** | Higher (large search space) | Lower (limited scope) |
| **Scalability** | Rebuild entire graph for new institute | Add new subgraph independently |
| **Retrieval changes** | Major (hub traversal logic) | Minor (existing logic per subgraph) |
| **Query latency** | Unknown (complex traversal) | Predictable (parallel subgraph queries) |
| **Accuracy** | Uncertain | High (domain isolation + dense connectivity) |

---

## Next Steps

1. **Review this plan** - Confirm approach aligns with your vision
2. **Answer clarification questions** (Questions 1-4 above)
3. **Approve implementation** - I'll start with Week 1 tasks
4. **Iterative testing** - Test after each week, adjust plan based on results

---

**Ready to implement?** Let me know if you want to proceed or need any modifications to this plan!
