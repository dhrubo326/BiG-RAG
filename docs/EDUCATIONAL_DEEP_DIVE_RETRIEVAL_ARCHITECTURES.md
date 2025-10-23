# Educational Deep Dive: Graph-R1 vs HyperGraphRAG vs BiG-RAG Retrieval Architectures

**Author:** Technical Analysis for Understanding Three GraphRAG Paradigms
**Date:** 2025-01-22
**Purpose:** Educational deep dive into retrieval processes and architectural paradigms

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Foundational Concepts](#foundational-concepts)
3. [Graph-R1: The RL-Agentic Paradigm](#graph-r1-the-rl-agentic-paradigm)
4. [HyperGraphRAG: The Single-Shot Paradigm](#hypergraphrag-the-single-shot-paradigm)
5. [BiG-RAG: The Algorithmic Adaptive Paradigm](#big-rag-the-algorithmic-adaptive-paradigm)
6. [Side-by-Side Comparison](#side-by-side-comparison)
7. [Practical Examples](#practical-examples)
8. [When to Use Which System](#when-to-use-which-system)
9. [Implementation Insights](#implementation-insights)

---

## Executive Summary

### The Three Paradigms

| Aspect | Graph-R1 | HyperGraphRAG | BiG-RAG |
|--------|----------|---------------|---------|
| **Philosophy** | RL-trained agent explores graph via multi-turn interaction | Single-shot dual-path retrieval + bidirectional expansion | Zero-training algorithmic with query-adaptive traversal |
| **Retrieval Style** | **Multi-turn iterative** (agent decides when to stop) | **Single-turn** (retrieve once, expand, generate) | **Single-turn adaptive** (depth varies by query complexity) |
| **Training Required** | ✅ Yes (GRPO/RL training ~72 hours) | ❌ No | ❌ No |
| **Query Processing** | LLM generates sub-queries in `<query>` tags | LLM extracts entities once | spaCy + LLM extract entities once |
| **Graph Traversal** | Agent decides via `<think>` steps | Fixed bidirectional expansion | Adaptive BFS (3-15 hops based on complexity) |
| **Strength** | Best accuracy on complex multi-hop queries | Fast single-shot retrieval | Best balance: no training + adaptive depth |
| **Weakness** | Requires RL training + GPU infrastructure | Fixed expansion may under/over-retrieve | More complex implementation |

### The Core Difference

The fundamental distinction is **not** about vector search vs filter-based retrieval (all three use vector similarity), but about:

1. **Interaction Model:**
   - **Graph-R1:** Multi-turn agent loop (think → query → retrieve → rethink → query again → answer)
   - **HyperGraphRAG:** Single retrieval → fixed expansion → answer
   - **BiG-RAG:** Single retrieval → adaptive expansion (depth varies by query) → answer

2. **Decision Making:**
   - **Graph-R1:** RL-trained LLM decides when to query and when to stop
   - **HyperGraphRAG:** Hardcoded expansion (always expands retrieved entities/hyperedges)
   - **BiG-RAG:** Linguistic analysis decides expansion depth before retrieval

3. **Optimization:**
   - **Graph-R1:** End-to-end RL with outcome-directed reward (F1 + format)
   - **HyperGraphRAG:** No optimization (uses pretrained LLM as-is)
   - **BiG-RAG:** Multi-factor coherence scoring (5 factors: similarity, hop distance, centrality, confidence, overlap)

---

## Foundational Concepts

### What is a Hypergraph?

Before diving into retrieval, understand the core data structure:

**Traditional Graph (Binary Relations):**
```
Entity A ──edge(relation)──> Entity B
```
Example: `(Dziga Vertov, directed, "In Memory of Sergo Ordzhonikidze")`

**Problem:** Real-world facts involve >2 entities:
```
"Male hypertensive patients with serum creatinine 115-133 µmol/L
 are diagnosed with mild serum creatinine elevation"
```

Decomposing into binary edges **loses semantic context**:
```
(Patient, Gender, Male)
(Patient, Has_Condition, Hypertension)
(Patient, Lab_Value, 115-133 µmol/L)
(Patient, Diagnosed_With, Mild elevation)
```

**Hypergraph (N-ary Relations):**
```
Hyperedge {
    entities: [Patient, Male, Hypertension, 115-133 µmol/L, Mild elevation]
    description: "Male hypertensive patients with serum creatinine 115-133 µmol/L..."
}
```

All three systems (Graph-R1, HyperGraphRAG, BiG-RAG) use hypergraphs to preserve n-ary relations.

### Bipartite Graph Representation

**Why Bipartite?**

A bipartite graph has two partitions where edges only connect nodes from different partitions.

**Structure:**
```
Partition 0 (Entities):        Partition 1 (Relations):
┌─────────────────┐            ┌──────────────────────────┐
│ Dziga Vertov    │────────────│ <hyperedge_1>:          │
│ Yelizaveta      │────────────│ "Dziga Vertov directed  │
│ Svilova         │────────────│  'In Memory of...' and  │
│ "In Memory of..." │───────────│   married Yelizaveta"   │
└─────────────────┘            └──────────────────────────┘
```

**Benefits:**
1. **No entity-entity edges** → Clear semantic separation
2. **Relation = hyperedge** → Preserves n-ary facts
3. **Efficient traversal** → Hop = entity → relation → entity → ...

**All three systems use bipartite graphs**, but differ in how they traverse them.

### Dual-Path Retrieval (Core Concept)

All three systems use **dual-path retrieval** but implement it differently:

**Two Retrieval Paths:**

1. **Entity-centric path:**
   - Extract entities from query (e.g., "Dziga Vertov")
   - Retrieve similar entities via vector similarity
   - Expand to connected hyperedges

2. **Relation-centric path:**
   - Embed full query or extract hyperedge-like descriptions
   - Retrieve similar hyperedges directly via vector similarity
   - Expand to participating entities

**Why Dual-Path?**

Entity search alone misses **relation-focused queries**:
```
Query: "What admission requirements apply to engineering programs?"
```
Entity search finds `[Engineering]` but misses specific requirements.
Relation search finds `<hyperedge>: "Engineering programs require Math A+, Physics B+..."`

---

## Graph-R1: The RL-Agentic Paradigm

### Core Philosophy

> **"Let an RL-trained agent decide when to query the graph and when to answer"**

Graph-R1 treats retrieval as a **Reinforcement Learning problem** where:
- **Agent:** LLM trained with GRPO (Group Relative Policy Optimization)
- **Environment:** Knowledge hypergraph G_H
- **Actions:** Think, Query, Retrieve, Answer
- **Reward:** F1 score (answer correctness) + format compliance

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER QUESTION                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
          ┌──────────────────────────────┐
          │   Graph-R1 Agent (LLM)       │
          │   Trained with RL (GRPO)     │
          └──────────────────────────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │   Multi-Turn Loop    │
              └──────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ↓                ↓                ↓
   ┌─────────┐     ┌──────────┐    ┌──────────┐
   │ <think> │ →   │ <query>  │ →  │ <answer> │
   │ Reflect │     │ Sub-Q    │    │ Terminate│
   └─────────┘     └──────────┘    └──────────┘
        │                │
        │                ↓
        │      ┌──────────────────┐
        │      │  Dual-Path       │
        │      │  Vector Search   │
        │      └──────────────────┘
        │                │
        └────────────────┴─── Loop until <answer>
```

### Retrieval Process (Step-by-Step)

#### Step 1: Agent Initialization

**Input:** User question `q`
**Output:** Initial state `s_1 = q`

**Example:**
```
q = "Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?"
s_1 = q  // Initial state = the question
```

#### Step 2: Multi-Turn Interaction

At each timestep `t`, the agent performs:

**Action Structure:**
```python
a_t = {
    "a_think": "<think>...</think>",     # Reflection
    "α_t": "continue" or "terminate",    # Decision
    "a_out": "<query>...</query>" or "<answer>...</answer>"
}
```

**Example Turn 1:**
```xml
<think>
I need to first find who directed the film 'In Memory of Sergo Ordzhonikidze'
</think>
<query>
Who directed the film 'In Memory of Sergo Ordzhonikidze'?
</query>
```

#### Step 3: Dual-Path Hypergraph Retrieval

**Given:** Query `a_query` from `<query>...</query>` tag

**(i) Entity-based Hyperedge Retrieval:**

```python
# Step 3a: Extract entities from query
V_query = extract_entities(a_query)
# Example: V_query = ["In Memory of Sergo Ordzhonikidze"]

# Step 3b: Retrieve top-k_v similar entities
R_v = argmax^{k_v}_{v in V} sim(φ(V_query), φ(v))
# Example: R_v = ["In Memory of Sergo Ordzhonikidze" (film entity)]

# Step 3c: Collect hyperedges connected to retrieved entities
F_v = ⋃_{v_i in R_v} {(e_h, V_eh) | v_i ∈ V_eh, e_h ∈ E_H}
# Example: F_v = {
#   (hyperedge_1, ["Dziga Vertov", "In Memory of...", "Director"]),
#   (hyperedge_2, ["Dziga Vertov", "Yelizaveta Svilova", "Spouse"])
# }
```

**(ii) Direct Hyperedge Retrieval:**

```python
# Step 3d: Retrieve top-k_h similar hyperedges directly
R_h = argmax^{k_h}_{e_h in E_H} sim(φ(a_query), φ(e_h))
# Example: R_h = [hyperedge_1: "Dziga Vertov directed 'In Memory of...'"]

# Step 3e: Collect hyperedge facts
F_h = ⋃_{e_i in R_h} {(e_i, V_ei) | V_ei ⊆ V}
```

**(iii) Reciprocal Rank Fusion (RRF):**

```python
# Step 3f: Merge results using RRF
a_ret = Top-k(F_v ∪ F_h, score(f) = 1/rank_v(f) + 1/rank_h(f))

# Example output:
# <knowledge>
# Hyperedge_1: Dziga Vertov directed "In Memory of Sergo Ordzhonikidze" in 1937.
# Entities: [Dziga Vertov, In Memory of Sergo Ordzhonikidze, Director]
# </knowledge>
```

#### Step 4: Agent Rethinks and Decides

**Agent receives:** `<knowledge>...</knowledge>` appended to state

```xml
<think>
Now I know Dziga Vertov directed the film. Next, I need to find his spouse.
</think>
<query>
Who is the spouse of Dziga Vertov?
</query>
```

**Graph-R1 performs another retrieval** (Steps 3a-3f repeat)

```xml
<knowledge>
Hyperedge_2: Dziga Vertov married Yelizaveta Svilova in 1923.
Entities: [Dziga Vertov, Yelizaveta Svilova, Spouse]
</knowledge>
```

#### Step 5: Agent Terminates

```xml
<think>
I have found the director (Dziga Vertov) and his spouse (Yelizaveta Svilova).
I can now answer the question.
</think>
<answer>
Yelizaveta Svilova
</answer>
```

### RL Training (GRPO)

**Objective:** Optimize policy π_θ to maximize:

```
J_GRPO(θ) = E[reward(trajectory)]
```

**Reward Function:**

```python
R(τ) = -1.0 + R_format(τ) + I{R_format = 1.0} * R_answer(a_ans)

# R_format: Checks each step has well-formed <think>, <query>/<answer>
# - 0.5 per valid step, capped at 1.0

# R_answer: F1 score between generated answer and ground truth
# - Token-level F1 = 2 * |tokens(answer) ∩ tokens(gold)| / (|answer| + |gold|)
```

**Training Process:**

1. **Sample trajectories:** Agent interacts with graph to generate N trajectories
2. **Compute rewards:** Calculate R(τ) for each trajectory
3. **Policy gradient:** Update LLM parameters θ to increase probability of high-reward trajectories
4. **Repeat:** Train for ~72 hours on 1x A100 GPU

**Result:** LLM learns:
- **When to query** (complex questions need multiple queries)
- **What to query** (decompose "spouse of director" into "find director" → "find spouse")
- **When to stop** (answer when sufficient knowledge retrieved)

### Key Advantages

✅ **Best accuracy on complex multi-hop queries** (57.8% avg F1 vs 46.2% for baselines)
✅ **Adaptive depth** (agent decides how many turns needed)
✅ **Learns from data** (RL training optimizes for answer quality)

### Key Limitations

❌ **Requires RL training** (~72 hours + GPU infrastructure)
❌ **Needs training data** (question-answer pairs for reward calculation)
❌ **More inference cost** (multiple LLM calls per query due to multi-turn interaction)

---

## HyperGraphRAG: The Single-Shot Paradigm

### Core Philosophy

> **"Retrieve once, expand bidirectionally, and generate - no training needed"**

HyperGraphRAG is the **simplest** paradigm:
- Extract entities from query → Retrieve → Expand → Generate
- **No RL training**
- **No multi-turn interaction**
- **Fixed expansion strategy**

### Architecture Overview

```
┌──────────────────────────────────────┐
│        USER QUESTION                 │
└──────────────────────────────────────┘
                │
                ↓
     ┌─────────────────────┐
     │ LLM Entity Extract  │
     └─────────────────────┘
                │
                ↓
     ┌─────────────────────────────┐
     │  Dual-Path Vector Search    │
     │  (Entity + Hyperedge)       │
     └─────────────────────────────┘
                │
                ↓
     ┌──────────────────────────────────┐
     │  Bidirectional Expansion         │
     │  - Expand hyperedges from entities│
     │  - Expand entities from hyperedges│
     └──────────────────────────────────┘
                │
                ↓
     ┌─────────────────────────┐
     │  Hybrid RAG Fusion      │
     │  (Hypergraph + Chunks)  │
     └─────────────────────────┘
                │
                ↓
     ┌──────────────────────┐
     │  LLM Generate Answer │
     └──────────────────────┘
```

### Retrieval Process (Step-by-Step)

#### Step 1: Entity Extraction from Query

**Prompt:**
```
Extract all entities mentioned in this question.
Return as JSON list.

Question: Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?
```

**LLM Output:**
```json
["In Memory of Sergo Ordzhonikidze"]
```

**Result:** `V_q = ["In Memory of Sergo Ordzhonikidze"]`

#### Step 2: Entity-Based Retrieval

**Retrieve top-k_V similar entities:**

```python
# Concatenate query entities into single text
h_Vq = embed(", ".join(V_q))
# h_Vq = embed("In Memory of Sergo Ordzhonikidze")

# Retrieve top-k_V entities from graph
R_V(q) = argmax^{k_V}_{v in V} (sim(h_Vq, h_v) ⊙ v^score)

# Example (k_V = 10):
R_V(q) = [
    "In Memory of Sergo Ordzhonikidze" (film),  # score: 0.98
    "Dziga Vertov",                              # score: 0.72
    "Yelizaveta Svilova",                        # score: 0.65
    ...
]
```

**Parameters:**
- `k_V = 10` (retrieve top 10 entities, configurable in HyperGraphRAG paper as 60)
- `sim(·,·)` = cosine similarity
- `v^score` = entity extraction confidence (boosts high-quality entities)

#### Step 3: Hyperedge-Based Retrieval

**Retrieve top-k_H similar hyperedges directly:**

```python
# Embed full query
h_q = embed("Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?")

# Retrieve top-k_H hyperedges
R_H(q) = argmax^{k_H}_{e_H in E_H} (sim(h_q, h_eH) ⊙ e_H^score)

# Example (k_H = 10):
R_H(q) = [
    hyperedge_1: "Dziga Vertov directed 'In Memory of...'",  # score: 0.85
    hyperedge_2: "Dziga Vertov married Yelizaveta Svilova",  # score: 0.73
    ...
]
```

#### Step 4: Bidirectional Expansion

**Goal:** Expand retrieval scope to capture complete n-ary facts

**(i) Expand Hyperedges from Retrieved Entities:**

```python
# For each retrieved entity, get ALL connected hyperedges
F_V = ⋃_{v_i in R_V(q)} {(e_H, V_eH) | v_i ∈ V_eH, e_H ∈ E_H}

# Example:
# R_V(q) = ["In Memory of...", "Dziga Vertov", "Yelizaveta Svilova"]
#
# F_V = {
#   (hyperedge_1, ["Dziga Vertov", "In Memory of...", "Director", "1937"]),
#   (hyperedge_2, ["Dziga Vertov", "Yelizaveta Svilova", "Spouse", "1923"]),
#   (hyperedge_3, ["Dziga Vertov", "Cinema Theorist", "Soviet Union"]),
#   ...
# }
```

**(ii) Expand Entities from Retrieved Hyperedges:**

```python
# For each retrieved hyperedge, get ALL participating entities
F_H = ⋃_{e_i in R_H(q)} {(e_i, V_ei) | V_ei ⊆ V}

# Example:
# R_H(q) = [hyperedge_1, hyperedge_2]
#
# F_H = {
#   (hyperedge_1, ["Dziga Vertov", "In Memory of...", "Director", "1937"]),
#   (hyperedge_2, ["Dziga Vertov", "Yelizaveta Svilova", "Spouse", "1923"])
# }
```

**(iii) Merge Results:**

```python
K_H = F_V ∪ F_H  # Union of both expansion paths

# K_H now contains ALL n-ary facts related to:
# - Retrieved entities (direct + expanded)
# - Retrieved hyperedges (direct + expanded)
```

**Key Point:** HyperGraphRAG **always expands** - no adaptive depth control.

#### Step 5: Hybrid RAG Fusion

**Combine hypergraph knowledge with chunk-based RAG:**

```python
# Retrieve traditional text chunks (standard RAG)
K_chunk = vector_search(q, text_chunks_db, top_k=5)

# Merge
K_final = K_H ∪ K_chunk
```

**Why Hybrid?**

Hypergraph captures **structured facts**, chunks capture **context/details**:
- Hypergraph: *"Dziga Vertov directed 'In Memory of...' in 1937"*
- Chunk: *"The film was a documentary about the Soviet politician..."*

#### Step 6: Generate Answer

**Prompt:**
```
Context:
{K_final}

Question:
Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?

Answer the question based on the context.
```

**LLM Output:**
```
Yelizaveta Svilova
```

### Key Advantages

✅ **No training required** (works with any pretrained LLM)
✅ **Fast single-shot retrieval** (no multi-turn interaction)
✅ **Simple implementation** (straightforward dual-path + expansion)
✅ **Hybrid fusion** (combines graph + chunks for comprehensive context)

### Key Limitations

❌ **Fixed expansion** (always expands, may over-retrieve for simple queries)
❌ **No adaptive depth** (can't adjust traversal based on query complexity)
❌ **May miss deep paths** (single-shot expansion limited to 2 hops effectively)

---

## BiG-RAG: The Algorithmic Adaptive Paradigm

### Core Philosophy

> **"Analyze query complexity linguistically, then adapt retrieval depth algorithmically - no training needed"**

BiG-RAG combines:
- **Query Complexity Classification** (SIMPLE / MODERATE / COMPLEX using spaCy)
- **Adaptive Multi-Hop Traversal** (3-15 hops based on complexity)
- **Multi-Factor Coherence Ranking** (5 factors beyond just similarity)

**No RL training, but more sophisticated than HyperGraphRAG's fixed expansion.**

### Architecture Overview

```
┌────────────────────────────────────────┐
│         USER QUESTION                  │
└────────────────────────────────────────┘
                 │
                 ↓
      ┌──────────────────────┐
      │  Query Complexity    │
      │  Classification      │
      │  (spaCy + Rules)     │
      └──────────────────────┘
                 │
       ┌─────────┼─────────┐
       │         │         │
    SIMPLE   MODERATE  COMPLEX
    3 hops   10 hops   Decompose
       │         │         │
       └─────────┼─────────┘
                 ↓
      ┌─────────────────────────┐
      │  Dual-Path Retrieval    │
      │  (Entity + Relation)    │
      └─────────────────────────┘
                 │
                 ↓
      ┌───────────────────────────────┐
      │  Adaptive Multi-Hop Traversal │
      │  (Structural Ranking)         │
      └───────────────────────────────┘
                 │
                 ↓
      ┌────────────────────────────┐
      │  Multi-Factor Coherence    │
      │  Ranking (5 factors)       │
      └────────────────────────────┘
                 │
                 ↓
      ┌──────────────────────┐
      │  LLM Generate Answer │
      └──────────────────────┘
```

### Retrieval Process (Step-by-Step)

#### Step 1: Query Complexity Classification

**Algorithm:**
```python
def classify_query(q):
    doc = spacy_parse(q)  # Dependency parsing

    # Feature extraction
    subordinate_markers = {"who", "which", "that", "whose", "of the"}
    has_subordinate = any(marker in q.lower() for marker in subordinate_markers)
    has_possessive = any(token.dep == "poss" for token in doc)
    num_entities = len(doc.ents)
    clause_depth = len([t for t in doc if t.dep in {"relcl", "acl", "ccomp"}])

    # Classification
    if has_subordinate and (has_possessive or clause_depth > 2):
        return "COMPLEX"
    elif num_entities > 2 or has_subordinate:
        return "MODERATE"
    else:
        return "SIMPLE"
```

**Example:**
```python
q = "Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?"

# Parsing results:
# - subordinate_markers: "of the" (2 occurrences)
# - has_possessive: True ("the director" → possessive dependency)
# - clause_depth: 2 (nested structure)

# Classification: COMPLEX
```

**Adaptive Parameters:**

| Complexity | max_hops | target_relations | Strategy |
|------------|----------|------------------|----------|
| SIMPLE     | 3        | 5                | Quick shallow search |
| MODERATE   | 10       | 15               | Standard multi-hop |
| COMPLEX    | 15       | 30               | Hierarchical decomposition |

#### Step 2: Dual-Path Initial Retrieval

**Same as HyperGraphRAG, but with optimized parameters:**

```python
# Extract entities from query
V_q = extract_entities(q)  # Example: ["In Memory of Sergo Ordzhonikidze"]

# Entity-based retrieval (dual_path_top_k = 10)
R_E = Top-10 entities by sim(φ(V_q), φ(e)) * e.conf

# Relation-based retrieval (dual_path_top_k = 10)
R_R = Top-10 relations by sim(φ(q), φ(r)) * r.conf

# Expand to connected nodes
F_E = expand_to_relations(R_E)
F_R = expand_to_entities(R_R)

# Reciprocal Rank Fusion
R_0 = RRF(F_E, F_R, k=10)
```

**Key Difference from HyperGraphRAG:**
- BiG-RAG uses **10** for initial retrieval (not 60)
- Why? Multi-hop expansion will discover additional relations

#### Step 3: Adaptive Multi-Hop Traversal

**For COMPLEX query → max_hops = 15, target_relations = 30**

**Algorithm 4: Adaptive Multi-Hop Traversal**

```python
def adaptive_multi_hop(R_0, max_hops, target_count):
    collected_relations = [(r, degree(r), weight(r), hop=0) for r in R_0]
    visited_entities = {e for r in R_0 for e in connected_entities(r)}
    visited_relations = set(R_0)
    current_frontier = visited_entities
    hop = 1

    while hop <= max_hops and len(collected_relations) < target_count:
        next_frontier = set()
        new_relations = []

        # Expand from entities to relations
        for e in current_frontier:
            neighbor_relations = get_relations_for_entity(e)

            for r in neighbor_relations:
                if r not in visited_relations:
                    visited_relations.add(r)

                    # Structural ranking (NOT cosine similarity)
                    edge_deg = degree(e) + degree(r)
                    weight = r.metadata['weight']

                    new_relations.append((r, edge_deg, weight, hop))

                    # Expand to next frontier
                    connected_ents = get_entities_for_relation(r)
                    next_frontier.update(connected_ents)

        # ✅ KEY: Rank new relations by (degree, weight) BEFORE adding
        new_relations.sort(key=lambda x: (x[1], x[2]), reverse=True)
        collected_relations.extend(new_relations)

        current_frontier = next_frontier
        hop += 1

    return collected_relations
```

**Example Execution:**

```
Hop 0 (Initial): R_0 = [hyperedge_1, hyperedge_2]
  - hyperedge_1: "Dziga Vertov directed 'In Memory of...'"
  - hyperedge_2: "Dziga Vertov married Yelizaveta Svilova"
  - visited_entities = ["Dziga Vertov", "In Memory of...", "Yelizaveta Svilova"]

Hop 1: Expand from visited_entities
  - From "Dziga Vertov":
    - hyperedge_3: "Dziga Vertov was a cinema theorist" (degree=12, weight=3)
    - hyperedge_4: "Dziga Vertov pioneered documentary film" (degree=8, weight=2)
  - From "Yelizaveta Svilova":
    - hyperedge_5: "Yelizaveta Svilova was a film editor" (degree=6, weight=1)

  # Sort by (degree, weight):
  new_relations = [hyperedge_3, hyperedge_4, hyperedge_5]
  collected_relations = [hyperedge_1, hyperedge_2, hyperedge_3, hyperedge_4, hyperedge_5]

  # Update frontier
  next_frontier = ["Cinema Theorist", "Documentary Film", "Film Editor", ...]

Hop 2: Continue expanding...
  (Repeat until max_hops=15 or target_count=30 reached)
```

**Key Innovation: Structural Ranking**

Instead of computing cosine similarity at each hop (expensive), BiG-RAG uses:

```python
ranking_score = (edge_degree, weight)

# edge_degree = degree(entity) + degree(relation)
# weight = frequency of relation occurrence in corpus
```

**Why?**
1. **100x faster** (no embedding API calls during traversal)
2. **Prevents topic drift** (central nodes = more connections = more important)
3. **Structural importance** (high-degree nodes are hubs in knowledge graph)

#### Step 4: Multi-Factor Coherence Ranking

After collecting relations from multi-hop traversal, rank by **5 factors**:

```python
def coherence_score(relation, query):
    # Factor 1: Semantic Similarity (40% weight)
    sim_score = cosine_similarity(embed(relation.desc), embed(query))

    # Factor 2: Hop Distance (20% weight)
    hop_score = 1.0 / (1 + relation.hop_distance)

    # Factor 3: Centrality (20% weight)
    centrality_score = degree(relation) / max_degree_in_graph

    # Factor 4: Extraction Confidence (10% weight)
    conf_score = relation.confidence

    # Factor 5: Entity Overlap (10% weight)
    overlap_score = |relation.entities ∩ query_entities| / |query_entities|

    # Weighted combination
    final_score = (
        0.4 * sim_score +
        0.2 * hop_score +
        0.2 * centrality_score +
        0.1 * conf_score +
        0.1 * overlap_score
    )

    return final_score
```

**Example:**

```python
# For query: "Who is the spouse of the director of 'In Memory of...'?"
#
# Relation: hyperedge_2 "Dziga Vertov married Yelizaveta Svilova"
# - sim_score: 0.85 (highly relevant to "spouse")
# - hop_score: 1.0 (hop_distance = 0, found in initial retrieval)
# - centrality_score: 0.7 (degree = 8, max_degree = 12)
# - conf_score: 0.95 (high extraction quality)
# - overlap_score: 1.0 (all query entities present)
#
# final_score = 0.4*0.85 + 0.2*1.0 + 0.2*0.7 + 0.1*0.95 + 0.1*1.0 = 0.875
```

#### Step 5: COMPLEX Query Decomposition

For **COMPLEX** queries, BiG-RAG uses **Hierarchical Decomposition**:

**Algorithm 5: Hierarchical Query Decomposition**

```python
def decompose_complex_query(q):
    # Identify dependency structure
    doc = spacy_parse(q)

    # Find nested clauses
    subordinate_clauses = [token for token in doc if token.dep in {"relcl", "acl"}]

    # Create sub-query sequence
    sub_queries = []
    for clause in subordinate_clauses:
        # Extract clause as sub-query
        sub_q = extract_clause_text(clause)
        sub_queries.append(sub_q)

    # Add final query
    sub_queries.append(q)

    return sub_queries
```

**Example:**

```python
q = "Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?"

# Decomposition:
sub_queries = [
    "Who directed 'In Memory of Sergo Ordzhonikidze'?",  # Sub-query 1
    "Who is the spouse of {answer_1}?"                   # Sub-query 2
]

# Execution:
answer_1 = adaptive_multi_hop(sub_queries[0])  # → "Dziga Vertov"
answer_2 = adaptive_multi_hop(sub_queries[1].format(answer_1=answer_1))  # → "Yelizaveta Svilova"
```

#### Step 6: Generate Answer

**Context Building:**
```python
# Select top-K relations after coherence ranking
top_relations = sorted(collected_relations, key=coherence_score, reverse=True)[:15]

# Build context
context = "\n".join([
    f"Fact {i+1}: {r.description}"
    for i, r in enumerate(top_relations)
])
```

**Prompt:**
```
Context:
Fact 1: Dziga Vertov directed "In Memory of Sergo Ordzhonikidze" in 1937.
Fact 2: Dziga Vertov married Yelizaveta Svilova in 1923.
...

Question: Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?

Answer based on the context.
```

**LLM Output:**
```
Yelizaveta Svilova
```

### Key Advantages

✅ **No training required** (zero-shot, uses pretrained LLM)
✅ **Adaptive depth** (3-15 hops based on query complexity)
✅ **Structural ranking** (100x faster than repeated embedding calls)
✅ **Multi-factor ranking** (5 factors > 1 factor similarity)
✅ **Handles complex queries** (hierarchical decomposition)

### Key Limitations

❌ **More complex implementation** (query classification, adaptive traversal, multi-factor ranking)
❌ **spaCy dependency** (requires English NLP models)
❌ **May over-traverse** (15 hops can be excessive for some queries)

---

## Side-by-Side Comparison

### Retrieval Depth Comparison

| Query Type | Graph-R1 (RL-Agentic) | HyperGraphRAG (Single-Shot) | BiG-RAG (Algorithmic) |
|------------|----------------------|----------------------------|----------------------|
| **"What is BUET?"** | Agent decides (likely 1 turn) | Fixed: 2 hops (entity → relation → entity) | SIMPLE: 3 hops max |
| **"BUET admission requirements?"** | Agent decides (likely 2 turns) | Fixed: 2 hops | MODERATE: 10 hops max |
| **"Spouse of director of film X?"** | Agent decides (likely 3 turns) | Fixed: 2 hops (may miss!) | COMPLEX: 15 hops + decomposition |

**Key Insight:**
- **Graph-R1:** Agent learns optimal depth via RL
- **HyperGraphRAG:** Always 2 hops (entity → relation → entity expansion)
- **BiG-RAG:** Linguistic analysis determines depth (3/10/15 hops)

### Vector Search vs Filter-Based Retrieval

**Common Misconception:** Graph-R1 uses "filter-based" retrieval instead of vector search.

**REALITY:** All three systems use **dual-path vector similarity search**:

```python
# Entity Retrieval (ALL THREE SYSTEMS)
R_E = argmax^k_{e in V} cosine_similarity(embed(query_entities), embed(e))

# Relation/Hyperedge Retrieval (ALL THREE SYSTEMS)
R_R = argmax^k_{r in R} cosine_similarity(embed(query), embed(r))
```

**The difference is NOT vector vs filter, but:**

| System | Retrieval Calls | When Vector Search Happens |
|--------|----------------|---------------------------|
| **Graph-R1** | Multiple (multi-turn) | At each `<query>` step (agent-driven) |
| **HyperGraphRAG** | Once | Single retrieval at start |
| **BiG-RAG** | Once | Single retrieval at start |

### Cost Comparison

**Assuming:**
- LLM: GPT-4o-mini ($0.15/1M input tokens, $0.60/1M output tokens)
- Embedding: text-embedding-3-small ($0.02/1M tokens)
- Query: "Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?"

| System | LLM Calls | Embedding Calls | Total Cost (per query) |
|--------|-----------|----------------|----------------------|
| **Graph-R1** (3 turns) | 6 (3 think + 3 query/answer) | 6 (3 entity extract + 3 vector searches) | ~$0.005 |
| **HyperGraphRAG** | 2 (1 entity extract + 1 generate) | 2 (1 entity + 1 hyperedge search) | ~$0.002 |
| **BiG-RAG** | 2 (1 entity extract + 1 generate) | 2 (1 entity + 1 relation search) | ~$0.002 |

**Training Cost:**
- **Graph-R1:** $800 one-time (72 hours A100 GPU)
- **HyperGraphRAG:** $0
- **BiG-RAG:** $0

### Accuracy Comparison (from papers)

**Dataset: 2WikiMultiHopQA (complex multi-hop queries)**

| Method | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| Standard RAG | 32.0% | - | - |
| HyperGraphRAG | 29.4% | - | - |
| BiG-RAG (Algorithmic) | ~35%* | - | - |
| Graph-R1 (RL-trained) | **57.8%** | - | - |

*Estimated based on architectural improvements over HyperGraphRAG

**Key Takeaway:** RL training (Graph-R1) achieves highest accuracy but requires significant upfront investment.

---

## Practical Examples

### Example 1: Simple Factoid Query

**Query:** "What is the capital of France?"

#### Graph-R1 Execution:
```xml
<think>
This is a simple factoid question. I should query for the capital of France.
</think>
<query>
What is the capital of France?
</query>

<knowledge>
Hyperedge_1: Paris is the capital and largest city of France.
</knowledge>

<think>
I have the answer. Paris is the capital of France.
</think>
<answer>
Paris
</answer>
```
**Turns:** 1 (agent recognized simplicity)
**Cost:** 2 LLM calls + 2 embedding calls

#### HyperGraphRAG Execution:
```python
# Step 1: Extract entities
V_q = ["France"]

# Step 2: Retrieve entities
R_V = ["France", "Paris", "French Republic"]

# Step 3: Retrieve hyperedges
R_H = [hyperedge_1: "Paris is the capital of France"]

# Step 4: Expand
F_V = {hyperedge_1, hyperedge_2: "France is in Europe", ...}
F_H = {hyperedge_1}

# Step 5: Generate
Answer: Paris
```
**Expansions:** Fixed 2 hops (may over-retrieve)
**Cost:** 2 LLM calls + 2 embedding calls

#### BiG-RAG Execution:
```python
# Step 1: Classify
complexity = "SIMPLE"  # No subordinate clauses, 1 entity
max_hops = 3
target_relations = 5

# Step 2: Retrieve
R_E = ["France"]
R_R = [hyperedge_1: "Paris is the capital of France"]

# Step 3: Adaptive traversal (stops at 3 hops or 5 relations)
collected_relations = [hyperedge_1]  # Sufficient after 1 hop

# Step 4: Generate
Answer: Paris
```
**Traversal:** Adaptive 1 hop (stopped early, target met)
**Cost:** 2 LLM calls + 2 embedding calls

**Winner for Simple Queries:** HyperGraphRAG (simplest, fast enough)

---

### Example 2: Multi-Hop Query

**Query:** "Who is the spouse of the director of 'In Memory of Sergo Ordzhonikidze'?"

#### Graph-R1 Execution:
```xml
Turn 1:
<think>
I need to find who directed the film first.
</think>
<query>
Who directed "In Memory of Sergo Ordzhonikidze"?
</query>
<knowledge>
Hyperedge_1: Dziga Vertov directed "In Memory of Sergo Ordzhonikidze" in 1937.
</knowledge>

Turn 2:
<think>
Now I know Dziga Vertov is the director. I need to find his spouse.
</think>
<query>
Who is the spouse of Dziga Vertov?
</query>
<knowledge>
Hyperedge_2: Dziga Vertov married Yelizaveta Svilova in 1923.
</knowledge>

Turn 3:
<think>
I have the answer. The spouse is Yelizaveta Svilova.
</think>
<answer>
Yelizaveta Svilova
</answer>
```
**Turns:** 3
**Cost:** 6 LLM calls + 6 embedding calls
**Accuracy:** High (agent decomposes naturally)

#### HyperGraphRAG Execution:
```python
# Step 1: Extract entities
V_q = ["In Memory of Sergo Ordzhonikidze"]

# Step 2: Retrieve entities (k_V=10)
R_V = ["In Memory of Sergo Ordzhonikidze", "Dziga Vertov", ...]

# Step 3: Retrieve hyperedges (k_H=10)
R_H = [hyperedge_1: "Dziga Vertov directed...", ...]

# Step 4: Expand
F_V = {
    hyperedge_1: "Dziga Vertov directed 'In Memory of...'",
    hyperedge_2: "Dziga Vertov married Yelizaveta Svilova",
    hyperedge_3: "Dziga Vertov was a cinema theorist",
    ... (all hyperedges connected to retrieved entities)
}

F_H = {hyperedge_1, ...}

# Step 5: Generate (LLM must reason over all expanded facts)
Answer: Yelizaveta Svilova
```
**Expansions:** Fixed 2 hops
**Cost:** 2 LLM calls + 2 embedding calls
**Risk:** May miss connection if "spouse" hyperedge not in initial retrieval or expansion

#### BiG-RAG Execution:
```python
# Step 1: Classify
complexity = "COMPLEX"  # has_subordinate=True, has_possessive=True
max_hops = 15
target_relations = 30

# Step 1b: Decompose
sub_queries = [
    "Who directed 'In Memory of Sergo Ordzhonikidze'?",
    "Who is the spouse of {answer_1}?"
]

# Sub-query 1:
R_E = ["In Memory of Sergo Ordzhonikidze"]
R_R = [hyperedge_1: "Dziga Vertov directed..."]

# Adaptive traversal (max 15 hops)
Hop 0: [hyperedge_1]
Hop 1: Expand from ["Dziga Vertov", "In Memory of..."]
  → [hyperedge_2: "Dziga Vertov was Soviet filmmaker", ...]
...

# Coherence ranking
answer_1 = "Dziga Vertov"  # Extracted from top-ranked relation

# Sub-query 2:
R_E = ["Dziga Vertov"]
R_R = [hyperedge_2: "Dziga Vertov married Yelizaveta Svilova"]

# Adaptive traversal
Hop 0: [hyperedge_2]
...

# Final answer
answer_2 = "Yelizaveta Svilova"
```
**Traversal:** Decomposed into 2 sub-queries, each with adaptive multi-hop
**Cost:** 4 LLM calls + 4 embedding calls
**Accuracy:** High (decomposition ensures logical steps)

**Winner for Multi-Hop Queries:** Graph-R1 (highest accuracy) or BiG-RAG (best balance)

---

### Example 3: Domain-Specific Query

**Query:** "What are the admission requirements for CSE at BUET?"

**Dataset:** Bangladesh university admission information (Bengali language)

#### Graph-R1 Execution:
```xml
<think>
I need to find admission requirements for Computer Science Engineering at BUET.
</think>
<query>
What are the admission requirements for CSE at BUET?
</query>

<knowledge>
Hyperedge_1: BUET CSE admission requires SSC GPA 5.00, HSC GPA 5.00, and Math A+ in both.
Hyperedge_2: BUET entrance exam has 200 MCQs (Math 60, Physics 60, Chemistry 40, English 40).
</knowledge>

<think>
I have comprehensive information about BUET CSE admission. I can answer now.
</think>
<answer>
BUET CSE admission requires:
- SSC GPA 5.00 with Math A+
- HSC GPA 5.00 with Math A+
- Pass BUET entrance exam (200 MCQs: Math 60, Physics 60, Chemistry 40, English 40)
</answer>
```
**Turns:** 1
**Advantage:** Agent can query in natural language, retrieves relevant hyperedges

#### HyperGraphRAG Execution:
```python
# Step 1: Extract entities
V_q = ["CSE", "BUET"]

# Step 2: Retrieve entities
R_V = ["BUET", "CSE", "Computer Science and Engineering"]

# Step 3: Retrieve hyperedges
R_H = [
    hyperedge_1: "BUET CSE admission requires SSC GPA 5.00, HSC GPA 5.00, Math A+",
    hyperedge_2: "BUET entrance exam format...",
    ...
]

# Step 4: Expand
F_V = {all hyperedges connected to BUET, CSE}

# Step 5: Generate
Answer: (Similar to Graph-R1 output)
```
**Expansions:** Fixed 2 hops
**Advantage:** Fast, works well for entity-focused queries

#### BiG-RAG Execution:
```python
# Step 1: Classify
complexity = "MODERATE"  # 2 entities (CSE, BUET), no subordinate clauses
max_hops = 10
target_relations = 15

# Step 2: Dual-path retrieval
R_E = ["BUET", "CSE"]
R_R = [
    hyperedge_1: "BUET CSE admission requires SSC GPA 5.00, HSC GPA 5.00, Math A+",
    hyperedge_2: "BUET entrance exam format...",
]

# Step 3: Adaptive traversal (max 10 hops)
Hop 0: [hyperedge_1, hyperedge_2]
Hop 1: Expand from ["BUET", "CSE", "SSC", "HSC", "Math A+", ...]
  → [hyperedge_3: "SSC is Secondary School Certificate", ...]
Hop 2: Continue...
  (Stops when 15 relations collected or 10 hops reached)

# Step 4: Coherence ranking
top_relations = [hyperedge_1, hyperedge_2, ...]  # Ranked by 5 factors

# Step 5: Generate
Answer: (Similar to Graph-R1 output)
```
**Traversal:** Adaptive 10 hops max (likely stops around 3-5 hops)
**Advantage:** Balances depth and efficiency

**Winner for Domain Queries:** HyperGraphRAG or BiG-RAG (both fast, no training needed)

---

## When to Use Which System

### Choose Graph-R1 If:

✅ **Highest accuracy is critical** (research, medical diagnosis, legal analysis)
✅ **You have training data** (question-answer pairs for your domain)
✅ **You have GPU infrastructure** (1x A100 for ~72 hours training)
✅ **Complex multi-hop queries are common** (50%+ of queries require 3+ reasoning steps)
✅ **Inference cost is acceptable** (multi-turn interaction = more LLM calls)

**Use Cases:**
- Medical diagnosis systems (multi-hop reasoning over symptoms → conditions → treatments)
- Legal case analysis (multi-step reasoning through precedents)
- Scientific literature QA (complex queries spanning multiple papers)

**Example:** *"What treatment is recommended for patients with condition X who are also taking medication Y?"*

---

### Choose HyperGraphRAG If:

✅ **Fast deployment is priority** (no training, works immediately)
✅ **Simple to moderate queries** (1-2 hops sufficient)
✅ **Cost-sensitive** (minimal LLM calls, minimal embedding calls)
✅ **Simplicity is valued** (easiest to implement and maintain)
✅ **Queries are entity-focused** ("What is X?", "Who is Y?")

**Use Cases:**
- Knowledge base chatbots (FAQ-style queries)
- Product information systems (entity-centric: "What are features of product X?")
- Educational content retrieval (simple factoid queries)

**Example:** *"What is the capital of France?"*, *"Who invented the telephone?"*

---

### Choose BiG-RAG If:

✅ **No training budget** (zero-shot, no RL training needed)
✅ **Mixed query complexity** (simple + moderate + complex queries)
✅ **Want adaptive depth** (automatic adjustment based on query)
✅ **Best balance** (accuracy vs cost vs complexity)
✅ **English language** (spaCy dependency parsing works best in English)

**Use Cases:**
- University admission systems (mix of simple and complex queries)
- Enterprise knowledge bases (varied query complexity)
- General-purpose RAG applications (need to handle all query types)

**Example:** *"What are BUET CSE admission requirements?"* (moderate), *"Who is the spouse of the director of film X?"* (complex)

---

## Implementation Insights

### Graph-R1 Implementation Details

**File Structure (hypothetical, based on paper):**
```
graphr1/
├── agent/
│   ├── policy.py          # LLM-based policy π_θ
│   ├── environment.py     # Hypergraph environment G_H
│   └── actions.py         # Action space (think, query, retrieve, answer)
├── training/
│   ├── grpo.py            # GRPO optimizer
│   ├── reward.py          # R(τ) = R_format + R_answer
│   └── trajectory.py      # Trajectory sampling
├── retrieval/
│   ├── dual_path.py       # Entity + hyperedge retrieval
│   └── rrf.py             # Reciprocal rank fusion
└── operate.py             # Main kg_query function
```

**Key Code Pattern:**
```python
# Agent interaction loop
def kg_query_rl(query, hypergraph, policy_model):
    state = query
    trajectory = []

    while True:
        # Agent generates action
        action = policy_model.generate(state)  # <think> + <query>/<answer>

        if action.type == "terminate":
            answer = action.content  # <answer>...</answer>
            break

        # Retrieve knowledge
        knowledge = dual_path_retrieval(
            query=action.content,  # <query>...</query>
            hypergraph=hypergraph
        )

        # Update state
        state = f"{state}\n<knowledge>{knowledge}</knowledge>"
        trajectory.append((state, action))

    return answer, trajectory
```

**Training Loop:**
```python
# RL training with GRPO
def train_graph_r1(dataset, hypergraph, epochs=10):
    policy_model = load_pretrained_llm("Qwen2.5-1.5B-Instruct")

    for epoch in range(epochs):
        for batch in dataset:
            # Sample N trajectories per question
            trajectories = []
            for n in range(N):
                answer, traj = kg_query_rl(batch.question, hypergraph, policy_model)
                reward = compute_reward(answer, batch.gold_answer, traj)
                trajectories.append((traj, reward))

            # Compute advantages
            advantages = normalize_rewards(trajectories)

            # Update policy with GRPO
            loss = grpo_loss(policy_model, trajectories, advantages)
            policy_model.backward(loss)
            policy_model.step()

    return policy_model
```

---

### HyperGraphRAG Implementation Details

**File Structure:**
```
hypergraphrag/
├── hypergraphrag.py       # Main HyperGraphRAG class
├── operate.py             # kg_query function
├── storage.py             # NanoVectorDBStorage, NetworkXStorage
├── base.py                # BaseVectorStorage, BaseGraphStorage
└── utils.py               # Embedding functions
```

**Key Code Pattern (from actual code):**
```python
async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    # Step 1: Extract entities from query
    entity_extract_prompt = PROMPTS["entity_extraction"].format(input_text=query)
    result = await llm_model_func(entity_extract_prompt)

    # Parse entities and hyperedges
    ll_keywords = []  # Entity keywords
    hl_keywords = []  # Hyperedge keywords
    # ... parsing logic ...

    # Step 2: Build context via dual-path retrieval
    context = await _build_query_context(
        [ll_keywords, hl_keywords],
        knowledge_graph_inst,
        entities_vdb,
        hyperedges_vdb,
        text_chunks_db,
        query_param,
    )

    # Step 3: Generate answer
    sys_prompt = PROMPTS["rag_response"].format(context_data=context)
    response = await llm_model_func(query, system_prompt=sys_prompt)

    return response
```

**Dual-Path Retrieval:**
```python
async def _build_query_context(
    query,  # [ll_keywords, hl_keywords]
    knowledge_graph_inst,
    entities_vdb,
    hyperedges_vdb,
    text_chunks_db,
    query_param,
):
    ll_keywords, hl_keywords = query[0], query[1]

    # Entity-based retrieval (local context)
    if query_param.mode in ["local", "hybrid"]:
        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        ) = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    # Hyperedge-based retrieval (global context)
    if query_param.mode in ["global", "hybrid"]:
        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        ) = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            hyperedges_vdb,
            text_chunks_db,
            query_param,
        )

    # Combine contexts
    if query_param.mode == "hybrid":
        context = combine_contexts(
            [ll_entities_context, hl_entities_context],
            [ll_relations_context, hl_relations_context],
            [ll_text_units_context, hl_text_units_context],
        )

    return context
```

---

### BiG-RAG Implementation Details

**File Structure:**
```
bigrag/
├── bigrag.py              # Main BiGRAG class
├── operate.py             # kg_query function
├── retrieval.py           # Dual-path + adaptive multi-hop
├── config.py              # BiGRAGConfig (complexity params)
├── entity_resolution.py   # Entity deduplication
├── storage.py             # NetworkXStorage (bipartite)
└── base.py                # BaseGraphStorage (bipartite methods)
```

**Key Code Pattern:**
```python
async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict,
):
    # Step 1: Classify query complexity
    complexity = classify_query_complexity(query)  # SIMPLE / MODERATE / COMPLEX

    # Step 2: Get adaptive parameters
    if complexity == "SIMPLE":
        max_hops = 3
        target_relations = 5
    elif complexity == "MODERATE":
        max_hops = 10
        target_relations = 15
    else:  # COMPLEX
        # Decompose query
        sub_queries = hierarchical_decompose(query)
        # Execute each sub-query recursively
        return await execute_decomposed_queries(sub_queries, ...)

    # Step 3: Dual-path initial retrieval
    initial_entities, initial_relations = await dual_path_retrieval(
        query,
        entities_vdb,
        hyperedges_vdb,
        dual_path_top_k=10
    )

    # Step 4: Adaptive multi-hop traversal
    collected_relations = await adaptive_multi_hop_traversal(
        initial_entities=initial_entities,
        initial_relations=initial_relations,
        knowledge_graph=knowledge_graph_inst,
        max_hops=max_hops,
        target_count=target_relations
    )

    # Step 5: Multi-factor coherence ranking
    ranked_relations = coherence_rank(
        collected_relations,
        query,
        factors=["similarity", "hop_distance", "centrality", "confidence", "overlap"]
    )

    # Step 6: Build context and generate
    context = build_context(ranked_relations[:15])
    response = await llm_model_func(query, system_prompt=f"Context:\n{context}")

    return response
```

**Adaptive Multi-Hop Traversal (from bigrag/retrieval.py):**
```python
async def adaptive_multi_hop_traversal(
    initial_entities: list[str],
    initial_relations: list[str],
    knowledge_graph: BaseGraphStorage,
    max_hops: int,
    target_count: int,
) -> list[dict]:
    # Initialize
    collected_relations = [
        {
            "id": r,
            "edge_degree": await knowledge_graph.get_node_degree(r),
            "weight": (await knowledge_graph.get_node(r)).get("weight", 1.0),
            "hop": 0
        }
        for r in initial_relations
    ]

    visited_entities = set(initial_entities)
    visited_relations = set(initial_relations)
    current_frontier = set(initial_entities)
    hop = 1

    # BFS traversal
    while hop <= max_hops and len(collected_relations) < target_count:
        next_frontier = set()
        new_relations = []

        # Expand from entities to relations
        for entity_id in current_frontier:
            # Get all relations connected to this entity
            neighbor_relations = await knowledge_graph.get_relations_for_entity(entity_id)

            for rel_id in neighbor_relations:
                if rel_id not in visited_relations:
                    visited_relations.add(rel_id)

                    # Compute structural ranking (NOT cosine similarity)
                    entity_degree = await knowledge_graph.get_node_degree(entity_id)
                    rel_degree = await knowledge_graph.get_node_degree(rel_id)
                    edge_degree = entity_degree + rel_degree

                    rel_node = await knowledge_graph.get_node(rel_id)
                    weight = rel_node.get("weight", 1.0)

                    new_relations.append({
                        "id": rel_id,
                        "edge_degree": edge_degree,
                        "weight": weight,
                        "hop": hop
                    })

                    # Get entities for next hop
                    connected_entities = await knowledge_graph.get_entities_for_relation(rel_id)
                    next_frontier.update(connected_entities)

        # ✅ KEY: Sort by (degree, weight) BEFORE adding
        new_relations.sort(key=lambda x: (x["edge_degree"], x["weight"]), reverse=True)
        collected_relations.extend(new_relations)

        current_frontier = next_frontier
        hop += 1

    return collected_relations
```

**Query Complexity Classification (from bigrag/operate.py):**
```python
def classify_query_complexity(query: str) -> str:
    """Classify query into SIMPLE, MODERATE, or COMPLEX."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)

    # Feature extraction
    subordinate_markers = {"who", "which", "that", "whose", "of the", "where", "when"}
    has_subordinate = any(marker in query.lower() for marker in subordinate_markers)
    has_possessive = any(token.dep_ == "poss" for token in doc)
    num_entities = len(doc.ents)

    # Clause depth
    relcl_tokens = [token for token in doc if token.dep_ in {"relcl", "acl", "ccomp"}]
    clause_depth = len(relcl_tokens)

    # Classification
    if has_subordinate and (has_possessive or clause_depth > 2):
        return "COMPLEX"
    elif num_entities > 2 or has_subordinate:
        return "MODERATE"
    else:
        return "SIMPLE"
```

---

## Conclusion

### The Three Paradigms Summarized

1. **Graph-R1 (RL-Agentic):** Train an agent to explore graphs intelligently
   - Best accuracy, requires training
   - Multi-turn interaction, adaptive depth
   - Use when accuracy > cost

2. **HyperGraphRAG (Single-Shot):** Retrieve once, expand fixed amount, generate
   - Simplest, fastest deployment
   - Fixed 2-hop expansion
   - Use when simplicity > adaptiveness

3. **BiG-RAG (Algorithmic Adaptive):** Analyze query linguistically, adapt algorithmically
   - Best balance, no training
   - Adaptive 3-15 hops based on complexity
   - Use when you need adaptiveness without training

### Key Learnings

**What They Share:**
- All use hypergraphs (n-ary relations)
- All use bipartite graph representation
- All use dual-path retrieval (entity + relation)
- All use vector similarity search
- All use reciprocal rank fusion (RRF)

**What Differentiates Them:**
- **Interaction model:** Multi-turn (Graph-R1) vs Single-shot (HyperGraphRAG, BiG-RAG)
- **Depth adaptation:** RL-learned (Graph-R1) vs Fixed (HyperGraphRAG) vs Linguistic (BiG-RAG)
- **Training requirement:** Yes (Graph-R1) vs No (HyperGraphRAG, BiG-RAG)
- **Complexity:** High (Graph-R1, BiG-RAG) vs Low (HyperGraphRAG)

### Future Directions

**Potential Hybrid Approaches:**

1. **BiG-RAG + RL:** Combine BiG-RAG's query classification with Graph-R1's multi-turn agent
2. **HyperGraphRAG + Adaptive Expansion:** Add complexity-based expansion to HyperGraphRAG
3. **Graph-R1 + Zero-Shot:** Use Graph-R1's multi-turn paradigm without RL training (prompt engineering)

---

**End of Deep Dive**

This document provides a comprehensive educational overview of Graph-R1, HyperGraphRAG, and BiG-RAG retrieval architectures. Use it as a reference for understanding the trade-offs and choosing the right system for your application.

For implementation questions, refer to the code examples and file structures provided in each section.
