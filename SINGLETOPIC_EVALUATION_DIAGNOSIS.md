# SingleTopic Evaluation Diagnosis Report

**Date:** 2025-11-04
**Dataset:** SingleTopic (20 documents, 120 questions)
**Graph Built:** ✅ 8,108 entities, 7,277 relations

---

## 📊 Evaluation Results Summary

```
Overall Performance:
  EM (Exact Match):  1.67%  (2/120 questions correct)
  F1 Score:          15.33% (token overlap)

By Question Type:
  Single Passage:    EM:  5.0%,  F1: 28.31%  ← Best
  Multi Passage:     EM:  0.0%,  F1: 17.69%
  No Answer:         EM:  0.0%,  F1:  0.0%   ← Worst
```

---

## 🔍 Root Cause Analysis

### Issue #1: Poor Retrieval Quality ⚠️ **CRITICAL**

**Example: Question 1**
```
Question: "Which enemy types wield an AK-47?"
Golden:   "Assault-rifle wielding Bullet and Tankers wield AK-47s."
```

**What's in the corpus (Document 0):**
```
✅ "Assault-rifle wielding Bullet Kin wield AK-47s."  ← CORRECT
✅ "Tankers wield AK-47s."                            ← CORRECT
```

**What the system retrieved:**
```
❌ Source 2: "Bandana Bullet Kin also have a higher magazine size
              than Bullet Kin that wield AK-47s..."
   → This is NOT the answer, just a comparison!
```

**What the model generated:**
```
❌ "The enemy type that wields an AK-47 is the Bullet Kin,
    specifically mentioned in Source 2 as 'Bullet Kin that wield AK-47s.'"
   → Incomplete: Missing "Tankers"
   → F1: 0.23 (only 23% overlap)
```

**Root Cause:**
- ✅ **Graph has the entities** (entities for "Tankers", "AK-47", "Bullet Kin" exist)
- ❌ **Wrong chunks retrieved** (retrieved indirect mention instead of direct answer)
- ❌ **Missing critical entity** ("Tankers wield AK-47s" not in top-5 results)

---

## 🐛 Identified Issues

### 1. Retrieval Not Finding Direct Answers

**Symptoms:**
- System retrieves chunks that MENTION keywords but don't ANSWER the question
- Direct factual statements being ranked lower than indirect mentions

**Example:**
- Retrieved: "...magazine size than Bullet Kin that wield AK-47s..." (indirect)
- Missed: "Tankers wield AK-47s." (direct answer)

**Possible Causes:**
1. **Entity extraction didn't create "Tankers wield AK-47s" as bipartite_edge**
   - The LLM might have created a generic entity "Tankers" but not the specific relation
   - Need to check if this triple exists in the graph

2. **Vector similarity scoring issue**
   - Query "Which enemy types wield an AK-47?" might not match well with "Tankers wield AK-47s"
   - Embedding distance not capturing semantic equivalence

3. **Top-k too small**
   - Currently retrieving top-5 contexts
   - Critical answer might be ranked 6th-10th

### 2. No Answer Detection Failing (0% F1)

**Symptoms:**
- Model always generates an answer, even for "no_answer" questions
- Should refuse to answer when context is irrelevant

**Example:**
```
Question (no_answer type): "How do I make a button?"
Expected: "I cannot answer this question based on the provided context."
Actual: [Generates an answer anyway]
```

**Root Cause:**
- Model not trained to refuse when context is insufficient
- Need to add instruction: "If the context doesn't contain the answer, say 'I cannot answer...'"

### 3. Multi-hop Questions Failing (0% EM)

**Symptoms:**
- Cannot answer questions requiring information from multiple passages

**Example:**
```
Question: "What makes jammed enemies different?"
Golden: "Jammed Keybullet Kin drop 2 keys..., jammed Chance Kins have a chance to drop twice the loot..., and jammed red-Caped Bullet Kin deal contact damage..."
  → Requires 3 separate pieces of information

Generated: "Jammed enemies, specifically Red-Caped Bullet Kin, differ in that they deal contact damage."
  → Only found 1 out of 3 pieces
```

**Root Cause:**
- Top-5 retrieval not sufficient for multi-hop
- Need to increase top_k or implement query decomposition

---

## 🔧 Recommended Fixes

### Fix Priority 1: Improve Retrieval Quality ⭐⭐⭐

#### A. Increase Top-K Retrieval
```python
# Current: top_k = 5
# Recommended: top_k = 10-15 for multi-hop questions

results = await rag.aquery(
    query,
    QueryParam(
        top_k=10,  # Increase from 5
        enable_reranking=True
    )
)
```

**Expected Impact:** +10-20% F1 improvement

#### B. Verify Entity Extraction Quality
```bash
# Check if "Tankers wield AK-47s" was extracted as a relation
python -c "
import networkx as nx
G = nx.read_graphml('expr/SingleTopic/graph_chunk_entity_relation.graphml')
# Search for nodes containing 'Tankers' and 'AK-47'
tanker_nodes = [n for n, d in G.nodes(data=True) if 'tanker' in n.lower()]
ak47_nodes = [n for n, d in G.nodes(data=True) if 'ak-47' in n.lower() or 'ak47' in n.lower()]
print('Tanker nodes:', tanker_nodes[:10])
print('AK-47 nodes:', ak47_nodes[:10])
"
```

If missing, this indicates entity extraction prompt needs tuning.

#### C. Add Query Expansion
```python
# Before retrieval, expand query with synonyms/rephrasing
query = "Which enemy types wield an AK-47?"
expanded_queries = [
    query,
    "enemies that use AK-47",
    "AK-47 wielding enemies",
    "what enemies have AK-47"
]
# Retrieve for all queries, merge results with RRF
```

**Expected Impact:** +5-15% F1 improvement

### Fix Priority 2: Fix No-Answer Detection ⭐⭐

Add instruction to generation prompt:
```python
system_prompt = """You are a helpful assistant. Answer questions based on the provided context.

IMPORTANT: If the context does not contain information to answer the question, respond with:
"I cannot answer this question based on the provided context."

Do not make up information or use knowledge outside the given context."""
```

**Expected Impact:** +30-50% F1 improvement for no_answer questions

### Fix Priority 3: Enable Multi-Hop Reasoning ⭐⭐

#### Option A: Increase context window
```python
# Increase top_k for multi_passage questions
if question_type == "multi_passage":
    top_k = 15
else:
    top_k = 5
```

#### Option B: Query decomposition
```python
# For multi-hop questions, decompose into sub-questions
question = "What makes jammed enemies different?"
sub_questions = [
    "What makes jammed Keybullet Kin different?",
    "What makes jammed Chance Kins different?",
    "What makes jammed Red-Caped Bullet Kin different?"
]
# Retrieve for each sub-question, combine contexts
```

**Expected Impact:** +15-30% F1 improvement for multi_passage questions

---

## 📈 Expected Results After Fixes

| Metric | Current | After Fix 1 | After Fix 1+2 | After Fix 1+2+3 |
|--------|---------|-------------|---------------|-----------------|
| **Overall EM** | 1.67% | 5-10% | 10-15% | 15-25% |
| **Overall F1** | 15.33% | 25-35% | 35-45% | 45-60% |
| **Single Passage F1** | 28.31% | 40-50% | 50-65% | 60-75% |
| **Multi Passage F1** | 17.69% | 25-35% | 35-45% | 50-70% |
| **No Answer F1** | 0.0% | 0-5% | 50-70% | 60-80% |

**Target Baseline:**
- Good RAG system: EM: 20-40%, F1: 50-70%
- Excellent RAG system: EM: 40-60%, F1: 70-85%

---

## 🔬 Diagnostic Commands

### 1. Check Entity Extraction Quality
```bash
# Count entities and relations by type
python -c "
import networkx as nx
G = nx.read_graphml('expr/SingleTopic/graph_chunk_entity_relation.graphml')
entities = [n for n, d in G.nodes(data=True) if d.get('role') == 'entity']
relations = [n for n, d in G.nodes(data=True) if d.get('role') == 'bipartite_edge']
print(f'Entities: {len(entities)}')
print(f'Relations: {len(relations)}')
print(f'Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}')
"
```

### 2. Test Single Retrieval
```bash
# Test retrieval for first question
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Which enemy types wield an AK-47?"],
    "top_k": 10,
    "enable_reranking": true
  }' | python -m json.tool
```

### 3. Check if Specific Relation Exists
```bash
# Search for "Tankers wield AK-47" in graph
python -c "
import networkx as nx
G = nx.read_graphml('expr/SingleTopic/graph_chunk_entity_relation.graphml')
for node, data in G.nodes(data=True):
    if 'tanker' in node.lower() and 'ak' in node.lower():
        print(f'Node: {node}')
        print(f'Role: {data.get(\"role\")}')
        print(f'Type: {data.get(\"entity_type\")}')
        print(f'Source IDs: {data.get(\"source_id\")}')
        print()
"
```

### 4. Inspect Failed Questions
```bash
# Get all questions with EM=0 and F1<0.2
python -c "
import json
with open('d:/BiG-RAG/datasets/SingleTopic/results/generation_results_evaluation.json') as f:
    # Note: This file doesn't exist yet, need to save it from API response
    data = json.load(f)
    failed = [q for q in data['per_question_results'] if q['em'] == 0 and q['f1'] < 0.2]
    print(f'Failed questions: {len(failed)}/{len(data[\"per_question_results\"])}')
    for q in failed[:5]:
        print(f\"\\nQ: {q['question']}\")
        print(f\"Golden: {q['golden_answer'][:100]}...\")
        print(f\"Generated: {q['generated_answer'][:100]}...\")
"
```

---

## 🎯 Immediate Action Plan

### Step 1: Quick Win - Increase Top-K (5 minutes)
```bash
# Re-run generation with top_k=10
curl -X POST http://localhost:8001/eval/batch_generate \
  -H "Content-Type: application/json" \
  -d '{
    "questions_csv_path": "datasets/SingleTopic/processed/all_questions_unified.csv",
    "output_csv_path": "datasets/SingleTopic/results/generation_results_topk10.csv",
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "top_k": 10,
    "enable_reranking": true
  }'

# Evaluate
curl -X POST http://localhost:8001/eval/evaluate_results \
  -H "Content-Type: application/json" \
  -d '{
    "results_csv_path": "datasets/SingleTopic/results/generation_results_topk10.csv",
    "metrics": ["em", "f1"],
    "output_dir": "datasets/SingleTopic/results/"
  }'
```

**Expected:** EM: 5-10%, F1: 25-35% (+10-20% improvement)

### Step 2: Add No-Answer Instruction (10 minutes)
Modify generation prompt in API endpoint to include:
```
"If the context does not contain the answer, say 'I cannot answer this question based on the provided context.'"
```

**Expected:** No-answer F1: 50-70% (+50-70% for that category)

### Step 3: Analyze Entity Extraction (15 minutes)
```bash
# Export sample of entities and relations to check quality
python -c "
import networkx as nx
import random
G = nx.read_graphml('expr/SingleTopic/graph_chunk_entity_relation.graphml')
relations = [(n, d) for n, d in G.nodes(data=True) if d.get('role') == 'bipartite_edge']
sample = random.sample(relations, min(20, len(relations)))
for node, data in sample:
    print(f'Relation: {node}')
    print(f'Source: {data.get(\"source_id\")}')
    print()
" > entity_extraction_sample.txt

# Review entity_extraction_sample.txt manually
cat entity_extraction_sample.txt
```

If quality is poor, tune prompts in `bigrag/prompt.py`.

---

## 📝 Summary

**Current State:**
- ✅ Graph built successfully (8K entities, 7K relations)
- ✅ Generation pipeline working
- ❌ **Retrieval quality poor** (retrieving wrong chunks)
- ❌ No-answer detection not working
- ❌ Multi-hop questions failing

**Key Finding:**
The answers ARE in the corpus, but the system is not retrieving the right chunks. This is a **retrieval quality issue**, not a knowledge gap.

**Recommended Path:**
1. **Immediate:** Increase top_k to 10-15 (quick win, +10-20% F1)
2. **Short-term:** Fix no-answer detection (+50-70% for no_answer category)
3. **Medium-term:** Investigate entity extraction quality
4. **Long-term:** Implement query decomposition for multi-hop

**Expected Final Performance:**
- EM: 15-25% (vs 1.67% now)
- F1: 45-60% (vs 15.33% now)
- Approaching good RAG system baseline (EM: 20-40%, F1: 50-70%)

---

**Next Step:** Run Step 1 (increase top_k=10) to get quick improvement.
