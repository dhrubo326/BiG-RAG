# Quick Subgraph FAQ
**Answers to Your Specific Questions**

---

## Question 1: How to Build Subgraphs Effectively?

### **Answer: Each Subgraph is a Separate Directory**

```
expr/
├── KUET/                    # First subgraph
│   ├── graph_chunk_entity_relation.graphml  ← One GraphML per subgraph
│   ├── vdb_entities.json
│   ├── vdb_relations.json
│   ├── vdb_chunks.json
│   ├── kv_store_full_docs.json
│   └── kv_store_text_chunks.json
│
├── BUET/                    # Second subgraph
│   ├── graph_chunk_entity_relation.graphml  ← Separate GraphML
│   ├── vdb_entities.json
│   └── ... (complete set of files)
│
├── football/                # Third subgraph (different domain)
│   ├── graph_chunk_entity_relation.graphml  ← Another separate GraphML
│   └── ...
│
└── master_map.json          # Registry of all subgraphs
```

**Key Points:**
- ✅ **One subgraph = One directory**
- ✅ **Each subgraph has its OWN GraphML file** (not shared!)
- ✅ **Each subgraph has its OWN VDB indices** (not shared!)
- ✅ **Complete isolation** between subgraphs

---

## Question 2: Is "demo_test" a Subgraph? What About "football"?

### **Answer: YES - Both are Subgraphs**

**Current System:**
```
expr/
└── demo_test/               # This IS a subgraph
    ├── graph_chunk_entity_relation.graphml
    ├── vdb_entities.json
    └── ...
```

**Future System (Multiple Subgraphs):**
```
expr/
├── demo_test/               # Subgraph 1 (rename to KUET or keep as demo)
├── football/                # Subgraph 2 (new)
├── BUET/                    # Subgraph 3 (new)
└── master_map.json          # NEW: Registry
```

**Migration Options:**

### **Option A: Rename demo_test to KUET**
```bash
# If demo_test contains only KUET data
mv expr/demo_test expr/KUET
```

### **Option B: Keep demo_test, Add New Subgraphs**
```bash
# demo_test stays as-is
# Add new subgraphs alongside
mkdir expr/football
mkdir expr/BUET
```

### **Option C: Rebuild Everything**
```bash
# Start fresh with clean subgraph names
rm -rf expr/demo_test
python script_build_subgraph.py --subgraph_name KUET ...
python script_build_subgraph.py --subgraph_name football ...
```

**My Recommendation:** Option A (rename demo_test → KUET) if it contains KUET data only.

---

## Question 3: How Many GraphML Files?

### **Answer: ONE GraphML per Subgraph**

```
expr/
├── KUET/
│   └── graph_chunk_entity_relation.graphml    ← KUET's graph
├── BUET/
│   └── graph_chunk_entity_relation.graphml    ← BUET's graph (separate!)
└── football/
    └── graph_chunk_entity_relation.graphml    ← Football's graph (separate!)
```

**NOT like this:**
```
expr/
├── graph_chunk_entity_relation.graphml        ❌ WRONG (shared file)
├── KUET/
├── BUET/
└── football/
```

---

## Question 4: How Does "demo_test" Fit In?

### **Current Situation:**

Your current `demo_test` dataset works like this:

```
datasets/demo_test/
├── raw/
│   └── corpus.jsonl         # Your source documents

expr/demo_test/              # Built knowledge graph
├── graph_chunk_entity_relation.graphml
├── vdb_entities.json
└── ...
```

**This IS a subgraph!** It's just not part of a federated system yet.

### **Transition to Federated System:**

**Step 1: Identify What demo_test Contains**
- Is it KUET data only? → Rename to `KUET`
- Is it mixed data? → Split into multiple subgraphs
- Is it test/experimental? → Keep as `demo_test` subgraph

**Step 2: Add to Master Map**
```json
{
  "subgraphs": {
    "demo_test": {
      "full_name": "Demo Test Dataset",
      "category": "test",
      "subgraph_path": "expr/demo_test"
    }
  }
}
```

**Step 3: Query Using Federated System**
```python
# Old way (direct)
rag = BiGRAG(working_dir="expr/demo_test")
results = await rag.aquery("...")

# New way (federated)
executor = FederatedQueryExecutor(federated_root="expr", ...)
results = await executor.query("...")  # Router picks demo_test
```

---

## Question 5: Creating "football" Subgraph Alongside demo_test

### **Steps:**

**1. Create Football Dataset:**
```bash
mkdir -p datasets/football/raw
# Add football documents (Messi bio, World Cup, etc.)
cp messi_biography.md datasets/football/raw/
cp world_cup_2022.md datasets/football/raw/
```

**2. Build Football Subgraph:**
```bash
python script_build_subgraph.py \
  --subgraph_name football \
  --input_dir datasets/football/raw \
  --output_dir expr/football
```

**3. Result:**
```
expr/
├── demo_test/               # Existing subgraph (untouched)
│   ├── graph_chunk_entity_relation.graphml
│   └── ...
│
├── football/                # NEW subgraph
│   ├── graph_chunk_entity_relation.graphml   ← Separate GraphML
│   ├── vdb_entities.json                     ← Separate VDB
│   └── ...
│
└── master_map.json          # NEW: Contains both subgraphs
```

**4. Master Map Content:**
```json
{
  "subgraphs": {
    "demo_test": {
      "full_name": "Demo Test Dataset",
      "subgraph_path": "expr/demo_test"
    },
    "football": {
      "full_name": "Football Knowledge Base",
      "subgraph_path": "expr/football"
    }
  }
}
```

**5. Query Both:**
```python
# Query demo_test only
results = await executor.query("KUET CSE seats")
# Router selects: ["demo_test"]

# Query football only
results = await executor.query("Who won 2022 World Cup?")
# Router selects: ["football"]

# No cross-contamination!
```

---

## Question 6: How to Manage Multiple Subgraphs?

### **Subgraph Operations:**

**List All Subgraphs:**
```bash
python manage_subgraphs.py --list

Output:
Available subgraphs:
- demo_test (expr/demo_test) - 450 entities, 380 relations
- football (expr/football) - 1200 entities, 980 relations
```

**Build New Subgraph:**
```bash
python script_build_subgraph.py \
  --subgraph_name NewSubgraph \
  --input_dir datasets/NewSubgraph/raw \
  --output_dir expr/NewSubgraph
```

**Update Existing Subgraph:**
```bash
# Add new documents to datasets/football/raw/
# Rebuild football subgraph (full rebuild)
python script_build_subgraph.py \
  --subgraph_name football \
  --input_dir datasets/football/raw \
  --output_dir expr/football \
  --force  # Overwrite existing
```

**Delete Subgraph:**
```bash
python manage_subgraphs.py --delete OldSubgraph

# Removes:
# - expr/OldSubgraph/ directory
# - Entry from master_map.json
```

---

## Visual Summary

### **Before (Single Graph):**
```
User Query: "KUET CSE seats"
    ↓
BiGRAG(working_dir="expr/demo_test")
    ↓
Searches entire graph
    ↓
May return mixed results (KUET + BUET + other data)
```

### **After (Federated Subgraphs):**
```
User Query: "KUET CSE seats"
    ↓
AgenticRouter analyzes master_map.json
    ↓
Selects relevant subgraph: ["KUET"]
    ↓
FederatedExecutor loads only KUET subgraph
    ↓
Searches ONLY KUET graph
    ↓
Returns KUET-specific results (no BUET/RUET contamination)
```

---

## Key Takeaways

1. **Each subgraph = Separate directory with complete files**
   - Own GraphML, VDB indices, KV storage

2. **demo_test IS a subgraph**
   - Just need to add it to master_map.json
   - Can rename to KUET if it contains KUET data

3. **football is another subgraph**
   - Built separately in expr/football/
   - Own GraphML, completely isolated from demo_test

4. **Master map is the registry**
   - Lists all subgraphs
   - Used by router to select relevant subgraphs

5. **No shared data between subgraphs**
   - KUET CSE ≠ BUET CSE (separate graphs)
   - Zero hallucination risk

---

## Next Action

**Should I proceed with:**

1. ✅ Updating FEDERATED_SUBGRAPH_PLAN.md (remove entity-entity edges) ← DONE
2. ✅ Creating SUBGRAPH_MANAGEMENT_GUIDE.md ← DONE
3. ✅ Creating this FAQ ← DONE
4. ⏳ Implementing `script_build_subgraph.py` (subgraph builder)?
5. ⏳ Implementing within-chunk entity-relation linking (Phase 1)?

**Let me know which to start with!**
