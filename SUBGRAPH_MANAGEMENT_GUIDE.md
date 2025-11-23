# Subgraph Management Guide
**Practical Guide: Build, Run, Search, and Update Subgraphs**
**Last Updated:** 2025-01-22

---

## Quick Overview

**What is a Subgraph?**
- KUET = one subgraph
- BUET = another subgraph
- DU = another subgraph
- football = another subgraph

**Each subgraph has:**
- Own GraphML file
- Own entity VDB
- Own relation VDB
- Own chunk VDB
- Own text chunks storage

**Complete isolation** - No data sharing between subgraphs.

---

## Directory Structure

```
expr/
├── KUET/                                    # KUET subgraph
│   ├── graph_chunk_entity_relation.graphml # Graph structure
│   ├── vdb_entities.json                   # Entity embeddings
│   ├── vdb_relations.json                  # Relation embeddings
│   ├── vdb_chunks.json                     # Chunk embeddings
│   ├── kv_store_full_docs.json             # Full documents
│   └── kv_store_text_chunks.json           # Text chunks
│
├── BUET/                                    # BUET subgraph
│   ├── graph_chunk_entity_relation.graphml
│   ├── vdb_entities.json
│   └── ... (same structure)
│
├── DU/                                      # DU subgraph
│   └── ...
│
├── football/                                # Football subgraph
│   └── ...
│
└── subgraph_registry.json                   # Registry of all subgraphs
```

---

## Part 1: Building Subgraphs

### Step 1: Prepare Documents

Create dataset directory:
```bash
mkdir -p datasets/KUET/raw
# Add your documents
cp KUET_Admission_info.md datasets/KUET/raw/
```

### Step 2: Build Subgraph

```bash
# Build KUET subgraph
python script_build.py \
  --data_source KUET \
  --input datasets/KUET/raw \
  --output expr/KUET

# Build BUET subgraph
python script_build.py \
  --data_source BUET \
  --input datasets/BUET/raw \
  --output expr/BUET

# Build football subgraph
python script_build.py \
  --data_source football \
  --input datasets/football/raw \
  --output expr/football
```

**What this does:**
1. Chunks documents (semantic chunking)
2. Extracts entities + relations (with within-chunk linking)
3. Applies entity canonicalization
4. Builds VDB indices
5. Saves all files to `expr/{subgraph_name}/`
6. Updates `subgraph_registry.json`

### Step 3: Verify Subgraph Built Successfully

```bash
# Check files exist
ls expr/KUET/
# Should see: graph_chunk_entity_relation.graphml, vdb_*.json, kv_store_*.json

# Check stats
python check_subgraph.py --subgraph KUET
# Output: 450 entities, 380 relations, 12 chunks
```

---

## Part 2: Running the Server

### Option 1: Single Subgraph Mode (Current Approach)

Run server for ONE specific subgraph:

```bash
# Run server for KUET only
cd backend
python server.py --data_source KUET

# Server loads: expr/KUET/
# API available at: http://localhost:8001
```

**Use case:** When you only have one subgraph, or testing specific subgraph.

---

### Option 2: Unified Mode (NEW - Multi-Subgraph)

Run server that can search ALL subgraphs:

```bash
# Run unified server
cd backend
python server.py --unified

# Server loads subgraph_registry.json
# Can query any subgraph: KUET, BUET, DU, football
# API available at: http://localhost:8001
```

**What happens:**
1. Server reads `expr/subgraph_registry.json`
2. Discovers all available subgraphs (KUET, BUET, DU, football)
3. Lazy loads subgraphs (only when queried)
4. Routes queries to relevant subgraphs using LLM agent

---

## Part 3: How LLM Agent Decides Which Subgraph to Search

### Subgraph Registry

**Location:** `expr/subgraph_registry.json`

**Example:**
```json
{ 
  "version": "1.0",
  "subgraphs": {
    "KUET": {
      "full_name": "Khulna University of Engineering and Technology",
      "full_name_bn": "খুলনা প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয়",
      "aliases": ["KUET", "খুলনা বিশ্ববিদ্যালয়", "Khulna Engineering"],
      "departments": ["CSE", "EEE", "ME", "CE"],
      "topics": ["admission", "seats", "departments", "eligibility"],
      "subgraph_path": "expr/KUET"
    },
    "BUET": {
      "full_name": "Bangladesh University of Engineering and Technology",
      "full_name_bn": "বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়",
      "aliases": ["BUET", "বুয়েট"],
      "departments": ["ARCH", "CE", "EEE"],
      "topics": ["admission", "seats", "departments"],
      "subgraph_path": "expr/BUET"
    },
    "football": {
      "full_name": "Football Knowledge Base",
      "aliases": ["football", "soccer", "ফুটবল"],
      "topics": ["players", "teams", "world cup", "leagues"],
      "subgraph_path": "expr/football"
    }
  }
}
```

### Routing Algorithm (How LLM Decides)

**Step 1: User sends query**
```
Query: "কুয়েটে CSE তে কতটি আসন আছে?"
```

**Step 2: LLM Router receives query + subgraph registry**

Router prompt:
```
You are a query routing agent. Analyze the query and select relevant subgraphs.

AVAILABLE SUBGRAPHS:
- KUET (খুলনা বিশ্ববিদ্যালয়): Departments: CSE, EEE, ME, CE. Topics: admission, seats
- BUET (বুয়েট): Departments: ARCH, CE, EEE. Topics: admission, seats
- football: Topics: players, teams, world cup

USER QUERY: "কুয়েটে CSE তে কতটি আসন আছে?"

TASK: Which subgraph(s) should be searched?

OUTPUT (JSON):
{
  "subgraphs": ["KUET"],
  "reasoning": "Query mentions KUET (কুয়েট) and CSE department"
}
```

**Step 3: Router returns decision**
```json
{
  "subgraphs": ["KUET"],
  "reasoning": "Query specifically mentions KUET and CSE department"
}
```

**Step 4: Server queries KUET subgraph only**
```
Load: expr/KUET/
Search: KUET graph for CSE seats
Return: "120 seats"
```

### Routing Examples

**Example 1: Single Subgraph Query**
```
Query: "How many seats in KUET CSE?"
Router decision: ["KUET"]
Reasoning: Query mentions KUET
```

**Example 2: Comparative Query (Multiple Subgraphs)**
```
Query: "Compare KUET and BUET CSE seat counts"
Router decision: ["KUET", "BUET"]
Reasoning: Query mentions both KUET and BUET
```

**Example 3: General Query (All Subgraphs)**
```
Query: "Which universities offer CSE in Bangladesh?"
Router decision: ["KUET", "BUET", "DU"]
Reasoning: General query about all universities
```

**Example 4: Different Domain**
```
Query: "Who won the 2022 World Cup?"
Router decision: ["football"]
Reasoning: Query is about football, not universities
```

**Example 5: Hallucination Prevention**
```
Query: "How many departments does Messi have?"
Router decision: ["football"]
Reasoning: Query mentions Messi (footballer)

Result: "Lionel Messi is a footballer, not a university..."
NO CONFUSION: Router does NOT select KUET/BUET (different domain)
```

---

## Part 4: Querying Subgraphs

### API Endpoint 1: Single Subgraph Query (Manual Selection)

**Current approach** - Directly query specific subgraph:

```bash
# Query KUET subgraph
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["How many seats in CSE?"],
    "subgraph": "KUET"
  }'
```

**Use case:** When you know exactly which subgraph to query (no routing needed).

---

### API Endpoint 2: Unified Query (Automatic Routing)

**NEW** - Let LLM router decide which subgraph(s):

```bash
# Query with automatic routing
curl -X POST http://localhost:8001/api/unified/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "কুয়েটে CSE তে কতটি আসন আছে?",
    "language": "Bangla"
  }'

# Response includes routing decision
{
  "query": "কুয়েটে CSE তে কতটি আসন আছে?",
  "routing": {
    "subgraphs": ["KUET"],
    "reasoning": "Query mentions KUET and CSE"
  },
  "results": [
    {
      "content": "KUET CSE department has 120 seats",
      "subgraph": "KUET",
      "type": "relation",
      "coherence": 0.95
    }
  ]
}
```

---

### API Endpoint 3: Force Multiple Subgraphs

Override router, query specific subgraphs:

```bash
# Force query KUET and BUET (ignore router)
curl -X POST http://localhost:8001/api/unified/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many CSE seats?",
    "force_subgraphs": ["KUET", "BUET"]
  }'
```

**Use case:** Admin/testing, or when you want specific subgraphs regardless of query.

---

### API Endpoint 4: List Available Subgraphs

```bash
# Get all available subgraphs
curl http://localhost:8001/api/unified/subgraphs

# Response
{
  "subgraphs": ["KUET", "BUET", "DU", "football"],
  "master_map": {
    "KUET": {
      "full_name": "Khulna University of Engineering and Technology",
      "entity_count": 450,
      "relation_count": 380
    },
    ...
  }
}
```

---

## Part 5: Updating Subgraphs

### Scenario 1: Add New Document to Existing Subgraph

**Example: New KUET document (2025-2026 admission info)**

```bash
# Step 1: Add new document to dataset
cp KUET_Admission_2025.md datasets/KUET/raw/

# Step 2: Rebuild KUET subgraph (full rebuild)
python script_build.py \
  --data_source KUET \
  --input datasets/KUET/raw \
  --output expr/KUET \
  --force  # Overwrite existing

# Step 3: Restart server to reload subgraph
# (Or use hot-reload API endpoint)
curl -X POST http://localhost:8001/api/unified/reload?subgraph=KUET
```

**Note:** Full rebuild is safest approach. Incremental updates can be added later.

---

### Scenario 2: Create New Subgraph

**Example: Add Chittagong University (CUET)**

```bash
# Step 1: Prepare dataset
mkdir -p datasets/CUET/raw
cp CUET_info.md datasets/CUET/raw/

# Step 2: Build subgraph
python script_build.py \
  --data_source CUET \
  --input datasets/CUET/raw \
  --output expr/CUET

# Step 3: Subgraph registry auto-updated (expr/subgraph_registry.json)
# Step 4: Restart server to discover new subgraph
# (Or use discovery API)
curl -X POST http://localhost:8001/api/unified/discover
```

**Result:** Router now routes queries mentioning CUET to CUET subgraph.

---

### Scenario 3: Delete Subgraph

```bash
# Step 1: Remove subgraph directory
rm -rf expr/OLD_SUBGRAPH/

# Step 2: Update master map (remove entry)
python manage_subgraphs.py --remove OLD_SUBGRAPH

# Step 3: Restart server
# Router no longer routes to OLD_SUBGRAPH
```

---

## Part 6: Server Startup Modes

### Mode 1: Single Subgraph (Simple)

```bash
# Start server for one subgraph
cd backend
python server.py --data_source KUET

# Loads: expr/KUET/
# API: http://localhost:8001
# Queries: Only search KUET
```

**Use when:**
- Testing specific subgraph
- Only have one subgraph
- Don't need routing

---

### Mode 2: Unified (Multi-Subgraph with Routing)

```bash
# Start unified server
cd backend
python server.py --unified

# Loads: subgraph_registry.json
# Discovers: All subgraphs (KUET, BUET, DU, football)
# API: http://localhost:8001
# Queries: Router selects relevant subgraphs
```

**Use when:**
- Have multiple subgraphs
- Want automatic routing
- Production system

---

### Mode 3: Unified with Specific Subgraphs

```bash
# Start server with only selected subgraphs
cd backend
python server.py --unified --subgraphs KUET BUET DU

# Loads: Only KUET, BUET, DU (ignores football)
# Router only considers these 3 subgraphs
```

**Use when:**
- Have many subgraphs but only want subset active
- Separate servers for different categories (universities vs sports)

---

## Part 7: Monitoring & Management

### Check Subgraph Status

```bash
# List all subgraphs
python manage_subgraphs.py --list

# Output:
# KUET (expr/KUET) - 450 entities, 380 relations, 12 chunks
# BUET (expr/BUET) - 520 entities, 445 relations, 15 chunks
# football (expr/football) - 1200 entities, 980 relations, 45 chunks
```

### Check Server Status

```bash
# Check which subgraphs are loaded
curl http://localhost:8001/api/unified/status

# Response:
{
  "mode": "unified",
  "loaded_subgraphs": ["KUET", "BUET", "DU"],
  "total_subgraphs": 4,
  "master_map_version": "1.0"
}
```

### Check Routing Decisions (Debug)

```bash
# Test routing without executing query
curl -X POST http://localhost:8001/api/unified/route \
  -H "Content-Type: application/json" \
  -d '{
    "query": "কুয়েটে CSE তে কতটি আসন আছে?"
  }'

# Response:
{
  "query": "কুয়েটে CSE তে কতটি আসন আছে?",
  "selected_subgraphs": ["KUET"],
  "reasoning": "Query mentions KUET and CSE department",
  "confidence": 0.95
}
```

---

## Part 8: Migration from Current System

### Current Setup (demo_test)

```bash
# Current command
cd backend
python server.py --data_source demo_test

# Loads: expr/demo_test/
```

### Migration Option A: Rename to First Subgraph

```bash
# If demo_test contains KUET data only
mv expr/demo_test expr/KUET

# Update server command
cd backend
python server.py --data_source KUET
```

### Migration Option B: Add Federated System

```bash
# Keep demo_test, add new subgraphs
python script_build.py --data_source BUET ...
python script_build.py --data_source DU ...

# Create subgraph_registry.json (manual or script)
# Switch to unified mode
cd backend
python server.py --unified
```

---

## Part 9: Query Flow Diagram

### Single Subgraph Mode
```
User Query: "How many CSE seats?"
    ↓
Server (--data_source KUET)
    ↓
Load KUET subgraph only
    ↓
Search KUET graph
    ↓
Return results
```

### Unified Mode
```
User Query: "Compare KUET and BUET CSE seats"
    ↓
Server (--unified)
    ↓
LLM Router analyzes query + subgraph_registry
    ↓
Router decision: ["KUET", "BUET"]
    ↓
Load KUET and BUET subgraphs (parallel)
    ↓
Query both graphs (parallel)
    ↓
Aggregate results
    ↓
Return combined results
```

---

## Part 10: Common Commands Summary

```bash
# BUILD subgraph
python script_build.py --data_source KUET --input datasets/KUET/raw --output expr/KUET

# RUN server (single subgraph)
cd backend && python server.py --data_source KUET

# RUN server (unified)
cd backend && python server.py --unified

# QUERY single subgraph (API)
curl -X POST http://localhost:8001/search -d '{"queries": ["..."], "subgraph": "KUET"}'

# QUERY unified (API)
curl -X POST http://localhost:8001/api/unified/query -d '{"query": "..."}'

# LIST subgraphs
python manage_subgraphs.py --list

# UPDATE subgraph (rebuild)
python script_build.py --data_source KUET --force

# DELETE subgraph
rm -rf expr/OLD_SUBGRAPH/ && python manage_subgraphs.py --remove OLD_SUBGRAPH

# RELOAD subgraph (hot reload)
curl -X POST http://localhost:8001/api/unified/reload?subgraph=KUET
```

---

## Summary

**Key Points:**
1. **One subgraph = One directory** (KUET, BUET, football)
2. **Each subgraph has complete files** (GraphML, VDB, KV storage)
3. **Subgraph registry lists all subgraphs** (for routing)
4. **LLM router decides which subgraphs to search** (based on query + registry)
5. **Two server modes:**
   - Single: `python server.py --data_source KUET`
   - Unified: `python server.py --unified`
6. **Queries routed automatically** in unified mode
7. **Complete isolation** - No cross-subgraph contamination

**Next Steps:**
1. Build your first subgraph (KUET)
2. Test single subgraph mode
3. Build second subgraph (BUET)
4. Create subgraph_registry.json
5. Switch to unified mode
6. Test routing with different queries
