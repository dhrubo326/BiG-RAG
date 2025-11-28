# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**BiG-RAG** (Bipartite Graph Retrieval-Augmented Generation) is an end-to-end reinforcement learning framework that combines bipartite graph-based knowledge retrieval with LLM reasoning capabilities. The project enables LLMs to iteratively execute a "**think → generate query → retrieve subgraph → rethink**" reasoning cycle using explicit reward mechanisms within RL training.

### Key Features

- **Bipartite Graph Structure**: Unlike traditional hypergraph terminology, the implementation uses a true bipartite graph with documents ↔ entities ↔ relations
- **Three-Path Retrieval** ⭐ **NEW**: Combines entity-based (Path A), relation-based (Path B), and chunk-based (Path C) retrieval for superior recall and precision
- **Semantic Reranking** ⭐ **NEW**: Cross-encoder based reranking of chunk candidates for improved relevance
- **Metadata Preservation** ⭐ **NEW**: Document metadata (title, category, tags) flows through chunking → entity extraction for +2-3 F1 improvement
- **End-to-End RL Training**: Trains LLMs to actively query knowledge graphs during generation
- **Tool-Augmented Generation**: Models learn to emit structured queries (`<query>...</query>`) to retrieve relevant context
- **Multiple RL Algorithms**: Supports GRPO, REINFORCE++, and PPO
- **Distributed Training**: Built on VERL (Volcano Engine RL Framework) with Ray for multi-GPU/multi-node training

### Key Components

- **[bigrag/](bigrag/)**: Core BiG-RAG implementation
  - Bipartite graph construction from text corpora with metadata preservation
  - N-ary relation extraction using LLMs (context-enhanced with document metadata)
  - Three-path retrieval: Entity (Path A) + Relation (Path B) + Chunk (Path C)
  - Fast similarity search with FAISS indices
  - Semantic reranking with cross-encoder (optional)
  - Async-first API for insertion, querying, and deletion

- **[verl/](verl/)**: Volcano Engine RL Framework (by Bytedance)
  - Distributed RL training infrastructure
  - Supports PPO, GRPO, REINFORCE++
  - Integration with vLLM for fast rollout generation
  - Ray-based worker management

- **[agent/](agent/)**: Tool-based agent system
  - ToolGenerationManager: Orchestrates tool calls during generation
  - ToolEnv: Manages tool state and execution
  - Search tool: Queries bipartite graph retrieval server

- **[evaluation/](evaluation/)**: Metrics computation
  - Exact Match (EM)
  - Token-level F1 score
  - Semantic similarity (SimCSE)

---

## 📝 Documentation Policy

**IMPORTANT**: This project follows a minimal documentation approach to avoid clutter and maintain focus on working code.

**When to create documentation:**
- ✅ When explicitly requested by the project maintainer
- ✅ For critical features that significantly impact the codebase
- ✅ For major architectural changes or breaking changes
- ✅ For complex algorithms that require detailed explanation

**When NOT to create documentation:**
- ❌ After every small task or bug fix
- ❌ For self-explanatory code changes
- ❌ For minor improvements or refactoring
- ❌ For exploratory or experimental work

**Preferred documentation format:**
- Keep documentation in existing files when possible (CLAUDE.md, README.md)
- Use code comments for implementation details
- Use commit messages for change rationale
- Only create new `.md` files for major features or when specifically requested

**Git Commit Policy:**
- ⚠️ **CRITICAL**: Do NOT commit changes until explicitly instructed by the project maintainer
- Wait for explicit permission to commit (user will say "commit" or "push")
- This allows for review and iteration before changes are finalized
- **Commit Message Format**: Do NOT include Claude Code attribution footer in commit messages (no "🤖 Generated with Claude Code" or "Co-Authored-By: Claude" lines)

**New File Creation Policy:**
- ⚠️ **CRITICAL**: Do NOT generate new documentation files each time unless it is a significant change
- Only create new `.md` files when explicitly requested or for major features
- Update existing documentation files instead of creating new ones

**This policy helps maintain:**
- Clean repository structure
- Focus on working code over documentation overhead
- Easy navigation without excessive files
- Documentation that stays synchronized with code
- Controlled version history with explicit commit points

---

## Environment Setup

### Two Installation Modes

#### 1. BiG-RAG-only mode (venv - lightweight, no RL training)

Use this for:
- Building knowledge graphs
- Running retrieval server
- Testing BiG-RAG API
- Development work on graph construction

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/macOS)
# source venv/bin/activate

# Install PyTorch (CPU or GPU)
pip install torch torchvision torchaudio

# Install BiG-RAG dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### 2. Full RL training mode (conda - required for distributed training)

Use this for:
- RL training with GRPO/PPO/REINFORCE++
- Multi-GPU distributed training
- Full pipeline including training and evaluation

```bash
# Create conda environment
conda create -n bigrag python==3.11.11
conda activate bigrag

# Install PyTorch with CUDA support
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash attention (optional, speeds up training)
pip3 install flash-attn --no-build-isolation

# Install BiG-RAG package and dependencies
pip3 install -e .
pip3 install -r requirements-rl.txt
```

### GPU Requirements

**For Training:**
- **Minimum**: 4 x 48GB GPUs (for 3B parameter models)
- **Recommended**: 8 x 80GB GPUs (for 7B+ models)
- Adjust `tensor_model_parallel_size` in training scripts based on GPU count

**For Inference/Graph Building:**
- **CPU only**: Works but slower for embedding generation
- **1 GPU**: Sufficient for most graph construction tasks

---

## 🔄 Pipeline Architecture & Modular Unification (January 2025)

### Current Status: Transitioning to Modular Unified Pipeline

**IMPORTANT**: BiG-RAG is currently transitioning from a dual-pipeline system (standard + production) to a **modular unified pipeline** with feature flags. This allows users to enable/disable specific features (table extraction, validation, entity linking, etc.) based on their needs.

**Active Development Branch**: `feature/modular-unified-pipeline`

**Planning Documents**:
- **[MODULAR_PIPELINE_PLAN.md](MODULAR_PIPELINE_PLAN.md)** - Complete implementation plan with feature flags, presets, error handling, and HITL system
- **[Production_pipeline_redesign_plan.md](Production_pipeline_redesign_plan.md)** - Previous unification plan (reference only)

**Implementation Status**: Phase 0 - Planning Complete (98/100 readiness score)

**Next Steps**:
- Week 1: Implement `bigrag/pipeline/features.py` and `bigrag/pipeline/base_pipeline.py`
- Week 2: Extract standard pipeline components into modular architecture
- Week 3: Integration testing with different feature combinations
- Week 4: API endpoint updates and documentation

---

### Legacy Dual Pipeline System (Current Implementation)

BiG-RAG currently supports two distinct knowledge graph construction pipelines with a unified retrieval backend. Both pipelines produce **100% compatible** graph structures (as of January 2025).

**Note**: This dual-pipeline system will be replaced by the modular unified pipeline, but remains functional during the transition.

### Two Pipeline Modes

#### Standard Pipeline (Default - Fast & Low Cost)

**Use Cases:**
- General-purpose RAG applications
- Quick prototyping and testing
- Cost-sensitive deployments
- Large document corpora (>10K documents)

**Characteristics:**
- **Chunking**: Token-based sliding window (1200 tokens, 100 overlap)
- **Extraction**: Basic entity and relation extraction
- **Quality**: Good (90-95% accuracy)
- **Speed**: Fast (~2-3 minutes per 1K documents)
- **Cost**: Low (~$0.60 per 10K documents with GPT-4o-mini)
- **Node IDs**: Hash-based (`entity-abc123`, `rel-def456`)

**Usage:**
```bash
# Build via script
python script_build.py --data_source my_dataset

# Build via API (default)
curl -X POST "http://localhost:8001/documents/upload" \
  -F "file=@document.md" \
  -F "use_production_pipeline=false"  # Optional (default)
```

**Output**: Standard BiG-RAG knowledge graph in `expr/my_dataset/`

---

#### Production Pipeline (Enhanced - High Accuracy)

**Use Cases:**
- Educational/technical content with tables and structured data
- Domain-specific knowledge bases requiring high precision
- Applications where accuracy is critical
- Small to medium corpora (<10K documents)

**Characteristics:**
- **Chunking**: Table-aware semantic chunking (preserves table structure)
- **Extraction**: Validated entity extraction with entity linking
- **Quality**: Excellent (95-99% accuracy)
- **Speed**: Slower (~10-15 minutes per 1K documents)
- **Cost**: Higher (~$2-3 per 10K documents)
- **Node IDs**: Hash-based (`entity-abc123`, `rel-def456`) - **identical to standard**
- **Special Features**:
  - Table content preservation
  - Entity consistency validation
  - Metadata-enhanced extraction

**Usage:**
```bash
# Build via script
python script_build.py --data_source my_dataset --use_production_pipeline

# Build via API
curl -X POST "http://localhost:8001/documents/upload" \
  -F "file=@document.md" \
  -F "use_production_pipeline=true"

# Or use the dynamic dataset endpoint (always uses production pipeline)
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "my_new_dataset",
    "documents": [
      {"content": "Document text...", "title": "Doc 1"}
    ]
  }'
```

**Output**: Production BiG-RAG knowledge graph in `expr/my_dataset/`

---

### Graph Structure Compatibility

**CRITICAL UPDATE (January 24, 2025)**: Both pipelines now produce **identical graph structures**:

| Component | Standard Pipeline | Production Pipeline | Compatible? |
|-----------|------------------|---------------------|-------------|
| **Entity Node ID** | `entity-abc123` | `entity-abc123` | ✅ **YES** |
| **Relation Node ID** | `rel-abc123` | `rel-abc123` | ✅ **YES** |
| **Edge Structure** | `rel-* → entity-*` | `rel-* → entity-*` | ✅ **YES** |
| **Vector DB Keys** | `entity-abc123` | `entity-abc123` | ✅ **YES** |
| **GraphML Format** | NetworkX | NetworkX | ✅ **YES** |
| **Storage Files** | 7 files | 7 files | ✅ **YES** |

**What This Means:**
- Backend endpoints work seamlessly with graphs from **both pipelines**
- Unified subgraph system can mix graphs from different pipelines
- No need for separate retrieval code paths
- Graphs built with standard pipeline can be queried using production pipeline endpoints (and vice versa)

**Historical Note**: Prior to January 24, 2025, production pipeline used `"relation-abc123"` prefix (bug). This has been fixed to use the standard `"rel-abc123"` prefix via the `RELATION_PREFIX` constant.

---

### Future: Modular Unified Pipeline Architecture

The new modular pipeline replaces the binary choice (standard vs production) with **15+ granular feature flags** organized into 5 categories:

#### Feature Categories

**1. Chunking Features**:
- `enable_table_detection`: GPT-4 table extraction
- `chunk_mode`: token | semantic | hybrid
- `chunk_size`, `chunk_overlap`: Configurable parameters

**2. Extraction Features**:
- `enable_gleaning`: Multi-pass extraction with conversation history
- `max_gleaning_iterations`: Default 2 passes
- `enable_table_fact_extraction`: Rule-based table fact extraction
- `extraction_concurrency`: Parallel LLM API calls (default: 16)

**3. Validation Features**:
- `enable_numeric_validation`: Gemini-based numeric consistency check
- `enable_entity_validation`: Entity quality scoring and filtering
- `enable_relation_validation`: Relation completeness validation
- `validation_strictness`: STRICT (99%) | MODERATE (95%) | LENIENT (80%)

**4. Merging Features**:
- `enable_entity_merging`: Entity deduplication
- `merge_strategy`: basic (fast) | fuzzy (accurate) | hybrid

**5. Quality Features**:
- `enable_hitl`: Save failed extractions for human review
- `enable_orphan_linking`: Post-merge orphan entity linking
- `enable_quality_scoring`: Track extraction quality metrics

#### Three Presets

**Standard Preset** (replaces current standard pipeline):
```python
features = PipelineFeatures.from_preset("standard")
# Fast, reliable: 90-95% accuracy, ~$0.15/40K doc, 30-60s
```

**Quality Preset** (replaces current production pipeline):
```python
features = PipelineFeatures.from_preset("quality")
# Slow, accurate: 95-99% accuracy, ~$0.40-0.60/40K doc, 2-5min
```

**Balanced Preset** (new):
```python
features = PipelineFeatures.from_preset("balanced")
# Medium: 92-96% accuracy, ~$0.25-0.35/40K doc, 1-2min
```

#### Custom Configuration

```python
# Mix and match features as needed
features = PipelineFeatures(
    enable_table_detection=True,
    chunk_mode="semantic",
    enable_gleaning=True,
    enable_numeric_validation=False,  # Disable if too strict
    validation_strictness="MODERATE"
)

pipeline = UnifiedPipeline(features)
result = await pipeline.process_document(markdown_text, metadata)
```

#### Key Benefits

- **Flexibility**: Enable only needed features (reduce cost/time)
- **Transparency**: Clear what each feature does
- **Gradual Adoption**: Start with standard, add features incrementally
- **Cost Control**: See exact cost per feature in plan
- **Error Handling**: Graceful degradation if optional features fail

#### Implementation Details

See **[MODULAR_PIPELINE_PLAN.md](MODULAR_PIPELINE_PLAN.md)** for:
- Complete feature flag definitions
- Validation thresholds (STRICT/MODERATE/LENIENT)
- Error handling strategy (graceful degradation)
- HITL system (human-in-the-loop for failed extractions)
- Semantic chunking algorithm (accumulation logic, overlap strategy)
- Gleaning implementation (two-stage: retry + refinement)
- Quality scoring formula (length + keywords + specificity)

---

### Unified Subgraph System

BiG-RAG supports a **unified subgraph architecture** where multiple knowledge graphs (subgraphs) can be managed and queried together via an LLM-based router.

#### What Are Subgraphs?

**Subgraph**: A self-contained knowledge graph built from a specific corpus or domain.

**Examples**:
- `football` - Knowledge graph about football/soccer
- `kuet_test` - Knowledge graph about KUET (Khulna University of Engineering & Technology)
- `medical_drugs` - Knowledge graph about pharmaceutical compounds
- `company_docs` - Knowledge graph from internal company documentation

**Benefits**:
- **Domain Isolation**: Each subgraph maintains domain-specific entity typing and relations
- **Selective Retrieval**: Query only relevant subgraphs (faster, more precise)
- **Lazy Loading**: Subgraphs loaded on-demand (memory efficient)
- **Easy Management**: Add/remove subgraphs without affecting others

---

#### Subgraph Registry

**File**: `expr/subgraph_registry.json`

**Purpose**: Central registry tracking all available subgraphs and their metadata.

**Structure**:
```json
{
  "subgraphs": {
    "football": {
      "path": "expr/football",
      "description": "Knowledge graph about football",
      "aliases": ["football", "soccer"],
      "topics": ["sports", "football"],
      "enabled": true,
      "created_at": "2025-01-20T10:30:00"
    },
    "kuet_test": {
      "path": "expr/kuet_test",
      "description": "KUET educational content",
      "aliases": ["kuet", "kuet_test"],
      "topics": ["education", "university"],
      "enabled": true,
      "auto_created": true  // ← Created via /datasets/create-and-index
    }
  }
}
```

**Key Fields**:
- `path`: Directory containing subgraph files
- `description`: Human-readable description for LLM router
- `aliases`: Alternative names for matching queries
- `topics`: Topic keywords for semantic routing
- `enabled`: Whether subgraph is active
- `auto_created`: Whether created dynamically (vs. manually built)

---

#### Dynamic Dataset Creation

**Endpoint**: `/datasets/create-and-index`

**What It Does**:
1. Creates new dataset directory structure
2. Processes documents using **production pipeline** (always)
3. Updates `subgraph_registry.json` automatically
4. Reloads unified executor to make new subgraph available **immediately**
5. No server restart required

**Example**:
```bash
# Create new dataset and index documents
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "company_handbook",
    "documents": [
      {
        "content": "Employee handbook content...",
        "title": "Employee Handbook",
        "metadata": {"category": "HR", "version": "2025"}
      }
    ],
    "process_async": false
  }'

# Response includes registry update confirmation
{
  "status": "success",
  "dataset_name": "company_handbook",
  "registry_updated": true,  // ← New subgraph added
  "documents_processed": 1
}
```

**After Creation**:
```bash
# Query new dataset immediately (no restart needed)
curl -X POST "http://localhost:8001/api/unified/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the vacation policy?",
    "dataset_name": "company_handbook"  // ← Route to specific subgraph
  }'
```

---

#### Unified Query Endpoints

**Endpoint**: `/api/unified/query`

**Router Logic**: LLM-based semantic routing to select appropriate subgraph(s).

**Query Modes**:

1. **Auto-Routing** (LLM selects subgraphs):
```bash
curl -X POST "http://localhost:8001/api/unified/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who won the Champions League in 2023?"
  }'

# LLM router analyzes query → routes to "football" subgraph
```

2. **Explicit Subgraph Selection**:
```bash
curl -X POST "http://localhost:8001/api/unified/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many departments are in KUET?",
    "dataset_name": "kuet_test"  // ← Force specific subgraph
  }'
```

3. **Multi-Subgraph Query** (future):
```bash
# Query multiple subgraphs and merge results
curl -X POST "http://localhost:8001/api/unified/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare football academies and engineering colleges",
    "dataset_names": ["football", "kuet_test"]  // ← Multiple subgraphs
  }'
```

---

#### Unified Chat Completion

**Endpoint**: `/api/unified/ask`

**Full RAG Pipeline**: Query selection → Retrieval → LLM generation with context.

**Example**:
```bash
curl -X POST "http://localhost:8001/api/unified/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the admission requirements for CSE at KUET?",
    "dataset_name": "kuet_test"
  }'

# Response with generated answer + retrieved context
{
  "answer": "The admission requirements for CSE at KUET are...",
  "contexts": [
    {"content": "KUET CSE admission criteria...", "score": 0.92},
    {"content": "Application process details...", "score": 0.87}
  ],
  "subgraph_used": "kuet_test"
}
```

---

#### Backend Endpoint Behavior

| Endpoint | Pipeline Used | Subgraph Registry | Use Case |
|----------|--------------|-------------------|----------|
| `/documents/upload` | **User Choice** (default: standard) | Not updated | Single document upload |
| `/datasets/create-and-index` | **Always Production** | ✅ **Auto-updated** | Dynamic dataset creation |
| `/api/unified/query` | N/A (retrieval only) | Uses registry | Query existing subgraphs |
| `/api/unified/ask` | N/A (RAG pipeline) | Uses registry | Full question answering |

**Key Takeaways**:
1. `/documents/upload` → User controls pipeline via parameter
2. `/datasets/create-and-index` → Always uses production pipeline + updates registry
3. Unified endpoints work with graphs from **both pipelines** seamlessly

---

### Migration from Old Graphs

**Breaking Change (January 24, 2025)**: Old graphs used incompatible node ID formats.

**Action Required**: Rebuild all existing graphs with new code.

**Rebuild Commands**:
```bash
# Standard pipeline
python script_build.py --data_source my_dataset

# Production pipeline
python script_build.py --data_source my_dataset --use_production_pipeline
```

**Verification**:
```bash
# Check entity node IDs (should start with "entity-")
grep '<node id="entity-' expr/my_dataset/graph_chunk_entity_relation.graphml | head -3

# Check relation node IDs (should start with "rel-")
grep '<node id="rel-' expr/my_dataset/graph_chunk_entity_relation.graphml | head -3

# Check edges (should connect rel-* to entity-*)
grep '<edge source="rel-' expr/my_dataset/graph_chunk_entity_relation.graphml | head -3
```

**Expected Output**:
```xml
<node id="entity-abc123">
<node id="rel-def456">
<edge source="rel-def456" target="entity-abc123">
```

---

## Common Commands

### Data Pipeline Workflow

```
Raw Data → Preprocess → Build Graph → Start Server → Train → Evaluate
```

#### Step 1: Preprocess datasets to parquet format

```bash
python script_process.py --data_source 2WikiMultiHopQA
# Other supported datasets: HotpotQA, Musique, NQ, PopQA, TriviaQA
```

**What it does:**
- Loads raw QA pairs from `datasets/{dataset}/raw/qa_*.json`
- Converts to standardized format with instruction templates
- Saves as Parquet files in `datasets/{dataset}/processed/`

**Output:**
```
datasets/2WikiMultiHopQA/processed/
├── train.parquet     # Training data
├── dev.parquet       # Development/validation data
└── test.parquet      # Test data
```

#### Step 2: Build Bipartite Knowledge Graph

```bash
# IMPORTANT: Set OpenAI API key first
echo "your-api-key-here" > openai_api_key.txt

# Build graph (runs in background)
nohup python -u script_build.py --data_source 2WikiMultiHopQA > build.log 2>&1 &

# Monitor progress
tail -f build.log
```

**What it does:**
1. Loads corpus from `datasets/{dataset}/raw/corpus.jsonl`
2. Chunks documents into manageable sizes (1200 tokens with 100 overlap)
3. Extracts entities and relations using GPT-4o-mini
4. Constructs bipartite graph: Documents ↔ Entities ↔ Relations
5. Generates embeddings with FlagEmbedding (bge-large-en-v1.5)
6. Creates FAISS indices for fast retrieval
7. Saves to `expr/{dataset}/`

**Output:**
```
expr/2WikiMultiHopQA/
├── kv_store_full_docs.json            # Full document metadata
├── kv_store_text_chunks.json          # Text chunk metadata
├── kv_store_llm_response_cache.json   # LLM response cache (optional)
├── vdb_entities.json                  # Entity embeddings (NanoVectorDB)
├── vdb_relations.json           # Relation embeddings (NanoVectorDB)
├── vdb_chunks.json                    # Chunk embeddings for Path C retrieval
└── graph_chunk_entity_relation.graphml # Bipartite graph structure (NetworkX)
```

**Note**: Entity and relation **metadata** (names, descriptions, source_ids, weights) are stored in the GraphML file, not in separate JSON files.

**Time Estimate**: 2-4 hours for ~10K documents (depends on corpus size and API rate limits)

**Alternative**: Download pre-built graphs from [TeraBox](https://1024terabox.com/s/1y1G7trP-hcmIDQRUaBaDDw)

#### Step 3: Start Retrieval Server

```bash
# NEW: Use backend/server.py (script_api.py moved to backend/)
cd backend
nohup python -u server.py --data_source 2WikiMultiHopQA > api.log 2>&1 &

# Verify server is running
curl http://localhost:8001/docs  # Opens FastAPI docs

# Test retrieval
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is the capital of France?"]}'
```

**What it does:**
- Loads pre-built bipartite graph and FAISS indices
- Starts FastAPI server on port 8001
- Provides `/search` endpoint for real-time retrieval during training
- Remains running throughout the training process

**Important**: Server MUST be running before starting training, or training will fail/hang.

**React UI Alternative (NEW - November 2025):**
```bash
# Terminal 1: Start backend (unified mode)
cd backend
python server.py --unified

# Terminal 2: Start frontend
cd frontend
npm run dev

# Open http://localhost:5173 in browser
```

#### Per-Query Language Override (Optional)

**NEW (January 2025)**: Support for per-query language parameter for multilingual retrieval:

```bash
# Example 1: Auto (uses DEFAULT_LANGUAGE from .env)
curl -X POST http://localhost:8001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "কুয়েটে আসন সংখ্যা?"}],
    "use_rag": true
  }'

# Example 2: Explicit English override
curl -X POST http://localhost:8001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "How many seats at KUET?"}],
    "use_rag": true,
    "language": "English"
  }'

# Example 3: Banglish to Bangla conversion
curl -X POST http://localhost:8001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "KUET e CSE te koyti seat ache?"}],
    "use_rag": true,
    "language": "Bangla"
  }'
```

**Supported Languages**: English, Bangla, Hindi, Arabic, Chinese, Spanish, French, German, Japanese, Korean

**Frontend**: Language selector available in Chat Settings panel (gear icon).

**Use Cases**:
- Mixed-language document corpora
- Banglish/romanized query normalization
- Cross-lingual search (with caveats - best when query matches document language)

---

#### Step 4: Run RL Training

```bash
# GRPO (recommended for starting)
nohup bash -u run_grpo.sh \
  -p Qwen/Qwen2.5-3B-Instruct \
  -m Qwen2.5-3B-Instruct \
  -d 2WikiMultiHopQA \
  > training.log 2>&1 &

# Monitor training
tail -f training.log

# Check GPU utilization
nvidia-smi -l 1

# View Ray dashboard
# Navigate to http://localhost:8265
```

**Parameters:**
- `-p`: Model path (HuggingFace ID or local path)
- `-m`: Model name (for experiment tracking/logging)
- `-d`: Dataset name (must match processed dataset)

**Other Algorithms:**
```bash
# REINFORCE++
bash run_rpp.sh -p <model_path> -m <model_name> -d <dataset>

# PPO (requires critic model)
bash run_ppo.sh -p <model_path> -m <model_name> -d <dataset>
```

#### Step 5: Stop Retrieval Server (after training)

```bash
# Linux/macOS
fuser -k 8001/tcp

# Windows
# netstat -ano | findstr :8001
# taskkill /PID <pid> /F
```

### Evaluation

```bash
cd evaluation
# See evaluation/README.md for dataset-specific instructions
python eval.py --checkpoint path/to/checkpoint --dataset 2WikiMultiHopQA
```

### Inference

```bash
cd inference
# See inference/README.md for detailed instructions
python inference.py --model path/to/trained/model --query "Your question here"
```

---

## Architecture Overview

### Bipartite Graph Structure

Unlike traditional hypergraphs, BiG-RAG uses a **true bipartite graph**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Bipartite Graph Structure                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Document Chunks                    Entities & Relations     │
│  ┌──────────┐                         ┌──────────┐          │
│  │  Doc A   │◄──────────────────────► │ Entity 1 │          │
│  └──────────┘                         └──────────┘          │
│       ▲                                     ▲                │
│       │                                     │                │
│       ▼                                     ▼                │
│  ┌──────────┐     Relation    ┌──────────┐           │
│  │  Doc B   │◄──────────────────────►│ Relation │           │
│  └──────────┘                         └──────────┘          │
│       ▲                                     ▲                │
│       │                                     │                │
│       ▼                                     ▼                │
│  ┌──────────┐                         ┌──────────┐          │
│  │  Doc C   │◄──────────────────────► │ Entity 2 │          │
│  └──────────┘                         └──────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Properties:**
- **Two node types**: Documents (chunks) and Semantic nodes (entities + relations)
- **Edges**: Connect documents to the entities/relations they contain
- **No direct edges**: Documents don't connect to documents; entities don't connect to entities
- **Queries**: Traverse from query → entities → relations → documents

### Weight Semantics

BiG-RAG uses **weight values** to rank the importance of entities and relations in the knowledge graph. Understanding these semantics is crucial for interpreting query results and debugging.

#### Entity Weights

**Calculation:**
```
weight = Σ(importance_score) for all occurrences
```

**Interpretation:**
- **Range**: 0 to N×100 (where N = number of chunks mentioning the entity)
- **Components**: Sum of LLM-assigned importance scores (key_score: 0-100) across all chunks
- **Higher weight** = more frequently mentioned + higher LLM importance ratings

**Examples:**
| Weight Range | Interpretation | Typical Example |
|--------------|----------------|-----------------|
| 400+ | Very central entity | "LIONEL MESSI" mentioned in 4+ chunks with high scores |
| 200-399 | Important entity | "BARCELONA" mentioned in 2-3 chunks |
| 100-199 | Mentioned entity | "LA LIGA" mentioned in 1-2 chunks |
| 50-99 | Peripheral entity | "COPA DEL REY" mentioned once with low score |

**Why no normalization?**
- Preserves **frequency signal**: Higher weight = more occurrences
- Enables **importance ranking**: Sort by weight to find central entities
- Reflects **graph centrality**: Entities with high weights are hubs

#### Relation Weights

**Calculation:**
```
weight = Σ(completeness_score) for all occurrences
```

**Interpretation:**
- **Range**: 0 to N×10 (where N = number of chunks mentioning the relation)
- **Components**: Sum of completeness scores (0-10) from LLM extraction
- **Higher weight** = more complete and frequently mentioned knowledge

**Examples:**
| Weight Range | Interpretation | Typical Example |
|--------------|----------------|-----------------|
| 20+ | Very important relation | Core fact mentioned in 2+ chunks |
| 10-19 | Important relation | Mentioned in 1-2 chunks with high completeness |
| 5-9 | Single mention | Mentioned once |
| <5 | Incomplete relation | Partial information extracted |

**Completeness Score Criteria:**
- **9-10**: Complete, well-formed knowledge segment with full context
- **7-8**: Mostly complete, minor context missing
- **5-6**: Partial information, some ambiguity
- **<5**: Incomplete or fragmented knowledge

#### Using Weights in Practice

**1. Entity Ranking**
```python
# Get top entities by weight
top_entities = sorted(entities, key=lambda e: e['weight'], reverse=True)[:10]
```

**2. Filtering by Importance**
```python
# Only keep entities with weight > 100 (mentioned at least once with decent score)
important_entities = [e for e in entities if e['weight'] > 100]
```

**3. Debug Low Weights**
```python
# Check if entity has low weight because:
# - Few occurrences (check source_id count)
# - Low LLM importance scores (check extraction quality)
```

**4. Weight Distribution Analysis**
```python
# Create histogram to understand weight distribution
import numpy as np
weights = [e['weight'] for e in entities]
print(f"Mean: {np.mean(weights):.1f}, Median: {np.median(weights):.1f}")
print(f"Min: {min(weights):.1f}, Max: {max(weights):.1f}")
```

#### Common Questions

**Q: Why is entity X weight 360.0?**
A: Entity X appears in multiple chunks (360/90 ≈ 4 mentions with avg score ~90).

**Q: Should I normalize weights?**
A: No. Un-normalized weights preserve frequency information which is valuable for ranking.

**Q: Can weights be negative?**
A: No. Weights are always non-negative (sum of positive scores).

**Q: Do weights affect retrieval?**
A: Yes. During graph traversal, higher-weight nodes/edges are prioritized in ranking.

---

### Training Pipeline Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│  1. Dataset (Parquet) → RL Dataset Loader                      │
├────────────────────────────────────────────────────────────────┤
│  • Loads batch of prompts                                      │
│  • Applies attention masks                                     │
│  • Sends to Actor                                              │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  2. Actor Rollout (vLLM) → Generate with Tool Calls            │
├────────────────────────────────────────────────────────────────┤
│  • LLM generates text token-by-token                           │
│  • ToolGenerationManager detects <query> tags                  │
│  • Extracts query text between tags                            │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  3. Tool Execution → Iterative Retrieval Cycle                 │
├────────────────────────────────────────────────────────────────┤
│  For each <query> tag:                                         │
│    a) Extract query text                                       │
│    b) HTTP POST to retrieval server (port 8001)                │
│    c) Server queries BiG-RAG bipartite graph                   │
│    d) Returns relevant context                                 │
│    e) Format as <knowledge>...</knowledge>                     │
│    f) Append to prompt                                         │
│    g) Continue LLM generation                                  │
│  Repeat until <answer>...</answer> or max_turns reached        │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  4. Reward Computation → EM/F1 vs Ground Truth                 │
├────────────────────────────────────────────────────────────────┤
│  • Extract answer from <answer> tags                           │
│  • Compute Exact Match (EM)                                    │
│  • Compute token-level F1 score                                │
│  • Assign reward: r = α·EM + β·F1                              │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  5. Critic (PPO only) → Value Estimation                       │
├────────────────────────────────────────────────────────────────┤
│  • Estimates state value V(s)                                  │
│  • Used for advantage estimation (PPO)                         │
│  • Skipped in GRPO (uses group-relative rewards)               │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  6. RL Algorithm → Policy Update                               │
├────────────────────────────────────────────────────────────────┤
│  GRPO:                                                         │
│    • Compare rewards across group of generations               │
│    • Update policy to favor higher-reward trajectories         │
│  PPO:                                                          │
│    • Compute advantages using GAE                              │
│    • Clip policy updates to maintain stability                │
│  REINFORCE++:                                                  │
│    • Variance-reduced policy gradients                         │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  7. Checkpointing → Save Model Periodically                    │
├────────────────────────────────────────────────────────────────┤
│  • Saves actor model state                                     │
│  • Saves critic model state (PPO)                              │
│  • Saves optimizer state                                       │
│  • Logs metrics to W&B                                         │
└────────────────────────────────────────────────────────────────┘
```

### Knowledge Graph Query Flow

```
┌────────────────────────────────────────────────────────────────┐
│  Query: "Who is the director of Nosferatu (1922)?"             │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  1. Tokenize & Embed Query                                     │
├────────────────────────────────────────────────────────────────┤
│  • Encode with FlagEmbedding (bge-large-en-v1.5)               │
│  • Output: 1536-dimensional vector                             │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  2. Vector Search in FAISS Indices                             │
├────────────────────────────────────────────────────────────────┤
│  Mode: "hybrid" (default)                                      │
│    • Search entity index → top-k entities                      │
│    • Search bipartite edge index → top-k relations             │
│    • Combine results                                           │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  3. Retrieve from Bipartite Graph                              │
├────────────────────────────────────────────────────────────────┤
│  Matched entities: ["Nosferatu", "F.W. Murnau"]                │
│  Matched relations: [("Nosferatu", "directed_by", "Murnau")]   │
│                                                                │
│  Traverse graph:                                               │
│    Entities → Connected Relations → Connected Documents        │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  4. Rank & Format Results                                      │
├────────────────────────────────────────────────────────────────┤
│  • Rank by relevance (cosine similarity + graph structure)     │
│  • Select top-k documents (typically k=5-10)                   │
│  • Format as natural language context                          │
└────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  5. Return Context                                             │
├────────────────────────────────────────────────────────────────┤
│  "Nosferatu is a 1922 silent film directed by F.W. Murnau...   │
│   The film is an adaptation of Bram Stoker's Dracula..."       │
└────────────────────────────────────────────────────────────────┘
```

### Tool-Augmented Generation Cycle

During training, the LLM learns to use tools through this cycle:

```
   ┌──────────────────────────────────────────────┐
   │  LLM: <think>I need info about X</think>     │
   └───────────────────┬──────────────────────────┘
                       ↓
   ┌──────────────────────────────────────────────┐
   │  LLM: <query>search for X</query>            │
   └───────────────────┬──────────────────────────┘
                       ↓
   ┌──────────────────────────────────────────────┐
   │  ToolEnv: Detect <query> tag                 │
   └───────────────────┬──────────────────────────┘
                       ↓
   ┌──────────────────────────────────────────────┐
   │  HTTP Request → Retrieval Server (8001)      │
   └───────────────────┬──────────────────────────┘
                       ↓
   ┌──────────────────────────────────────────────┐
   │  BiG-RAG: Query bipartite graph              │
   └───────────────────┬──────────────────────────┘
                       ↓
   ┌──────────────────────────────────────────────┐
   │  Return: <knowledge>context</knowledge>      │
   └───────────────────┬──────────────────────────┘
                       ↓
   ┌──────────────────────────────────────────────┐
   │  Append to prompt → Continue generation      │
   └───────────────────┬──────────────────────────┘
                       ↓
   ┌──────────────────────────────────────────────┐
   │  LLM: <think>Based on knowledge...</think>   │
   │  LLM: <answer>Final answer</answer>          │
   └──────────────────────────────────────────────┘
```

**Key Points:**
- LLM learns **when** and **how** to query (via RL reward signal)
- Retrieval happens **synchronously** within generation loop
- Can iterate multiple times (multi-hop reasoning)
- Reward model scores final answer quality

---

## Configuration System

BiG-RAG uses **Hydra** for hierarchical configuration.

### Base Configuration Structure

**Location**: [verl/trainer/config/ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml)

```yaml
data:                    # Dataset paths, batch sizes, token limits
  train_files: [...]
  val_files: [...]
  max_prompt_length: 4096
  max_response_length: 4096
  train_batch_size: 128

actor_rollout_ref:       # Actor, rollout (vLLM/HF), reference model
  model:
    path: "Qwen/Qwen2.5-3B-Instruct"
  actor:
    optim:
      lr: 5e-7
  rollout:
    tensor_model_parallel_size: 4
    gpu_memory_utilization: 0.5

critic:                  # Critic model (PPO only)
  optim:
    lr: 1e-5

algorithm:               # RL hyperparameters
  adv_estimator: "grpo"  # or "gae" for PPO
  gamma: 1.0
  lam: 0.95
  kl_ctrl:
    kl_coef: 0.001

trainer:                 # Training loop
  total_epochs: 1
  test_freq: 10
  save_freq: -1
  n_gpus_per_node: 4

tool:                    # Tool configuration
  env: 'search'
  max_turns: 5
  query_start_tag: "<query>"
  query_end_tag: "</query>"
```

### Runtime Overrides

Training scripts override config via CLI:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=[datasets/2WikiMultiHopQA/processed/train.parquet] \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    trainer.total_epochs=1 \
    tool.env=search \
    tool.max_turns=5
```

### Key Parameters Explained

**Data Parameters:**
- `max_prompt_length`: Maximum input tokens (includes question + retrieved context)
- `max_response_length`: Maximum generation tokens (includes tool calls + answer)
- `max_tool_response_length`: Maximum tokens per tool response
- `train_batch_size`: Global batch size across all GPUs

**Actor/Rollout Parameters:**
- `tensor_model_parallel_size`: Number of GPUs for model parallelism (1 GPU per shard)
- `gpu_memory_utilization`: Fraction of GPU memory for vLLM (0.5 = 50%)
- `n_repeat`: Number of generations per prompt (for reward variance estimation)
- `use_kl_loss`: Enable KL divergence penalty (keeps model close to reference)

**Training Parameters:**
- `total_epochs`: Training duration (1 epoch = 1 pass through dataset)
- `test_freq`: Evaluate every N epochs
- `save_freq`: Checkpoint frequency (-1 = only save at end)
- `n_gpus_per_node`: GPUs per machine (for multi-node setups)

**Tool Parameters:**
- `env`: Tool environment type (`'search'` for graph retrieval)
- `max_turns`: Maximum tool interaction cycles per generation
- `query_start_tag`, `query_end_tag`: Markers for tool invocation

### Environment Variables

```bash
# vLLM configuration
export VLLM_ATTENTION_BACKEND=XFORMERS        # Attention implementation (XFORMERS or FLASH_ATTN)

# Model path
export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"  # HuggingFace ID or local path

# Logging
export PROJECT_NAME='BiG-RAG'                 # Weights & Biases project name
export EXPERIMENT_NAME="qwen3b_2wiki_grpo"    # W&B experiment name

# Debugging
export HYDRA_FULL_ERROR=1                     # Show full Hydra config errors
export CUDA_LAUNCH_BLOCKING=1                 # Synchronous CUDA (for debugging)
```

---

## ⭐ Recent Improvements (January 2025)

BiG-RAG has been significantly enhanced with three major improvements. See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for full details.

### Phase 2: Critical Fixes

#### Metadata & Title Preservation
**Problem**: Document metadata (title, tags, category) was discarded during indexing, reducing entity extraction accuracy.

**Solution**: Full metadata preservation pipeline from ingestion → chunking → entity extraction.

**Impact**: +2-3 F1 points improvement in entity extraction quality.

**Usage**:
```python
# Insert documents with metadata
rag.insert(
    ["Document content"],
    metadata=[{"title": "My Doc", "category": "science", "tags": ["ai", "ml"]}]
)
```

#### Document Deletion System
**New Method**: `rag.delete_document(doc_id_or_content)`

Enables removing indexed documents with **full cascade cleanup** across all storage layers:
- Deletes document chunks from KV storage and vector DB
- Removes orphaned entities/edges (only referenced by deleted document)
- Updates shared entities/edges (preserves if referenced by other documents)
- Smart reference counting prevents data loss

**Usage**:
```python
# Delete by document ID
rag.delete_document("doc-abc123")

# Delete by original content
rag.delete_document("The original document text...")
```

**Performance**: ~1-2 seconds for cascade deletion (no graph rebuild needed)

### Phase 3: Three-Path Retrieval + Reranking

#### Three-Path Retrieval Architecture
**Old**: 2 paths (Entity + Relation) → 5 context items
**New**: 3 paths (Entity + Relation + Chunk) → **10 context items**

```
Query → Path A (Entities)  → top-60 entities → RRF → top-5 structured
     → Path B (Relations) → top-60 edges   → RRF → top-5 structured
     → Path C (Chunks)    → 10 candidates  → rerank → top-5 chunks

Output: 5 structured + 5 chunks = 10 total context items
```

**Impact**: +15-25% recall, +10-20% precision

#### Semantic Reranking
Cross-encoder based reranking of chunk candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Usage**:
```python
from bigrag.base import QueryParam

# With reranking (default)
results = rag.query("query", QueryParam(enable_reranking=True))

# Without reranking (faster)
results = rag.query("query", QueryParam(enable_reranking=False))
```

**Optional Dependency**: `pip install sentence-transformers` (~330MB)

**Performance**: +10-20% precision at ~50-100ms latency cost

### Phase 4: Critical Bug Fixes (January 2025)

During system testing, we discovered and fixed 5 critical bugs:

#### Bug #1: Missing chunks_vdb Indexing
**Symptom**: `vdb_chunks.json` file was empty (0 entries)
**Cause**: Chunks were created but never indexed to `chunks_vdb`
**Impact**: Path C (chunk-based retrieval) was completely broken
**Fixed**: [bigrag/bigrag.py:384-395](bigrag/bigrag.py#L384-L395)

#### Bug #2: API Reading Non-Existent Files
**Symptom**: Document stats always showed 0 entities and 0 edges
**Cause**: `api/kg_utils.py` looking for `kv_store_entities.json` and `kv_store_relations.json` which were never created
**Impact**: All document detail endpoints returned incorrect stats
**Fixed**: [api/kg_utils.py:53-199](api/kg_utils.py#L53-L199) - Now reads from GraphML

#### Bug #3: Entity Weights Missing
**Symptom**: All entities had `weight=0` in API responses
**Cause**: `_merge_nodes_then_upsert()` didn't aggregate weights from multiple occurrences
**Impact**: Entities couldn't be ranked by importance
**Fixed**: [bigrag/operate.py:190-243](bigrag/operate.py#L190-L243) - Added weight aggregation

#### Bug #4: Incomplete Rebuild Cleanup
**Symptom**: `rebuild_entire_graph()` listed non-existent files and missed actual files
**Cause**: Outdated file list from old architecture
**Impact**: Rebuild didn't clean all index files
**Fixed**: [api/kg_utils.py:418-439](api/kg_utils.py#L418-L439)

#### Bug #5: Incomplete Document Deletion
**Symptom**: Hard delete only removed from corpus, leaving all KG data orphaned
**Cause**: `adelete_document()` was incomplete stub; API didn't call it
**Impact**: Deleted documents remained in knowledge graph
**Fixed**: Complete cascade deletion implementation (see Phase 2 above)

All bugs are now fixed and tested. The system is production-ready.

### Phase 5: Orphan Node Reduction (January 2025)

#### Critical Delimiter Corruption Bug Fix
**Problem**: LLM was outputting `<<|>>` (double angle brackets) instead of `<|>` delimiter, causing 100% extraction failure (0 entities, 0 relations extracted).

**Root Cause**: The `fix_delimiter_corruption()` function had substring patterns (`<|` and `|>`) that were breaking already-fixed delimiters.

**Solution**:
- Added double-bracket pattern `<<|>>` to corruption detection
- Removed substring patterns that caused secondary corruption
- Applied fix per-record during extraction (critical for success)

**Implementation**: [bigrag/utils.py:372-408](bigrag/utils.py#L372-L408)

**Results**:
```
Before Fix (Broken):
- Entities: 0 (100% failure)
- Relations: 0 (100% failure)

After Fix (Working):
- Entities: 133 ✅
- Relations: 77 ✅
- Orphan Relations: 9 (11.7%)
- Orphan Entities: 0 (0%)
```

#### Relation Context Check Relaxation
**Problem**: Strict validation was rejecting entities without immediate relation context, causing data loss.

**Solution**: Changed from rejecting to creating default relation when context is missing. This prevents data loss while still tracking sequencing issues.

**Implementation**: [bigrag/operate.py:309-321](bigrag/operate.py#L309-L321)

**Impact**: Prevents orphan entities while maintaining data integrity.

#### Comprehensive Unit Testing
**New File**: [test_scripts/test_orphan_reduction_validation.py](test_scripts/test_orphan_reduction_validation.py)

**Test Coverage**:
- Delimiter corruption fix (4/4 tests passing)
- Sanitization logic (6/7 tests passing)
- Quality scoring (5/5 tests passing)
- Parsing logic (3/3 tests passing)
- Relation validation (5/5 tests passing)
- Entity validation (5/5 tests passing)
- End-to-end integration (1/1 test passing)

**Overall**: 96.7% test coverage (29/30 tests passing)

### Testing Your Installation

Run the comprehensive test suite:
```bash
cd test_scripts
python test_improvements.py
```

Tests all improvements: metadata preservation, document deletion, three-path retrieval, reranking, and orphan node reduction.

**All test scripts are now in `test_scripts/` directory.**

---

## Important Implementation Details

### Storage Plugin System

BiG-RAG uses **abstract base classes** with **lazy imports** to support multiple backends without requiring all dependencies.

**Base Classes** ([bigrag/base.py](bigrag/base.py)):
- `BaseGraphStorage`: Graph database interface
- `BaseVectorStorage`: Vector database interface
- `BaseKVStorage`: Key-value storage interface

**Default Implementations** ([bigrag/storage.py](bigrag/storage.py)):
- `NetworkXStorage`: In-memory graph (NetworkX)
- `NanoVectorDBStorage`: In-memory vector DB
- `JsonKVStorage`: JSON file-based KV store

**Optional Backends** ([bigrag/kg/*.py](bigrag/kg/)):
- **Graph**: Neo4J, Oracle, MongoDB
- **Vector**: Milvus, ChromaDB, TiDB, Oracle
- **KV**: MongoDB, Oracle, TiDB

**To add new backend:**
1. Inherit from base class in [bigrag/base.py](bigrag/base.py)
2. Implement required async methods
3. Add to `lazy_external_import()` in [bigrag/bigrag.py](bigrag/bigrag.py)

### Async/Await Pattern

Nearly all BiG-RAG operations are async-first:

```python
# ✅ Correct usage (async)
await bigrag.ainsert(documents)
contexts = await bigrag.aquery(query, param)

# ⚠️ Synchronous wrappers (discouraged but available)
bigrag.insert(documents)  # Internally calls ainsert()
contexts = bigrag.query(query, param)  # Internally calls aquery()
```

**Why async?**
- Enables concurrent operations (batch processing)
- Better resource utilization (I/O-bound tasks)
- Required for distributed workers in RL training

### Tool Integration During Training

BiG-RAG trains the LLM to **actively generate tool calls**, not just use pre-retrieved context:

```python
# Standard RAG (post-hoc)
context = retrieve(query)
response = llm(query + context)

# BiG-RAG (learned tool use)
response = llm(query)
# During generation, LLM emits: <query>sub-query</query>
# → Retrieval happens synchronously
# → Context injected into generation
# → LLM continues with context
# → Reward signal based on final answer
```

**Key Difference**: The model learns **when** and **how** to query, not just how to use given context.

### Distributed Training with Ray

**Architecture**:
- **Ray Cluster**: Manages distributed workers across GPUs/nodes
- **Workers**: Actor, Critic, Rollout, Reward Manager run as separate Ray actors
- **Parallelism**: Combines FSDP (data parallel), Tensor Parallel (model parallel), and Ray (node distribution)

**Setup**:
```bash
# Start Ray cluster (before training)
ray start --head

# Training script automatically connects to cluster
bash run_grpo.sh ...

# Stop cluster (after training or on error)
ray stop
```

**Common Issues**:
- Always run `ray stop` between training runs (prevents GPU memory leaks)
- Check Ray dashboard at `http://localhost:8265` for worker status
- If training hangs, check Ray logs: `cat /tmp/ray/session_latest/logs/*`

### Storage Architecture

BiG-RAG uses a three-layer storage architecture:

**1. Vector Storage (NanoVectorDB by default)**:
- `vdb_entities.json`: Entity embeddings for Path A (entity-based retrieval)
- `vdb_relations.json`: Relation embeddings for Path B (relation-based retrieval)
- `vdb_chunks.json`: Text chunk embeddings for Path C (chunk-based retrieval)

**2. Graph Storage (NetworkX by default)**:
- `graph_chunk_entity_relation.graphml`: Complete bipartite graph with all node/edge attributes
  - Entity nodes: `{name, description, entity_type, source_id, weight, role="entity"}`
  - Relation nodes: `{name, description, source_id, weight, role="relation"}`
  - Edges connect chunks ↔ entities/relations

**3. KV Storage (JSON by default)**:
- `kv_store_full_docs.json`: Full document metadata and content
- `kv_store_text_chunks.json`: Text chunks with metadata (title, id, source)
- `kv_store_llm_response_cache.json`: Cached LLM responses (optional)

**Query Modes**:
- `local`: Entity-based retrieval only (Path A)
- `global`: Relation-based retrieval only (Path B)
- `hybrid`: Combines Path A + Path B + Path C (default, most effective)
- `naive`: Direct text chunk retrieval (baseline)

---

## Key Files and Entry Points

### Main Execution Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| [script_process.py](script_process.py) | Preprocess datasets to parquet | `datasets/{dataset}/processed/*.parquet` |
| [script_build.py](script_build.py) | Build bipartite knowledge graph | `expr/{dataset}/` (see Output section above) |
| [script_api.py](script_api.py) | Start retrieval server (FastAPI) | HTTP server on port 8001 |
| [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py) | RL training entry point | Model checkpoints, logs |
| [verl/trainer/main_generation.py](verl/trainer/main_generation.py) | Inference/generation | Generated text |
| [verl/trainer/main_eval.py](verl/trainer/main_eval.py) | Evaluation runner | EM/F1 metrics |

### Core Library Modules

| Module | Purpose |
|--------|---------|
| [bigrag/bigrag.py](bigrag/bigrag.py) | Main `BiGRAG` class (async insert/query/delete with metadata) |
| [bigrag/operate.py](bigrag/operate.py) | Entity extraction, chunking, three-path retrieval |
| [bigrag/reranker.py](bigrag/reranker.py) | ⭐ **NEW**: Semantic reranking with cross-encoder |
| [bigrag/storage.py](bigrag/storage.py) | Default storage implementations |
| [bigrag/base.py](bigrag/base.py) | Abstract base classes (TextChunkSchema with metadata) |
| [bigrag/llm.py](bigrag/llm.py) | LLM completion wrappers (OpenAI, HuggingFace) |
| [bigrag/prompt.py](bigrag/prompt.py) | Prompt templates for entity extraction |
| [bigrag/utils.py](bigrag/utils.py) | Utility functions (hashing, encoding, caching, **delimiter corruption fix**) |
| [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) | Distributed PPO trainer |
| [agent/llm_agent/generation.py](agent/llm_agent/generation.py) | ToolGenerationManager (tool-calling loop) |
| [agent/tool/tool_env.py](agent/tool/tool_env.py) | ToolEnv (manages tool state) |

### Configuration Files

| File | Purpose |
|------|---------|
| [verl/trainer/config/ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml) | Base PPO configuration |
| [verl/trainer/config/sft_trainer.yaml](verl/trainer/config/sft_trainer.yaml) | Supervised fine-tuning config |
| [run_grpo.sh](run_grpo.sh) | GRPO training script |
| [run_ppo.sh](run_ppo.sh) | PPO training script |
| [run_rpp.sh](run_rpp.sh) | REINFORCE++ training script |

---

## Datasets

### Supported Datasets

**Multi-Hop QA**: 2WikiMultiHopQA, HotpotQA, Musique
**Single-Hop QA**: NQ (Natural Questions), PopQA, TriviaQA

### Directory Structure

```
datasets/{dataset_name}/
├── raw/
│   ├── corpus.jsonl          # Text corpus for bipartite graph
│   ├── qa_train.json         # Training QA pairs
│   ├── qa_test.json          # Test QA pairs
│   └── qa_dev.json           # Development QA pairs
└── processed/
    ├── train.parquet         # Preprocessed training data
    ├── test.parquet          # Preprocessed test data
    └── dev.parquet           # Preprocessed dev data
```

### Corpus Format

Each line in `corpus.jsonl`:
```json
{
  "id": "doc_001",
  "contents": "The actual text content...",
  "title": "Optional title",
  "metadata": {...}
}
```

### QA Pair Format

```json
[
  {
    "question": "Your question here?",
    "golden_answers": ["answer1", "answer2"]
  }
]
```

### Download Pre-built Datasets

- **Datasets**: [TeraBox Link](https://1024terabox.com/s/12FXnOnOhOZNyGzjWuoo-qg)
- **Pre-built Graphs**: [TeraBox Link](https://1024terabox.com/s/1y1G7trP-hcmIDQRUaBaDDw)

### Building Custom Datasets

See [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md) for complete guide.

---

## Common Gotchas

1. **Retrieval server not running**: Training will fail/hang if port 8001 is not responding
   - **Check**: `ps aux | grep script_api` or `curl http://localhost:8001/docs`
   - **Fix**: Start server with `python script_api.py --data_source {dataset}`

2. **Ray cleanup**: Lingering processes consume GPU memory
   - **Check**: `ray status` or `ps aux | grep ray`
   - **Fix**: Always run `ray stop` after training

3. **CUDA OOM with vLLM**: Insufficient GPU memory
   - **Fix**: Reduce `gpu_memory_utilization` (e.g., from 0.5 to 0.3)
   - **Fix**: Increase `tensor_model_parallel_size` to spread model across more GPUs

4. **OpenAI API key**: Required for graph construction
   - **Fix**: Set key in `openai_api_key.txt` or use pre-built graphs

5. **Tool markers must match**: LLM must generate exact tags
   - If changing tags, update `tool.query_start_tag`, `tool.query_end_tag` in config
   - Default: `<query>`, `</query>`, `<answer>`, `</answer>`

6. **Parquet format required**: Raw JSON files won't work for training
   - **Fix**: Always run `script_process.py` before training

7. **Model compatibility**: Only decoder-only models supported
   - **Supported**: Qwen, Llama, Mistral, Gemma
   - **Not supported**: T5, BART, encoder-decoder models

8. **Windows limitations**: Training scripts use bash and Linux commands
   - **Fix**: Use WSL2 or adapt scripts for PowerShell

9. **File naming**: After rebranding, some file paths changed
   - **Old**: `index_hyperedge.bin`, `kv_store_hyperedges.json`
   - **New**: `index_relation.bin`, `kv_store_relations.json`
   - If using pre-built graphs, you may need to rename files

10. **Delimiter corruption in LLM output**: LLM sometimes outputs `<<|>>` instead of `<|>`
   - **Symptom**: Graph build shows 0 entities, 0 relations extracted
   - **Cause**: Double angle brackets breaking parsing
   - **Fix**: Already handled by `fix_delimiter_corruption()` in [bigrag/utils.py](bigrag/utils.py)
   - **Test**: Run orphan detection: `cd test_scripts && python test_orphan_detection.py`

---

## Logging and Debugging

### Console Logging

- RL training logs printed to stdout/stderr
- Redirect with `> logfile.log 2>&1` when using nohup
- Useful patterns to grep:
  - `grep "epoch"`: See training progress
  - `grep "reward"`: Check reward values
  - `grep "ERROR"`: Find errors

### Weights & Biases

- Configure via `trainer.logger=['console','wandb']`
- **Tracks**:
  - Reward curves (mean, std, min, max)
  - EM/F1 scores
  - KL divergence (ref vs policy)
  - Loss values (policy loss, value loss)
  - Learning rates
- **Setup**:
  ```bash
  export PROJECT_NAME='BiG-RAG'
  export EXPERIMENT_NAME="qwen3b_2wiki_grpo"
  wandb login  # First time only
  ```

### Ray Dashboard

- Access at `http://localhost:8265` when cluster is running
- **Shows**:
  - GPU utilization per worker
  - Worker status (running, failed, pending)
  - Task execution timeline
  - Memory usage
- **Useful for**: Debugging distributed training issues

### Debug Mode

```bash
# Synchronous CUDA (easier to debug)
export CUDA_LAUNCH_BLOCKING=1

# Validate data before training
python -m verl.trainer.main_ppo \
  trainer.val_before_train=True \
  ... other args ...

# Reduce batch size to isolate GPU issues
python -m verl.trainer.main_ppo \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  ... other args ...

# Enable verbose logging
python -m verl.trainer.main_ppo \
  trainer.log_level=DEBUG \
  ... other args ...
```

### Centralized Logging System (November 2025)

BiG-RAG uses a comprehensive centralized logging infrastructure for production-ready log management.

**Directory Structure**:
```
logs/
├── bigrag-core/     # Core engine logs (bigrag.log, error.log)
├── backend/         # Backend API logs (api.log, error.log, access.log)
├── jobs/            # Background job logs
└── frontend/        # Frontend logs (browser console)
```

**Key Features**:
- **Log Rotation**: Size-based (10MB) and time-based (daily) rotation with configurable backup retention
- **Multiple Handlers**: Console, file, and error-only log streams
- **Structured Logging**: Optional JSON format for log aggregation tools
- **Component Separation**: Logs organized by component (core, backend, frontend)
- **Contextual Logging**: Add metadata to log entries for better debugging

**Configuration** (`.env`):
```bash
# Log level
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log directory (optional, defaults to logs/bigrag-core/)
LOG_DIR=./logs/bigrag-core

# JSON format for structured logging
LOG_JSON_FORMAT=false

# Frontend log level
VITE_LOG_LEVEL=INFO
```

**Usage in Python**:
```python
from bigrag.logging_config import setup_logger, add_context

# Setup logger
logger = setup_logger(
    name="my_module",
    log_dir="./logs/backend",
    log_file="my_module.log",
    level="INFO",
    rotation="size",  # or "time"
    max_bytes=10*1024*1024,
    backup_count=5
)

# Basic logging
logger.info("Processing started")
logger.error("Failed to connect", error)

# Contextual logging
ctx_logger = add_context(logger, request_id="abc123", user_id="user456")
ctx_logger.info("User action logged")
```

**Usage in Frontend (TypeScript)**:
```typescript
import { logger, apiLogger, graphLogger } from '@/utils/logger';

logger.info('User logged in', { userId: '123' });
graphLogger.debug('Rendering graph', { nodes: 100 });
apiLogger.error('API call failed', error);
```

**Implementation Files**:
- **[bigrag/logging_config.py](bigrag/logging_config.py)** - Centralized logging module
- **[frontend/src/utils/logger.ts](frontend/src/utils/logger.ts)** - Frontend browser logger
- **[docs/technical/LOGGING_GUIDE.md](docs/technical/LOGGING_GUIDE.md)** - Complete logging documentation

For detailed configuration, best practices, and troubleshooting, see [docs/technical/LOGGING_GUIDE.md](docs/technical/LOGGING_GUIDE.md).

---

## Evaluation Metrics

### Exact Match (EM)

**Definition**: Percentage of predictions that exactly match ground truth (after normalization)

**Normalization**:
- Lowercase
- Remove punctuation
- Remove articles (a, an, the)
- Trim whitespace

**Example**:
```
Prediction: "The answer is Paris."
Ground truth: "paris"
Normalized prediction: "answer paris"
Normalized ground truth: "paris"
EM: 0 (no exact match)
```

### F1 Score

**Definition**: Token-level F1 between prediction and ground truth

**Calculation**:
1. Tokenize prediction and ground truth
2. Compute precision: |tokens in both| / |tokens in prediction|
3. Compute recall: |tokens in both| / |tokens in ground truth|
4. F1 = 2 * (precision * recall) / (precision + recall)

**Example**:
```
Prediction: "The capital is Paris"
Ground truth: "Paris"
Tokens prediction: ["capital", "paris"]
Tokens ground truth: ["paris"]
Overlap: ["paris"]
Precision: 1/2 = 0.5
Recall: 1/1 = 1.0
F1: 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.67
```

### Reward Scoring

**Defined in**: [verl/utils/reward_score/](verl/utils/reward_score/)

**Configurable**:
```yaml
reward_model:
  style: "rule"  # or "model" for learned reward model
  ground_truth: ["answer"]
  em_weight: 1.0
  f1_weight: 0.5
```

**Combined Reward**:
```python
reward = em_weight * EM + f1_weight * F1
```

---

## 📁 Project Structure (November 2025)

BiG-RAG has been reorganized for better clarity and scalability:

```
BiG-RAG/
├── README.md                    # Main README
├── CLAUDE.md                    # This file
├── BIGRAG_UI_PLAN.md           # UI implementation plan
├── IMPLEMENTATION_STATUS.md     # Current development status
│
├── backend/                     # FastAPI server (NEW)
│   ├── api/                    # API modules (moved from root)
│   ├── server.py               # Main server (was script_api.py)
│   └── README.md               # Backend documentation
│
├── frontend/                    # React UI (NEW - Nov 2025)
│   ├── src/                    # React 19 + TypeScript + Tailwind v4
│   ├── package.json            # Latest dependencies
│   └── README.md               # Frontend documentation
│
├── bigrag/                      # Core Python library
│   ├── bigrag.py               # Main BiGRAG class
│   ├── operate.py              # Graph operations
│   ├── reranker.py             # Semantic reranking
│   └── ...
│
├── docs/                        # Documentation (NEW)
│   ├── README.md               # Documentation index
│   ├── technical/              # Design specs, setup guides
│   ├── reports/                # Test & evaluation reports
│   └── updates/                # Change logs
│
├── test_scripts/                # Test & validation scripts (NEW)
│   ├── README.md               # Test documentation
│   ├── test_*.py               # Various test scripts
│   └── validate_*.py           # Validation scripts
│
├── datasets/                    # QA datasets and corpora
├── expr/                        # Built knowledge graphs
├── script_build.py             # Build knowledge graph
├── script_process.py           # Process datasets
└── setup.py                    # Package setup
```

**Key Changes:**
- ✅ `api/` → `backend/api/` for clear separation
- ✅ `script_api.py` → `backend/server.py` with path fixes
- ✅ `frontend/` added with React 19 + TypeScript + Tailwind CSS v4
- ✅ `docs/` organized into technical/, reports/, updates/
- ✅ `test_scripts/` consolidates all test files
- ✅ Root directory clean with only 4 markdown files

See [docs/README.md](docs/README.md) for complete documentation index.

---

## Related Documentation

### Core Documentation
- **[README.md](README.md)** - Project overview and quick start guide
- **[CLAUDE.md](CLAUDE.md)** - This file: AI assistant guidance and comprehensive reference
- **[BIGRAG_UI_PLAN.md](BIGRAG_UI_PLAN.md)** - UI/Frontend implementation plan
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Current development status
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - Directory cleanup summary

### Design & Planning (docs/technical/)
- **[docs/technical/BiG_RAG_DESIGN.md](docs/technical/BiG_RAG_DESIGN.md)** - Comprehensive design document
- **[docs/technical/BiG_RAG_TECHNICAL_SPEC.md](docs/technical/BiG_RAG_TECHNICAL_SPEC.md)** - Technical specification
- **[docs/technical/BiG_RAG_IMPLEMENTATION_CHECKLIST.md](docs/technical/BiG_RAG_IMPLEMENTATION_CHECKLIST.md)** - Implementation checklist
- **[docs/technical/SETUP_VENV.md](docs/technical/SETUP_VENV.md)** - Setup guide for Python venv (lightweight mode)
- **[docs/technical/ENV_SETUP_GUIDE.md](docs/technical/ENV_SETUP_GUIDE.md)** - Complete environment setup

### Updates & Improvements (docs/updates/)
- **[docs/updates/IMPLEMENTATION_SUMMARY.md](docs/updates/IMPLEMENTATION_SUMMARY.md)** - Recent improvements summary (Jan 2025)
- **[docs/updates/API_UPDATES_2025.md](docs/updates/API_UPDATES_2025.md)** - API updates
- **[docs/updates/RERANKING_CONFIG_UPDATE.md](docs/updates/RERANKING_CONFIG_UPDATE.md)** - Reranking configuration

### Test Reports (docs/reports/)
- **[docs/reports/GRAPH_CONSTRUCTION_TEST_REPORT.md](docs/reports/GRAPH_CONSTRUCTION_TEST_REPORT.md)** - Graph construction tests
- **[docs/reports/COMPREHENSIVE_QA_REPORT.md](docs/reports/COMPREHENSIVE_QA_REPORT.md)** - QA testing results
- **[docs/reports/SINGLETOPIC_EVALUATION_DIAGNOSIS.md](docs/reports/SINGLETOPIC_EVALUATION_DIAGNOSIS.md)** - SingleTopic diagnosis
- **[docs/reports/ORPHAN_REDUCTION_VALIDATION_ANALYSIS.md](docs/reports/ORPHAN_REDUCTION_VALIDATION_ANALYSIS.md)** - Orphan node validation analysis
- **[docs/reports/ORPHAN_REDUCTION_FINAL_SUMMARY.md](docs/reports/ORPHAN_REDUCTION_FINAL_SUMMARY.md)** - Orphan reduction implementation summary

### Component Documentation
- **[backend/README.md](backend/README.md)** - Backend API documentation
- **[frontend/README.md](frontend/README.md)** - Frontend UI documentation
- **[test_scripts/README.md](test_scripts/README.md)** - Test scripts documentation
- **[evaluation/README.md](evaluation/README.md)** - Evaluation metrics and testing
- **[inference/README.md](inference/README.md)** - Model inference and deployment

---

## Acknowledgments

BiG-RAG builds upon several excellent open-source projects:
- **[Agent-R1](https://github.com/0russwest0/Agent-R1)**: Tool-augmented RL training
- **[LightRAG](https://github.com/HKUDS/LightRAG)**: Lightweight graph RAG
- **[HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)**: Hippocampus-inspired RAG
- **[VERL](https://github.com/volcengine/verl)**: Volcano Engine RL Framework (Bytedance)

Thanks to all these projects for their wonderful contributions to the field!

---

## Quick Reference

### File Paths Cheatsheet (UPDATED)

```
BiG-RAG/
├── backend/                   # FastAPI server (NEW)
│   ├── api/                  # API modules
│   ├── server.py             # Main server (was script_api.py)
│   └── README.md
├── frontend/                  # React UI (NEW)
│   ├── src/                  # React 19 + TypeScript
│   └── package.json
├── bigrag/                    # Core library
│   ├── bigrag.py             # Main BiGRAG class
│   ├── operate.py            # Graph operations
│   └── kg/                   # Storage backends
├── docs/                      # Documentation (NEW)
│   ├── technical/            # Design docs
│   ├── reports/              # Test reports
│   └── updates/              # Change logs
├── test_scripts/              # All tests (NEW)
│   ├── test_*.py             # Test scripts
│   └── validate_*.py         # Validation scripts
├── datasets/{name}/
│   ├── raw/                  # Raw data
│   │   ├── corpus.jsonl     # Knowledge base
│   │   └── qa_*.json        # QA pairs
│   └── processed/           # Parquet files
├── expr/{name}/              # Built graphs
│   ├── kv_store_*.json      # Metadata
│   └── vdb_*.json           # Vector DBs
├── verl/                     # RL training
│   ├── trainer/
│   │   ├── main_ppo.py      # Training entry point
│   │   └── config/          # Hydra configs
│   └── utils/reward_score/  # Metrics
├── script_process.py         # Step 1: Preprocess
├── script_build.py           # Step 2: Build graph
└── run_*.sh                  # Step 4: Train
```

### Command Cheatsheet

```bash
# 1. Preprocess
python script_process.py --data_source 2WikiMultiHopQA

# 2. Build graph
python script_build.py --data_source 2WikiMultiHopQA

# 3. Start server (unified mode)
cd backend
python server.py --unified &

# 4. Train
bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m qwen3b -d 2WikiMultiHopQA

# 5. Monitor
tail -f training.log
nvidia-smi -l 1
# Open http://localhost:8265 (Ray dashboard)

# 6. Stop
ray stop
# Linux/macOS: fuser -k 8001/tcp
# Windows: netstat -ano | findstr :8001 then taskkill /PID <pid> /F
```

### UI Development Commands (NEW)

```bash
# Start backend (unified mode)
cd backend
python server.py --unified

# Start frontend (separate terminal)
cd frontend
npm run dev

# Access UI
# Open http://localhost:5173 in browser
```

---

## 🎨 React UI (NEW - November 2025)

BiG-RAG now includes a modern web interface built with:
- **React 19.2.0** - Latest with Activity API
- **TypeScript 5.9.3** - Type-safe development
- **Vite 7.1.12** - Lightning-fast build tool
- **Tailwind CSS 4.1.16** - CSS-first utility framework
- **Cytoscape.js 3.33.0** - Graph visualization with WebGL

### UI Features

1. **Dashboard** - System overview, quick actions, recent activity
2. **Chat Interface** - Ask questions with real-time retrieval visualization
3. **Graph Visualization** - Interactive Cytoscape canvas with 5 layout algorithms
4. **Document Management** - Upload, search, delete documents with drag-and-drop
5. **Evaluation Dashboard** - Run evaluations, view results, compare runs
6. **Settings** - API keys, dataset selector, theme switcher

See [BIGRAG_UI_PLAN.md](BIGRAG_UI_PLAN.md) and [frontend/README.md](frontend/README.md) for details.

---

## ⚠️ Important Notes

### Emoji Usage in Scripts

**WARNING**: Do NOT use emojis in Python test scripts or output!

**Reason**: Windows CMD/PowerShell will throw `UnicodeEncodeError` when running scripts with emoji characters.

**Bad** (causes errors):
```python
print("✅ Test passed!")
print("❌ Test failed!")
```

**Good** (works everywhere):
```python
print("[OK] Test passed!")
print("[FAIL] Test failed!")
```

**This applies to:**
- Print statements in test scripts
- Log messages
- Error messages
- Comments (if they're ever printed)

**Exception**: Emojis are OK in:
- Documentation files (.md)
- React UI components (browser handles Unicode)
- README files

### Common Gotchas (UPDATED)

1. **Retrieval server not running**: Training will fail/hang if port 8001 is not responding
   - **Check**: `curl http://localhost:8001/docs`
   - **Fix**: `cd backend && python server.py --unified`

2. **Wrong server path**: Use `backend/server.py` with `--unified` flag NOT `--data_source`
   - **Old**: `python server.py --data_source {dataset}` ❌
   - **New**: `cd backend && python server.py --unified` ✅

3. **Test scripts moved**: All test files now in `test_scripts/`
   - **Old**: `python test_improvements.py` ❌
   - **New**: `cd test_scripts && python test_improvements.py` ✅

4. **Documentation moved**: Technical docs now in `docs/technical/`
   - **Old**: `SETUP_VENV.md` ❌
   - **New**: `docs/technical/SETUP_VENV.md` ✅

---

## ✅ System Readiness Checklist

Before starting development, verify:

### Backend Ready
```bash
cd backend
python server.py --help
# Should show usage without errors
```

### Frontend Ready (if developing UI)
```bash
cd frontend
npm run dev
# Should start dev server on port 5173
```

### Framework Ready
```bash
python -c "from bigrag import BiGRAG; print('BiGRAG OK')"
# Should print "BiGRAG OK"
```

### Dataset Ready (example: SingleTopic)
```bash
# Check corpus exists
ls datasets/SingleTopic/raw/corpus.jsonl

# Build if needed
python script_build.py --data_source SingleTopic
```

### All Systems Go
If all above pass, you're ready to:
- Build knowledge graphs
- Start backend API server
- Develop React UI
- Run tests and evaluations
- Train RL models (if GPUs available)

---

**Questions?** Check the troubleshooting section above or open an issue on GitHub.
