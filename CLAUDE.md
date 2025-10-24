# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**BiG-RAG** (Bipartite Graph Retrieval-Augmented Generation) is an end-to-end reinforcement learning framework that combines bipartite graph-based knowledge retrieval with LLM reasoning capabilities. The project enables LLMs to iteratively execute a "**think → generate query → retrieve subgraph → rethink**" reasoning cycle using explicit reward mechanisms within RL training.

### Key Features

- **Bipartite Graph Structure**: Unlike traditional hypergraph terminology, the implementation uses a true bipartite graph with documents ↔ entities ↔ relations
- **End-to-End RL Training**: Trains LLMs to actively query knowledge graphs during generation
- **Tool-Augmented Generation**: Models learn to emit structured queries (`<query>...</query>`) to retrieve relevant context
- **Multiple RL Algorithms**: Supports GRPO, REINFORCE++, and PPO
- **Distributed Training**: Built on VERL (Volcano Engine RL Framework) with Ray for multi-GPU/multi-node training

### Key Components

- **[bigrag/](bigrag/)**: Core BiG-RAG implementation
  - Bipartite graph construction from text corpora
  - N-ary relation extraction using LLMs
  - Fast similarity search with FAISS indices
  - Async-first API for insertion and querying

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
pip install -r requirements_graphrag_only.txt

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
pip3 install -r requirements.txt
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
├── kv_store_entities.json          # Entity metadata
├── kv_store_bipartite_edges.json   # Relation metadata
├── kv_store_text_chunks.json       # Text chunk metadata
├── index_entity.bin                # FAISS index for entities
├── index_bipartite_edge.bin        # FAISS index for relations
├── index.bin                       # FAISS index for chunks
├── corpus.npy                      # Chunk embeddings
├── corpus_entity.npy               # Entity embeddings
└── corpus_bipartite_edge.npy       # Edge embeddings
```

**Time Estimate**: 2-4 hours for ~10K documents (depends on corpus size and API rate limits)

**Alternative**: Download pre-built graphs from [TeraBox](https://1024terabox.com/s/1y1G7trP-hcmIDQRUaBaDDw)

#### Step 3: Start Retrieval Server

```bash
# Start server (must run during training)
nohup python -u script_api.py --data_source 2WikiMultiHopQA > api.log 2>&1 &

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
│  ┌──────────┐     Bipartite Edge    ┌──────────┐           │
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

### FAISS Index Management

BiG-RAG stores bipartite graph as multiple FAISS indices for fast similarity search:

**Index Files**:
- `index_entity.bin`: Entity node embeddings (for entity-based retrieval)
- `index_bipartite_edge.bin`: Relation edge embeddings (for relation-based retrieval)
- `index.bin`: Text chunk embeddings (for naive text search)

**Metadata Files** (JSON):
- `kv_store_entities.json`: Entity names, descriptions, source IDs
- `kv_store_bipartite_edges.json`: Relation structures, connected entities
- `kv_store_text_chunks.json`: Original text, chunk metadata

**Query Modes**:
- `local`: Entity-based retrieval only
- `global`: Relation-based retrieval only
- `hybrid`: Combines both (default, most effective)
- `naive`: Direct text chunk retrieval (baseline)

---

## Key Files and Entry Points

### Main Execution Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| [script_process.py](script_process.py) | Preprocess datasets to parquet | `datasets/{dataset}/processed/*.parquet` |
| [script_build.py](script_build.py) | Build bipartite knowledge graph | `expr/{dataset}/kv_store_*.json`, `index*.bin` |
| [script_api.py](script_api.py) | Start retrieval server (FastAPI) | HTTP server on port 8001 |
| [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py) | RL training entry point | Model checkpoints, logs |
| [verl/trainer/main_generation.py](verl/trainer/main_generation.py) | Inference/generation | Generated text |
| [verl/trainer/main_eval.py](verl/trainer/main_eval.py) | Evaluation runner | EM/F1 metrics |

### Core Library Modules

| Module | Purpose |
|--------|---------|
| [bigrag/bigrag.py](bigrag/bigrag.py) | Main `BiGRAG` class (async insert/query) |
| [bigrag/operate.py](bigrag/operate.py) | Entity extraction, chunking, graph operations |
| [bigrag/storage.py](bigrag/storage.py) | Default storage implementations |
| [bigrag/base.py](bigrag/base.py) | Abstract base classes for storage plugins |
| [bigrag/llm.py](bigrag/llm.py) | LLM completion wrappers (OpenAI, HuggingFace) |
| [bigrag/prompt.py](bigrag/prompt.py) | Prompt templates for entity extraction |
| [bigrag/utils.py](bigrag/utils.py) | Utility functions (hashing, encoding, caching) |
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
   - **New**: `index_bipartite_edge.bin`, `kv_store_bipartite_edges.json`
   - If using pre-built graphs, you may need to rename files

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

## Related Documentation

### Core Documentation
- **[README.md](README.md)** - Project overview and quick start guide
- **[DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md)** - Technical details, architecture, and developer guidance
- **[SETUP_VENV.md](SETUP_VENV.md)** - Setup guide for Python venv (lightweight mode)

### Specialized Guides
- **[docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)** - Dataset preparation and corpus building
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

### File Paths Cheatsheet

```
BiG-RAG/
├── bigrag/                      # Core library
│   ├── bigrag.py               # Main BiGRAG class
│   ├── operate.py              # Graph operations
│   └── kg/                     # Storage backends
├── datasets/{name}/
│   ├── raw/                    # Raw data
│   │   ├── corpus.jsonl       # Knowledge base
│   │   └── qa_*.json          # QA pairs
│   └── processed/             # Parquet files
├── expr/{name}/               # Built graphs
│   ├── kv_store_*.json       # Metadata
│   └── index*.bin            # FAISS indices
├── verl/                      # RL training
│   ├── trainer/
│   │   ├── main_ppo.py       # Training entry point
│   │   └── config/           # Hydra configs
│   └── utils/reward_score/   # Metrics
├── script_process.py          # Step 1: Preprocess
├── script_build.py            # Step 2: Build graph
├── script_api.py              # Step 3: Start server
└── run_*.sh                   # Step 4: Train
```

### Command Cheatsheet

```bash
# 1. Preprocess
python script_process.py --data_source 2WikiMultiHopQA

# 2. Build graph
python script_build.py --data_source 2WikiMultiHopQA

# 3. Start server
python script_api.py --data_source 2WikiMultiHopQA &

# 4. Train
bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m qwen3b -d 2WikiMultiHopQA

# 5. Monitor
tail -f training.log
nvidia-smi -l 1
# Open http://localhost:8265 (Ray dashboard)

# 6. Stop
ray stop
fuser -k 8001/tcp
```

---

**Questions?** Check the troubleshooting section above or open an issue on GitHub.
