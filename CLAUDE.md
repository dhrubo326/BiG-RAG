# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Graph-R1** is an end-to-end reinforcement learning framework that combines GraphRAG (knowledge hypergraph retrieval) with LLM reasoning capabilities. The project enables LLMs to iteratively execute a "think–generate query–retrieve subgraph–rethink" reasoning cycle using explicit reward mechanisms within RL training.

**Key Components:**
- **bigrag/**: Core GraphRAG implementation with n-ary relation extraction and knowledge hypergraph management
- **verl/**: Volcano Engine RL Framework (Bytedance) - distributed RL training with PPO/GRPO/REINFORCE++
- **agent/**: Tool-based agent system for iterative retrieval during training
- **evaluation/**: Metrics computation (EM, F1, semantic similarity)

## Environment Setup

Two installation modes available:

### GraphRAG-only mode (venv - lightweight, no RL training)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements_graphrag_only.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Full RL training mode (conda - required for distributed training)
```bash
conda create -n graphr1 python==3.11.11
conda activate graphr1
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install -r requirements.txt
```

**GPU Requirements for Training:**
- Minimum: 4 x 48GB GPUs for 3B parameter models
- Adjust `tensor_model_parallel_size` in training scripts for different GPU configurations

## Common Commands

### Data Pipeline

#### 1. Preprocess datasets to parquet format
```bash
python script_process.py --data_source 2WikiMultiHopQA
# Other datasets: HotpotQA, Musique, NQ, PopQA, TriviaQA
```
**Output:** `datasets/{dataset}/processed/train.parquet`, `test.parquet`, `dev.parquet`

#### 2. Build Knowledge HyperGraph (requires OpenAI API key)
```bash
# Set API key in openai_api_key.txt file first
nohup python -u script_build.py --data_source 2WikiMultiHopQA > result_build_2WikiMultiHopQA.log 2>&1 &
```
**Output:** `expr/{dataset}/kv_store_*.json`, `index*.bin` (FAISS indices)
**Alternative:** Download pre-built graphs from TeraBox link in README

#### 3. Start retrieval server (required before training)
```bash
nohup python -u script_api.py --data_source 2WikiMultiHopQA > result_api_2WikiMultiHopQA.log 2>&1 &
```
**Port:** 8001 (FastAPI server for graph retrieval during training)

#### 4. Run RL training
```bash
# GRPO (recommended starting point)
nohup bash -u run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m Qwen2.5-3B-Instruct -d 2WikiMultiHopQA > result_run.log 2>&1 &

# REINFORCE++
# bash run_rpp.sh -p <model_path> -m <model_name> -d <dataset>

# PPO (requires critic model)
# bash run_ppo.sh -p <model_path> -m <model_name> -d <dataset>
```
**Parameters:**
- `-p`: Model path (HuggingFace model ID or local path)
- `-m`: Model name (for experiment tracking)
- `-d`: Dataset name (must match processed dataset in `datasets/`)

#### 5. Cleanup retrieval server
```bash
fuser -k 8001/tcp  # Linux/macOS
# On Windows: netstat -ano | findstr :8001, then taskkill /PID <pid> /F
```

### Evaluation
```bash
# See evaluation/README.md for detailed instructions
cd evaluation
# Run evaluation scripts specific to trained models
```

### Inference
```bash
# See inference/README.md for detailed instructions
cd inference
# Load trained checkpoints and run inference
```

## Architecture Overview

### Data Flow: Training Pipeline

```
Dataset (Parquet)
    ↓
RL Dataset Loader → batch prompts with masks
    ↓
Actor Rollout (vLLM) → generates with tool calls
    ↓
ToolGenerationManager → iterative retrieval cycle:
    1. Extract <query>...</query> from LLM output
    2. Call retrieval server (port 8001) → get graph context
    3. Format as <knowledge>...</knowledge>
    4. Continue generation until <answer>...</answer>
    ↓
Reward Manager → compute EM/F1 vs ground truth
    ↓
Critic → value estimation (PPO only, not GRPO)
    ↓
PPO/GRPO Algorithm → policy update with GAE advantages
    ↓
Checkpoints saved (configurable frequency)
```

### Knowledge Graph Query Flow

```
GraphR1.aquery(query_text, QueryParam)
    ↓
Tokenize query → encode with FlagEmbedding
    ↓
Vector search modes:
    - local: entity-based retrieval
    - global: relation/hyperedge retrieval
    - hybrid: combines both (default)
    - naive: text chunk retrieval
    ↓
Retrieve top-k results from FAISS indices
    ↓
Format context (combine local + global graph structures)
    ↓
Return prompt-ready context
```

### Tool-based Agent Cycle (During Training)

```
LLM generates text
    ↓
ToolEnv detects <query>tag</query> markers
    ↓
Extract query → call registered tool (search)
    ↓
Search tool → HTTP request to retrieval server (port 8001)
    ↓
Server queries GraphR1 → returns subgraph context
    ↓
Format as <knowledge>context</knowledge>
    ↓
Append to prompt → continue generation
    ↓
Repeat until <answer>tag</answer> or max_turns reached
```

## Configuration System

The project uses **Hydra** for hierarchical configuration management.

### Base Configuration
- **Location:** [verl/trainer/config/ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml)
- **Structure:**
  ```yaml
  data:                    # Dataset paths, batch sizes, token limits
  actor_rollout_ref:       # Actor, rollout (vLLM/HF), reference model
  critic:                  # Critic model settings (PPO only)
  reward_model:            # Optional separate reward model
  algorithm:               # RL hyperparameters (gamma, lam, KL control)
  trainer:                 # Epochs, logging, checkpointing
  tool:                    # Tool configuration (max_turns, markers, env type)
  ```

### Runtime Overrides (CLI)
Training scripts override config via command-line arguments:
```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=datasets/2WikiMultiHopQA/processed/train.parquet \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    trainer.total_epochs=1 \
    tool.env='search'
```

### Key Configuration Parameters

**Data:**
- `max_prompt_length=4096`: Maximum input tokens
- `max_response_length=4096`: Maximum generation tokens
- `max_tool_response_length=4096`: Maximum tool output tokens
- `train_batch_size=128`: Global batch size

**Actor/Rollout:**
- `tensor_model_parallel_size=4`: Number of GPUs for model parallelism
- `gpu_memory_utilization=0.5`: vLLM memory fraction
- `n_repeat=5`: Number of generations per prompt (for reward variance)
- `use_kl_loss=True`: Enable KL penalty in GRPO

**Training:**
- `total_epochs=1`: Training duration
- `test_freq=10`: Evaluation every N epochs
- `save_freq=-1`: Checkpoint frequency (-1 = only at end)
- `n_gpus_per_node=4`: GPUs per machine

**Tool:**
- `env='search'`: Tool environment type (search = graph retrieval)
- `max_turns`: Maximum tool interaction cycles per generation

### Environment Variables
```bash
export VLLM_ATTENTION_BACKEND=XFORMERS        # Attention implementation
export BASE_MODEL="path/to/model"             # Model path for training
export PROJECT_NAME='Graph-R1'                # W&B project name
export EXPERIMENT_NAME="model_dataset_algo"   # W&B experiment name
export HYDRA_FULL_ERROR=1                     # Show full Hydra errors
export CUDA_LAUNCH_BLOCKING=1                 # Synchronous CUDA (for debugging)
```

## Important Implementation Details

### Storage Plugin System (bigrag/)
The GraphRAG layer uses abstract base classes with lazy imports to support multiple backends:

**Graph Storage:**
- `BaseGraphStorage` → NetworkXStorage (default), Neo4JStorage, OracleGraphStorage, MongoKnowledgeGraph

**Vector Storage:**
- `BaseVectorStorage` → NanoVectorDBStorage (default), FaissStorage, MilvusStorage, ChromaStorage, TiDBVectorStorage

**KV Storage:**
- `BaseKVStorage` → JsonKVStorage (default), MongoKVStorage, OracleKVStorage, TiDBKVStorage

To add new backends, inherit from base classes in [bigrag/base.py](bigrag/base.py) and implement required methods.

### Async/Await Pattern
Nearly all bigrag/ operations are async-compatible:
```python
# Correct usage
await graph_r1.insert(text, metadata)
contexts = await graph_r1.aquery(query, param)

# Synchronous wrappers available but discouraged
```

### Tool Integration During Training
Unlike post-hoc retrieval augmentation, Graph-R1 **trains the LLM to generate tool calls**:
1. LLM learns to emit `<query>search terms</query>` tags
2. ToolGenerationManager intercepts these during generation
3. Retrieval happens synchronously within the generation loop
4. LLM continues generation with retrieved context
5. Reward model scores the final answer, providing RL signal for tool use

This creates a tight feedback loop where the model learns when/how to query the knowledge graph.

### Distributed Training with Ray
- **Ray Cluster:** Started with `ray start --head` before training
- **Workers:** Actor, Critic, Rollout, Reward Manager run as separate Ray actors
- **Parallelism:** Combines FSDP (data parallel), Tensor Parallel (model parallel), and Ray (node distribution)
- **Cleanup:** Always run `ray stop` between training runs to prevent resource leaks

### FAISS Index Management
Knowledge graphs are stored as multiple FAISS indices:
- `index_entity.bin`: Entity embeddings (for local retrieval)
- `index_hyperedge.bin`: Hyperedge embeddings (for global retrieval)
- `index_text.bin`: Raw text chunk embeddings (for naive retrieval)

Corresponding metadata in JSON files:
- `kv_store_entity.json`: Entity descriptions
- `kv_store_hyperedge.json`: Relation structures
- `kv_store_text.json`: Original text chunks

## Key Files and Entry Points

### Main Execution Scripts
- [script_process.py](script_process.py): Dataset preprocessing
- [script_build.py](script_build.py): Knowledge graph construction
- [script_api.py](script_api.py): Retrieval server (FastAPI on port 8001)
- [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py): RL training entry point
- [verl/trainer/main_generation.py](verl/trainer/main_generation.py): Inference/generation
- [verl/trainer/main_eval.py](verl/trainer/main_eval.py): Evaluation runner

### Core Library Modules
- [bigrag/graphr1.py](bigrag/graphr1.py): Main GraphR1 class (async insert/query interface)
- [bigrag/operate.py](bigrag/operate.py): Entity extraction, chunking, graph operations
- [bigrag/storage.py](bigrag/storage.py): Default storage implementations
- [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py): Distributed PPO trainer
- [agent/llm_agent/generation.py](agent/llm_agent/generation.py): ToolGenerationManager (tool-calling loop)
- [agent/tool/tool_env.py](agent/tool/tool_env.py): ToolEnv (manages tool state and execution)

### Configuration Files
- [verl/trainer/config/ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml): Base PPO config
- [verl/trainer/config/sft_trainer.yaml](verl/trainer/config/sft_trainer.yaml): Supervised fine-tuning config
- Training scripts: [run_grpo.sh](run_grpo.sh), [run_ppo.sh](run_ppo.sh), [run_rpp.sh](run_rpp.sh)

## Datasets

**Supported:** 2WikiMultiHopQA, HotpotQA, Musique, NQ, PopQA, TriviaQA

**Directory Structure:**
```
datasets/{dataset_name}/
├── raw/
│   ├── corpus.jsonl          # Text corpus for knowledge graph
│   ├── qa_train.json         # Training QA pairs
│   ├── qa_test.json          # Test QA pairs
│   └── qa_dev.json           # Development QA pairs
└── processed/
    ├── train.parquet         # Preprocessed training data
    ├── test.parquet          # Preprocessed test data
    └── dev.parquet           # Preprocessed dev data
```

**Download:** See TeraBox links in README.md for pre-built datasets and knowledge graphs.

## Common Gotchas

1. **Retrieval server must be running:** Training will fail silently or hang if port 8001 is not responding. Always check `ps aux | grep script_api` before training.

2. **Ray cleanup:** If training crashes, run `ray stop` before restarting. Lingering Ray processes consume GPU memory.

3. **CUDA OOM with vLLM:** Reduce `gpu_memory_utilization` (default 0.5) or `tensor_model_parallel_size` if running out of memory.

4. **OpenAI API key:** Required for knowledge graph construction (`script_build.py`). Set in `openai_api_key.txt` file or use pre-built graphs.

5. **Tool markers:** The model must generate exact tags `<query>`, `</query>`, `<answer>`, `</answer>`. If changing these, update `tool.query_start_tag`, `tool.query_end_tag` in config.

6. **Parquet format:** Datasets must be preprocessed to parquet before training. Raw JSON files will not work directly.

7. **Model compatibility:** Only decoder-only models supported (Qwen, Llama, Mistral). Encoder-decoder models (T5, BART) will not work with current vLLM setup.

8. **Windows limitations:** Training scripts use bash and Linux-specific commands (`fuser`, `nohup`). Windows users should use WSL2 or adapt scripts.

## Logging and Debugging

**Console Logging:**
- RL training logs printed to stdout/stderr
- Redirect with `> logfile.log 2>&1` when using nohup

**Weights & Biases:**
- Configured via `trainer.logger=['console','wandb']`
- Tracks: reward curves, EM/F1 scores, KL divergence, loss values
- Project/experiment names set via environment variables

**Ray Dashboard:**
- Access at `http://localhost:8265` when Ray cluster is running
- Shows: GPU utilization, worker status, task execution timeline

**Debug Mode:**
- Set `export CUDA_LAUNCH_BLOCKING=1` for synchronous CUDA execution
- Add `trainer.val_before_train=True` to validate data loading before training
- Use `ppo_micro_batch_size_per_gpu=1` to isolate per-GPU issues

## Evaluation Metrics

**EM (Exact Match):** Percentage of predictions that exactly match ground truth (after normalization)

**F1 Score:** Token-level F1 between prediction and ground truth (measures partial credit)

**Reward Scoring:** Defined in [verl/utils/reward_score/](verl/utils/reward_score/)
- Can combine multiple metrics (EM + F1 + format validation)
- Configurable via `reward_model` section in config

## Related Documentation

- **Full paper:** [arXiv:2507.21892](https://arxiv.org/abs/2507.21892)
- **Evaluation details:** [evaluation/README.md](evaluation/README.md)
- **Inference guide:** [inference/README.md](inference/README.md)
- **Setup guide (venv):** [SETUP_VENV.md](SETUP_VENV.md)

## Acknowledgments

This codebase builds on: Agent-R1, HyperGraphRAG, FlashRAG, LightRAG, HippoRAG2, R1-Searcher, Search-R1, and VERL (Volcano Engine RL Framework by Bytedance).
