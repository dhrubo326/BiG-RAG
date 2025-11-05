# BiG-RAG Requirements Guide

## Overview

BiG-RAG uses a modular requirements structure to support different use cases:

### Root Directory Requirements

1. **`requirements.txt`** (Main - BiG-RAG without RL)
   - For knowledge graph construction and retrieval
   - Works with OpenAI/Claude/Gemini APIs
   - Includes all BiG-RAG algorithmic features
   - **This is what most users need**

2. **`requirements-rl.txt`** (Advanced - RL Training Mode)
   - For reinforcement learning training with small LLMs
   - Includes vLLM, Ray, DeepSpeed for distributed training
   - Only needed if training custom models

### Backend Directory Requirements

**`backend/requirements.txt`**
- Minimal dependencies for the FastAPI server
- FastAPI, Uvicorn, Pydantic only
- BiG-RAG framework must be installed separately from root

## Installation Guide

### Standard Installation (Recommended)

For BiG-RAG knowledge graph and retrieval features:

```bash
# From root directory
pip install -r requirements.txt
```

### Backend API Server

```bash
# From backend directory
cd backend
pip install -r requirements.txt
```

### RL Training Mode (Advanced)

For training LLMs with reinforcement learning:

```bash
# Requires conda environment
conda create -n bigrag-rl python==3.11.11
conda activate bigrag-rl
pip install -r requirements-rl.txt
```

## Which Requirements File Do I Need?

| Use Case | Requirements File | Location |
|----------|------------------|----------|
| Building knowledge graphs | `requirements.txt` | Root |
| Running retrieval API | `requirements.txt` + `backend/requirements.txt` | Root + Backend |
| Testing BiG-RAG | `requirements.txt` | Root |
| Training with RL | `requirements-rl.txt` | Root |

## Log Files Organization

All log files are now organized in the `logs/` directory:

```
logs/
├── api_demo.log          # API server logs for demo dataset
├── api_singletopic.log   # API server logs for SingleTopic
├── build_demo.log        # Graph building logs
├── bigrag.log           # Main BiG-RAG framework logs
└── test_*.log           # Test execution logs
```

Log files are automatically ignored by git (via `.gitignore`).

## Removed Files

The following files were removed during reorganization:
- `requirements_graphrag_only.txt` → Renamed to `requirements.txt`
- `requirements_test.txt` → Not needed (main requirements covers testing)
- `backend/.env` → Using root `.env` only
- `backend/.env.example` → Using root `.env` only

## Notes

1. The root `.env` file is the single source of configuration
2. All paths (INPUT_DIR, WORKING_DIR) are configurable via `.env`
3. Python-dotenv is included in requirements for environment variable management
4. The backend automatically loads the root `.env` file on startup