# BiG-RAG Setup Guide - Python venv Edition

This guide explains how to set up BiG-RAG using Python's built-in `venv` instead of Anaconda.

## Prerequisites

- Python 3.11, 3.12, or 3.13 installed on your system
- Git installed
- Windows, Linux, or macOS

## Setup Steps

### 1. Clone or Navigate to the Repository

```bash
cd d:\BiG-RAG
```

### 2. Create Python Virtual Environment

**On Windows:**
```bash
python -m venv venv
```

**On Linux/macOS:**
```bash
python3 -m venv venv
```

This creates a `venv` folder in your project directory (already added to `.gitignore`).

### 3. Activate the Virtual Environment

**On Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**On Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**On Linux/macOS:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt.

### 4. Upgrade pip (Recommended)

```bash
python -m pip install --upgrade pip
```

### 5. Install PyTorch

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

**For CUDA 11.8 (NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1 (NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 6. Install BiG-RAG Dependencies

```bash
pip install -r requirements_graphrag_only.txt
```

### 7. Download Required NLP Models

**Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

**Download NLTK data:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 8. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# On Windows
copy .env.example .env

# On Linux/macOS
cp .env.example .env
```

Then edit `.env` and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 9. Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; import transformers; import networkx; import faiss; print('All imports successful!')"
```

## Running BiG-RAG

With the virtual environment activated, you can now run BiG-RAG scripts:

```bash
python your_script.py
```

## Deactivating the Virtual Environment

When you're done working:

```bash
deactivate
```

## Reactivating Later

Every time you open a new terminal to work on BiG-RAG, activate the environment again:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

## Troubleshooting

### PowerShell Execution Policy Error

If you get an error about execution policies on Windows PowerShell:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python Version Issues

Check your Python version:

```bash
python --version
```

Make sure it's 3.11, 3.12, or 3.13. If you have multiple Python versions installed, you may need to use `py -3.11 -m venv venv` on Windows.

### Package Installation Failures

If packages fail to install, try:

1. Upgrade pip: `python -m pip install --upgrade pip`
2. Install packages one by one to identify which one is failing
3. Check that you're using a compatible Python version

## Key Differences from Anaconda Setup

- **Environment creation:** `python -m venv venv` instead of `conda create`
- **Activation:** Use OS-specific activation scripts instead of `conda activate`
- **Package management:** All packages installed via `pip` instead of mix of `conda` and `pip`
- **Lighter weight:** venv is minimal compared to Anaconda's full distribution
- **No conda-specific packages:** All dependencies are pip-installable (as specified in requirements_graphrag_only.txt)

## What's Included in requirements_graphrag_only.txt

This file contains ONLY the dependencies needed for BiG-RAG Algorithmic Mode (works with GPT-4, Claude, etc. - no training required).

Excluded from this setup:
- RL training dependencies (verl, vllm, deepspeed, ray)
- Compiled packages that require C++ compiler (gensim, hnswlib, graspologic)

See [requirements_graphrag_only.txt](requirements_graphrag_only.txt) for the complete list of dependencies.
