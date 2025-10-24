# BiG-RAG LLM Configuration Guide

This guide explains how to configure and switch between different LLM providers in BiG-RAG. The framework supports multiple providers with minimal code changes.

## Table of Contents

1. [Overview](#overview)
2. [Supported LLM Providers](#supported-llm-providers)
3. [Configuration Methods](#configuration-methods)
4. [Provider-Specific Setup](#provider-specific-setup)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Overview

BiG-RAG's LLM module ([bigrag/llm.py](../bigrag/llm.py)) provides a **unified interface** for multiple LLM providers:

- **OpenAI** (GPT-4, GPT-4o, GPT-4o-mini)
- **Azure OpenAI**
- **AWS Bedrock** (Claude, etc.)
- **Google Gemini** (via OpenAI-compatible API)
- **DeepSeek** (via OpenAI-compatible API)
- **Anthropic Claude** (via Bedrock)
- **Ollama** (local models)
- **HuggingFace** (local models)
- **Zhipu AI** (ChatGLM)
- **NVIDIA NIM** (Nemotron)
- **LMDeploy** (for inference optimization)

All providers use the same **async interface**, making it easy to switch between them.

---

## Supported LLM Providers

### Cloud Providers

| Provider | Models | API Key Required | Base URL |
|----------|--------|-----------------|----------|
| **OpenAI** | gpt-4, gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| **Azure OpenAI** | Your deployed models | `AZURE_OPENAI_API_KEY` | Custom endpoint |
| **AWS Bedrock** | Claude, Titan | AWS credentials | AWS region endpoint |
| **Zhipu AI** | glm-4, glm-4-flashx | `ZHIPUAI_API_KEY` | `https://open.bigmodel.cn/api/paas/v4` |
| **NVIDIA NIM** | Nemotron-70B | `OPENAI_API_KEY` | `https://integrate.api.nvidia.com/v1` |

### OpenAI-Compatible Providers

These providers use OpenAI's API format with custom `base_url`:

| Provider | Base URL | Notes |
|----------|----------|-------|
| **DeepSeek** | `https://api.deepseek.com/v1` | Use `deepseek-chat` model |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` | Requires Google API key as `OPENAI_API_KEY` |
| **Together AI** | `https://api.together.xyz/v1` | OpenAI-compatible |
| **Perplexity** | `https://api.perplexity.ai` | OpenAI-compatible |

### Local Providers

| Provider | Installation | Models |
|----------|-------------|--------|
| **Ollama** | `ollama pull <model>` | llama3, mistral, qwen, etc. |
| **HuggingFace** | `pip install transformers torch` | Any HF model |
| **LMDeploy** | `pip install lmdeploy` | Optimized inference |

---

## Configuration Methods

### Method 1: Environment Variables (Recommended)

Set environment variables before running BiG-RAG:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"

# Zhipu AI
export ZHIPUAI_API_KEY="..."
```

### Method 2: Direct Function Calls

Use specific completion functions from `bigrag.llm`:

```python
from bigrag.llm import (
    gpt_4o_mini_complete,      # GPT-4o-mini
    gpt_4o_complete,           # GPT-4o
    openai_complete,           # Generic OpenAI
    azure_openai_complete,     # Azure OpenAI
    bedrock_complete,          # AWS Bedrock
    ollama_model_complete,     # Ollama
    hf_model_complete,         # HuggingFace
    zhipu_complete,            # Zhipu AI
)

# Example: GPT-4o-mini
response = await gpt_4o_mini_complete(
    prompt="What is the capital of France?",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=150
)
```

### Method 3: BiGRAG Instance Configuration

Configure LLM when creating BiGRAG instance:

```python
from bigrag import BiGRAG
from bigrag.llm import gpt_4o_mini_complete, ollama_model_complete

# Option A: Use pre-defined function
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=gpt_4o_mini_complete,
    llm_model_name="gpt-4o-mini"  # Optional
)

# Option B: Use custom function with base_url for OpenAI-compatible APIs
from bigrag.llm import openai_complete_if_cache

async def deepseek_complete(prompt, system_prompt=None, **kwargs):
    return await openai_complete_if_cache(
        model="deepseek-chat",
        prompt=prompt,
        system_prompt=system_prompt,
        base_url="https://api.deepseek.com/v1",
        api_key="your-deepseek-api-key",
        **kwargs
    )

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=deepseek_complete
)
```

### Method 4: MultiModel Load Balancing

Distribute requests across multiple API keys or models:

```python
from bigrag.llm import Model, MultiModel, openai_complete_if_cache
import os

# Create multiple model instances
models = [
    Model(
        gen_func=openai_complete_if_cache,
        kwargs={"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY_1"]}
    ),
    Model(
        gen_func=openai_complete_if_cache,
        kwargs={"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY_2"]}
    ),
    Model(
        gen_func=openai_complete_if_cache,
        kwargs={"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY_3"]}
    ),
]

# Create MultiModel instance
multi_model = MultiModel(models)

# Use in BiGRAG
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=multi_model.llm_model_func
)
```

---

## Provider-Specific Setup

### OpenAI (GPT-4o-mini, GPT-4o, GPT-4)

**Setup:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Usage:**
```python
from bigrag.llm import gpt_4o_mini_complete

response = await gpt_4o_mini_complete(
    prompt="Explain quantum computing",
    temperature=0.7,
    max_tokens=500
)
```

**Models:**
- `gpt-4o-mini` - Fast, cost-effective (default for BiG-RAG)
- `gpt-4o` - Most capable
- `gpt-4` - Previous generation

---

### DeepSeek

**Setup:**
```bash
export OPENAI_API_KEY="your-deepseek-api-key"
```

**Usage:**
```python
from bigrag.llm import openai_complete_if_cache

response = await openai_complete_if_cache(
    model="deepseek-chat",
    prompt="Explain machine learning",
    base_url="https://api.deepseek.com/v1",
    temperature=0.7
)
```

**Custom Function:**
```python
async def deepseek_complete(prompt, system_prompt=None, **kwargs):
    from bigrag.llm import openai_complete_if_cache
    return await openai_complete_if_cache(
        model="deepseek-chat",
        prompt=prompt,
        system_prompt=system_prompt,
        base_url="https://api.deepseek.com/v1",
        **kwargs
    )

# Use in BiGRAG
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=deepseek_complete
)
```

---

### Google Gemini

**Setup:**
```bash
export OPENAI_API_KEY="your-google-api-key"
```

**Usage:**
```python
from bigrag.llm import openai_complete_if_cache

response = await openai_complete_if_cache(
    model="gemini-1.5-pro",  # or gemini-1.5-flash
    prompt="What is photosynthesis?",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0.7
)
```

**Custom Function:**
```python
async def gemini_complete(prompt, system_prompt=None, **kwargs):
    from bigrag.llm import openai_complete_if_cache
    return await openai_complete_if_cache(
        model="gemini-1.5-flash",
        prompt=prompt,
        system_prompt=system_prompt,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        **kwargs
    )

rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=gemini_complete
)
```

---

### Anthropic Claude (via AWS Bedrock)

**Setup:**
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

**Usage:**
```python
from bigrag.llm import bedrock_complete

response = await bedrock_complete(
    prompt="Explain relativity",
    system_prompt="You are a physics expert.",
    temperature=0.7,
    max_tokens=500
)
```

**Default Model:** `anthropic.claude-3-haiku-20240307-v1:0`

**Change Model:**
```python
from bigrag.llm import bedrock_complete_if_cache

response = await bedrock_complete_if_cache(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    prompt="Explain relativity",
    temperature=0.7
)
```

---

### Ollama (Local Models)

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3
ollama pull mistral
ollama pull qwen2.5
```

**Usage:**
```python
from bigrag.llm import ollama_model_complete

# Simple usage
response = await ollama_model_complete(
    prompt="What is machine learning?",
    model="llama3",  # or mistral, qwen2.5, etc.
    temperature=0.7
)

# With custom host
response = await ollama_model_complete(
    prompt="Explain quantum physics",
    model="llama3",
    host="http://localhost:11434",  # Default Ollama host
    timeout=30.0
)
```

**BiGRAG Configuration:**
```python
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3"  # Model name
)
```

---

### Azure OpenAI

**Setup:**
```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

**Usage:**
```python
from bigrag.llm import azure_openai_complete

response = await azure_openai_complete(
    prompt="Explain AI",
    temperature=0.7
)
```

**Custom Deployment:**
```python
from bigrag.llm import azure_openai_complete_if_cache

response = await azure_openai_complete_if_cache(
    model="your-deployment-name",  # Your Azure deployment name
    prompt="Explain AI",
    temperature=0.7
)
```

---

### Zhipu AI (ChatGLM)

**Setup:**
```bash
export ZHIPUAI_API_KEY="..."
```

**Usage:**
```python
from bigrag.llm import zhipu_complete

response = await zhipu_complete(
    prompt="解释人工智能",  # Supports Chinese
    temperature=0.7
)
```

**Models:**
- `glm-4-flashx` (default) - Fast, cost-effective
- `glm-4` - Most capable

---

## Advanced Usage

### Custom LLM Function

Create a fully custom LLM function:

```python
async def my_custom_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    """
    Custom LLM function template

    Args:
        prompt: User query
        system_prompt: System instruction (optional)
        history_messages: Chat history (optional)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        str: LLM response
    """
    # Your custom logic here
    # Call external API, local model, etc.

    # Example: Call OpenAI with custom settings
    from bigrag.llm import openai_complete_if_cache

    return await openai_complete_if_cache(
        model="gpt-4o-mini",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 500)
    )

# Use in BiGRAG
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=my_custom_llm
)
```

### Switching Models at Runtime

```python
from bigrag import BiGRAG
from bigrag.llm import gpt_4o_mini_complete, gpt_4o_complete, ollama_model_complete

# Start with GPT-4o-mini
rag = BiGRAG(
    working_dir="expr/2WikiMultiHopQA",
    llm_model_func=gpt_4o_mini_complete
)

# Query with GPT-4o-mini
result1 = await rag.aquery("What is AI?")

# Switch to GPT-4o
rag.llm_model_func = gpt_4o_complete

# Query with GPT-4o
result2 = await rag.aquery("Explain quantum mechanics")

# Switch to local Ollama
rag.llm_model_func = ollama_model_complete
rag.llm_model_name = "llama3"

# Query with Ollama
result3 = await rag.aquery("What is machine learning?")
```

### Model-Specific Parameters

Different providers support different parameters:

```python
# OpenAI
response = await gpt_4o_mini_complete(
    prompt="Test",
    temperature=0.7,        # 0-2
    max_tokens=150,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5
)

# Bedrock
response = await bedrock_complete(
    prompt="Test",
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    stop_sequences=["</answer>"]
)

# Ollama
response = await ollama_model_complete(
    prompt="Test",
    model="llama3",
    temperature=0.7,
    num_predict=150,  # max tokens
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1
)
```

---

## Troubleshooting

### Issue: `ImportError: No module named 'openai'`

**Solution:**
```bash
pip install openai
```

### Issue: `AuthenticationError: Invalid API key`

**Solution:**
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Set API key
export OPENAI_API_KEY="sk-..."
```

### Issue: `RateLimitError: Rate limit exceeded`

**Solution 1: Use MultiModel** (distributes load across multiple keys)
```python
from bigrag.llm import Model, MultiModel
# See MultiModel example above
```

**Solution 2: Add retry logic** (already built-in)
```python
# Retries are automatic with exponential backoff
# Default: 3 retries with 4-10 second wait
```

### Issue: Ollama connection error

**Solution:**
```bash
# Check Ollama is running
ollama list

# Start Ollama server
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Issue: DeepSeek/Gemini not working

**Solution:**
Ensure `base_url` is correctly set:
```python
# DeepSeek
base_url="https://api.deepseek.com/v1"

# Gemini
base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
```

### Issue: Azure OpenAI authentication error

**Solution:**
```bash
# Ensure all three variables are set
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

---

## Quick Reference

### Minimal Configuration for Each Provider

**OpenAI:**
```python
export OPENAI_API_KEY="sk-..."
from bigrag.llm import gpt_4o_mini_complete
response = await gpt_4o_mini_complete("Hello")
```

**DeepSeek:**
```python
export OPENAI_API_KEY="your-deepseek-key"
from bigrag.llm import openai_complete_if_cache
response = await openai_complete_if_cache(
    "deepseek-chat", "Hello", base_url="https://api.deepseek.com/v1"
)
```

**Gemini:**
```python
export OPENAI_API_KEY="your-google-api-key"
from bigrag.llm import openai_complete_if_cache
response = await openai_complete_if_cache(
    "gemini-1.5-flash", "Hello",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

**Claude (Bedrock):**
```python
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
from bigrag.llm import bedrock_complete
response = await bedrock_complete("Hello")
```

**Ollama:**
```bash
ollama pull llama3
```
```python
from bigrag.llm import ollama_model_complete
response = await ollama_model_complete("Hello", model="llama3")
```

---

## See Also

- [bigrag/llm.py](../bigrag/llm.py) - Full LLM implementation
- [script_api.py](../script_api.py) - API server with GPT-4o-mini endpoint
- [CLAUDE.md](../CLAUDE.md) - Complete BiG-RAG documentation
