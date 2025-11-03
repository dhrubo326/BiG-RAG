# BiG-RAG Dataset Preparation Guide

**Complete guide to preparing, structuring, and managing datasets for BiG-RAG knowledge graph construction**

**Version:** 1.0
**Last Updated:** 2025-10-30

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Datasets and Corpora](#understanding-datasets-and-corpora)
3. [Dataset Structure](#dataset-structure)
4. [Available Tools](#available-tools)
5. [Complete Pipelines](#complete-pipelines)
6. [Building Custom Datasets](#building-custom-datasets)
7. [Extending Existing Datasets](#extending-existing-datasets)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Advanced Topics](#advanced-topics)

---

## Introduction

**Why This Guide Matters:**

Creating and managing datasets properly is **one of the most crucial parts** of the BiG-RAG framework. Without proper dataset preparation:
- ❌ The indexing pipeline will fail
- ❌ Entity extraction will produce poor results
- ❌ Retrieval will be ineffective
- ❌ Model training will not converge

**This guide covers:**
- ✅ What datasets and corpora are
- ✅ How to structure them correctly
- ✅ Which tools to use and when
- ✅ Complete step-by-step workflows
- ✅ Troubleshooting common issues

---

## Understanding Datasets and Corpora

### What is a Corpus?

A **corpus** (plural: *corpora*) is a collection of text documents that serves as the **raw knowledge source** for BiG-RAG.

**Think of it as**: A library of information from which the system retrieves relevant facts to answer questions.

**Key Characteristics:**
- **Format**: JSONL (JSON Lines) - one JSON object per line
- **Content**: Domain-specific or general knowledge documents
- **Size**: Typically 10-100K documents (but can scale to millions)
- **Purpose**: Provides factual grounding for question answering

**Example Corpus Entry:**
```json
{
  "id": "doc_001",
  "contents": "Albert Einstein was born on March 14, 1879, in Ulm, Germany. He developed the theory of relativity and won the Nobel Prize in Physics in 1921.",
  "title": "Albert Einstein Biography",
  "metadata": {
    "source": "Wikipedia",
    "date": "2024-01-15"
  }
}
```

### What is a Dataset?

A **dataset** in BiG-RAG consists of:
1. **Corpus**: The knowledge base (text documents)
2. **QA Pairs**: Question-answer pairs for training/evaluation

**Relationship:**
```
Dataset = Corpus (knowledge base) + QA Pairs (questions + ground truth answers)
```

**During Training:**
- Model sees **questions** from QA pairs
- Model queries **corpus** (via knowledge graph) to find answers
- Model evaluated by comparing outputs to **ground truth answers**

---

## Dataset Structure

BiG-RAG uses a **standardized directory structure** for all datasets:

```
datasets/{dataset_name}/
├── raw/                          # Raw, unprocessed data
│   ├── corpus.jsonl              # ✅ REQUIRED: Knowledge base
│   ├── qa_train.json             # ✅ REQUIRED: Training questions
│   ├── qa_dev.json               # ✅ REQUIRED: Development questions
│   └── qa_test.json              # ✅ REQUIRED: Test questions
│
└── processed/                    # Preprocessed data (auto-generated)
    ├── train.parquet             # Processed training data
    ├── dev.parquet               # Processed dev data
    └── test.parquet              # Processed test data
```

### File Format Specifications

#### 1. `corpus.jsonl` (Knowledge Base)

**Format**: JSONL (one JSON per line)

**Required Fields:**
```json
{
  "id": "unique_document_id",
  "contents": "The actual text content of the document"
}
```

**Optional Fields:**
```json
{
  "title": "Document title",
  "url": "Source URL",
  "metadata": {
    "author": "...",
    "date": "...",
    "category": "..."
  }
}
```

**Example corpus.jsonl:**
```jsonl
{"id": "doc_001", "contents": "Paris is the capital of France, located on the Seine River.", "title": "Paris"}
{"id": "doc_002", "contents": "The Eiffel Tower was built in 1889 for the World's Fair.", "title": "Eiffel Tower"}
{"id": "doc_003", "contents": "Napoleon Bonaparte was born in Corsica in 1769.", "title": "Napoleon"}
```

**Important Notes:**
- ✅ Each document on a **single line** (JSONL format)
- ✅ `contents` field must be **non-empty**
- ✅ `id` must be **unique** across corpus
- ❌ Do NOT use pretty-printed JSON (multi-line)

#### 2. `qa_train.json`, `qa_dev.json`, `qa_test.json` (QA Pairs)

**Format**: JSON array

**Required Fields:**
```json
[
  {
    "question": "What is the capital of France?",
    "golden_answers": ["Paris"]
  },
  {
    "question": "When was the Eiffel Tower built?",
    "golden_answers": ["1889", "in 1889"]
  }
]
```

**Important Notes:**
- ✅ `golden_answers` is an **array** (supports multiple correct answers)
- ✅ Questions should be **answerable from corpus**
- ✅ Multi-hop questions require reasoning across multiple documents
- ✅ Typically split: 80% train, 10% dev, 10% test

---

## Available Tools

BiG-RAG provides **4 tools** for dataset preparation:

### Tool Comparison

| Tool | Purpose | Input | Output | When to Use |
|------|---------|-------|--------|-------------|
| **convert_text_to_corpus.py** | Convert text files to corpus | `.txt`, `.md` files or directory | `corpus.jsonl` | You have plain text files |
| **script_process.py** | Preprocess QA pairs | `qa_*.json` | `*.parquet` | Always (required step) |
| **script_build.py** | Build knowledge graph from corpus | `corpus.jsonl` | KG files in `expr/` | Always (required step) |
| **build_kg_from_corpus.py** | Alternative KG builder | `corpus.jsonl` | KG files in `expr/` | Same as script_build.py |

### Tool 1: `convert_text_to_corpus.py`

**Purpose**: Convert plain text files to BiG-RAG corpus format (JSONL)

**Features:**
- ✅ Single file or entire directory
- ✅ Supports `.txt`, `.md`, `.text` extensions
- ✅ Split large documents by paragraphs or sentences
- ✅ Auto-generates unique IDs
- ✅ UTF-8 encoding handling

**Basic Usage:**
```bash
# Convert directory of text files
python convert_text_to_corpus.py \
  --input-dir my_documents/ \
  --output datasets/my_data/raw/corpus.jsonl

# Convert single file
python convert_text_to_corpus.py \
  --input article.txt \
  --output datasets/my_data/raw/corpus.jsonl
```

**Advanced Usage (Split Large Documents):**
```bash
# Split by paragraphs (recommended for books/long documents)
python convert_text_to_corpus.py \
  --input large_book.txt \
  --split-by-paragraphs \
  --min-paragraph-length 200 \
  --output datasets/book/raw/corpus.jsonl

# Split by sentences
python convert_text_to_corpus.py \
  --input document.txt \
  --split-by-sentences \
  --sentences-per-chunk 5 \
  --output datasets/data/raw/corpus.jsonl
```

**Parameters:**
- `--input` or `--input-dir`: Source file(s)
- `--output`: Output JSONL file path
- `--split-by-paragraphs`: Split large docs by paragraphs
- `--split-by-sentences`: Split large docs by sentences
- `--min-paragraph-length`: Minimum chars per paragraph (default: 100)
- `--sentences-per-chunk`: Sentences per chunk when splitting (default: 3)

### Tool 2: `script_process.py`

**Purpose**: Preprocess QA pairs for training (convert JSON → Parquet)

**What it does:**
1. Loads `qa_train.json`, `qa_dev.json`, `qa_test.json`
2. Adds instruction template with `<think>`, `<query>`, `<answer>` tags
3. Formats for RL training framework (VERL)
4. Converts to Parquet (efficient columnar format)

**Usage:**
```bash
python script_process.py --data_source MyDataset
```

**Output:**
```
datasets/MyDataset/processed/
├── train.parquet
├── dev.parquet
└── test.parquet
```

**Important**: This step is **REQUIRED** before training, even if you have custom datasets.

### Tool 3: `script_build.py`

**Purpose**: Build bipartite knowledge graph from corpus

**What it does:**
1. Loads `corpus.jsonl`
2. Chunks documents (1200 tokens, 100 overlap)
3. Extracts entities via GPT-4o-mini
4. Extracts n-ary relations
5. Builds bipartite graph (NetworkX)
6. Generates embeddings (OpenAI text-embedding-3-large)
7. Creates FAISS indices for fast retrieval
8. Saves to `expr/{dataset}/`

**Usage:**
```bash
# Basic (uses GPT-4o-mini + text-embedding-3-large)
python script_build.py --data_source MyDataset

# Custom batch size (faster with more API concurrency)
python script_build.py --data_source MyDataset --batch-size 10
```

**Requirements:**
- OpenAI API key in `openai_api_key.txt` or `OPENAI_API_KEY` env var
- Corpus file at `datasets/{dataset}/raw/corpus.jsonl`

**Output Files:**
```
expr/MyDataset/
├── kv_store_full_docs.json            # Full document metadata
├── kv_store_text_chunks.json          # Text chunk metadata
├── kv_store_llm_response_cache.json   # LLM response cache (optional)
├── vdb_entities.json                  # Entity embeddings (NanoVectorDB)
├── vdb_bipartite_edges.json           # Relation embeddings (NanoVectorDB)
├── vdb_chunks.json                    # Chunk embeddings for Path C retrieval
└── graph_chunk_entity_relation.graphml # Bipartite graph structure (NetworkX)
```

**Note**: Entity and relation **metadata** (names, descriptions, source_ids, weights) are stored in the GraphML file, not in separate JSON files.

**Time Estimate**: 10-30 minutes for ~1K documents (depends on API speed)

### Tool 4: `build_kg_from_corpus.py`

**Purpose**: Alternative knowledge graph builder (similar to script_build.py)

**Differences from script_build.py:**
- More verbose logging
- Additional retry logic
- Batch processing options

**Usage:**
```bash
python build_kg_from_corpus.py --data-source MyDataset --batch-size 10
```

**Note**: Choose **either** script_build.py **or** build_kg_from_corpus.py (not both). They produce the same output.

---

## Complete Pipelines

### Pipeline 1: Text Files → Knowledge Graph → Training

**Use case**: You have plain text files (articles, documents, books)

```bash
# Step 1: Convert text files to corpus
python convert_text_to_corpus.py \
  --input-dir my_documents/ \
  --output datasets/my_data/raw/corpus.jsonl

# Step 2: Create QA pairs manually (see section below)
# Edit: datasets/my_data/raw/qa_train.json
# Edit: datasets/my_data/raw/qa_dev.json
# Edit: datasets/my_data/raw/qa_test.json

# Step 3: Preprocess QA pairs
python script_process.py --data_source my_data

# Step 4: Set OpenAI API key
echo "sk-your-api-key" > openai_api_key.txt

# Step 5: Build knowledge graph
python script_build.py --data_source my_data

# Step 6: Start retrieval server
python script_api.py --data_source my_data &

# Step 7: Train model
bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m qwen3b -d my_data
```

### Pipeline 2: Existing Corpus → Knowledge Graph

**Use case**: You already have `corpus.jsonl` and QA pairs

```bash
# Verify files exist
ls datasets/my_data/raw/
# Should show: corpus.jsonl, qa_train.json, qa_dev.json, qa_test.json

# Step 1: Preprocess QA pairs
python script_process.py --data_source my_data

# Step 2: Build knowledge graph
python script_build.py --data_source my_data

# Step 3: Start retrieval server
python script_api.py --data_source my_data &

# Step 4: Train
bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m qwen3b -d my_data
```

### Pipeline 3: Incremental Updates

**Use case**: Adding new documents to existing knowledge graph

**Option A: Rebuild from scratch (simpler)**
```bash
# 1. Append new documents to corpus.jsonl
cat new_docs.jsonl >> datasets/my_data/raw/corpus.jsonl

# 2. Rebuild graph
python script_build.py --data_source my_data

# 3. Restart server
pkill -f "script_api.*my_data"
python script_api.py --data_source my_data &
```

**Option B: Incremental insert (Python API)**
```python
from bigrag import BiGRAG

# Load existing graph
rag = BiGRAG(working_dir="expr/my_data")

# Add new documents
new_docs = [
    "New article about quantum computing",
    "Another article about blockchain"
]

# Insert (auto-chunks, extracts, embeds, updates graph)
rag.insert(new_docs)

# Save
# (Graph auto-saves on insert completion)

# Query immediately
result = rag.query("What is quantum computing?")
print(result)
```

---

## Building Custom Datasets

### Complete Step-by-Step Guide

#### Step 1: Choose a Dataset Name

```bash
export DATASET_NAME="MyCustomDataset"
```

**Naming Guidelines:**
- Use CamelCase (e.g., `MedicalQA`, `LegalDocs`, `TechSupport`)
- No spaces or special characters
- Descriptive and memorable

#### Step 2: Create Directory Structure

```bash
mkdir -p datasets/$DATASET_NAME/raw
mkdir -p datasets/$DATASET_NAME/processed
```

#### Step 3: Prepare Corpus

**Option A: From Text Files**

```bash
# Collect your text files in a directory
my_docs/
├── article1.txt
├── article2.md
├── notes.txt
└── book.txt

# Convert to corpus
python convert_text_to_corpus.py \
  --input-dir my_docs/ \
  --output datasets/$DATASET_NAME/raw/corpus.jsonl
```

**Option B: Create corpus.jsonl Programmatically**

```python
import json

corpus = [
    {
        "id": "doc_001",
        "contents": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "title": "Machine Learning Introduction",
        "metadata": {"category": "AI", "level": "beginner"}
    },
    {
        "id": "doc_002",
        "contents": "Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "title": "Deep Learning Basics",
        "metadata": {"category": "AI", "level": "intermediate"}
    },
    # Add more documents...
]

# Write to JSONL
with open(f"datasets/{DATASET_NAME}/raw/corpus.jsonl", "w", encoding="utf-8") as f:
    for doc in corpus:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"Created corpus with {len(corpus)} documents")
```

**Corpus Quality Guidelines:**
- ✅ Each document should be **self-contained** (complete sentences)
- ✅ Length: **100-2000 tokens** per document (longer docs get auto-chunked)
- ✅ Include **diverse information** relevant to your questions
- ✅ Avoid **duplicate** or near-duplicate documents
- ✅ Use **natural language** (avoid keyword lists)
- ✅ Ensure **factual accuracy**

**Poor Corpus Examples:**
- ❌ Tables without context (just numbers)
- ❌ Code snippets (unless task is code QA)
- ❌ Extremely short documents (<20 tokens)
- ❌ Highly repetitive content
- ❌ Broken formatting (HTML tags, escape characters)

#### Step 4: Prepare QA Pairs

**Create Training/Dev/Test Splits:**

```python
import json

# Create QA pairs
qa_pairs = [
    {
        "question": "What is machine learning?",
        "golden_answers": [
            "A subset of artificial intelligence",
            "A method that enables computers to learn from data"
        ]
    },
    {
        "question": "How does deep learning work?",
        "golden_answers": [
            "Uses artificial neural networks with multiple layers",
            "Extracts higher-level features progressively"
        ]
    },
    # Add more questions (aim for 100+ total)
]

# Split: 80% train, 10% dev, 10% test
n_train = int(len(qa_pairs) * 0.8)
n_dev = int(len(qa_pairs) * 0.1)

train_data = qa_pairs[:n_train]
dev_data = qa_pairs[n_train:n_train + n_dev]
test_data = qa_pairs[n_train + n_dev:]

# Save splits
with open(f"datasets/{DATASET_NAME}/raw/qa_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open(f"datasets/{DATASET_NAME}/raw/qa_dev.json", "w") as f:
    json.dump(dev_data, f, indent=2)

with open(f"datasets/{DATASET_NAME}/raw/qa_test.json", "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Created {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test QA pairs")
```

**QA Pair Guidelines:**
- ✅ Questions should be **answerable from corpus**
- ✅ Include **multiple acceptable answer forms** in `golden_answers`
- ✅ Balance question difficulty (mix easy/hard, single-hop/multi-hop)
- ✅ Minimum **~100 questions per split** for meaningful evaluation
- ✅ Questions should be **natural language** (not keyword queries)

**Question Types:**
- **Single-hop**: Answerable from one document
  - Example: "What is the capital of France?" → "Paris"
- **Multi-hop**: Requires reasoning across multiple documents
  - Example: "Who is the spouse of the director of Nosferatu?" → Requires finding director, then finding spouse

#### Step 5: Preprocess Dataset

```bash
python script_process.py --data_source $DATASET_NAME
```

**Verify Output:**
```python
import pandas as pd

# Check processed data
df = pd.read_parquet(f"datasets/{DATASET_NAME}/processed/train.parquet")
print(df.head())
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
```

#### Step 6: Build Knowledge Graph

```bash
# Set OpenAI API key (one time)
echo "sk-your-api-key-here" > openai_api_key.txt

# Build graph (takes 10-30 minutes for ~1K docs)
python script_build.py --data_source $DATASET_NAME
```

**Monitor Progress:**
```bash
# Watch logs
tail -f build_graph.log

# Check output files
ls -lh expr/$DATASET_NAME/
```

#### Step 7: Verify Graph Construction

```bash
# Check all files exist
ls expr/$DATASET_NAME/

# Expected output:
# kv_store_full_docs.json
# kv_store_text_chunks.json
# kv_store_llm_response_cache.json
# vdb_entities.json
# vdb_bipartite_edges.json
# vdb_chunks.json
# graph_chunk_entity_relation.graphml

# Check statistics
python -c "
import json
import networkx as nx

# Count entities and relations from GraphML
G = nx.read_graphml('expr/$DATASET_NAME/graph_chunk_entity_relation.graphml')
entities = sum(1 for _, attrs in G.nodes(data=True) if attrs.get('role') == 'entity')
relations = sum(1 for _, attrs in G.nodes(data=True) if attrs.get('role') == 'bipartite_edge')
print(f'Entities: {entities}')
print(f'Relations: {relations}')

# Count text chunks
with open('expr/$DATASET_NAME/kv_store_text_chunks.json') as f:
    chunks = json.load(f)
print(f'Text Chunks: {len(chunks)}')
"
```

#### Step 8: Test Retrieval

```bash
# Start server
python script_api.py --data_source $DATASET_NAME &

# Wait for server to start
sleep 5

# Test query
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is machine learning?"], "top_k": 5}'

# Stop server when done testing
pkill -f "script_api.*$DATASET_NAME"
```

#### Step 9: Train Model

```bash
# Start server (keep running during training)
python script_api.py --data_source $DATASET_NAME &

# Train with GRPO
bash run_grpo.sh \
  -p Qwen/Qwen2.5-3B-Instruct \
  -m Qwen2.5-3B \
  -d $DATASET_NAME

# Monitor training
tail -f training.log
```

---

## Extending Existing Datasets

### Scenario 1: Adding New Documents

**When**: New information becomes available, need to expand knowledge base

**Steps:**

1. **Append to corpus.jsonl:**
```python
import json

new_docs = [
    {"id": "doc_new_001", "contents": "New information about quantum computing..."},
    {"id": "doc_new_002", "contents": "More information about blockchain..."}
]

with open("datasets/MyDataset/raw/corpus.jsonl", "a", encoding="utf-8") as f:
    for doc in new_docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"Added {len(new_docs)} new documents")
```

2. **Rebuild knowledge graph:**
```bash
python script_build.py --data_source MyDataset
```

3. **Restart retrieval server:**
```bash
pkill -f "script_api.*MyDataset"
python script_api.py --data_source MyDataset &
```

### Scenario 2: Adding New QA Pairs

**When**: Expanding training set, adding new evaluation questions

**Steps:**

1. **Append to qa_train.json:**
```python
import json

# Load existing
with open("datasets/MyDataset/raw/qa_train.json") as f:
    qa_data = json.load(f)

# Add new questions
qa_data.extend([
    {"question": "What is quantum entanglement?", "golden_answers": ["A phenomenon where particles become correlated"]},
    {"question": "How does blockchain work?", "golden_answers": ["A distributed ledger technology"]},
])

# Save
with open("datasets/MyDataset/raw/qa_train.json", "w") as f:
    json.dump(qa_data, f, indent=2)

print(f"Total questions: {len(qa_data)}")
```

2. **Reprocess QA data:**
```bash
python script_process.py --data_source MyDataset
```

3. **Retrain or continue training:**
```bash
bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m Qwen2.5-3B -d MyDataset
```

### Scenario 3: Updating Existing Documents

**When**: Fixing errors, updating outdated information

**Steps:**

1. **Edit corpus.jsonl:**
```python
import json

# Load corpus
docs = []
with open("datasets/MyDataset/raw/corpus.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

# Update specific document
for doc in docs:
    if doc["id"] == "doc_005":
        doc["contents"] = "Updated information here..."

# Rewrite corpus
with open("datasets/MyDataset/raw/corpus.jsonl", "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
```

2. **Rebuild graph:**
```bash
python script_build.py --data_source MyDataset
```

---

## Troubleshooting

### Issue 1: "KeyError: 'contents'" when building graph

**Symptom**: script_build.py fails with KeyError

**Cause**: Corpus JSONL missing required `contents` field

**Fix:**
```python
# Check corpus format
import json
with open("datasets/MyDataset/raw/corpus.jsonl") as f:
    first_doc = json.loads(f.readline())
    print("Fields:", first_doc.keys())
    print("Has 'contents':", 'contents' in first_doc)

# Fix: Rename field
# If you have "text" instead of "contents":
docs = []
with open("datasets/MyDataset/raw/corpus.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        if "text" in doc:
            doc["contents"] = doc.pop("text")  # Rename text → contents
        docs.append(doc)

# Rewrite
with open("datasets/MyDataset/raw/corpus.jsonl", "w") as f:
    for doc in docs:
        f.write(json.dumps(doc) + "\n")
```

### Issue 2: "No module named 'datasets'" when preprocessing

**Symptom**: script_process.py fails with ImportError

**Cause**: Missing Hugging Face datasets library

**Fix:**
```bash
pip install datasets
```

### Issue 3: Empty FAISS indices after build

**Symptom**: FAISS indices have size 0 or ntotal=0

**Cause**: Entity extraction failed or returned no entities

**Fix:**
1. **Check OpenAI API key:**
```bash
cat openai_api_key.txt  # Should show your API key
# Or
echo $OPENAI_API_KEY
```

2. **Verify corpus has substantive content:**
```python
import json
with open("datasets/MyDataset/raw/corpus.jsonl") as f:
    for i, line in enumerate(f):
        doc = json.loads(line)
        content_length = len(doc.get("contents", ""))
        print(f"Doc {i}: {content_length} chars")
        if content_length < 50:
            print(f"  ⚠️ Warning: Very short content")
```

3. **Check build logs for errors:**
```bash
tail -100 build_graph.log | grep -i error
```

### Issue 4: "Connection refused" to port 8001 during training

**Symptom**: Training fails with connection error

**Cause**: Retrieval server not running

**Fix:**
```bash
# Check if server is running
ps aux | grep script_api

# If not running, start it
python script_api.py --data_source MyDataset &

# Wait for startup message
# Expected: "Uvicorn running on http://0.0.0.0:8001"

# Test connection
curl http://localhost:8001/health
```

### Issue 5: "Out of memory" during graph construction

**Symptom**: Process killed or OOM error

**Cause**: Too many documents processed simultaneously

**Fix:**
```bash
# Reduce batch size
python script_build.py --data_source MyDataset --batch-size 3

# Or chunk corpus into smaller files and build separately
split -l 1000 datasets/MyDataset/raw/corpus.jsonl corpus_part_
# Then build each part separately
```

### Issue 6: Retrieval returns empty results

**Symptom**: Query returns [] or empty context

**Cause**: FAISS indices not loaded or embedding mismatch

**Fix:**
```python
# Verify storage files exist and are non-empty
import os
import json
import networkx as nx

dataset = "MyDataset"
files = {
    "graph": f"expr/{dataset}/graph_chunk_entity_relation.graphml",
    "vdb_entities": f"expr/{dataset}/vdb_entities.json",
    "vdb_bipartite_edges": f"expr/{dataset}/vdb_bipartite_edges.json",
    "vdb_chunks": f"expr/{dataset}/vdb_chunks.json",
    "text_chunks": f"expr/{dataset}/kv_store_text_chunks.json"
}

for name, path in files.items():
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"{name}: {'✅' if exists else '❌'} ({size} bytes)")

# Check GraphML contents
if os.path.exists(files["graph"]):
    G = nx.read_graphml(files["graph"])
    entities = sum(1 for _, attrs in G.nodes(data=True) if attrs.get('role') == 'entity')
    relations = sum(1 for _, attrs in G.nodes(data=True) if attrs.get('role') == 'bipartite_edge')
    print(f"Graph: {entities} entities, {relations} relations, {G.number_of_edges()} edges")

# Check vector DB contents
if os.path.exists(files["vdb_entities"]):
    with open(files["vdb_entities"]) as f:
        vdb_data = json.load(f)
        print(f"Entity VDB: {len(vdb_data.get('data', []))} vectors")
```

### Issue 7: Training converges slowly or not at all

**Symptom**: Loss doesn't decrease, EM/F1 scores remain low

**Possible Causes & Fixes:**

1. **Corpus quality is poor:**
   - Check corpus has relevant information
   - Verify entities were extracted correctly

2. **QA pairs don't match corpus:**
   - Ensure questions are answerable from corpus
   - Check a few examples manually

3. **Retrieval server not responding:**
   - Check server logs: `tail -f api.log`
   - Test retrieval manually

4. **Hyperparameters need tuning:**
   - Try smaller learning rate
   - Increase number of training epochs

---

## Best Practices

### Corpus Construction

**1. Document Granularity:**
- ✅ **Good**: One concept or topic per document (100-1000 tokens)
- ❌ **Bad**: Entire books as single documents

**2. Content Quality:**
- ✅ **Good**: Natural language, complete sentences, factual
- ❌ **Bad**: Keyword lists, broken formatting, opinion pieces

**3. Diversity:**
- ✅ **Good**: Varied sources, writing styles, coverage
- ❌ **Bad**: Repetitive content, single source

**4. Metadata:**
- ✅ **Good**: Include source, date, category when available
- ❌ **Bad**: No metadata (harder to debug issues)

### QA Pair Design

**1. Question Clarity:**
- ✅ **Good**: "What year was the Eiffel Tower built?" (specific, clear)
- ❌ **Bad**: "Eiffel Tower info?" (vague, not a question)

**2. Answer Coverage:**
- ✅ **Good**: Multiple acceptable forms: ["1889", "in 1889", "eighteen eighty-nine"]
- ❌ **Bad**: Single rigid answer: ["1889"]

**3. Difficulty Balance:**
- ✅ **Good**: Mix of easy (30%), medium (50%), hard (20%)
- ❌ **Bad**: All easy or all impossible questions

**4. Corpus Alignment:**
- ✅ **Good**: Every question answerable from corpus
- ❌ **Bad**: Questions require external knowledge

### Storage and Organization

**1. Naming Conventions:**
- ✅ **Good**: `MedicalQA`, `LegalDocs`, `TechSupport` (CamelCase, descriptive)
- ❌ **Bad**: `data1`, `test`, `temp` (generic, non-descriptive)

**2. Version Control:**
- ✅ **Good**: Track corpus and QA changes in git
- ❌ **Bad**: No versioning (can't reproduce results)

**3. Documentation:**
- ✅ **Good**: README in dataset folder explaining source, stats, quirks
- ❌ **Bad**: No documentation

### Performance Optimization

**1. API Costs:**
- ✅ Use GPT-4o-mini instead of GPT-4 (15x cheaper)
- ✅ Use text-embedding-3-small instead of large (cheaper, faster)
- ✅ Enable caching (default, but verify it's working)

**2. Build Speed:**
- ✅ Increase batch size: `--batch-size 10` (more parallelism)
- ✅ Pre-chunk long documents before building
- ✅ Use local models (Ollama) for large corpora

**3. Memory Usage:**
- ✅ Smaller chunks: `chunk_token_size=800`
- ✅ Process in batches for very large corpora
- ✅ Use external vector DB (Milvus) for >100K docs

---

## Advanced Topics

### Multi-Hop Dataset Construction

**Definition**: Questions requiring reasoning across multiple documents

**Example:**
```
Question: "Who is the spouse of the director of Nosferatu (1922)?"
Required docs:
  1. Doc about Nosferatu → identifies F.W. Murnau as director
  2. Doc about F.W. Murnau → mentions spouse Enno Patalas
Answer: "Enno Patalas"
```

**Corpus Construction Strategy:**
1. Include **supporting documents** (contain answer components)
2. Add **distractor documents** (seem relevant but aren't)
3. Ensure **reasoning path exists** (entity chains connect docs)

**QA Pair Format:**
```json
{
  "question": "Who is the spouse of the director of Nosferatu (1922)?",
  "golden_answers": ["Enno Patalas"],
  "supporting_facts": [
    ["Nosferatu (1922)", "directed by F.W. Murnau"],
    ["F.W. Murnau", "spouse Enno Patalas"]
  ]
}
```

### Domain-Specific Corpora

**Medical Example:**
```python
medical_corpus = [
    {
        "id": "med_001",
        "contents": "Diabetes mellitus is a metabolic disorder characterized by elevated blood glucose levels...",
        "metadata": {
            "category": "endocrinology",
            "icd_code": "E11",
            "severity": "chronic"
        }
    }
]
```

**Legal Example:**
```python
legal_corpus = [
    {
        "id": "law_001",
        "contents": "Section 230 of the Communications Decency Act provides immunity for website publishers...",
        "metadata": {
            "jurisdiction": "US Federal",
            "statute": "47 U.S.C. § 230",
            "year": "1996"
        }
    }
]
```

### Corpus Update Strategies

**Incremental Updates (Recommended for Large Corpora):**
```python
from bigrag import BiGRAG

# Load existing graph
rag = BiGRAG(working_dir="expr/MyDataset")

# Track processed IDs
processed_ids = set(rag.full_docs.list_keys())  # Get existing doc IDs

# Load new documents
new_docs_to_add = []
with open("datasets/MyDataset/raw/corpus.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        if doc["id"] not in processed_ids:
            new_docs_to_add.append(doc["contents"])

# Insert only new documents
if new_docs_to_add:
    rag.insert(new_docs_to_add)
    print(f"Added {len(new_docs_to_add)} new documents")
```

**Full Rebuild (Simpler but Slower):**
```bash
# Just rebuild entire graph
python script_build.py --data_source MyDataset
```

**When to Use Each:**
- **Incremental**: >10K docs, frequent updates, minimal changes
- **Full rebuild**: <10K docs, major changes, fresh start needed

### Multi-Modal Data Handling

BiG-RAG currently supports **text-only**, but you can preprocess multi-modal data:

**Images → Text:**
```python
# Use OCR or image captioning
from PIL import Image
import pytesseract  # or use API like GPT-4 Vision

def image_to_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Add to corpus
corpus.append({
    "id": "img_001",
    "contents": image_to_text("diagram.png"),
    "metadata": {"source": "image", "original": "diagram.png"}
})
```

**Tables → Text:**
```python
import pandas as pd

def table_to_text(csv_path):
    df = pd.read_csv(csv_path)
    # Convert to natural language descriptions
    text = f"The table contains {len(df)} rows. "
    for _, row in df.iterrows():
        text += f"{row['column1']} has value {row['column2']}. "
    return text
```

**PDFs → Text:**
```python
import PyPDF2

def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text
```

---

## Summary Checklist

### Building a New Dataset (Complete Checklist)

- [ ] **Step 1**: Create directory structure
  ```bash
  mkdir -p datasets/{name}/raw
  mkdir -p datasets/{name}/processed
  ```

- [ ] **Step 2**: Prepare corpus (choose one)
  - [ ] Option A: Use `convert_text_to_corpus.py` for text files
  - [ ] Option B: Create `corpus.jsonl` manually with proper format

- [ ] **Step 3**: Create QA pairs
  - [ ] `qa_train.json` (80% of data)
  - [ ] `qa_dev.json` (10% of data)
  - [ ] `qa_test.json` (10% of data)

- [ ] **Step 4**: Preprocess QA pairs
  ```bash
  python script_process.py --data_source {name}
  ```

- [ ] **Step 5**: Set OpenAI API key
  ```bash
  echo "your-key" > openai_api_key.txt
  ```

- [ ] **Step 6**: Build knowledge graph
  ```bash
  python script_build.py --data_source {name}
  ```

- [ ] **Step 7**: Verify output files in `expr/{name}/`
  - [ ] `kv_store_*.json` files exist
  - [ ] `index_*.bin` files exist
  - [ ] `corpus_*.npy` files exist

- [ ] **Step 8**: Test retrieval
  ```bash
  python script_api.py --data_source {name} &
  curl http://localhost:8001/search -d '{"queries": ["test"]}'
  ```

- [ ] **Step 9**: Train model
  ```bash
  bash run_grpo.sh -p model -m name -d {name}
  ```

---

## Related Documentation

- **[Implementation Structure Guide](IMPLEMENTATION_STRUCTURE_GUIDE.md)** - Complete A-to-Z implementation reference
- **[PART1: Graph Construction](PART1_GRAPH_CONSTRUCTION.md)** - Deep dive into graph building
- **[PART2: Retrieval System](PART2_RETRIEVAL_SYSTEM.md)** - Understanding retrieval mechanics
- **[LLM Configuration Guide](LLM_CONFIGURATION_GUIDE.md)** - Switching LLM providers
- **[Main README](../README.md)** - Project overview
- **[CLAUDE.md](../CLAUDE.md)** - Complete workflow guide

---

## Questions?

**Need help?**
- Check the [Troubleshooting](#troubleshooting) section above
- Review related documentation links
- Open an issue on GitHub
- Check existing issues for similar problems

**Remember**: Dataset preparation is crucial! Take time to ensure your corpus and QA pairs are high quality. The system's performance depends on it.

---

**End of Guide** | Version 1.0 | Last Updated: 2025-10-30
