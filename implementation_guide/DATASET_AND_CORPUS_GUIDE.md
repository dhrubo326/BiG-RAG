# BiG-RAG Dataset and Corpus Guide

**Complete guide to understanding, building, and extending datasets and corpora in BiG-RAG**

---

## Table of Contents

1. [Overview](#overview)
2. [What is a Corpus?](#what-is-a-corpus)
3. [Dataset Structure](#dataset-structure)
4. [Corpus Construction](#corpus-construction)
5. [Data Pipeline](#data-pipeline)
6. [Building Your Own Dataset](#building-your-own-dataset)
7. [Extending Existing Datasets](#extending-existing-datasets)
8. [Troubleshooting](#troubleshooting)

---

## Overview

BiG-RAG operates on two main data components:
1. **Datasets**: Question-answer pairs for training and evaluation
2. **Corpus**: Text documents that form the knowledge base

The **corpus** is processed into a **bipartite knowledge graph**, which the model queries during training and inference.

---

## What is a Corpus?

### Definition

A **corpus** (plural: *corpora*) is a collection of text documents that serves as the raw knowledge source for BiG-RAG. Think of it as a library of information from which the system can retrieve relevant facts to answer questions.

### Key Characteristics

- **Format**: Plain text or structured text (usually JSONL)
- **Content**: Domain-specific or general knowledge documents
- **Size**: Typically thousands to millions of documents
- **Purpose**: Provides factual grounding for question answering

### Example Corpus Entry

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

---

## Dataset Structure

BiG-RAG uses a standardized directory structure for each dataset:

```
datasets/{dataset_name}/
├── raw/
│   ├── corpus.jsonl              # ← Raw text corpus (knowledge base)
│   ├── qa_train.json             # ← Training question-answer pairs
│   ├── qa_dev.json               # ← Development/validation QA pairs
│   └── qa_test.json              # ← Test QA pairs
└── processed/
    ├── train.parquet             # ← Processed training data (from script_process.py)
    ├── dev.parquet               # ← Processed development data
    └── test.parquet              # ← Processed test data
```

### File Descriptions

#### 1. `corpus.jsonl` (Raw Corpus)

The knowledge base that BiG-RAG will build a bipartite graph from.

**Format**: JSONL (JSON Lines - one JSON object per line)

**Required Fields**:
```json
{
  "contents": "The actual text content of the document",
  "id": "unique_document_identifier"
}
```

**Optional Fields**:
```json
{
  "title": "Document title",
  "url": "Source URL",
  "metadata": { ... }
}
```

**Example**:
```jsonl
{"id": "doc_001", "contents": "Paris is the capital of France..."}
{"id": "doc_002", "contents": "The Eiffel Tower was built in 1889..."}
{"id": "doc_003", "contents": "Napoleon Bonaparte was born in Corsica..."}
```

#### 2. `qa_train.json`, `qa_dev.json`, `qa_test.json` (QA Pairs)

Question-answer pairs for training and evaluation.

**Format**: JSON array

**Required Fields**:
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

**Notes**:
- `golden_answers` is an array to support multiple correct answers
- Questions should be answerable from the corpus
- Multi-hop questions require reasoning across multiple documents

---

## Corpus Construction

### How Corpus is Built for Each Dataset

The corpus construction process varies by dataset type, but follows these general principles:

#### 1. **Multi-Hop QA Datasets** (2WikiMultiHopQA, HotpotQA, Musique)

**Source**: Wikipedia articles or curated document collections

**Construction Process**:
1. **Identify Supporting Documents**: For each question, find all documents that contain relevant information
2. **Include Distractor Documents**: Add documents that seem relevant but aren't (increases difficulty)
3. **Deduplicate**: Remove duplicate documents
4. **Clean**: Remove formatting artifacts, normalize text

**Example** (2WikiMultiHopQA):
```json
{
  "question": "Who is the spouse of the director of film Nosferatu (1922)?",
  "golden_answers": ["Enno Patalas"],
  "supporting_docs": ["doc_001", "doc_045"]
}
```

Corpus would include:
- `doc_001`: Information about F.W. Murnau (director)
- `doc_045`: Information about Enno Patalas (spouse)
- Additional documents about film, actors, etc. (distractors)

#### 2. **Single-Hop QA Datasets** (NQ, TriviaQA, PopQA)

**Source**: Web snippets, Wikipedia paragraphs

**Construction Process**:
1. **Retrieve Relevant Passages**: Use BM25 or dense retrieval to find passages
2. **Include Top-K**: Keep top 50-100 passages per question
3. **Pool Across Questions**: Combine passages from all questions
4. **Deduplicate**: Remove exact duplicates

**Example** (Natural Questions):
```json
{
  "question": "when was the first star wars movie released",
  "golden_answers": ["May 25, 1977"]
}
```

Corpus includes passages about:
- Star Wars original trilogy
- George Lucas
- 1977 film releases
- Science fiction movies

#### 3. **Custom Domain Corpus**

For domain-specific applications (medical, legal, financial):

**Construction Process**:
1. **Collect Domain Documents**: PDFs, web pages, databases
2. **Extract Text**: Use OCR, HTML parsing, or APIs
3. **Chunk Long Documents**: Split into manageable pieces (500-2000 tokens)
4. **Annotate Metadata**: Add source, date, author, section
5. **Quality Filter**: Remove low-quality or duplicate content

---

## Data Pipeline

### Complete Flow: Raw Data → Trained Model

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Prepare Raw Data                                       │
├─────────────────────────────────────────────────────────────────┤
│  • datasets/{dataset}/raw/corpus.jsonl                          │
│  • datasets/{dataset}/raw/qa_train.json                         │
│  • datasets/{dataset}/raw/qa_dev.json                           │
│  • datasets/{dataset}/raw/qa_test.json                          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Preprocess QA Pairs (script_process.py)                │
├─────────────────────────────────────────────────────────────────┤
│  Command: python script_process.py --data_source DatasetName    │
│  Output: datasets/{dataset}/processed/*.parquet                 │
│  • Converts JSON to Parquet format                              │
│  • Adds instruction templates                                   │
│  • Formats for RL training                                      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Build Bipartite Knowledge Graph (script_build.py)      │
├─────────────────────────────────────────────────────────────────┤
│  Command: python script_build.py --data_source DatasetName      │
│  Output: expr/{dataset}/                                        │
│  • Chunks corpus documents                                      │
│  • Extracts entities & relations (via GPT-4o-mini)              │
│  • Builds bipartite graph structure                             │
│  • Creates FAISS indices for fast retrieval                     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Start Retrieval Server (script_api.py)                 │
├─────────────────────────────────────────────────────────────────┤
│  Command: python script_api.py --data_source DatasetName        │
│  Server: FastAPI on port 8001                                   │
│  • Loads bipartite graph and FAISS indices                      │
│  • Provides /search endpoint for training                       │
│  • Remains running during training                              │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: RL Training (run_grpo.sh / run_ppo.sh)                 │
├─────────────────────────────────────────────────────────────────┤
│  • Loads processed Parquet data                                 │
│  • Trains LLM to query graph during generation                  │
│  • Evaluates with EM/F1 metrics                                 │
│  • Saves checkpoints                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Pipeline Steps

#### Step 2: Preprocessing Details (`script_process.py`)

**Input**: Raw JSON files
**Output**: Parquet files with standardized schema

**What it does**:
1. Loads `qa_train.json`, `qa_dev.json`, `qa_test.json`
2. Converts each QA pair to training format:
   ```python
   {
       "data_source": "2WikiMultiHopQA",
       "prompt": [{"role": "user", "content": "Question: ..."}],
       "ability": "multihop_qa",
       "reward_model": {
           "style": "rule",
           "ground_truth": ["Paris"]
       },
       "extra_info": {
           "split": "train",
           "index": "0",
           "answer": ["Paris"],
           "question": "What is the capital of France?"
       }
   }
   ```
3. Adds instruction template with tool-calling format
4. Saves as Parquet (efficient columnar format)

**Running**:
```bash
python script_process.py --data_source 2WikiMultiHopQA
```

#### Step 3: Graph Construction Details (`script_build.py`)

**Input**: `corpus.jsonl`
**Output**: Knowledge graph + FAISS indices

**What it does**:

1. **Chunking** (via `BiGRAG.insert()`):
   ```python
   # Splits long documents into chunks
   chunk_size = 1200 tokens
   overlap = 100 tokens
   ```

2. **Entity Extraction** (via GPT-4o-mini):
   ```
   Document Chunk → LLM → Entities + Relations

   Example:
   Input: "Paris is the capital of France, located on the Seine River."
   Output:
     Entities: ["Paris", "France", "Seine River"]
     Relations: [
       ("Paris", "capital_of", "France"),
       ("Paris", "located_on", "Seine River")
     ]
   ```

3. **Bipartite Graph Construction**:
   ```
   Documents ←→ Entities ←→ Relations ←→ Entities ←→ Documents

   Bipartite structure:
   - One side: Document chunks
   - Other side: Entities + Relations (bipartite edges)
   - Edges connect documents to entities they mention
   ```

4. **Embedding & Indexing**:
   ```python
   # Embed all entities using FlagEmbedding (bge-large-en-v1.5)
   entity_embeddings = model.encode(entities)

   # Create FAISS index for fast similarity search
   index_entity = faiss.IndexFlatIP(dimension=1536)
   index_entity.add(entity_embeddings)
   faiss.write_index(index_entity, "index_entity.bin")
   ```

**Output Files**:
```
expr/{dataset}/
├── kv_store_entities.json          # Entity metadata
├── kv_store_bipartite_edges.json   # Relation/edge metadata
├── kv_store_text_chunks.json       # Original text chunks
├── index_entity.bin                # FAISS index for entities
├── index_bipartite_edge.bin        # FAISS index for edges
├── index.bin                       # FAISS index for chunks
├── corpus.npy                      # Chunk embeddings
├── corpus_entity.npy               # Entity embeddings
└── corpus_bipartite_edge.npy       # Edge embeddings
```

**Running**:
```bash
# Requires OpenAI API key in openai_api_key.txt
python script_build.py --data_source 2WikiMultiHopQA
```

**Time Estimate**: 2-4 hours for 10K documents (depends on corpus size and API rate limits)

#### Step 4: Retrieval Server (`script_api.py`)

**What it does**:
1. Loads pre-built bipartite graph
2. Loads FAISS indices into memory
3. Starts FastAPI server on port 8001
4. Provides `/search` endpoint:
   ```python
   POST /search
   Body: {"queries": ["What is the capital of France?"]}
   Response: [{"results": "Paris is the capital..."}]
   ```

**Running**:
```bash
python script_api.py --data_source 2WikiMultiHopQA
# Keep running in background during training
```

---

## Building Your Own Dataset

### Step-by-Step Guide

#### 1. Choose a Name

```bash
export DATASET_NAME="MyCustomDataset"
```

#### 2. Create Directory Structure

```bash
mkdir -p datasets/$DATASET_NAME/raw
mkdir -p datasets/$DATASET_NAME/processed
```

#### 3. Prepare Corpus

Create `datasets/MyCustomDataset/raw/corpus.jsonl`:

```python
import json

corpus = [
    {
        "id": "doc_001",
        "contents": "Your document text here...",
        "title": "Document Title",
        "metadata": {"source": "your_source"}
    },
    # Add more documents...
]

with open("datasets/MyCustomDataset/raw/corpus.jsonl", "w") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")
```

**Best Practices**:
- Each document should be self-contained (complete sentences)
- Length: 100-2000 tokens per document (longer docs get chunked)
- Include diverse information relevant to your questions
- Avoid duplicate or near-duplicate documents

#### 4. Prepare QA Pairs

Create `datasets/MyCustomDataset/raw/qa_train.json`:

```python
import json

qa_pairs = [
    {
        "question": "Your question here?",
        "golden_answers": ["answer1", "answer2"]  # List of acceptable answers
    },
    # Add more QA pairs...
]

# Save train, dev, test splits (80/10/10 typical)
with open("datasets/MyCustomDataset/raw/qa_train.json", "w") as f:
    json.dump(qa_pairs[:800], f, indent=2)

with open("datasets/MyCustomDataset/raw/qa_dev.json", "w") as f:
    json.dump(qa_pairs[800:900], f, indent=2)

with open("datasets/MyCustomDataset/raw/qa_test.json", "w") as f:
    json.dump(qa_pairs[900:], f, indent=2)
```

**Best Practices**:
- Questions should be answerable from corpus
- Include multiple acceptable answer forms
- Balance question difficulty (mix easy/hard)
- Minimum ~100 questions per split for meaningful evaluation

#### 5. Preprocess Dataset

```bash
python script_process.py --data_source MyCustomDataset
```

**Verification**:
```python
import pandas as pd

# Check processed data
df = pd.read_parquet("datasets/MyCustomDataset/processed/train.parquet")
print(df.head())
print(f"Total samples: {len(df)}")
```

#### 6. Build Knowledge Graph

```bash
# Ensure OpenAI API key is set
echo "your-api-key-here" > openai_api_key.txt

# Build graph (this will take time)
python script_build.py --data_source MyCustomDataset
```

**Monitor Progress**:
- Check logs for entity extraction progress
- Typical: ~10-50 documents per minute (depends on API speed)

#### 7. Test Retrieval

```bash
# Start server
python script_api.py --data_source MyCustomDataset &

# Test query
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": ["test query"]}'

# Stop server when done testing
pkill -f script_api.py
```

#### 8. Train on Your Dataset

```bash
# Update training script with your dataset name
bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct \
                 -m Qwen2.5-3B \
                 -d MyCustomDataset
```

---

## Extending Existing Datasets

### Adding New Documents to Corpus

#### Scenario: New information becomes available

**Steps**:

1. **Append to corpus.jsonl**:
   ```python
   import json

   new_docs = [
       {"id": "doc_new_001", "contents": "New information..."},
       {"id": "doc_new_002", "contents": "More new information..."}
   ]

   with open("datasets/MyDataset/raw/corpus.jsonl", "a") as f:
       for doc in new_docs:
           f.write(json.dumps(doc) + "\n")
   ```

2. **Rebuild knowledge graph**:
   ```bash
   # This will reprocess entire corpus (including new docs)
   python script_build.py --data_source MyDataset
   ```

3. **Restart retrieval server**:
   ```bash
   pkill -f "script_api.py.*MyDataset"
   python script_api.py --data_source MyDataset &
   ```

### Adding New QA Pairs

#### Scenario: Expanding training set

**Steps**:

1. **Append to qa_train.json**:
   ```python
   import json

   # Load existing
   with open("datasets/MyDataset/raw/qa_train.json") as f:
       qa_data = json.load(f)

   # Add new
   qa_data.extend([
       {"question": "New question?", "golden_answers": ["answer"]},
       # More questions...
   ])

   # Save
   with open("datasets/MyDataset/raw/qa_train.json", "w") as f:
       json.dump(qa_data, f, indent=2)
   ```

2. **Reprocess QA data**:
   ```bash
   python script_process.py --data_source MyDataset
   ```

3. **Retrain** (or continue training from checkpoint):
   ```bash
   bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct \
                    -m Qwen2.5-3B \
                    -d MyDataset
   ```

---

## Troubleshooting

### Common Issues

#### 1. "KeyError: 'contents'" when building graph

**Cause**: Corpus JSONL missing required `contents` field

**Fix**:
```python
# Check corpus format
import json
with open("datasets/MyDataset/raw/corpus.jsonl") as f:
    first_doc = json.loads(f.readline())
    print(first_doc.keys())  # Should include 'contents'
```

#### 2. "No module named 'datasets'" when preprocessing

**Cause**: Missing Hugging Face datasets library

**Fix**:
```bash
pip install datasets
```

#### 3. Empty FAISS indices

**Cause**: Entity extraction failed or returned no entities

**Fix**:
- Check OpenAI API key is valid
- Verify corpus has substantive content (not just titles)
- Check logs for LLM extraction errors

#### 4. Retrieval server returns empty results

**Cause**: FAISS indices not loaded or query embeddings failing

**Fix**:
```python
# Test index files exist
import os
dataset = "MyDataset"
files = [
    f"expr/{dataset}/index_entity.bin",
    f"expr/{dataset}/index_bipartite_edge.bin",
    f"expr/{dataset}/kv_store_entities.json"
]
for f in files:
    print(f"{f}: {os.path.exists(f)}")
```

#### 5. Training fails with "connection refused" to port 8001

**Cause**: Retrieval server not running

**Fix**:
```bash
# Check if server is running
ps aux | grep script_api

# If not, start it
python script_api.py --data_source MyDataset &

# Wait for "Uvicorn running on http://0.0.0.0:8001" message
```

---

## Advanced Topics

### Corpus Quality Guidelines

**Good Corpus Properties**:
- ✅ Factual, accurate information
- ✅ Diverse topics and writing styles
- ✅ Natural language (avoid keyword lists)
- ✅ Proper sentence structure
- ✅ Unique documents (minimal duplication)

**Poor Corpus Properties**:
- ❌ Tables without context (just numbers)
- ❌ Code snippets (unless task is code QA)
- ❌ Extremely short documents (<20 tokens)
- ❌ Highly repetitive content
- ❌ Broken formatting (e.g., HTML tags)

### Multi-Modal Corpus (Future)

BiG-RAG currently supports text-only corpora. For multi-modal data:

**Images**: Extract captions and OCR text → add to corpus as text documents
**Tables**: Convert to natural language descriptions → add to corpus
**PDFs**: Use PDF parsing libraries to extract clean text

### Corpus Update Strategies

**Incremental Updates** (recommended for large corpora):
1. Track document IDs already processed
2. Only build graph for new documents
3. Merge new graph with existing graph

**Full Rebuild** (simpler but slower):
1. Rebuild entire graph from scratch
2. Recommended when <10K documents or major changes

---

## Summary Checklist

Building a new dataset:
- [ ] Create directory structure: `datasets/{name}/raw/` and `.../processed/`
- [ ] Prepare `corpus.jsonl` with `id` and `contents` fields
- [ ] Prepare `qa_train.json`, `qa_dev.json`, `qa_test.json` with questions
- [ ] Run `script_process.py --data_source {name}`
- [ ] Set OpenAI API key in `openai_api_key.txt`
- [ ] Run `script_build.py --data_source {name}`
- [ ] Verify output files in `expr/{name}/`
- [ ] Start `script_api.py --data_source {name}`
- [ ] Test retrieval with curl or Python requests
- [ ] Train model with `run_grpo.sh` or `run_ppo.sh`

---

## Related Documentation

- [BiG-RAG Architecture Overview](../README.md)
- [Setup Guide](./SETUP_AND_TESTING_GUIDE.md)
- [Training Guide](../CLAUDE.md)
- [API Reference](../docs/Helper_code/README.md)

---

**Questions or issues?** Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) or open an issue on GitHub.
