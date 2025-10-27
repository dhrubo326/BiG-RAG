# Single-Topic Dataset Evaluation Guide

## Overview

This guide documents the complete workflow for evaluating BiG-RAG on single-topic datasets using OpenAI models (no local models required).

**Purpose**: Test the BiG-RAG framework end-to-end with a focused dataset to validate:
- Knowledge graph construction
- Entity and relation extraction
- Retrieval quality across different modes
- Evaluation metrics implementation

---

## Quick Start

### 1. Setup (One-time)

```bash
cd d:\BiG-RAG
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install pandas matplotlib seaborn

# Set OpenAI API key
echo "sk-your-api-key" > openai_api_key.txt
```

**Models Used** (all via OpenAI API):
- **LLM**: GPT-4o-mini (entity/relation extraction)
- **Embedding**: text-embedding-3-large (3072 dimensions)

### 2. Build Knowledge Graph (10-30 minutes)

```bash
# Convert CSV corpus to JSONL format
cd retrieval_eval
python convert_csv_to_corpus.py --csv ../datasets/Single-Topic/raw/documents.csv

# Build bipartite graph with OpenAI models
cd ..
python script_build.py --data_source Single-Topic
```

**Output**: `expr/Single-Topic/` (graph files with entities, relations, embeddings)

### 3. Evaluate (5 minutes)

```bash
# Quick comparative evaluation (sample 20 questions per type)
cd retrieval_eval
python script_quick_eval.py --data_source Single-Topic --sample 20

# Full evaluation (all questions)
python script_evaluate_single_topic.py --data_source Single-Topic

# Visualize results
python script_visualize_results.py --comparative ../expr/Single-Topic/comparative_results.json --output_dir figures
```

---

## Dataset Structure

### Input Files

```
datasets/Single-Topic/raw/
├── documents.csv                          → Corpus for graph building (20 docs)
├── single_passage_answer_questions.csv    → Single-doc questions (63)
├── multi_passage_answer_questions.csv     → Multi-doc questions (60)
└── no_answer_questions.csv                → Unanswerable questions (41)
```

### File Formats

**documents.csv** (for graph construction):
```csv
index,source_url,text
0,https://example.com,Document text here...
```

**question files** (for evaluation):
```csv
document_index,question,answer
0,What do keybullet kin drop?,Keybullet kin drop a key...
```

---

## Evaluation Metrics

### 5 Core Metrics

1. **Relevance (F1)**: Are retrieved documents correct?
   - Precision: % of retrieved docs that are relevant
   - Recall: % of relevant docs that were retrieved
   - F1: Harmonic mean
   - **Target**: >0.80

2. **Comprehensiveness**: Did we retrieve ALL necessary documents?
   - Measures recall (coverage of ground truth)
   - **Target**: >0.90

3. **Diversity**: For multi-passage questions, did we retrieve from multiple sources?
   - Measures retrieval from diverse sources
   - **Target**: >0.85

4. **Logicality**: Ratio of relevant to total retrieved documents (low noise)
   - High score = most retrieved docs are relevant
   - **Target**: >0.75

5. **Coherence**: Are relevant chunks ranked high and grouped together?
   - Uses average precision-like metric
   - **Target**: >0.85

### Retrieval Modes

- **Hybrid** (default): Entity-based + Relation-based retrieval (full bipartite graph)
- **Local**: Entity-based retrieval only
- **Global**: Relation-based retrieval only
- **Naive**: Direct text similarity (no graph structure)

---

## Scripts Reference

All evaluation scripts are in `retrieval_eval/` directory.

### Core Scripts

#### 1. `retrieval_eval/convert_csv_to_corpus.py`

Converts CSV documents to BiG-RAG corpus format (JSONL).

```bash
cd retrieval_eval
python convert_csv_to_corpus.py --csv ../datasets/Single-Topic/raw/documents.csv
python convert_csv_to_corpus.py --csv path/to/docs.csv --output path/to/corpus.jsonl --overwrite
```

**What it does**:
- Reads documents.csv (index, text, source_url columns)
- Converts to corpus.jsonl format
- Each line: `{"id": "0", "contents": "...", "title": "...", "metadata": {...}}`

#### 2. `script_build.py`

Builds bipartite knowledge graph using OpenAI models.

```bash
python script_build.py --data_source Single-Topic
python script_build.py --data_source Single-Topic --batch_size 5 --chunk_size 1200
```

**What it does**:
1. Loads corpus from `datasets/{dataset}/raw/corpus.jsonl`
2. Chunks documents (default: 1200 tokens, 100 overlap)
3. Extracts entities and relations via GPT-4o-mini
4. Constructs bipartite graph: Documents ↔ Entities ↔ Relations
5. Generates embeddings via text-embedding-3-large
6. Saves to `expr/{dataset}/`

**Output files**:
```
expr/Single-Topic/
├── kv_store_text_chunks.json          # Text chunk metadata
├── vdb_entities.json                  # Entity embeddings (NanoVectorDB)
├── vdb_bipartite_edges.json           # Relation embeddings (NanoVectorDB)
└── graph_chunk_entity_relation.graphml # Graph structure (NetworkX)
```

#### 3. `retrieval_eval/script_quick_eval.py`

Quick comparative evaluation across retrieval modes.

```bash
cd retrieval_eval
python script_quick_eval.py --data_source Single-Topic --sample 20    # Sample 20 per type
python script_quick_eval.py --data_source Single-Topic --full         # All questions
python script_quick_eval.py --data_source Single-Topic --sample 10 --top_k 15  # Custom parameters
```

**What it does**:
- Evaluates all 4 retrieval modes (hybrid, local, global, naive)
- Tests on 3 question types (single-passage, multi-passage, no-answer)
- Computes 5 metrics for each combination
- Generates comparison table showing best mode per metric
- Saves results to `expr/{dataset}/comparative_results.json`

**Output**:
```
COMPARATIVE RESULTS
================================================================================
Metric               Hybrid      Local     Global      Naive
--------------------------------------------------------------------------------
Relevance            0.768*      0.732     0.654      0.621
Comprehensiveness    0.845*      0.798     0.723      0.712
Diversity            0.892*      0.867     0.801      0.789
Logicality           0.701*      0.665     0.598      0.571
Coherence            0.823*      0.789     0.734      0.698

* = Best performance
```

#### 4. `retrieval_eval/script_evaluate_single_topic.py`

Full evaluation with detailed per-question-type analysis.

```bash
cd retrieval_eval
python script_evaluate_single_topic.py --data_source Single-Topic
python script_evaluate_single_topic.py --data_source Single-Topic --top_k 15 --rebuild
```

**What it does**:
- Separate evaluation for each question type
- Detailed statistics (mean, std, median) per metric
- Saves to `expr/{dataset}/evaluation_results.json`

#### 5. `retrieval_eval/script_visualize_results.py`

Generates charts and visualizations from evaluation results.

```bash
cd retrieval_eval
# Visualize comparative results
python script_visualize_results.py --comparative ../expr/Single-Topic/comparative_results.json --output_dir figures

# Visualize full evaluation results
python script_visualize_results.py --results ../expr/Single-Topic/evaluation_results.json --output_dir figures
```

**Generates**:
- Bar charts comparing metrics
- Radar charts showing performance profiles
- Heatmaps for detailed analysis
- Text summary reports

**Requirements**: `pip install matplotlib seaborn`

---

## Implementation Details

### Bipartite Graph Structure

BiG-RAG uses a **true bipartite graph** (not a hypergraph):

```
Document Chunks ←→ Entities/Relations
```

**Key properties**:
- Two node types: Documents (chunks) and Semantic nodes (entities + relations)
- Edges connect documents to the entities/relations they contain
- No direct edges between documents or between entities
- Queries traverse: query → entities/relations → documents

### Query Flow

```
1. User Query: "What do keybullet kin drop?"
       ↓
2. Embed query with text-embedding-3-large (3072 dim)
       ↓
3. Search in NanoVectorDB:
   - Entity embeddings (local mode)
   - Relation embeddings (global mode)
   - Both (hybrid mode - default)
       ↓
4. Retrieve matched entities/relations
       ↓
5. Traverse graph to find connected documents
       ↓
6. Rank and return top-k document chunks
```

### OpenAI API Integration

**LLM (GPT-4o-mini)** - Used for:
- Entity extraction from text chunks
- Relation extraction between entities
- Summarization of entities

**Embedding (text-embedding-3-large)** - Used for:
- Document chunk embeddings (3072 dim)
- Entity embeddings (3072 dim)
- Relation embeddings (3072 dim)
- Query embeddings (3072 dim)

All embeddings stored in NanoVectorDB (in-memory vector database).

---

## Troubleshooting

### Common Issues

**1. "OpenAI API key not found"**
```bash
echo sk-your-api-key > openai_api_key.txt
```

**2. "FileNotFoundError: corpus.jsonl"**
```bash
python convert_csv_to_corpus.py --csv datasets/Single-Topic/raw/documents.csv
```

**3. "FileNotFoundError: documents.csv"**
```bash
# Check files exist
dir datasets\Single-Topic\raw\
```

**4. "vdb_entities.json not found"**
```bash
# Rebuild graph
python script_build.py --data_source Single-Topic
```

**5. "No module named 'pandas'"**
```bash
venv\Scripts\activate
pip install pandas matplotlib seaborn
```

---

## Change Log

### Issues Fixed

1. **FlagEmbedding Dependency** → Switched to OpenAI text-embedding-3-large
2. **Unicode Errors (Windows)** → Added UTF-8 encoding support
3. **Wrong File Check** → Check for `vdb_entities.json` (new format)
4. **Document Format Error** → Pass strings to `ainsert()`, not dicts
5. **QueryParam Error** → Use `QueryParam(top_k=k, mode=m)` object
6. **API Key Loading** → Load from `openai_api_key.txt` in evaluation scripts

### Key Decisions

**Why OpenAI models?**
- No local model installation required
- Consistent quality (GPT-4o-mini is reliable for entity extraction)
- High-quality embeddings (text-embedding-3-large, 3072 dim)
- Faster setup for users without GPU

**Why NanoVectorDB?**
- In-memory vector database (no external dependencies)
- Simple JSON storage format
- Fast enough for small-to-medium datasets (<100K chunks)
- Easy to inspect and debug

**Why Single-Topic dataset?**
- Small size (20 docs) for quick testing
- Focused domain (single topic) for clear evaluation
- Diverse question types (single, multi, no-answer)
- Good test case for full pipeline validation

---

## Next Steps

### After successful evaluation

1. **Analyze results**: Which metrics are weak? Which retrieval mode performs best?

2. **Tune parameters**: Try different values:
   ```bash
   cd retrieval_eval
   python script_quick_eval.py --data_source Single-Topic --sample 20 --top_k 15
   cd ..
   python script_build.py --data_source Single-Topic --chunk_size 800
   ```

3. **Try larger datasets**: Apply to multi-hop QA datasets:
   ```bash
   python script_process.py --data_source 2WikiMultiHopQA
   python script_build.py --data_source 2WikiMultiHopQA
   ```

4. **Integrate with RL training**: Use retrieval API in tool-augmented generation
   ```bash
   python script_api.py --data_source Single-Topic  # Start retrieval server
   bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m qwen3b -d Single-Topic
   ```

---

## Additional Resources

- **Main README**: [README.md](README.md) - Project overview
- **Development Notes**: [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) - Technical architecture
- **Claude Instructions**: [CLAUDE.md](CLAUDE.md) - Complete workflow guide
- **Dataset Guide**: [docs/DATASET_AND_CORPUS_GUIDE.md](docs/DATASET_AND_CORPUS_GUIDE.md)

---

**Questions?** Open an issue on GitHub or check the main documentation.
