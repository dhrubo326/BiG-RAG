# Retrieval Evaluation Tools

This directory contains tools for evaluating **BiG-RAG retrieval quality** (knowledge graph construction and querying).

**Note**: This is separate from `evaluation/` which evaluates **RL-trained model outputs** (EM/F1 metrics).

---

## Purpose

Test BiG-RAG's retrieval performance across:
- Different datasets (Single-Topic, 2WikiMultiHopQA, HotpotQA, etc.)
- Different retrieval modes (hybrid, local, global, naive)
- Multiple quality dimensions (relevance, comprehensiveness, diversity, logicality, coherence)

---

## Scripts

### 1. `convert_csv_to_corpus.py`

Convert CSV documents to BiG-RAG corpus format (JSONL).

```bash
python retrieval_eval/convert_csv_to_corpus.py --csv datasets/Single-Topic/raw/documents.csv
python retrieval_eval/convert_csv_to_corpus.py --csv path/to/docs.csv --output path/to/corpus.jsonl
```

**Input**: CSV with `index`, `text`, `source_url` columns
**Output**: JSONL with `{"id": "...", "contents": "...", "title": "...", "metadata": {...}}`

### 2. `script_evaluate_single_topic.py`

Full evaluation with detailed per-question-type analysis.

```bash
cd retrieval_eval
python script_evaluate_single_topic.py --data_source Single-Topic
python script_evaluate_single_topic.py --data_source Single-Topic --top_k 15 --rebuild
```

**Features**:
- Separate evaluation for each question type (single-passage, multi-passage, no-answer)
- Detailed statistics (mean, std, median) per metric
- Saves to `expr/{dataset}/evaluation_results.json`

### 3. `script_quick_eval.py`

Quick comparative evaluation across retrieval modes.

```bash
cd retrieval_eval
python script_quick_eval.py --data_source Single-Topic --sample 20
python script_quick_eval.py --data_source Single-Topic --full
```

**Features**:
- Compares all 4 retrieval modes (hybrid, local, global, naive)
- Samples questions for fast testing (or use all with `--full`)
- Shows comparison table with best mode per metric
- Saves to `expr/{dataset}/comparative_results.json`

**Output**:
```
COMPARATIVE RESULTS
================================================================================
Metric               Hybrid      Local     Global      Naive
--------------------------------------------------------------------------------
Relevance            0.768*      0.732     0.654      0.621
Comprehensiveness    0.845*      0.798     0.723      0.712
...
```

### 4. `script_visualize_results.py`

Generate charts and visualizations from evaluation results.

```bash
cd retrieval_eval
python script_visualize_results.py --comparative expr/Single-Topic/comparative_results.json --output_dir figures
python script_visualize_results.py --results expr/Single-Topic/evaluation_results.json --output_dir figures
```

**Generates**:
- Bar charts comparing metrics
- Radar charts showing performance profiles
- Heatmaps for detailed analysis
- Text summary reports

**Requirements**: `pip install matplotlib seaborn`

---

## Evaluation Metrics

### 5 Core Metrics

1. **Relevance (F1)**: Precision/recall of retrieved documents
   - Target: >0.80
   - Measures: Are retrieved docs correct?

2. **Comprehensiveness (Recall)**: Coverage of all necessary documents
   - Target: >0.90
   - Measures: Did we get ALL needed docs?

3. **Diversity**: Multi-source retrieval for complex questions
   - Target: >0.85
   - Measures: Retrieved from multiple sources?

4. **Logicality (Precision)**: Signal-to-noise ratio
   - Target: >0.75
   - Measures: Low noise in results?

5. **Coherence (Average Precision)**: Ranking quality
   - Target: >0.85
   - Measures: Relevant docs ranked high?

---

## Workflow: Evaluate New Dataset

### Step 1: Prepare Dataset

Create directory structure:
```
datasets/YourDataset/raw/
├── documents.csv                          # Corpus (required)
├── single_passage_answer_questions.csv    # Single-doc questions
├── multi_passage_answer_questions.csv     # Multi-doc questions
└── no_answer_questions.csv                # Unanswerable questions
```

**documents.csv format**:
```csv
index,source_url,text
0,https://example.com,"Document text..."
```

**Question file format**:
```csv
document_index,question,answer
0,"Question text?","Answer text."
```

### Step 2: Convert to Corpus

```bash
cd retrieval_eval
python convert_csv_to_corpus.py --csv ../datasets/YourDataset/raw/documents.csv
```

Output: `datasets/YourDataset/raw/corpus.jsonl`

### Step 3: Build Knowledge Graph

```bash
cd ..
python script_build.py --data_source YourDataset --batch_size 5
```

Output: `expr/YourDataset/` (graph files)

**Time**: 10-30 minutes (depends on corpus size, uses OpenAI API)

### Step 4: Quick Evaluation

```bash
cd retrieval_eval
python script_quick_eval.py --data_source YourDataset --sample 20
```

Output: `expr/YourDataset/comparative_results.json`

### Step 5: Full Evaluation (Optional)

```bash
python script_evaluate_single_topic.py --data_source YourDataset
```

Output: `expr/YourDataset/evaluation_results.json`

### Step 6: Visualize (Optional)

```bash
python script_visualize_results.py --comparative expr/YourDataset/comparative_results.json --output_dir figures/YourDataset
```

Output: `figures/YourDataset/*.png`

---

## Dataset Requirements

### Required Files

- ✅ `documents.csv` - Corpus for building knowledge graph
- ✅ At least ONE question file (single/multi/no-answer)
- ✅ `openai_api_key.txt` in root directory (for embeddings)

### CSV Formats

**documents.csv** (required columns):
- `index`: Unique document ID (integer)
- `text`: Document content (string)
- `source_url`: (optional) Source URL

**Question files** (required columns):
- `document_index`: Ground truth document ID
- `question`: Question text
- `answer`: Answer text (not required for no_answer_questions.csv)

---

## Retrieval Modes

- **Hybrid** (default): Entity + Relation retrieval (full bipartite graph)
- **Local**: Entity-based retrieval only
- **Global**: Relation-based retrieval only
- **Naive**: Direct text similarity (no graph, baseline)

**Recommendation**: Hybrid mode usually performs best.

---

## Troubleshooting

### "FileNotFoundError: corpus.jsonl"
```bash
cd retrieval_eval
python convert_csv_to_corpus.py --csv ../datasets/YourDataset/raw/documents.csv
```

### "OpenAI API key not found"
```bash
echo sk-your-api-key > openai_api_key.txt  # In root directory
```

### "vdb_entities.json not found"
```bash
python script_build.py --data_source YourDataset  # Build graph first
```

### "No module named 'pandas'"
```bash
pip install pandas matplotlib seaborn
```

### Import error when running scripts
```bash
# Always run from retrieval_eval directory
cd retrieval_eval
python script_quick_eval.py ...

# Or use python -m
python -m retrieval_eval.script_quick_eval ...
```

---

## Comparison: retrieval_eval vs evaluation

| Directory | Purpose | Evaluates | Metrics | When to Use |
|-----------|---------|-----------|---------|-------------|
| **retrieval_eval/** | Test retrieval quality | Knowledge graph querying | Relevance, Comprehensiveness, Diversity, Logicality, Coherence | Before RL training, dataset validation |
| **evaluation/** | Test trained models | RL-trained model outputs | EM, F1, SimCSE | After RL training |

**Example**:
```bash
# 1. Test retrieval quality (retrieval_eval)
cd retrieval_eval
python script_quick_eval.py --data_source Single-Topic --sample 20

# 2. Train model with RL
cd ..
bash run_grpo.sh -p Qwen/Qwen2.5-3B-Instruct -m qwen3b -d Single-Topic

# 3. Evaluate trained model outputs (evaluation)
cd evaluation
python get_remote_score.py --dir ../expr_results/Qwen2.5-3B-Instruct_Single-Topic_grpo
```

---

## Complete Documentation

See [SINGLE_TOPIC_EVALUATION.md](../SINGLE_TOPIC_EVALUATION.md) for:
- Complete workflow guide
- Implementation details
- OpenAI API integration
- Troubleshooting guide
- Change log

---

## Future Enhancements

- [ ] Support more question types (reasoning, factoid, yes/no)
- [ ] Add cross-dataset comparison
- [ ] Integrate with MLflow for experiment tracking
- [ ] Add statistical significance tests
- [ ] Support batch evaluation across multiple datasets

---

**Questions?** Open an issue or check [SINGLE_TOPIC_EVALUATION.md](../SINGLE_TOPIC_EVALUATION.md)
