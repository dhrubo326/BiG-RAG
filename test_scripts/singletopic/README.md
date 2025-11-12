# SingleTopic Dataset Evaluation Scripts

This directory contains scripts to evaluate BiG-RAG's performance on the SingleTopic dataset.

## Dataset Overview

**SingleTopic** is a question-answering dataset about video game wikis (Enter the Gungeon).

- **Documents**: 9,374 wiki pages
- **Questions**: 160 total
  - 62 single-passage questions (answer from one document)
  - 58 multi-passage questions (answer from multiple documents)
  - 39 no-answer questions (unanswerable, expects refusal)

**Location**: `datasets/SingleTopic/`

## Files in This Directory

| Script | Purpose | Time |
|--------|---------|------|
| `3_build_knowledge_graph.sh` | Build BiG-RAG knowledge graph from corpus | 2-4 hours |
| `4_generate_answers.py` | Query BiG-RAG for all questions, save results | ~5-10 min |
| `5_evaluate_results.py` | Calculate metrics (EM, F1, etc.) | <1 min |
| `run_full_evaluation.sh` | **Master script** - runs all steps | 2-4 hours |
| `README.md` | This file | - |

---

## Quick Start

### Option 1: Full Evaluation (Automated)

```bash
# From project root
bash test_scripts/singletopic/run_full_evaluation.sh
```

This will:
1. Build knowledge graph (if not exists)
2. Check if server is running
3. Generate answers for all 160 questions
4. Calculate evaluation metrics
5. Save results to `datasets/SingleTopic/results/`

### Option 2: Step-by-Step (Manual)

```bash
# Step 1: Build knowledge graph (one-time, takes 2-4 hours)
bash test_scripts/singletopic/3_build_knowledge_graph.sh

# Step 2: Start server (in separate terminal)
cd backend
python server.py --data_source SingleTopic

# Step 3: Generate answers (~5-10 minutes)
python test_scripts/singletopic/4_generate_answers.py

# Step 4: Evaluate results (<1 minute)
python test_scripts/singletopic/5_evaluate_results.py
```

---

## Prerequisites

### 1. Environment Setup

```bash
# Ensure Python virtual environment is activated
# (from project root)
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### 2. OpenAI API Key

```bash
# Create API key file (required for graph building)
echo "sk-your-api-key-here" > openai_api_key.txt
```

### 3. Verify Data Files

```bash
# Check if corpus and questions exist
ls datasets/SingleTopic/raw/corpus.jsonl  # Should exist (717KB)
ls datasets/SingleTopic/processed/all_questions_unified.csv  # Should exist (161 rows)
```

---

## Output Files

After running evaluation, you'll find:

```
datasets/SingleTopic/
├── results/
│   ├── generation_results.csv          # BiG-RAG's answers for all questions
│   ├── evaluation_report.json          # Detailed metrics (machine-readable)
│   ├── evaluation_report.csv           # Metrics by question type
│   └── evaluation_report.md            # Human-readable report
```

---

## Understanding the Metrics

### Answer Quality Metrics (for answerable questions)

1. **Exact Match (EM)**: Strictest metric - 1.0 if normalized answer exactly matches golden answer
   - Used for benchmarking against papers
   - Very sensitive to wording differences

2. **F1 Score**: Standard QA metric - measures token overlap
   - Balances precision and recall
   - **Recommended primary metric**

3. **Partial Match**: 1.0 if all golden answer words appear in prediction (any order)
   - More lenient - rewards correct facts even with extra details
   - Good for assessing if core information is present

4. **Fuzzy Match**: 1.0 if prediction is ≥80% similar to golden answer
   - Handles typos and minor wording differences
   - Uses edit distance algorithm

### Retrieval Quality Metrics

- **Retrieval Success Rate**: % of questions where BiG-RAG successfully retrieved context

### No-Answer Handling Metrics

- **Refusal Rate**: % of no-answer questions where model correctly said "no answer" / "unanswerable"
- **Hallucination Rate**: % of no-answer questions where model gave an answer (incorrect behavior)

### Interpretation Guide

**Good Performance**:
- EM: 0.40-0.60 (40-60%)
- F1: 0.60-0.80 (60-80%)
- Partial Match: 0.70-0.90 (70-90%)
- Retrieval Success: >0.90 (>90%)
- No-Answer Refusal: >0.70 (>70%)
- No-Answer Hallucination: <0.30 (<30%)

**Why use multiple metrics?**
- EM is too strict (penalizes valid alternative wordings)
- F1 balances strictness and leniency
- Partial/Fuzzy help understand system behavior
- Retrieval quality shows if graph is working
- No-answer handling tests if model knows its limits

---

## Example Output

After running evaluation, `evaluation_report.md` will look like:

```markdown
# SingleTopic Evaluation Report

**Generated**: 2025-01-12 14:30:52
**Dataset**: SingleTopic

## Overall Metrics

- **Total Questions**: 160
- **Successful**: 158 (98.8%)
- **Failed**: 2

### Answer Quality (Answerable Questions)

- **Exact Match (EM)**: 0.4583 (45.83%)
- **F1 Score**: 0.6725 (67.25%)
- **Partial Match**: 0.7814 (78.14%)
- **Fuzzy Match (80%)**: 0.5472 (54.72%)

### Retrieval Quality

- **Retrieval Success Rate**: 0.9430 (94.30%)

### No-Answer Handling

- **Total No-Answer Questions**: 39
- **Refusal Rate**: 0.7250 (72.50%)
- **Hallucination Rate**: 0.2750 (27.50%)

## Metrics by Question Type

| Question Type | Count | EM | F1 | Partial | Fuzzy | Refusal Rate | Hallucination Rate |
|---------------|-------|----|----|---------|-------|--------------|---------------------|
| single_passage | 62 | 0.5500 | 0.7200 | 0.8200 | 0.6100 | - | - |
| multi_passage | 58 | 0.3800 | 0.6300 | 0.7500 | 0.5000 | - | - |
| no_answer | 39 | - | - | - | - | 72.5% | 27.5% |
```

---

## Troubleshooting

### Problem: "Server is not running"

**Solution**: Start the backend server

```bash
cd backend
python server.py --data_source SingleTopic
```

Check if running:
```bash
curl http://localhost:8001/
```

### Problem: "OpenAI API rate limit"

**Solution**: Wait a few minutes, then retry. OpenAI has rate limits per minute.

Alternative: Use pre-built knowledge graph (if available)

### Problem: "Row count mismatch"

**Cause**: Something went wrong during answer generation (network error, timeout, etc.)

**Solution**: Delete `generation_results.csv` and re-run step 4:

```bash
rm datasets/SingleTopic/results/generation_results.csv
python test_scripts/singletopic/4_generate_answers.py
```

### Problem: "Knowledge graph build failed"

**Common causes**:
1. No OpenAI API key or invalid key
2. Insufficient disk space (need ~5GB)
3. Corpus file is corrupted

**Solution**:
```bash
# Check API key
cat openai_api_key.txt  # Should show sk-...

# Check disk space
df -h  # Need at least 5GB free

# Verify corpus
head -1 datasets/SingleTopic/raw/corpus.jsonl | python -m json.tool
```

### Problem: "All metrics are 0.0"

**Cause**: Server is using wrong dataset or generation failed

**Solution**:
1. Check server dataset:
   ```bash
   curl http://localhost:8001/ | grep dataset
   # Should show "dataset": "SingleTopic"
   ```

2. Check if answers were generated:
   ```bash
   head -2 datasets/SingleTopic/results/generation_results.csv
   # Should show question, golden_answer, generated_answer columns
   ```

---

## Advanced Usage

### Rebuild Knowledge Graph

```bash
# Delete existing graph
rm -rf expr/SingleTopic

# Rebuild
bash test_scripts/singletopic/3_build_knowledge_graph.sh
```

### Generate Answers Only (skip evaluation)

```bash
python test_scripts/singletopic/4_generate_answers.py
```

### Evaluate Existing Results

```bash
# If you already have generation_results.csv
python test_scripts/singletopic/5_evaluate_results.py
```

### Customize Evaluation Thresholds

Edit `5_evaluate_results.py`:

```python
# Line 19-20
FUZZY_MATCH_THRESHOLD = 0.8  # Default: 80% similarity
PARTIAL_MATCH_MIN_WORDS = 2  # Default: minimum 2 words
```

### Analyze Failed Questions

```python
import pandas as pd

# Load results
df = pd.read_csv("datasets/SingleTopic/results/generation_results.csv")

# Filter failed questions
failed = df[df['error'] != '']
print(f"Failed: {len(failed)}")
print(failed[['question', 'error']])

# Filter low F1 scores (if you have per-question metrics)
low_f1 = df[df['f1_score'] < 0.3]  # Requires adding f1_score column
```

---

## Customizing for Other Datasets

To adapt these scripts for other datasets:

### 1. Create New Directory

```bash
mkdir test_scripts/your_dataset
```

### 2. Copy Scripts

```bash
cp test_scripts/singletopic/* test_scripts/your_dataset/
```

### 3. Update Configuration

Edit each script to change:

```python
# In Python scripts:
DATASET = "your_dataset"  # Change from "SingleTopic"

# In bash scripts:
# Change dataset name references
```

### 4. Verify Data Format

Ensure your dataset has:
- `datasets/your_dataset/raw/corpus.jsonl` (BiG-RAG format)
- `datasets/your_dataset/processed/all_questions_unified.csv` (with columns: question, golden_answer, document_index, question_type)

---

## Performance Tips

### Speed Up Answer Generation

1. **Reduce timeout**: Edit `4_generate_answers.py` line 27
   ```python
   TIMEOUT_SECONDS = 30  # Default: 60
   ```

2. **Disable reranking**: Edit line 29
   ```python
   ENABLE_RERANKING = False  # Default: True
   ```

3. **Reduce top_k**: Edit line 28
   ```python
   TOP_K = 3  # Default: 5
   ```

### Speed Up Knowledge Graph Building

1. **Use smaller corpus**: Filter corpus.jsonl to fewer documents
2. **Use faster embedding model**: Edit `script_build.py` (not recommended - affects quality)
3. **Skip relation extraction**: Edit `script_build.py` to disable relation extraction (not recommended)

---

## Citation

If you use these scripts in research, please cite:

```bibtex
@software{bigrag_singletopic_eval,
  title={SingleTopic Evaluation Scripts for BiG-RAG},
  author={BiG-RAG Contributors},
  year={2025},
  url={https://github.com/dhrubo326/BiG-RAG}
}
```

---

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/dhrubo326/BiG-RAG/issues
- **Documentation**: See main README.md and CLAUDE.md

---

**Last Updated**: 2025-01-12
**Author**: BiG-RAG Development Team
