# BiG-RAG Evaluation Endpoints - Testing Guide

**Version:** 1.0
**Date:** 2025-10-30
**Status:** ✅ Ready for Testing

---

## Quick Start

### 1. Server Already Running
Your server is running at: `http://localhost:8001`

### 2. Open Swagger UI
Navigate to: **http://localhost:8001/docs**

Scroll to the **"Evaluation"** section - you should see 4 new endpoints.

---

## Test Scenario 1: Simple Retrieval Evaluation (2 minutes)

**Purpose:** Test if retrieval metrics work correctly

**Steps:**
1. In Swagger UI, click **POST /eval/retrieval**
2. Click "Try it out"
3. Paste this JSON:

```json
{
  "queries": [
    {
      "question": "What is Artificial Intelligence?",
      "ground_truth_docs": ["doc_001"]
    }
  ],
  "dataset": "demo_test",
  "mode": "hybrid",
  "top_k": 5,
  "metrics": ["precision", "recall", "mrr"]
}
```

4. Click **"Execute"**

**Expected Result:**
```json
{
  "success": true,
  "total_queries": 1,
  "metrics": {
    "precision@5": 0.2,
    "recall@5": 1.0,
    "mrr": 1.0
  },
  "per_query_results": [
    {
      "question": "What is Artificial Intelligence?",
      "retrieved_docs": ["doc_001", "doc_003", "doc_005", "doc_007", "doc_010"],
      "relevant_retrieved": ["doc_001"],
      "metrics": {
        "precision@5": 0.2,
        "recall@5": 1.0,
        "mrr": 1.0
      }
    }
  ],
  "evaluation_time": 1.5
}
```

✅ **Success if:**
- `success: true`
- `mrr: 1.0` (found relevant doc at rank 1)
- `recall@5: 1.0` (retrieved the only relevant doc)
- Execution time < 5 seconds

---

## Test Scenario 2: Answer Quality Evaluation (3 minutes)

**Purpose:** Test answer generation and quality metrics

**Steps:**
1. Click **POST /eval/answer**
2. Click "Try it out"
3. Paste this JSON:

```json
{
  "test_cases": [
    {
      "question": "What is TensorFlow?",
      "ground_truth": "an open-source machine learning framework developed by Google",
      "use_rag": true
    }
  ],
  "dataset": "demo_test",
  "llm_provider": "openai",
  "model": "gpt-4o-mini",
  "metrics": ["em", "f1", "rouge_l"]
}
```

4. Click **"Execute"**

**Expected Result:**
```json
{
  "success": true,
  "total_questions": 1,
  "aggregate_metrics": {
    "exact_match": 0.0,
    "f1": 0.6,
    "rouge_l": 0.5
  },
  "per_question_results": [
    {
      "question": "What is TensorFlow?",
      "ground_truth": "an open-source machine learning framework developed by Google",
      "predicted_answer": "TensorFlow is an open-source machine learning framework created by Google...",
      "metrics": {
        "exact_match": 0.0,
        "f1": 0.6,
        "rouge_l": 0.5
      },
      "retrieval_used": true,
      "num_contexts_used": 3,
      "generation_time": 2.1
    }
  ],
  "total_time": 3.5
}
```

✅ **Success if:**
- `predicted_answer` mentions TensorFlow and Google
- `retrieval_used: true`
- `f1 > 0.4` (good token overlap)
- `num_contexts_used > 0`

---

## Test Scenario 3: Compare Retrieval Modes (4 minutes)

**Purpose:** Find which retrieval mode works best

**Steps:**
1. Click **POST /eval/compare**
2. Click "Try it out"
3. Paste this JSON:

```json
{
  "queries": [
    {
      "question": "What is deep learning?",
      "ground_truth_docs": ["doc_003", "doc_007"]
    }
  ],
  "dataset": "demo_test",
  "configurations": [
    {"name": "hybrid", "mode": "hybrid", "top_k": 5},
    {"name": "local", "mode": "local", "top_k": 5},
    {"name": "global", "mode": "global", "top_k": 5},
    {"name": "naive", "mode": "naive", "top_k": 5}
  ],
  "metrics": ["precision", "recall", "mrr"]
}
```

4. Click **"Execute"**

**Expected Result:**
```json
{
  "success": true,
  "comparison_results": {
    "hybrid": {
      "precision@5": 0.6,
      "recall@5": 1.0,
      "mrr": 0.5
    },
    "local": {
      "precision@5": 0.4,
      "recall@5": 0.8,
      "mrr": 0.5
    },
    "global": {
      "precision@5": 0.2,
      "recall@5": 0.5,
      "mrr": 0.33
    },
    "naive": {
      "precision@5": 0.4,
      "recall@5": 1.0,
      "mrr": 1.0
    }
  },
  "best_configuration": "hybrid",
  "ranking": ["hybrid", "naive", "local", "global"]
}
```

✅ **Success if:**
- All 4 modes have results
- `best_configuration` is identified
- Hybrid mode typically ranks in top 2

---

## Test Scenario 4: Batch Evaluation (5 minutes)

**Purpose:** Test batch processing on full dataset

**Steps:**
1. Click **POST /eval/batch**
2. Click "Try it out"
3. Paste this JSON:

```json
{
  "dataset_file": "test_datasets/eval_qa.json",
  "data_source": "demo_test",
  "mode": "hybrid",
  "top_k": 5,
  "metrics": ["em", "f1", "precision", "recall"],
  "use_llm": true,
  "save_results": true,
  "output_file": "evaluation_results/test_results.json",
  "limit": 10
}
```

4. Click **"Execute"**
5. Wait 30-60 seconds (processes 10 questions)

**Expected Result:**
```json
{
  "success": true,
  "dataset": "test_datasets/eval_qa.json",
  "total_questions": 10,
  "processed": 10,
  "failed": 0,
  "metrics": {
    "retrieval": {
      "precision@5": 0.68,
      "recall@5": 0.85
    },
    "answer": {
      "exact_match": 0.3,
      "f1": 0.62
    }
  },
  "performance": {
    "total_time": 45.2,
    "avg_time_per_query": 4.52,
    "total_llm_calls": 10,
    "total_embedding_calls": 10
  },
  "results_saved_to": "evaluation_results/test_results.json"
}
```

✅ **Success if:**
- `processed: 10` (all questions processed)
- `failed: 0` (no failures)
- Retrieval precision > 0.5
- Answer F1 > 0.5
- Results file created

**Verify Results Saved:**
```bash
# Windows
type evaluation_results\test_results.json

# Linux/Mac
cat evaluation_results/test_results.json
```

---

## Metric Interpretation

### Retrieval Metrics

| Metric | Range | Good Score | Meaning |
|--------|-------|------------|---------|
| **Precision@5** | 0.0-1.0 | > 0.6 | % of retrieved docs that are relevant |
| **Recall@5** | 0.0-1.0 | > 0.7 | % of relevant docs that were retrieved |
| **MRR** | 0.0-1.0 | > 0.7 | 1 / rank of first relevant doc |
| **NDCG@5** | 0.0-1.0 | > 0.7 | Quality of ranking (higher=better) |

### Answer Metrics

| Metric | Range | Good Score | Meaning |
|--------|-------|------------|---------|
| **Exact Match** | 0 or 1 | > 0.3 | Binary: exact match after normalization |
| **Token F1** | 0.0-1.0 | > 0.5 | Token-level overlap (precision + recall) |
| **ROUGE-L** | 0.0-1.0 | > 0.4 | Longest common subsequence |

---

## Troubleshooting

### Issue: "Dataset file not found"

```bash
# Create the test dataset directory
mkdir test_datasets

# Verify file exists
dir test_datasets\eval_qa.json  # Windows
ls test_datasets/eval_qa.json   # Linux/Mac
```

The file should have been created during implementation. Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md).

---

### Issue: Low Metric Scores (all near 0)

**Check 1:** Verify knowledge graph exists
```bash
curl "http://localhost:8001/graph/stats?dataset=demo_test"
```

Expected: Should show entities > 0, edges > 0

**Check 2:** List documents in corpus
```bash
python -c "import json; f=open('datasets/demo_test/raw/corpus.jsonl'); [print(json.loads(line).get('id')) for line in f]"
```

Expected: Should see doc_001, doc_002, ... doc_010

**Check 3:** Verify ground truth docs match corpus
- Test dataset uses doc_001 through doc_010
- Your corpus should have these IDs

---

### Issue: "OpenAI API Error"

**Solution:**
```bash
# Check if API key file exists
type openai_api_key.txt  # Windows
cat openai_api_key.txt   # Linux/Mac

# If missing, create it
echo "your-api-key-here" > openai_api_key.txt

# Restart server
python script_api.py --data_source demo_test
```

---

### Issue: Slow Performance (> 10 sec per query)

**Try smaller batch:**
```json
{
  "dataset_file": "test_datasets/eval_qa.json",
  "limit": 3,
  "use_llm": false
}
```

**Or test retrieval only:**
```bash
curl -X POST "http://localhost:8001/eval/retrieval" \
  -H "Content-Type: application/json" \
  -d '{"queries":[{"question":"What is AI?","ground_truth_docs":["doc_001"]}],"metrics":["precision"]}'
```

---

## Testing Checklist

- [ ] ✅ Server running (`curl http://localhost:8001/health`)
- [ ] ✅ Swagger UI accessible (http://localhost:8001/docs)
- [ ] ✅ Test dataset exists (`test_datasets/eval_qa.json`)
- [ ] ✅ Test 1: Retrieval evaluation works
- [ ] ✅ Test 2: Answer evaluation works
- [ ] ✅ Test 3: Mode comparison works
- [ ] ✅ Test 4: Batch evaluation works
- [ ] ✅ All metrics between 0.0 and 1.0
- [ ] ✅ Results saving works
- [ ] ✅ Performance acceptable (< 60 sec for 10 questions)

---

## Implementation Progress

**✅ Completed:**
1. Created `api/metrics.py` - Metric calculation functions
2. Created `api/models_eval.py` - Pydantic models
3. Created `api/evaluation.py` - Evaluation logic
4. Added 4 endpoints to `script_api.py`
5. Created `test_datasets/eval_qa.json` - Test dataset
6. Created this testing guide

**Total Code:** ~1,500 lines of evaluation code

**New Files:**
- `api/metrics.py` (485 lines)
- `api/models_eval.py` (350 lines)
- `api/evaluation.py` (350 lines)
- `test_datasets/eval_qa.json` (125 lines)
- `script_api.py` (320 lines added)

---

## Next Steps

After testing succeeds:

1. **Analyze Results** - Which mode performs best?
2. **Tune Parameters** - Try different `top_k` values
3. **Create Custom Datasets** - Build eval sets for your domain
4. **Monitor Over Time** - Track metrics as you add documents
5. **Optimize** - Use `/eval/compare` to find best configuration

---

## Summary

**All evaluation endpoints are now implemented and ready for testing!**

Start with **Test Scenario 1** (simplest), then progress through scenarios 2-4.

If all 4 scenarios pass, you have a fully functional evaluation system for measuring:
- ✅ Retrieval quality (IR metrics)
- ✅ Answer accuracy (NLP metrics)
- ✅ Configuration comparison
- ✅ Large-scale batch evaluation

**Happy Testing! 🚀**
