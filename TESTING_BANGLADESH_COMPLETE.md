# Complete Testing Guide - Bangladesh.txt

**Version:** 2.0 - Updated 2025-10-31
**Status:** ✅ All bugs fixed - Ready to use!

This guide walks you through testing ALL important endpoints using Bangladesh.txt.

## 🔥 What Changed (IMPORTANT!)

This guide has been updated to reflect critical bug fixes:

**✅ Fixed:**
1. **Document ID Format** - Now uses `doc-` prefix with 32-char hash (was `upload-` with 16 chars)
2. **ID Matching** - Document IDs now match between registry and knowledge graph
3. **Source ID Tracking** - Evaluation endpoints now properly track document sources
4. **Empty Dataset Startup** - Server now starts correctly with empty datasets

**🚨 Critical New Step:**
- **Step 2.5** added - MUST verify document has non-zero stats before testing
- If stats are 0, evaluation will fail (empty knowledge graph)

**Prerequisites:**
- Server running: `python script_api.py --data_source demo_test`
- OpenAI API key configured
- Bangladesh.txt file in current directory
- **Latest code changes committed** (document ID fixes applied)

---

## Step 1: Upload Bangladesh.txt

**What this does:** Uploads the document and starts background processing (KG building)

```bash
curl -X POST "http://localhost:8001/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@Bangladesh.txt;type=text/plain" \
  -F "title=Bangladesh - Country Overview" \
  -F "data_source=demo_test" \
  -F "process_async=true" \
  -F 'metadata={"category":"Geography","tags":["Bangladesh","South Asia","Country"]}'
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Document queued for processing",
  "document_id": "doc-53a0479813a7da9e631fcac2f7c0a80d",
  "job_id": "job--53e973922dfaabba",
  "filename": "Bangladesh.txt",
  "status": "pending"
}
```

**CRITICAL - Document ID Format:**
- ✅ **NEW format**: `doc-` prefix with 32-character hash (e.g., `doc-53a0479813a7da9e631fcac2f7c0a80d`)
- ❌ **OLD format**: `upload-` with 16 chars (deprecated, won't work with evaluation)

**ACTION REQUIRED:**
- 📝 **WRITE DOWN BOTH IDs** - you'll need them for all tests!
- Copy the `job_id` (e.g., `job--53e973922dfaabba`)
- Copy the `document_id` (e.g., `doc-53a0479813a7da9e631fcac2f7c0a80d`)

---

## Step 2: Monitor Processing Status

**What this does:** Checks if document processing is complete

Replace `JOB_ID` with your actual job_id from Step 1:

```bash
# Replace JOB_ID with your actual job_id
curl "http://localhost:8001/status/YOUR_JOB_ID_HERE"
```

**Example:**
```bash
curl "http://localhost:8001/status/job--5c553d5b175cbb32"
```

**Expected Response (Processing):**
```json
{
  "job_id": "job-xxxxx",
  "status": "processing",
  "stage": "extracting_entities",
  "progress": 0.4,
  "started_at": "2025-10-31T...",
  "completed_at": null
}
```

**Expected Response (Completed):**
```json
{
  "job_id": "job-xxxxx",
  "status": "completed",
  "stage": "completed",
  "progress": 1.0,
  "completed_at": "2025-10-31T..."
}
```

**Wait time:** 2-5 minutes (document is ~23,000 characters)

**Keep checking every 30 seconds until status = "completed"**

---

## Step 2.5: 🚨 CRITICAL - Verify Document Has Stats (NEW STEP)

**BEFORE proceeding to testing, you MUST verify the document was indexed properly!**

```bash
# Replace YOUR_DOC_ID with your document_id from Step 1
curl "http://localhost:8001/documents/YOUR_DOC_ID?include_entities=true"
```

**Example:**
```bash
curl "http://localhost:8001/documents/doc-53a0479813a7da9e631fcac2f7c0a80d?include_entities=true"
```

**CRITICAL - Check Stats:**
```json
{
  "document_id": "doc-53a0479813a7da9e631fcac2f7c0a80d",
  "status": "indexed",
  "stats": {
    "chunks": 15,      // ❗ MUST be > 0
    "entities": 80,    // ❗ MUST be > 0
    "edges": 60,       // ❗ MUST be > 0
    "tokens": 6000     // ❗ MUST be > 0
  },
  "top_entities": [
    {"name": "BANGLADESH", "type": "LOCATION"},
    {"name": "DHAKA", "type": "LOCATION"},
    ...
  ]
}
```

**🚨 IF ALL STATS ARE 0:**
```json
"stats": {
  "chunks": 0,    // ❌ BAD!
  "entities": 0,  // ❌ BAD!
  "edges": 0      // ❌ BAD!
}
```

**This means the document failed to index! You must:**
1. Check server logs for errors
2. Check job status for error message
3. **DO NOT continue testing** - evaluation will fail!

**✅ ONLY proceed to Step 3 if stats show non-zero values!**

---

## Step 3: Test Search/Retrieval Endpoint

**What this does:** Tests if Bangladesh document is retrievable

### Test 3a: Simple Search

```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is Bangladesh?"],
    "mode": "hybrid",
    "top_k": 5,
    "dataset": "demo_test"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "results": [
    {
      "query": "What is Bangladesh?",
      "results": [
        {
          "id": "upload-xxxxx",
          "title": "Bangladesh - Country Overview",
          "content": "Bangladesh, officially the People's Republic of Bangladesh...",
          "score": 0.85
        }
      ]
    }
  ]
}
```

**Look for:** Your Bangladesh document should appear in top results

### Test 3b: Entity-based Search

```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Dhaka capital of Bangladesh"],
    "mode": "local",
    "top_k": 5,
    "dataset": "demo_test"
  }'
```

### Test 3c: Relation-based Search

```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Bangladesh independence 1971"],
    "mode": "global",
    "top_k": 5,
    "dataset": "demo_test"
  }'
```

### Test 3d: Naive Text Search (Baseline)

```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Bengali language movement 1952"],
    "mode": "naive",
    "top_k": 5,
    "dataset": "demo_test"
  }'
```

---

## Step 4: Test Q&A / Chat Endpoint

**What this does:** Uses LLM to generate answers based on retrieved context

### Test 4a: Simple Question

```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of Bangladesh?",
    "dataset": "demo_test",
    "mode": "hybrid",
    "use_llm": true,
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "stream": false
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "question": "What is the capital of Bangladesh?",
  "answer": "The capital of Bangladesh is Dhaka.",
  "contexts": [
    {
      "content": "Dhaka, the capital and largest city...",
      "score": 0.92
    }
  ],
  "generation_time": 1.5
}
```

### Test 4b: Historical Question

```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "When did Bangladesh gain independence?",
    "dataset": "demo_test",
    "mode": "hybrid",
    "use_llm": true,
    "llm_provider": "openai",
    "model": "gpt-4o-mini"
  }'
```

**Expected Answer:** "December 16, 1971" or similar

### Test 4c: Complex Multi-hop Question

```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the Bengali language movement and what year did it happen?",
    "dataset": "demo_test",
    "mode": "hybrid",
    "use_llm": true,
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "top_k": 10
  }'
```

### Test 4d: Geographic Question

```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which countries border Bangladesh?",
    "dataset": "demo_test",
    "mode": "hybrid",
    "use_llm": true,
    "llm_provider": "openai",
    "model": "gpt-4o-mini"
  }'
```

**Expected Answer:** "India, Myanmar" or similar

---

## Step 5: Test Evaluation - Retrieval Quality

**What this does:** Measures how well the system retrieves relevant documents

### Test 5a: Evaluate Single Query Retrieval

```bash
curl -X POST "http://localhost:8001/eval/retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "question": "What is the capital of Bangladesh?",
        "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"]
      }
    ],
    "dataset": "demo_test",
    "mode": "hybrid",
    "top_k": 5,
    "metrics": ["precision", "recall", "f1", "mrr", "ndcg"]
  }'
```

**IMPORTANT:** Replace `doc-53a0479813a7da9e631fcac2f7c0a80d` with your actual document_id from Step 1
- ✅ Use the FULL 32-character hash
- ✅ Must start with `doc-` prefix

**Expected Response:**
```json
{
  "success": true,
  "total_queries": 1,
  "metrics": {
    "precision@5": 0.2,
    "recall@5": 1.0,
    "f1@5": 0.333,
    "mrr": 1.0,
    "ndcg@5": 1.0
  },
  "per_query_results": [...]
}
```

**Good signs:**
- `recall@5` close to 1.0 (found the document)
- `mrr` close to 1.0 (document ranked first)
- `ndcg@5` > 0.8 (good ranking quality)

### Test 5b: Multiple Query Evaluation

```bash
# IMPORTANT: Replace doc-xxxxx with YOUR actual document_id!
curl -X POST "http://localhost:8001/eval/retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "question": "What is the capital of Bangladesh?",
        "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"]
      },
      {
        "question": "When did Bangladesh gain independence?",
        "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"]
      },
      {
        "question": "Which countries border Bangladesh?",
        "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"]
      }
    ],
    "dataset": "demo_test",
    "mode": "hybrid",
    "top_k": 5,
    "metrics": ["precision", "recall", "mrr"]
  }'
```

---

## Step 6: Test Evaluation - Answer Quality

**What this does:** Measures LLM answer quality against ground truth

### Test 6a: Evaluate Single Answer

```bash
curl -X POST "http://localhost:8001/eval/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "question": "What is the capital of Bangladesh?",
        "ground_truth": "Dhaka",
        "use_rag": true
      }
    ],
    "dataset": "demo_test",
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "metrics": ["em", "f1", "rouge_l"]
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "total_questions": 1,
  "aggregate_metrics": {
    "exact_match": 1.0,
    "f1_score": 1.0,
    "rouge_l": 0.95
  },
  "per_question_results": [
    {
      "question": "What is the capital of Bangladesh?",
      "ground_truth": "Dhaka",
      "predicted_answer": "Dhaka",
      "metrics": {
        "exact_match": 1.0,
        "f1_score": 1.0,
        "rouge_l": 1.0
      }
    }
  ]
}
```

### Test 6b: Evaluate Multiple Answers

```bash
curl -X POST "http://localhost:8001/eval/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "question": "What is the capital of Bangladesh?",
        "ground_truth": "Dhaka",
        "use_rag": true
      },
      {
        "question": "When did Bangladesh gain independence?",
        "ground_truth": "16 December 1971",
        "use_rag": true
      },
      {
        "question": "What is the second-largest city in Bangladesh?",
        "ground_truth": "Chittagong",
        "use_rag": true
      }
    ],
    "dataset": "demo_test",
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "metrics": ["em", "f1", "rouge_l"]
  }'
```

**Good scores:**
- `exact_match` > 0.7 (70%+ exact matches)
- `f1_score` > 0.8 (80%+ token overlap)
- `rouge_l` > 0.75 (good sequence matching)

---

## Step 7: Test Comparison - Retrieval Modes

**What this does:** Compares hybrid vs local vs global vs naive modes

```bash
# IMPORTANT: Replace doc-xxxxx with YOUR actual document_id!
curl -X POST "http://localhost:8001/eval/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "question": "What is Bangladesh?",
        "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"]
      }
    ],
    "dataset": "demo_test",
    "configurations": [
      {
        "name": "hybrid",
        "mode": "hybrid",
        "top_k": 5
      },
      {
        "name": "entity_only",
        "mode": "local",
        "top_k": 5
      },
      {
        "name": "relation_only",
        "mode": "global",
        "top_k": 5
      },
      {
        "name": "naive_text",
        "mode": "naive",
        "top_k": 5
      }
    ],
    "metrics": ["precision", "recall", "mrr"]
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "comparison_results": {
    "hybrid": {
      "precision@5": 0.4,
      "recall@5": 1.0,
      "mrr": 1.0
    },
    "entity_only": {
      "precision@5": 0.2,
      "recall@5": 1.0,
      "mrr": 0.5
    },
    "relation_only": {
      "precision@5": 0.2,
      "recall@5": 0.5,
      "mrr": 0.33
    },
    "naive_text": {
      "precision@5": 0.2,
      "recall@5": 1.0,
      "mrr": 1.0
    }
  },
  "best_configuration": "hybrid"
}
```

**What to look for:**
- Hybrid should usually perform best (highest MRR/precision)
- Local (entity) good for factual queries
- Global (relation) good for relationship queries
- Naive is baseline (no graph structure)

---

## Step 8: List All Documents

**What this does:** Shows all documents in the dataset

```bash
curl -X GET "http://localhost:8001/documents?dataset=demo_test"
```

**Expected Response:**
```json
{
  "success": true,
  "total": 13,
  "documents": [
    {
      "document_id": "upload-xxxxx",
      "title": "Bangladesh - Country Overview",
      "filename": "Bangladesh.txt",
      "upload_date": "2025-10-31T...",
      "status": "indexed",
      "content_length": 6953
    },
    ...
  ]
}
```

**Look for:** Your Bangladesh document should be in the list with status "indexed"

---

## Step 9: Get Document Details

**What this does:** Retrieves detailed information about your uploaded document

Replace `DOCUMENT_ID` with your document_id from Step 1:

```bash
curl -X GET "http://localhost:8001/documents/YOUR_DOCUMENT_ID_HERE?dataset=demo_test"
```

**Example:**
```bash
curl -X GET "http://localhost:8001/documents/upload-56304e0ee590ff89?dataset=demo_test"
```

**Expected Response:**
```json
{
  "success": true,
  "document": {
    "document_id": "upload-xxxxx",
    "title": "Bangladesh - Country Overview",
    "filename": "Bangladesh.txt",
    "upload_date": "2025-10-31T...",
    "indexed_date": "2025-10-31T...",
    "status": "indexed",
    "content_length": 6953,
    "metadata": {
      "category": "Geography",
      "tags": ["Bangladesh", "South Asia", "Country"]
    },
    "stats": {
      "total_chunks": 8,
      "total_entities": 50,
      "total_relations": 35
    }
  }
}
```

---

## Step 10: Test Batch Evaluation (Optional)

**What this does:** Evaluates multiple questions from a JSON file

First, create a test file:

**🚨 CRITICAL: You MUST replace `doc-xxxxx` below with YOUR actual document_id before running!**

```bash
cat > bangladesh_test_questions.json << 'EOF'
{
  "name": "Bangladesh QA Test Set",
  "version": "1.0",
  "dataset": "demo_test",
  "total_questions": 5,
  "questions": [
    {
      "id": "bd_q001",
      "question": "What is the capital of Bangladesh?",
      "ground_truth_answer": "Dhaka",
      "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"],
      "difficulty": "easy",
      "type": "factual"
    },
    {
      "id": "bd_q002",
      "question": "When did Bangladesh gain independence?",
      "ground_truth_answer": "16 December 1971",
      "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"],
      "difficulty": "easy",
      "type": "historical"
    },
    {
      "id": "bd_q003",
      "question": "Which countries border Bangladesh?",
      "ground_truth_answer": "India and Myanmar",
      "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"],
      "difficulty": "medium",
      "type": "geographic"
    },
    {
      "id": "bd_q004",
      "question": "What is the second-largest city in Bangladesh?",
      "ground_truth_answer": "Chittagong",
      "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"],
      "difficulty": "medium",
      "type": "factual"
    },
    {
      "id": "bd_q005",
      "question": "What was the Bengali language movement?",
      "ground_truth_answer": "A political movement in 1952 to establish Bengali as an official language of Pakistan",
      "ground_truth_docs": ["doc-53a0479813a7da9e631fcac2f7c0a80d"],
      "difficulty": "hard",
      "type": "historical"
    }
  ]
}
EOF
```

**BEFORE running this command:**
1. Replace ALL instances of `doc-53a0479813a7da9e631fcac2f7c0a80d` with YOUR actual document_id
2. Use Find & Replace in your text editor to replace all occurrences at once

Then run batch evaluation:

```bash
curl -X POST "http://localhost:8001/eval/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_file": "bangladesh_test_questions.json",
    "data_source": "demo_test",
    "mode": "hybrid",
    "top_k": 5,
    "metrics": ["em", "f1", "precision", "recall"],
    "use_llm": true,
    "llm_provider": "openai",
    "save_results": true,
    "output_file": "evaluation_results/bangladesh_eval.json"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "dataset": "demo_test",
  "total_questions": 5,
  "processed": 5,
  "failed": 0,
  "metrics": {
    "retrieval": {
      "precision@5": 0.8,
      "recall@5": 1.0
    },
    "answer": {
      "exact_match": 0.6,
      "f1_score": 0.8
    }
  },
  "results_saved_to": "evaluation_results/bangladesh_eval.json"
}
```

---

## Step 11: Check Queue Stats

**What this does:** Shows processing queue statistics

```bash
curl -X GET "http://localhost:8001/queue/stats"
```

**Expected Response:**
```json
{
  "success": true,
  "queue_stats": {
    "pending": 0,
    "processing": 0,
    "completed": 1,
    "failed": 0,
    "total": 1
  }
}
```

---

## Step 12: Test Delete Document (Optional - Only if you want to remove)

**What this does:** Removes document from the dataset

**WARNING:** This will delete your Bangladesh document!

```bash
curl -X DELETE "http://localhost:8001/documents/YOUR_DOCUMENT_ID_HERE?dataset=demo_test"
```

**Only run if you want to clean up!**

---

## Troubleshooting

### Issue 1: Upload fails with 500 error

**Check:**
- Is OpenAI API key set? (Check `openai_api_key.txt`)
- Is server running? (`python script_api.py --data_source demo_test`)

### Issue 2: Status stuck at "pending"

**Solution:**
- Check server logs: Look at terminal where server is running
- Common cause: OpenAI API rate limit or invalid key

### Issue 3: Search returns empty results

**Solution:**
- Wait for processing to complete (status must be "completed")
- Check document was indexed: `curl "http://localhost:8001/documents?dataset=demo_test"`

### Issue 4: Evaluation endpoints return 404

**Solution:**
- Make sure you replaced `upload-xxxxx` with your actual document_id
- Check document exists: `curl "http://localhost:8001/documents?dataset=demo_test"`

### Issue 5: LLM answers are wrong

**Possible causes:**
- Not enough context retrieved (increase `top_k`)
- Wrong retrieval mode (try `hybrid` instead of `local` or `global`)
- Document not properly indexed (check stats in Step 9)

---

## Success Metrics

After completing all tests, you should see:

✅ **Upload:** Status = "completed", progress = 1.0
✅ **Search:** Bangladesh document appears in top 3 results
✅ **Q&A:** LLM provides accurate answers about Bangladesh
✅ **Retrieval Eval:** Recall@5 > 0.8, MRR > 0.8
✅ **Answer Eval:** F1 > 0.7, Exact Match > 0.5
✅ **Comparison:** Hybrid mode performs best
✅ **List:** Bangladesh document visible with status "indexed"

---

## Quick Reference: Important IDs

**📝 Fill these in as you go:**

```
Job ID (from Step 1):
  Example: job--53e973922dfaabba
  Your ID: ___________________________

Document ID (from Step 1):
  Example: doc-53a0479813a7da9e631fcac2f7c0a80d
  Your ID: ___________________________
```

**🚨 CRITICAL CHECKS:**

Before proceeding with tests, verify:
- [ ] Document ID starts with `doc-` (NOT `upload-`)
- [ ] Document ID is 36 characters total (4 chars "doc-" + 32 char hash)
- [ ] Status = "completed" in Step 2
- [ ] Stats show chunks > 0, entities > 0 in Step 2.5

**If ANY of these fail, DO NOT continue testing - fix the issue first!**

---

## Summary of Endpoints Tested

| Endpoint | Purpose | Step |
|----------|---------|------|
| POST /upload | Upload document | 1 |
| GET /status/{job_id} | Check processing status | 2 |
| POST /search | Retrieve relevant documents | 3 |
| POST /ask | Generate answers with LLM | 4 |
| POST /eval/retrieval | Evaluate retrieval quality | 5 |
| POST /eval/answer | Evaluate answer quality | 6 |
| POST /eval/compare | Compare retrieval modes | 7 |
| GET /documents | List all documents | 8 |
| GET /documents/{id} | Get document details | 9 |
| POST /eval/batch | Batch evaluation | 10 |
| GET /queue/stats | Queue statistics | 11 |
| DELETE /documents/{id} | Delete document | 12 |

---

**You're all set! Start from Step 1 and paste the results/errors here if you need help.**
