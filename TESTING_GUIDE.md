# KUET Document Testing Guide

**Date:** 2025-01-22
**Purpose:** Test document indexing using EXACT backend workflow

---

## What Went Wrong Before

### test_kuet_production_indexing.py (WRONG APPROACH)
- Tried to use ProductionKGPipeline directly
- Did NOT save to corpus.jsonl
- Did NOT follow backend workflow
- Result: VALIDATION FAILED, no graph files created

### Why It Failed
1. **Table validation failing:** Missing Bangla numerals (৭, ২, ৬, ৩)
2. **Paragraph extraction failing:** Only 65% numeric coverage
3. **Negative consistency score:** -82.14% (logic error in validator)

---

## Correct Approach

### Your Backend API Workflow (How It Actually Works)

```
POST /documents/upload
    |
    v
1. add_document_to_corpus()
    Save to: datasets/demo_test/raw/corpus.jsonl
    Format: {"id": "...", "contents": "...", "title": "...", "metadata": {...}}
    |
    v
2. process_document_background()
    Call: rag_instance.ainsert(content, metadata)
    This uses: OLD chunking-first approach (current BiGRAG)
    |
    v
3. Output Files
    Location: expr/demo_test/
    Files:
    - graph_chunk_entity_relation.graphml
    - vdb_entities.json
    - vdb_relations.json
    - vdb_chunks.json
    - kv_store_full_docs.json
    - kv_store_text_chunks.json
```

---

## How to Test Correctly

### test_kuet_backend_workflow.py (CORRECT APPROACH)

This script replicates the EXACT backend workflow:

**Step 1:** Save to corpus.jsonl
```python
await add_document_to_corpus(
    data_source="demo_test",
    doc_id="kuet_admission_2024_25",
    content=kuet_content,
    title="KUET Admission Info",
    metadata={...}
)
```

**Step 2:** Process with BiGRAG (OLD approach)
```python
rag = BiGRAG(working_dir="expr/demo_test", ...)
await rag.ainsert(content, metadata=doc_metadata)
```

**Step 3:** Inspect generated files in expr/demo_test/

**Step 4:** Load in backend API and test queries

---

## Running the Test

### Command

```bash
python test_kuet_backend_workflow.py
```

### Expected Output

```
[STEP 1/4] Adding document to corpus...
[OK] Document added to corpus
     Corpus file: datasets/demo_test/raw/corpus.jsonl
     Document ID: kuet_admission_2024_25
     Title: KUET Admission Information 2024-2025

[STEP 2/4] Processing with BiGRAG...
[INFO] Initializing BiGRAG...
       Working dir: expr/demo_test
       Chunk size: 1200 tokens
       Overlap: 100 tokens
       Model: gpt-4o

[INFO] Calling BiGRAG.ainsert()...
[OK] Processing completed in 60-120 seconds

PROCESSING SUMMARY
Total Entities: 50-150
Total Relations: 50-150
Total Chunks: 10-30
Processing Time: 60-120 seconds
Output Directory: expr/demo_test

[STEP 3/4] Inspecting output files...
[INFO] Files in expr/demo_test:
       - graph_chunk_entity_relation.graphml (50,000+ bytes)
       - vdb_entities.json (20,000+ bytes)
       - vdb_relations.json (20,000+ bytes)
       - vdb_chunks.json (15,000+ bytes)
       - kv_store_full_docs.json (10,000+ bytes)
       - kv_store_text_chunks.json (15,000+ bytes)

[INFO] Critical files check:
       [OK] graph_chunk_entity_relation.graphml
       [OK] vdb_entities.json
       [OK] vdb_relations.json
       [OK] vdb_chunks.json
       [OK] kv_store_full_docs.json
       [OK] kv_store_text_chunks.json

[OK] All critical files present

[STEP 4/4] Saving test results...
[OK] Test results saved to: test_results_kuet_backend.json

TEST COMPLETION SUMMARY
[OK] Test completed successfully

Generated files:
  - Corpus: datasets/demo_test/raw/corpus.jsonl
  - Graph: expr/demo_test/graph_chunk_entity_relation.graphml
  - Vector DBs: expr/demo_test/vdb_*.json
  - KV stores: expr/demo_test/kv_store_*.json

You can now:
  1. Start backend: cd backend && python server.py --data_source demo_test
  2. Start frontend: cd frontend && npm run dev
  3. View documents at: http://localhost:3000/documents
  4. Test queries at: http://localhost:3000/
```

---

## After Test Passes

### Verify in UI

1. **Start Backend:**
   ```bash
   cd backend
   python server.py --data_source demo_test
   ```

2. **Start Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open Browser:**
   ```
   http://localhost:3000/documents
   ```

4. **Check Document Appears:**
   - Should see "KUET Admission Information 2024-2025"
   - Should show entity/relation counts
   - Should be queryable

### Test Some Queries

**Query 1:** "KUET CSE তে কত আসন আছে?"
- Expected: Should find seat count from document

**Query 2:** "KUET তে কোন কোন বিভাগ আছে?"
- Expected: Should list departments (CSE, EEE, ME, CE, etc.)

**Query 3:** "KUET ভর্তির জন্য কি যোগ্যতা লাগে?"
- Expected: Should retrieve eligibility requirements

---

## Why This Approach is Correct

### Matches Production Workflow
- Uses SAME corpus.jsonl format as backend
- Uses SAME BiGRAG.ainsert() as backend
- Generates SAME output files as backend
- Can be loaded in backend API immediately

### Tests Real System
- Not testing ProductionKGPipeline in isolation
- Testing the ACTUAL workflow users will use
- Validates end-to-end integration

### Allows Comparison
- Can compare with OLD demo_test data (if you have backup)
- Can see if quality improved
- Can decide if ready to integrate ProductionKGPipeline

---

## Next Steps After Test

### If Test Succeeds (Files Created)

1. **Inspect Graph Quality:**
   - Check entity names are correct
   - Check relations are meaningful
   - Check no critical data missing

2. **Test Queries in UI:**
   - Run 10-20 test questions
   - Verify answers are correct
   - Compare with expectations

3. **Decide on ProductionKGPipeline:**
   - If OLD approach works well → Keep as-is
   - If OLD approach has issues → Integrate ProductionKGPipeline
   - If unsure → Test both and compare

### If Test Fails

1. **Check Error Logs:**
   - Look at console output
   - Check what stage failed (chunking, extraction, graph building)

2. **Debug Issues:**
   - If chunking fails → Check document format
   - If extraction fails → Check API key, model availability
   - If graph building fails → Check storage permissions

3. **Get Help:**
   - Share error logs
   - Share test_results_kuet_backend.json
   - Check CLAUDE.md troubleshooting section

---

## Files Created

1. **test_kuet_backend_workflow.py** - Main test script (RUN THIS)
2. **TESTING_GUIDE.md** - This documentation
3. **test_results_kuet_backend.json** - Test output (created after run)

---

## Important Notes

- This test uses the OLD BiGRAG approach (current production)
- This is NOT testing the NEW ProductionKGPipeline yet
- First validate OLD approach works on KUET document
- Then later we can integrate ProductionKGPipeline

---

## Questions?

If you have issues:
1. Check openai_api_key.txt exists and has valid key
2. Check KUET_Admission_info.md exists in project root
3. Check you have write permissions to datasets/ and expr/
4. Check no other process is using those directories
