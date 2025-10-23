# How to Run BiG-RAG Tests (Step-by-Step Guide)

## Prerequisites Check

Before running tests, ensure you have:
1. ✓ Virtual environment created (`venv` folder exists)
2. ✓ OpenAI API key in `openai_api_key.txt`
3. ✓ Demo dataset created in `datasets/demo_test/`

## Step-by-Step Instructions

### Step 1: Activate Virtual Environment

```bash
# On Windows:
venv\Scripts\activate

# You should see (venv) at the start of your command prompt
```

### Step 2: Install Missing Dependencies (if needed)

```bash
# Check if packages are installed
python -c "import torch; import openai; import faiss; print('All packages OK')"

# If you get errors, install:
pip install torch numpy openai tiktoken tenacity networkx faiss-cpu aiohttp pydantic
```

### Step 3: Verify OpenAI API Key

```bash
# Check if openai_api_key.txt exists and has content
type openai_api_key.txt
```

The file should contain your OpenAI API key starting with `sk-`

### Step 4: Run Test 1 - Build Knowledge Graph

```bash
python test_build_graph.py
```

**What to expect:**
- Takes 3-8 minutes
- Creates knowledge graph in `expr/demo_test/`
- Logs saved to `build_graph.log`

**Look for:**
- ✓ Messages showing successful batch processing
- ✓ Final message: "BUILD SUCCESSFUL!"
- ✓ Files created: `kv_store_text_chunks.json`, `kv_store_entities.json`, `kv_store_bipartite_edges.json`

**If errors occur:**
- Check `build_graph.log` for details
- Verify API key is valid
- Report the error message to me

### Step 5: Run Test 2 - Test Retrieval

```bash
python test_retrieval.py
```

**What to expect:**
- Takes 1-2 minutes
- Tests query functionality
- Logs saved to `test_retrieval.log`
 
**Look for:**
- ✓ Retrieved results for each query
- ✓ Coherence scores (0-1)
- ✓ Success rate: 100%

**If errors occur:**
- Check `test_retrieval.log`
- Ensure Step 4 completed successfully
- Report the error message to me

### Step 6: Run Test 3 - End-to-End RAG Test

```bash
python test_end_to_end.py
```

**What to expect:**
- Takes 2-4 minutes
- Tests complete RAG pipeline (retrieval + answer generation)
- Logs saved to `test_end_to_end.log`

**Look for:**
- ✓ Retrieved context for each question
- ✓ Generated answers using gpt-4o-mini
- ✓ Match rate with expected answers

**If errors occur:**
- Check `test_end_to_end.log`
- Ensure Steps 4 and 5 completed successfully
- Report the error message to me

## Quick Command Reference

```bash
# Activate venv (Windows)
venv\Scripts\activate

# Run all tests in sequence
python test_build_graph.py
python test_retrieval.py
python test_end_to_end.py

# Check logs if errors occur
type build_graph.log
type test_retrieval.log
type test_end_to_end.log

# Verify output files
dir expr\demo_test\
```

## What to Report Back to Me

For each test, please tell me:

1. **Did it complete successfully?** (Yes/No)
2. **Any error messages?** (Copy the error if any)
3. **Final statistics** (from the console output)

Example:
```
Test 1: ✓ Success
- Text Chunks: 25
- Entities: 127
- Relations: 89

Test 2: ✓ Success
- Success rate: 100%
- Average coherence: 0.8234

Test 3: ✓ Success
- Answer matches: 8/10 (80%)
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Install missing package:
```bash
pip install <package-name>
```

### Issue: "OpenAI API key not found"
**Solution:** Create `openai_api_key.txt` with your API key:
```bash
echo sk-your-key-here > openai_api_key.txt
```

### Issue: "Knowledge graph not found"
**Solution:** Run `test_build_graph.py` first before other tests

### Issue: Script hangs or is very slow
**Solution:**
- This is normal during build (3-8 minutes)
- Check `build_graph.log` to see progress
- Press Ctrl+C to cancel if needed

## Cost Estimate

Running all tests on demo dataset:
- **Total cost: ~$0.02-0.05 USD**
- Entity extraction: ~$0.01
- Embeddings: ~$0.001
- Answer generation: ~$0.01

## Next Steps After Successful Tests

Once all tests pass, we can:
1. Create FastAPI server for REST API endpoints
2. Scale to your own dataset
3. Integrate with your applications
4. Optimize performance and accuracy

---

**Ready to start?** Activate your venv and run the first test!

```bash
venv\Scripts\activate
python test_build_graph.py
```
