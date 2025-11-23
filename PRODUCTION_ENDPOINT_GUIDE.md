# Production Dataset Management Endpoint

## Overview

The new `/datasets/create-and-index` endpoint enables **dynamic dataset creation and document indexing** in production environments, perfect for scenarios where datasets cannot be pre-defined.

---

## Key Features

✅ **Dynamic Dataset Creation** - Creates datasets on-the-fly
✅ **Auto-Registration** - Automatically adds to `expr/subgraph_registry.json`
✅ **Production Pipeline Only** - Always uses table-aware extraction (no fallback)
✅ **Corpus Persistence** - Saves to `datasets/{data_source}/raw/corpus.jsonl`
✅ **Incremental Indexing** - Add documents one at a time
✅ **Unified Mode Ready** - Works seamlessly with multi-subgraph queries

---

## Requirements

### Server Configuration
```bash
# Start server in unified mode
python server.py --unified
```

### Environment Variables
```bash
# Required: OPENAI_API_KEY must be set in .env
OPENAI_API_KEY=sk-proj-your-key-here
```

**Important:** Production pipeline will **fail with error** if `OPENAI_API_KEY` is not set (no silent fallback to standard pipeline).

---

## API Endpoint

### `POST /datasets/create-and-index`

Creates a new dataset (if it doesn't exist) and indexes a document using the production pipeline.

**Endpoint:** `http://localhost:8001/datasets/create-and-index`

**Method:** POST (multipart/form-data)

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | ✅ Yes | Document file (.txt or .md, max 50MB) |
| `data_source` | String | ✅ Yes | Dataset name (created if doesn't exist) |
| `title` | String | No | Document title (defaults to filename) |
| `metadata` | JSON String | No | Document metadata |
| `process_async` | Boolean | No | Background processing (default: true) |

---

## Example Usage

### 1. Basic Usage - Create New Dataset

```bash
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@university_admission.md" \
  -F "data_source=dhaka_university" \
  -F "title=Dhaka University Admission Guide"
```

**What Happens:**
1. Creates `expr/dhaka_university/` directory
2. Creates `datasets/dhaka_university/raw/` directory
3. Adds `dhaka_university` to `expr/subgraph_registry.json`
4. Saves document to `corpus.jsonl`
5. Indexes with **Production Pipeline** (table-aware)
6. Returns `job_id` for tracking

---

### 2. With Metadata

```bash
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@medical_info.md" \
  -F "data_source=medical_kb" \
  -F "title=Medical Encyclopedia" \
  -F 'metadata={"category":"healthcare","tags":["medicine","health"],"author":"Dr. Smith"}'
```

---

### 3. Multiple Documents to Same Dataset

```bash
# First document - creates dataset
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@doc1.md" \
  -F "data_source=tech_kb" \
  -F "title=Introduction to AI"

# Second document - appends to existing dataset
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@doc2.md" \
  -F "data_source=tech_kb" \
  -F "title=Machine Learning Basics"

# Third document - continues building the graph
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@doc3.md" \
  -F "data_source=tech_kb" \
  -F "title=Deep Learning Guide"
```

---

## Response Format

### Success Response

```json
{
  "success": true,
  "message": "Document queued for indexing in dataset 'dhaka_university' (production pipeline)",
  "dataset_name": "dhaka_university",
  "document_id": "doc-abc123",
  "job_id": "job-xyz789",
  "filename": "university_admission.md",
  "title": "Dhaka University Admission Guide",
  "content_preview": "Dhaka University is one of the premier...",
  "content_length": 15234,
  "status": "pending",
  "metadata": {
    "category": "education",
    "tags": ["admission", "university"]
  },
  "upload_date": "2025-01-23T15:30:00",
  "pipeline_mode": "production"
}
```

---

## Track Processing Status

Use the `job_id` to track processing status:

```bash
curl "http://localhost:8001/status/job-xyz789"
```

**Response:**
```json
{
  "job_id": "job-xyz789",
  "document_id": "doc-abc123",
  "dataset": "dhaka_university",
  "status": "completed",
  "progress": 1.0,
  "stage": "completed",
  "started_at": "2025-01-23T15:30:05",
  "completed_at": "2025-01-23T15:32:15",
  "stats": {
    "entities": 127,
    "relations": 84,
    "chunks": 8
  }
}
```

---

## Query the Indexed Data

After indexing completes, query using unified endpoint:

```bash
curl -X POST "http://localhost:8001/api/unified/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the admission requirements for Dhaka University?",
    "top_k": 5
  }'
```

The unified mode will automatically route to the `dhaka_university` subgraph!

---

## Directory Structure Created

For `data_source=dhaka_university`, the following structure is created:

```
BiG-RAG/
├── expr/
│   ├── subgraph_registry.json          # Updated with new dataset
│   └── dhaka_university/               # Graph storage (auto-created)
│       ├── graph_chunk_entity_relation.graphml
│       ├── kv_store_full_docs.json
│       ├── kv_store_text_chunks.json
│       ├── kv_store_llm_response_cache.json
│       ├── vdb_entities.json
│       ├── vdb_relations.json
│       └── vdb_chunks.json
│
└── datasets/
    └── dhaka_university/               # Corpus storage (auto-created)
        └── raw/
            └── corpus.jsonl            # Appended with new documents
```

---

## Subgraph Registry Entry

When a new dataset is created, it's automatically added to `expr/subgraph_registry.json`:

```json
{
  "subgraphs": {
    "dhaka_university": {
      "path": "expr/dhaka_university",
      "description": "Auto-created dataset: dhaka_university",
      "aliases": ["dhaka_university"],
      "topics": ["dhaka_university", "auto-created"],
      "enabled": true,
      "created_at": "2025-01-23T15:30:00",
      "auto_created": true
    }
  }
}
```

**Note:** The unified executor automatically reloads the registry after adding a new dataset.

---

## Error Handling

### Missing API Key

**Request:**
```bash
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@doc.md" \
  -F "data_source=test_kb"
```

**Response (if OPENAI_API_KEY not in .env):**
```json
{
  "detail": "[Production Pipeline] OPENAI_API_KEY not found in environment variables. Production pipeline requires OpenAI API key. Please set OPENAI_API_KEY in your .env file or environment."
}
```

**Solution:** Add `OPENAI_API_KEY=sk-...` to your `.env` file and restart server.

---

### Server Not in Unified Mode

**Response:**
```json
{
  "detail": "This endpoint requires unified mode. Start server with --unified flag."
}
```

**Solution:** Restart server with `python server.py --unified`

---

### Invalid File Type

**Response:**
```json
{
  "detail": "Only .txt and .md files are supported"
}
```

**Solution:** Convert file to .txt or .md format.

---

## Pre-Create Dataset (Optional)

If you want to create a dataset with custom metadata before indexing:

```bash
curl -X POST "http://localhost:8001/datasets/create" \
  -F "dataset_name=medical_kb" \
  -F "description=Comprehensive medical knowledge base" \
  -F "topics=medicine,health,medical,healthcare,diagnosis" \
  -F "aliases=medical,med,health,healthcare"
```

Then index documents:

```bash
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@medical_doc.md" \
  -F "data_source=medical_kb"
```

---

## Production Workflow

### Step-by-Step Production Deployment

**1. Start Server**
```bash
cd backend
python server.py --unified
```

**2. Index Initial Documents**
```bash
# University 1
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@dhaka_univ.md" \
  -F "data_source=dhaka_university"

# University 2
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@buet.md" \
  -F "data_source=buet"

# University 3
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@chittagong_univ.md" \
  -F "data_source=chittagong_university"
```

**3. Add More Documents Incrementally**
```bash
# Add to existing dataset
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@dhaka_univ_faq.md" \
  -F "data_source=dhaka_university"
```

**4. Query Across All Datasets**
```bash
curl -X POST "http://localhost:8001/api/unified/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Which universities offer CSE programs?", "top_k": 10}'
```

The unified mode automatically searches across **all** indexed datasets!

---

## Comparison: Old vs New

### Old Approach (Pre-defined Datasets)
```bash
# Step 1: Manually create directories
mkdir -p expr/my_dataset
mkdir -p datasets/my_dataset/raw

# Step 2: Manually create corpus.jsonl
echo '{"id":"doc1","contents":"...","title":"..."}' > datasets/my_dataset/raw/corpus.jsonl

# Step 3: Manually add to subgraph_registry.json
# (edit JSON file manually)

# Step 4: Build graph
python script_build.py --data_source my_dataset

# Step 5: Restart server with new dataset
python server.py --unified --prewarm my_dataset
```

### New Approach (Dynamic Creation) ✅
```bash
# One command - everything done automatically!
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@document.md" \
  -F "data_source=my_dataset"

# No restart needed - dataset immediately available for queries!
```

---

## Benefits for Production

✅ **No Manual Setup** - No need to pre-create directories or edit JSON files
✅ **No Server Restart** - New datasets available immediately
✅ **Scalable** - Add unlimited datasets on-demand
✅ **API-Driven** - Perfect for SaaS platforms
✅ **Incremental** - Build knowledge base document by document
✅ **Production Pipeline** - Higher accuracy (table-aware, 95%+ validation)
✅ **Fail-Fast** - Clear errors if API key missing (no silent degradation)

---

## Use Cases

### 1. Multi-Tenant SaaS
Each customer gets their own dataset:
```bash
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@customer_data.md" \
  -F "data_source=customer_${customer_id}"
```

### 2. Educational Platform
Each institution gets a dataset:
```bash
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@school_info.md" \
  -F "data_source=school_${school_id}"
```

### 3. Document Management System
Organize by category/department:
```bash
curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@hr_policy.md" \
  -F "data_source=hr_documents"

curl -X POST "http://localhost:8001/datasets/create-and-index" \
  -F "file=@tech_spec.md" \
  -F "data_source=engineering_docs"
```

---

## Testing Checklist

- [ ] Server starts in unified mode (`--unified` flag)
- [ ] OPENAI_API_KEY set in `.env`
- [ ] Endpoint accessible at `/datasets/create-and-index`
- [ ] New dataset creates directories
- [ ] Registry updated automatically
- [ ] Corpus.jsonl created and appended
- [ ] Production pipeline runs (check logs for "Production Pipeline")
- [ ] Job tracking works (`/status/{job_id}`)
- [ ] Unified queries route to new dataset
- [ ] Multiple documents to same dataset work
- [ ] Error on missing API key
- [ ] Error on wrong file type

---

## Troubleshooting

### Dataset Not Appearing in Queries

**Check:**
```bash
curl "http://localhost:8001/api/unified/subgraphs"
```

If dataset missing, manually reload registry:
```bash
curl -X POST "http://localhost:8001/api/unified/registry/reload"
```

---

### Production Pipeline Falling Back

**Issue:** Logs show "No OpenAI API key found"

**Solution:**
1. Check `.env` file has `OPENAI_API_KEY=...`
2. Restart server to reload environment variables
3. Verify with: `echo $OPENAI_API_KEY` (Linux) or `echo %OPENAI_API_KEY%` (Windows)

---

### Files Created in Wrong Location

**Check working directory:**
- Graph files should be in: `D:\BiG-RAG\expr\{dataset_name}\`
- Corpus should be in: `D:\BiG-RAG\datasets\{dataset_name}\raw\`

If wrong location, check server start command (should be run from `backend/` directory).

---

## API Documentation

After server starts, view interactive API docs:

**Swagger UI:** http://localhost:8001/docs
**ReDoc:** http://localhost:8001/redoc

Search for "Dataset Management (Production)" to see all endpoints.

---

## Security Considerations

1. **API Key Protection:** Never commit `.env` to version control
2. **Input Validation:** Endpoint validates file size (max 50MB) and type (.txt/.md only)
3. **Dataset Naming:** Sanitize dataset names in production (avoid special characters)
4. **Rate Limiting:** Consider adding rate limits for production deployments

---

## Performance

- **First Document:** ~2-4 minutes (creates dataset + indexes)
- **Subsequent Documents:** ~2-3 minutes (incremental indexing)
- **Dataset Creation Only:** <1 second

**Production Pipeline Cost:** ~$0.16-0.40 per document (OpenAI API)

---

## Next Steps

1. Test with sample document
2. Monitor job processing logs
3. Query indexed data via unified endpoint
4. Integrate into your application

**Need Help?** Check logs in `logs/backend/api.log` or raise an issue on GitHub.
