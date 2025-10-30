# BiG-RAG API Enhancement Plan
## Complete Document Management System for RAG Endpoints

**Version:** 1.0
**Created:** 2025-10-30
**Status:** Ready for Implementation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Knowledge Graph Architecture](#knowledge-graph-architecture)
4. [Complete Data Pipeline Flow](#complete-data-pipeline-flow)
5. [Proposed API Endpoints](#proposed-api-endpoints)
6. [Supporting Infrastructure](#supporting-infrastructure)
7. [File Structure & Organization](#file-structure--organization)
8. [Implementation Phases](#implementation-phases)
9. [Testing Strategy](#testing-strategy)
10. [Error Handling & Edge Cases](#error-handling--edge-cases)

---

## 1. Executive Summary

### Objective
Transform `script_api.py` into a production-ready RAG system API with complete document lifecycle management, supporting both real-time and batch operations with full async job tracking.

### Key Enhancements
- ✅ **Markdown & Text Support**: Process `.md` and `.txt` files
- ✅ **Async Job Processing**: Background document processing with status tracking
- ✅ **Document Registry**: Centralized metadata management
- ✅ **Batch Operations**: Multi-file upload and deletion
- ✅ **Complete CRUD**: Create, Read, Update, Delete for documents
- ✅ **KG Statistics**: Real-time graph analytics
- ✅ **Production-Ready**: Comprehensive error handling, validation, logging

### Scope
**Focus:** API endpoints and document management (no UI, no RL training)
**Testing:** Swagger UI (`/docs`) for all endpoints
**Deployment:** Single-server FastAPI application

---

## 2. Current State Analysis

### Existing Functionality

**File:** `script_api.py` (1042 lines)

#### Current Endpoints
| Endpoint | Method | Status | Purpose |
|----------|--------|--------|---------|
| `/upload` | POST | ✅ Working | Upload single `.txt` file |
| `/rebuild` | POST | ✅ Working | Rebuild knowledge graph |
| `/ask` | POST | ✅ Working | Q&A with retrieval |
| `/search` | POST | ✅ Working | Batch retrieval |
| `/chat/completions` | POST | ✅ Working | OpenAI-compatible chat |
| `/health` | GET | ✅ Working | Health check |

#### Current Upload Flow
```python
# script_api.py: Lines 657-730
/upload → validate .txt → read content → generate doc_id (MD5) →
  add_document_to_corpus() → rebuild_knowledge_graph_incremental() → response
```

**Helper Functions:**
```python
def compute_doc_id(content: str, prefix: str = "upload") -> str:
    """Generate unique ID from content hash"""
    # Line 531-534

async def add_document_to_corpus(data_source, doc_id, content, title):
    """Add document to corpus.jsonl"""
    # Line 537-558
    # Appends to: datasets/{data_source}/raw/corpus.jsonl

async def rebuild_knowledge_graph_incremental(data_source, new_contents):
    """Incrementally update KG using rag.ainsert()"""
    # Line 561-593
    # Calls: await rag.ainsert(batch)
```

### BiGRAG Core Integration

**File:** `bigrag/bigrag.py`

```python
class BiGRAG:
    async def ainsert(self, docs: list[dict]) -> None:
        """
        Main insertion pipeline

        Input: [{"content": str, "title": str, "metadata": dict}]

        Steps:
        1. Generate MD5 hash IDs
        2. Filter already-processed docs (via full_docs storage)
        3. Chunk documents (1200 tokens, 100 overlap)
        4. Extract entities (GPT-4o-mini with gleaning)
        5. Merge duplicate entities
        6. Build bipartite graph
        7. Generate embeddings (OpenAI text-embedding-3-large)
        8. Create FAISS indices
        9. Save metadata to JSON files
        """
```

**Key Insight:** `rag.ainsert()` uses **MD5 content hashing** to automatically skip duplicates!

---

## 3. Knowledge Graph Architecture

### Critical Understanding: One KG Per Dataset

**Architecture:**
```
BiG-RAG System
│
├── Dataset A (e.g., "medical_qa")
│   ├── datasets/medical_qa/raw/corpus.jsonl    ← All source documents
│   └── expr/medical_qa/                         ← Single unified KG
│       ├── kv_store_entities.json
│       ├── kv_store_bipartite_edges.json
│       ├── index_entity.bin
│       └── ...
│
├── Dataset B (e.g., "legal_docs")
│   ├── datasets/legal_docs/raw/corpus.jsonl
│   └── expr/legal_docs/                         ← Separate KG
│       └── ...
│
└── Dataset C (user uploads via API)
    ├── datasets/uploaded_docs/raw/corpus.jsonl
    └── expr/uploaded_docs/                      ← Unified KG for all uploads
        └── ...
```

**Key Points:**
- **One dataset = One knowledge graph**
- All documents uploaded to the same `data_source` → **same KG**
- Different `data_source` values → **separate KGs**
- Documents are **incrementally added** to existing KG
- Duplicate detection via MD5 hash (automatic in `rag.ainsert()`)

**Default for API Uploads:**
```python
# Option 1: Single shared KG for all user uploads
data_source = "user_uploads"  # All API uploads go here

# Option 2: User-specified datasets (allow multiple KGs)
data_source = request.data_source or "user_uploads"
```

**Recommendation:** Use **single shared KG** (`user_uploads`) for simplicity, with metadata tagging for organization.

---

## 4. Complete Data Pipeline Flow

### Upload → Indexing → Retrieval Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: Document Upload (API)                                    │
├──────────────────────────────────────────────────────────────────┤
│ POST /upload                                                      │
│   • File: document.md or document.txt                            │
│   • Title: "My Document"                                         │
│   • Metadata: {"category": "research", "author": "John"}         │
│                                                                   │
│ Validation:                                                       │
│   ✓ File type (.txt, .md)                                        │
│   ✓ File size (< 50 MB)                                          │
│   ✓ Content not empty                                            │
│   ✓ UTF-8 encoding                                               │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: Markdown Processing (if .md)                             │
├──────────────────────────────────────────────────────────────────┤
│ import markdown                                                   │
│ from bs4 import BeautifulSoup                                    │
│                                                                   │
│ html = markdown.markdown(md_content)                             │
│ soup = BeautifulSoup(html, 'html.parser')                        │
│ plain_text = soup.get_text()                                     │
│                                                                   │
│ Preserves: Headings, lists, emphasis                            │
│ Removes: Syntax markers (**, ##, etc.)                           │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: Document Registry Update                                 │
├──────────────────────────────────────────────────────────────────┤
│ File: expr/{data_source}/documents_registry.json                │
│                                                                   │
│ {                                                                 │
│   "upload-a1b2c3d4e5f6": {                                       │
│     "document_id": "upload-a1b2c3d4e5f6",                        │
│     "filename": "document.md",                                   │
│     "title": "My Document",                                      │
│     "content_length": 15420,                                     │
│     "upload_date": "2025-10-30T10:30:00",                        │
│     "status": "pending",                                         │
│     "metadata": {"category": "research"},                        │
│     "job_id": "job-xyz789"                                       │
│   }                                                               │
│ }                                                                 │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: Add to Corpus (JSONL)                                    │
├──────────────────────────────────────────────────────────────────┤
│ File: datasets/{data_source}/raw/corpus.jsonl                   │
│                                                                   │
│ APPEND (not overwrite):                                          │
│ {"id":"upload-a1b2c3d4e5f6","contents":"...plain text...",      │
│  "title":"My Document","upload_date":"2025-10-30T10:30:00",     │
│  "source":"upload","metadata":{...}}                             │
│                                                                   │
│ Format: JSON Lines (one object per line, newline-delimited)     │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5: Background Job Creation                                  │
├──────────────────────────────────────────────────────────────────┤
│ class ProcessingJob:                                              │
│   job_id: str = "job-xyz789"                                     │
│   document_id: str = "upload-a1b2c3d4e5f6"                       │
│   status: JobStatus = "pending"                                  │
│   progress: float = 0.0                                          │
│   stage: str = "queued"                                          │
│   started_at: datetime                                           │
│   completed_at: Optional[datetime] = None                        │
│   error: Optional[str] = None                                    │
│                                                                   │
│ processing_jobs[job_id] = job                                    │
│ BackgroundTasks.add_task(process_document_background, job_id)   │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 6: Knowledge Graph Construction (BiGRAG.ainsert)           │
├──────────────────────────────────────────────────────────────────┤
│ await rag.ainsert([{"content": plain_text, "title": title}])    │
│                                                                   │
│ Sub-steps (bigrag/bigrag.py):                                   │
│   1. MD5 Hash Check (skip if exists in full_docs)               │
│   2. Chunking (1200 tokens, 100 overlap)                        │
│      └─> Stage: "chunking" (progress: 0.15)                     │
│   3. Entity Extraction (GPT-4o-mini + gleaning)                 │
│      └─> Stage: "extracting" (progress: 0.40)                   │
│   4. Relation Extraction                                         │
│      └─> Stage: "extracting_relations" (progress: 0.60)         │
│   5. Graph Merging & Updates                                     │
│      └─> Stage: "graph_building" (progress: 0.75)               │
│   6. Embedding Generation (OpenAI)                               │
│      └─> Stage: "embedding" (progress: 0.85)                    │
│   7. FAISS Indexing                                              │
│      └─> Stage: "indexing" (progress: 0.95)                     │
│   8. Finalization                                                │
│      └─> Stage: "completed" (progress: 1.0)                     │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 7: Knowledge Graph Files Created/Updated                    │
├──────────────────────────────────────────────────────────────────┤
│ expr/{data_source}/                                              │
│ ├── kv_store_entities.json          (entities metadata)         │
│ ├── kv_store_bipartite_edges.json   (relations metadata)        │
│ ├── kv_store_text_chunks.json       (chunks metadata)           │
│ ├── index_entity.bin                 (FAISS entity index)        │
│ ├── index_bipartite_edge.bin         (FAISS edge index)          │
│ ├── index.bin                         (FAISS chunk index)         │
│ ├── corpus.npy                        (chunk embeddings)          │
│ ├── corpus_entity.npy                 (entity embeddings)         │
│ ├── corpus_bipartite_edge.npy        (edge embeddings)           │
│ └── graph_chunk_entity_relation.graphml (NetworkX graph)        │
│                                                                   │
│ Notes:                                                            │
│ • Files are UPDATED incrementally (not replaced)                │
│ • Embeddings are ADDED to existing indices                      │
│ • Graph is MERGED with existing structure                       │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 8: Registry Update (Success/Failure)                        │
├──────────────────────────────────────────────────────────────────┤
│ Update documents_registry.json:                                  │
│                                                                   │
│ Success:                                                          │
│   "status": "indexed",                                           │
│   "indexed_at": "2025-10-30T10:35:00",                          │
│   "stats": {                                                      │
│     "chunks_created": 12,                                        │
│     "entities_extracted": 35,                                    │
│     "edges_created": 28                                          │
│   }                                                               │
│                                                                   │
│ Failure:                                                          │
│   "status": "failed",                                            │
│   "error": "OpenAI API rate limit exceeded"                     │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 9: Ready for Retrieval                                      │
├──────────────────────────────────────────────────────────────────┤
│ GET /ask?question=What is in my document?                        │
│                                                                   │
│ Flow:                                                             │
│ 1. Embed query (OpenAI)                                          │
│ 2. Search FAISS indices (entities + edges)                      │
│ 3. Traverse bipartite graph                                     │
│ 4. Retrieve connected text chunks                               │
│ 5. Rank by relevance                                             │
│ 6. Return context                                                │
└──────────────────────────────────────────────────────────────────┘
```

### Time Estimates (per document)

| Stage | Time (small doc) | Time (large doc) | Bottleneck |
|-------|------------------|------------------|------------|
| Upload & Validation | <1s | <2s | File I/O |
| MD Processing | <1s | <2s | CPU |
| Chunking | <1s | 5s | Tokenization |
| Entity Extraction | 10-20s | 60-120s | **GPT-4o-mini API** |
| Embedding | 3-5s | 10-15s | **OpenAI API** |
| FAISS Indexing | <1s | 2-3s | CPU |
| **Total** | **15-30s** | **80-150s** | **API calls** |

**Note:** Background processing essential for large documents!

---

## 5. Proposed API Endpoints

### 5.1 Document Upload & Management

#### **1. Enhanced Upload (Single Document)**

**Endpoint:** `POST /upload`

**Changes from Current:**
- ✅ Add Markdown (`.md`) support
- ✅ Return `job_id` for status tracking
- ✅ Optional `process_async` parameter (default: true)
- ✅ Enhanced metadata support

**Request:**
```python
POST /upload
Content-Type: multipart/form-data

file: UploadFile              # .txt or .md file
title: Optional[str]          # Document title (defaults to filename)
data_source: Optional[str]    # Dataset name (defaults to "user_uploads")
process_async: bool = True    # Process in background
metadata: Optional[str]       # JSON string: {"category": "research", "tags": [...]}
```

**Response:**
```json
{
  "success": true,
  "message": "Document queued for processing",
  "document_id": "upload-a1b2c3d4e5f6",
  "job_id": "job-xyz789",
  "filename": "document.md",
  "title": "My Document",
  "content_preview": "This is the first 200 characters of the document...",
  "content_length": 15420,
  "dataset": "user_uploads",
  "status": "pending",
  "metadata": {"category": "research"},
  "upload_date": "2025-10-30T10:30:00Z"
}
```

**Implementation:**
```python
@app.post("/upload", response_model=UploadResponse, tags=["Document Management"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    data_source: str = Form(None),
    process_async: bool = Form(True),
    metadata: str = Form(None)  # JSON string
):
    """
    Upload a document (.txt or .md) and add to knowledge graph

    - **file**: Text or Markdown file to upload
    - **title**: Optional title (defaults to filename)
    - **data_source**: Dataset name (defaults to "user_uploads")
    - **process_async**: Process in background (recommended for large files)
    - **metadata**: Optional JSON metadata

    Returns job_id for tracking processing status
    """
    # Validate file extension
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(400, "Only .txt and .md files supported")

    # Read content
    content_bytes = await file.read()

    try:
        content_text = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 encoded")

    # Process Markdown if .md
    if file.filename.endswith('.md'):
        content_text = process_markdown(content_text)

    # Validate content
    if not content_text.strip():
        raise HTTPException(400, "File is empty")

    # Parse metadata
    doc_metadata = json.loads(metadata) if metadata else {}

    # Generate IDs
    doc_id = compute_doc_id(content_text, prefix="upload")
    job_id = f"job-{compute_doc_id(doc_id + str(datetime.now()), prefix='')}"

    # Determine dataset
    target_dataset = data_source or "user_uploads"
    doc_title = title or file.filename

    # Add to corpus.jsonl
    await add_document_to_corpus(
        data_source=target_dataset,
        doc_id=doc_id,
        content=content_text,
        title=doc_title,
        metadata=doc_metadata
    )

    # Add to document registry
    await registry.add_document(
        document_id=doc_id,
        filename=file.filename,
        title=doc_title,
        content_length=len(content_text),
        dataset=target_dataset,
        metadata=doc_metadata,
        job_id=job_id,
        status="pending"
    )

    # Create job
    job = ProcessingJob(
        job_id=job_id,
        document_id=doc_id,
        status="pending",
        progress=0.0,
        stage="queued"
    )
    processing_jobs[job_id] = job

    # Process
    if process_async:
        background_tasks.add_task(
            process_document_background,
            job_id=job_id,
            content=content_text,
            title=doc_title,
            dataset=target_dataset
        )
    else:
        # Synchronous processing (blocks response)
        await process_document_background(job_id, content_text, doc_title, target_dataset)

    return UploadResponse(
        success=True,
        message="Document queued for processing" if process_async else "Document processed",
        document_id=doc_id,
        job_id=job_id,
        filename=file.filename,
        title=doc_title,
        content_preview=content_text[:200] + "...",
        content_length=len(content_text),
        dataset=target_dataset,
        status=job.status,
        metadata=doc_metadata,
        upload_date=datetime.now().isoformat()
    )
```

---

#### **2. Batch Upload (NEW)**

**Endpoint:** `POST /upload/batch`

**Purpose:** Upload multiple files in one request

**Request:**
```python
POST /upload/batch
Content-Type: multipart/form-data

files: List[UploadFile]       # Multiple .txt or .md files
data_source: Optional[str]    # Dataset for all files
default_metadata: Optional[str] # JSON string (applied to all)
```

**Response:**
```json
{
  "success": true,
  "message": "Batch upload complete",
  "batch_id": "batch-abc123",
  "total_files": 10,
  "accepted_files": 9,
  "rejected_files": ["invalid.pdf"],
  "job_ids": ["job-1", "job-2", ..., "job-9"],
  "documents": [
    {
      "document_id": "upload-xyz1",
      "filename": "doc1.txt",
      "job_id": "job-1",
      "status": "pending"
    },
    ...
  ]
}
```

**Implementation:**
```python
@app.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    data_source: str = Form(None),
    default_metadata: str = Form(None)
):
    """Upload multiple documents at once"""

    batch_id = f"batch-{compute_doc_id(str(datetime.now()), prefix='')}"
    target_dataset = data_source or "user_uploads"
    metadata = json.loads(default_metadata) if default_metadata else {}

    accepted = []
    rejected = []
    job_ids = []

    for file in files:
        # Validate extension
        if not file.filename.endswith(('.txt', '.md')):
            rejected.append(file.filename)
            continue

        # Read content
        content_bytes = await file.read()

        try:
            content_text = content_bytes.decode('utf-8')
        except:
            rejected.append(file.filename)
            continue

        # Process MD if needed
        if file.filename.endswith('.md'):
            content_text = process_markdown(content_text)

        if not content_text.strip():
            rejected.append(file.filename)
            continue

        # Generate IDs
        doc_id = compute_doc_id(content_text, prefix="upload")
        job_id = f"job-{compute_doc_id(doc_id + str(datetime.now()), prefix='')}"

        # Add to corpus & registry
        await add_document_to_corpus(
            target_dataset, doc_id, content_text, file.filename, metadata
        )

        await registry.add_document(
            document_id=doc_id,
            filename=file.filename,
            title=file.filename,
            content_length=len(content_text),
            dataset=target_dataset,
            metadata={**metadata, "batch_id": batch_id},
            job_id=job_id,
            status="pending"
        )

        # Create job
        job = ProcessingJob(
            job_id=job_id,
            document_id=doc_id,
            status="pending",
            batch_id=batch_id
        )
        processing_jobs[job_id] = job

        # Queue processing
        background_tasks.add_task(
            process_document_background,
            job_id, content_text, file.filename, target_dataset
        )

        accepted.append({
            "document_id": doc_id,
            "filename": file.filename,
            "job_id": job_id,
            "status": "pending"
        })
        job_ids.append(job_id)

    # Store batch info
    batch_info[batch_id] = {
        "batch_id": batch_id,
        "total_files": len(files),
        "accepted": len(accepted),
        "rejected": rejected,
        "job_ids": job_ids,
        "created_at": datetime.now().isoformat()
    }

    return BatchUploadResponse(
        success=True,
        message=f"Batch upload complete: {len(accepted)}/{len(files)} files accepted",
        batch_id=batch_id,
        total_files=len(files),
        accepted_files=len(accepted),
        rejected_files=rejected,
        job_ids=job_ids,
        documents=accepted
    )
```

---

### 5.2 Job Status & Monitoring

#### **3. Job Status (NEW)**

**Endpoint:** `GET /status/{job_id}`

**Purpose:** Check processing status for a document

**Response:**
```json
{
  "job_id": "job-xyz789",
  "document_id": "upload-a1b2c3d4e5f6",
  "status": "processing",
  "progress": 0.65,
  "stage": "embedding",
  "started_at": "2025-10-30T10:30:05Z",
  "completed_at": null,
  "error": null,
  "stats": {
    "chunks_created": 12,
    "entities_extracted": 35,
    "edges_created": 28,
    "processing_time_seconds": 45.2
  }
}
```

**Implementation:**
```python
@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get processing status for a job"""

    if job_id not in processing_jobs:
        raise HTTPException(404, f"Job not found: {job_id}")

    job = processing_jobs[job_id]

    return JobStatusResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        stats=job.stats
    )
```

---

#### **4. Batch Status (NEW)**

**Endpoint:** `GET /status/batch/{batch_id}`

**Response:**
```json
{
  "batch_id": "batch-abc123",
  "total_documents": 10,
  "completed": 7,
  "processing": 2,
  "failed": 1,
  "pending": 0,
  "overall_progress": 0.75,
  "documents": [
    {
      "job_id": "job-1",
      "document_id": "upload-xyz1",
      "filename": "doc1.txt",
      "status": "completed",
      "progress": 1.0
    },
    ...
  ]
}
```

---

### 5.3 Document Discovery & Retrieval

#### **5. List Documents (NEW)**

**Endpoint:** `GET /documents`

**Purpose:** List all indexed documents with filtering and pagination

**Query Parameters:**
```python
page: int = 1
page_size: int = 20
search: Optional[str]         # Search in title/metadata
category: Optional[str]       # Filter by metadata.category
tags: Optional[List[str]]     # Filter by metadata.tags
status: Optional[str]         # indexed|failed|processing
date_from: Optional[datetime]
date_to: Optional[datetime]
sort_by: str = "date"         # date|title|size
data_source: Optional[str]    # Filter by dataset
```

**Response:**
```json
{
  "total": 156,
  "page": 1,
  "page_size": 20,
  "total_pages": 8,
  "documents": [
    {
      "document_id": "upload-abc123",
      "title": "Research Paper on AI",
      "filename": "ai_research.md",
      "upload_date": "2025-10-30T10:30:00Z",
      "indexed_date": "2025-10-30T10:32:15Z",
      "content_length": 15420,
      "status": "indexed",
      "dataset": "user_uploads",
      "metadata": {
        "category": "research",
        "tags": ["AI", "ML"]
      },
      "stats": {
        "chunks": 12,
        "entities": 35,
        "edges": 28
      }
    },
    ...
  ]
}
```

**Implementation:**
```python
@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = Query("date", regex="^(date|title|size)$"),
    data_source: Optional[str] = None
):
    """
    List documents with filtering and pagination

    - **page**: Page number (starts at 1)
    - **page_size**: Items per page (max 100)
    - **search**: Search in title/filename
    - **category**: Filter by metadata.category
    - **tags**: Filter by tags (comma-separated)
    - **status**: Filter by status (indexed, failed, processing)
    - **sort_by**: Sort by date, title, or size
    """

    # Parse tags
    tag_list = tags.split(',') if tags else None

    # Get documents from registry
    all_docs = await registry.list_documents(
        search=search,
        category=category,
        tags=tag_list,
        status=status,
        date_from=date_from,
        date_to=date_to,
        data_source=data_source
    )

    # Sort
    if sort_by == "date":
        all_docs.sort(key=lambda d: d.get("upload_date", ""), reverse=True)
    elif sort_by == "title":
        all_docs.sort(key=lambda d: d.get("title", "").lower())
    elif sort_by == "size":
        all_docs.sort(key=lambda d: d.get("content_length", 0), reverse=True)

    # Paginate
    total = len(all_docs)
    total_pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    page_docs = all_docs[start:end]

    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        documents=page_docs
    )
```

---

#### **6. Get Document Details (NEW)**

**Endpoint:** `GET /documents/{document_id}`

**Response:**
```json
{
  "document_id": "upload-abc123",
  "title": "Research Paper on AI",
  "filename": "ai_research.md",
  "content": "Full document content here...",
  "content_length": 15420,
  "upload_date": "2025-10-30T10:30:00Z",
  "indexed_date": "2025-10-30T10:32:15Z",
  "last_modified": "2025-10-30T10:32:15Z",
  "status": "indexed",
  "dataset": "user_uploads",
  "metadata": {
    "category": "research",
    "tags": ["AI", "ML"],
    "author": "John Doe"
  },
  "stats": {
    "chunks": 12,
    "entities": 35,
    "edges": 28,
    "tokens": 3840
  },
  "top_entities": [
    {"name": "Artificial Intelligence", "type": "category", "weight": 95},
    {"name": "Machine Learning", "type": "category", "weight": 87}
  ],
  "related_documents": [
    {"id": "upload-xyz789", "title": "ML Basics", "similarity": 0.83}
  ]
}
```

**Implementation:**
```python
@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_details(document_id: str):
    """Get full details for a specific document"""

    # Get from registry
    doc = await registry.get_document(document_id)

    if not doc:
        raise HTTPException(404, f"Document not found: {document_id}")

    # Get content from corpus
    content = await get_document_content_from_corpus(
        data_source=doc["dataset"],
        document_id=document_id
    )

    # Get stats from KG
    stats = await get_document_stats_from_kg(
        data_source=doc["dataset"],
        document_id=document_id
    )

    # Get top entities
    top_entities = await get_document_entities(
        data_source=doc["dataset"],
        document_id=document_id,
        top_k=10
    )

    # Get related documents (via entity overlap)
    related = await find_related_documents(
        data_source=doc["dataset"],
        document_id=document_id,
        top_k=5
    )

    return DocumentDetailResponse(
        document_id=document_id,
        title=doc["title"],
        filename=doc["filename"],
        content=content,
        content_length=doc["content_length"],
        upload_date=doc["upload_date"],
        indexed_date=doc.get("indexed_date"),
        last_modified=doc.get("last_modified"),
        status=doc["status"],
        dataset=doc["dataset"],
        metadata=doc.get("metadata", {}),
        stats=stats,
        top_entities=top_entities,
        related_documents=related
    )
```

---

### 5.4 Document Deletion

#### **7. Delete Document (NEW)**

**Endpoint:** `DELETE /documents/{document_id}`

**Query Parameters:**
```python
hard_delete: bool = False     # Remove from corpus.jsonl
rebuild_graph: bool = True    # Rebuild KG after deletion
```

**Response:**
```json
{
  "success": true,
  "message": "Document deleted successfully",
  "document_id": "upload-abc123",
  "deleted_at": "2025-10-30T11:00:00Z",
  "deletion_type": "soft",
  "deleted_items": {
    "chunks": 12,
    "entities": 35,
    "edges": 28
  },
  "rebuild_required": true,
  "rebuild_job_id": "job-rebuild-xyz"
}
```

**Implementation:**
```python
@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    background_tasks: BackgroundTasks,
    document_id: str,
    hard_delete: bool = Query(False),
    rebuild_graph: bool = Query(True)
):
    """
    Delete a document from the system

    - **hard_delete**: If true, removes from corpus.jsonl (cannot undo)
    - **rebuild_graph**: If true, rebuilds KG after deletion (recommended)

    Soft delete: Marks as deleted in registry, keeps in corpus
    Hard delete: Removes from corpus.jsonl, rebuilds KG from scratch
    """

    # Get document
    doc = await registry.get_document(document_id)

    if not doc:
        raise HTTPException(404, f"Document not found: {document_id}")

    dataset = doc["dataset"]

    # Count items to be deleted
    deleted_stats = await get_document_stats_from_kg(dataset, document_id)

    if hard_delete:
        # Remove from corpus.jsonl
        await remove_from_corpus(dataset, document_id)

        # Mark in registry
        await registry.delete_document(document_id, hard=True)

        if rebuild_graph:
            # Rebuild entire KG from updated corpus
            rebuild_job_id = f"job-rebuild-{compute_doc_id(str(datetime.now()))}"

            background_tasks.add_task(
                rebuild_entire_graph,
                dataset=dataset,
                job_id=rebuild_job_id
            )

            deletion_type = "hard"
            message = "Document removed from corpus, rebuilding graph..."
        else:
            deletion_type = "hard"
            message = "Document removed from corpus (graph rebuild required)"
            rebuild_job_id = None
    else:
        # Soft delete (mark in registry only)
        await registry.delete_document(document_id, hard=False)
        deletion_type = "soft"
        message = "Document marked as deleted (still in corpus)"
        rebuild_job_id = None

    return DeleteResponse(
        success=True,
        message=message,
        document_id=document_id,
        deleted_at=datetime.now().isoformat(),
        deletion_type=deletion_type,
        deleted_items=deleted_stats,
        rebuild_required=rebuild_graph and hard_delete,
        rebuild_job_id=rebuild_job_id
    )
```

---

#### **8. Batch Delete (NEW)**

**Endpoint:** `DELETE /documents/batch`

**Request Body:**
```json
{
  "document_ids": ["upload-abc1", "upload-abc2", "upload-abc3"],
  "hard_delete": false,
  "rebuild_graph": true
}
```

---

### 5.5 Graph Management & Statistics

#### **9. Graph Statistics (NEW)**

**Endpoint:** `GET /graph/stats`

**Query Parameters:**
```python
data_source: Optional[str] = None  # Specific dataset or all
```

**Response:**
```json
{
  "dataset": "user_uploads",
  "total_documents": 156,
  "indexed_documents": 150,
  "failed_documents": 6,
  "total_chunks": 1872,
  "total_entities": 5420,
  "total_edges": 4180,
  "entity_types": {
    "person": 1240,
    "geo": 890,
    "organization": 1520,
    "event": 680,
    "category": 1090
  },
  "storage_size": {
    "corpus": "12.5 MB",
    "indices": "85.3 MB",
    "metadata": "3.2 MB",
    "total": "101.0 MB"
  },
  "last_updated": "2025-10-30T10:35:12Z",
  "embedding_mode": "openai",
  "graph_density": 0.15,
  "avg_entities_per_doc": 34.7
}
```

**Implementation:**
```python
@app.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_statistics(data_source: Optional[str] = None):
    """Get knowledge graph statistics"""

    target_dataset = data_source or args.data_source

    # Load metadata files
    entities_file = f"expr/{target_dataset}/kv_store_entities.json"
    edges_file = f"expr/{target_dataset}/kv_store_bipartite_edges.json"
    chunks_file = f"expr/{target_dataset}/kv_store_text_chunks.json"

    # Count items
    with open(entities_file) as f:
        entities = json.load(f)

    with open(edges_file) as f:
        edges = json.load(f)

    with open(chunks_file) as f:
        chunks = json.load(f)

    # Entity type distribution
    entity_types = defaultdict(int)
    for entity_id, entity_data in entities.items():
        entity_type = entity_data.get("entity_type", "unknown")
        entity_types[entity_type] += 1

    # Storage sizes
    corpus_size = os.path.getsize(f"datasets/{target_dataset}/raw/corpus.jsonl")

    index_sizes = 0
    for idx_file in ["index_entity.bin", "index_bipartite_edge.bin", "index.bin"]:
        path = f"expr/{target_dataset}/{idx_file}"
        if os.path.exists(path):
            index_sizes += os.path.getsize(path)

    metadata_sizes = sum([
        os.path.getsize(entities_file),
        os.path.getsize(edges_file),
        os.path.getsize(chunks_file)
    ])

    # Document counts
    doc_stats = await registry.get_stats(target_dataset)

    return GraphStatsResponse(
        dataset=target_dataset,
        total_documents=doc_stats["total"],
        indexed_documents=doc_stats["indexed"],
        failed_documents=doc_stats["failed"],
        total_chunks=len(chunks),
        total_entities=len(entities),
        total_edges=len(edges),
        entity_types=dict(entity_types),
        storage_size={
            "corpus": f"{corpus_size / 1024 / 1024:.1f} MB",
            "indices": f"{index_sizes / 1024 / 1024:.1f} MB",
            "metadata": f"{metadata_sizes / 1024 / 1024:.1f} MB",
            "total": f"{(corpus_size + index_sizes + metadata_sizes) / 1024 / 1024:.1f} MB"
        },
        last_updated=datetime.now().isoformat(),
        embedding_mode=embedding_manager.mode,
        graph_density=len(edges) / (len(entities) * len(edges)) if entities and edges else 0,
        avg_entities_per_doc=len(entities) / doc_stats["total"] if doc_stats["total"] > 0 else 0
    )
```

---

#### **10. Clear Graph (NEW)**

**Endpoint:** `POST /graph/clear`

**Request:**
```json
{
  "confirm": "DELETE_ALL",
  "create_backup": true,
  "data_source": "user_uploads"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Knowledge graph cleared successfully",
  "dataset": "user_uploads",
  "backup_location": "backups/user_uploads_2025-10-30_10-35-12.zip",
  "deleted_items": {
    "documents": 156,
    "chunks": 1872,
    "entities": 5420,
    "edges": 4180
  },
  "cleared_at": "2025-10-30T10:35:12Z"
}
```

---

### 5.6 Enhanced Existing Endpoints

#### **11. Enhanced `/health`**

**Current:** Basic status check
**Enhanced:** Add processing queue status, disk space, API health

**Response:**
```json
{
  "status": "healthy",
  "dataset": "user_uploads",
  "entities_count": 5420,
  "edges_count": 4180,
  "chunks_count": 1872,
  "embedding_mode": "openai",
  "available_providers": ["openai", "anthropic"],
  "default_provider": "openai",
  "processing_queue": {
    "pending": 3,
    "processing": 2,
    "completed_today": 45,
    "failed_today": 1
  },
  "disk_space": {
    "total_gb": 500,
    "used_gb": 120,
    "available_gb": 380,
    "usage_percent": 24.0
  },
  "api_keys": {
    "openai": "configured",
    "anthropic": "not_configured"
  },
  "uptime_seconds": 86400
}
```

---

## 6. Supporting Infrastructure

### 6.1 Job Queue System

**File:** `api/jobs.py`

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingStage(str, Enum):
    QUEUED = "queued"
    CHUNKING = "chunking"
    EXTRACTING = "extracting_entities"
    EXTRACTING_RELATIONS = "extracting_relations"
    GRAPH_BUILDING = "graph_building"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"

@dataclass
class ProcessingJob:
    """Represents a document processing job"""
    job_id: str
    document_id: str
    dataset: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    stage: ProcessingStage = ProcessingStage.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    batch_id: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs):
        """Update job fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Global job storage (in-memory)
# For production, use Redis or database
processing_jobs: Dict[str, ProcessingJob] = {}
batch_info: Dict[str, Dict] = {}

# Progress mapping by stage
STAGE_PROGRESS = {
    ProcessingStage.QUEUED: 0.0,
    ProcessingStage.CHUNKING: 0.15,
    ProcessingStage.EXTRACTING: 0.40,
    ProcessingStage.EXTRACTING_RELATIONS: 0.60,
    ProcessingStage.GRAPH_BUILDING: 0.75,
    ProcessingStage.EMBEDDING: 0.85,
    ProcessingStage.INDEXING: 0.95,
    ProcessingStage.FINALIZING: 0.98,
    ProcessingStage.COMPLETED: 1.0
}

async def process_document_background(
    job_id: str,
    content: str,
    title: str,
    dataset: str
):
    """
    Background task for document processing

    Updates job status throughout processing
    """
    job = processing_jobs[job_id]

    try:
        # Start processing
        job.update(
            status=JobStatus.PROCESSING,
            started_at=datetime.now(),
            stage=ProcessingStage.CHUNKING,
            progress=STAGE_PROGRESS[ProcessingStage.CHUNKING]
        )

        logger.info(f"[Job {job_id}] Starting processing for {title}")

        # Stage 1: Chunking (handled internally by rag.ainsert)
        # No explicit action needed

        # Stage 2: Call BiGRAG ainsert
        # This handles: chunking, extraction, graph building, embedding, indexing

        # Update to extraction stage
        job.update(
            stage=ProcessingStage.EXTRACTING,
            progress=STAGE_PROGRESS[ProcessingStage.EXTRACTING]
        )

        # Process document
        await rag.ainsert([{
            "content": content,
            "title": title
        }])

        # Since ainsert doesn't provide progress callbacks, we estimate
        # Update to final stages sequentially

        job.update(
            stage=ProcessingStage.GRAPH_BUILDING,
            progress=STAGE_PROGRESS[ProcessingStage.GRAPH_BUILDING]
        )

        await asyncio.sleep(0.1)  # Small delay for status updates

        job.update(
            stage=ProcessingStage.EMBEDDING,
            progress=STAGE_PROGRESS[ProcessingStage.EMBEDDING]
        )

        await asyncio.sleep(0.1)

        job.update(
            stage=ProcessingStage.INDEXING,
            progress=STAGE_PROGRESS[ProcessingStage.INDEXING]
        )

        await asyncio.sleep(0.1)

        # Get stats from KG
        stats = await get_document_stats_from_kg(dataset, job.document_id)

        # Complete
        job.update(
            status=JobStatus.COMPLETED,
            stage=ProcessingStage.COMPLETED,
            progress=1.0,
            completed_at=datetime.now(),
            stats=stats
        )

        # Update registry
        await registry.update_document(
            job.document_id,
            status="indexed",
            indexed_date=datetime.now().isoformat(),
            stats=stats
        )

        logger.info(f"[Job {job_id}] Completed successfully")

    except Exception as e:
        # Handle failure
        logger.error(f"[Job {job_id}] Failed: {str(e)}")

        job.update(
            status=JobStatus.FAILED,
            completed_at=datetime.now(),
            error=str(e)
        )

        # Update registry
        await registry.update_document(
            job.document_id,
            status="failed",
            error=str(e)
        )
```

---

### 6.2 Document Registry

**File:** `api/registry.py`

```python
import json
import os
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path

class DocumentRegistry:
    """
    Manages document metadata and lifecycle

    Stores metadata in: expr/{data_source}/documents_registry.json
    """

    def __init__(self, working_dir: str = "expr"):
        self.working_dir = working_dir

    def _get_registry_path(self, dataset: str) -> Path:
        """Get registry file path for dataset"""
        return Path(self.working_dir) / dataset / "documents_registry.json"

    async def _load_registry(self, dataset: str) -> Dict:
        """Load registry from disk"""
        path = self._get_registry_path(dataset)

        if not path.exists():
            return {}

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def _save_registry(self, dataset: str, data: Dict):
        """Save registry to disk"""
        path = self._get_registry_path(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def add_document(
        self,
        document_id: str,
        filename: str,
        title: str,
        content_length: int,
        dataset: str,
        metadata: Dict,
        job_id: str,
        status: str = "pending"
    ):
        """Add new document to registry"""

        registry = await self._load_registry(dataset)

        registry[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "title": title,
            "content_length": content_length,
            "upload_date": datetime.now().isoformat(),
            "indexed_date": None,
            "last_modified": datetime.now().isoformat(),
            "status": status,
            "dataset": dataset,
            "metadata": metadata,
            "job_id": job_id,
            "stats": {}
        }

        await self._save_registry(dataset, registry)

    async def get_document(self, document_id: str, dataset: Optional[str] = None) -> Optional[Dict]:
        """Get document by ID"""

        if dataset:
            registry = await self._load_registry(dataset)
            return registry.get(document_id)
        else:
            # Search across all datasets
            for dataset_dir in Path(self.working_dir).iterdir():
                if dataset_dir.is_dir():
                    registry = await self._load_registry(dataset_dir.name)
                    if document_id in registry:
                        return registry[document_id]
            return None

    async def update_document(self, document_id: str, **kwargs):
        """Update document fields"""

        # Find document's dataset
        doc = await self.get_document(document_id)

        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        dataset = doc["dataset"]
        registry = await self._load_registry(dataset)

        # Update fields
        for key, value in kwargs.items():
            if key in registry[document_id]:
                registry[document_id][key] = value

        registry[document_id]["last_modified"] = datetime.now().isoformat()

        await self._save_registry(dataset, registry)

    async def delete_document(self, document_id: str, hard: bool = False):
        """Delete or mark document as deleted"""

        doc = await self.get_document(document_id)

        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        dataset = doc["dataset"]
        registry = await self._load_registry(dataset)

        if hard:
            # Remove from registry
            del registry[document_id]
        else:
            # Mark as deleted
            registry[document_id]["status"] = "deleted"
            registry[document_id]["deleted_at"] = datetime.now().isoformat()

        await self._save_registry(dataset, registry)

    async def list_documents(
        self,
        dataset: Optional[str] = None,
        search: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict]:
        """List documents with filtering"""

        all_docs = []

        if dataset:
            datasets = [dataset]
        else:
            # All datasets
            datasets = [d.name for d in Path(self.working_dir).iterdir() if d.is_dir()]

        for ds in datasets:
            registry = await self._load_registry(ds)

            for doc_id, doc in registry.items():
                # Apply filters
                if status and doc.get("status") != status:
                    continue

                if search and search.lower() not in doc.get("title", "").lower():
                    continue

                if category and doc.get("metadata", {}).get("category") != category:
                    continue

                if tags:
                    doc_tags = doc.get("metadata", {}).get("tags", [])
                    if not any(tag in doc_tags for tag in tags):
                        continue

                if date_from and doc.get("upload_date", "") < date_from:
                    continue

                if date_to and doc.get("upload_date", "") > date_to:
                    continue

                all_docs.append(doc)

        return all_docs

    async def get_stats(self, dataset: str) -> Dict:
        """Get document statistics for dataset"""

        registry = await self._load_registry(dataset)

        total = len(registry)
        indexed = sum(1 for d in registry.values() if d.get("status") == "indexed")
        failed = sum(1 for d in registry.values() if d.get("status") == "failed")
        processing = sum(1 for d in registry.values() if d.get("status") == "processing")

        return {
            "total": total,
            "indexed": indexed,
            "failed": failed,
            "processing": processing,
            "pending": total - indexed - failed - processing
        }

# Global registry instance
registry = DocumentRegistry()
```

---

### 6.3 Markdown Processor

**File:** `api/utils.py`

```python
import markdown
from bs4 import BeautifulSoup
import re

def process_markdown(md_content: str) -> str:
    """
    Convert Markdown to plain text for indexing

    Preserves structure while removing syntax

    Args:
        md_content: Markdown text

    Returns:
        Plain text suitable for knowledge graph construction
    """
    # Convert MD → HTML
    html = markdown.markdown(
        md_content,
        extensions=[
            'extra',           # Tables, fenced code, etc.
            'nl2br',           # Newline to <br>
            'sane_lists'       # Better list handling
        ]
    )

    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Extract text
    text = soup.get_text(separator='\n')

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = text.strip()

    return text

def validate_file_upload(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 50
) -> tuple[bool, Optional[str]]:
    """
    Validate uploaded file

    Returns:
        (is_valid, error_message)
    """
    # Check size
    size_mb = len(file_bytes) / 1024 / 1024

    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f} MB (max {max_size_mb} MB)"

    # Check extension
    if not filename.endswith(('.txt', '.md')):
        return False, "Only .txt and .md files are supported"

    # Check UTF-8 encoding
    try:
        file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return False, "File must be UTF-8 encoded"

    # Check not empty
    if len(file_bytes.strip()) == 0:
        return False, "File is empty"

    return True, None
```

---

### 6.4 Helper Functions for KG Queries

**File:** `api/kg_utils.py`

```python
import json
from typing import Dict, List, Optional

async def get_document_content_from_corpus(
    data_source: str,
    document_id: str
) -> Optional[str]:
    """
    Retrieve document content from corpus.jsonl

    Args:
        data_source: Dataset name
        document_id: Document ID

    Returns:
        Document content or None if not found
    """
    corpus_file = f"datasets/{data_source}/raw/corpus.jsonl"

    if not os.path.exists(corpus_file):
        return None

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("id") == document_id:
                return doc.get("contents", "")

    return None

async def get_document_stats_from_kg(
    data_source: str,
    document_id: str
) -> Dict:
    """
    Get statistics about a document in the knowledge graph

    Returns:
        {chunks, entities, edges, tokens}
    """
    chunks_file = f"expr/{data_source}/kv_store_text_chunks.json"
    entities_file = f"expr/{data_source}/kv_store_entities.json"
    edges_file = f"expr/{data_source}/kv_store_bipartite_edges.json"

    stats = {
        "chunks": 0,
        "entities": 0,
        "edges": 0,
        "tokens": 0
    }

    # Count chunks
    if os.path.exists(chunks_file):
        with open(chunks_file) as f:
            chunks = json.load(f)

        # Filter by document ID
        doc_chunks = [
            c for c_id, c in chunks.items()
            if document_id in c_id or c.get("doc_id") == document_id
        ]
        stats["chunks"] = len(doc_chunks)
        stats["tokens"] = sum(c.get("tokens", 0) for c in doc_chunks)

    # Count entities (by source_id)
    if os.path.exists(entities_file):
        with open(entities_file) as f:
            entities = json.load(f)

        doc_entities = [
            e for e_id, e in entities.items()
            if document_id in e.get("source_id", "")
        ]
        stats["entities"] = len(doc_entities)

    # Count edges (by source_id)
    if os.path.exists(edges_file):
        with open(edges_file) as f:
            edges = json.load(f)

        doc_edges = [
            edge for edge_id, edge in edges.items()
            if document_id in edge.get("source_id", "")
        ]
        stats["edges"] = len(doc_edges)

    return stats

async def get_document_entities(
    data_source: str,
    document_id: str,
    top_k: int = 10
) -> List[Dict]:
    """
    Get top entities for a document

    Returns:
        List of {name, type, weight} dicts
    """
    entities_file = f"expr/{data_source}/kv_store_entities.json"

    if not os.path.exists(entities_file):
        return []

    with open(entities_file) as f:
        entities = json.load(f)

    # Filter by document
    doc_entities = [
        {
            "name": e.get("entity_name"),
            "type": e.get("entity_type"),
            "weight": e.get("weight", 0)
        }
        for e_id, e in entities.items()
        if document_id in e.get("source_id", "")
    ]

    # Sort by weight
    doc_entities.sort(key=lambda x: x["weight"], reverse=True)

    return doc_entities[:top_k]

async def find_related_documents(
    data_source: str,
    document_id: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Find documents related by entity overlap

    Returns:
        List of {id, title, similarity} dicts
    """
    # Get entities for this document
    entities_file = f"expr/{data_source}/kv_store_entities.json"
    chunks_file = f"expr/{data_source}/kv_store_text_chunks.json"

    if not os.path.exists(entities_file):
        return []

    with open(entities_file) as f:
        entities = json.load(f)

    # Get this document's entities
    doc_entities = set([
        e.get("entity_name")
        for e_id, e in entities.items()
        if document_id in e.get("source_id", "")
    ])

    # Find other documents with overlapping entities
    doc_scores = {}

    for e_id, e in entities.items():
        entity_name = e.get("entity_name")

        if entity_name not in doc_entities:
            continue

        # Get all documents mentioning this entity
        source_ids = e.get("source_id", "").split(GRAPH_FIELD_SEP)

        for source_id in source_ids:
            if source_id == document_id:
                continue

            if source_id not in doc_scores:
                doc_scores[source_id] = 0

            doc_scores[source_id] += 1

    # Normalize scores
    max_score = max(doc_scores.values()) if doc_scores else 1

    related = [
        {
            "id": doc_id,
            "title": await get_document_title(data_source, doc_id),
            "similarity": score / max_score
        }
        for doc_id, score in doc_scores.items()
    ]

    # Sort by similarity
    related.sort(key=lambda x: x["similarity"], reverse=True)

    return related[:top_k]

async def get_document_title(data_source: str, document_id: str) -> str:
    """Get document title from registry or corpus"""

    # Try registry first
    doc = await registry.get_document(document_id, dataset=data_source)

    if doc:
        return doc.get("title", document_id)

    # Fallback to corpus
    corpus_file = f"datasets/{data_source}/raw/corpus.jsonl"

    if os.path.exists(corpus_file):
        with open(corpus_file) as f:
            for line in f:
                doc = json.loads(line)
                if doc.get("id") == document_id:
                    return doc.get("title", document_id)

    return document_id

async def remove_from_corpus(data_source: str, document_id: str):
    """
    Remove document from corpus.jsonl (hard delete)

    Rewrites corpus without the specified document
    """
    corpus_file = f"datasets/{data_source}/raw/corpus.jsonl"
    temp_file = f"{corpus_file}.tmp"

    if not os.path.exists(corpus_file):
        return

    # Rewrite corpus without this document
    with open(corpus_file, 'r', encoding='utf-8') as f_in:
        with open(temp_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                doc = json.loads(line)
                if doc.get("id") != document_id:
                    f_out.write(line)

    # Replace original
    os.replace(temp_file, corpus_file)

async def rebuild_entire_graph(dataset: str, job_id: str):
    """
    Rebuild entire knowledge graph from corpus (for hard deletes)

    WARNING: This clears existing graph and rebuilds from scratch
    """
    job = ProcessingJob(
        job_id=job_id,
        document_id="rebuild",
        dataset=dataset,
        status=JobStatus.PROCESSING
    )
    processing_jobs[job_id] = job

    try:
        # Clear existing graph files
        working_dir = f"expr/{dataset}"

        for file in [
            "kv_store_entities.json",
            "kv_store_bipartite_edges.json",
            "kv_store_text_chunks.json",
            "index_entity.bin",
            "index_bipartite_edge.bin",
            "index.bin"
        ]:
            path = os.path.join(working_dir, file)
            if os.path.exists(path):
                os.remove(path)

        # Reload all documents from corpus
        corpus_file = f"datasets/{dataset}/raw/corpus.jsonl"

        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        documents = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents.append({
                    "content": doc.get("contents", ""),
                    "title": doc.get("title", "")
                })

        # Rebuild graph
        # Process in batches
        batch_size = 10

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            await rag.ainsert(batch)

            progress = (i + len(batch)) / len(documents)
            job.update(progress=progress)

        job.update(
            status=JobStatus.COMPLETED,
            progress=1.0,
            completed_at=datetime.now()
        )

    except Exception as e:
        job.update(
            status=JobStatus.FAILED,
            error=str(e),
            completed_at=datetime.now()
        )
```

---

## 7. File Structure & Organization

```
d:\BiG-RAG\
├── script_api.py                    # Main API server (enhanced)
│
├── api/                              # NEW: API support modules
│   ├── __init__.py
│   ├── models.py                     # Pydantic request/response models
│   ├── jobs.py                       # Job queue management
│   ├── registry.py                   # Document registry
│   ├── kg_utils.py                   # KG helper functions
│   └── utils.py                      # Markdown processing, validation
│
├── datasets/                         # Document storage
│   ├── user_uploads/                 # Default for API uploads
│   │   ├── raw/
│   │   │   └── corpus.jsonl         # APPEND-only document storage
│   │   └── processed/
│   │
│   └── {custom_dataset}/
│       └── ...
│
├── expr/                             # Knowledge graph files
│   ├── user_uploads/
│   │   ├── documents_registry.json  # NEW: Document metadata index
│   │   ├── kv_store_entities.json
│   │   ├── kv_store_bipartite_edges.json
│   │   ├── kv_store_text_chunks.json
│   │   ├── index_entity.bin
│   │   ├── index_bipartite_edge.bin
│   │   └── graph_chunk_entity_relation.graphml
│   │
│   └── {custom_dataset}/
│       └── ...
│
├── bigrag/                           # Core BiG-RAG library (no changes)
│   ├── bigrag.py
│   ├── operate.py
│   └── ...
│
└── requirements.txt                  # Add: markdown, beautifulsoup4
```

---

## 8. Implementation Phases

### **Phase 1: Core Infrastructure** (Priority 1)

**Estimated Time:** 2-3 days

**Tasks:**
1. Create `api/` module structure
2. Implement `ProcessingJob` and job queue (`api/jobs.py`)
3. Implement `DocumentRegistry` (`api/registry.py`)
4. Implement Markdown processor (`api/utils.py`)
5. Implement KG helper functions (`api/kg_utils.py`)
6. Create Pydantic models (`api/models.py`)

**Deliverables:**
- ✅ Background job processing
- ✅ Document registry system
- ✅ Markdown support
- ✅ Helper utilities

---

### **Phase 2: Core Endpoints** (Priority 1)

**Estimated Time:** 3-4 days

**Tasks:**
1. Enhanced `/upload` with Markdown support
2. `/status/{job_id}` endpoint
3. `/documents` list endpoint
4. `/documents/{id}` details endpoint
5. `/documents/{id}` delete endpoint
6. `/graph/stats` endpoint

**Deliverables:**
- ✅ Full CRUD for documents
- ✅ Job status tracking
- ✅ Graph statistics

---

### **Phase 3: Batch Operations** (Priority 2)

**Estimated Time:** 2 days

**Tasks:**
1. `/upload/batch` endpoint
2. `/status/batch/{batch_id}` endpoint
3. `/documents/batch` delete endpoint
4. Batch processing optimizations

**Deliverables:**
- ✅ Batch upload
- ✅ Batch status tracking
- ✅ Batch deletion

---

### **Phase 4: Advanced Features** (Priority 3)

**Estimated Time:** 2-3 days

**Tasks:**
1. Enhanced `/health` with queue stats
2. `/graph/clear` endpoint
3. Document export functionality
4. Related document discovery
5. Advanced filtering

**Deliverables:**
- ✅ Complete API surface
- ✅ Production-ready monitoring

---

### **Phase 5: Testing & Documentation** (Priority 1)

**Estimated Time:** 2 days

**Tasks:**
1. Unit tests for all endpoints
2. Integration tests
3. Swagger UI examples
4. API documentation
5. Performance testing

**Deliverables:**
- ✅ Test coverage >80%
- ✅ Complete API docs
- ✅ Performance benchmarks

---

## 9. Testing Strategy

### Unit Tests

**Test File:** `tests/test_api_endpoints.py`

```python
import pytest
from fastapi.testclient import TestClient
from script_api import app

client = TestClient(app)

def test_upload_txt_file():
    """Test uploading a .txt file"""

    with open("test_data/sample.txt", "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("sample.txt", f, "text/plain")},
            data={"title": "Test Document"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "job_id" in data
    assert "document_id" in data

def test_upload_markdown_file():
    """Test uploading a .md file"""

    md_content = "# Test\n\nThis is **markdown**."

    response = client.post(
        "/upload",
        files={"file": ("test.md", md_content.encode(), "text/markdown")},
        data={"title": "Markdown Test"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.md"

def test_upload_invalid_file_type():
    """Test that non-.txt/.md files are rejected"""

    response = client.post(
        "/upload",
        files={"file": ("test.pdf", b"fake pdf", "application/pdf")}
    )

    assert response.status_code == 400
    assert "Only .txt and .md files supported" in response.text

def test_job_status():
    """Test job status endpoint"""

    # First upload a document
    with open("test_data/sample.txt", "rb") as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("sample.txt", f, "text/plain")}
        )

    job_id = upload_response.json()["job_id"]

    # Check status
    status_response = client.get(f"/status/{job_id}")

    assert status_response.status_code == 200
    data = status_response.json()
    assert data["job_id"] == job_id
    assert "status" in data
    assert "progress" in data

def test_list_documents():
    """Test document listing with pagination"""

    response = client.get("/documents?page=1&page_size=10")

    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)

def test_document_details():
    """Test getting document details"""

    # Upload a document first
    with open("test_data/sample.txt", "rb") as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("sample.txt", f, "text/plain")}
        )

    doc_id = upload_response.json()["document_id"]

    # Get details
    response = client.get(f"/documents/{doc_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == doc_id
    assert "content" in data
    assert "stats" in data

def test_delete_document():
    """Test document deletion"""

    # Upload
    with open("test_data/sample.txt", "rb") as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("sample.txt", f, "text/plain")}
        )

    doc_id = upload_response.json()["document_id"]

    # Delete (soft)
    response = client.delete(f"/documents/{doc_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["deletion_type"] == "soft"

def test_graph_stats():
    """Test graph statistics endpoint"""

    response = client.get("/graph/stats")

    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_entities" in data
    assert "storage_size" in data
```

---

### Integration Tests

```python
def test_full_upload_flow():
    """Test complete upload → index → query flow"""

    # 1. Upload document
    md_content = """
    # Artificial Intelligence

    AI is the simulation of human intelligence by machines.
    Machine learning is a subset of AI.
    """

    upload_response = client.post(
        "/upload",
        files={"file": ("ai.md", md_content.encode(), "text/markdown")},
        data={
            "title": "AI Introduction",
            "metadata": json.dumps({"category": "education"})
        }
    )

    assert upload_response.status_code == 200
    job_id = upload_response.json()["job_id"]
    doc_id = upload_response.json()["document_id"]

    # 2. Wait for processing (poll status)
    import time
    max_wait = 60
    elapsed = 0

    while elapsed < max_wait:
        status_response = client.get(f"/status/{job_id}")
        status = status_response.json()["status"]

        if status == "completed":
            break
        elif status == "failed":
            pytest.fail("Document processing failed")

        time.sleep(2)
        elapsed += 2

    assert status == "completed"

    # 3. Verify document in list
    list_response = client.get("/documents?status=indexed")
    docs = list_response.json()["documents"]

    assert any(d["document_id"] == doc_id for d in docs)

    # 4. Query knowledge graph
    query_response = client.post(
        "/ask",
        json={"question": "What is machine learning?", "top_k": 5}
    )

    assert query_response.status_code == 200
    contexts = query_response.json()["retrieved_contexts"]

    # Should retrieve our document
    assert len(contexts) > 0
    assert any("machine learning" in c["context"].lower() for c in contexts)

    # 5. Check stats
    stats_response = client.get("/graph/stats")
    stats = stats_response.json()

    assert stats["total_documents"] >= 1
    assert stats["total_entities"] > 0
```

---

## 10. Error Handling & Edge Cases

### Error Scenarios

**1. Duplicate Document Upload**

```python
# Same content uploaded twice
# BiGRAG handles via MD5 hash check in full_docs
# No error thrown, but ainsert() skips processing

# API Response:
{
  "success": true,
  "message": "Document already indexed (duplicate detected)",
  "document_id": "upload-abc123",
  "job_id": null,
  "status": "indexed",
  "is_duplicate": true
}
```

**2. OpenAI API Rate Limit**

```python
# Caught in process_document_background()

try:
    await rag.ainsert([doc])
except Exception as e:
    if "rate_limit" in str(e).lower():
        # Retry with exponential backoff
        await asyncio.sleep(5)
        await rag.ainsert([doc])
    else:
        raise
```

**3. Large File Upload**

```python
# Validate size before processing

if len(content_bytes) > 50 * 1024 * 1024:  # 50 MB
    raise HTTPException(
        413,
        f"File too large: {len(content_bytes)/1024/1024:.1f} MB (max 50 MB)"
    )
```

**4. Corrupt Markdown File**

```python
# Handle markdown parsing errors

try:
    plain_text = process_markdown(md_content)
except Exception as e:
    logger.error(f"Markdown parsing failed: {e}")
    # Fallback: use raw content
    plain_text = md_content
```

**5. Missing Corpus File (Delete Scenario)**

```python
# When corpus.jsonl is deleted but registry still has entries

content = await get_document_content_from_corpus(dataset, doc_id)

if content is None:
    raise HTTPException(
        404,
        f"Document corpus missing for {doc_id}. Rebuild required."
    )
```

**6. Graph Files Missing (Rebuild Required)**

```python
# Check for essential files before querying

required_files = [
    f"expr/{dataset}/kv_store_entities.json",
    f"expr/{dataset}/index_entity.bin"
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    raise HTTPException(
        500,
        f"Knowledge graph incomplete. Missing files: {missing}. Run /rebuild"
    )
```

---

## 11. Dependencies to Add

**Update:** `requirements.txt`

```txt
# Existing dependencies
fastapi
uvicorn
pydantic
python-multipart

# NEW for API enhancements
markdown>=3.5.0           # Markdown processing
beautifulsoup4>=4.12.0    # HTML parsing
python-dateutil>=2.8.0    # Date parsing
```

**Install:**
```bash
pip install markdown beautifulsoup4 python-dateutil
```

---

## 12. Deployment Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create `api/` directory structure
- [ ] Implement all modules in `api/`
- [ ] Enhance `script_api.py` with new endpoints
- [ ] Test all endpoints via Swagger UI (`/docs`)
- [ ] Run unit tests: `pytest tests/test_api_endpoints.py`
- [ ] Run integration tests
- [ ] Set environment variables (API keys)
- [ ] Start server: `python script_api.py --data_source user_uploads`
- [ ] Verify health check: `curl http://localhost:8001/health`
- [ ] Test upload: Upload sample .txt and .md files
- [ ] Monitor logs for errors
- [ ] Test document lifecycle: upload → status → list → delete

---

## 13. Performance Considerations

### Optimization Strategies

**1. Background Processing**
- Always use `process_async=True` for documents >10 KB
- Batch processing for multiple files reduces overhead

**2. Caching**
- LLM response caching enabled by default (60-70% cache hit rate)
- Embedding cache (if enabled) reduces API costs

**3. Pagination**
- Default `page_size=20`, max `page_size=100`
- Prevents memory issues with large document collections

**4. Incremental Graph Updates**
- `rag.ainsert()` updates existing graph (no rebuild needed)
- Only rebuild on hard deletes

**5. FAISS Index Optimization**
- Use `IndexFlatIP` for <100K vectors (exact search)
- Use `IndexIVFFlat` for larger datasets (approximate search)

---

## 14. Monitoring & Logging

### Log Levels

```python
import logging

# In script_api.py
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log important events
logger.info(f"Document uploaded: {doc_id}")
logger.warning(f"Processing slow for job {job_id}: {elapsed}s")
logger.error(f"Failed to process {doc_id}: {error}")
```

### Metrics to Track

- **Upload Rate**: Documents/hour
- **Processing Time**: Average seconds per document
- **Success Rate**: % completed vs failed jobs
- **Queue Depth**: Pending + processing jobs
- **API Response Time**: p50, p95, p99 latency
- **Storage Growth**: MB/day

---

## 15. Security Considerations

### Input Validation

```python
# File size limits
MAX_FILE_SIZE_MB = 50

# File type whitelist (no blacklist)
ALLOWED_EXTENSIONS = {'.txt', '.md'}

# Content validation
def sanitize_metadata(metadata: dict) -> dict:
    """Remove potentially dangerous fields"""

    dangerous_keys = ['__proto__', 'constructor', 'prototype']

    return {
        k: v for k, v in metadata.items()
        if k not in dangerous_keys
    }

# Rate limiting (per IP)
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/upload")
@limiter.limit("100/hour")  # Max 100 uploads/hour per IP
async def upload_document(...):
    ...
```

---

## 16. Future Enhancements (Out of Scope)

**Not included in this plan, but possible extensions:**

- [ ] Authentication & API keys
- [ ] Multi-user support with permissions
- [ ] Document versioning
- [ ] Automatic entity linking across datasets
- [ ] GraphQL API
- [ ] WebSocket support for real-time status updates
- [ ] Document annotations & comments
- [ ] Advanced analytics dashboard
- [ ] Export to Neo4J, ArangoDB
- [ ] PDF/DOCX support
- [ ] Image OCR for diagrams in Markdown

---

## Summary

This plan provides a **complete, production-ready document management system** for BiG-RAG, transforming `script_api.py` into a comprehensive RAG API with:

✅ **Markdown & Text Support**
✅ **Async Job Processing with Status Tracking**
✅ **Document Registry for Metadata Management**
✅ **Complete CRUD Operations**
✅ **Batch Upload & Delete**
✅ **Graph Statistics & Monitoring**
✅ **Production-Ready Error Handling**

**Total Implementation Time:** 10-12 days (with testing)

**Ready to proceed?** This plan can be executed in phases, with Phase 1 & 2 (Core Infrastructure + Core Endpoints) providing immediate value in ~5-7 days.

---

**END OF PLAN** | Version 1.0 | 2025-10-30
