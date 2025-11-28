"""
Document management routes
"""

import json
from datetime import datetime
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from bigrag.utils import logger

from ..core.dependencies import RAGDep, get_data_source, get_working_dir
from ..models.models import (
    UploadResponse as EnhancedUploadResponse,
    DeleteResponse,
    RebuildResponse,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentSummary,
    DocumentFilter
)
from ..services.jobs import (
    ProcessingJob, JobStatus, ProcessingStage,
    processing_jobs, process_document_background
)
from ..services.registry import registry
from ..services.utils import process_markdown, validate_file_upload, truncate_text
from ..services.kg_utils import (
    get_document_stats_from_kg,
    get_document_entities,
    find_related_documents,
    get_document_content_from_corpus,
    rebuild_entire_graph,
    compute_doc_id,
    add_document_to_corpus,
    remove_from_corpus
)


router = APIRouter(prefix="/documents", tags=["Document Management"])


@router.post("/upload", response_model=EnhancedUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Text or Markdown file to upload (.txt or .md)"),
    title: str = Form(None, description="Optional document title (defaults to filename)"),
    data_source: str = Form(None, description="Dataset name (defaults to current dataset)"),
    process_async: bool = Form(True, description="Process in background (recommended for large files)"),
    metadata: str = Form(None, description="Optional JSON metadata: {\"category\": \"research\", \"tags\": [...]}"),
    preset: str = Form("standard", description="Pipeline preset: 'standard' (fast) | 'quality' (accurate) | 'balanced' (medium)")
):
    """
    Upload a document (.txt or .md) and add it to the knowledge graph.

    **Enhanced Features:**
    - Supports both .txt and .md (Markdown) files
    - Background processing with job tracking
    - Document registry for metadata management
    - Optional metadata (category, tags, custom fields)
    - **NEW:** Production Pipeline support (table-aware, higher accuracy)
    - Progress tracking via /status/{job_id}

    **Example usage:**
    ```bash
    # Basic upload (standard preset - default)
    curl -X POST "http://localhost:8001/documents/upload" \\
      -F "file=@document.md" \\
      -F "title=My Research Paper"

    # With metadata (standard preset)
    curl -X POST "http://localhost:8001/documents/upload" \\
      -F "file=@document.md" \\
      -F "title=BiG-RAG Paper" \\
      -F 'metadata={"category":"research","tags":["RAG","NLP"]}'

    # With quality preset (table-aware, higher accuracy for educational content)
    curl -X POST "http://localhost:8001/documents/upload" \\
      -F "file=@kuet_admission.md" \\
      -F "title=KUET Admission Guide" \\
      -F 'metadata={"category":"education","tags":["KUET","admission"]}' \\
      -F "preset=quality"

    # With balanced preset (medium speed/quality)
    curl -X POST "http://localhost:8001/documents/upload" \\
      -F "file=@my_doc.md" \\
      -F "preset=balanced"
    ```

    **Pipeline Presets:**
    - **standard (default):** Fast, token-based chunking, gleaning, ~30-60s per 40K doc
    - **quality:** Table-aware, validation, fuzzy merging, ~2-5min per 40K doc, highest accuracy
    - **balanced:** Table detection, single-pass extraction, ~1-2min per 40K doc, good speed/quality trade-off

    **Returns:** job_id for tracking processing status via /status/{job_id}
    """
    try:
        # Check if we're in unified mode
        from ..core.dependencies import get_unified_executor
        unified_executor = get_unified_executor()

        # Get or create RAG instance based on mode
        if unified_executor:
            # UNIFIED MODE: data_source is required, create RAG instance on-demand
            if not data_source or data_source == "string":
                raise HTTPException(
                    status_code=400,
                    detail="data_source parameter is required in unified mode (e.g., 'kuet_test', 'football')"
                )
            target_dataset = data_source
            current_data_source = data_source

            # Create temporary RAG instance for this dataset
            from bigrag import BiGRAG
            from bigrag.llm import gpt_4o_mini_complete
            from bigrag.config import config
            from pathlib import Path

            # Path from backend/api/routes/documents.py -> D:\BiG-RAG
            PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
            working_dir = str(PROJECT_ROOT / "expr" / target_dataset)

            # Create pipeline features from preset
            from bigrag.pipeline.features import PipelineFeatures
            import os

            pipeline_features = PipelineFeatures.from_preset(
                preset,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                gemini_api_key=os.getenv("GEMINI_API_KEY")
            )

            logger.info(f"[Unified Mode] Creating RAG instance for dataset: {target_dataset} with preset={preset}")
            rag = BiGRAG(
                working_dir=working_dir,
                llm_model_func=gpt_4o_mini_complete,
                chunk_token_size=config.chunk_size,
                chunk_overlap_token_size=config.chunk_overlap_size,
                enable_llm_cache=config.enable_llm_cache,
                addon_params={"language": config.default_language},
                pipeline_features=pipeline_features
            )
        else:
            # SINGLE MODE: use injected RAG instance
            from ..core.dependencies import get_rag_instance
            rag = get_rag_instance()
            current_data_source = get_data_source()

        # Validate file extension
        if not file.filename.endswith(('.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .md files are supported"
            )

        # Read file content
        content_bytes = await file.read()

        # Validate file
        is_valid, error_msg = validate_file_upload(
            content_bytes,
            file.filename,
            max_size_mb=50
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Decode content
        content_text = content_bytes.decode('utf-8')

        # Process Markdown if .md
        if file.filename.endswith('.md'):
            logger.info(f"Processing Markdown file: {file.filename}")
            content_text = process_markdown(content_text)

        # Validate content not empty
        if not content_text.strip():
            raise HTTPException(status_code=400, detail="File is empty after processing")

        # Parse metadata
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON in metadata field"
                )

        # Determine target dataset
        target_dataset = data_source if (data_source and data_source != "string") else current_data_source

        # Ensure dataset exists and is registered (same as /datasets/create-and-index)
        dataset_info = None
        if unified_executor:
            from .datasets import ensure_dataset_exists
            logger.info(f"[Upload] Ensuring dataset exists: {target_dataset}")
            dataset_info = await ensure_dataset_exists(target_dataset)
            logger.info(f"[Upload] Dataset check complete. Registry updated: {dataset_info['registry_updated']}")

        # Generate IDs
        doc_id = compute_doc_id(content_text, prefix="doc")
        job_id = f"job-{compute_doc_id(doc_id + str(datetime.now()), prefix='')}"

        # Use filename as title if not provided
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

        # Create processing job
        job = ProcessingJob(
            job_id=job_id,
            document_id=doc_id,
            dataset=target_dataset,
            status=JobStatus.PENDING,
            progress=0.0,
            stage=ProcessingStage.QUEUED
        )
        processing_jobs[job_id] = job

        # Process document
        if process_async:
            # Background processing
            background_tasks.add_task(
                process_document_background,
                job_id=job_id,
                content=content_text,
                title=doc_title,
                dataset=target_dataset,
                rag_instance=rag,
                registry_instance=registry,
                metadata=doc_metadata
            )
            message = f"Document queued for processing (preset={preset})"
        else:
            # Synchronous processing
            await process_document_background(
                job_id=job_id,
                content=content_text,
                title=doc_title,
                dataset=target_dataset,
                rag_instance=rag,
                registry_instance=registry,
                metadata=doc_metadata
            )
            message = f"Document processed successfully (preset={preset})"

        # Reload registry in unified executor if dataset was just added
        if unified_executor and dataset_info.get('registry_updated', False):
            try:
                unified_executor.reload_registry()
                logger.info(f"[Upload] Reloaded unified executor registry after adding new dataset")
            except Exception as e:
                logger.warning(f"[Upload] Failed to reload registry: {e}")

        # Return response
        return EnhancedUploadResponse(
            success=True,
            message=message,
            document_id=doc_id,
            job_id=job_id,
            filename=file.filename,
            title=doc_title,
            content_preview=truncate_text(content_text, 200),
            content_length=len(content_text),
            dataset=target_dataset,
            status=job.status.value if isinstance(job.status, JobStatus) else job.status,
            metadata=doc_metadata,
            upload_date=datetime.now().isoformat()
        )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File encoding error. Please upload UTF-8 encoded files"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_graph(
    rag: RAGDep,
    data_source: str = Form(None, description="Dataset name (defaults to current dataset)"),
    force_full_rebuild: bool = Form(False, description="Force full rebuild instead of incremental")
):
    """
    Manually trigger knowledge graph rebuild.

    By default, performs incremental update (adds new documents from corpus.jsonl).
    Use `force_full_rebuild=true` to rebuild entire graph from scratch.

    **Example usage:**
    ```bash
    # Incremental rebuild
    curl -X POST "http://localhost:8001/documents/rebuild"

    # Full rebuild
    curl -X POST "http://localhost:8001/documents/rebuild" \\
      -F "force_full_rebuild=true"
    ```
    """
    try:
        current_data_source = get_data_source()
        target_dataset = data_source or current_data_source

        if force_full_rebuild:
            success, message = await rebuild_entire_graph(target_dataset, rag)
        else:
            # Incremental rebuild logic here
            # For now, just indicate incremental is not yet implemented
            success = False
            message = "Incremental rebuild not yet implemented. Use force_full_rebuild=true."

        return RebuildResponse(
            success=success,
            message=message,
            dataset=target_dataset,
            rebuild_type="full" if force_full_rebuild else "incremental"
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    dataset: Optional[str] = None,
    search: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List all documents with optional filtering and pagination.

    **Filters:**
    - `dataset`: Filter by dataset name
    - `search`: Search in title or filename
    - `category`: Filter by category
    - `tags`: Comma-separated tags (e.g., "RAG,NLP")
    - `status`: Filter by status (pending, processing, indexed, failed, deleted)
    - `date_from`, `date_to`: Date range filter (ISO format)
    - `limit`: Results per page (1-500, default 50)
    - `offset`: Pagination offset (default 0)

    **Example usage:**
    ```bash
    # Get all documents
    curl "http://localhost:8001/documents"

    # Search with filters
    curl "http://localhost:8001/documents?search=research&category=science&limit=20"

    # Get documents by status
    curl "http://localhost:8001/documents?status=indexed"
    ```
    """
    try:
        # Parse tags
        tags_list = tags.split(',') if tags else None

        # Get documents from registry
        docs = await registry.list_documents(
            dataset=dataset,
            search=search,
            category=category,
            tags=tags_list,
            status=status,
            date_from=date_from,
            date_to=date_to
        )

        # Apply pagination
        total = len(docs)
        paginated = docs[offset:offset+limit]

        # Convert to DocumentSummary models
        summaries = [
            DocumentSummary(
                document_id=doc["document_id"],
                filename=doc["filename"],
                title=doc["title"],
                content_length=doc["content_length"],
                upload_date=doc["upload_date"],
                indexed_date=doc.get("indexed_date"),
                last_modified=doc["last_modified"],
                status=doc["status"],
                dataset=doc["dataset"],
                metadata=doc.get("metadata"),
                job_id=doc["job_id"]
            )
            for doc in paginated
        ]

        return DocumentListResponse(
            total=total,
            limit=limit,
            offset=offset,
            documents=summaries
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.) as-is
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document_details(
    document_id: str,
    include_entities: bool = True,
    include_related: bool = True
):
    """
    Get detailed information about a specific document.

    **Includes:**
    - Document metadata
    - Content preview
    - Knowledge graph statistics
    - Extracted entities (optional)
    - Related documents (optional)

    **Example usage:**
    ```bash
    curl "http://localhost:8001/documents/doc-abc123?include_entities=true&include_related=true"
    ```
    """
    try:
        current_data_source = get_data_source()
        working_dir = get_working_dir()

        # Get from registry
        doc = await registry.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        # Get the document's dataset (may differ from current dataset)
        doc_dataset = doc.get("dataset", current_data_source)

        # Get KG stats
        kg_stats = await get_document_stats_from_kg(doc_dataset, document_id)

        # Get entities if requested
        entities_list = []
        if include_entities:
            entities_list = await get_document_entities(doc_dataset, document_id)

        # Get related documents if requested
        related_docs = []
        if include_related:
            related_docs = await find_related_documents(doc_dataset, document_id, top_k=5)

        # Get content preview
        content_preview = await get_document_content_from_corpus(doc_dataset, document_id)

        return DocumentDetailResponse(
            document_id=document_id,
            title=doc.get("title", ""),
            filename=doc.get("filename", ""),
            dataset=doc.get("dataset", current_data_source),
            upload_date=doc.get("created_at", ""),
            last_modified=doc.get("updated_at", ""),
            content_length=doc.get("content_length", 0),
            content_preview=content_preview[:500] if content_preview else "",
            status=doc.get("status", "active"),
            metadata=doc.get("metadata"),
            job_id=doc.get("job_id"),
            stats=kg_stats,
            top_entities=entities_list,
            related_documents=related_docs
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document details: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    rag: RAGDep,
    hard_delete: bool = False
):
    """
    Delete a document from the system.

    **Delete Modes:**
    - Soft delete (default): Marks as deleted, preserves data
    - Hard delete: Removes from graph, corpus, and indices

    **Example usage:**
    ```bash
    # Soft delete
    curl -X DELETE "http://localhost:8001/documents/doc-abc123"

    # Hard delete
    curl -X DELETE "http://localhost:8001/documents/doc-abc123?hard_delete=true"
    ```
    """
    try:
        # Check if document exists
        doc = await registry.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        if hard_delete:
            # Hard delete: Remove from KG, corpus, and registry
            doc_dataset = doc.get("dataset", get_data_source())

            # 1. Delete from knowledge graph (chunks, entities, edges, vectors)
            await rag.adelete_document(document_id)

            # 2. Remove from corpus.jsonl (prevents resurrection on rebuild)
            await remove_from_corpus(doc_dataset, document_id)

            # 3. Delete from document registry
            await registry.delete_document(document_id, hard=True)

            message = f"Document {document_id} permanently deleted from all storage layers"
        else:
            # Soft delete: Mark as deleted
            await registry.delete_document(document_id, hard=False)
            message = f"Document {document_id} marked as deleted"

        return DeleteResponse(
            success=True,
            message=message,
            document_id=document_id,
            hard_delete=hard_delete,
            rebuild_required=False  # Cascade deletion handles everything
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
