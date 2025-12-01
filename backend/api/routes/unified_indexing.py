"""
Unified Indexing Endpoint with Feature Flags

Provides granular control over indexing without presets.
Users enable specific features via form parameters.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel

from bigrag.utils import logger
from bigrag.config import IndexingConfig  # NEW: 16 independent features
from bigrag import BiGRAG
from ..services.utils import process_markdown, validate_file_upload, truncate_text
from ..services.kg_utils import compute_doc_id, add_document_to_corpus
from ..services.registry import registry


router = APIRouter(prefix="/indexing", tags=["Unified Indexing"])


# Response Model
class IndexingResponse(BaseModel):
    """Response for indexing endpoint"""
    success: bool
    message: str
    dataset_name: str
    document_id: str
    job_id: str
    filename: str
    title: str
    content_preview: str
    content_length: int
    status: str
    metadata: Optional[Dict[str, Any]] = None
    upload_date: str
    features_enabled: Dict[str, Any]
    estimated_time: str
    estimated_cost: str


@router.post("/index-document", response_model=IndexingResponse)
async def index_document_with_features(
    background_tasks: BackgroundTasks,

    # Required
    file: UploadFile = File(..., description="Document to index (.txt or .md)"),
    data_source: str = Form(..., description="Dataset name"),

    # Optional metadata
    title: str = Form(None, description="Document title"),
    metadata: str = Form(None, description="JSON metadata"),
    process_async: bool = Form(True, description="Process in background"),

    # ========================================
    # GROUP A: CHUNKING FEATURES (2 features)
    # ========================================
    chunking_strategy: str = Form("semantic", description="Chunking: 'token' (fast) | 'semantic' (accurate)"),
    enable_table_detection: bool = Form(True, description="GPT-4 table extraction (requires API key)"),

    # ========================================
    # GROUP B: EXTRACTION FEATURES (4 features)
    # ========================================
    extraction_strategy: str = Form("gleaning", description="Extraction: 'strict' (single-pass) | 'gleaning' (multi-pass)"),
    enable_table_fact_extraction: bool = Form(False, description="Rule-based table→facts (requires table detection)"),
    enable_multilingual: bool = Form(True, description="Multilingual extraction support"),

    # ========================================
    # GROUP C: VALIDATION FEATURES (3 features)
    # ========================================
    enable_numeric_validation: bool = Form(False, description="Gemini numeric validation (expensive)"),
    enable_entity_validation: bool = Form(True, description="Entity quality validation (cheap)"),
    enable_relation_validation: bool = Form(True, description="Relation validation (cheap)"),

    # ========================================
    # GROUP D: MERGING FEATURES (2 features)
    # ========================================
    enable_entity_merging: bool = Form(True, description="Enable entity deduplication"),
    enable_fuzzy_matching: bool = Form(True, description="Fuzzy string matching for merging"),

    # ========================================
    # GROUP E: QUALITY FEATURES (3 features)
    # ========================================
    enable_hitl: bool = Form(True, description="Human-in-the-loop failure tracking"),
    enable_orphan_linking: bool = Form(True, description="Link orphan entities"),
    enable_quality_scoring: bool = Form(True, description="Track quality metrics"),

    # ========================================
    # PARAMETERS (not features)
    # ========================================
    # Chunking parameters
    chunk_size: int = Form(1200, description="Chunk size in tokens"),
    chunk_overlap: int = Form(100, description="Overlap between chunks"),

    # Extraction parameters
    gleaning_iterations: int = Form(2, description="Gleaning passes (if extraction_strategy='gleaning')"),
    extraction_concurrency: int = Form(16, description="Max concurrent LLM calls"),

    # Validation parameters
    validation_strictness: str = Form("MODERATE", description="STRICT (99%) | MODERATE (95%) | LENIENT (80%)"),
    numeric_validation_mode: str = Form("document", description="chunk | document (if numeric validation enabled)"),

    # Merging parameters
    fuzzy_similarity_threshold: float = Form(0.9, description="Fuzzy match threshold 0-1 (if fuzzy matching enabled)")
):
    """
    Index document with explicit feature control (16 independent features).

    **NEW (January 2025)**: Updated to 16 independent boolean features with clear
    separation from configuration parameters.

    **Architecture**: Uses modular indexing system (IndexingConfig + BiGRAG).
    All features are independent with explicit dependency validation.

    **Feature Groups**:
    - Group A: Chunking (2 features) - How documents are split
    - Group B: Extraction (4 features) - How entities/relations are extracted
    - Group C: Validation (3 features) - What validations are applied
    - Group D: Merging (2 features) - How duplicates are merged
    - Group E: Quality (3 features) - Quality tracking and HITL

    Example 1 (Fast & Cheap - Default):
    ```bash
    curl -X POST "http://localhost:8001/indexing/index-document" \\
      -F "file=@doc.md" \\
      -F "data_source=my_data"
      # Defaults: semantic chunking + table detection + gleaning + entity/relation validation
    ```

    Example 2 (Maximum Speed - Token Chunking):
    ```bash
    curl -X POST "http://localhost:8001/indexing/index-document" \\
      -F "file=@doc.md" \\
      -F "data_source=my_data" \\
      -F "chunking_strategy=token" \\
      -F "enable_table_detection=false" \\
      -F "extraction_strategy=strict" \\
      -F "enable_fuzzy_matching=false"
    ```

    Example 3 (Maximum Quality - All Features):
    ```bash
    curl -X POST "http://localhost:8001/indexing/index-document" \\
      -F "file=@doc.md" \\
      -F "data_source=my_data" \\
      -F "chunking_strategy=semantic" \\
      -F "enable_table_detection=true" \\
      -F "extraction_strategy=gleaning" \\
      -F "enable_table_fact_extraction=true" \\
      -F "enable_numeric_validation=true" \\
      -F "enable_fuzzy_matching=true"
    ```
    """
    try:
        # Step 1: Validate file
        if not file.filename.endswith(('.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .md files supported"
            )

        # Step 2: Read content
        content_bytes = await file.read()
        is_valid, error_msg = validate_file_upload(content_bytes, file.filename, max_size_mb=50)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        content_text = content_bytes.decode('utf-8')

        # Process markdown if needed
        if file.filename.endswith('.md'):
            try:
                content_text = process_markdown(content_text)
            except Exception as e:
                logger.warning(f"Markdown processing failed: {e}. Using raw content.")

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="Empty file")

        # Step 3: Parse metadata
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON metadata")

        # Step 4: Build IndexingConfig directly from form parameters (NEW - 16 independent features)
        indexing_config = IndexingConfig(
            # Group A: Chunking
            chunking_strategy=chunking_strategy,
            enable_table_detection=enable_table_detection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,

            # Group B: Extraction
            extraction_strategy=extraction_strategy,
            enable_table_fact_extraction=enable_table_fact_extraction,
            enable_multilingual=enable_multilingual,
            gleaning_iterations=gleaning_iterations,
            extraction_concurrency=extraction_concurrency,

            # Group C: Validation
            enable_numeric_validation=enable_numeric_validation,
            enable_entity_validation=enable_entity_validation,
            enable_relation_validation=enable_relation_validation,
            validation_strictness=validation_strictness,
            numeric_validation_mode=numeric_validation_mode,

            # Group D: Merging
            enable_entity_merging=enable_entity_merging,
            enable_fuzzy_matching=enable_fuzzy_matching,
            fuzzy_similarity_threshold=fuzzy_similarity_threshold,

            # Group E: Quality
            enable_hitl=enable_hitl,
            enable_orphan_linking=enable_orphan_linking,
            enable_quality_scoring=enable_quality_scoring,

            # API Keys
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            gemini_api_key=os.getenv('GEMINI_API_KEY'),

            # Dataset path (for HITL)
            dataset_path=os.path.join(os.getenv('WORKING_DIR', './expr'), data_source)
        )

        # Validation happens in IndexingConfig.__post_init__() automatically
        # Any dependency errors will raise ValueError with clear message

        # Step 5: Ensure dataset exists
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
        expr_dir = PROJECT_ROOT / "expr" / data_source
        dataset_dir = PROJECT_ROOT / "datasets" / data_source / "raw"
        expr_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Step 6: Generate IDs
        doc_id = compute_doc_id(content_text, prefix="doc")
        job_id = f"job-{compute_doc_id(doc_id + str(datetime.now()), prefix='')}"
        doc_title = title or file.filename

        # Step 7: Add to corpus
        await add_document_to_corpus(
            data_source=data_source,
            doc_id=doc_id,
            content=content_text,
            title=doc_title,
            metadata=doc_metadata
        )

        # Step 8: Add to registry
        await registry.add_document(
            document_id=doc_id,
            filename=file.filename,
            title=doc_title,
            content_length=len(content_text),
            dataset=data_source,
            metadata=doc_metadata,
            job_id=job_id,
            status="pending"
        )

        # Step 9: Estimate time and cost
        estimated_time, estimated_cost = _estimate_time_cost(indexing_config, len(content_text))

        # Step 10: Process document
        if process_async:
            # Background processing
            background_tasks.add_task(
                _process_with_bigrag,
                str(expr_dir),  # Pass absolute path (not dataset name)
                doc_id,
                content_text,
                doc_title,
                doc_metadata,
                indexing_config
            )
            status = "processing"
            logger.info(f"[Indexing] Queued job: {job_id}")
        else:
            # Synchronous processing
            await _process_with_bigrag(
                str(expr_dir),  # Pass absolute path (not dataset name)
                doc_id,
                content_text,
                doc_title,
                doc_metadata,
                indexing_config
            )
            status = "completed"
            logger.info(f"[Indexing] Completed synchronously")

        # Step 11: Build response
        return IndexingResponse(
            success=True,
            message=f"Document {'queued' if process_async else 'indexed'}",
            dataset_name=data_source,
            document_id=doc_id,
            job_id=job_id,
            filename=file.filename,
            title=doc_title,
            content_preview=truncate_text(content_text, 200),
            content_length=len(content_text),
            status=status,
            metadata=doc_metadata,
            upload_date=datetime.now().isoformat(),
            features_enabled={
                # Group A: Chunking
                "chunking_strategy": chunking_strategy,
                "table_detection": enable_table_detection,
                # Group B: Extraction
                "extraction_strategy": extraction_strategy,
                "table_fact_extraction": enable_table_fact_extraction,
                "multilingual": enable_multilingual,
                # Group C: Validation
                "numeric_validation": enable_numeric_validation,
                "entity_validation": enable_entity_validation,
                "relation_validation": enable_relation_validation,
                # Group D: Merging
                "entity_merging": enable_entity_merging,
                "fuzzy_matching": enable_fuzzy_matching,
                # Group E: Quality
                "hitl": enable_hitl,
                "orphan_linking": enable_orphan_linking,
                "quality_scoring": enable_quality_scoring
            },
            estimated_time=estimated_time,
            estimated_cost=estimated_cost
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Indexing] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


async def _process_with_bigrag(
    working_dir: str,
    document_id: str,
    content: str,
    title: str,
    metadata: Dict[str, Any],
    config: IndexingConfig  # NEW: Accepts IndexingConfig directly
) -> Dict[str, Any]:
    """
    Process document using BiGRAG with modular indexing system.

    UPDATED (January 2025): Now uses IndexingConfig directly with 16 independent features.

    Args:
        working_dir: Absolute path to dataset directory (e.g., D:/BiG-RAG/expr/kuet_test)
                    This ensures files are created in the correct location regardless of CWD.
        document_id: Unique document identifier
        content: Document content (markdown text)
        title: Document title
        metadata: Document metadata dict
        config: IndexingConfig with 16 independent features

    Returns:
        Dict with processing result

    Architecture:
        1. Ensure dataset path is set in config
        2. Initialize BiGRAG with modular system
        3. Use index_document() method (modular pipeline)
        4. Return success
    """
    try:
        logger.info(f"[Indexing] Processing {document_id} with modular system (16 features)")
        logger.debug(f"[Indexing] Working directory: {working_dir}")

        # STEP 1: Ensure dataset path is set
        config.dataset_path = working_dir

        logger.info(f"[Indexing] Config: chunking={config.chunking_strategy}, extraction={config.extraction_strategy}")
        logger.info(f"[Indexing] Features: table_detect={config.enable_table_detection}, "
                   f"numeric_val={config.enable_numeric_validation}, "
                   f"fuzzy={config.enable_fuzzy_matching}")

        # STEP 2: Initialize BiGRAG with modular system
        rag = BiGRAG(
            working_dir=working_dir,
            indexing_config=config
        )

        # STEP 3: Index document using modular pipeline
        result = await rag.index_document(
            text=content,
            metadata={
                "title": title,
                "document_id": document_id,
                **metadata
            }
        )

        logger.info(f"[Indexing] Completed: {document_id}")
        logger.info(f"[Indexing] Statistics: {result.get('statistics', {})}")
        logger.info(f"[Indexing] Validation: {result.get('validation', {})}")

        return {
            "success": True,
            "document_id": document_id,
            "statistics": result.get('statistics', {}),
            "validation": result.get('validation', {})
        }

    except Exception as e:
        logger.error(f"[Indexing] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def _estimate_time_cost(config: IndexingConfig, content_length: int) -> tuple:
    """
    Estimate processing time and cost based on enabled features.

    Args:
        config: IndexingConfig with 16 independent features
        content_length: Document length in characters

    Returns:
        Tuple of (time_str, cost_str)
    """
    base_time = 30  # seconds
    base_cost = 0.05  # USD

    # Table detection (if enabled)
    if config.enable_table_detection:
        base_time += 30
        base_cost += 0.05

    # Gleaning extraction (if multi-pass)
    if config.extraction_strategy == "gleaning":
        base_time += 60 * config.gleaning_iterations
        base_cost += 0.10 * config.gleaning_iterations

    # Numeric validation (expensive)
    if config.enable_numeric_validation:
        base_time += 20
        base_cost += 0.05

    # Fuzzy matching (adds processing time)
    if config.enable_fuzzy_matching:
        base_time += 10
        base_cost += 0.02

    # Scale by document size
    scale = content_length / 40000
    total_time = int(base_time * scale)
    total_cost = base_cost * scale

    # Format
    if total_time < 60:
        time_str = f"{total_time}s"
    else:
        time_str = f"{total_time // 60}min {total_time % 60}s"

    cost_str = f"${total_cost:.3f}" if total_cost < 0.10 else f"${total_cost:.2f}"

    return time_str, cost_str
