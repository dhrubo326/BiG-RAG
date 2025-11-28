"""
Unified Indexing Endpoint with Explicit Feature Flags

This endpoint provides granular control over the indexing pipeline without presets.
Users can enable/disable individual features based on their needs.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field

from bigrag.utils import logger
from bigrag.pipeline.features import PipelineFeatures
from bigrag.pipeline.base_pipeline import UnifiedPipeline
from ..core.dependencies import get_unified_executor
from ..models.models import UploadResponse as EnhancedUploadResponse
from ..services.jobs import ProcessingJob, JobStatus, ProcessingStage, processing_jobs
from ..services.registry import registry
from ..services.utils import process_markdown, validate_file_upload, truncate_text
from ..services.kg_utils import compute_doc_id, add_document_to_corpus


router = APIRouter(prefix="/indexing", tags=["Unified Indexing (Modular)"])


# ============================================================================
# Request/Response Models
# ============================================================================

class UnifiedIndexingResponse(BaseModel):
    """Response for unified indexing endpoint"""
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


# ============================================================================
# Helper Functions
# ============================================================================

async def ensure_dataset_exists(dataset_name: str) -> Dict[str, Any]:
    """
    Ensure dataset directory structure exists and is registered.
    Same as datasets.py implementation.
    """
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

    # Create directories
    expr_dir = PROJECT_ROOT / "expr" / dataset_name
    dataset_dir = PROJECT_ROOT / "datasets" / dataset_name / "raw"

    expr_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[UnifiedIndexing] Created directories for: {dataset_name}")

    # Update subgraph registry
    registry_path = PROJECT_ROOT / "expr" / "subgraph_registry.json"
    registry_updated = False

    try:
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
        else:
            registry_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "description": "Registry of all available subgraphs",
                "subgraphs": {}
            }

        if dataset_name not in registry_data["subgraphs"]:
            registry_data["subgraphs"][dataset_name] = {
                "path": f"expr/{dataset_name}",
                "description": f"Auto-created dataset: {dataset_name}",
                "aliases": [dataset_name],
                "topics": [dataset_name, "auto-created"],
                "enabled": True,
                "created_at": datetime.now().isoformat(),
                "auto_created": True
            }

            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)

            registry_updated = True
            logger.info(f"[UnifiedIndexing] Added {dataset_name} to subgraph registry")

    except Exception as e:
        logger.error(f"[UnifiedIndexing] Failed to update registry: {e}")
        raise HTTPException(status_code=500, detail=f"Registry update failed: {e}")

    return {
        "expr_dir": str(expr_dir),
        "dataset_dir": str(dataset_dir),
        "registry_updated": registry_updated
    }


def estimate_time_and_cost(features: PipelineFeatures, content_length: int) -> tuple[str, str]:
    """
    Estimate processing time and cost based on enabled features.

    Args:
        features: Pipeline feature configuration
        content_length: Document length in characters

    Returns:
        (estimated_time, estimated_cost) as human-readable strings
    """
    # Base estimates for 40K character document
    base_doc_size = 40000
    scale = content_length / base_doc_size

    # Time estimates (in seconds)
    base_time = 30  # Token chunking, basic extraction

    if features.enable_table_detection:
        base_time += 30  # Table detection adds ~30s

    if features.enable_gleaning:
        base_time += 60 * features.max_gleaning_iterations  # 60s per gleaning pass

    if features.enable_table_fact_extraction:
        base_time += 10  # Rule-based, very fast

    if features.enable_numeric_validation:
        base_time += 20  # LLM validation

    if features.merge_strategy == "fuzzy":
        base_time += 30  # Fuzzy matching slower than basic

    total_seconds = int(base_time * scale)

    # Format time
    if total_seconds < 60:
        time_str = f"{total_seconds}s"
    elif total_seconds < 300:
        time_str = f"{total_seconds // 60}min {total_seconds % 60}s"
    else:
        time_str = f"{total_seconds // 60}min"

    # Cost estimates (in USD for 40K doc)
    base_cost = 0.05  # Minimal chunking + basic extraction

    if features.enable_table_detection:
        base_cost += 0.05  # GPT-4 table detection

    if features.enable_gleaning:
        base_cost += 0.10 * features.max_gleaning_iterations  # Extra LLM calls

    if features.enable_numeric_validation:
        base_cost += 0.05  # Gemini validation

    total_cost = base_cost * scale

    # Format cost
    if total_cost < 0.10:
        cost_str = f"${total_cost:.3f}"
    else:
        cost_str = f"${total_cost:.2f}"

    return time_str, cost_str


async def process_document_with_unified_pipeline(
    working_dir: str,
    document_id: str,
    content: str,
    title: str,
    metadata: Dict[str, Any],
    features: PipelineFeatures
) -> Dict[str, Any]:
    """
    Process document using BiGRAG which handles full graph persistence.

    CRITICAL: Uses BiGRAG.ainsert() instead of UnifiedPipeline directly because:
    - BiGRAG.ainsert() handles extraction + graph building + storage
    - UnifiedPipeline only extracts entities/relations (no persistence)
    - Must persist to: GraphML, vector DBs, KV stores (7 files total)

    Args:
        working_dir: Absolute path to dataset directory (from ensure_dataset_exists)
        document_id: Document ID
        content: Document content
        title: Document title
        metadata: Document metadata
        features: Pipeline feature configuration

    Returns:
        Processing result dict
    """
    from bigrag import BiGRAG
    from bigrag.llm import gpt_4o_mini_complete
    from bigrag.config import config

    try:
        logger.info(f"[UnifiedIndexing] Processing {document_id} with unified pipeline")
        logger.info(f"[UnifiedIndexing] Features: {_summarize_features(features)}")

        # Initialize BiGRAG with unified pipeline features
        # Working dir is already absolute path from ensure_dataset_exists()
        # This follows the same pattern as /datasets/create-and-index endpoint
        rag = BiGRAG(
            working_dir=working_dir,
            llm_model_func=gpt_4o_mini_complete,
            chunk_token_size=config.chunk_size,
            chunk_overlap_token_size=config.chunk_overlap_size,
            enable_llm_cache=config.enable_llm_cache,
            addon_params={"language": config.default_language},
            pipeline_features=features  # Custom features from endpoint
        )

        # Prepare metadata
        doc_metadata = {
            "title": title,
            "document_id": document_id,
            **metadata
        }

        # Insert document (handles: extraction + graph building + storage)
        # This will:
        # 1. Call UnifiedPipeline.process_document()
        # 2. Remap chunk IDs to hash-based IDs
        # 3. Call build_bipartite_graph_from_pipeline()
        # 4. Store chunks to text_chunks KV store
        # 5. Store entities to vdb_entities vector DB
        # 6. Store relations to vdb_relations vector DB
        # 7. Save GraphML file
        # FIXED: Use 'metadata' not 'metadatas' (singular, accepts list internally)
        await rag.ainsert([content], metadata=[doc_metadata])

        logger.info(f"[UnifiedIndexing] Document indexed successfully")
        logger.info(f"  - Graph files updated in: {working_dir}")
        logger.info(f"  - Knowledge graph ready for queries")

        return {
            "success": True,
            "message": "Document indexed successfully with full graph persistence",
            "dataset": dataset_name,
            "document_id": document_id,
            "working_dir": working_dir
        }

    except Exception as e:
        logger.error(f"[UnifiedIndexing] Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def _summarize_features(features: PipelineFeatures) -> str:
    """Summarize enabled features for logging."""
    enabled = []
    if features.enable_table_detection:
        enabled.append("table_detection")
    if features.enable_gleaning:
        enabled.append(f"gleaning(x{features.max_gleaning_iterations})")
    if features.enable_table_fact_extraction:
        enabled.append("table_facts")
    if features.enable_numeric_validation:
        enabled.append("numeric_validation")
    if features.enable_entity_validation:
        enabled.append("entity_validation")
    if features.enable_relation_validation:
        enabled.append("relation_validation")
    if features.merge_strategy == "fuzzy":
        enabled.append("fuzzy_merging")
    if features.enable_hitl:
        enabled.append("hitl")

    return ", ".join(enabled) if enabled else "basic"


# ============================================================================
# Endpoint
# ============================================================================

@router.post("/index-document", response_model=UnifiedIndexingResponse)
async def index_document_with_features(
    background_tasks: BackgroundTasks,
    # Required fields
    file: UploadFile = File(..., description="Document to index (.txt or .md)"),
    data_source: str = Form(..., description="Dataset name"),

    # Optional metadata
    title: str = Form(None, description="Document title (defaults to filename)"),
    metadata: str = Form(None, description="JSON metadata"),
    process_async: bool = Form(True, description="Process in background"),

    # Chunking features
    need_table_extraction: bool = Form(False, description="Extract tables using GPT-4"),
    need_dynamic_chunking: bool = Form(False, description="Use semantic chunking (vs fixed token)"),

    # Extraction features
    need_gleaning: bool = Form(False, description="Multi-pass extraction for better recall"),
    gleaning_iterations: int = Form(2, description="Number of gleaning passes (if enabled)"),
    need_table_fact_extraction: bool = Form(False, description="Rule-based table fact extraction (0% hallucination)"),
    extraction_concurrency: int = Form(16, description="Parallel LLM API calls"),

    # Validation features
    need_numeric_validation: bool = Form(False, description="Validate numeric accuracy with Gemini"),
    need_semantic_validation: bool = Form(False, description="Validate entity/relation quality"),
    validation_strictness: str = Form("MODERATE", description="STRICT | MODERATE | LENIENT"),

    # Merging features
    merge_strategy: str = Form("basic", description="basic (fast) | fuzzy (accurate)"),

    # Quality features
    enable_hitl: bool = Form(False, description="Save failed extractions for human review"),
    enable_orphan_linking: bool = Form(False, description="Detect orphan entities"),
    enable_quality_scoring: bool = Form(False, description="Track extraction quality metrics")
):
    """
    **Unified Indexing Endpoint with Explicit Feature Flags**

    This endpoint gives you complete control over the indexing pipeline by allowing
    you to enable/disable individual features instead of using presets.

    **Feature Categories:**

    1. **Chunking Features:**
       - `need_table_extraction`: Extract tables with GPT-4 (adds ~30s, $0.05)
       - `need_dynamic_chunking`: Semantic vs token-based chunking (auto-optimized)

    2. **Extraction Features:**
       - `need_gleaning`: Multi-pass extraction (adds ~60s/pass, 0.10/pass)
       - `gleaning_iterations`: How many passes (default: 2)
       - `need_table_fact_extraction`: Rule-based table extraction (0% hallucination)
       - `extraction_concurrency`: Parallel LLM calls (default: 16)

    3. **Validation Features:**
       - `need_numeric_validation`: Gemini-based numeric accuracy check
       - `need_semantic_validation`: Entity/relation quality filtering
       - `validation_strictness`: STRICT (99%) | MODERATE (95%) | LENIENT (80%)

    4. **Merging Features:**
       - `merge_strategy`: basic (fast, O(n)) | fuzzy (accurate, O(n²))

    5. **Quality Features:**
       - `enable_hitl`: Human-in-the-loop for failed extractions
       - `enable_orphan_linking`: Detect entities without relations
       - `enable_quality_scoring`: Track extraction metrics

    **Quick Recipes:**

    **Recipe 1: Fast & Cheap (like Standard Preset)**
    ```bash
    curl -X POST "http://localhost:8001/indexing/index-document" \\
      -F "file=@document.md" \\
      -F "data_source=my_dataset" \\
      -F "title=My Document" \\
      # All features default to false/basic
    ```

    **Recipe 2: High Quality (like Quality Preset)**
    ```bash
    curl -X POST "http://localhost:8001/indexing/index-document" \\
      -F "file=@KUET_Admission.md" \\
      -F "data_source=kuet" \\
      -F "title=KUET Admission Info" \\
      -F "need_table_extraction=true" \\
      -F "need_dynamic_chunking=true" \\
      -F "need_gleaning=true" \\
      -F "gleaning_iterations=2" \\
      -F "need_table_fact_extraction=true" \\
      -F "need_numeric_validation=true" \\
      -F "need_semantic_validation=true" \\
      -F "merge_strategy=fuzzy" \\
      -F "enable_hitl=true"
    ```

    **Recipe 3: Custom Mix**
    ```bash
    curl -X POST "http://localhost:8001/indexing/index-document" \\
      -F "file=@report.md" \\
      -F "data_source=reports" \\
      -F "need_table_extraction=true" \\
      -F "need_numeric_validation=true" \\
      # Tables + numeric validation only, skip gleaning for speed
    ```

    **Response:**
    - `job_id`: Track with /jobs/status/{job_id}
    - `features_enabled`: Summary of what's active
    - `estimated_time`: Expected processing time
    - `estimated_cost`: Estimated API cost

    **Requirements:**
    - OPENAI_API_KEY in .env (required)
    - GEMINI_API_KEY in .env (if need_numeric_validation=true)
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

        # Process Markdown
        if file.filename.endswith('.md'):
            content_text = process_markdown(content_text)

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="Empty file")

        # Step 3: Parse metadata
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON metadata")

        # Step 4: Build feature configuration
        features = PipelineFeatures(
            # Chunking - use fixed safe values (not exposed to users)
            enable_table_detection=need_table_extraction,
            chunk_mode="semantic" if need_dynamic_chunking else "token",
            chunk_size=1200,      # Fixed safe default
            chunk_overlap=100,    # Fixed safe default

            # Extraction
            enable_gleaning=need_gleaning,
            max_gleaning_iterations=gleaning_iterations if need_gleaning else 1,
            enable_table_fact_extraction=need_table_fact_extraction,
            extraction_concurrency=extraction_concurrency,

            # Validation
            enable_numeric_validation=need_numeric_validation,
            enable_entity_validation=need_semantic_validation,
            enable_relation_validation=need_semantic_validation,
            validation_strictness=validation_strictness,

            # Merging
            merge_strategy=merge_strategy,

            # Quality
            enable_hitl=enable_hitl,
            enable_orphan_linking=enable_orphan_linking,
            enable_quality_scoring=enable_quality_scoring,

            # API keys from environment
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            gemini_api_key=os.getenv('GEMINI_API_KEY')
        )

        # Validate feature configuration
        warnings = features.validate()
        if warnings:
            logger.warning(f"[UnifiedIndexing] Feature warnings: {warnings}")

        # Step 5: Ensure dataset exists
        dataset_info = await ensure_dataset_exists(data_source)

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

        # Step 8: Add to document registry
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
        estimated_time, estimated_cost = estimate_time_and_cost(features, len(content_text))

        # Step 10: Process (sync or async)
        if process_async:
            # Create job (using correct ProcessingJob parameters)
            job = ProcessingJob(
                job_id=job_id,
                dataset=data_source,  # FIXED: 'dataset' not 'dataset_name'
                document_id=doc_id,
                status=JobStatus.PENDING,
                stage=ProcessingStage.QUEUED
            )
            # Store additional metadata in stats field
            job.stats = {
                "filename": file.filename,
                "title": doc_title,
                "content_length": len(content_text),
                "metadata": doc_metadata,
                "features_enabled": _summarize_features(features)
            }
            processing_jobs[job_id] = job

            # Schedule background processing
            background_tasks.add_task(
                process_document_with_unified_pipeline,
                dataset_info["expr_dir"],  # Pass absolute path from ensure_dataset_exists()
                doc_id,
                content_text,
                doc_title,
                doc_metadata,
                features
            )

            status = "processing"
            logger.info(f"[UnifiedIndexing] Queued job: {job_id}")
        else:
            # Process synchronously
            result = await process_document_with_unified_pipeline(
                dataset_info["expr_dir"],  # Pass absolute path from ensure_dataset_exists()
                doc_id,
                content_text,
                doc_title,
                doc_metadata,
                features
            )
            status = "completed"
            logger.info(f"[UnifiedIndexing] Completed synchronous processing")

        # Step 11: Build response
        return UnifiedIndexingResponse(
            success=True,
            message=f"Document {'queued for processing' if process_async else 'processed successfully'}",
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
                "table_extraction": need_table_extraction,
                "dynamic_chunking": need_dynamic_chunking,
                "gleaning": need_gleaning,
                "gleaning_iterations": gleaning_iterations if need_gleaning else 0,
                "table_fact_extraction": need_table_fact_extraction,
                "numeric_validation": need_numeric_validation,
                "semantic_validation": need_semantic_validation,
                "merge_strategy": merge_strategy,
                "hitl": enable_hitl,
                "orphan_linking": enable_orphan_linking
            },
            estimated_time=estimated_time,
            estimated_cost=estimated_cost
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UnifiedIndexing] Endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
