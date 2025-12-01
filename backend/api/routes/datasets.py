"""
Dataset Management Routes (Production-Ready)

Provides endpoints for dynamic dataset creation and indexing in unified mode.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field

from bigrag.utils import logger
from ..core.dependencies import get_unified_executor
from ..models.models import UploadResponse as EnhancedUploadResponse
from ..services.jobs import ProcessingJob, JobStatus, ProcessingStage, processing_jobs, process_document_background
from ..services.registry import registry
from ..services.utils import process_markdown, validate_file_upload, truncate_text
from ..services.kg_utils import compute_doc_id, add_document_to_corpus


router = APIRouter(prefix="/datasets", tags=["Dataset Management (Production)"])


# ============================================================================
# Request/Response Models
# ============================================================================

class DatasetCreateResponse(BaseModel):
    """Response for dataset creation"""
    success: bool
    message: str
    dataset_name: str
    registry_updated: bool
    corpus_created: bool
    graph_directory: str


class DatasetIndexResponse(BaseModel):
    """Response for document indexing"""
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
    pipeline_mode: str = "production"


# ============================================================================
# Helper Functions
# ============================================================================

async def ensure_dataset_exists(dataset_name: str) -> Dict[str, Any]:
    """
    Ensure dataset directory structure exists and is registered in subgraph registry.

    Creates:
    - expr/{dataset_name}/
    - datasets/{dataset_name}/raw/
    - Adds to expr/subgraph_registry.json if not exists

    Returns:
        Dict with creation status
    """
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

    # Create directories
    expr_dir = PROJECT_ROOT / "expr" / dataset_name
    dataset_dir = PROJECT_ROOT / "datasets" / dataset_name / "raw"

    expr_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Dataset] Created directories for: {dataset_name}")

    # Update subgraph registry
    registry_path = PROJECT_ROOT / "expr" / "subgraph_registry.json"
    registry_updated = False

    try:
        # Load existing registry
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
        else:
            # Create new registry
            registry_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "description": "Registry of all available subgraphs for unified query routing",
                "subgraphs": {},
                "routing_config": {
                    "default_strategy": "llm_based",
                    "fallback_subgraph": "demo_test",
                    "max_subgraphs_per_query": 3,
                    "enable_parallel_search": True
                }
            }

        # Check if dataset already registered
        if dataset_name not in registry_data["subgraphs"]:
            # Add new subgraph entry with default metadata
            registry_data["subgraphs"][dataset_name] = {
                "path": f"expr/{dataset_name}",
                "description": f"Auto-created dataset: {dataset_name}",
                "aliases": [dataset_name],
                "topics": [dataset_name, "auto-created"],
                "enabled": True,
                "created_at": datetime.now().isoformat(),
                "auto_created": True
            }

            # Save updated registry
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)

            registry_updated = True
            logger.info(f"[Dataset] Added {dataset_name} to subgraph registry")
        else:
            logger.info(f"[Dataset] {dataset_name} already exists in registry")

    except Exception as e:
        logger.error(f"[Dataset] Failed to update registry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update subgraph registry: {e}")

    return {
        "expr_dir": str(expr_dir),
        "dataset_dir": str(dataset_dir),
        "registry_updated": registry_updated,
        "exists": True
    }


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/create-and-index", response_model=DatasetIndexResponse)
async def create_and_index_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document to index (.txt or .md)"),
    data_source: str = Form(..., description="Dataset name (will be created if doesn't exist)"),

    # Preset-based configuration (SIMPLE - 90% of users)
    preset: str = Form("balanced", description="Indexing preset: fast | balanced | quality"),

    # Advanced overrides (OPTIONAL - 10% of users)
    chunker: str = Form(None, description="Override chunker: token | semantic | hybrid"),
    extractor: str = Form(None, description="Override extractor: strict | gleaning | hybrid"),
    validators: str = Form(None, description="Override validators (comma-separated): numeric,semantic"),
    merger: str = Form(None, description="Override merger: basic | fuzzy | hybrid"),
    validation_strictness: str = Form(None, description="Override strictness: STRICT | MODERATE | LENIENT"),

    # Metadata
    title: str = Form(None, description="Document title (defaults to filename)"),
    metadata: str = Form(None, description="Optional JSON metadata"),
    process_async: bool = Form(True, description="Process in background (recommended)"),
    language: str = Form(None, description="Language for entity extraction (auto-detected if not specified)")
):
    """
    **Production-Ready Endpoint: Create Dataset & Index Document with Presets**

    Creates new datasets dynamically and indexes documents with flexible preset-based configuration.
    Perfect for multi-subgraph management where different subgraphs have different quality/cost needs.

    **Architecture**: Uses NEW modular indexing system (IndexingConfig + BiGRAG).

    **Features:**
    - ✅ Creates new dataset if doesn't exist
    - ✅ Auto-registers to subgraph registry (unified mode)
    - ✅ Saves to corpus.jsonl for persistence
    - ✅ **Preset-based configuration** (fast/balanced/quality)
    - ✅ **Optional overrides** for fine-tuned control
    - ✅ Requires OPENAI_API_KEY in .env

    **Presets:**
    - **fast**: Token chunking, strict extraction, no validation (~$0.60/10K docs, 2-3 min, 90-95% accuracy)
    - **balanced**: Semantic chunking, gleaning, semantic validation (~$2.50/10K docs, 10-15 min, 92-96% accuracy) **[DEFAULT]**
    - **quality**: Hybrid extraction, full validation, fuzzy merging (~$4-6/10K docs, 20-30 min, 95-99% accuracy)

    **SIMPLE USAGE (90% of cases):**
    ```bash
    # Fast indexing (blog posts, simple text)
    curl -X POST "http://localhost:8001/datasets/create-and-index" \\
      -F "file=@blog.md" \\
      -F "data_source=blogs" \\
      -F "preset=fast"

    # Balanced (recommended default)
    curl -X POST "http://localhost:8001/datasets/create-and-index" \\
      -F "file=@doc.md" \\
      -F "data_source=my_dataset" \\
      -F "preset=balanced"

    # Quality (educational content with tables)
    curl -X POST "http://localhost:8001/datasets/create-and-index" \\
      -F "file=@kuet_admission.md" \\
      -F "data_source=kuet_test" \\
      -F "preset=quality"
    ```

    **ADVANCED USAGE (10% of cases):**
    ```bash
    # Start with balanced, disable validation (save cost)
    curl -F "preset=balanced" -F "validators=" -F "file=@doc.md" ...

    # Start with fast, enable tables
    curl -F "preset=fast" -F "chunker=semantic" -F "file=@doc.md" ...

    # Fully custom (override all)
    curl -F "chunker=hybrid" -F "extractor=gleaning" -F "validators=numeric,semantic" ...
    ```

    **Overrides:**
    - `chunker`: Override chunking strategy (token | semantic | hybrid)
    - `extractor`: Override extraction strategy (strict | gleaning | hybrid)
    - `validators`: Override validators (comma-separated: numeric,semantic or empty for none)
    - `merger`: Override merger strategy (basic | fuzzy | hybrid)
    - `validation_strictness`: Override strictness (STRICT | MODERATE | LENIENT)

    **Requirements:**
    - OPENAI_API_KEY must be set in .env
    - Server must run in unified mode (--unified flag)
    - For numeric validation: GEMINI_API_KEY required
    """
    try:
        # Step 1: Verify unified mode
        unified_executor = get_unified_executor()
        if not unified_executor:
            raise HTTPException(
                status_code=503,
                detail="This endpoint requires unified mode. Start server with --unified flag."
            )

        # Step 2: Validate file
        if not file.filename.endswith(('.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .md files are supported"
            )

        # Step 3: Read and validate content
        content_bytes = await file.read()
        is_valid, error_msg = validate_file_upload(content_bytes, file.filename, max_size_mb=50)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        content_text = content_bytes.decode('utf-8')

        # Process Markdown
        if file.filename.endswith('.md'):
            logger.info(f"Processing Markdown file: {file.filename}")
            content_text = process_markdown(content_text)

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="File is empty after processing")

        # Step 4: Parse metadata
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata field")

        # Step 5: Ensure dataset exists and is registered
        logger.info(f"[Create-and-Index] Ensuring dataset exists: {data_source}")
        dataset_info = await ensure_dataset_exists(data_source)

        # Step 6: Generate IDs
        doc_id = compute_doc_id(content_text, prefix="doc")
        job_id = f"job-{compute_doc_id(doc_id + str(datetime.now()), prefix='')}"
        doc_title = title or file.filename

        # Step 7: Add to corpus.jsonl
        await add_document_to_corpus(
            data_source=data_source,
            doc_id=doc_id,
            content=content_text,
            title=doc_title,
            metadata=doc_metadata
        )
        logger.info(f"[Create-and-Index] Added document to corpus: {doc_id}")

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

        # Step 9: Build IndexingConfig from preset + overrides (NEW: 16-feature system)
        import os
        from bigrag import BiGRAG
        from bigrag.config import IndexingConfig

        working_dir = dataset_info["expr_dir"]
        logger.info(f"[Create-and-Index] Building config for: {data_source} (preset={preset})")

        # Build base config from preset
        if preset == "fast":
            indexing_config = IndexingConfig.preset_fast(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                gemini_api_key=os.getenv('GEMINI_API_KEY'),
                dataset_path=working_dir
            )
        elif preset == "quality":
            indexing_config = IndexingConfig.preset_quality(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                gemini_api_key=os.getenv('GEMINI_API_KEY'),
                dataset_path=working_dir
            )
        else:  # "balanced" (default)
            indexing_config = IndexingConfig.preset_balanced(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                gemini_api_key=os.getenv('GEMINI_API_KEY'),
                dataset_path=working_dir
            )

        # Apply overrides (if provided) - Map old names to new 16-feature system
        if chunker:
            # OLD: chunker="semantic" → NEW: chunking_strategy="semantic"
            indexing_config.chunking_strategy = chunker
            logger.info(f"[Create-and-Index] Override: chunking_strategy={chunker}")
        if extractor:
            # OLD: extractor="gleaning" → NEW: extraction_strategy="gleaning"
            indexing_config.extraction_strategy = extractor
            logger.info(f"[Create-and-Index] Override: extraction_strategy={extractor}")
        if validators is not None:  # Allow empty string to disable
            # OLD: validators=["numeric","semantic"] → NEW: enable_numeric_validation, enable_entity_validation
            validator_list = validators.split(',') if validators else []
            indexing_config.enable_numeric_validation = 'numeric' in validator_list
            indexing_config.enable_entity_validation = 'semantic' in validator_list or 'entity' in validator_list
            indexing_config.enable_relation_validation = 'semantic' in validator_list or 'relation' in validator_list
            logger.info(f"[Create-and-Index] Override: validators={validator_list}")
        if merger:
            # OLD: merger="fuzzy" → NEW: enable_fuzzy_matching=True
            indexing_config.enable_fuzzy_matching = (merger in ['fuzzy', 'hybrid'])
            logger.info(f"[Create-and-Index] Override: merger={merger}")
        if validation_strictness:
            indexing_config.validation_strictness = validation_strictness
            logger.info(f"[Create-and-Index] Override: validation_strictness={validation_strictness}")

        # Create RAG instance with modular system
        rag = BiGRAG(
            working_dir=working_dir,
            indexing_config=indexing_config
        )
        logger.info(f"[Create-and-Index] Initialized (16 features): chunking={indexing_config.chunking_strategy}, "
                   f"extraction={indexing_config.extraction_strategy}, "
                   f"numeric_val={indexing_config.enable_numeric_validation}")

        # Step 10: Create processing job
        job = ProcessingJob(
            job_id=job_id,
            document_id=doc_id,
            dataset=data_source,
            status=JobStatus.PENDING,
            progress=0.0,
            stage=ProcessingStage.QUEUED
        )
        processing_jobs[job_id] = job

        # Step 11: Process with MODULAR PIPELINE (Preset-based)
        # Pipeline configuration built from preset + overrides above
        if process_async:
            background_tasks.add_task(
                process_document_background,
                job_id=job_id,
                content=content_text,
                title=doc_title,
                dataset=data_source,
                rag_instance=rag,
                registry_instance=registry,
                metadata=doc_metadata,
                language=language
            )
            message = f"Document queued for indexing in dataset '{data_source}' (preset: {preset}, chunking: {indexing_config.chunking_strategy}, extraction: {indexing_config.extraction_strategy}, language: {language})"
        else:
            await process_document_background(
                job_id=job_id,
                content=content_text,
                title=doc_title,
                dataset=data_source,
                rag_instance=rag,
                registry_instance=registry,
                metadata=doc_metadata,
                language=language
            )
            message = f"Document indexed in dataset '{data_source}' (preset: {preset}, chunking: {indexing_config.chunking_strategy}, extraction: {indexing_config.extraction_strategy}, language: {language})"

        # Step 12: Reload registry in unified executor (if dataset was just added)
        if dataset_info["registry_updated"]:
            try:
                unified_executor.reload_registry()
                logger.info(f"[Create-and-Index] Reloaded unified executor registry")

                # NEW: Pre-load the new subgraph immediately
                await unified_executor.cache.get(data_source)
                logger.info(f"[Create-and-Index] Pre-loaded new subgraph: {data_source}")

            except Exception as e:
                logger.warning(f"[Create-and-Index] Failed to reload/prewarm: {e}")

        return DatasetIndexResponse(
            success=True,
            message=message,
            dataset_name=data_source,
            document_id=doc_id,
            job_id=job_id,
            filename=file.filename,
            title=doc_title,
            content_preview=truncate_text(content_text, 200),
            content_length=len(content_text),
            status=job.status.value if isinstance(job.status, JobStatus) else job.status,
            metadata=doc_metadata,
            upload_date=datetime.now().isoformat(),
            pipeline_mode="enhanced"  # Updated from "production" to reflect Phase 1 changes
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Create-and-Index] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/create", response_model=DatasetCreateResponse)
async def create_dataset(
    dataset_name: str = Form(..., description="Name of the dataset to create"),
    description: str = Form(None, description="Optional description"),
    topics: str = Form(None, description="Comma-separated topics"),
    aliases: str = Form(None, description="Comma-separated aliases")
):
    """
    Create a new dataset without indexing documents.

    This endpoint only creates the directory structure and registers the dataset.
    Use this when you want to pre-create datasets before indexing documents.

    Example:
    ```bash
    curl -X POST "http://localhost:8001/datasets/create" \\
      -F "dataset_name=medical_kb" \\
      -F "description=Medical knowledge base" \\
      -F "topics=medicine,health,medical" \\
      -F "aliases=medical,med,health"
    ```
    """
    try:
        # Verify unified mode
        unified_executor = get_unified_executor()
        if not unified_executor:
            raise HTTPException(
                status_code=503,
                detail="This endpoint requires unified mode. Start server with --unified flag."
            )

        # Create dataset structure
        dataset_info = await ensure_dataset_exists(dataset_name)

        # Update registry with custom metadata if provided
        if description or topics or aliases:
            PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
            registry_path = PROJECT_ROOT / "expr" / "subgraph_registry.json"

            with open(registry_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)

            if dataset_name in registry_data["subgraphs"]:
                if description:
                    registry_data["subgraphs"][dataset_name]["description"] = description
                if topics:
                    registry_data["subgraphs"][dataset_name]["topics"] = [t.strip() for t in topics.split(',')]
                if aliases:
                    registry_data["subgraphs"][dataset_name]["aliases"] = [a.strip() for a in aliases.split(',')]

                with open(registry_path, 'w', encoding='utf-8') as f:
                    json.dump(registry_data, f, indent=2, ensure_ascii=False)

                logger.info(f"[Dataset] Updated metadata for {dataset_name}")

        # Reload registry
        try:
            unified_executor.reload_registry()
        except Exception as e:
            logger.warning(f"Failed to reload registry: {e}")

        # Create corpus file
        corpus_path = Path(dataset_info["dataset_dir"]) / "corpus.jsonl"
        corpus_created = False
        if not corpus_path.exists():
            corpus_path.touch()
            corpus_created = True

        return DatasetCreateResponse(
            success=True,
            message=f"Dataset '{dataset_name}' created successfully",
            dataset_name=dataset_name,
            registry_updated=dataset_info["registry_updated"],
            corpus_created=corpus_created,
            graph_directory=dataset_info["expr_dir"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Create Dataset] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Dataset creation failed: {str(e)}")
