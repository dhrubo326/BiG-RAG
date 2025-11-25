"""
HITL (Human-in-the-Loop) API Routes

Provides REST API endpoints for managing failed extractions:
- View failed extractions
- Submit corrections
- Reprocess corrected data
- Get statistics

Part of Phase 1 Step 6: HITL System
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, List
from pydantic import BaseModel
from pathlib import Path

from bigrag.hitl.failed_extraction_store import FailedExtractionStore


router = APIRouter(prefix="/hitl", tags=["Human-in-the-Loop"])


# Request/Response Models

class CorrectionSubmission(BaseModel):
    """Corrected extraction data submitted by human reviewer."""
    corrected_entities: Optional[List[Dict]] = None
    corrected_relations: Optional[List[Dict]] = None
    corrected_table_data: Optional[Dict] = None
    reviewer_notes: Optional[str] = None


class ReprocessRequest(BaseModel):
    """Request to reprocess corrected extraction."""
    merge_with_existing: bool = True  # Merge with existing graph or replace


# Helper function to get store

def get_store(dataset_name: str) -> FailedExtractionStore:
    """Get HITL store for dataset."""
    dataset_path = Path("expr") / dataset_name

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found at {dataset_path}"
        )

    return FailedExtractionStore(str(dataset_path))


# API Endpoints

@router.get("/failed-extractions/{dataset_name}")
async def get_failed_extractions(
    dataset_name: str,
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    extraction_type: Optional[str] = Query(None, description="Filter by type (chunk/table)"),
    status: Optional[str] = Query(None, description="Filter by status (pending_review/reviewed/corrected)")
):
    """
    Get failed extractions for human review.

    Query Parameters:
        - document_id: Filter by specific document (optional)
        - extraction_type: Filter by 'chunk' or 'table' (optional)
        - status: Filter by status (optional)

    Returns:
        List of failed extraction records with full details
    """
    try:
        store = get_store(dataset_name)
        failures = store.get_failed_extractions(
            document_id=document_id,
            extraction_type=extraction_type,
            status=status
        )

        return {
            "dataset": dataset_name,
            "filters": {
                "document_id": document_id,
                "extraction_type": extraction_type,
                "status": status
            },
            "total_failures": len(failures),
            "failures": failures
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/review-queue/{dataset_name}")
async def get_review_queue(dataset_name: str):
    """
    Get pending review queue for dataset.

    Returns only extractions with status='pending_review'.
    """
    try:
        store = get_store(dataset_name)
        queue = store.get_review_queue()

        return {
            "dataset": dataset_name,
            "pending_count": len(queue),
            "review_queue": queue
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/{dataset_name}")
async def get_statistics(dataset_name: str):
    """
    Get statistics about failed extractions.

    Returns counts by type, status, and document.
    """
    try:
        store = get_store(dataset_name)
        stats = store.get_statistics()

        return {
            "dataset": dataset_name,
            "statistics": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extraction/{dataset_name}/{extraction_id}")
async def get_extraction_details(dataset_name: str, extraction_id: str):
    """
    Get detailed information about a specific extraction.

    Returns full extraction record including content and validation details.
    """
    try:
        store = get_store(dataset_name)
        all_failures = store.get_failed_extractions()

        # Find matching extraction
        extraction = next(
            (f for f in all_failures if f["extraction_id"] == extraction_id),
            None
        )

        if not extraction:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction '{extraction_id}' not found"
            )

        return {
            "dataset": dataset_name,
            "extraction": extraction
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mark-reviewed/{dataset_name}/{extraction_id}")
async def mark_reviewed(
    dataset_name: str,
    extraction_id: str,
    correction: Optional[CorrectionSubmission] = None
):
    """
    Mark extraction as human-reviewed.

    Optionally provide corrected data. Status will be set to 'reviewed'.
    """
    try:
        store = get_store(dataset_name)

        corrected_data = None
        if correction:
            corrected_data = {
                "entities": correction.corrected_entities,
                "relations": correction.corrected_relations,
                "table_data": correction.corrected_table_data
            }

        success = store.mark_reviewed(
            extraction_id=extraction_id,
            corrected_data=corrected_data,
            reviewer_notes=correction.reviewer_notes if correction else None
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction '{extraction_id}' not found"
            )

        return {
            "status": "success",
            "message": "Extraction marked as reviewed",
            "extraction_id": extraction_id,
            "has_corrections": corrected_data is not None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit-correction/{dataset_name}/{extraction_id}")
async def submit_correction(
    dataset_name: str,
    extraction_id: str,
    correction: CorrectionSubmission
):
    """
    Submit human correction for extraction.

    Marks extraction as 'corrected' and ready for reprocessing.
    """
    try:
        store = get_store(dataset_name)

        corrected_data = {
            "entities": correction.corrected_entities,
            "relations": correction.corrected_relations,
            "table_data": correction.corrected_table_data
        }

        success = store.mark_corrected(
            extraction_id=extraction_id,
            corrected_data=corrected_data,
            correction_notes=correction.reviewer_notes
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction '{extraction_id}' not found"
            )

        return {
            "status": "success",
            "message": "Correction saved, ready for reprocessing",
            "extraction_id": extraction_id,
            "next_step": f"POST /hitl/reprocess/{dataset_name}/{extraction_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reprocess/{dataset_name}/{extraction_id}")
async def reprocess_extraction(
    dataset_name: str,
    extraction_id: str,
    request: Optional[ReprocessRequest] = None
):
    """
    Reprocess corrected extraction into knowledge graph.

    This is a placeholder - actual reprocessing logic would depend on
    the pipeline implementation. In practice, this would:
    1. Load corrected data from HITL store
    2. Insert entities/relations into graph
    3. Update vector indices
    4. Mark extraction as 'reprocessed'
    """
    try:
        store = get_store(dataset_name)

        # Get extraction
        all_failures = store.get_failed_extractions()
        extraction = next(
            (f for f in all_failures if f["extraction_id"] == extraction_id),
            None
        )

        if not extraction:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction '{extraction_id}' not found"
            )

        if extraction["status"] != "corrected":
            raise HTTPException(
                status_code=400,
                detail=f"Extraction must be corrected before reprocessing (current status: {extraction['status']})"
            )

        # TODO: Actual reprocessing logic
        # This would involve:
        # 1. Load corrected_data from extraction
        # 2. Insert into graph (bigrag.chunk_entity_relation_graph)
        # 3. Update vector DBs (bigrag.vdb_entities, bigrag.vdb_relations)
        # 4. Update KV stores

        # For now, just mark as reprocessed
        store._update_status_in_file(
            store.chunks_file if extraction["type"] == "chunk" else store.tables_file,
            extraction_id,
            "reprocessed",
            None,
            "Reprocessed via API"
        )

        return {
            "status": "success",
            "message": "Extraction reprocessed successfully",
            "extraction_id": extraction_id,
            "note": "Actual reprocessing logic needs to be implemented based on pipeline"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/extraction/{dataset_name}/{extraction_id}")
async def delete_extraction(dataset_name: str, extraction_id: str):
    """
    Delete extraction record (use with caution).

    This permanently removes the extraction from the failed extractions store.
    """
    try:
        store = get_store(dataset_name)

        success = store.delete_extraction(extraction_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction '{extraction_id}' not found"
            )

        return {
            "status": "success",
            "message": "Extraction deleted permanently",
            "extraction_id": extraction_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/{dataset_name}")
async def hitl_health_check(dataset_name: str):
    """
    Health check for HITL system.

    Returns status and basic information about the HITL store.
    """
    try:
        store = get_store(dataset_name)
        stats = store.get_statistics()

        return {
            "status": "healthy",
            "dataset": dataset_name,
            "store_path": str(store.base_path),
            "total_failures": stats["total_failures"],
            "pending_review": stats["by_status"]["pending_review"]
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "dataset": dataset_name,
            "error": str(e)
        }
