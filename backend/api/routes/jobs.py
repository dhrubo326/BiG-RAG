"""
Job management routes
"""

from fastapi import APIRouter, HTTPException

from ..models.models import JobStatusResponse, JobStatistics
from ..services.jobs import processing_jobs


router = APIRouter(prefix="/jobs", tags=["Job Management"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get processing status for a document upload job.

    **Example usage:**
    ```bash
    curl "http://localhost:8001/jobs/job-abc123"
    ```

    **Returns:** Job status with progress (0.0 to 1.0) and current stage
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = processing_jobs[job_id]

    # Convert stats to JobStatistics model if present
    job_stats = None
    if job.stats:
        relations_count = job.stats.get("relations", job.stats.get("edges", 0))
        job_stats = JobStatistics(
            chunks_created=job.stats.get("chunks", 0),
            entities_extracted=job.stats.get("entities", 0),
            relations_created=relations_count,
            edges_created=relations_count,  # Deprecated: alias for backward compatibility
            tokens_processed=job.stats.get("tokens", 0)
        )

    return JobStatusResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        dataset=job.dataset,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error,
        stats=job_stats
    )
