"""
Job Queue Management System

Handles background document processing with status tracking
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStage(str, Enum):
    """Document processing stages"""
    QUEUED = "queued"
    CHUNKING = "chunking"
    EXTRACTING = "extracting_entities"
    EXTRACTING_RELATIONS = "extracting_relations"
    GRAPH_BUILDING = "graph_building"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


# Progress mapping by stage (0.0 to 1.0)
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

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "dataset": self.dataset,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "progress": self.progress,
            "stage": self.stage.value if isinstance(self.stage, ProcessingStage) else self.stage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "batch_id": self.batch_id,
            "stats": self.stats
        }


# Global job storage (in-memory)
# For production, use Redis or database
processing_jobs: Dict[str, ProcessingJob] = {}
batch_info: Dict[str, Dict] = {}


async def process_document_background(
    job_id: str,
    content: str,
    title: str,
    dataset: str,
    rag_instance,
    registry_instance,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Background task for document processing

    Updates job status throughout processing

    Args:
        job_id: Job identifier
        content: Document content (plain text)
        title: Document title
        dataset: Dataset/data_source name
        rag_instance: BiGRAG instance (with pipeline mode pre-configured)
        registry_instance: DocumentRegistry instance
        metadata: Optional document metadata (Phase 1: metadata preservation)

    Note:
        Pipeline mode (standard/enhanced) is configured during BiGRAG initialization.
        No need to override pipeline settings here.
    """
    job = processing_jobs.get(job_id)

    if not job:
        logger.error(f"Job {job_id} not found in processing_jobs")
        return

    try:
        # Start processing
        job.update(
            status=JobStatus.PROCESSING,
            started_at=datetime.now(),
            stage=ProcessingStage.CHUNKING,
            progress=STAGE_PROGRESS[ProcessingStage.CHUNKING]
        )

        logger.info(f"[Job {job_id}] Starting processing for document: {title}")

        # Detect pipeline mode from RAG instance
        pipeline_mode = "ENHANCED (Phase 1)" if getattr(rag_instance, 'use_enhanced_pipeline', False) else "STANDARD"
        logger.info(f"[Job {job_id}] Pipeline mode: {pipeline_mode}")

        # Update to extraction stage
        job.update(
            stage=ProcessingStage.EXTRACTING,
            progress=STAGE_PROGRESS[ProcessingStage.EXTRACTING]
        )

        # Process document with BiGRAG
        # This handles: chunking, entity extraction, graph building, embedding, indexing
        # Phase 1: Pass metadata to improve entity extraction (+2-3 F1 points)
        doc_metadata = metadata or {}
        if title and "title" not in doc_metadata:
            doc_metadata["title"] = title

        # Process with pre-configured pipeline (no override needed)
        await rag_instance.ainsert(content, metadata=doc_metadata)

        # Update progress through remaining stages
        job.update(
            stage=ProcessingStage.EXTRACTING_RELATIONS,
            progress=STAGE_PROGRESS[ProcessingStage.EXTRACTING_RELATIONS]
        )

        await asyncio.sleep(0.1)  # Small delay for status updates

        job.update(
            stage=ProcessingStage.GRAPH_BUILDING,
            progress=STAGE_PROGRESS[ProcessingStage.GRAPH_BUILDING]
        )

        await asyncio.sleep(0.1)

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

        job.update(
            stage=ProcessingStage.FINALIZING,
            progress=STAGE_PROGRESS[ProcessingStage.FINALIZING]
        )

        # Get stats from KG (import here to avoid circular dependency)
        try:
            from .kg_utils import get_document_stats_from_kg
            stats = await get_document_stats_from_kg(dataset, job.document_id)
        except Exception as e:
            logger.warning(f"Could not get document stats: {e}")
            stats = {}

        # Complete
        job.update(
            status=JobStatus.COMPLETED,
            stage=ProcessingStage.COMPLETED,
            progress=1.0,
            completed_at=datetime.now(),
            stats=stats
        )

        # Update registry
        try:
            await registry_instance.update_document(
                job.document_id,
                status="indexed",
                indexed_date=datetime.now().isoformat(),
                stats=stats
            )
        except Exception as e:
            logger.warning(f"Could not update registry: {e}")

        logger.info(f"[Job {job_id}] Completed successfully")

    except Exception as e:
        # Handle failure
        logger.error(f"[Job {job_id}] Failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        job.update(
            status=JobStatus.FAILED,
            completed_at=datetime.now(),
            error=str(e)
        )

        # Update registry
        try:
            await registry_instance.update_document(
                job.document_id,
                status="failed",
                error=str(e)
            )
        except Exception as reg_error:
            logger.error(f"Could not update registry after failure: {reg_error}")


def get_queue_stats() -> Dict[str, int]:
    """Get statistics about the processing queue"""
    stats = {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "total": len(processing_jobs)
    }

    for job in processing_jobs.values():
        status = job.status if isinstance(job.status, str) else job.status.value
        stats[status] = stats.get(status, 0) + 1

    return stats


def cleanup_old_jobs(max_age_hours: int = 24):
    """
    Remove completed/failed jobs older than max_age_hours

    Args:
        max_age_hours: Maximum age in hours for completed jobs
    """
    from datetime import timedelta

    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    jobs_to_remove = []

    for job_id, job in processing_jobs.items():
        if job.completed_at and job.completed_at < cutoff_time:
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                jobs_to_remove.append(job_id)

    for job_id in jobs_to_remove:
        del processing_jobs[job_id]

    logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    return len(jobs_to_remove)
