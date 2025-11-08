"""
Pydantic Models for API Requests and Responses

Defines all request and response models for the BiG-RAG API endpoints
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ==============================================================================
# Enums
# ==============================================================================

class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


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


# ==============================================================================
# Request Models
# ==============================================================================

class UploadMetadata(BaseModel):
    """Optional metadata for document upload"""
    category: Optional[str] = Field(None, description="Document category")
    tags: Optional[List[str]] = Field(None, description="Document tags")
    author: Optional[str] = Field(None, description="Document author")
    source: Optional[str] = Field(None, description="Document source/origin")
    language: Optional[str] = Field("en", description="Document language")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom metadata fields")


class DocumentFilter(BaseModel):
    """Filter parameters for document listing"""
    dataset: Optional[str] = Field(None, description="Filter by dataset")
    search: Optional[str] = Field(None, description="Search in title or filename")
    category: Optional[str] = Field(None, description="Filter by category")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    status: Optional[DocumentStatus] = Field(None, description="Filter by status")
    date_from: Optional[str] = Field(None, description="Filter by upload date (ISO format)")
    date_to: Optional[str] = Field(None, description="Filter by upload date (ISO format)")
    limit: int = Field(50, ge=1, le=500, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class DeleteBatchRequest(BaseModel):
    """Request model for batch deletion"""
    document_ids: List[str] = Field(..., min_items=1, description="List of document IDs to delete")
    hard_delete: bool = Field(False, description="Permanently delete from KG (requires rebuild)")


# ==============================================================================
# Response Models - Upload Endpoints
# ==============================================================================

class UploadResponse(BaseModel):
    """Response for single document upload"""
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    document_id: str = Field(..., description="Unique document identifier")
    job_id: str = Field(..., description="Job ID for tracking processing status")
    filename: str = Field(..., description="Original filename")
    title: str = Field(..., description="Document title")
    content_preview: str = Field(..., description="First 200 characters of content")
    content_length: int = Field(..., description="Content length in characters")
    dataset: str = Field(..., description="Dataset/data_source name")
    status: str = Field(..., description="Initial processing status (usually 'pending')")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    upload_date: str = Field(..., description="Upload timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document queued for processing",
                "document_id": "upload-a1b2c3d4e5f6",
                "job_id": "job-xyz789",
                "filename": "research_paper.md",
                "title": "BiG-RAG Research Paper",
                "content_preview": "This paper introduces BiG-RAG, a novel approach to...",
                "content_length": 15420,
                "dataset": "user_uploads",
                "status": "pending",
                "metadata": {"category": "research", "tags": ["RAG", "NLP"]},
                "upload_date": "2025-10-30T10:30:00Z"
            }
        }


class BatchUploadResponse(BaseModel):
    """Response for batch document upload"""
    success: bool = Field(..., description="Whether batch upload was successful")
    message: str = Field(..., description="Status message")
    batch_id: str = Field(..., description="Batch ID for tracking all uploads")
    total_files: int = Field(..., description="Total files in batch")
    accepted_files: int = Field(..., description="Number of files accepted")
    rejected_files: int = Field(..., description="Number of files rejected")
    documents: List[UploadResponse] = Field(..., description="Individual upload responses")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Errors for rejected files")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Batch upload completed: 8/10 files accepted",
                "batch_id": "batch-abc123",
                "total_files": 10,
                "accepted_files": 8,
                "rejected_files": 2,
                "documents": [],  # Would contain UploadResponse objects
                "errors": [
                    {"filename": "invalid.pdf", "error": "Only .txt and .md files supported"},
                    {"filename": "empty.txt", "error": "File is empty"}
                ]
            }
        }


# ==============================================================================
# Response Models - Job Status
# ==============================================================================

class JobStatistics(BaseModel):
    """Statistics from completed processing job"""
    chunks_created: int = Field(0, description="Number of text chunks created")
    entities_extracted: int = Field(0, description="Number of entities extracted")
    edges_created: int = Field(0, description="Number of bipartite edges created")
    tokens_processed: int = Field(0, description="Total tokens processed")


class JobStatusResponse(BaseModel):
    """Response for job status query"""
    job_id: str = Field(..., description="Job identifier")
    document_id: str = Field(..., description="Associated document ID")
    dataset: str = Field(..., description="Dataset name")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress (0.0 to 1.0)")
    stage: ProcessingStage = Field(..., description="Current processing stage")
    started_at: Optional[str] = Field(None, description="Job start time (ISO format)")
    completed_at: Optional[str] = Field(None, description="Job completion time (ISO format)")
    error: Optional[str] = Field(None, description="Error message if failed")
    stats: Optional[JobStatistics] = Field(None, description="Processing statistics (when completed)")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-xyz789",
                "document_id": "upload-a1b2c3d4e5f6",
                "dataset": "user_uploads",
                "status": "processing",
                "progress": 0.65,
                "stage": "embedding",
                "started_at": "2025-10-30T10:30:05Z",
                "completed_at": None,
                "error": None,
                "stats": None
            }
        }


class BatchStatusResponse(BaseModel):
    """Response for batch status query"""
    batch_id: str = Field(..., description="Batch identifier")
    total_jobs: int = Field(..., description="Total jobs in batch")
    completed: int = Field(..., description="Completed jobs")
    processing: int = Field(..., description="Currently processing jobs")
    pending: int = Field(..., description="Pending jobs")
    failed: int = Field(..., description="Failed jobs")
    overall_progress: float = Field(..., ge=0.0, le=1.0, description="Overall batch progress")
    jobs: List[JobStatusResponse] = Field(..., description="Individual job statuses")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "batch-abc123",
                "total_jobs": 10,
                "completed": 7,
                "processing": 2,
                "pending": 0,
                "failed": 1,
                "overall_progress": 0.75,
                "jobs": []  # Would contain JobStatusResponse objects
            }
        }


# ==============================================================================
# Response Models - Document Management
# ==============================================================================

class DocumentSummary(BaseModel):
    """Summary information for document listing"""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    title: str = Field(..., description="Document title")
    content_length: int = Field(..., description="Content length in characters")
    upload_date: str = Field(..., description="Upload timestamp")
    indexed_date: Optional[str] = Field(None, description="Indexing completion timestamp")
    last_modified: str = Field(..., description="Last modification timestamp")
    status: DocumentStatus = Field(..., description="Document status")
    dataset: str = Field(..., description="Dataset name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    job_id: str = Field(..., description="Associated job ID")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "upload-a1b2c3d4e5f6",
                "filename": "research_paper.md",
                "title": "BiG-RAG Research Paper",
                "content_length": 15420,
                "upload_date": "2025-10-30T10:30:00Z",
                "indexed_date": "2025-10-30T10:35:00Z",
                "last_modified": "2025-10-30T10:35:00Z",
                "status": "indexed",
                "dataset": "user_uploads",
                "metadata": {"category": "research"},
                "job_id": "job-xyz789"
            }
        }


class DocumentListResponse(BaseModel):
    """Response for document listing"""
    total: int = Field(..., description="Total documents matching filter")
    limit: int = Field(..., description="Results per page")
    offset: int = Field(..., description="Current offset")
    documents: List[DocumentSummary] = Field(..., description="Document summaries")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 42,
                "limit": 50,
                "offset": 0,
                "documents": []  # Would contain DocumentSummary objects
            }
        }


class KGStatistics(BaseModel):
    """Knowledge graph statistics for document"""
    chunks: int = Field(0, description="Number of chunks")
    entities: int = Field(0, description="Number of entities")
    edges: int = Field(0, description="Number of bipartite edges")
    tokens: int = Field(0, description="Total tokens")


class EntityInfo(BaseModel):
    """Entity information"""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    weight: float = Field(..., description="Entity weight/importance")


class RelatedDocument(BaseModel):
    """Related document information"""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")


class DocumentDetailResponse(BaseModel):
    """Detailed response for single document"""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    title: str = Field(..., description="Document title")
    content_length: int = Field(..., description="Content length in characters")
    content_preview: str = Field(..., description="Content preview")
    upload_date: str = Field(..., description="Upload timestamp")
    indexed_date: Optional[str] = Field(None, description="Indexing completion timestamp")
    last_modified: str = Field(..., description="Last modification timestamp")
    status: DocumentStatus = Field(..., description="Document status")
    dataset: str = Field(..., description="Dataset name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    job_id: Optional[str] = Field(None, description="Associated job ID (if uploaded via API)")
    stats: Optional[KGStatistics] = Field(None, description="Knowledge graph statistics")
    top_entities: Optional[List[EntityInfo]] = Field(None, description="Top entities extracted")
    related_documents: Optional[List[RelatedDocument]] = Field(None, description="Related documents")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "upload-a1b2c3d4e5f6",
                "filename": "research_paper.md",
                "title": "BiG-RAG Research Paper",
                "content_length": 15420,
                "content_preview": "This paper introduces BiG-RAG...",
                "upload_date": "2025-10-30T10:30:00Z",
                "indexed_date": "2025-10-30T10:35:00Z",
                "last_modified": "2025-10-30T10:35:00Z",
                "status": "indexed",
                "dataset": "user_uploads",
                "metadata": {"category": "research"},
                "job_id": "job-xyz789",
                "stats": {
                    "chunks": 12,
                    "entities": 35,
                    "edges": 28,
                    "tokens": 3500
                },
                "top_entities": [
                    {"name": "BiG-RAG", "type": "TECHNOLOGY", "weight": 0.95}
                ],
                "related_documents": [
                    {"id": "upload-xyz123", "title": "RAG Survey", "similarity": 0.78}
                ]
            }
        }


class DeleteResponse(BaseModel):
    """Response for document deletion"""
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")
    document_id: str = Field(..., description="Deleted document ID")
    hard_delete: bool = Field(..., description="Whether it was a hard delete")
    rebuild_required: bool = Field(..., description="Whether KG rebuild is required")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document marked as deleted (soft delete)",
                "document_id": "upload-a1b2c3d4e5f6",
                "hard_delete": False,
                "rebuild_required": False
            }
        }


class BatchDeleteResponse(BaseModel):
    """Response for batch deletion"""
    success: bool = Field(..., description="Whether batch deletion was successful")
    message: str = Field(..., description="Status message")
    total_requested: int = Field(..., description="Total documents requested for deletion")
    deleted: int = Field(..., description="Successfully deleted documents")
    failed: int = Field(..., description="Failed deletions")
    hard_delete: bool = Field(..., description="Whether it was a hard delete")
    rebuild_required: bool = Field(..., description="Whether KG rebuild is required")
    results: List[DeleteResponse] = Field(..., description="Individual deletion results")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Deletion errors")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Batch deletion completed: 8/10 succeeded",
                "total_requested": 10,
                "deleted": 8,
                "failed": 2,
                "hard_delete": False,
                "rebuild_required": False,
                "results": [],
                "errors": [
                    {"document_id": "upload-missing", "error": "Document not found"}
                ]
            }
        }


# ==============================================================================
# Response Models - Graph Statistics
# ==============================================================================

class DatasetStats(BaseModel):
    """Statistics for a single dataset"""
    dataset: str = Field(..., description="Dataset name")
    total_documents: int = Field(..., description="Total documents in dataset")
    indexed_documents: int = Field(..., description="Successfully indexed documents")
    pending_documents: int = Field(..., description="Pending documents")
    failed_documents: int = Field(..., description="Failed documents")
    total_chunks: int = Field(..., description="Total text chunks")
    total_entities: int = Field(..., description="Total entities")
    total_edges: int = Field(..., description="Total bipartite edges")
    total_tokens: int = Field(..., description="Total tokens processed")


class GraphStatsResponse(BaseModel):
    """Response for graph statistics"""
    success: bool = Field(..., description="Whether query was successful")
    total_datasets: int = Field(..., description="Total number of datasets")
    global_stats: Dict[str, int] = Field(..., description="Global statistics across all datasets")
    datasets: List[DatasetStats] = Field(..., description="Per-dataset statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_datasets": 3,
                "global_stats": {
                    "total_documents": 150,
                    "total_entities": 4520,
                    "total_edges": 3890,
                    "total_chunks": 1250
                },
                "datasets": [
                    {
                        "dataset": "user_uploads",
                        "total_documents": 50,
                        "indexed_documents": 48,
                        "pending_documents": 1,
                        "failed_documents": 1,
                        "total_chunks": 420,
                        "total_entities": 1500,
                        "total_edges": 1200,
                        "total_tokens": 105000
                    }
                ]
            }
        }


# ==============================================================================
# Response Models - System
# ==============================================================================

class RAGInstanceInfo(BaseModel):
    """RAG instance information"""
    dataset: str = Field(..., description="Dataset name")
    status: str = Field(..., description="Instance status")
    total_documents: int = Field(0, description="Total documents in KG")
    indices_loaded: bool = Field(..., description="Whether FAISS indices are loaded")


class HealthResponse(BaseModel):
    """Response for health check"""
    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current server time")
    rag_instances: Dict[str, RAGInstanceInfo] = Field(..., description="RAG instance information")
    job_queue: Dict[str, int] = Field(..., description="Job queue statistics")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-10-30T10:30:00Z",
                "rag_instances": {
                    "user_uploads": {
                        "dataset": "user_uploads",
                        "status": "active",
                        "total_documents": 50,
                        "indices_loaded": True
                    }
                },
                "job_queue": {
                    "pending": 2,
                    "processing": 3,
                    "completed": 45,
                    "failed": 1
                },
                "uptime_seconds": 86400.5
            }
        }


# ==============================================================================
# Error Response Models
# ==============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information"""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = Field(False, description="Always False for errors")
    error: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Validation failed",
                "details": [
                    {
                        "field": "file",
                        "message": "Only .txt and .md files are supported",
                        "error_code": "INVALID_FILE_TYPE"
                    }
                ],
                "timestamp": "2025-10-30T10:30:00Z"
            }
        }

# ==============================================================================
# Additional Request/Response Models for Retrieval and LLM
# ==============================================================================

class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    question: str
    top_k: Optional[int] = 5
    mode: Optional[str] = "hybrid"
    llm_provider: Optional[str] = None
    enable_reranking: Optional[bool] = False

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is Artificial Intelligence?",
                "top_k": 5,
                "mode": "hybrid",
                "llm_provider": "openai",
                "enable_reranking": False
            }
        }


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    question: str
    retrieved_contexts: List[Dict[str, Any]]
    num_results: int
    mode: str
    llm_provider_used: Optional[str] = None
    message: Optional[str] = None


class SearchRequest(BaseModel):
    """Request model for /search endpoint"""
    queries: List[str]


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # "system", "user", or "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for /chat/completions endpoint"""
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    llm_provider: Optional[str] = None
    use_rag: Optional[bool] = True
    enable_reranking: Optional[bool] = False

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is Artificial Intelligence?"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500,
                "use_rag": True,
                "enable_reranking": False
            }
        }


class RebuildResponse(BaseModel):
    """Response model for /rebuild endpoint"""
    success: bool
    message: str
    documents_processed: Optional[int] = 0
    dataset: str
    rebuild_type: str
