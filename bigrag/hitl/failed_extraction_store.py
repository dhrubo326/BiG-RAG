"""
Failed Extraction Store - HITL System

Stores failed extractions (chunks and tables) for human review and later reprocessing.

Storage Structure:
    expr/{dataset}/failed_extractions/
    ├── failed_chunks.json       # Paragraph extraction failures
    ├── failed_tables.json       # Table extraction failures
    └── review_queue.json        # Pending human review

Part of Phase 1 Step 6: HITL System
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from bigrag.utils import logger


class FailedExtractionStore:
    """
    Store and manage failed extractions for human review.

    Usage:
        store = FailedExtractionStore("expr/my_dataset")

        # Save failed chunk
        store.save_failed_chunk(
            chunk_id="chunk_001",
            chunk_content="The paragraph text...",
            failure_reason="Validation failed: numeric mismatch",
            validation_details={...},
            document_id="doc_123"
        )

        # Retrieve failures
        failures = store.get_failed_extractions(document_id="doc_123")

        # Mark as reviewed
        store.mark_reviewed(extraction_id, corrected_data)
    """

    def __init__(self, dataset_path: str):
        """
        Initialize HITL store for a dataset.

        Args:
            dataset_path: Path to dataset directory (e.g., "expr/my_dataset")
        """
        self.base_path = Path(dataset_path) / "failed_extractions"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.chunks_file = self.base_path / "failed_chunks.json"
        self.tables_file = self.base_path / "failed_tables.json"
        self.queue_file = self.base_path / "review_queue.json"

        logger.info(f"[HITL] Initialized store at {self.base_path}")

    def save_failed_chunk(
        self,
        chunk_id: str,
        chunk_content: str,
        failure_reason: str,
        validation_details: Dict,
        document_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save failed chunk extraction for human review.

        Args:
            chunk_id: Unique chunk identifier
            chunk_content: Full text content of the chunk
            failure_reason: Human-readable reason for failure
            validation_details: Full validation result dictionary
            document_id: Parent document ID
            metadata: Optional chunk metadata (title, category, etc.)

        Returns:
            extraction_id: Unique ID for this failure record
        """
        timestamp = datetime.now()
        extraction_id = f"chunk_{chunk_id}_{int(timestamp.timestamp())}"

        failure_record = {
            "extraction_id": extraction_id,
            "type": "chunk",
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": chunk_content,
            "failure_reason": failure_reason,
            "validation_details": validation_details,
            "metadata": metadata or {},
            "timestamp": timestamp.isoformat(),
            "status": "pending_review"
        }

        self._append_to_file(self.chunks_file, failure_record)
        self._add_to_review_queue(failure_record)

        logger.warning(
            f"[HITL] Saved failed chunk: {chunk_id} "
            f"(doc: {document_id}, reason: {failure_reason})"
        )

        return extraction_id

    def save_failed_table(
        self,
        table_id: str,
        table_data: Dict,
        failure_reason: str,
        document_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save failed table extraction for human review.

        Args:
            table_id: Unique table identifier
            table_data: Table structure (headers, rows, caption)
            failure_reason: Human-readable reason for failure
            document_id: Parent document ID
            metadata: Optional table metadata

        Returns:
            extraction_id: Unique ID for this failure record
        """
        timestamp = datetime.now()
        extraction_id = f"table_{table_id}_{int(timestamp.timestamp())}"

        failure_record = {
            "extraction_id": extraction_id,
            "type": "table",
            "table_id": table_id,
            "document_id": document_id,
            "table_data": table_data,
            "failure_reason": failure_reason,
            "metadata": metadata or {},
            "timestamp": timestamp.isoformat(),
            "status": "pending_review"
        }

        self._append_to_file(self.tables_file, failure_record)
        self._add_to_review_queue(failure_record)

        logger.warning(
            f"[HITL] Saved failed table: {table_id} "
            f"(doc: {document_id}, reason: {failure_reason})"
        )

        return extraction_id

    def get_failed_extractions(
        self,
        document_id: Optional[str] = None,
        extraction_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve failed extractions with optional filtering.

        Args:
            document_id: Filter by document ID (None = all documents)
            extraction_type: Filter by type ('chunk', 'table', None = both)
            status: Filter by status ('pending_review', 'reviewed', 'corrected', None = all)

        Returns:
            List of failure records
        """
        all_failures = []

        # Load chunks
        if extraction_type in [None, 'chunk']:
            if self.chunks_file.exists():
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    all_failures.extend(json.load(f))

        # Load tables
        if extraction_type in [None, 'table']:
            if self.tables_file.exists():
                with open(self.tables_file, 'r', encoding='utf-8') as f:
                    all_failures.extend(json.load(f))

        # Apply filters
        if document_id:
            all_failures = [f for f in all_failures if f.get("document_id") == document_id]

        if status:
            all_failures = [f for f in all_failures if f.get("status") == status]

        return all_failures

    def get_review_queue(self) -> List[Dict]:
        """
        Get all items pending human review.

        Returns:
            List of extraction records with status='pending_review'
        """
        if not self.queue_file.exists():
            return []

        with open(self.queue_file, 'r', encoding='utf-8') as f:
            queue = json.load(f)

        # Filter to only pending items
        pending = [item for item in queue if item.get("status") == "pending_review"]

        return pending

    def mark_reviewed(
        self,
        extraction_id: str,
        corrected_data: Optional[Dict] = None,
        reviewer_notes: Optional[str] = None
    ) -> bool:
        """
        Mark extraction as human-reviewed with optional corrections.

        Args:
            extraction_id: ID of the extraction to mark
            corrected_data: Optional corrected extraction data
            reviewer_notes: Optional human reviewer notes

        Returns:
            True if successfully marked, False if not found
        """
        updated = False

        # Update in chunks file
        if self.chunks_file.exists():
            updated = self._update_status_in_file(
                self.chunks_file,
                extraction_id,
                "reviewed",
                corrected_data,
                reviewer_notes
            )

        # Update in tables file if not found in chunks
        if not updated and self.tables_file.exists():
            updated = self._update_status_in_file(
                self.tables_file,
                extraction_id,
                "reviewed",
                corrected_data,
                reviewer_notes
            )

        # Update review queue
        if updated:
            self._update_queue_status(extraction_id, "reviewed")
            logger.info(f"[HITL] Marked {extraction_id} as reviewed")

        return updated

    def mark_corrected(
        self,
        extraction_id: str,
        corrected_data: Dict,
        correction_notes: Optional[str] = None
    ) -> bool:
        """
        Mark extraction as corrected and ready for reprocessing.

        Args:
            extraction_id: ID of the extraction
            corrected_data: Corrected extraction data
            correction_notes: Optional correction notes

        Returns:
            True if successfully marked
        """
        updated = False

        # Update in appropriate file
        if self.chunks_file.exists():
            updated = self._update_status_in_file(
                self.chunks_file,
                extraction_id,
                "corrected",
                corrected_data,
                correction_notes
            )

        if not updated and self.tables_file.exists():
            updated = self._update_status_in_file(
                self.tables_file,
                extraction_id,
                "corrected",
                corrected_data,
                correction_notes
            )

        if updated:
            self._update_queue_status(extraction_id, "corrected")
            logger.info(f"[HITL] Marked {extraction_id} as corrected")

        return updated

    def get_statistics(self) -> Dict:
        """
        Get statistics about failed extractions.

        Returns:
            Dictionary with counts by type, status, and document
        """
        all_failures = self.get_failed_extractions()

        stats = {
            "total_failures": len(all_failures),
            "by_type": {
                "chunk": len([f for f in all_failures if f["type"] == "chunk"]),
                "table": len([f for f in all_failures if f["type"] == "table"])
            },
            "by_status": {
                "pending_review": len([f for f in all_failures if f["status"] == "pending_review"]),
                "reviewed": len([f for f in all_failures if f["status"] == "reviewed"]),
                "corrected": len([f for f in all_failures if f["status"] == "corrected"])
            },
            "by_document": {}
        }

        # Count by document
        for failure in all_failures:
            doc_id = failure.get("document_id", "unknown")
            stats["by_document"][doc_id] = stats["by_document"].get(doc_id, 0) + 1

        return stats

    def delete_extraction(self, extraction_id: str) -> bool:
        """
        Delete an extraction record (use with caution).

        Args:
            extraction_id: ID of extraction to delete

        Returns:
            True if deleted, False if not found
        """
        deleted = False

        # Try chunks file
        if self.chunks_file.exists():
            deleted = self._delete_from_file(self.chunks_file, extraction_id)

        # Try tables file
        if not deleted and self.tables_file.exists():
            deleted = self._delete_from_file(self.tables_file, extraction_id)

        # Remove from queue
        if deleted:
            self._delete_from_file(self.queue_file, extraction_id)
            logger.info(f"[HITL] Deleted {extraction_id}")

        return deleted

    # Helper methods

    def _append_to_file(self, file_path: Path, record: Dict):
        """Append record to JSON file."""
        records = []
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    records = json.load(f)
                except json.JSONDecodeError:
                    records = []

        records.append(record)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def _add_to_review_queue(self, record: Dict):
        """Add to review queue."""
        # Only add essential fields to queue (keep it lightweight)
        queue_item = {
            "extraction_id": record["extraction_id"],
            "type": record["type"],
            "document_id": record["document_id"],
            "failure_reason": record["failure_reason"],
            "timestamp": record["timestamp"],
            "status": record["status"]
        }

        self._append_to_file(self.queue_file, queue_item)

    def _update_status_in_file(
        self,
        file_path: Path,
        extraction_id: str,
        new_status: str,
        corrected_data: Optional[Dict],
        notes: Optional[str]
    ) -> bool:
        """Update status of extraction in file."""
        if not file_path.exists():
            return False

        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        updated = False
        for record in records:
            if record.get("extraction_id") == extraction_id:
                record["status"] = new_status
                record["reviewed_at"] = datetime.now().isoformat()

                if corrected_data:
                    record["corrected_data"] = corrected_data

                if notes:
                    record["reviewer_notes"] = notes

                updated = True
                break

        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        return updated

    def _update_queue_status(self, extraction_id: str, new_status: str):
        """Update status in review queue."""
        if not self.queue_file.exists():
            return

        with open(self.queue_file, 'r', encoding='utf-8') as f:
            queue = json.load(f)

        for item in queue:
            if item.get("extraction_id") == extraction_id:
                item["status"] = new_status
                break

        with open(self.queue_file, 'w', encoding='utf-8') as f:
            json.dump(queue, f, indent=2, ensure_ascii=False)

    def _delete_from_file(self, file_path: Path, extraction_id: str) -> bool:
        """Delete extraction from file."""
        if not file_path.exists():
            return False

        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        original_len = len(records)
        records = [r for r in records if r.get("extraction_id") != extraction_id]

        if len(records) < original_len:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            return True

        return False
