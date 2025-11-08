"""
Document Registry System

Manages document metadata and lifecycle
Stores metadata in: expr/{data_source}/documents_registry.json
"""

import json
import os
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
import logging

# Get project root directory (2 levels up from api/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from root .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """
    Manages document metadata and lifecycle

    Stores metadata in: expr/{data_source}/documents_registry.json
    """

    def __init__(self, working_dir: str = None):
        # Use configurable working directory from environment
        if working_dir is None:
            working_dir_base = os.getenv('WORKING_DIR', './expr').lstrip('./')
            self.working_dir = str(PROJECT_ROOT / working_dir_base)
        else:
            self.working_dir = working_dir

    def _get_registry_path(self, dataset: str) -> Path:
        """Get registry file path for dataset"""
        return Path(self.working_dir) / dataset / "documents_registry.json"

    async def _load_registry(self, dataset: str) -> Dict:
        """Load registry from disk"""
        path = self._get_registry_path(dataset)

        if not path.exists():
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error loading registry for {dataset}: {e}")
            return {}

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
        logger.info(f"Added document {document_id} to registry for dataset {dataset}")

    async def get_document(self, document_id: str, dataset: Optional[str] = None) -> Optional[Dict]:
        """Get document by ID"""

        if dataset:
            registry = await self._load_registry(dataset)

            # If not in registry, check kv_store_full_docs.json
            if document_id not in registry:
                full_docs_path = Path(self.working_dir) / dataset / "kv_store_full_docs.json"
                if full_docs_path.exists():
                    try:
                        with open(full_docs_path, 'r', encoding='utf-8') as f:
                            full_docs = json.load(f)
                            if document_id in full_docs:
                                doc_data = full_docs[document_id]
                                # Convert to registry format
                                return {
                                    "document_id": document_id,
                                    "filename": doc_data.get("metadata", {}).get("filename", "corpus_document"),
                                    "title": doc_data.get("metadata", {}).get("title", document_id),
                                    "content_length": len(doc_data.get("content", "")),
                                    "upload_date": doc_data.get("metadata", {}).get("upload_date", "corpus"),
                                    "indexed_date": doc_data.get("metadata", {}).get("indexed_date", "corpus"),
                                    "last_modified": doc_data.get("metadata", {}).get("last_modified", "corpus"),
                                    "status": "indexed",
                                    "dataset": dataset,
                                    "metadata": doc_data.get("metadata", {}),
                                    "job_id": "corpus_build",
                                    "stats": {}
                                }
                    except Exception as e:
                        logger.debug(f"Could not load from full_docs: {e}")

            return registry.get(document_id)
        else:
            # Search across all datasets
            working_path = Path(self.working_dir)

            if not working_path.exists():
                return None

            for dataset_dir in working_path.iterdir():
                if dataset_dir.is_dir():
                    registry = await self._load_registry(dataset_dir.name)
                    if document_id in registry:
                        return registry[document_id]

                    # Also check kv_store_full_docs.json
                    full_docs_path = dataset_dir / "kv_store_full_docs.json"
                    if full_docs_path.exists():
                        try:
                            with open(full_docs_path, 'r', encoding='utf-8') as f:
                                full_docs = json.load(f)
                                if document_id in full_docs:
                                    doc_data = full_docs[document_id]
                                    return {
                                        "document_id": document_id,
                                        "filename": doc_data.get("metadata", {}).get("filename", "corpus_document"),
                                        "title": doc_data.get("metadata", {}).get("title", document_id),
                                        "content_length": len(doc_data.get("content", "")),
                                        "upload_date": doc_data.get("metadata", {}).get("upload_date", "corpus"),
                                        "indexed_date": doc_data.get("metadata", {}).get("indexed_date", "corpus"),
                                        "last_modified": doc_data.get("metadata", {}).get("last_modified", "corpus"),
                                        "status": "indexed",
                                        "dataset": dataset_dir.name,
                                        "metadata": doc_data.get("metadata", {}),
                                        "job_id": "corpus_build",
                                        "stats": {}
                                    }
                        except Exception as e:
                            logger.debug(f"Could not load from full_docs: {e}")
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
            registry[document_id][key] = value

        registry[document_id]["last_modified"] = datetime.now().isoformat()

        await self._save_registry(dataset, registry)
        logger.info(f"Updated document {document_id} in registry")

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
            logger.info(f"Hard deleted document {document_id} from registry")
        else:
            # Mark as deleted
            registry[document_id]["status"] = "deleted"
            registry[document_id]["deleted_at"] = datetime.now().isoformat()
            logger.info(f"Soft deleted document {document_id}")

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
            working_path = Path(self.working_dir)

            if not working_path.exists():
                return []

            datasets = [d.name for d in working_path.iterdir() if d.is_dir()]

        for ds in datasets:
            registry = await self._load_registry(ds)

            # If registry is empty, try to load from kv_store_full_docs.json (corpus-built KG)
            if not registry:
                full_docs_path = Path(self.working_dir) / ds / "kv_store_full_docs.json"
                if full_docs_path.exists():
                    try:
                        with open(full_docs_path, 'r', encoding='utf-8') as f:
                            full_docs = json.load(f)
                            # Convert to registry format
                            for doc_id, doc_data in full_docs.items():
                                registry[doc_id] = {
                                    "document_id": doc_id,
                                    "filename": doc_data.get("metadata", {}).get("filename", "corpus_document"),
                                    "title": doc_data.get("metadata", {}).get("title", doc_id),
                                    "content_length": len(doc_data.get("content", "")),
                                    "upload_date": doc_data.get("metadata", {}).get("upload_date", "corpus"),
                                    "indexed_date": doc_data.get("metadata", {}).get("indexed_date", "corpus"),
                                    "last_modified": doc_data.get("metadata", {}).get("last_modified", "corpus"),
                                    "status": "indexed",  # Corpus docs are already indexed
                                    "dataset": ds,
                                    "metadata": doc_data.get("metadata", {}),
                                    "job_id": "corpus_build",
                                    "stats": {}
                                }
                    except Exception as e:
                        logger.debug(f"Could not load full_docs for {ds}: {e}")

            for doc_id, doc in registry.items():
                # Apply filters
                if status and doc.get("status") != status:
                    continue

                if search and search.lower() not in doc.get("title", "").lower():
                    if search.lower() not in doc.get("filename", "").lower():
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
        pending = sum(1 for d in registry.values() if d.get("status") == "pending")

        # If registry is empty, check kv_store_full_docs.json for corpus-built documents
        if total == 0:
            full_docs_path = Path(self.working_dir) / dataset / "kv_store_full_docs.json"
            if full_docs_path.exists():
                try:
                    with open(full_docs_path, 'r', encoding='utf-8') as f:
                        full_docs = json.load(f)
                        total = len(full_docs)
                        indexed = total  # All corpus docs are indexed
                except Exception as e:
                    logger.debug(f"Could not load full_docs stats: {e}")

        return {
            "total": total,
            "indexed": indexed,
            "failed": failed,
            "processing": processing,
            "pending": pending
        }


# Global registry instance
registry = DocumentRegistry()
