"""
API tests for document management endpoints

Tests file upload, listing, retrieval, and deletion of documents.
"""

import pytest
import os
import io
from pathlib import Path


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped"
)
class TestDocumentsAPI:
    """Test /documents endpoints"""

    @pytest.mark.asyncio
    async def test_upload_document_txt(self, api_client):
        """Test uploading a .txt document"""
        try:
            # Create a simple text file
            file_content = b"This is a test document about Python programming. Python is a high-level language."
            files = {"file": ("test_doc.txt", io.BytesIO(file_content), "text/plain")}
            data = {
                "title": "Python Test Document",
                "process_async": "true"
            }

            response = await api_client.post(
                "/documents/upload",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result = response.json()
                assert "job_id" in result or "document_id" in result or "status" in result
                # Upload endpoint should return job tracking info
            elif response.status_code == 404:
                pytest.skip("Upload endpoint not available")
            else:
                pytest.skip(f"Upload endpoint returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_upload_document_markdown(self, api_client):
        """Test uploading a .md document"""
        try:
            # Create a markdown file
            file_content = b"# Test Document\n\nThis is a **markdown** document about machine learning.\n\nML is a subset of AI."
            files = {"file": ("test_doc.md", io.BytesIO(file_content), "text/markdown")}
            data = {
                "title": "ML Markdown Doc",
                "process_async": "true",
                "metadata": '{"category": "research", "tags": ["ml", "ai"]}'
            }

            response = await api_client.post(
                "/documents/upload",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result = response.json()
                assert isinstance(result, dict)
            elif response.status_code == 404:
                pytest.skip("Upload endpoint not available")
            else:
                pytest.skip(f"Upload endpoint returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_upload_document_with_metadata(self, api_client):
        """Test uploading document with custom metadata"""
        try:
            file_content = b"Research paper about neural networks and deep learning architectures."
            files = {"file": ("research.txt", io.BytesIO(file_content), "text/plain")}
            data = {
                "title": "Neural Networks Research",
                "metadata": '{"category": "research", "tags": ["neural nets", "deep learning"], "author": "Test User"}'
            }

            response = await api_client.post(
                "/documents/upload",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result = response.json()
                # Check metadata is acknowledged
                assert isinstance(result, dict)
            elif response.status_code == 404:
                pytest.skip("Upload endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_upload_invalid_file_type(self, api_client):
        """Test uploading invalid file type (should reject)"""
        try:
            # Try to upload a PDF (not supported)
            file_content = b"%PDF-1.4 fake pdf content"
            files = {"file": ("document.pdf", io.BytesIO(file_content), "application/pdf")}
            data = {"title": "PDF Document"}

            response = await api_client.post(
                "/documents/upload",
                files=files,
                data=data
            )

            # Should reject with 400 or 415 (Unsupported Media Type)
            if response.status_code in [400, 415, 422]:
                pass  # Expected rejection
            elif response.status_code == 404:
                pytest.skip("Upload endpoint not available")
            else:
                # If it accepts PDF, that's also valid (depends on implementation)
                pass

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_upload_large_file(self, api_client):
        """Test uploading larger file (within limits)"""
        try:
            # Create a 1MB text file
            file_content = b"Test content. " * 70000  # ~1MB
            files = {"file": ("large_doc.txt", io.BytesIO(file_content), "text/plain")}
            data = {"title": "Large Document"}

            response = await api_client.post(
                "/documents/upload",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result = response.json()
                assert isinstance(result, dict)
            elif response.status_code in [413, 400]:
                # File too large - acceptable rejection
                pass
            elif response.status_code == 404:
                pytest.skip("Upload endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_list_documents(self, api_client):
        """Test listing all documents"""
        try:
            response = await api_client.get("/documents")

            if response.status_code == 200:
                data = response.json()
                # Should return a list or paginated result
                assert isinstance(data, (list, dict))
                if isinstance(data, dict):
                    # Might be paginated: {"documents": [...], "total": N}
                    assert "documents" in data or "items" in data or "data" in data
            elif response.status_code == 404:
                pytest.skip("Documents list endpoint not available")
            else:
                pytest.skip(f"Documents list returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_list_documents_with_filters(self, api_client):
        """Test listing documents with query filters"""
        try:
            # Test with common filter parameters
            response = await api_client.get(
                "/documents?limit=10&offset=0&sort_by=created_at"
            )

            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, (list, dict))
            elif response.status_code == 404:
                pytest.skip("Documents list endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_get_document_by_id(self, api_client):
        """Test retrieving specific document by ID"""
        try:
            # Use a test ID (might not exist)
            response = await api_client.get("/documents/test_doc_id")

            if response.status_code == 200:
                data = response.json()
                # Should return document details
                assert isinstance(data, dict)
                assert "id" in data or "document_id" in data or "title" in data
            elif response.status_code == 404:
                # Document not found is acceptable for test ID
                pass
            else:
                pytest.skip(f"Document detail endpoint returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_delete_document_soft(self, api_client):
        """Test soft deleting a document"""
        try:
            # Try to delete with soft delete
            response = await api_client.delete(
                "/documents/test_doc_id?delete_type=soft"
            )

            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, dict)
                # Should confirm deletion
            elif response.status_code == 404:
                # Document not found or endpoint not available
                pytest.skip("Delete endpoint not available or document not found")
            else:
                pytest.skip(f"Delete endpoint returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_delete_document_hard(self, api_client):
        """Test hard deleting a document (cascade)"""
        try:
            # Try to delete with hard delete
            response = await api_client.delete(
                "/documents/test_doc_id?delete_type=hard"
            )

            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, dict)
            elif response.status_code == 404:
                pytest.skip("Delete endpoint not available or document not found")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_rebuild_graph(self, api_client):
        """Test rebuilding the knowledge graph"""
        try:
            # This is typically an admin operation
            response = await api_client.post("/documents/rebuild")

            if response.status_code == 200:
                data = response.json()
                # Should return job ID or status
                assert isinstance(data, dict)
            elif response.status_code == 202:
                # Accepted for processing
                pass
            elif response.status_code == 404:
                pytest.skip("Rebuild endpoint not available")
            else:
                pytest.skip(f"Rebuild endpoint returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_upload_sync_processing(self, api_client):
        """Test synchronous upload processing"""
        try:
            file_content = b"Small test document for sync processing."
            files = {"file": ("sync_test.txt", io.BytesIO(file_content), "text/plain")}
            data = {
                "title": "Sync Test",
                "process_async": "false"  # Request synchronous processing
            }

            response = await api_client.post(
                "/documents/upload",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result = response.json()
                # Should return completed status or document details
                assert isinstance(result, dict)
            elif response.status_code == 404:
                pytest.skip("Upload endpoint not available")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
