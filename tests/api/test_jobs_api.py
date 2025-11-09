"""
API tests for job management endpoints

Tests async job tracking and status monitoring.
"""

import pytest
import os
import asyncio
import io


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped"
)
class TestJobsAPI:
    """Test /jobs endpoints"""

    @pytest.fixture
    async def test_job_id(self, api_client):
        """Create a test job by uploading a document"""
        # Upload a small test document to create a job
        file_content = b"Test document for job testing. Lionel Messi is a football player."
        files = {"file": ("test_job_doc.txt", io.BytesIO(file_content), "text/plain")}
        data = {"metadata": '{"title": "Test Job Document"}'}

        response = await api_client.post("/documents/upload", files=files, data=data)

        if response.status_code == 200:
            response_data = response.json()
            job_id = response_data.get("job_id")
            if job_id:
                # Wait a moment for job to start processing
                await asyncio.sleep(1)
                return job_id

        # If upload failed, skip jobs tests
        pytest.skip(f"Could not create test job (upload returned {response.status_code})")

    @pytest.mark.asyncio
    async def test_get_job_status_existing(self, api_client, test_job_id):
        """Test getting status of an existing job"""
        response = await api_client.get(f"/jobs/{test_job_id}")

        # Should return job status (200)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "job_id" in data
        assert data["job_id"] == test_job_id
        assert "status" in data
        assert "progress" in data
        assert data["status"] in ["pending", "processing", "completed", "failed"]

    @pytest.mark.asyncio
    async def test_get_job_status_nonexistent(self, api_client):
        """Test getting status of non-existent job"""
        response = await api_client.get("/jobs/nonexistent_job_id_12345")

        # Should return 404 (job not found)
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_job_status_format(self, api_client, test_job_id):
        """Test job status response format"""
        response = await api_client.get(f"/jobs/{test_job_id}")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data

        # Check progress is a number between 0 and 1
        assert isinstance(data["progress"], (int, float))
        assert 0 <= data["progress"] <= 1

        # Check status is valid
        valid_statuses = ["pending", "processing", "completed", "failed"]
        assert data["status"] in valid_statuses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
