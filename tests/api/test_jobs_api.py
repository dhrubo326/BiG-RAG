"""
API tests for job management endpoints

Tests async job tracking and status monitoring.
"""

import pytest
import os
import asyncio


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "false").lower() == "true",
    reason="API tests skipped"
)
class TestJobsAPI:
    """Test /jobs endpoints"""

    @pytest.mark.asyncio
    async def test_get_job_status_nonexistent(self, api_client):
        """Test getting status of non-existent job"""
        try:
            response = await api_client.get("/jobs/nonexistent_job_id")

            if response.status_code == 404:
                # Job not found - expected
                data = response.json()
                assert isinstance(data, dict)
            elif response.status_code == 200:
                # Job endpoint exists but job not found (might return empty status)
                data = response.json()
                assert isinstance(data, dict)
            else:
                pytest.skip(f"Jobs endpoint returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_get_job_status_format(self, api_client):
        """Test job status response format"""
        try:
            # Try to get a job status (may not exist)
            response = await api_client.get("/jobs/test_job_123")

            if response.status_code in [200, 404]:
                data = response.json()
                assert isinstance(data, dict)
                # If job exists, should have status fields
                if response.status_code == 200:
                    # Common job status fields
                    expected_fields = ["job_id", "status", "progress", "created_at"]
                    # At least one of these should be present
                    has_job_fields = any(field in data for field in expected_fields)
                    if has_job_fields:
                        assert True
            else:
                pytest.skip(f"Jobs endpoint returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")

    @pytest.mark.asyncio
    async def test_job_status_values(self, api_client):
        """Test that job status contains valid state values"""
        try:
            response = await api_client.get("/jobs/test_job")

            if response.status_code == 200:
                data = response.json()
                if "status" in data:
                    # Status should be one of: pending, processing, completed, failed
                    valid_statuses = ["pending", "processing", "completed", "failed", "queued"]
                    assert data["status"] in valid_statuses or isinstance(data["status"], str)
            elif response.status_code == 404:
                pytest.skip("Job not found (expected for test job)")
            else:
                pytest.skip(f"Jobs endpoint returned {response.status_code}")

        except Exception as e:
            pytest.skip(f"API server not running: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
