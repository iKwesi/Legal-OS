"""
Integration tests for orchestration API endpoints.

Tests the complete workflow:
1. Upload document
2. Trigger analysis
3. Poll status
4. Retrieve results
"""

import pytest
import time
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.endpoints.analyze import (
    session_states,
    session_results,
    session_documents,
    session_lock,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_session_stores():
    """Clear session stores before each test."""
    with session_lock:
        session_states.clear()
        session_results.clear()
        session_documents.clear()
    yield
    with session_lock:
        session_states.clear()
        session_results.clear()
        session_documents.clear()


@pytest.fixture
def sample_document():
    """Path to a sample test document."""
    # Use existing test document (relative to backend directory)
    doc_path = Path("../data/Freedom_Final_Asset_Agreement.pdf")
    if not doc_path.exists():
        pytest.skip(f"Test document not found: {doc_path}")
    return doc_path


class TestAnalyzeEndpoint:
    """Tests for POST /api/v1/analyze endpoint."""

    def test_analyze_without_upload(self):
        """Test analyze endpoint fails when session doesn't exist."""
        response = client.post(
            "/api/v1/analyze",
            json={"session_id": "nonexistent-session"}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_analyze_with_invalid_session_id(self):
        """Test analyze endpoint validates session_id format."""
        response = client.post(
            "/api/v1/analyze",
            json={"session_id": ""}
        )
        assert response.status_code == 422  # Validation error

    def test_analyze_success(self, sample_document):
        """Test successful analysis trigger."""
        # First upload a document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]

        # Trigger analysis
        analyze_response = client.post(
            "/api/v1/analyze",
            json={"session_id": session_id}
        )

        assert analyze_response.status_code == 201
        data = analyze_response.json()
        assert data["session_id"] == session_id
        assert "status_url" in data
        assert f"/api/v1/status/{session_id}" in data["status_url"]

    def test_analyze_idempotent(self, sample_document):
        """Test calling analyze multiple times is idempotent."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # First analyze call
        response1 = client.post(
            "/api/v1/analyze",
            json={"session_id": session_id}
        )
        assert response1.status_code == 201

        # Second analyze call should return same response
        response2 = client.post(
            "/api/v1/analyze",
            json={"session_id": session_id}
        )
        assert response2.status_code == 201
        assert response2.json()["session_id"] == session_id


class TestStatusEndpoint:
    """Tests for GET /api/v1/status/{session_id} endpoint."""

    def test_status_nonexistent_session(self):
        """Test status endpoint with nonexistent session."""
        response = client.get("/api/v1/status/nonexistent-session")
        assert response.status_code == 404

    def test_status_pending(self, sample_document):
        """Test status shows pending before analysis starts."""
        # Upload and trigger analysis
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Manually set pending state (simulating analyze endpoint)
        with session_lock:
            session_states[session_id] = {
                "status": "pending",
                "progress": 0,
                "current_step": "start",
                "message": None
            }

        # Check status
        response = client.get(f"/api/v1/status/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["progress"] == 0

    def test_status_processing(self, sample_document):
        """Test status shows progress during processing."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Simulate processing state
        with session_lock:
            session_states[session_id] = {
                "status": "processing",
                "progress": 40,
                "current_step": "clause_extraction",
                "completed_steps": ["ingestion", "clause_extraction"],
                "message": None
            }

        # Check status
        response = client.get(f"/api/v1/status/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] >= 40  # Progress calculated from completed steps
        assert data["current_step"] == "clause_extraction"

    def test_status_completed(self, sample_document):
        """Test status shows completed state."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Simulate completed state
        with session_lock:
            session_states[session_id] = {
                "status": "completed",
                "progress": 100,
                "current_step": "complete",
                "completed_steps": ["ingestion", "clause_extraction", "risk_scoring", "summary", "provenance", "checklist"],
                "message": None
            }

        # Check status
        response = client.get(f"/api/v1/status/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100

    def test_status_failed(self, sample_document):
        """Test status shows failed state with error message."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Simulate failed state
        error_msg = "Test error message"
        with session_lock:
            session_states[session_id] = {
                "status": "failed",
                "progress": 0,
                "current_step": "ingestion",
                "message": error_msg
            }

        # Check status
        response = client.get(f"/api/v1/status/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["message"] == error_msg


class TestResultsEndpoint:
    """Tests for GET /api/v1/results/{session_id} endpoint."""

    def test_results_nonexistent_session(self):
        """Test results endpoint with nonexistent session."""
        response = client.get("/api/v1/results/nonexistent-session")
        assert response.status_code == 404

    def test_results_not_started(self, sample_document):
        """Test results endpoint when analysis not started."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Set pending state
        with session_lock:
            session_states[session_id] = {
                "status": "pending",
                "progress": 0
            }

        # Try to get results
        response = client.get(f"/api/v1/results/{session_id}")
        assert response.status_code == 400
        assert "not been started" in response.json()["detail"]

    def test_results_still_processing(self, sample_document):
        """Test results endpoint when analysis still in progress."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Set processing state
        with session_lock:
            session_states[session_id] = {
                "status": "processing",
                "progress": 50
            }

        # Try to get results
        response = client.get(f"/api/v1/results/{session_id}")
        assert response.status_code == 202
        assert "in progress" in response.json()["detail"]

    def test_results_failed(self, sample_document):
        """Test results endpoint when analysis failed."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Set failed state
        with session_lock:
            session_states[session_id] = {
                "status": "failed",
                "message": "Test failure"
            }

        # Try to get results
        response = client.get(f"/api/v1/results/{session_id}")
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()

    def test_results_success(self, sample_document):
        """Test successful results retrieval."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Simulate completed state with mock results
        with session_lock:
            session_states[session_id] = {
                "status": "completed",
                "progress": 100,
                "started_at": "2024-01-01T00:00:00Z"
            }
            session_results[session_id] = {
                "summary": {
                    "executive_summary": "Test summary"
                },
                "extracted_clauses": [],
                "scored_clauses": [],
                "checklist": [
                    {"question": "Test question", "category": "Legal"}
                ],
                "metadata": {},
                "completed_at": "2024-01-01T00:01:00Z"
            }

        # Get results
        response = client.get(f"/api/v1/results/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "completed"
        assert "summary_memo" in data
        assert "extracted_clauses" in data
        assert "red_flags" in data
        assert "checklist" in data
        assert len(data["checklist"]) == 1


class TestCompleteWorkflow:
    """Integration tests for complete upload → analyze → status → results workflow."""

    @pytest.mark.slow
    def test_complete_workflow_mock(self, sample_document):
        """Test complete workflow with mocked orchestration."""
        # Step 1: Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]

        # Step 2: Trigger analysis
        analyze_response = client.post(
            "/api/v1/analyze",
            json={"session_id": session_id}
        )
        assert analyze_response.status_code == 201

        # Step 3: Poll status (simulate processing)
        # Manually update state to simulate orchestration progress
        with session_lock:
            session_states[session_id]["status"] = "processing"
            session_states[session_id]["progress"] = 50
            session_states[session_id]["current_step"] = "risk_scoring"

        status_response = client.get(f"/api/v1/status/{session_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "processing"

        # Step 4: Simulate completion
        with session_lock:
            session_states[session_id]["status"] = "completed"
            session_states[session_id]["progress"] = 100
            session_results[session_id] = {
                "summary": {"executive_summary": "Mock summary"},
                "extracted_clauses": [],
                "scored_clauses": [],
                "checklist": [],
                "metadata": {},
                "completed_at": "2024-01-01T00:01:00Z"
            }

        # Step 5: Get results
        results_response = client.get(f"/api/v1/results/{session_id}")
        assert results_response.status_code == 200
        assert results_response.json()["status"] == "completed"


class TestProgressCalculation:
    """Tests for progress calculation logic."""

    def test_progress_calculation(self, sample_document):
        """Test that progress is calculated correctly from completed steps."""
        # Upload document
        with open(sample_document, "rb") as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"files": ("test.pdf", f, "application/pdf")}
            )
        session_id = upload_response.json()["session_id"]

        # Test different completion stages
        test_cases = [
            (["ingestion"], 20),
            (["ingestion", "clause_extraction"], 40),
            (["ingestion", "clause_extraction", "risk_scoring"], 60),
            (["ingestion", "clause_extraction", "risk_scoring", "summary"], 80),
            (["ingestion", "clause_extraction", "risk_scoring", "summary", "provenance"], 90),
            (["ingestion", "clause_extraction", "risk_scoring", "summary", "provenance", "checklist"], 100),
        ]

        for completed_steps, expected_progress in test_cases:
            with session_lock:
                session_states[session_id] = {
                    "status": "processing",
                    "progress": 0,
                    "completed_steps": completed_steps
                }

            response = client.get(f"/api/v1/status/{session_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["progress"] >= expected_progress
