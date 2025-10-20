"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from io import BytesIO

from app.main import app


class TestUploadEndpoint:
    """Test suite for upload endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_ingestion_agent(self):
        """Mock IngestionAgent for testing."""
        with patch("app.api.v1.endpoints.upload.IngestionAgent") as mock:
            mock_instance = Mock()
            mock_instance.ingest_documents.return_value = {
                "session_id": "test-session",
                "total_documents": 1,
                "successful": 1,
                "failed": 0,
                "total_chunks": 5,
                "results": [],
                "failures": [],
            }
            mock.return_value = mock_instance
            yield mock_instance

    def test_upload_success(self, client, mock_ingestion_agent):
        """Test successful file upload."""
        # Create fake file
        file_content = b"Test document content"
        files = {"files": ("test.txt", BytesIO(file_content), "text/plain")}

        response = client.post("/api/v1/upload", files=files)

        # Assertions
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert "file_names" in data
        assert "test.txt" in data["file_names"]

    def test_upload_no_files(self, client):
        """Test upload with no files returns 400."""
        response = client.post("/api/v1/upload", files={})

        assert response.status_code == 422  # FastAPI validation error

    def test_upload_unsupported_format(self, client, mock_ingestion_agent):
        """Test upload with unsupported file format."""
        # Create fake file with unsupported extension
        file_content = b"Test content"
        files = {"files": ("test.xyz", BytesIO(file_content), "application/octet-stream")}

        # Mock to return no successful ingestions
        mock_ingestion_agent.ingest_documents.return_value = {
            "session_id": "test-session",
            "total_documents": 0,
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "results": [],
            "failures": [],
        }

        response = client.post("/api/v1/upload", files=files)

        # Should return 400 for no valid files
        assert response.status_code == 400
        assert "No valid files" in response.json()["detail"]

    def test_upload_multiple_files(self, client, mock_ingestion_agent):
        """Test uploading multiple files."""
        files = [
            ("files", ("test1.txt", BytesIO(b"Content 1"), "text/plain")),
            ("files", ("test2.txt", BytesIO(b"Content 2"), "text/plain")),
        ]

        mock_ingestion_agent.ingest_documents.return_value = {
            "session_id": "test-session",
            "total_documents": 2,
            "successful": 2,
            "failed": 0,
            "total_chunks": 10,
            "results": [],
            "failures": [],
        }

        response = client.post("/api/v1/upload", files=files)

        assert response.status_code == 201
        data = response.json()
        assert len(data["file_names"]) == 2


class TestQueryEndpoint:
    """Test suite for query endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Mock RAGPipeline for testing."""
        with patch("app.api.v1.endpoints.query.rag_pipeline") as mock:
            mock.query.return_value = {
                "answer": "The payment terms are net 30 days.",
                "sources": [
                    {
                        "text": "Payment terms: net 30 days...",
                        "score": 0.95,
                        "source": "contract.pdf",
                        "chunk_index": 0,
                    }
                ],
                "metadata": {
                    "chunks_retrieved": 1,
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                },
            }
            yield mock

    def test_query_success(self, client, mock_rag_pipeline):
        """Test successful query."""
        request_data = {
            "session_id": "test-session",
            "query": "What are the payment terms?",
            "top_k": 5,
        }

        response = client.post("/api/v1/query", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["query"] == "What are the payment terms?"
        assert "answer" in data
        assert "sources" in data
        assert "metadata" in data

    def test_query_missing_fields(self, client):
        """Test query with missing required fields."""
        request_data = {
            "query": "What are the payment terms?",
            # Missing session_id
        }

        response = client.post("/api/v1/query", json=request_data)

        # Should return 422 for validation error
        assert response.status_code == 422

    def test_query_with_default_top_k(self, client, mock_rag_pipeline):
        """Test query uses default top_k when not specified."""
        request_data = {
            "session_id": "test-session",
            "query": "Test query",
            # top_k not specified, should use default
        }

        response = client.post("/api/v1/query", json=request_data)

        assert response.status_code == 200
        # Verify the pipeline was called
        mock_rag_pipeline.query.assert_called_once()


class TestHealthEndpoints:
    """Test suite for health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns health status."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Legal-OS API"

    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
