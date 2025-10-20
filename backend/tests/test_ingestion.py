"""
Tests for the Ingestion Agent.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from app.pipelines.ingestion_pipeline import IngestionPipeline
from app.rag.chunking import DocumentChunker
from app.rag.vector_store import VectorStore


class TestIngestionPipeline:
    """Test suite for IngestionPipeline."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore for testing."""
        with patch("app.pipelines.ingestion_pipeline.VectorStore") as mock:
            mock_instance = Mock(spec=VectorStore)
            mock_instance.add_chunks.return_value = ["chunk1", "chunk2"]
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_chunker(self):
        """Mock DocumentChunker for testing."""
        with patch("app.pipelines.ingestion_pipeline.DocumentChunker") as mock:
            mock_instance = Mock(spec=DocumentChunker)
            mock_instance.chunk_document.return_value = [
                {
                    "chunk_id": "chunk1",
                    "document_id": "doc1",
                    "text": "Test chunk 1",
                    "chunk_index": 0,
                    "metadata": {},
                },
                {
                    "chunk_id": "chunk2",
                    "document_id": "doc1",
                    "text": "Test chunk 2",
                    "chunk_index": 1,
                    "metadata": {},
                },
            ]
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def pipeline(self, mock_vector_store, mock_chunker):
        """Create IngestionPipeline instance with mocked dependencies."""
        return IngestionPipeline()

    def test_supported_extensions(self):
        """Test that supported extensions are correctly defined."""
        assert ".pdf" in IngestionPipeline.SUPPORTED_EXTENSIONS
        assert ".docx" in IngestionPipeline.SUPPORTED_EXTENSIONS
        assert ".txt" in IngestionPipeline.SUPPORTED_EXTENSIONS

    def test_load_document_file_not_found(self, pipeline):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            pipeline.load_document("nonexistent_file.pdf")

    def test_load_document_unsupported_format(self, pipeline, tmp_path):
        """Test loading unsupported file format raises ValueError."""
        # Create a temporary file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            pipeline.load_document(str(test_file))

    @patch("app.pipelines.ingestion_pipeline.PyMuPDFLoader")
    def test_load_pdf_document(self, mock_loader, pipeline, tmp_path):
        """Test loading a PDF document."""
        # Create a temporary PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        # Mock the loader
        mock_doc = Mock()
        mock_doc.page_content = "Test PDF content"
        mock_loader.return_value.load.return_value = [mock_doc]

        # Load document
        result = pipeline.load_document(str(test_file))

        # Assertions
        assert result["file_name"] == "test.pdf"
        assert result["content"] == "Test PDF content"
        assert result["metadata"]["file_type"] == "pdf"
        assert "document_id" in result

    @patch("app.pipelines.ingestion_pipeline.TextLoader")
    def test_load_txt_document(self, mock_loader, pipeline, tmp_path):
        """Test loading a TXT document."""
        # Create a temporary TXT file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test text content")

        # Mock the loader
        mock_doc = Mock()
        mock_doc.page_content = "Test text content"
        mock_loader.return_value.load.return_value = [mock_doc]

        # Load document
        result = pipeline.load_document(str(test_file))

        # Assertions
        assert result["file_name"] == "test.txt"
        assert result["content"] == "Test text content"
        assert result["metadata"]["file_type"] == "txt"

    def test_ingest_document(self, pipeline, mock_chunker, mock_vector_store, tmp_path):
        """Test full document ingestion pipeline."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for ingestion")

        # Mock document loading
        with patch.object(pipeline, "load_document") as mock_load:
            mock_load.return_value = {
                "document_id": "doc1",
                "file_name": "test.txt",
                "content": "Test content for ingestion",
                "metadata": {"file_type": "txt"},
            }

            # Ingest document
            session_id = str(uuid4())
            result = pipeline.ingest_document(str(test_file), session_id)

            # Assertions
            assert result["status"] == "success"
            assert result["document_id"] == "doc1"
            assert result["file_name"] == "test.txt"
            assert result["chunk_count"] == 2
            mock_chunker.chunk_document.assert_called_once()
            mock_vector_store.add_chunks.assert_called_once()

    def test_ingest_documents_batch(
        self, pipeline, mock_chunker, mock_vector_store, tmp_path
    ):
        """Test batch document ingestion."""
        # Create temporary files
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        # Mock document loading
        with patch.object(pipeline, "ingest_document") as mock_ingest:
            mock_ingest.side_effect = [
                {
                    "document_id": "doc1",
                    "file_name": "test1.txt",
                    "chunk_count": 2,
                    "status": "success",
                },
                {
                    "document_id": "doc2",
                    "file_name": "test2.txt",
                    "chunk_count": 3,
                    "status": "success",
                },
            ]

            # Ingest batch
            session_id = str(uuid4())
            result = pipeline.ingest_documents([str(file1), str(file2)], session_id)

            # Assertions
            assert result["total_documents"] == 2
            assert result["successful"] == 2
            assert result["failed"] == 0
            assert result["total_chunks"] == 5

    def test_ingest_documents_with_failures(self, pipeline, tmp_path):
        """Test batch ingestion with some failures."""
        # Create one valid file
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("Valid content")

        # Use non-existent file for failure
        invalid_file = "nonexistent.txt"

        # Mock successful ingestion for valid file
        with patch.object(pipeline, "ingest_document") as mock_ingest:
            mock_ingest.side_effect = [
                {
                    "document_id": "doc1",
                    "file_name": "valid.txt",
                    "chunk_count": 2,
                    "status": "success",
                },
                FileNotFoundError("File not found"),
            ]

            # Ingest batch
            session_id = str(uuid4())
            result = pipeline.ingest_documents(
                [str(valid_file), invalid_file], session_id
            )

            # Assertions
            assert result["total_documents"] == 2
            assert result["successful"] == 1
            assert result["failed"] == 1
            assert len(result["failures"]) == 1
