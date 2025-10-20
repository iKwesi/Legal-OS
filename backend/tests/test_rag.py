"""
Tests for RAG pipeline components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from app.rag.chunking import DocumentChunker
from app.rag.retrievers import NaiveRetriever
from app.rag.pipeline import RAGPipeline


class TestDocumentChunker:
    """Test suite for DocumentChunker."""

    @pytest.fixture
    def chunker(self):
        """Create DocumentChunker instance."""
        return DocumentChunker(chunk_size=100, chunk_overlap=20)

    def test_chunk_document(self, chunker):
        """Test chunking a document."""
        text = "This is a test document. " * 20  # Create text > 100 chars
        document_id = "doc1"
        metadata = {"source": "test.txt"}

        chunks = chunker.chunk_document(document_id, text, metadata)

        # Assertions
        assert len(chunks) > 0
        assert all(chunk["document_id"] == document_id for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)

    def test_chunk_empty_document(self, chunker):
        """Test chunking an empty document returns empty list."""
        chunks = chunker.chunk_document("doc1", "", {})
        assert chunks == []

    def test_chunk_documents_batch(self, chunker):
        """Test chunking multiple documents."""
        documents = [
            {
                "document_id": "doc1",
                "text": "Document 1 content. " * 10,
                "metadata": {"source": "doc1.txt"},
            },
            {
                "document_id": "doc2",
                "text": "Document 2 content. " * 10,
                "metadata": {"source": "doc2.txt"},
            },
        ]

        chunks = chunker.chunk_documents(documents)

        # Assertions
        assert len(chunks) > 0
        doc1_chunks = [c for c in chunks if c["document_id"] == "doc1"]
        doc2_chunks = [c for c in chunks if c["document_id"] == "doc2"]
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0


class TestNaiveRetriever:
    """Test suite for NaiveRetriever."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore for testing."""
        mock = Mock()
        mock.search.return_value = [
            {
                "chunk_id": "chunk1",
                "document_id": "doc1",
                "text": "Test chunk 1",
                "score": 0.95,
                "metadata": {"file_name": "test.pdf"},
                "chunk_index": 0,
            },
            {
                "chunk_id": "chunk2",
                "document_id": "doc1",
                "text": "Test chunk 2",
                "score": 0.85,
                "metadata": {"file_name": "test.pdf"},
                "chunk_index": 1,
            },
        ]
        return mock

    @pytest.fixture
    def retriever(self, mock_vector_store):
        """Create NaiveRetriever with mocked vector store."""
        with patch("backend.app.rag.retrievers.VectorStore") as mock_vs_class:
            mock_vs_class.return_value = mock_vector_store
            return NaiveRetriever(vector_store=mock_vector_store)

    def test_retrieve(self, retriever, mock_vector_store):
        """Test retrieving chunks."""
        query = "What is the test about?"
        session_id = "session1"

        results = retriever.retrieve(query, session_id=session_id, top_k=5)

        # Assertions
        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk1"
        assert results[0]["score"] == 0.95
        mock_vector_store.search.assert_called_once()

    def test_format_context(self, retriever):
        """Test formatting retrieved chunks into context."""
        chunks = [
            {
                "text": "First chunk content",
                "score": 0.95,
                "metadata": {"file_name": "doc1.pdf"},
            },
            {
                "text": "Second chunk content",
                "score": 0.85,
                "metadata": {"file_name": "doc2.pdf"},
            },
        ]

        context = retriever.format_context(chunks)

        # Assertions
        assert "First chunk content" in context
        assert "Second chunk content" in context
        assert "doc1.pdf" in context
        assert "doc2.pdf" in context
        assert "0.950" in context  # Score formatting

    def test_format_context_empty(self, retriever):
        """Test formatting empty chunks returns empty string."""
        context = retriever.format_context([])
        assert context == ""


class TestRAGPipeline:
    """Test suite for RAGPipeline."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock NaiveRetriever for testing."""
        mock = Mock(spec=NaiveRetriever)
        mock.retrieve.return_value = [
            {
                "chunk_id": "chunk1",
                "document_id": "doc1",
                "text": "The payment terms are net 30 days.",
                "score": 0.95,
                "metadata": {"file_name": "contract.pdf"},
                "chunk_index": 0,
            },
        ]
        mock.format_context.return_value = (
            "[Document 1] (Source: contract.pdf, Relevance: 0.950)\n"
            "The payment terms are net 30 days."
        )
        return mock

    @pytest.fixture
    def mock_llm(self):
        """Mock ChatOpenAI for testing."""
        mock = Mock()
        mock_response = Mock()
        mock_response.content = "The payment terms are net 30 days according to the contract."
        mock.invoke.return_value = mock_response
        return mock

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_llm):
        """Create RAGPipeline with mocked dependencies."""
        with patch("backend.app.rag.pipeline.NaiveRetriever") as mock_ret_class:
            with patch("backend.app.rag.pipeline.ChatOpenAI") as mock_llm_class:
                mock_ret_class.return_value = mock_retriever
                mock_llm_class.return_value = mock_llm
                return RAGPipeline(retriever=mock_retriever, llm=mock_llm)

    def test_query_success(self, pipeline, mock_retriever, mock_llm):
        """Test successful RAG query."""
        question = "What are the payment terms?"
        session_id = "session1"

        result = pipeline.query(question, session_id=session_id)

        # Assertions
        assert "answer" in result
        assert "sources" in result
        assert "metadata" in result
        assert result["answer"] == "The payment terms are net 30 days according to the contract."
        assert len(result["sources"]) == 1
        mock_retriever.retrieve.assert_called_once()
        mock_llm.invoke.assert_called_once()

    def test_query_no_results(self, pipeline, mock_retriever):
        """Test query with no retrieved chunks."""
        mock_retriever.retrieve.return_value = []

        result = pipeline.query("What is this about?", session_id="session1")

        # Assertions
        assert "couldn't find any relevant information" in result["answer"]
        assert len(result["sources"]) == 0
        assert result["metadata"]["chunks_retrieved"] == 0

    def test_query_with_custom_top_k(self, pipeline, mock_retriever):
        """Test query with custom top_k parameter."""
        question = "Test question"
        session_id = "session1"
        custom_top_k = 10

        pipeline.query(question, session_id=session_id, top_k=custom_top_k)

        # Verify retrieve was called with custom top_k
        call_args = mock_retriever.retrieve.call_args
        assert call_args[1]["top_k"] == custom_top_k
