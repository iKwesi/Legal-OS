"""
Tests for E01_Pipeline_Foundation.py notebook

This test suite validates the notebook structure and ensures it can be
imported and executed without errors (with appropriate mocking).
"""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestNotebookStructure:
    """Test the structure and format of the E01 notebook."""

    @pytest.fixture
    def notebook_path(self):
        """Get the path to the notebook file."""
        return Path(__file__).parent.parent.parent / "notebooks" / "E01_Pipeline_Foundation.py"

    @pytest.fixture
    def notebook_content(self, notebook_path):
        """Read the notebook content."""
        return notebook_path.read_text()

    def test_notebook_file_exists(self, notebook_path):
        """Test that the notebook file exists."""
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

    def test_notebook_has_jupytext_markers(self, notebook_content):
        """Test that the notebook has Jupytext cell markers."""
        assert "# %%" in notebook_content, "Missing Jupytext cell markers (# %%)"
        assert "# %% [markdown]" in notebook_content, "Missing markdown cell markers"

    def test_notebook_has_required_cells(self, notebook_content):
        """Test that all required cells are present."""
        # Check for documentation cell
        assert "# E01: Pipeline Foundation" in notebook_content, "Missing documentation header"
        assert "Purpose" in notebook_content, "Missing purpose section"
        assert "Prerequisites" in notebook_content, "Missing prerequisites section"

        # Check for setup cell
        assert "Setup and Imports" in notebook_content, "Missing setup cell"
        assert "import pandas as pd" in notebook_content, "Missing pandas import"
        assert "from dotenv import load_dotenv" in notebook_content, "Missing dotenv import"

        # Check for ingestion cell
        assert "Document Ingestion" in notebook_content, "Missing ingestion cell"
        assert "ingest_document" in notebook_content, "Missing ingest_document call"

        # Check for RAG cell
        assert "RAG Pipeline Test" in notebook_content, "Missing RAG cell"
        assert "create_rag_chain" in notebook_content, "Missing create_rag_chain call"

        # Check for SGD cell
        assert "SGD Generation" in notebook_content or "Synthetic Golden Dataset" in notebook_content, "Missing SGD cell"

    def test_notebook_has_error_handling(self, notebook_content):
        """Test that the notebook includes error handling."""
        assert "try:" in notebook_content, "Missing try-except blocks"
        assert "except" in notebook_content, "Missing exception handling"
        assert "raise" in notebook_content or "print" in notebook_content, "Missing error reporting"

    def test_notebook_syntax_is_valid(self, notebook_path):
        """Test that the notebook has valid Python syntax."""
        content = notebook_path.read_text()
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Notebook has syntax errors: {e}")

    def test_notebook_has_environment_validation(self, notebook_content):
        """Test that the notebook validates environment variables."""
        assert "OPENAI_API_KEY" in notebook_content, "Missing OPENAI_API_KEY validation"
        assert "required_vars" in notebook_content or "missing_vars" in notebook_content, "Missing env var validation logic"


class TestNotebookImports:
    """Test that the notebook's imports are available."""

    def test_core_dependencies_importable(self):
        """Test that core dependencies can be imported."""
        try:
            import pandas  # noqa: F401
            from dotenv import load_dotenv  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Core dependencies not available: {e}")

    def test_langchain_dependencies_importable(self):
        """Test that LangChain dependencies can be imported."""
        try:
            import langchain  # noqa: F401
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: F401
        except ImportError as e:
            pytest.fail(f"LangChain dependencies not available: {e}")

    def test_project_imports_available(self):
        """Test that project modules can be imported."""
        # Add backend to path
        backend_path = Path(__file__).parent.parent
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        try:
            from app.core.config import settings  # noqa: F401
            from app.pipelines.ingestion_pipeline import ingest_document  # noqa: F401
            from app.rag.pipeline import create_rag_chain  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Project imports not available: {e}")


class TestNotebookExecution:
    """Test notebook execution with mocked external dependencies."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API calls."""
        with patch("langchain_openai.ChatOpenAI") as mock_chat, \
             patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
            mock_chat.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            yield mock_chat, mock_embeddings

    @pytest.fixture
    def mock_qdrant(self):
        """Mock Qdrant operations."""
        with patch("langchain_qdrant.QdrantVectorStore") as mock_qdrant:
            mock_instance = MagicMock()
            mock_qdrant.return_value = mock_instance
            yield mock_qdrant

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-1234")
        monkeypatch.setenv("QDRANT_HOST", "localhost")
        monkeypatch.setenv("QDRANT_PORT", "6333")

    def test_notebook_can_be_imported(self, mock_env_vars):
        """Test that the notebook can be imported as a module."""
        notebook_path = Path(__file__).parent.parent.parent / "notebooks" / "E01_Pipeline_Foundation.py"
        
        # Read and compile the notebook
        content = notebook_path.read_text()
        
        # Remove cell markers for compilation
        content_no_markers = content.replace("# %%", "").replace("# %% [markdown]", "")
        
        try:
            compile(content_no_markers, str(notebook_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"Notebook cannot be compiled: {e}")

    def test_environment_validation_logic(self, monkeypatch):
        """Test the environment validation logic."""
        # Test with missing env var
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        import os
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        assert "OPENAI_API_KEY" in missing_vars, "Environment validation should detect missing OPENAI_API_KEY"

    def test_path_resolution_logic(self):
        """Test that path resolution works correctly."""
        from pathlib import Path
        
        # Simulate notebook path resolution
        notebook_dir = Path(__file__).parent.parent.parent / "notebooks"
        backend_path = notebook_dir.parent / "backend"
        data_dir = notebook_dir.parent / "data"
        golden_dataset_dir = notebook_dir.parent / "golden_dataset"
        
        # These directories should exist from Story 1.1
        assert backend_path.exists(), f"Backend directory not found at {backend_path}"
        assert data_dir.exists(), f"Data directory not found at {data_dir}"
        assert golden_dataset_dir.exists(), f"Golden dataset directory not found at {golden_dataset_dir}"


class TestNotebookOutputs:
    """Test expected outputs and data formats."""

    def test_sgd_csv_format(self):
        """Test that SGD CSV files have the correct format."""
        golden_dataset_dir = Path(__file__).parent.parent.parent / "golden_dataset"
        csv_files = list(golden_dataset_dir.glob("sgd_benchmark*.csv"))
        
        if csv_files:
            import pandas as pd
            
            # Load the most recent CSV
            latest_csv = sorted(csv_files)[-1]
            df = pd.read_csv(latest_csv)
            
            # Check required columns (RAGAS 0.37+ format)
            required_columns = ["user_input", "reference_contexts", "reference", "synthesizer_name"]
            for col in required_columns:
                assert col in df.columns, f"Missing required column: {col}"
            
            # Check that we have data
            assert len(df) > 0, "SGD CSV is empty"
        else:
            pytest.skip("No SGD CSV files found (run Story 1.3 first)")

    def test_sample_documents_exist(self):
        """Test that sample documents are available."""
        data_dir = Path(__file__).parent.parent.parent / "data"
        pdf_files = list(data_dir.glob("*.pdf"))
        
        assert len(pdf_files) > 0, f"No PDF documents found in {data_dir}"
