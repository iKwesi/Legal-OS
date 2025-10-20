"""
Tests for Synthetic Golden Dataset (SGD) Generation Script

This module contains unit and integration tests for the generate_sgd.py script,
including document loading, RAGAS integration, CSV output, and error handling.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import sys

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.scripts.generate_sgd import (
    load_documents_from_directory,
    generate_testset,
    save_testset_to_csv,
    parse_arguments,
)


class TestDocumentLoading:
    """Tests for document loading functionality."""

    def test_load_documents_from_directory_success(self, tmp_path):
        """Test successful loading of documents from directory."""
        # Create test directory with a text file
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        test_file = test_dir / "test.txt"
        test_file.write_text("This is a test document.")

        # Mock the document loaders
        with patch("app.scripts.generate_sgd.TextLoader") as mock_text_loader:
            mock_loader_instance = Mock()
            mock_loader_instance.load.return_value = [
                Mock(
                    page_content="This is a test document.",
                    metadata={}
                )
            ]
            mock_text_loader.return_value = mock_loader_instance

            documents = load_documents_from_directory(str(test_dir))

            assert len(documents) == 1
            assert documents[0].page_content == "This is a test document."
            assert documents[0].metadata["file_name"] == "test.txt"

    def test_load_documents_directory_not_found(self):
        """Test error handling when directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            load_documents_from_directory("/nonexistent/directory")

    def test_load_documents_not_a_directory(self, tmp_path):
        """Test error handling when path is not a directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Path is not a directory"):
            load_documents_from_directory(str(test_file))

    def test_load_documents_no_supported_files(self, tmp_path):
        """Test error handling when no supported files found."""
        test_dir = tmp_path / "empty_data"
        test_dir.mkdir()

        # No mocking needed - directory is actually empty
        with pytest.raises(ValueError, match="No supported documents found"):
            load_documents_from_directory(str(test_dir))

    def test_load_documents_handles_individual_failures(self, tmp_path):
        """Test that loading continues when individual documents fail."""
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        
        # Create two test files
        (test_dir / "good.txt").write_text("Good document")
        (test_dir / "bad.txt").write_text("Bad document")

        with patch("app.scripts.generate_sgd.TextLoader") as mock_text_loader:
            # First call succeeds, second fails
            mock_loader_good = Mock()
            mock_loader_good.load.return_value = [
                Mock(page_content="Good document", metadata={})
            ]
            
            mock_loader_bad = Mock()
            mock_loader_bad.load.side_effect = Exception("Failed to load")
            
            # Return different loaders for each file
            mock_text_loader.side_effect = [mock_loader_good, mock_loader_bad]

            documents = load_documents_from_directory(str(test_dir))

            # Should have loaded only the successful document
            assert len(documents) == 1
            assert documents[0].page_content == "Good document"


class TestTestsetGeneration:
    """Tests for RAGAS testset generation."""

    @patch("app.scripts.generate_sgd.TestsetGenerator")
    @patch("app.scripts.generate_sgd.ChatOpenAI")
    @patch("app.scripts.generate_sgd.OpenAIEmbeddings")
    def test_generate_testset_success(
        self, mock_embeddings, mock_chat, mock_generator_class
    ):
        """Test successful testset generation."""
        # Mock documents
        mock_docs = [
            Mock(page_content="Test content 1", metadata={}),
            Mock(page_content="Test content 2", metadata={}),
        ]

        # Mock generator
        mock_generator = Mock()
        mock_testset = Mock()
        mock_generator.generate_with_langchain_docs.return_value = mock_testset
        mock_generator_class.from_langchain.return_value = mock_generator

        # Generate testset
        result = generate_testset(
            documents=mock_docs,
            test_size=10,
        )

        # Verify generator was called correctly
        assert result == mock_testset
        mock_generator.generate_with_langchain_docs.assert_called_once()

    @patch("app.scripts.generate_sgd.TestsetGenerator")
    @patch("app.scripts.generate_sgd.ChatOpenAI")
    @patch("app.scripts.generate_sgd.OpenAIEmbeddings")
    def test_generate_testset_handles_errors(
        self, mock_embeddings, mock_chat, mock_generator_class
    ):
        """Test error handling during testset generation."""
        mock_docs = [Mock(page_content="Test", metadata={})]

        # Mock generator to raise exception
        mock_generator = Mock()
        mock_generator.generate_with_langchain_docs.side_effect = Exception(
            "API Error"
        )
        mock_generator_class.from_langchain.return_value = mock_generator

        with pytest.raises(Exception, match="Failed to generate testset"):
            generate_testset(
                documents=mock_docs,
                test_size=10,
            )


class TestCSVOutput:
    """Tests for CSV output functionality."""

    def test_save_testset_to_csv_success(self, tmp_path):
        """Test successful saving of testset to CSV."""
        output_file = tmp_path / "output" / "test.csv"

        # Mock testset with pandas DataFrame
        mock_testset = Mock()
        mock_df = pd.DataFrame({
            "user_input": ["Question 1", "Question 2"],
            "reference_contexts": [["Context 1"], ["Context 2"]],
            "reference": ["Answer 1", "Answer 2"],
            "synthesizer_name": ["simple", "reasoning"],
        })
        mock_testset.to_pandas.return_value = mock_df

        # Save testset
        save_testset_to_csv(mock_testset, str(output_file))

        # Verify file was created
        assert output_file.exists()

        # Verify content
        saved_df = pd.read_csv(output_file)
        assert len(saved_df) == 2
        assert list(saved_df.columns) == [
            "user_input",
            "reference_contexts",
            "reference",
            "synthesizer_name",
        ]

    def test_save_testset_creates_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_file = tmp_path / "new_dir" / "subdir" / "test.csv"

        mock_testset = Mock()
        mock_df = pd.DataFrame({
            "user_input": ["Q1"],
            "reference_contexts": [["C1"]],
            "reference": ["A1"],
            "synthesizer_name": ["simple"],
        })
        mock_testset.to_pandas.return_value = mock_df

        save_testset_to_csv(mock_testset, str(output_file))

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_save_testset_handles_missing_columns(self, tmp_path):
        """Test warning when expected columns are missing."""
        output_file = tmp_path / "test.csv"

        mock_testset = Mock()
        mock_df = pd.DataFrame({
            "user_input": ["Q1"],
            "reference": ["A1"],
            # Missing reference_contexts and synthesizer_name
        })
        mock_testset.to_pandas.return_value = mock_df

        # Should still save but log warning
        save_testset_to_csv(mock_testset, str(output_file))

        assert output_file.exists()


class TestArgumentParsing:
    """Tests for command-line argument parsing."""

    def test_parse_arguments_defaults(self):
        """Test default argument values."""
        with patch("sys.argv", ["generate_sgd.py"]):
            args = parse_arguments()

            assert args.test_size == 3
            assert args.output is None  # Default is None, timestamp added in main()
            assert args.data_dir == "data/"
            assert args.verbose is False

    def test_parse_arguments_custom_values(self):
        """Test parsing custom argument values."""
        with patch(
            "sys.argv",
            [
                "generate_sgd.py",
                "--test-size",
                "100",
                "--output",
                "custom.csv",
                "--data-dir",
                "custom_data/",
                "--verbose",
            ],
        ):
            args = parse_arguments()

            assert args.test_size == 100
            assert args.output == "custom.csv"
            assert args.data_dir == "custom_data/"
            assert args.verbose is True


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch("app.scripts.generate_sgd.TestsetGenerator")
    @patch("app.scripts.generate_sgd.ChatOpenAI")
    @patch("app.scripts.generate_sgd.OpenAIEmbeddings")
    def test_end_to_end_workflow(
        self, mock_embeddings, mock_chat, mock_generator_class, tmp_path
    ):
        """Test complete workflow from loading to saving."""
        # Setup test data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test.txt").write_text("Test document content")

        # Setup output path
        output_file = tmp_path / "output" / "sgd.csv"

        # Mock document loaders
        with patch("app.scripts.generate_sgd.PyMuPDFLoader"), \
             patch("app.scripts.generate_sgd.Docx2txtLoader"), \
             patch("app.scripts.generate_sgd.TextLoader") as mock_text_loader:
            
            # Mock TextLoader to return a document
            mock_loader_instance = Mock()
            mock_loader_instance.load.return_value = [
                Mock(page_content="Test document content", metadata={})
            ]
            mock_text_loader.return_value = mock_loader_instance

            # Mock RAGAS generator
            mock_generator = Mock()
            mock_testset = Mock()
            mock_df = pd.DataFrame({
                "user_input": ["Q1", "Q2"],
                "reference_contexts": [["C1"], ["C2"]],
                "reference": ["A1", "A2"],
                "synthesizer_name": ["simple", "reasoning"],
            })
            mock_testset.to_pandas.return_value = mock_df
            mock_generator.generate_with_langchain_docs.return_value = mock_testset
            mock_generator_class.from_langchain.return_value = mock_generator

            # Execute workflow
            documents = load_documents_from_directory(str(data_dir))
            testset = generate_testset(
                documents=documents,
                test_size=2,
            )
            save_testset_to_csv(testset, str(output_file))

            # Verify results
            assert output_file.exists()
            saved_df = pd.read_csv(output_file)
            assert len(saved_df) == 2
            assert "user_input" in saved_df.columns
