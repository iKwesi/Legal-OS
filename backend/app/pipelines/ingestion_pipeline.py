"""
Ingestion Pipeline for document loading, processing, and storage.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from app.core.config import settings
from app.rag.chunking import DocumentChunker
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Pipeline responsible for document ingestion workflow.

    Handles:
    - Document loading from various formats (PDF, DOCX, TXT)
    - Text chunking using configured strategy
    - Embedding generation and storage in vector store
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def __init__(self):
        """Initialize the Ingestion Pipeline with chunker and vector store."""
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore()
        logger.info("IngestionPipeline initialized")

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a document from file path.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with document_id, file_name, content, and metadata

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        try:
            path = Path(file_path)

            # Validate file exists
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Validate file extension
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file format: {path.suffix}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
                )

            logger.info(f"Loading document: {file_path}")

            # Select appropriate loader based on file extension
            loader = self._get_loader(file_path)

            # Load document
            documents = loader.load()

            # Extract text content
            content = "\n\n".join([doc.page_content for doc in documents])

            # Get file metadata
            file_size = path.stat().st_size
            file_type = path.suffix.lower().lstrip(".")

            # Prepare metadata
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "file_type": file_type,
                "file_size": file_size,
                "total_pages": len(documents) if file_type == "pdf" else None,
            }

            # Generate document ID
            document_id = str(uuid4())

            logger.info(
                f"Successfully loaded document {path.name} "
                f"({len(content)} chars, {len(documents)} pages/sections)"
            )

            return {
                "document_id": document_id,
                "file_name": path.name,
                "content": content,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def _get_loader(self, file_path: str):
        """
        Get appropriate document loader based on file extension.

        Args:
            file_path: Path to the document

        Returns:
            LangChain document loader instance
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".pdf":
            return PyMuPDFLoader(file_path)
        elif extension == ".docx":
            return Docx2txtLoader(file_path)
        elif extension == ".txt":
            return TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    def ingest_document(
        self,
        file_path: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Complete ingestion pipeline for a single document.

        Args:
            file_path: Path to the document file
            session_id: Session identifier for grouping documents

        Returns:
            Dictionary with ingestion results including document_id, chunk_count, etc.
        """
        try:
            logger.info(f"Starting ingestion for: {file_path}")

            # Step 1: Load document
            document = self.load_document(file_path)

            # Step 2: Chunk document
            chunks = self.chunker.chunk_document(
                document_id=document["document_id"],
                text=document["content"],
                metadata=document["metadata"],
            )

            if not chunks:
                logger.warning(f"No chunks created for document: {file_path}")
                return {
                    "document_id": document["document_id"],
                    "file_name": document["file_name"],
                    "chunk_count": 0,
                    "status": "no_chunks",
                }

            # Step 3: Store chunks in vector store
            chunk_ids = self.vector_store.add_chunks(
                chunks=chunks,
                session_id=session_id,
            )

            logger.info(
                f"Successfully ingested {document['file_name']}: "
                f"{len(chunk_ids)} chunks stored"
            )

            return {
                "document_id": document["document_id"],
                "file_name": document["file_name"],
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            raise

    def ingest_documents(
        self,
        file_paths: List[str],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents in a batch.

        Args:
            file_paths: List of document file paths
            session_id: Session identifier for grouping documents

        Returns:
            Dictionary with batch ingestion results
        """
        try:
            logger.info(f"Starting batch ingestion for {len(file_paths)} documents")

            results = []
            total_chunks = 0
            failed_documents = []

            for file_path in file_paths:
                try:
                    result = self.ingest_document(
                        file_path=file_path,
                        session_id=session_id,
                    )
                    results.append(result)
                    total_chunks += result["chunk_count"]
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")
                    failed_documents.append({
                        "file_path": file_path,
                        "error": str(e),
                    })

            logger.info(
                f"Batch ingestion complete: {len(results)} successful, "
                f"{len(failed_documents)} failed, {total_chunks} total chunks"
            )

            return {
                "session_id": session_id,
                "total_documents": len(file_paths),
                "successful": len(results),
                "failed": len(failed_documents),
                "total_chunks": total_chunks,
                "results": results,
                "failures": failed_documents,
            }

        except Exception as e:
            logger.error(f"Error in batch ingestion: {e}")
            raise
