"""
Text chunking strategies for document processing.
"""

import logging
from typing import List, Dict, Any
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Handles document chunking using various strategies."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Size of each chunk (defaults to settings.chunk_size)
            chunk_overlap: Overlap between chunks (defaults to settings.chunk_overlap)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Initialize RecursiveCharacterTextSplitter as default strategy
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"Initialized DocumentChunker with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def chunk_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document using Recursive Character Text Splitting.

        Args:
            document_id: Unique identifier for the document
            text: Full text content to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text, IDs, and metadata
        """
        try:
            if not text or not text.strip():
                logger.warning(f"Empty text provided for document {document_id}")
                return []

            logger.info(f"Chunking document {document_id} (length: {len(text)} chars)")

            # Split text into chunks
            text_chunks = self.splitter.split_text(text)

            logger.info(f"Created {len(text_chunks)} chunks for document {document_id}")

            # Create chunk dictionaries with metadata
            chunks = []
            for idx, chunk_text in enumerate(text_chunks):
                chunk_id = str(uuid4())

                # Prepare chunk metadata
                chunk_metadata = {
                    "source": metadata.get("source", "") if metadata else "",
                    "chunk_size": len(chunk_text),
                    "total_chunks": len(text_chunks),
                }

                # Add any additional metadata from the document
                if metadata:
                    chunk_metadata.update(metadata)

                chunk_dict = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": chunk_text,
                    "chunk_index": idx,
                    "metadata": chunk_metadata,
                }

                chunks.append(chunk_dict)

            logger.info(
                f"Successfully created {len(chunks)} chunks for document {document_id}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Error chunking document {document_id}: {e}")
            raise

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries with 'document_id', 'text', 'metadata'

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            document_id = doc.get("document_id")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            chunks = self.chunk_document(
                document_id=document_id,
                text=text,
                metadata=metadata,
            )
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks"
        )
        return all_chunks
