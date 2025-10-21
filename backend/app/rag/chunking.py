"""
Text chunking strategies for document processing.
"""

import logging
from typing import List, Dict, Any, Literal
from uuid import uuid4
from abc import ABC, abstractmethod

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["naive", "semantic"]


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

    def chunk_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document using the specific chunking strategy.

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

            # Split text into chunks using strategy-specific method
            text_chunks = self.split_text(text)

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


class NaiveChunker(BaseChunker):
    """
    Naive chunking using Recursive Character Text Splitting.
    
    This is the baseline chunking strategy that splits text based on
    character count with fixed overlap.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Initialize the Naive Chunker.

        Args:
            chunk_size: Size of each chunk (defaults to settings.chunk_size)
            chunk_overlap: Overlap between chunks (defaults to settings.chunk_overlap)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Initialize RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"Initialized NaiveChunker with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def split_text(self, text: str) -> List[str]:
        """Split text using RecursiveCharacterTextSplitter."""
        return self.splitter.split_text(text)


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using embedding-based similarity.
    
    This strategy identifies semantic boundaries in text by analyzing
    embedding similarities between sentences, creating chunks that
    preserve semantic coherence.
    
    Note: This is a placeholder implementation. The actual SemanticChunker
    from langchain_experimental will be used when available in the environment.
    For now, it falls back to sentence-based chunking.
    """

    def __init__(
        self,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        embedding_model: str | None = None,
    ):
        """
        Initialize the Semantic Chunker.

        Args:
            breakpoint_threshold_type: Type of threshold ("percentile", "standard_deviation", "interquartile")
            breakpoint_threshold_amount: Threshold value for breakpoint detection
            embedding_model: OpenAI embedding model (defaults to settings.embedding_model)
        """
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.embedding_model_name = embedding_model or settings.embedding_model

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=settings.openai_api_key,
        )

        # Try to import and use the actual SemanticChunker
        try:
            from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
            self.splitter = LangChainSemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
            )
            self.use_langchain_semantic = True
        except (ImportError, AttributeError):
            # Fallback to sentence-based splitting
            logger.warning(
                "SemanticChunker from langchain_experimental not available. "
                "Using sentence-based fallback implementation."
            )
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            )
            self.use_langchain_semantic = False

        logger.info(
            f"Initialized SemanticChunker (using_langchain={self.use_langchain_semantic}) "
            f"with threshold_type={breakpoint_threshold_type}, "
            f"threshold_amount={breakpoint_threshold_amount}, "
            f"embedding_model={self.embedding_model_name}"
        )

    def split_text(self, text: str) -> List[str]:
        """Split text using semantic similarity analysis or fallback."""
        return self.splitter.split_text(text)


def get_chunker(
    strategy: ChunkingStrategy = "naive",
    **kwargs,
) -> BaseChunker:
    """
    Factory function to create a chunker based on strategy.

    Args:
        strategy: Chunking strategy ("naive" or "semantic")
        **kwargs: Additional parameters for the specific chunker

    Returns:
        Chunker instance

    Raises:
        ValueError: If strategy is not recognized
    """
    if strategy == "naive":
        return NaiveChunker(**kwargs)
    elif strategy == "semantic":
        return SemanticChunker(**kwargs)
    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Must be one of: 'naive', 'semantic'"
        )


# Maintain backward compatibility with DocumentChunker
class DocumentChunker(NaiveChunker):
    """
    Legacy DocumentChunker class for backward compatibility.
    
    This class is now an alias for NaiveChunker.
    """
    pass
