"""
Retriever implementations for RAG pipeline.
"""

import logging
from typing import List, Dict, Any, Optional

from app.rag.vector_store import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)


class NaiveRetriever:
    """
    Naive (similarity-based) retriever using vector similarity search.

    This is the baseline retriever that performs simple cosine similarity
    search in the vector store.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the Naive Retriever.

        Args:
            vector_store: Optional VectorStore instance (creates new if not provided)
        """
        self.vector_store = vector_store or VectorStore()
        logger.info("NaiveRetriever initialized")

    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using similarity search.

        Args:
            query: Search query text
            session_id: Optional session ID to filter results
            top_k: Number of results to return (defaults to settings.retrieval_top_k)
            score_threshold: Minimum similarity score (defaults to settings.similarity_threshold)

        Returns:
            List of retrieved chunks with scores and metadata
        """
        try:
            k = top_k or settings.retrieval_top_k
            threshold = score_threshold or settings.similarity_threshold

            logger.info(
                f"Retrieving top {k} chunks for query (threshold: {threshold})"
            )

            # Perform similarity search
            results = self.vector_store.search(
                query=query,
                session_id=session_id,
                top_k=k,
                score_threshold=threshold,
            )

            logger.info(f"Retrieved {len(results)} chunks")
            return results

        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            raise

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string for LLM.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries

        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return ""

        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            source = chunk.get("metadata", {}).get("file_name", "Unknown")

            context_parts.append(
                f"[Document {idx}] (Source: {source}, Relevance: {score:.3f})\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)
