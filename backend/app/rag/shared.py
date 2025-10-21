"""
Shared RAG components for Legal-OS.

This module provides singleton instances of RAG components to ensure
data consistency across different endpoints.
"""

import logging
from typing import Optional

from app.rag.vector_store import VectorStore
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Global shared instances
_SHARED_VECTOR_STORE: Optional[VectorStore] = None
_SHARED_RAG_PIPELINE: Optional[RAGPipeline] = None


def get_shared_vector_store() -> VectorStore:
    """
    Get or create the shared vector store instance.
    
    This ensures all endpoints use the same in-memory vector store,
    allowing ingestion and query to share data.
    
    Returns:
        Shared VectorStore instance
    """
    global _SHARED_VECTOR_STORE
    if _SHARED_VECTOR_STORE is None:
        logger.info("Creating shared vector store instance")
        _SHARED_VECTOR_STORE = VectorStore(use_memory=True)
    return _SHARED_VECTOR_STORE


def get_shared_rag_pipeline() -> RAGPipeline:
    """
    Get or create the shared RAG pipeline instance.
    
    Uses the shared vector store to ensure consistency.
    
    Returns:
        Shared RAGPipeline instance
    """
    global _SHARED_RAG_PIPELINE
    if _SHARED_RAG_PIPELINE is None:
        logger.info("Creating shared RAG pipeline instance")
        vector_store = get_shared_vector_store()
        _SHARED_RAG_PIPELINE = RAGPipeline(vector_store=vector_store)
    return _SHARED_RAG_PIPELINE


def reset_shared_instances():
    """
    Reset shared instances (useful for testing).
    """
    global _SHARED_VECTOR_STORE, _SHARED_RAG_PIPELINE
    _SHARED_VECTOR_STORE = None
    _SHARED_RAG_PIPELINE = None
    logger.info("Reset shared RAG instances")
