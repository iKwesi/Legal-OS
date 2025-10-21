"""
Retriever implementations for RAG pipeline using LangChain patterns.

This module provides a swappable retriever architecture using LangChain's
BaseRetriever interface. All retrievers implement the standard LangChain
retriever interface for seamless integration with RAG pipelines.
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from abc import ABC

from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.rag.vector_store import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)

RetrieverType = Literal["naive", "bm25"]


def get_naive_retriever(
    vector_store: VectorStore,
    top_k: int = 10,
) -> Any:
    """
    Create a naive (vector similarity) retriever.
    
    Args:
        vector_store: VectorStore instance
        top_k: Number of results to return (default 10 based on evaluation)
        
    Returns:
        LangChain retriever object
    """
    logger.info(f"Creating naive retriever with k={top_k}")
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def get_bm25_retriever(
    documents: List[Document],
    top_k: int = 10,
) -> LangChainBM25Retriever:
    """
    Create a BM25 (keyword-based) retriever.
    
    Args:
        documents: List of LangChain Document objects
        top_k: Number of results to return (default 10)
        
    Returns:
        BM25Retriever instance
    """
    logger.info(f"Creating BM25 retriever with k={top_k}")
    retriever = LangChainBM25Retriever.from_documents(documents)
    retriever.k = top_k
    return retriever


def get_retriever(
    retriever_type: RetrieverType = "naive",
    vector_store: Optional[VectorStore] = None,
    documents: Optional[List[Document]] = None,
    top_k: int = 10,
    **kwargs,
) -> Any:
    """
    Factory function to create a retriever based on type.
    
    Args:
        retriever_type: Type of retriever ("naive" or "bm25")
        vector_store: VectorStore instance (required for naive retriever)
        documents: List of documents (required for BM25 retriever)
        top_k: Number of results to return (default 10 based on evaluation)
        **kwargs: Additional parameters (currently unused)
        
    Returns:
        LangChain retriever instance
        
    Raises:
        ValueError: If retriever_type is not recognized or required params missing
    """
    if retriever_type == "naive":
        if not vector_store:
            raise ValueError("vector_store is required for naive retriever")
        return get_naive_retriever(vector_store, top_k)
    
    elif retriever_type == "bm25":
        if not documents:
            raise ValueError("documents list is required for BM25 retriever")
        return get_bm25_retriever(documents, top_k)
    
    else:
        raise ValueError(
            f"Unknown retriever type: {retriever_type}. "
            f"Must be one of: 'naive', 'bm25'"
        )


# Backward compatibility: Keep old class names as aliases
class BaseRetriever:
    """Deprecated: Use get_retriever() factory function instead."""
    pass


class BM25Retriever:
    """
    Deprecated: Use get_bm25_retriever() instead.
    
    This class is kept for backward compatibility but should not be used in new code.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        vector_store: Optional[VectorStore] = None,
    ):
        """Initialize BM25 retriever."""
        self.k1 = k1
        self.b = b
        self.vector_store = vector_store or VectorStore()
        self.bm25_retriever = None
        self.documents_cache = []
        logger.warning(
            "BM25Retriever class is deprecated. "
            "Use get_bm25_retriever() instead."
        )
    
    def _initialize_bm25(self, session_id: Optional[str] = None):
        """Initialize BM25 with documents from vector store."""
        try:
            # Get all documents from vector store
            all_docs = self.vector_store.get_all_documents(session_id=session_id)
            
            if not all_docs:
                logger.warning("No documents found for BM25 initialization")
                return
            
            self.documents_cache = all_docs
            self.bm25_retriever = LangChainBM25Retriever.from_documents(all_docs)
            self.bm25_retriever.k = 10
            
            logger.info(f"BM25 initialized with {len(all_docs)} documents")
        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve using BM25."""
        k = top_k or 10
        
        if not self.bm25_retriever:
            self._initialize_bm25(session_id)
        
        if not self.bm25_retriever:
            return []
        
        self.bm25_retriever.k = k
        docs = self.bm25_retriever.get_relevant_documents(query)
        
        # Convert to old format
        results = []
        for idx, doc in enumerate(docs):
            results.append({
                "text": doc.page_content,
                "score": 1.0 - (idx / max(len(docs), 1)),
                "metadata": doc.metadata,
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "document_id": doc.metadata.get("document_id", ""),
                "chunk_index": doc.metadata.get("chunk_index", idx),
            })
        
        return results
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
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


class NaiveRetriever:
    """
    Deprecated: Use get_naive_retriever() or vector_store.as_retriever() instead.
    
    This class is kept for backward compatibility but should not be used in new code.
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Initialize with vector store."""
        self.vector_store = vector_store or VectorStore()
        logger.warning(
            "NaiveRetriever class is deprecated. "
            "Use vector_store.as_retriever() or get_naive_retriever() instead."
        )
    
    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve using LangChain retriever."""
        k = top_k or 10
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        
        # Convert to old format for compatibility
        results = []
        for idx, doc in enumerate(docs):
            results.append({
                "text": doc.page_content,
                "score": 1.0 - (idx / max(len(docs), 1)),  # Rank-based score
                "metadata": doc.metadata,
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "document_id": doc.metadata.get("document_id", ""),
                "chunk_index": doc.metadata.get("chunk_index", idx),
            })
        
        return results
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
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
