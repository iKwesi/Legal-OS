"""
LangChain-based Qdrant vector store for Legal-OS.

This module provides vector store operations using LangChain's native
Qdrant integration for better performance and compatibility.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper for LangChain's Qdrant vector store.
    
    Provides a simplified interface for document storage and retrieval
    using LangChain's battle-tested Qdrant integration.
    """

    def __init__(self, use_memory: bool = True, collection_name: Optional[str] = None):
        """
        Initialize Qdrant vector store.
        
        Args:
            use_memory: If True, use in-memory Qdrant. Otherwise connect to remote.
                       Defaults to True for simpler development. Set to False for production.
            collection_name: Optional collection name (defaults to settings)
        """
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        
        # Create Qdrant client
        if use_memory:
            logger.info("Initializing in-memory Qdrant client")
            self.client = QdrantClient(":memory:")
            self.location = ":memory:"
            self.url = None
        else:
            logger.info(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
            self.location = None
            self.url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        
        # Initialize empty vectorstore (will be populated via from_documents)
        self.vectorstore = None
        
    def from_documents(
        self,
        documents: List[Document],
        session_id: Optional[str] = None,
    ) -> "VectorStore":
        """
        Create vector store from LangChain documents.
        
        Args:
            documents: List of LangChain Document objects
            session_id: Optional session ID to add to metadata
            
        Returns:
            Self for method chaining
        """
        try:
            # Add session_id to all document metadata if provided
            if session_id:
                for doc in documents:
                    doc.metadata["session_id"] = session_id
            
            logger.info(f"Creating vectorstore from {len(documents)} documents")
            
            # Create Qdrant vectorstore from documents
            if self.location == ":memory:":
                self.vectorstore = Qdrant.from_documents(
                    documents,
                    self.embeddings,
                    location=":memory:",
                    collection_name=self.collection_name,
                )
            else:
                self.vectorstore = Qdrant.from_documents(
                    documents,
                    self.embeddings,
                    url=self.url,
                    collection_name=self.collection_name,
                )
            
            logger.info(f"Vectorstore created with {len(documents)} documents")
            return self
            
        except Exception as e:
            logger.error(f"Error creating vectorstore from documents: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of LangChain Document objects
            session_id: Optional session ID to add to metadata
            
        Returns:
            List of document IDs
        """
        try:
            if not self.vectorstore:
                # If vectorstore doesn't exist, create it
                self.from_documents(documents, session_id)
                return [doc.metadata.get("chunk_id", str(i)) for i, doc in enumerate(documents)]
            
            # Add session_id to metadata if provided
            if session_id:
                for doc in documents:
                    doc.metadata["session_id"] = session_id
            
            logger.info(f"Adding {len(documents)} documents to vectorstore")
            ids = self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(ids)} documents")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get LangChain retriever interface.
        
        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 10})
            
        Returns:
            LangChain retriever object
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call from_documents() first.")
        
        # Default to k=10 based on evaluation results
        kwargs = search_kwargs or {"k": 10}
        logger.info(f"Creating retriever with search_kwargs: {kwargs}")
        
        return self.vectorstore.as_retriever(search_kwargs=kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results (default 10 based on evaluation)
            filter: Optional metadata filter
            
        Returns:
            List of matching documents
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call from_documents() first.")
        
        return self.vectorstore.similarity_search(query, k=k, filter=filter)
    
    def get_all_documents(self, session_id: Optional[str] = None) -> List[Document]:
        """
        Get all documents from the vector store.
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            List of all documents
        """
        if not self.vectorstore:
            logger.warning("Vectorstore not initialized, returning empty list")
            return []
        
        try:
            # Get collection info to know how many documents exist
            collection_info = self.client.get_collection(self.collection_name)
            total_docs = collection_info.points_count
            
            if total_docs == 0:
                return []
            
            # Use similarity search with high k to get all docs
            filter_dict = {"session_id": session_id} if session_id else None
            results = self.vectorstore.similarity_search(
                query="",  # Empty query
                k=total_docs,
                filter=filter_dict,
            )
            
            logger.info(f"Retrieved {len(results)} documents from vectorstore")
            return results
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []
