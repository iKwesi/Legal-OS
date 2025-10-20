"""
Qdrant vector store wrapper for document chunk storage and retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper for Qdrant vector store operations."""

    def __init__(self):
        """Initialize Qdrant client and embeddings."""
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = settings.qdrant_collection_name
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        session_id: str,
    ) -> List[str]:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with 'text', 'metadata', etc.
            session_id: Session identifier to associate chunks with

        Returns:
            List of chunk IDs that were added
        """
        try:
            if not chunks:
                logger.warning("No chunks provided to add_chunks")
                return []

            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.embeddings.embed_documents(texts)

            # Prepare points for Qdrant
            points = []
            chunk_ids = []

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = chunk.get("chunk_id", str(uuid4()))
                chunk_ids.append(chunk_id)

                # Prepare payload with metadata
                payload = {
                    "text": chunk["text"],
                    "chunk_id": chunk_id,
                    "document_id": chunk.get("document_id", ""),
                    "chunk_index": chunk.get("chunk_index", idx),
                    "session_id": session_id,
                    "metadata": chunk.get("metadata", {}),
                }

                points.append(
                    PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            # Upload to Qdrant
            logger.info(f"Uploading {len(points)} points to Qdrant")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Successfully added {len(chunk_ids)} chunks to vector store")
            return chunk_ids

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector store.

        Args:
            query: Search query text
            session_id: Optional session ID to filter results
            top_k: Number of results to return
            score_threshold: Optional minimum similarity score

        Returns:
            List of matching chunks with scores and metadata
        """
        try:
            # Generate query embedding
            logger.info(f"Searching for: {query[:100]}...")
            query_embedding = self.embeddings.embed_query(query)

            # Prepare filter if session_id provided
            query_filter = None
            if session_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id),
                        )
                    ]
                )

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold or settings.similarity_threshold,
            )

            # Format results
            results = []
            for result in search_results:
                results.append({
                    "chunk_id": result.payload.get("chunk_id"),
                    "document_id": result.payload.get("document_id"),
                    "text": result.payload.get("text"),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_index": result.payload.get("chunk_index"),
                })

            logger.info(f"Found {len(results)} matching chunks")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def delete_session(self, session_id: str) -> bool:
        """
        Delete all chunks associated with a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deletion was successful
        """
        try:
            logger.info(f"Deleting chunks for session: {session_id}")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id),
                        )
                    ]
                ),
            )
            logger.info(f"Successfully deleted chunks for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting session chunks: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
