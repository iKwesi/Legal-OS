"""
Query endpoint for RAG question answering.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.api import QueryRequest, QueryResponse
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize RAG pipeline (singleton for the endpoint)
rag_pipeline = RAGPipeline()


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query documents using RAG.

    This endpoint:
    1. Receives a user question and session ID
    2. Retrieves relevant chunks from the vector store
    3. Generates an answer using the LLM
    4. Returns the answer with sources

    Args:
        request: QueryRequest with session_id, query, and optional top_k

    Returns:
        QueryResponse with answer, sources, and metadata

    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(
            f"Processing query for session {request.session_id}: {request.query[:100]}..."
        )

        # Execute RAG query
        result = rag_pipeline.query(
            question=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
        )

        # Format response
        response = QueryResponse(
            session_id=request.session_id,
            query=request.query,
            answer=result["answer"],
            sources=[source["text"] for source in result["sources"]],
            metadata=result["metadata"],
        )

        logger.info(f"Successfully processed query for session {request.session_id}")
        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )
