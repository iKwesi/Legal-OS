"""
Query endpoint for RAG question answering with swappable retriever support.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.api import QueryRequest, QueryResponse
from app.rag.pipeline import RAGPipeline
from app.rag.shared import get_shared_vector_store, get_shared_rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query documents using RAG with optional retriever configuration.

    This endpoint supports swappable retriever strategies:
    1. Receives a user question, session ID, and optional retriever configuration
    2. Creates a RAG pipeline with the specified retriever (or uses default)
    3. Retrieves relevant chunks from the vector store
    4. Generates an answer using the LLM
    5. Returns the answer with sources

    Args:
        request: QueryRequest with session_id, query, optional top_k, and optional retriever_config

    Returns:
        QueryResponse with answer, sources, and metadata

    Raises:
        HTTPException: If query processing fails
        
    Examples:
        Default (naive) retriever:
        ```json
        {
            "session_id": "session123",
            "query": "What are the payment terms?"
        }
        ```
        
        Custom BM25 retriever:
        ```json
        {
            "session_id": "session123",
            "query": "What are the payment terms?",
            "retriever_config": {
                "retriever_type": "bm25",
                "top_k": 10,
                "params": {"k1": 1.5, "b": 0.75}
            }
        }
        ```
    """
    try:
        logger.info(
            f"Processing query for session {request.session_id}: {request.query[:100]}..."
        )

        # Determine which pipeline to use
        if request.retriever_config:
            # Create a new pipeline with the specified retriever configuration
            logger.info(
                f"Using custom retriever: {request.retriever_config.get_description()}"
            )
            pipeline = RAGPipeline(
                retriever_config=request.retriever_config,
                vector_store=get_shared_vector_store(),
            )
        else:
            # Use shared default pipeline
            logger.info("Using shared default naive retriever")
            pipeline = get_shared_rag_pipeline()

        # Execute RAG query
        result = pipeline.query(
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
