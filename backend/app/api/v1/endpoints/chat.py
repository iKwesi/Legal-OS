"""
Chat endpoint for RAG-based document Q&A.

This endpoint provides a chat interface for asking questions about analyzed documents.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.models.api import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Import query function from query endpoint to reuse RAG logic
from app.api.v1.endpoints.query import query_documents


class ChatRequest(BaseModel):
    """Request model for chat message."""
    message: str = Field(..., description="User's question or message")


class ChatResponse(BaseModel):
    """Response model for chat message."""
    role: str = Field(default="assistant", description="Role of the responder")
    content: str = Field(..., description="Response content")
    provenance: list = Field(default_factory=list, description="Source references")


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat_with_document(session_id: str, request: ChatRequest) -> ChatResponse:
    """
    Send a chat message and get a response from the RAG pipeline.
    
    This endpoint wraps the query endpoint to provide a chat-friendly interface.
    
    Args:
        session_id: Session identifier for the analyzed document
        request: Chat request with user message
        
    Returns:
        ChatResponse with assistant's answer and provenance
        
    Raises:
        HTTPException: If chat processing fails
    """
    try:
        logger.info(f"Chat request for session {session_id}: {request.message[:100]}...")
        
        # Create QueryRequest and call the existing query endpoint
        query_request = QueryRequest(
            session_id=session_id,
            query=request.message,
            top_k=5
        )
        
        # Reuse the query endpoint logic
        query_response = await query_documents(query_request)
        
        # Format provenance for chat response
        provenance = []
        for source_text in query_response.sources[:3]:  # Limit to top 3 sources
            provenance.append({
                "source": source_text[:100] + "..." if len(source_text) > 100 else source_text,
                "page": None  # Page info not available in current format
            })
        
        response = ChatResponse(
            role="assistant",
            content=query_response.answer,
            provenance=provenance
        )
        
        logger.info(f"Chat response generated for session {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}"
        )
