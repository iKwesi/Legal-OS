"""
API request/response Pydantic models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import UploadFile

from app.models.retriever import RetrieverConfig


class Upload(BaseModel):
    """Request model for document upload (used for validation)."""

    # Note: FastAPI handles UploadFile directly in endpoint parameters
    # This model is for documentation purposes
    pass


class UploadResponse(BaseModel):
    """Response model for document upload."""

    session_id: str = Field(..., description="Unique session identifier for this upload")
    file_names: List[str] = Field(..., description="List of uploaded file names")
    message: str = Field(
        default="Documents uploaded and ingestion started",
        description="Status message",
    )


class QueryRequest(BaseModel):
    """Request model for RAG query with optional retriever configuration."""

    session_id: str = Field(..., description="Session identifier for document context")
    query: str = Field(..., description="User question or query")
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve (deprecated: use retriever_config.top_k instead)",
        ge=1,
        le=20,
    )
    retriever_config: Optional[RetrieverConfig] = Field(
        default=None,
        description="Optional retriever configuration for swappable retrieval strategies"
    )


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(
        default_factory=list,
        description="Source chunks used for answer generation",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (e.g., retrieval scores, latency)",
    )
