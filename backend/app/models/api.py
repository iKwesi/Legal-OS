"""
API request/response Pydantic models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import UploadFile
from datetime import datetime

from app.models.retriever import RetrieverConfig
from app.models.agent import ExtractedClause, ScoredClause, DiligenceMemo


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


# Story 5.0: Orchestration API Models


class AnalyzeRequest(BaseModel):
    """Request model for triggering document analysis."""

    session_id: str = Field(
        ...,
        description="Session identifier from document upload",
        min_length=1
    )


class AnalyzeResponse(BaseModel):
    """Response model for analysis request."""

    status_url: str = Field(
        ...,
        description="URL to poll for analysis status"
    )
    session_id: str = Field(
        ...,
        description="Session identifier for this analysis"
    )


class StatusResponse(BaseModel):
    """Response model for analysis status polling."""

    status: str = Field(
        ...,
        description="Current status: 'pending', 'processing', 'completed', or 'failed'"
    )
    progress: int = Field(
        ...,
        description="Progress percentage (0-100)",
        ge=0,
        le=100
    )
    current_step: Optional[str] = Field(
        None,
        description="Current processing step"
    )
    message: Optional[str] = Field(
        None,
        description="Additional status message or error details"
    )


class RedFlag(BaseModel):
    """Red flag identified in the document."""

    flag_id: str = Field(..., description="Unique identifier for this red flag")
    description: str = Field(..., description="Description of the red flag")
    risk_score: str = Field(..., description="Risk level: Low, Medium, High, Critical")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source tracking information"
    )


class ChecklistItem(BaseModel):
    """Follow-up checklist item."""

    item_id: str = Field(..., description="Unique identifier for this checklist item")
    text: str = Field(..., description="Checklist question or item text")
    related_flag_id: Optional[str] = Field(
        None,
        description="Related red flag ID if applicable"
    )


class AnalysisReport(BaseModel):
    """Complete analysis report with all agent outputs."""

    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Analysis status")
    summary_memo: str = Field(..., description="Executive summary memo")
    extracted_clauses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of extracted clauses with metadata"
    )
    red_flags: List[RedFlag] = Field(
        default_factory=list,
        description="List of identified red flags"
    )
    checklist: List[ChecklistItem] = Field(
        default_factory=list,
        description="Follow-up checklist items"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing metadata (timing, counts, etc.)"
    )
    created_at: str = Field(..., description="Analysis start timestamp (ISO format)")
    completed_at: Optional[str] = Field(
        None,
        description="Analysis completion timestamp (ISO format)"
    )
