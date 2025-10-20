"""
Document-related Pydantic models.
"""

from typing import Any, Dict
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    source: str = Field(..., description="Source file path or name")
    page: int | None = Field(default=None, description="Page number (for PDFs)")
    total_pages: int | None = Field(default=None, description="Total pages in document")
    file_type: str = Field(..., description="File type (pdf, docx, txt)")
    file_size: int | None = Field(default=None, description="File size in bytes")


class Document(BaseModel):
    """Internal representation of a document."""

    document_id: str = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    content: str = Field(..., description="Full text content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the document",
    )


class DocumentChunk(BaseModel):
    """Internal representation of a document chunk."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    text: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata including position, source, etc.",
    )
    chunk_index: int = Field(..., description="Index of chunk within document")
