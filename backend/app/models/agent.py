"""
Data models for agent outputs.

This module defines Pydantic models for structured outputs from various agents
in the Legal-OS system, including clause extraction, risk scoring, and more.
"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


# Clause Extraction Models

class ExtractedClause(BaseModel):
    """
    Represents a single extracted clause from a legal document.
    
    Attributes:
        clause_text: The actual text of the clause
        clause_type: Category of the clause (e.g., payment_terms, warranties)
        location: Location metadata (page, section, etc.)
        confidence: Confidence score for the extraction (0.0 to 1.0)
        source_chunk_ids: IDs of document chunks this clause was extracted from
    """
    clause_text: str = Field(
        description="The actual text content of the extracted clause"
    )
    clause_type: str = Field(
        description="Type/category of the clause (e.g., 'payment_terms', 'warranties', 'indemnification')"
    )
    location: Dict[str, Any] = Field(
        default_factory=dict,
        description="Location metadata such as page number, section, paragraph"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this extraction (0.0 to 1.0)"
    )
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs from which this clause was extracted"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "clause_text": "The purchase price shall be $10,000,000 payable in cash at closing.",
                    "clause_type": "payment_terms",
                    "location": {"page": 5, "section": "2.1"},
                    "confidence": 0.95,
                    "source_chunk_ids": ["chunk_123", "chunk_124"]
                }
            ]
        }
    }


class RedFlag(BaseModel):
    """
    Represents a detected red flag or potential issue in a legal document.
    
    Attributes:
        description: Description of the red flag
        severity: Severity level (Low, Medium, High, Critical)
        clause_reference: Reference to the related clause (if any)
        recommendation: Suggested action or mitigation
        category: Category of the red flag (e.g., 'missing_protection', 'unfavorable_terms')
    """
    description: str = Field(
        description="Detailed description of the red flag or issue"
    )
    severity: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Severity level of the red flag"
    )
    clause_reference: Optional[str] = Field(
        default=None,
        description="Reference to the related clause or section"
    )
    recommendation: str = Field(
        description="Recommended action or mitigation strategy"
    )
    category: str = Field(
        description="Category of the red flag (e.g., 'missing_protection', 'unfavorable_terms', 'ambiguous_language')"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "No cap on indemnification liability",
                    "severity": "Critical",
                    "clause_reference": "Section 8.2 - Indemnification",
                    "recommendation": "Negotiate a reasonable cap on indemnification obligations, typically 1-2x purchase price",
                    "category": "unfavorable_terms"
                }
            ]
        }
    }


class ClauseExtractionResult(BaseModel):
    """
    Complete result from the clause extraction agent.
    
    Attributes:
        clauses: List of extracted clauses
        red_flags: List of detected red flags
        metadata: Additional metadata about the extraction process
        timestamp: When the extraction was performed
        document_id: ID of the document that was analyzed
    """
    clauses: List[ExtractedClause] = Field(
        default_factory=list,
        description="List of all extracted clauses"
    )
    red_flags: List[RedFlag] = Field(
        default_factory=list,
        description="List of all detected red flags"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., processing time, model used, chunk count)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the extraction was performed"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="ID of the document that was analyzed"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "clauses": [
                        {
                            "clause_text": "The purchase price shall be $10,000,000.",
                            "clause_type": "payment_terms",
                            "location": {"page": 5},
                            "confidence": 0.95,
                            "source_chunk_ids": ["chunk_123"]
                        }
                    ],
                    "red_flags": [
                        {
                            "description": "No cap on indemnification",
                            "severity": "Critical",
                            "clause_reference": "Section 8.2",
                            "recommendation": "Negotiate a cap",
                            "category": "unfavorable_terms"
                        }
                    ],
                    "metadata": {
                        "processing_time_seconds": 12.5,
                        "model": "gpt-4o-mini",
                        "total_chunks_analyzed": 45
                    },
                    "timestamp": "2025-10-21T00:00:00Z",
                    "document_id": "doc_abc123"
                }
            ]
        }
    }
