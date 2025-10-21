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


# Risk Scoring Models

class RiskFactor(BaseModel):
    """
    Represents an individual risk factor identified in a clause.
    
    Attributes:
        factor_name: Name of the risk factor
        description: Description of why this is a risk
        score_impact: Numerical impact on risk score (0-100)
        detected: Whether this factor was detected in the clause
    """
    factor_name: str = Field(
        description="Name of the risk factor (e.g., 'unlimited_liability', 'short_survival_period')"
    )
    description: str = Field(
        description="Explanation of why this factor represents a risk"
    )
    score_impact: int = Field(
        ge=0,
        le=100,
        description="Impact on risk score if this factor is present (0-100)"
    )
    detected: bool = Field(
        default=False,
        description="Whether this risk factor was detected in the clause"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "factor_name": "unlimited_liability",
                    "description": "No cap on indemnification liability",
                    "score_impact": 50,
                    "detected": True
                }
            ]
        }
    }


class RiskScore(BaseModel):
    """
    Complete risk assessment for a clause.
    
    Attributes:
        score: Numerical risk score (0-100)
        category: Risk category (Low, Medium, High, Critical)
        factors: List of risk factors that contributed to the score
        justification: Explanation of the risk score
    """
    score: int = Field(
        ge=0,
        le=100,
        description="Numerical risk score (0-100)"
    )
    category: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Risk category based on score: Low (0-25), Medium (26-50), High (51-75), Critical (76-100)"
    )
    factors: List[RiskFactor] = Field(
        default_factory=list,
        description="List of risk factors that contributed to this score"
    )
    justification: str = Field(
        description="Detailed explanation of why this risk score was assigned"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "score": 75,
                    "category": "High",
                    "factors": [
                        {
                            "factor_name": "unlimited_liability",
                            "description": "No cap on indemnification",
                            "score_impact": 50,
                            "detected": True
                        }
                    ],
                    "justification": "High risk due to unlimited indemnification liability without any cap"
                }
            ]
        }
    }


class ScoredClause(BaseModel):
    """
    An extracted clause enriched with risk scoring information.
    
    Attributes:
        clause: The original extracted clause
        risk_score: The risk assessment for this clause
    """
    clause: ExtractedClause = Field(
        description="The original extracted clause"
    )
    risk_score: RiskScore = Field(
        description="Risk assessment for this clause"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "clause": {
                        "clause_text": "Seller shall indemnify Buyer for all losses.",
                        "clause_type": "indemnification",
                        "location": {"page": 8},
                        "confidence": 0.9,
                        "source_chunk_ids": ["chunk_200"]
                    },
                    "risk_score": {
                        "score": 75,
                        "category": "High",
                        "factors": [],
                        "justification": "Unlimited indemnification without cap"
                    }
                }
            ]
        }
    }


class RiskScoringResult(BaseModel):
    """
    Complete result from the risk scoring agent.
    
    Attributes:
        scored_clauses: List of clauses with risk scores
        overall_risk_score: Overall document risk score (0-100)
        overall_risk_category: Overall risk category
        metadata: Additional metadata about the scoring process
        timestamp: When the scoring was performed
        document_id: ID of the document that was analyzed
    """
    scored_clauses: List[ScoredClause] = Field(
        default_factory=list,
        description="List of all clauses with their risk scores"
    )
    overall_risk_score: int = Field(
        ge=0,
        le=100,
        description="Overall document risk score (0-100), calculated as average of clause scores"
    )
    overall_risk_category: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Overall risk category based on overall_risk_score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., processing time, model used)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the scoring was performed"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="ID of the document that was analyzed"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scored_clauses": [],
                    "overall_risk_score": 65,
                    "overall_risk_category": "High",
                    "metadata": {
                        "processing_time_seconds": 8.5,
                        "model": "gpt-4o-mini",
                        "total_clauses_scored": 5
                    },
                    "timestamp": "2025-10-21T00:00:00Z",
                    "document_id": "doc_abc123"
                }
            ]
        }
    }
