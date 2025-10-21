"""
Data models for agent outputs.

This module defines Pydantic models for structured outputs from various agents
in the Legal-OS system, including clause extraction, risk scoring, and more.
"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, UTC
from pydantic import BaseModel, Field


# Source Tracking Models

class SourceReference(BaseModel):
    """
    Reference to a specific source location in a document.
    
    Attributes:
        document_id: Unique identifier for the source document
        page: Page number in the original document (if applicable)
        section: Section or heading in the document
        chunk_id: Specific chunk identifier from vector store
        text_snippet: Short excerpt showing the source text
        confidence: Confidence score for the source attribution (0.0 to 1.0)
    """
    document_id: str = Field(
        description="Unique identifier for the source document"
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number in the original document"
    )
    section: Optional[str] = Field(
        default=None,
        description="Section or heading in the document"
    )
    chunk_id: str = Field(
        description="Specific chunk identifier from vector store"
    )
    text_snippet: str = Field(
        description="Short excerpt (100-200 chars) showing the source text"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Confidence score for the source attribution (0.0 to 1.0)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_id": "doc_abc123",
                    "page": 5,
                    "section": "Section 2.1 - Purchase Price",
                    "chunk_id": "chunk_123",
                    "text_snippet": "The purchase price shall be $10,000,000 payable in cash at closing.",
                    "confidence": 0.95
                }
            ]
        }
    }


class SourceMetadata(BaseModel):
    """
    Complete provenance information for an item.
    
    Attributes:
        sources: List of source references supporting this item
        confidence: Overall confidence score (0.0 to 1.0)
        extraction_method: Method used to extract/generate this item
        timestamp: When this provenance was recorded
    """
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="List of source references supporting this item"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Overall confidence score for this item (0.0 to 1.0)"
    )
    extraction_method: str = Field(
        default="llm_extraction",
        description="Method used to extract/generate this item (e.g., 'llm_extraction', 'rule_based', 'aggregation')"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this provenance was recorded"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sources": [
                        {
                            "document_id": "doc_abc123",
                            "page": 5,
                            "section": "Section 2.1",
                            "chunk_id": "chunk_123",
                            "text_snippet": "The purchase price shall be $10,000,000...",
                            "confidence": 0.95
                        }
                    ],
                    "confidence": 0.95,
                    "extraction_method": "llm_extraction",
                    "timestamp": "2025-10-21T00:00:00Z"
                }
            ]
        }
    }


class SourceLink(BaseModel):
    """
    Frontend-renderable link to a source.
    
    Attributes:
        link_id: Unique identifier for this link
        link_text: Display text for the link
        link_url: URL or reference to navigate to the source
        tooltip: Tooltip text with additional context
        document_id: ID of the source document
        page: Page number (if applicable)
        section: Section reference (if applicable)
    """
    link_id: str = Field(
        description="Unique identifier for this link"
    )
    link_text: str = Field(
        description="Display text for the link (e.g., 'Section 2.1, Page 5')"
    )
    link_url: str = Field(
        description="URL or reference to navigate to the source"
    )
    tooltip: str = Field(
        description="Tooltip text with additional context (e.g., text snippet)"
    )
    document_id: str = Field(
        description="ID of the source document"
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number in the document"
    )
    section: Optional[str] = Field(
        default=None,
        description="Section reference"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "link_id": "link_abc123",
                    "link_text": "Section 2.1, Page 5",
                    "link_url": "/documents/doc_abc123#page=5&section=2.1",
                    "tooltip": "The purchase price shall be $10,000,000...",
                    "document_id": "doc_abc123",
                    "page": 5,
                    "section": "Section 2.1"
                }
            ]
        }
    }


class SourcedItem(BaseModel):
    """
    Generic item with attached provenance metadata.
    
    Attributes:
        content: The actual content/data of the item
        provenance: Provenance metadata for this item
        item_type: Type of item (e.g., 'clause', 'finding', 'recommendation')
    """
    content: Dict[str, Any] = Field(
        description="The actual content/data of the item"
    )
    provenance: SourceMetadata = Field(
        description="Provenance metadata for this item"
    )
    item_type: str = Field(
        description="Type of item (e.g., 'clause', 'finding', 'recommendation')"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": {
                        "clause_text": "The purchase price shall be $10,000,000.",
                        "clause_type": "payment_terms"
                    },
                    "provenance": {
                        "sources": [
                            {
                                "document_id": "doc_abc123",
                                "page": 5,
                                "section": "Section 2.1",
                                "chunk_id": "chunk_123",
                                "text_snippet": "The purchase price shall be $10,000,000...",
                                "confidence": 0.95
                            }
                        ],
                        "confidence": 0.95,
                        "extraction_method": "llm_extraction",
                        "timestamp": "2025-10-21T00:00:00Z"
                    },
                    "item_type": "clause"
                }
            ]
        }
    }


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
        provenance: Provenance metadata for this clause
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
    provenance: Optional[SourceMetadata] = Field(
        default=None,
        description="Provenance metadata for this clause"
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
        provenance: Provenance metadata for this red flag
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
    provenance: Optional[SourceMetadata] = Field(
        default=None,
        description="Provenance metadata for this red flag"
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


# Summary Agent Models

class KeyFinding(BaseModel):
    """
    Represents a key finding from the document analysis.
    
    Attributes:
        finding: Description of the finding
        severity: Severity level (Low, Medium, High, Critical)
        clause_reference: Reference to related clause(s)
        impact: Business impact description
        provenance: Provenance metadata for this finding
    """
    finding: str = Field(
        description="Description of the key finding"
    )
    severity: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Severity level of the finding"
    )
    clause_reference: Optional[str] = Field(
        default=None,
        description="Reference to the related clause or section"
    )
    impact: str = Field(
        description="Description of the business impact"
    )
    provenance: Optional[SourceMetadata] = Field(
        default=None,
        description="Provenance metadata for this finding"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "finding": "Unlimited indemnification liability without cap",
                    "severity": "Critical",
                    "clause_reference": "Indemnification - Section 8.2",
                    "impact": "Exposes buyer to unlimited financial liability"
                }
            ]
        }
    }


class Recommendation(BaseModel):
    """
    Represents an actionable recommendation.
    
    Attributes:
        recommendation: The recommended action
        priority: Priority level (Low, Medium, High, Critical)
        rationale: Explanation of why this is recommended
        related_findings: References to related findings
        provenance: Provenance metadata for this recommendation
    """
    recommendation: str = Field(
        description="The recommended action to take"
    )
    priority: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Priority level for this recommendation"
    )
    rationale: str = Field(
        description="Explanation of why this recommendation is important"
    )
    related_findings: List[str] = Field(
        default_factory=list,
        description="References to related key findings"
    )
    provenance: Optional[SourceMetadata] = Field(
        default=None,
        description="Provenance metadata for this recommendation"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "recommendation": "Negotiate a cap on indemnification liability at 1-2x purchase price",
                    "priority": "Critical",
                    "rationale": "Unlimited liability exposes buyer to catastrophic financial risk",
                    "related_findings": ["Unlimited indemnification liability"]
                }
            ]
        }
    }


class ExecutiveSummary(BaseModel):
    """
    High-level executive summary of the document.
    
    Attributes:
        overview: Brief overview of the document and transaction
        critical_findings: Top 3-5 most critical findings
        primary_recommendations: Top 3-5 most important recommendations
        overall_risk_assessment: Overall risk level and brief explanation
    """
    overview: str = Field(
        description="Brief overview of the document and transaction"
    )
    critical_findings: List[str] = Field(
        description="Top 3-5 most critical findings requiring immediate attention"
    )
    primary_recommendations: List[str] = Field(
        description="Top 3-5 most important recommendations"
    )
    overall_risk_assessment: str = Field(
        description="Overall risk level and brief explanation"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "overview": "Asset purchase agreement for $10M acquisition with standard M&A terms",
                    "critical_findings": [
                        "Unlimited indemnification liability",
                        "Short 12-month survival period for warranties",
                        "Broad non-compete restrictions"
                    ],
                    "primary_recommendations": [
                        "Negotiate indemnification cap",
                        "Extend warranty survival period",
                        "Narrow non-compete scope"
                    ],
                    "overall_risk_assessment": "High risk (65/100) - Several critical issues require negotiation"
                }
            ]
        }
    }


class ClauseSummary(BaseModel):
    """
    Summary of a specific clause type.
    
    Attributes:
        clause_type: Type of clause (e.g., payment_terms, warranties)
        summary: Concise summary of the clause provisions
        risk_level: Risk level for this clause type
        key_points: Key points or notable provisions
    """
    clause_type: str = Field(
        description="Type of clause (e.g., 'payment_terms', 'warranties', 'indemnification')"
    )
    summary: str = Field(
        description="Concise summary of the clause provisions"
    )
    risk_level: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Risk level for this clause type"
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Key points or notable provisions in this clause"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "clause_type": "indemnification",
                    "summary": "Seller provides unlimited indemnification for breaches of warranties",
                    "risk_level": "Critical",
                    "key_points": [
                        "No cap on liability",
                        "12-month survival period",
                        "Broad scope of indemnification"
                    ]
                }
            ]
        }
    }


class DiligenceMemo(BaseModel):
    """
    Complete M&A diligence memo with all analysis sections.
    
    Attributes:
        executive_summary: High-level summary with critical findings
        clause_summaries: Summaries for each clause type
        key_findings: Detailed list of all key findings
        recommendations: Detailed list of all recommendations
        overall_assessment: Final assessment and recommendation
        metadata: Additional metadata about the memo generation
        timestamp: When the memo was generated
        document_id: ID of the document that was analyzed
    """
    executive_summary: ExecutiveSummary = Field(
        description="High-level executive summary"
    )
    clause_summaries: List[ClauseSummary] = Field(
        default_factory=list,
        description="Summaries for each clause type analyzed"
    )
    key_findings: List[KeyFinding] = Field(
        default_factory=list,
        description="Detailed list of all key findings"
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Detailed list of all recommendations"
    )
    overall_assessment: str = Field(
        description="Final assessment with recommendation (Proceed/Proceed with Caution/Do Not Proceed)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., processing time, model used)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the memo was generated"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="ID of the document that was analyzed"
    )
    
    def to_markdown(self) -> str:
        """
        Convert the diligence memo to a formatted Markdown document.
        
        Returns:
            Formatted Markdown string
        """
        md = "# M&A Due Diligence Memo\n\n"
        
        # Executive Summary
        md += "## Executive Summary\n\n"
        md += f"{self.executive_summary.overview}\n\n"
        md += f"**Overall Risk Assessment:** {self.executive_summary.overall_risk_assessment}\n\n"
        
        md += "### Critical Findings\n\n"
        for i, finding in enumerate(self.executive_summary.critical_findings, 1):
            md += f"{i}. {finding}\n"
        md += "\n"
        
        md += "### Primary Recommendations\n\n"
        for i, rec in enumerate(self.executive_summary.primary_recommendations, 1):
            md += f"{i}. {rec}\n"
        md += "\n"
        
        # Clause-by-Clause Analysis
        md += "## Clause-by-Clause Analysis\n\n"
        for clause_summary in self.clause_summaries:
            md += f"### {clause_summary.clause_type.replace('_', ' ').title()}\n\n"
            md += f"**Risk Level:** {clause_summary.risk_level}\n\n"
            md += f"{clause_summary.summary}\n\n"
            if clause_summary.key_points:
                md += "**Key Points:**\n\n"
                for point in clause_summary.key_points:
                    md += f"- {point}\n"
                md += "\n"
        
        # Key Findings
        md += "## Key Findings\n\n"
        for finding in self.key_findings:
            md += f"### {finding.finding}\n\n"
            md += f"**Severity:** {finding.severity}\n\n"
            if finding.clause_reference:
                md += f"**Reference:** {finding.clause_reference}\n\n"
            md += f"**Impact:** {finding.impact}\n\n"
        
        # Recommendations
        md += "## Recommendations\n\n"
        for rec in self.recommendations:
            md += f"### {rec.recommendation}\n\n"
            md += f"**Priority:** {rec.priority}\n\n"
            md += f"**Rationale:** {rec.rationale}\n\n"
            if rec.related_findings:
                md += f"**Related Findings:** {', '.join(rec.related_findings)}\n\n"
        
        # Overall Assessment
        md += "## Overall Assessment\n\n"
        md += f"{self.overall_assessment}\n\n"
        
        # Metadata
        md += "---\n\n"
        md += f"*Generated: {self.timestamp.isoformat()}*\n"
        if self.document_id:
            md += f"*Document ID: {self.document_id}*\n"
        
        return md
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "executive_summary": {
                        "overview": "Asset purchase agreement for $10M acquisition",
                        "critical_findings": ["Unlimited indemnification"],
                        "primary_recommendations": ["Negotiate cap"],
                        "overall_risk_assessment": "High risk (65/100)"
                    },
                    "clause_summaries": [],
                    "key_findings": [],
                    "recommendations": [],
                    "overall_assessment": "Proceed with Caution - Address critical issues before closing",
                    "metadata": {"processing_time_seconds": 45.2},
                    "timestamp": "2025-10-21T00:00:00Z",
                    "document_id": "doc_abc123"
                }
            ]
        }
    }
