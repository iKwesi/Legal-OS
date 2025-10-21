"""
Data models for orchestration state and workflow.

This module defines TypedDict models for the orchestration state used by
the LangGraph-based supervisor agent to coordinate multi-agent workflows.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from app.models.agent import (
    ExtractedClause,
    RiskScore,
    ScoredClause,
    DiligenceMemo,
    KeyFinding,
    Recommendation,
)


class ChecklistItem(TypedDict, total=False):
    """
    Represents a single checklist item.
    
    Attributes:
        question: The checklist question
        category: Category of the question
        priority: Priority level
        rationale: Why this question is important
    """
    question: str
    category: str
    priority: str
    rationale: str


class OrchestrationState(TypedDict, total=False):
    """
    State model for the orchestration workflow.
    
    This TypedDict defines all state fields that are passed between agents
    during the multi-agent orchestration process. The supervisor agent uses
    this state to track progress and route tasks to specialized agents.
    
    Attributes:
        # Document info
        document_id: Unique identifier for the document being processed
        document_path: File path to the document
        document_chunks: List of text chunks from the document
        
        # Workflow tracking
        current_step: Current step in the workflow
        completed_steps: List of completed workflow steps
        next_agent: Next agent to invoke (set by supervisor)
        
        # Agent outputs
        extracted_clauses: List of clauses extracted by Clause Extraction Agent
        risk_scores: List of risk scores from Risk Scoring Agent
        scored_clauses: List of clauses with risk scores attached
        summary: Diligence memo from Summary Agent
        provenance_data: Source tracking data from Provenance Agent
        checklist: List of follow-up questions from Checklist Agent
        
        # Error handling
        errors: List of error messages encountered during processing
        retry_count: Number of retries attempted for current step
        
        # Metadata
        metadata: Additional metadata about the orchestration
        started_at: Timestamp when orchestration started
        completed_at: Timestamp when orchestration completed (if done)
        
        # Messages (for LangGraph compatibility)
        messages: List of messages exchanged during orchestration
    """
    # Document info
    document_id: str
    document_path: Optional[str]
    document_chunks: List[str]
    
    # Workflow tracking
    current_step: str
    completed_steps: List[str]
    next_agent: Optional[str]
    
    # Agent outputs
    extracted_clauses: List[ExtractedClause]
    risk_scores: List[RiskScore]
    scored_clauses: List[ScoredClause]
    summary: Optional[DiligenceMemo]
    provenance_data: Dict[str, Any]
    checklist: List[ChecklistItem]
    
    # Error handling
    errors: List[str]
    retry_count: int
    
    # Metadata
    metadata: Dict[str, Any]
    started_at: str
    completed_at: Optional[str]
    
    # Messages (for LangGraph)
    messages: List[Any]
