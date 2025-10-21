"""
Analysis endpoints for orchestration workflow.

This module provides endpoints for triggering document analysis,
polling status, and retrieving results.
"""

import logging
from typing import Dict, Any, List
from threading import Lock
from datetime import datetime, UTC
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, status

from app.models.api import (
    AnalyzeRequest,
    AnalyzeResponse,
    StatusResponse,
    AnalysisReport,
    RedFlag,
    ChecklistItem,
)
from app.orchestration.pipeline import DocumentOrchestrator
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session stores (MVP implementation)
# TODO: Post-MVP - Replace with Redis or database for production
session_states: Dict[str, dict] = {}
session_results: Dict[str, dict] = {}
session_documents: Dict[str, List[str]] = {}  # Store document paths by session_id
session_lock = Lock()

# Progress mapping for workflow steps
STEP_PROGRESS = {
    "start": 0,
    "ingestion": 20,
    "clause_extraction": 40,
    "risk_scoring": 60,
    "summary": 80,
    "provenance": 90,
    "checklist": 100,
}


def get_session_state(session_id: str) -> dict:
    """
    Get session state from in-memory store.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session state dictionary
        
    Raises:
        HTTPException: If session not found
    """
    with session_lock:
        if session_id not in session_states:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        return session_states[session_id].copy()


def update_session_state(session_id: str, state: dict) -> None:
    """
    Update session state in in-memory store.
    
    Args:
        session_id: Session identifier
        state: Updated state dictionary
    """
    with session_lock:
        session_states[session_id] = state


def get_session_results(session_id: str) -> dict:
    """
    Get session results from in-memory store.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session results dictionary
        
    Raises:
        HTTPException: If results not found
    """
    with session_lock:
        if session_id not in session_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Results for session {session_id} not found"
            )
        return session_results[session_id].copy()


def store_session_document(session_id: str, document_path: str) -> None:
    """
    Store document path for a session.
    
    Args:
        session_id: Session identifier
        document_path: Path to the uploaded document
    """
    with session_lock:
        if session_id not in session_documents:
            session_documents[session_id] = []
        session_documents[session_id].append(document_path)


def get_session_document(session_id: str) -> str:
    """
    Get document path for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Document path
        
    Raises:
        HTTPException: If no document found for session
    """
    with session_lock:
        if session_id not in session_documents or not session_documents[session_id]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found for session {session_id}"
            )
        # Return first document (MVP supports single document per session)
        return session_documents[session_id][0]


def calculate_progress(completed_steps: List[str]) -> int:
    """
    Calculate progress percentage based on completed steps.
    
    Args:
        completed_steps: List of completed workflow steps
        
    Returns:
        Progress percentage (0-100)
    """
    if not completed_steps:
        return 0
    
    # Get max progress from completed steps
    max_progress = 0
    for step in completed_steps:
        step_progress = STEP_PROGRESS.get(step, 0)
        if step_progress > max_progress:
            max_progress = step_progress
    
    return max_progress


def run_orchestration_task(session_id: str) -> None:
    """
    Background task to run document orchestration.
    
    Args:
        session_id: Session identifier
    """
    try:
        logger.info(f"Starting orchestration for session {session_id}")
        
        # Get document path
        document_path = get_session_document(session_id)
        
        # Update state to processing
        with session_lock:
            session_states[session_id]["status"] = "processing"
            session_states[session_id]["current_step"] = "ingestion"
            session_states[session_id]["updated_at"] = datetime.now(UTC).isoformat()
        
        # Run orchestrator
        orchestrator = DocumentOrchestrator()
        results = orchestrator.run_orchestration(
            document_path=document_path,
            document_id=session_id
        )
        
        # Store results
        with session_lock:
            session_results[session_id] = results
            
            # Update state based on results
            if results.get("status") == "failed":
                session_states[session_id]["status"] = "failed"
                session_states[session_id]["message"] = "; ".join(results.get("errors", []))
                session_states[session_id]["progress"] = 0
            else:
                session_states[session_id]["status"] = "completed"
                session_states[session_id]["progress"] = 100
                session_states[session_id]["current_step"] = "complete"
                session_states[session_id]["completed_steps"] = results.get("completed_steps", [])
            
            session_states[session_id]["updated_at"] = datetime.now(UTC).isoformat()
        
        logger.info(f"Orchestration completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in orchestration task for session {session_id}: {e}", exc_info=True)
        
        # Update state to failed
        with session_lock:
            session_states[session_id]["status"] = "failed"
            session_states[session_id]["message"] = str(e)
            session_states[session_id]["updated_at"] = datetime.now(UTC).isoformat()


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_201_CREATED)
async def analyze_document(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
) -> AnalyzeResponse:
    """
    Trigger full document analysis for a session.
    
    This endpoint:
    1. Validates the session exists (from upload)
    2. Initializes analysis state
    3. Triggers orchestration in background
    4. Returns status URL for polling
    
    Args:
        request: Analysis request with session_id
        background_tasks: FastAPI background tasks
        
    Returns:
        AnalyzeResponse with status_url and session_id
        
    Raises:
        HTTPException: If session not found or already processing
    """
    try:
        session_id = request.session_id
        
        logger.info(f"Received analysis request for session {session_id}")
        
        # Validate session exists (has uploaded document)
        try:
            get_session_document(session_id)
        except HTTPException:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found. Please upload a document first."
            )
        
        # Check if analysis already in progress or completed
        with session_lock:
            if session_id in session_states:
                existing_state = session_states[session_id]
                if existing_state["status"] in ["processing", "completed"]:
                    logger.warning(f"Analysis already {existing_state['status']} for session {session_id}")
                    return AnalyzeResponse(
                        status_url=f"/api/v1/status/{session_id}",
                        session_id=session_id
                    )
        
        # Initialize session state
        initial_state = {
            "session_id": session_id,
            "status": "pending",
            "current_step": "start",
            "completed_steps": [],
            "progress": 0,
            "message": None,
            "started_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        
        with session_lock:
            session_states[session_id] = initial_state
        
        # Trigger orchestration in background
        background_tasks.add_task(run_orchestration_task, session_id)
        
        logger.info(f"Analysis triggered for session {session_id}")
        
        return AnalyzeResponse(
            status_url=f"/api/v1/status/{session_id}",
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error triggering analysis: {str(e)}"
        )


@router.get("/status/{session_id}", response_model=StatusResponse)
async def get_analysis_status(session_id: str) -> StatusResponse:
    """
    Get analysis status for a session.
    
    This endpoint provides pollable status updates including:
    - Current status (pending, processing, completed, failed)
    - Progress percentage (0-100)
    - Current processing step
    - Error messages if failed
    
    Args:
        session_id: Session identifier
        
    Returns:
        StatusResponse with current status and progress
        
    Raises:
        HTTPException: If session not found
    """
    try:
        logger.debug(f"Status check for session {session_id}")
        
        # Get session state
        state = get_session_state(session_id)
        
        # Calculate progress if processing
        if state["status"] == "processing":
            completed_steps = state.get("completed_steps", [])
            progress = calculate_progress(completed_steps)
            state["progress"] = progress
        
        return StatusResponse(
            status=state["status"],
            progress=state["progress"],
            current_step=state.get("current_step"),
            message=state.get("message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving status: {str(e)}"
        )


@router.get("/results/{session_id}", response_model=AnalysisReport)
async def get_analysis_results(session_id: str) -> AnalysisReport:
    """
    Get complete analysis results for a session.
    
    This endpoint returns the full analysis report including:
    - Summary memo
    - Extracted clauses
    - Red flags
    - Checklist items
    - Processing metadata
    
    Args:
        session_id: Session identifier
        
    Returns:
        AnalysisReport with all agent outputs
        
    Raises:
        HTTPException: If session not found or analysis not complete
    """
    try:
        logger.info(f"Results request for session {session_id}")
        
        # Check session state
        state = get_session_state(session_id)
        
        # Validate analysis is complete
        if state["status"] == "pending":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Analysis has not been started. Please call /analyze first."
            )
        elif state["status"] == "processing":
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Analysis is still in progress. Please poll /status endpoint."
            )
        elif state["status"] == "failed":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {state.get('message', 'Unknown error')}"
            )
        
        # Get results
        results = get_session_results(session_id)
        
        # Transform to AnalysisReport format
        summary = results.get("summary")
        summary_text = ""
        if summary:
            logger.info(f"Summary type: {type(summary)}, has overview: {hasattr(summary, 'overview')}")
            # Handle different summary formats
            if hasattr(summary, "executive_summary"):
                # DiligenceMemo format - but executive_summary might be an object!
                exec_summary = summary.executive_summary
                logger.info(f"exec_summary type: {type(exec_summary)}, is string: {isinstance(exec_summary, str)}")
                if isinstance(exec_summary, str):
                    summary_text = exec_summary
                    logger.info("exec_summary is already a string")
                elif hasattr(exec_summary, "overview"):
                    # It's an ExecutiveSummary object nested inside DiligenceMemo
                    overview = exec_summary.overview if hasattr(exec_summary, "overview") else ""
                    assessment = exec_summary.overall_risk_assessment if hasattr(exec_summary, "overall_risk_assessment") else ""
                    summary_text = f"{overview}\n\nRisk Assessment: {assessment}"
                    logger.info(f"Converted ExecutiveSummary to string, length: {len(summary_text)}")
                else:
                    summary_text = str(exec_summary)
                    logger.info("Used str() fallback")
                logger.info(f"DiligenceMemo format - summary_text type after conversion: {type(summary_text)}")
            elif hasattr(summary, "overview"):
                # ExecutiveSummary format - convert to string
                overview = getattr(summary, "overview", "")
                assessment = getattr(summary, "assessment", "")
                summary_text = f"{overview}\n\nRisk Assessment: {assessment}"
                logger.info(f"Using ExecutiveSummary format, text length: {len(summary_text)}")
            elif isinstance(summary, dict):
                # Dictionary format
                if "executive_summary" in summary:
                    summary_text = summary["executive_summary"]
                elif "overview" in summary:
                    overview = summary.get("overview", "")
                    assessment = summary.get("assessment", "")
                    summary_text = f"{overview}\n\nRisk Assessment: {assessment}"
                logger.info("Using dict format")
            elif isinstance(summary, str):
                # Already a string
                summary_text = summary
                logger.info("Summary already string")
            else:
                # Fallback: convert to string
                summary_text = str(summary)
                logger.info("Using fallback str() conversion")
        
        logger.info(f"Final summary_text type: {type(summary_text)}, is string: {isinstance(summary_text, str)}")
        
        # Extract clauses
        extracted_clauses = []
        for clause in results.get("extracted_clauses", []):
            if hasattr(clause, "model_dump"):
                extracted_clauses.append(clause.model_dump())
            elif isinstance(clause, dict):
                extracted_clauses.append(clause)
        
        # Build red flags from scored clauses
        red_flags = []
        for scored_clause in results.get("scored_clauses", []):
            if hasattr(scored_clause, "risk_score"):
                risk_score = scored_clause.risk_score
                # Only include high/critical risks as red flags
                if hasattr(risk_score, "risk_category") and risk_score.risk_category in ["High", "Critical"]:
                    red_flag = RedFlag(
                        flag_id=f"flag_{hash(scored_clause.clause.clause_text) % 10000}",
                        description=scored_clause.clause.clause_text[:200],
                        risk_score=risk_score.risk_category,
                        provenance={}
                    )
                    red_flags.append(red_flag)
        
        # Build checklist
        checklist_items = []
        for idx, item in enumerate(results.get("checklist", [])):
            if isinstance(item, dict):
                checklist_item = ChecklistItem(
                    item_id=f"item_{idx}",
                    text=item.get("question", item.get("item", "")),
                    related_flag_id=None
                )
                checklist_items.append(checklist_item)
        
        return AnalysisReport(
            session_id=session_id,
            status=state["status"],
            summary_memo=summary_text,
            extracted_clauses=extracted_clauses,
            red_flags=red_flags,
            checklist=checklist_items,
            metadata=results.get("metadata", {}),
            created_at=state.get("started_at", datetime.now(UTC).isoformat()),
            completed_at=results.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in results endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving results: {str(e)}"
        )
