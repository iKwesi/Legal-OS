"""
Document Orchestrator using LangGraph for multi-agent coordination.

This module implements a supervisor-based orchestration pattern using LangGraph
to coordinate specialized agents for M&A document analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, UTC
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models.orchestration import OrchestrationState, ChecklistItem
from app.models.agent import (
    ClauseExtractionResult,
    RiskScoringResult,
    DiligenceMemo,
    ExtractedClause,
    ScoredClause,
)
from app.core.config import settings

# Import agents
from app.agents.clause_extraction import ClauseExtractionAgent
from app.agents.risk_scoring import RiskScoringAgent
from app.agents.summary import SummaryAgent
from app.agents.checklist import ChecklistAgent
from app.utils.source_tracker import SourceTracker
from app.rag.vector_store import VectorStore
from app.pipelines.ingestion_pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


class DocumentOrchestrator:
    """
    Orchestrator for coordinating multi-agent document analysis workflow.
    
    Uses LangGraph StateGraph with a supervisor agent to route tasks to
    specialized agents in a ReAct-style workflow.
    
    Workflow:
        START → Ingestion → Clause Extraction → Risk Scoring → 
        Summary → Provenance → Checklist → END
    """
    
    # Workflow step constants
    STEP_START = "start"
    STEP_INGESTION = "ingestion"
    STEP_CLAUSE_EXTRACTION = "clause_extraction"
    STEP_RISK_SCORING = "risk_scoring"
    STEP_SUMMARY = "summary"
    STEP_PROVENANCE = "provenance"
    STEP_CHECKLIST = "checklist"
    STEP_COMPLETE = "complete"
    
    # Agent names
    AGENT_SUPERVISOR = "supervisor"
    AGENT_INGESTION = "ingestion_agent"
    AGENT_CLAUSE_EXTRACTION = "clause_extraction_agent"
    AGENT_RISK_SCORING = "risk_scoring_agent"
    AGENT_SUMMARY = "summary_agent"
    AGENT_PROVENANCE = "provenance_agent"
    AGENT_CHECKLIST = "checklist_agent"
    
    # Maximum iterations to prevent infinite loops
    MAX_ITERATIONS = 50
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize the document orchestrator.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation
            vector_store: Optional vector store instance (creates new if None)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key
        )
        
        # Initialize vector store
        self.vector_store = vector_store or VectorStore()
        
        # Initialize agents
        self.ingestion_pipeline = IngestionPipeline(vector_store=self.vector_store)
        self.clause_extraction_agent = ClauseExtractionAgent(
            vector_store=self.vector_store,
            model_name=model_name,
            temperature=temperature
        )
        self.risk_scoring_agent = RiskScoringAgent(
            model_name=model_name,
            temperature=temperature
        )
        self.summary_agent = SummaryAgent(
            model_name=model_name,
            temperature=temperature
        )
        # Note: SourceTracker is created per-document in _provenance_node
        self.checklist_agent = ChecklistAgent(
            model_name=model_name,
            temperature=temperature
        )
        
        # Build the orchestration graph
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=MemorySaver())
        
        logger.info(f"DocumentOrchestrator initialized with model={model_name}")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph orchestration graph.
        
        Returns:
            Compiled StateGraph for orchestration
        """
        # Create graph with OrchestrationState
        workflow = StateGraph(OrchestrationState)
        
        # Add nodes for each agent
        workflow.add_node(self.AGENT_INGESTION, self._ingestion_node)
        workflow.add_node(self.AGENT_CLAUSE_EXTRACTION, self._clause_extraction_node)
        workflow.add_node(self.AGENT_RISK_SCORING, self._risk_scoring_node)
        workflow.add_node(self.AGENT_SUMMARY, self._summary_node)
        workflow.add_node(self.AGENT_PROVENANCE, self._provenance_node)
        workflow.add_node(self.AGENT_CHECKLIST, self._checklist_node)
        workflow.add_node(self.AGENT_SUPERVISOR, self._supervisor_node)
        
        # Set entry point
        workflow.set_entry_point(self.AGENT_SUPERVISOR)
        
        # Add edges from supervisor to agents
        workflow.add_conditional_edges(
            self.AGENT_SUPERVISOR,
            self._route_next_agent,
            {
                self.AGENT_INGESTION: self.AGENT_INGESTION,
                self.AGENT_CLAUSE_EXTRACTION: self.AGENT_CLAUSE_EXTRACTION,
                self.AGENT_RISK_SCORING: self.AGENT_RISK_SCORING,
                self.AGENT_SUMMARY: self.AGENT_SUMMARY,
                self.AGENT_PROVENANCE: self.AGENT_PROVENANCE,
                self.AGENT_CHECKLIST: self.AGENT_CHECKLIST,
                END: END,
            }
        )
        
        # Add edges from agents back to supervisor
        workflow.add_edge(self.AGENT_INGESTION, self.AGENT_SUPERVISOR)
        workflow.add_edge(self.AGENT_CLAUSE_EXTRACTION, self.AGENT_SUPERVISOR)
        workflow.add_edge(self.AGENT_RISK_SCORING, self.AGENT_SUPERVISOR)
        workflow.add_edge(self.AGENT_SUMMARY, self.AGENT_SUPERVISOR)
        workflow.add_edge(self.AGENT_PROVENANCE, self.AGENT_SUPERVISOR)
        workflow.add_edge(self.AGENT_CHECKLIST, self.AGENT_SUPERVISOR)
        
        return workflow
    
    def _supervisor_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Supervisor node that decides which agent to invoke next.
        
        Args:
            state: Current orchestration state
            
        Returns:
            Updated state with next_agent set
        """
        try:
            current_step = state.get("current_step", self.STEP_START)
            completed_steps = state.get("completed_steps", [])
            errors = state.get("errors", [])
            
            logger.info(f"Supervisor analyzing state: current_step={current_step}, completed={completed_steps}")
            
            # Check for max iterations
            if len(completed_steps) >= self.MAX_ITERATIONS:
                logger.warning(f"Max iterations ({self.MAX_ITERATIONS}) reached")
                state["errors"] = errors + ["Max iterations reached"]
                state["next_agent"] = END
                return state
            
            # Determine next agent based on workflow
            next_agent = self._determine_next_agent(current_step, completed_steps, errors)
            
            logger.info(f"Supervisor decision: next_agent={next_agent}")
            state["next_agent"] = next_agent
            
            return state
            
        except Exception as e:
            logger.error(f"Error in supervisor node: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Supervisor error: {str(e)}"]
            state["next_agent"] = END
            return state
    
    def _determine_next_agent(
        self,
        current_step: str,
        completed_steps: List[str],
        errors: List[str]
    ) -> str:
        """
        Determine the next agent to invoke based on workflow state.
        
        Args:
            current_step: Current workflow step
            completed_steps: List of completed steps
            errors: List of errors encountered
            
        Returns:
            Name of next agent to invoke or END
        """
        # If there are critical errors, end workflow
        if len(errors) > 3:
            logger.warning("Too many errors, ending workflow")
            return END
        
        # Standard workflow progression
        if self.STEP_INGESTION not in completed_steps:
            return self.AGENT_INGESTION
        elif self.STEP_CLAUSE_EXTRACTION not in completed_steps:
            return self.AGENT_CLAUSE_EXTRACTION
        elif self.STEP_RISK_SCORING not in completed_steps:
            return self.AGENT_RISK_SCORING
        elif self.STEP_SUMMARY not in completed_steps:
            return self.AGENT_SUMMARY
        elif self.STEP_PROVENANCE not in completed_steps:
            return self.AGENT_PROVENANCE
        elif self.STEP_CHECKLIST not in completed_steps:
            return self.AGENT_CHECKLIST
        else:
            # All steps completed
            return END
    
    def _route_next_agent(self, state: OrchestrationState) -> str:
        """
        Route to the next agent based on supervisor decision.
        
        Args:
            state: Current orchestration state
            
        Returns:
            Name of next agent or END
        """
        next_agent = state.get("next_agent", END)
        logger.info(f"Routing to: {next_agent}")
        return next_agent
    
    def _ingestion_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Ingestion agent node - processes document and creates chunks.
        
        Args:
            state: Current orchestration state
            
        Returns:
            Updated state with document chunks
        """
        try:
            logger.info("Executing ingestion agent")
            
            document_path = state.get("document_path")
            document_id = state.get("document_id")
            
            if not document_path:
                raise ValueError("document_path is required for ingestion")
            
            # Run ingestion pipeline
            result = self.ingestion_pipeline.ingest_document(
                file_path=document_path,
                session_id=document_id  # Use document_id as session_id
            )
            
            # Result is a dict with chunk_count and chunk_ids
            chunk_count = result.get("chunk_count", 0)
            
            # Update state (we don't store actual chunks, just the count)
            state["document_chunks"] = []  # Chunks are in vector store
            state["current_step"] = self.STEP_CLAUSE_EXTRACTION
            state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_INGESTION]
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["ingestion"] = {
                "total_chunks": chunk_count,
                "processing_time": 0  # IngestionPipeline doesn't track time
            }
            state["metadata"] = metadata
            
            logger.info(f"Ingestion complete: {chunk_count} chunks created")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in ingestion node: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Ingestion error: {str(e)}"]
            state["current_step"] = self.STEP_CLAUSE_EXTRACTION  # Try to continue
            return state
    
    def _clause_extraction_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Clause extraction agent node - extracts clauses and detects red flags.
        
        Args:
            state: Current orchestration state
            
        Returns:
            Updated state with extracted clauses
        """
        try:
            logger.info("Executing clause extraction agent")
            
            document_id = state.get("document_id")
            
            # Run clause extraction agent
            result = self.clause_extraction_agent.extract_clauses(
                document_id=document_id
            )
            
            # Update state
            state["extracted_clauses"] = result.clauses
            state["current_step"] = self.STEP_RISK_SCORING
            state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_CLAUSE_EXTRACTION]
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["clause_extraction"] = {
                "total_clauses": len(result.clauses),
                "total_red_flags": len(result.red_flags),
                "processing_time": result.metadata.get("processing_time_seconds", 0)
            }
            state["metadata"] = metadata
            
            logger.info(f"Clause extraction complete: {len(result.clauses)} clauses, {len(result.red_flags)} red flags")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in clause extraction node: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Clause extraction error: {str(e)}"]
            state["current_step"] = self.STEP_RISK_SCORING  # Try to continue
            return state
    
    def _risk_scoring_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Risk scoring agent node - assigns risk scores to clauses.
        
        Args:
            state: Current orchestration state
            
        Returns:
            Updated state with risk scores
        """
        try:
            logger.info("Executing risk scoring agent")
            
            extracted_clauses = state.get("extracted_clauses", [])
            
            if not extracted_clauses:
                logger.warning("No clauses to score, skipping risk scoring")
                state["current_step"] = self.STEP_SUMMARY
                state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_RISK_SCORING]
                return state
            
            # Create ClauseExtractionResult for risk scoring agent
            from app.models.agent import ClauseExtractionResult
            clause_result = ClauseExtractionResult(
                clauses=extracted_clauses,
                red_flags=[],
                metadata={},
                timestamp=datetime.now(UTC),
                document_id=state.get("document_id")
            )
            
            # Run risk scoring agent
            result = self.risk_scoring_agent.score_risks(clause_result)
            
            # Update state
            state["scored_clauses"] = result.scored_clauses
            state["risk_scores"] = [sc.risk_score for sc in result.scored_clauses]
            state["current_step"] = self.STEP_SUMMARY
            state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_RISK_SCORING]
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["risk_scoring"] = {
                "total_scored_clauses": len(result.scored_clauses),
                "overall_risk_score": result.overall_risk_score,
                "overall_risk_category": result.overall_risk_category,
                "processing_time": result.metadata.get("processing_time_seconds", 0)
            }
            state["metadata"] = metadata
            
            logger.info(f"Risk scoring complete: {len(result.scored_clauses)} clauses scored, overall risk={result.overall_risk_score}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in risk scoring node: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Risk scoring error: {str(e)}"]
            state["current_step"] = self.STEP_SUMMARY  # Try to continue
            return state
    
    def _summary_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Summary agent node - generates diligence memo.
        
        Args:
            state: Current orchestration state
            
        Returns:
            Updated state with summary
        """
        try:
            logger.info("Executing summary agent")
            
            scored_clauses = state.get("scored_clauses", [])
            
            if not scored_clauses:
                logger.warning("No scored clauses available, skipping summary")
                state["current_step"] = self.STEP_PROVENANCE
                state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_SUMMARY]
                return state
            
            # Create RiskScoringResult for summary agent
            from app.models.agent import RiskScoringResult
            risk_result = RiskScoringResult(
                scored_clauses=scored_clauses,
                overall_risk_score=state.get("metadata", {}).get("risk_scoring", {}).get("overall_risk_score", 0),
                overall_risk_category=state.get("metadata", {}).get("risk_scoring", {}).get("overall_risk_category", "Low"),
                metadata={},
                timestamp=datetime.now(UTC),
                document_id=state.get("document_id")
            )
            
            # Run summary agent
            result = self.summary_agent.generate_summary(risk_result)
            
            # Update state
            state["summary"] = result
            state["current_step"] = self.STEP_PROVENANCE
            state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_SUMMARY]
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["summary"] = {
                "total_findings": len(result.key_findings),
                "total_recommendations": len(result.recommendations),
                "processing_time": result.metadata.get("processing_time_seconds", 0)
            }
            state["metadata"] = metadata
            
            logger.info(f"Summary complete: {len(result.key_findings)} findings, {len(result.recommendations)} recommendations")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in summary node: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Summary error: {str(e)}"]
            state["current_step"] = self.STEP_PROVENANCE  # Try to continue
            return state
    
    def _provenance_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Provenance agent node - adds source tracking to outputs.
        
        Args:
            state: Current orchestration state
            
        Returns:
            Updated state with provenance data
        """
        try:
            logger.info("Executing provenance agent")
            
            extracted_clauses = state.get("extracted_clauses", [])
            document_id = state.get("document_id")
            
            if not extracted_clauses:
                logger.warning("No clauses available for provenance tracking")
                state["current_step"] = self.STEP_CHECKLIST
                state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_PROVENANCE]
                return state
            
            # Create source tracker for this document
            source_tracker = SourceTracker(document_id=document_id)
            
            # Track sources for each clause
            provenance_data = {}
            for clause in extracted_clauses:
                clause_id = f"{clause.clause_type}_{hash(clause.clause_text) % 10000}"
                # Create source references from clause metadata
                sources = []
                for chunk_id in clause.source_chunk_ids[:3]:  # Top 3 sources
                    source_ref = source_tracker.create_source_reference(
                        chunk_id=chunk_id,
                        text_snippet=clause.clause_text[:200],
                        confidence=clause.confidence
                    )
                    sources.append(source_ref)
                
                provenance_data[clause_id] = {
                    "clause_type": clause.clause_type,
                    "sources": [s.model_dump() for s in sources]
                }
            
            # Update state
            state["provenance_data"] = provenance_data
            state["current_step"] = self.STEP_CHECKLIST
            state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_PROVENANCE]
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["provenance"] = {
                "total_items_tracked": len(provenance_data)
            }
            state["metadata"] = metadata
            
            logger.info(f"Provenance tracking complete: {len(provenance_data)} items tracked")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in provenance node: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Provenance error: {str(e)}"]
            state["current_step"] = self.STEP_CHECKLIST  # Try to continue
            return state
    
    def _checklist_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Checklist agent node - generates follow-up questions.
        
        Args:
            state: Current orchestration state
            
        Returns:
            Updated state with checklist
        """
        try:
            logger.info("Executing checklist agent")
            
            summary = state.get("summary")
            
            if not summary:
                logger.warning("No summary available for checklist generation")
                state["current_step"] = self.STEP_COMPLETE
                state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_CHECKLIST]
                return state
            
            # Run checklist agent
            result = self.checklist_agent.generate_checklist(summary)
            
            # Convert to ChecklistItem format
            checklist_items = []
            for item in result.checklist_items:
                checklist_items.append({
                    "question": item.item,  # ChecklistItem uses 'item' not 'question'
                    "category": item.category,
                    "priority": item.priority,
                    "rationale": item.rationale if hasattr(item, 'rationale') else ""
                })
            
            # Update state
            state["checklist"] = checklist_items
            state["current_step"] = self.STEP_COMPLETE
            state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_CHECKLIST]
            state["completed_at"] = datetime.now(UTC).isoformat()
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["checklist"] = {
                "total_questions": len(checklist_items),
                "processing_time": result.metadata.get("processing_time_seconds", 0)
            }
            state["metadata"] = metadata
            
            logger.info(f"Checklist complete: {len(checklist_items)} questions generated")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in checklist node: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [f"Checklist error: {str(e)}"]
            state["current_step"] = self.STEP_COMPLETE
            state["completed_steps"] = state.get("completed_steps", []) + [self.STEP_CHECKLIST]
            return state
    
    def _initialize_state(self, document_path: str, document_id: Optional[str] = None) -> OrchestrationState:
        """
        Initialize the orchestration state.
        
        Args:
            document_path: Path to the document to process
            document_id: Optional document ID (generated if not provided)
            
        Returns:
            Initialized OrchestrationState
        """
        if not document_id:
            import hashlib
            document_id = hashlib.md5(document_path.encode()).hexdigest()[:12]
        
        return {
            # Document info
            "document_id": document_id,
            "document_path": document_path,
            "document_chunks": [],
            
            # Workflow tracking
            "current_step": self.STEP_START,
            "completed_steps": [],
            "next_agent": None,
            
            # Agent outputs
            "extracted_clauses": [],
            "risk_scores": [],
            "scored_clauses": [],
            "summary": None,
            "provenance_data": {},
            "checklist": [],
            
            # Error handling
            "errors": [],
            "retry_count": 0,
            
            # Metadata
            "metadata": {},
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            
            # Messages
            "messages": [],
        }
    
    def run_orchestration(
        self,
        document_path: str,
        document_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete orchestration workflow for a document.
        
        Args:
            document_path: Path to the document to process
            document_id: Optional document ID (generated if not provided)
            config: Optional configuration for the workflow
            
        Returns:
            Dictionary containing all agent results and metadata
        """
        try:
            logger.info(f"Starting orchestration for document: {document_path}")
            
            # Initialize state
            initial_state = self._initialize_state(document_path, document_id)
            
            # Run the graph
            config = config or {"configurable": {"thread_id": document_id or "default"}}
            final_state = self.app.invoke(initial_state, config=config)
            
            # Extract results
            results = {
                "document_id": final_state.get("document_id"),
                "status": "completed" if not final_state.get("errors") else "completed_with_errors",
                "extracted_clauses": final_state.get("extracted_clauses", []),
                "scored_clauses": final_state.get("scored_clauses", []),
                "summary": final_state.get("summary"),
                "provenance_data": final_state.get("provenance_data", {}),
                "checklist": final_state.get("checklist", []),
                "errors": final_state.get("errors", []),
                "metadata": final_state.get("metadata", {}),
                "started_at": final_state.get("started_at"),
                "completed_at": final_state.get("completed_at"),
                "completed_steps": final_state.get("completed_steps", []),
            }
            
            logger.info(f"Orchestration complete: {len(results['completed_steps'])} steps completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in orchestration: {e}", exc_info=True)
            return {
                "document_id": document_id,
                "status": "failed",
                "errors": [str(e)],
                "metadata": {},
            }
    
    def get_graph_visualization(self) -> Any:
        """
        Get a visualization of the orchestration graph.
        
        Returns:
            Graph visualization (IPython Image if available, None otherwise)
        """
        try:
            from IPython.display import Image
            return Image(self.app.get_graph().draw_mermaid_png())
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}")
            return None
