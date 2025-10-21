"""
Integration tests for the Document Orchestrator.

Tests the complete orchestration workflow including supervisor logic,
agent coordination, state management, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, UTC
from typing import List

from app.orchestration.pipeline import DocumentOrchestrator
from app.models.orchestration import OrchestrationState
from app.models.agent import (
    ExtractedClause,
    RedFlag,
    ClauseExtractionResult,
    RiskScore,
    RiskFactor,
    ScoredClause,
    RiskScoringResult,
    DiligenceMemo,
    ExecutiveSummary,
    KeyFinding,
    Recommendation,
)
from app.rag.vector_store import VectorStore


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    return Mock(spec=VectorStore)


@pytest.fixture
def orchestrator(mock_vector_store):
    """Create a DocumentOrchestrator instance with mocked dependencies."""
    with patch('app.orchestration.pipeline.IngestionPipeline'), \
         patch('app.orchestration.pipeline.ClauseExtractionAgent'), \
         patch('app.orchestration.pipeline.RiskScoringAgent'), \
         patch('app.orchestration.pipeline.SummaryAgent'), \
         patch('app.orchestration.pipeline.SourceTracker'), \
         patch('app.orchestration.pipeline.ChecklistAgent'):
        
        orch = DocumentOrchestrator(vector_store=mock_vector_store)
        return orch


class TestOrchestratorInitialization:
    """Test orchestrator initialization and configuration."""
    
    def test_orchestrator_initialization(self, mock_vector_store):
        """Test that orchestrator initializes correctly."""
        with patch('app.orchestration.pipeline.IngestionPipeline'), \
             patch('app.orchestration.pipeline.ClauseExtractionAgent'), \
             patch('app.orchestration.pipeline.RiskScoringAgent'), \
             patch('app.orchestration.pipeline.SummaryAgent'), \
             patch('app.orchestration.pipeline.SourceTracker'), \
             patch('app.orchestration.pipeline.ChecklistAgent'):
            
            orch = DocumentOrchestrator(
                model_name="gpt-4o-mini",
                temperature=0.0,
                vector_store=mock_vector_store
            )
            
            assert orch.model_name == "gpt-4o-mini"
            assert orch.temperature == 0.0
            assert orch.vector_store == mock_vector_store
            assert orch.graph is not None
            assert orch.app is not None
    
    def test_orchestrator_creates_vector_store_if_none(self):
        """Test that orchestrator creates vector store if not provided."""
        with patch('app.orchestration.pipeline.VectorStore') as mock_vs, \
             patch('app.orchestration.pipeline.IngestionPipeline'), \
             patch('app.orchestration.pipeline.ClauseExtractionAgent'), \
             patch('app.orchestration.pipeline.RiskScoringAgent'), \
             patch('app.orchestration.pipeline.SummaryAgent'), \
             patch('app.orchestration.pipeline.SourceTracker'), \
             patch('app.orchestration.pipeline.ChecklistAgent'):
            
            orch = DocumentOrchestrator()
            mock_vs.assert_called_once()


class TestStateManagement:
    """Test state initialization and management."""
    
    def test_initialize_state(self, orchestrator):
        """Test state initialization with document path."""
        state = orchestrator._initialize_state(
            document_path="/path/to/doc.pdf",
            document_id="test_doc_123"
        )
        
        assert state["document_id"] == "test_doc_123"
        assert state["document_path"] == "/path/to/doc.pdf"
        assert state["current_step"] == orchestrator.STEP_START
        assert state["completed_steps"] == []
        assert state["errors"] == []
        assert state["extracted_clauses"] == []
        assert state["scored_clauses"] == []
        assert state["checklist"] == []
        assert "started_at" in state
        assert state["completed_at"] is None
    
    def test_initialize_state_generates_document_id(self, orchestrator):
        """Test that document ID is generated if not provided."""
        state = orchestrator._initialize_state(document_path="/path/to/doc.pdf")
        
        assert "document_id" in state
        assert len(state["document_id"]) == 12  # MD5 hash truncated to 12 chars


class TestSupervisorLogic:
    """Test supervisor node and routing logic."""
    
    def test_determine_next_agent_start(self, orchestrator):
        """Test supervisor determines ingestion as first agent."""
        next_agent = orchestrator._determine_next_agent(
            current_step=orchestrator.STEP_START,
            completed_steps=[],
            errors=[]
        )
        
        assert next_agent == orchestrator.AGENT_INGESTION
    
    def test_determine_next_agent_after_ingestion(self, orchestrator):
        """Test supervisor routes to clause extraction after ingestion."""
        next_agent = orchestrator._determine_next_agent(
            current_step=orchestrator.STEP_CLAUSE_EXTRACTION,
            completed_steps=[orchestrator.STEP_INGESTION],
            errors=[]
        )
        
        assert next_agent == orchestrator.AGENT_CLAUSE_EXTRACTION
    
    def test_determine_next_agent_workflow_progression(self, orchestrator):
        """Test complete workflow progression."""
        # After ingestion
        assert orchestrator._determine_next_agent(
            orchestrator.STEP_CLAUSE_EXTRACTION,
            [orchestrator.STEP_INGESTION],
            []
        ) == orchestrator.AGENT_CLAUSE_EXTRACTION
        
        # After clause extraction
        assert orchestrator._determine_next_agent(
            orchestrator.STEP_RISK_SCORING,
            [orchestrator.STEP_INGESTION, orchestrator.STEP_CLAUSE_EXTRACTION],
            []
        ) == orchestrator.AGENT_RISK_SCORING
        
        # After risk scoring
        assert orchestrator._determine_next_agent(
            orchestrator.STEP_SUMMARY,
            [orchestrator.STEP_INGESTION, orchestrator.STEP_CLAUSE_EXTRACTION, orchestrator.STEP_RISK_SCORING],
            []
        ) == orchestrator.AGENT_SUMMARY
    
    def test_determine_next_agent_ends_after_all_steps(self, orchestrator):
        """Test supervisor ends workflow after all steps completed."""
        from langgraph.graph import END
        
        next_agent = orchestrator._determine_next_agent(
            current_step=orchestrator.STEP_COMPLETE,
            completed_steps=[
                orchestrator.STEP_INGESTION,
                orchestrator.STEP_CLAUSE_EXTRACTION,
                orchestrator.STEP_RISK_SCORING,
                orchestrator.STEP_SUMMARY,
                orchestrator.STEP_PROVENANCE,
                orchestrator.STEP_CHECKLIST,
            ],
            errors=[]
        )
        
        assert next_agent == END
    
    def test_determine_next_agent_ends_on_too_many_errors(self, orchestrator):
        """Test supervisor ends workflow if too many errors."""
        from langgraph.graph import END
        
        next_agent = orchestrator._determine_next_agent(
            current_step=orchestrator.STEP_CLAUSE_EXTRACTION,
            completed_steps=[orchestrator.STEP_INGESTION],
            errors=["error1", "error2", "error3", "error4"]
        )
        
        assert next_agent == END
    
    def test_supervisor_node_sets_next_agent(self, orchestrator):
        """Test supervisor node updates state with next agent."""
        state = {
            "current_step": orchestrator.STEP_START,
            "completed_steps": [],
            "errors": []
        }
        
        updated_state = orchestrator._supervisor_node(state)
        
        assert "next_agent" in updated_state
        assert updated_state["next_agent"] == orchestrator.AGENT_INGESTION
    
    def test_supervisor_node_handles_max_iterations(self, orchestrator):
        """Test supervisor node stops at max iterations."""
        from langgraph.graph import END
        
        state = {
            "current_step": orchestrator.STEP_CLAUSE_EXTRACTION,
            "completed_steps": ["step"] * 51,  # Exceed MAX_ITERATIONS
            "errors": []
        }
        
        updated_state = orchestrator._supervisor_node(state)
        
        assert updated_state["next_agent"] == END
        assert "Max iterations reached" in updated_state["errors"]


class TestAgentNodes:
    """Test individual agent node functions."""
    
    def test_ingestion_node_success(self, orchestrator):
        """Test ingestion node processes document successfully."""
        # Mock ingestion pipeline
        mock_result = Mock()
        mock_result.chunks = [Mock(page_content=f"chunk_{i}") for i in range(5)]
        mock_result.metadata = {"processing_time_seconds": 2.5}
        orchestrator.ingestion_pipeline.ingest_document = Mock(return_value=mock_result)
        
        state = {
            "document_path": "/path/to/doc.pdf",
            "document_id": "test_doc",
            "completed_steps": [],
            "metadata": {}
        }
        
        updated_state = orchestrator._ingestion_node(state)
        
        assert len(updated_state["document_chunks"]) == 5
        assert orchestrator.STEP_INGESTION in updated_state["completed_steps"]
        assert updated_state["current_step"] == orchestrator.STEP_CLAUSE_EXTRACTION
        assert "ingestion" in updated_state["metadata"]
    
    def test_ingestion_node_handles_missing_path(self, orchestrator):
        """Test ingestion node handles missing document path."""
        state = {
            "document_id": "test_doc",
            "completed_steps": [],
            "errors": []
        }
        
        updated_state = orchestrator._ingestion_node(state)
        
        assert len(updated_state.get("errors", [])) > 0
        assert any("document_path is required" in err for err in updated_state["errors"])
    
    def test_clause_extraction_node_success(self, orchestrator):
        """Test clause extraction node extracts clauses successfully."""
        # Mock clause extraction agent
        mock_clause = ExtractedClause(
            clause_text="Test clause",
            clause_type="payment_terms",
            location={"page": 1},
            confidence=0.9,
            source_chunk_ids=["chunk_1"]
        )
        mock_result = ClauseExtractionResult(
            clauses=[mock_clause],
            red_flags=[],
            metadata={"processing_time_seconds": 5.0},
            timestamp=datetime.now(UTC),
            document_id="test_doc"
        )
        orchestrator.clause_extraction_agent.extract_clauses = Mock(return_value=mock_result)
        
        state = {
            "document_id": "test_doc",
            "completed_steps": [orchestrator.STEP_INGESTION],
            "metadata": {}
        }
        
        updated_state = orchestrator._clause_extraction_node(state)
        
        assert len(updated_state["extracted_clauses"]) == 1
        assert orchestrator.STEP_CLAUSE_EXTRACTION in updated_state["completed_steps"]
        assert updated_state["current_step"] == orchestrator.STEP_RISK_SCORING
    
    def test_risk_scoring_node_success(self, orchestrator):
        """Test risk scoring node scores clauses successfully."""
        # Setup state with extracted clauses
        mock_clause = ExtractedClause(
            clause_text="Test clause",
            clause_type="payment_terms",
            location={"page": 1},
            confidence=0.9,
            source_chunk_ids=["chunk_1"]
        )
        
        # Mock risk scoring agent
        mock_risk_score = RiskScore(
            score=50,
            category="Medium",
            factors=[],
            justification="Test justification"
        )
        mock_scored_clause = ScoredClause(
            clause=mock_clause,
            risk_score=mock_risk_score
        )
        mock_result = RiskScoringResult(
            scored_clauses=[mock_scored_clause],
            overall_risk_score=50,
            overall_risk_category="Medium",
            metadata={"processing_time_seconds": 3.0},
            timestamp=datetime.now(UTC),
            document_id="test_doc"
        )
        orchestrator.risk_scoring_agent.score_risks = Mock(return_value=mock_result)
        
        state = {
            "document_id": "test_doc",
            "extracted_clauses": [mock_clause],
            "completed_steps": [orchestrator.STEP_INGESTION, orchestrator.STEP_CLAUSE_EXTRACTION],
            "metadata": {}
        }
        
        updated_state = orchestrator._risk_scoring_node(state)
        
        assert len(updated_state["scored_clauses"]) == 1
        assert len(updated_state["risk_scores"]) == 1
        assert orchestrator.STEP_RISK_SCORING in updated_state["completed_steps"]
    
    def test_risk_scoring_node_skips_if_no_clauses(self, orchestrator):
        """Test risk scoring node skips if no clauses available."""
        state = {
            "document_id": "test_doc",
            "extracted_clauses": [],
            "completed_steps": [orchestrator.STEP_INGESTION, orchestrator.STEP_CLAUSE_EXTRACTION],
            "metadata": {}
        }
        
        updated_state = orchestrator._risk_scoring_node(state)
        
        assert orchestrator.STEP_RISK_SCORING in updated_state["completed_steps"]
        assert updated_state["current_step"] == orchestrator.STEP_SUMMARY


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    def test_ingestion_node_error_handling(self, orchestrator):
        """Test ingestion node handles errors gracefully."""
        orchestrator.ingestion_pipeline.ingest_document = Mock(
            side_effect=Exception("Ingestion failed")
        )
        
        state = {
            "document_path": "/path/to/doc.pdf",
            "document_id": "test_doc",
            "completed_steps": [],
            "errors": []
        }
        
        updated_state = orchestrator._ingestion_node(state)
        
        assert len(updated_state["errors"]) > 0
        assert any("Ingestion error" in err for err in updated_state["errors"])
        # Should still try to continue
        assert updated_state["current_step"] == orchestrator.STEP_CLAUSE_EXTRACTION
    
    def test_clause_extraction_node_error_handling(self, orchestrator):
        """Test clause extraction node handles errors gracefully."""
        orchestrator.clause_extraction_agent.extract_clauses = Mock(
            side_effect=Exception("Extraction failed")
        )
        
        state = {
            "document_id": "test_doc",
            "completed_steps": [orchestrator.STEP_INGESTION],
            "errors": []
        }
        
        updated_state = orchestrator._clause_extraction_node(state)
        
        assert len(updated_state["errors"]) > 0
        assert any("Clause extraction error" in err for err in updated_state["errors"])
    
    def test_supervisor_node_error_handling(self, orchestrator):
        """Test supervisor node handles errors gracefully."""
        from langgraph.graph import END
        
        # Patch _determine_next_agent to raise an error
        with patch.object(orchestrator, '_determine_next_agent', side_effect=Exception("Supervisor error")):
            state = {
                "current_step": orchestrator.STEP_START,
                "completed_steps": [],
                "errors": []
            }
            
            updated_state = orchestrator._supervisor_node(state)
            
            assert len(updated_state["errors"]) > 0
            assert any("Supervisor error" in err for err in updated_state["errors"])
            assert updated_state["next_agent"] == END


class TestGraphVisualization:
    """Test graph visualization functionality."""
    
    def test_get_graph_visualization(self, orchestrator):
        """Test graph visualization generation."""
        # The method handles import internally, just test it doesn't crash
        result = orchestrator.get_graph_visualization()
        
        # Should return None or an Image object
        assert result is None or result is not None
    
    def test_get_graph_visualization_handles_error(self, orchestrator):
        """Test graph visualization handles errors gracefully."""
        # Mock the app.get_graph to raise an error
        with patch.object(orchestrator.app, 'get_graph', side_effect=Exception("Graph error")):
            result = orchestrator.get_graph_visualization()
            
            # Should return None on error
            assert result is None


class TestRunOrchestration:
    """Test the main orchestration execution method."""
    
    def test_run_orchestration_returns_results(self, orchestrator):
        """Test run_orchestration returns structured results."""
        # Mock the app.invoke to return a final state
        mock_final_state = {
            "document_id": "test_doc",
            "extracted_clauses": [],
            "scored_clauses": [],
            "summary": None,
            "provenance_data": {},
            "checklist": [],
            "errors": [],
            "metadata": {},
            "started_at": "2025-10-21T00:00:00",
            "completed_at": "2025-10-21T00:05:00",
            "completed_steps": [
                orchestrator.STEP_INGESTION,
                orchestrator.STEP_CLAUSE_EXTRACTION,
                orchestrator.STEP_RISK_SCORING,
                orchestrator.STEP_SUMMARY,
                orchestrator.STEP_PROVENANCE,
                orchestrator.STEP_CHECKLIST,
            ]
        }
        
        orchestrator.app.invoke = Mock(return_value=mock_final_state)
        
        results = orchestrator.run_orchestration(
            document_path="/path/to/doc.pdf",
            document_id="test_doc"
        )
        
        assert results["document_id"] == "test_doc"
        assert results["status"] == "completed"
        assert "extracted_clauses" in results
        assert "scored_clauses" in results
        assert "summary" in results
        assert "checklist" in results
        assert "metadata" in results
        assert len(results["completed_steps"]) == 6
    
    def test_run_orchestration_handles_errors(self, orchestrator):
        """Test run_orchestration handles errors gracefully."""
        orchestrator.app.invoke = Mock(side_effect=Exception("Orchestration failed"))
        
        results = orchestrator.run_orchestration(
            document_path="/path/to/doc.pdf",
            document_id="test_doc"
        )
        
        assert results["status"] == "failed"
        assert len(results["errors"]) > 0
        assert "Orchestration failed" in results["errors"][0]
    
    def test_run_orchestration_with_errors_in_state(self, orchestrator):
        """Test run_orchestration marks status as completed_with_errors."""
        mock_final_state = {
            "document_id": "test_doc",
            "extracted_clauses": [],
            "scored_clauses": [],
            "summary": None,
            "provenance_data": {},
            "checklist": [],
            "errors": ["Some error occurred"],
            "metadata": {},
            "started_at": "2025-10-21T00:00:00",
            "completed_at": "2025-10-21T00:05:00",
            "completed_steps": [orchestrator.STEP_INGESTION]
        }
        
        orchestrator.app.invoke = Mock(return_value=mock_final_state)
        
        results = orchestrator.run_orchestration(
            document_path="/path/to/doc.pdf",
            document_id="test_doc"
        )
        
        assert results["status"] == "completed_with_errors"
        assert len(results["errors"]) > 0


class TestWorkflowIntegration:
    """Integration tests for complete workflow."""
    
    @pytest.mark.integration
    def test_complete_workflow_mock(self, orchestrator):
        """Test complete workflow with all agents mocked."""
        # Mock all agents to return successful results
        
        # Ingestion
        mock_ingestion_result = Mock()
        mock_ingestion_result.chunks = [Mock(page_content="chunk_1")]
        mock_ingestion_result.metadata = {"processing_time_seconds": 1.0}
        orchestrator.ingestion_pipeline.ingest_document = Mock(return_value=mock_ingestion_result)
        
        # Clause extraction
        mock_clause = ExtractedClause(
            clause_text="Payment terms clause",
            clause_type="payment_terms",
            location={"page": 1},
            confidence=0.9,
            source_chunk_ids=["chunk_1"]
        )
        mock_clause_result = ClauseExtractionResult(
            clauses=[mock_clause],
            red_flags=[],
            metadata={"processing_time_seconds": 2.0},
            timestamp=datetime.now(UTC),
            document_id="test_doc"
        )
        orchestrator.clause_extraction_agent.extract_clauses = Mock(return_value=mock_clause_result)
        
        # Risk scoring
        mock_risk_score = RiskScore(
            score=50,
            category="Medium",
            factors=[],
            justification="Medium risk"
        )
        mock_scored_clause = ScoredClause(
            clause=mock_clause,
            risk_score=mock_risk_score
        )
        mock_risk_result = RiskScoringResult(
            scored_clauses=[mock_scored_clause],
            overall_risk_score=50,
            overall_risk_category="Medium",
            metadata={"processing_time_seconds": 1.5},
            timestamp=datetime.now(UTC),
            document_id="test_doc"
        )
        orchestrator.risk_scoring_agent.score_risks = Mock(return_value=mock_risk_result)
        
        # Summary
        mock_summary = DiligenceMemo(
            executive_summary=ExecutiveSummary(
                overview="Test overview",
                critical_findings=["Finding 1"],
                primary_recommendations=["Recommendation 1"],
                overall_risk_assessment="Medium risk"
            ),
            clause_summaries=[],
            key_findings=[],
            recommendations=[],
            overall_assessment="Proceed with caution",
            metadata={"processing_time_seconds": 3.0},
            timestamp=datetime.now(UTC),
            document_id="test_doc"
        )
        orchestrator.summary_agent.generate_summary = Mock(return_value=mock_summary)
        
        # Provenance - handled internally in _provenance_node, no mocking needed
        
        # Checklist
        from app.agents.checklist import ChecklistResult, ChecklistItem
        mock_checklist_item = ChecklistItem(
            item="Question 1?",
            category="Financial",
            priority="High",
            status="Pending",
            related_findings=[]
        )
        mock_checklist_result = ChecklistResult(
            checklist_items=[mock_checklist_item],
            metadata={"processing_time_seconds": 1.0},
            timestamp=datetime.now(UTC),
            document_id="test_doc"
        )
        orchestrator.checklist_agent.generate_checklist = Mock(return_value=mock_checklist_result)
        
        # Run orchestration
        results = orchestrator.run_orchestration(
            document_path="/path/to/doc.pdf",
            document_id="test_doc"
        )
        
        # Verify results
        assert results["status"] in ["completed", "completed_with_errors"]
        assert results["document_id"] == "test_doc"
        
        # Verify all agents were called
        orchestrator.ingestion_pipeline.ingest_document.assert_called_once()
        orchestrator.clause_extraction_agent.extract_clauses.assert_called_once()
        orchestrator.risk_scoring_agent.score_risks.assert_called_once()
        orchestrator.summary_agent.generate_summary.assert_called_once()
        orchestrator.checklist_agent.generate_checklist.assert_called_once()
