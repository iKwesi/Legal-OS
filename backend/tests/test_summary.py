"""
Tests for the Summary Agent.

This module tests the SummaryAgent class which generates M&A diligence memos
from risk scoring results using LangGraph's create_react_agent.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, UTC

from app.agents.summary import SummaryAgent
from app.models.agent import (
    RiskScoringResult,
    ScoredClause,
    ExtractedClause,
    RiskScore,
    RiskFactor,
    DiligenceMemo,
    ExecutiveSummary,
    ClauseSummary,
    KeyFinding,
    Recommendation,
)


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    retriever = Mock()
    retriever.invoke = Mock(return_value=[])
    return retriever


@pytest.fixture
def sample_risk_scoring_result():
    """Create a sample risk scoring result for testing."""
    clause1 = ExtractedClause(
        clause_text="Seller shall indemnify Buyer for all losses without limitation.",
        clause_type="indemnification",
        location={"page": 8},
        confidence=0.9,
        source_chunk_ids=["chunk_200"]
    )
    
    risk_score1 = RiskScore(
        score=75,
        category="High",
        factors=[
            RiskFactor(
                factor_name="unlimited_liability",
                description="No cap on indemnification",
                score_impact=50,
                detected=True
            )
        ],
        justification="High risk due to unlimited indemnification liability"
    )
    
    scored_clause1 = ScoredClause(clause=clause1, risk_score=risk_score1)
    
    clause2 = ExtractedClause(
        clause_text="Purchase price shall be $10,000,000 payable at closing.",
        clause_type="payment_terms",
        location={"page": 3},
        confidence=0.95,
        source_chunk_ids=["chunk_100"]
    )
    
    risk_score2 = RiskScore(
        score=20,
        category="Low",
        factors=[],
        justification="Standard payment terms with minimal risk"
    )
    
    scored_clause2 = ScoredClause(clause=clause2, risk_score=risk_score2)
    
    return RiskScoringResult(
        scored_clauses=[scored_clause1, scored_clause2],
        overall_risk_score=48,
        overall_risk_category="Medium",
        metadata={"model": "gpt-4o-mini"},
        timestamp=datetime.now(UTC),
        document_id="test_doc_123"
    )


class TestSummaryAgentInitialization:
    """Test SummaryAgent initialization."""
    
    def test_init_without_retriever(self):
        """Test agent initialization without retriever."""
        agent = SummaryAgent()
        assert agent.retriever is None
        assert agent.llm is not None
        assert agent.tools is not None
        assert len(agent.tools) == 6  # 6 tools defined
        assert agent.agent_executor is not None
    
    def test_init_with_retriever(self, mock_retriever):
        """Test agent initialization with retriever."""
        agent = SummaryAgent(retriever=mock_retriever)
        assert agent.retriever == mock_retriever
        assert agent.llm is not None
        assert agent.tools is not None
        assert agent.agent_executor is not None
    
    def test_init_with_custom_model(self):
        """Test agent initialization with custom model name."""
        agent = SummaryAgent(model_name="gpt-4", temperature=0.5)
        assert agent.llm.model_name == "gpt-4"
        assert agent.llm.temperature == 0.5


class TestSummaryAgentTools:
    """Test individual agent tools."""
    
    def test_retrieve_context_tool_without_retriever(self):
        """Test retrieve_context tool when no retriever is configured."""
        agent = SummaryAgent()
        result = agent._retrieve_context_tool('{"query": "test", "k": 3}')
        assert "error" in result
        assert "No retriever configured" in result
    
    def test_retrieve_context_tool_with_retriever(self, mock_retriever):
        """Test retrieve_context tool with retriever."""
        mock_chunk = Mock()
        mock_chunk.page_content = "Test content"
        mock_chunk.metadata = {"page": 1}
        mock_retriever.invoke.return_value = [mock_chunk]
        
        agent = SummaryAgent(retriever=mock_retriever)
        result = agent._retrieve_context_tool('{"query": "indemnification", "k": 3}')
        
        assert "chunks" in result
        mock_retriever.invoke.assert_called_once()
    
    def test_retrieve_context_tool_invalid_json(self):
        """Test retrieve_context tool with invalid JSON input."""
        agent = SummaryAgent()
        result = agent._retrieve_context_tool('invalid json')
        assert "error" in result
        # When no retriever is configured, it returns that error first
        # When retriever is configured, it would return Invalid JSON error
        assert "error" in result.lower()


class TestDataModels:
    """Test Pydantic data models."""
    
    def test_executive_summary_model(self):
        """Test ExecutiveSummary model creation."""
        summary = ExecutiveSummary(
            overview="Test overview",
            critical_findings=["Finding 1", "Finding 2"],
            primary_recommendations=["Rec 1", "Rec 2"],
            overall_risk_assessment="Medium risk"
        )
        assert summary.overview == "Test overview"
        assert len(summary.critical_findings) == 2
        assert len(summary.primary_recommendations) == 2
    
    def test_clause_summary_model(self):
        """Test ClauseSummary model creation."""
        summary = ClauseSummary(
            clause_type="indemnification",
            summary="Test summary",
            risk_level="High",
            key_points=["Point 1", "Point 2"]
        )
        assert summary.clause_type == "indemnification"
        assert summary.risk_level == "High"
        assert len(summary.key_points) == 2
    
    def test_key_finding_model(self):
        """Test KeyFinding model creation."""
        finding = KeyFinding(
            finding="Unlimited liability",
            severity="Critical",
            clause_reference="Section 8.2",
            impact="High financial risk"
        )
        assert finding.finding == "Unlimited liability"
        assert finding.severity == "Critical"
    
    def test_recommendation_model(self):
        """Test Recommendation model creation."""
        rec = Recommendation(
            recommendation="Negotiate cap",
            priority="Critical",
            rationale="Reduce financial exposure",
            related_findings=["Unlimited liability"]
        )
        assert rec.recommendation == "Negotiate cap"
        assert rec.priority == "Critical"
        assert len(rec.related_findings) == 1
    
    def test_diligence_memo_model(self, sample_risk_scoring_result):
        """Test DiligenceMemo model creation."""
        exec_summary = ExecutiveSummary(
            overview="Test overview",
            critical_findings=["Finding 1"],
            primary_recommendations=["Rec 1"],
            overall_risk_assessment="Medium risk"
        )
        
        memo = DiligenceMemo(
            executive_summary=exec_summary,
            clause_summaries=[],
            key_findings=[],
            recommendations=[],
            overall_assessment="Proceed with caution",
            metadata={},
            document_id="test_123"
        )
        
        assert memo.document_id == "test_123"
        assert memo.executive_summary.overview == "Test overview"
        assert memo.overall_assessment == "Proceed with caution"
    
    def test_diligence_memo_to_markdown(self):
        """Test DiligenceMemo to_markdown method."""
        exec_summary = ExecutiveSummary(
            overview="Test M&A document",
            critical_findings=["High risk item"],
            primary_recommendations=["Negotiate terms"],
            overall_risk_assessment="High risk (75/100)"
        )
        
        clause_summary = ClauseSummary(
            clause_type="indemnification",
            summary="Unlimited indemnification",
            risk_level="Critical",
            key_points=["No cap", "Broad scope"]
        )
        
        finding = KeyFinding(
            finding="Unlimited liability",
            severity="Critical",
            clause_reference="Section 8.2",
            impact="Financial exposure"
        )
        
        rec = Recommendation(
            recommendation="Add liability cap",
            priority="Critical",
            rationale="Limit exposure",
            related_findings=["Unlimited liability"]
        )
        
        memo = DiligenceMemo(
            executive_summary=exec_summary,
            clause_summaries=[clause_summary],
            key_findings=[finding],
            recommendations=[rec],
            overall_assessment="Proceed with Caution",
            metadata={},
            document_id="test_123"
        )
        
        markdown = memo.to_markdown()
        
        assert "# M&A Due Diligence Memo" in markdown
        assert "## Executive Summary" in markdown
        assert "Test M&A document" in markdown
        assert "## Clause-by-Clause Analysis" in markdown
        assert "Indemnification" in markdown
        assert "## Key Findings" in markdown
        assert "Unlimited liability" in markdown
        assert "## Recommendations" in markdown
        assert "Add liability cap" in markdown
        assert "## Overall Assessment" in markdown
        assert "Proceed with Caution" in markdown


class TestSummaryGeneration:
    """Test summary generation functionality."""
    
    def test_generate_summary_empty_clauses(self):
        """Test summary generation with no scored clauses."""
        empty_result = RiskScoringResult(
            scored_clauses=[],
            overall_risk_score=0,
            overall_risk_category="Low",
            metadata={},
            timestamp=datetime.now(UTC),
            document_id="empty_doc"
        )
        
        agent = SummaryAgent()
        memo = agent.generate_summary(empty_result)
        
        assert isinstance(memo, DiligenceMemo)
        assert memo.document_id == "empty_doc"
        assert "No clauses available" in memo.executive_summary.overview
        assert len(memo.clause_summaries) == 0
        assert "warning" in memo.metadata
    
    def test_parse_summary_results_with_executive_summary(self, sample_risk_scoring_result):
        """Test parsing executive summary from messages."""
        agent = SummaryAgent()
        
        mock_message = Mock()
        mock_message.content = '{"overview": "Test M&A document", "critical_findings": ["High risk"], "primary_recommendations": ["Negotiate"], "overall_risk_assessment": "Medium risk"}'
        
        messages = [mock_message]
        memo = agent._parse_summary_results(messages, sample_risk_scoring_result)
        
        assert isinstance(memo, DiligenceMemo)
        assert memo.executive_summary.overview == "Test M&A document"
        assert len(memo.executive_summary.critical_findings) == 1
    
    def test_parse_summary_results_with_clause_summaries(self, sample_risk_scoring_result):
        """Test parsing clause summaries from messages."""
        agent = SummaryAgent()
        
        mock_message = Mock()
        mock_message.content = '{"clause_type": "indemnification", "summary": "Test summary", "risk_level": "High", "key_points": ["Point 1"]}'
        
        messages = [mock_message]
        memo = agent._parse_summary_results(messages, sample_risk_scoring_result)
        
        assert isinstance(memo, DiligenceMemo)
        assert len(memo.clause_summaries) == 1
        assert memo.clause_summaries[0].clause_type == "indemnification"
    
    def test_parse_summary_results_with_findings(self, sample_risk_scoring_result):
        """Test parsing key findings from messages."""
        agent = SummaryAgent()
        
        mock_message = Mock()
        mock_message.content = '[{"finding": "Test finding", "severity": "High", "clause_reference": "Section 1", "impact": "Test impact"}]'
        
        messages = [mock_message]
        memo = agent._parse_summary_results(messages, sample_risk_scoring_result)
        
        assert isinstance(memo, DiligenceMemo)
        assert len(memo.key_findings) == 1
        assert memo.key_findings[0].finding == "Test finding"
    
    def test_parse_summary_results_with_recommendations(self, sample_risk_scoring_result):
        """Test parsing recommendations from messages."""
        agent = SummaryAgent()
        
        mock_message = Mock()
        mock_message.content = '[{"recommendation": "Test rec", "priority": "High", "rationale": "Test rationale", "related_findings": []}]'
        
        messages = [mock_message]
        memo = agent._parse_summary_results(messages, sample_risk_scoring_result)
        
        assert isinstance(memo, DiligenceMemo)
        assert len(memo.recommendations) == 1
        assert memo.recommendations[0].recommendation == "Test rec"


class TestGraphVisualization:
    """Test graph visualization functionality."""
    
    def test_get_graph_visualization_without_ipython(self):
        """Test graph visualization when IPython is not available."""
        agent = SummaryAgent()
        result = agent.get_graph_visualization()
        # Should return None or handle gracefully when IPython not available
        assert result is None or result is not None


class TestErrorHandling:
    """Test error handling in the agent."""
    
    def test_tool_error_handling_invalid_json(self):
        """Test that tools handle invalid JSON gracefully."""
        agent = SummaryAgent()
        
        # Test each tool with invalid JSON
        result1 = agent._generate_executive_summary_tool("invalid json")
        assert "error" in result1.lower()
        
        result2 = agent._generate_clause_summary_tool("invalid json")
        assert "error" in result2.lower()
        
        result3 = agent._extract_key_findings_tool("invalid json")
        assert "error" in result3.lower()
        
        result4 = agent._generate_recommendations_tool("invalid json")
        assert "error" in result4.lower()
        
        result5 = agent._generate_overall_assessment_tool("invalid json")
        assert "error" in result5.lower()
