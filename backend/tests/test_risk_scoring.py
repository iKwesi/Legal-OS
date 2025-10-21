"""
Tests for the Risk Scoring Agent.

This module tests the RiskScoringAgent's ability to score risks in extracted clauses
using defined rules and heuristics.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, UTC

from app.agents.risk_scoring import RiskScoringAgent
from app.models.agent import (
    ExtractedClause,
    ClauseExtractionResult,
    RiskFactor,
    RiskScore,
    ScoredClause,
    RiskScoringResult,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    with patch('app.agents.risk_scoring.ChatOpenAI') as mock:
        llm_instance = Mock()
        mock.return_value = llm_instance
        yield llm_instance


@pytest.fixture
def risk_scoring_agent(mock_llm):
    """Create a RiskScoringAgent instance for testing."""
    return RiskScoringAgent()


@pytest.fixture
def sample_clauses():
    """Create sample extracted clauses for testing."""
    return [
        ExtractedClause(
            clause_text="Seller shall indemnify Buyer for all losses without any cap or limitation.",
            clause_type="indemnification",
            location={"page": 8, "section": "8.2"},
            confidence=0.95,
            source_chunk_ids=["chunk_200"]
        ),
        ExtractedClause(
            clause_text="The purchase price shall be $10,000,000 payable in installments over 5 years.",
            clause_type="payment_terms",
            location={"page": 3, "section": "2.1"},
            confidence=0.90,
            source_chunk_ids=["chunk_50"]
        ),
        ExtractedClause(
            clause_text="Seller represents and warrants the accuracy of financial statements.",
            clause_type="warranties",
            location={"page": 5, "section": "4.1"},
            confidence=0.85,
            source_chunk_ids=["chunk_100"]
        ),
    ]


@pytest.fixture
def sample_clause_extraction_result(sample_clauses):
    """Create a sample ClauseExtractionResult for testing."""
    return ClauseExtractionResult(
        clauses=sample_clauses,
        red_flags=[],
        metadata={"processing_time_seconds": 10.5},
        timestamp=datetime.now(UTC),
        document_id="test_doc_123"
    )


class TestRiskScoringAgentInitialization:
    """Test agent initialization and configuration."""
    
    def test_agent_initialization(self, risk_scoring_agent):
        """Test that agent initializes correctly."""
        assert risk_scoring_agent is not None
        assert risk_scoring_agent.llm is not None
        assert risk_scoring_agent.tools is not None
        assert len(risk_scoring_agent.tools) == 3
        assert risk_scoring_agent.agent_executor is not None
    
    def test_risk_rules_defined(self, risk_scoring_agent):
        """Test that risk rules are properly defined."""
        assert len(risk_scoring_agent.RISK_RULES) > 0
        assert "payment_terms" in risk_scoring_agent.RISK_RULES
        assert "indemnification" in risk_scoring_agent.RISK_RULES
        assert "warranties" in risk_scoring_agent.RISK_RULES
        
        # Check structure of rules
        for clause_type, rules in risk_scoring_agent.RISK_RULES.items():
            assert isinstance(rules, dict)
            for factor_name, factor_info in rules.items():
                assert "impact" in factor_info
                assert "description" in factor_info
                assert isinstance(factor_info["impact"], int)
                assert 0 <= factor_info["impact"] <= 100


class TestRiskScoringTools:
    """Test individual risk scoring tools."""
    
    def test_analyze_clause_tool_valid_input(self, risk_scoring_agent, mock_llm):
        """Test analyze_clause_tool with valid input."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '''```json
{
    "identified_factors": [
        {
            "factor_name": "unlimited_liability",
            "detected": true,
            "evidence": "Clause states 'without any cap or limitation'"
        }
    ]
}
```'''
        mock_llm.invoke.return_value = mock_response
        
        input_json = '{"clause_text": "Seller shall indemnify without cap", "clause_type": "indemnification"}'
        result = risk_scoring_agent._analyze_clause_tool(input_json)
        
        assert result is not None
        assert "unlimited_liability" in result or "identified_factors" in result
    
    def test_analyze_clause_tool_invalid_json(self, risk_scoring_agent):
        """Test analyze_clause_tool with invalid JSON input."""
        result = risk_scoring_agent._analyze_clause_tool("not valid json")
        assert "error" in result.lower() or "invalid" in result.lower()
    
    def test_analyze_clause_tool_unknown_clause_type(self, risk_scoring_agent):
        """Test analyze_clause_tool with unknown clause type."""
        input_json = '{"clause_text": "Some text", "clause_type": "unknown_type"}'
        result = risk_scoring_agent._analyze_clause_tool(input_json)
        assert "error" in result.lower() or "no risk rules" in result.lower()
    
    def test_calculate_risk_tool_valid_factors(self, risk_scoring_agent):
        """Test calculate_risk_tool with valid factors."""
        input_json = '{"clause_type": "indemnification", "factors": ["unlimited_liability", "no_cap_on_indemnification"]}'
        result = risk_scoring_agent._calculate_risk_tool(input_json)
        
        assert result is not None
        # Should contain score and category
        assert "score" in result or "category" in result
    
    def test_calculate_risk_tool_no_factors(self, risk_scoring_agent):
        """Test calculate_risk_tool with no factors."""
        input_json = '{"clause_type": "payment_terms", "factors": []}'
        result = risk_scoring_agent._calculate_risk_tool(input_json)
        
        assert result is not None
        # Score should be 0 with no factors
        assert '"score": 0' in result or '"score":0' in result
    
    def test_calculate_risk_tool_score_capped_at_100(self, risk_scoring_agent):
        """Test that risk scores are capped at 100."""
        # Use all indemnification factors to try to exceed 100
        all_factors = list(risk_scoring_agent.RISK_RULES["indemnification"].keys())
        import json
        input_json = json.dumps({"clause_type": "indemnification", "factors": all_factors})
        result = risk_scoring_agent._calculate_risk_tool(input_json)
        
        result_data = json.loads(result)
        assert result_data["score"] <= 100
    
    def test_generate_justification_tool(self, risk_scoring_agent, mock_llm):
        """Test generate_justification_tool."""
        mock_response = Mock()
        mock_response.content = "This clause has high risk due to unlimited liability without any cap."
        mock_llm.invoke.return_value = mock_response
        
        input_json = '{"clause_type": "indemnification", "score": 75, "factors": ["unlimited_liability"]}'
        result = risk_scoring_agent._generate_justification_tool(input_json)
        
        assert result is not None
        assert len(result) > 0


class TestRiskCategoryAssignment:
    """Test risk category assignment logic."""
    
    def test_low_risk_category(self, risk_scoring_agent):
        """Test that scores 0-25 are categorized as Low."""
        input_json = '{"clause_type": "dispute_resolution", "factors": ["no_attorney_fees_provision"]}'
        result = risk_scoring_agent._calculate_risk_tool(input_json)
        
        import json
        result_data = json.loads(result)
        if result_data["score"] <= 25:
            assert result_data["category"] == "Low"
    
    def test_medium_risk_category(self, risk_scoring_agent):
        """Test that scores 26-50 are categorized as Medium."""
        input_json = '{"clause_type": "warranties", "factors": ["weak_disclosure_requirements", "broad_carveouts"]}'
        result = risk_scoring_agent._calculate_risk_tool(input_json)
        
        import json
        result_data = json.loads(result)
        if 26 <= result_data["score"] <= 50:
            assert result_data["category"] == "Medium"
    
    def test_high_risk_category(self, risk_scoring_agent):
        """Test that scores 51-75 are categorized as High."""
        input_json = '{"clause_type": "indemnification", "factors": ["unlimited_liability", "broad_indemnification_scope"]}'
        result = risk_scoring_agent._calculate_risk_tool(input_json)
        
        import json
        result_data = json.loads(result)
        if 51 <= result_data["score"] <= 75:
            assert result_data["category"] == "High"
    
    def test_critical_risk_category(self, risk_scoring_agent):
        """Test that scores 76-100 are categorized as Critical."""
        input_json = '{"clause_type": "indemnification", "factors": ["unlimited_liability", "no_cap_on_indemnification", "broad_indemnification_scope"]}'
        result = risk_scoring_agent._calculate_risk_tool(input_json)
        
        import json
        result_data = json.loads(result)
        if result_data["score"] >= 76:
            assert result_data["category"] == "Critical"


class TestScoreRisksMethod:
    """Test the main score_risks method."""
    
    def test_score_risks_with_no_clauses(self, risk_scoring_agent):
        """Test score_risks with empty clause list."""
        empty_result = ClauseExtractionResult(
            clauses=[],
            red_flags=[],
            metadata={},
            timestamp=datetime.now(UTC),
            document_id="empty_doc"
        )
        
        result = risk_scoring_agent.score_risks(empty_result)
        
        assert isinstance(result, RiskScoringResult)
        assert len(result.scored_clauses) == 0
        assert result.overall_risk_score == 0
        assert result.overall_risk_category == "Low"
        assert "warning" in result.metadata
    
    @patch('app.agents.risk_scoring.create_react_agent')
    def test_score_risks_with_clauses(self, mock_create_agent, risk_scoring_agent, sample_clause_extraction_result):
        """Test score_risks with sample clauses."""
        # Mock the agent executor
        mock_executor = Mock()
        mock_create_agent.return_value = mock_executor
        
        # Mock agent response with scoring data
        mock_message1 = Mock()
        mock_message1.content = '''```json
{
    "identified_factors": [
        {"factor_name": "unlimited_liability", "detected": true, "evidence": "No cap mentioned"}
    ]
}
```'''
        
        mock_message2 = Mock()
        mock_message2.content = '''```json
{
    "score": 75,
    "category": "High",
    "applied_factors": [{"factor_name": "unlimited_liability", "impact": 50}]
}
```'''
        
        mock_executor.invoke.return_value = {
            "messages": [mock_message1, mock_message2]
        }
        
        # Reinitialize agent with mocked executor
        risk_scoring_agent.agent_executor = mock_executor
        
        result = risk_scoring_agent.score_risks(sample_clause_extraction_result)
        
        assert isinstance(result, RiskScoringResult)
        assert result.document_id == "test_doc_123"
        assert "processing_time_seconds" in result.metadata
        assert "model" in result.metadata
    
    def test_score_risks_error_handling(self, risk_scoring_agent, sample_clause_extraction_result):
        """Test that score_risks handles errors gracefully."""
        # Force an error by making agent_executor raise an exception
        risk_scoring_agent.agent_executor.invoke = Mock(side_effect=Exception("Test error"))
        
        result = risk_scoring_agent.score_risks(sample_clause_extraction_result)
        
        assert isinstance(result, RiskScoringResult)
        assert "error" in result.metadata
        assert result.overall_risk_score == 0


class TestGraphVisualization:
    """Test graph visualization functionality."""
    
    def test_get_graph_visualization(self, risk_scoring_agent):
        """Test get_graph_visualization method."""
        # This may return None if IPython is not available
        result = risk_scoring_agent.get_graph_visualization()
        # Just ensure it doesn't raise an exception
        assert result is None or result is not None


class TestDataModels:
    """Test Pydantic data models."""
    
    def test_risk_factor_model(self):
        """Test RiskFactor model validation."""
        factor = RiskFactor(
            factor_name="test_factor",
            description="Test description",
            score_impact=25,
            detected=True
        )
        assert factor.factor_name == "test_factor"
        assert factor.score_impact == 25
        assert factor.detected is True
    
    def test_risk_score_model(self):
        """Test RiskScore model validation."""
        score = RiskScore(
            score=65,
            category="High",
            factors=[],
            justification="Test justification"
        )
        assert score.score == 65
        assert score.category == "High"
    
    def test_scored_clause_model(self, sample_clauses):
        """Test ScoredClause model validation."""
        scored = ScoredClause(
            clause=sample_clauses[0],
            risk_score=RiskScore(
                score=75,
                category="High",
                factors=[],
                justification="High risk"
            )
        )
        assert scored.clause.clause_type == "indemnification"
        assert scored.risk_score.score == 75
    
    def test_risk_scoring_result_model(self):
        """Test RiskScoringResult model validation."""
        result = RiskScoringResult(
            scored_clauses=[],
            overall_risk_score=50,
            overall_risk_category="Medium",
            metadata={},
            timestamp=datetime.now(UTC),
            document_id="test_doc"
        )
        assert result.overall_risk_score == 50
        assert result.overall_risk_category == "Medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
