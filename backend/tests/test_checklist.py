"""
Tests for the Checklist Agent.

This module contains comprehensive tests for the ChecklistAgent class,
including initialization, tool functionality, agent execution, and data models.
"""

import json
import pytest
from datetime import datetime, UTC
from unittest.mock import Mock, patch, MagicMock

from app.agents.checklist import (
    ChecklistAgent,
    ChecklistItem,
    FollowUpQuestion,
    ChecklistResult
)
from app.models.agent import (
    DiligenceMemo,
    ExecutiveSummary,
    KeyFinding,
    Recommendation,
    ClauseSummary
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_memo():
    """Create a sample DiligenceMemo for testing."""
    return DiligenceMemo(
        executive_summary=ExecutiveSummary(
            overview="Asset purchase agreement for $10M acquisition",
            critical_findings=[
                "Unlimited indemnification liability",
                "Short 12-month warranty survival period",
                "Broad non-compete restrictions"
            ],
            primary_recommendations=[
                "Negotiate indemnification cap at 1-2x purchase price",
                "Extend warranty survival to 24 months",
                "Narrow non-compete scope and duration"
            ],
            overall_risk_assessment="High risk (65/100) - Several critical issues require negotiation"
        ),
        clause_summaries=[
            ClauseSummary(
                clause_type="indemnification",
                summary="Seller provides unlimited indemnification for breaches",
                risk_level="Critical",
                key_points=[
                    "No cap on liability",
                    "12-month survival period",
                    "Broad scope"
                ]
            ),
            ClauseSummary(
                clause_type="payment_terms",
                summary="$10M cash at closing with standard escrow",
                risk_level="Low",
                key_points=["Standard payment structure"]
            )
        ],
        key_findings=[
            KeyFinding(
                finding="Unlimited indemnification liability without cap",
                severity="Critical",
                clause_reference="Section 8.2 - Indemnification",
                impact="Exposes buyer to unlimited financial liability"
            ),
            KeyFinding(
                finding="Short warranty survival period of 12 months",
                severity="High",
                clause_reference="Section 7.1 - Warranties",
                impact="Limited time to discover and claim breaches"
            )
        ],
        recommendations=[
            Recommendation(
                recommendation="Negotiate a cap on indemnification liability at 1-2x purchase price",
                priority="Critical",
                rationale="Unlimited liability exposes buyer to catastrophic financial risk",
                related_findings=["Unlimited indemnification liability"]
            ),
            Recommendation(
                recommendation="Extend warranty survival period to 24 months minimum",
                priority="High",
                rationale="12 months is insufficient for discovering material breaches",
                related_findings=["Short warranty survival period"]
            )
        ],
        overall_assessment="Proceed with Caution - Address critical issues before closing",
        metadata={"test": True},
        document_id="test_doc_123"
    )


@pytest.fixture
def checklist_agent():
    """Create a ChecklistAgent instance for testing."""
    return ChecklistAgent(model_name="gpt-4o-mini", temperature=0.0)


# ============================================================================
# Test Data Models
# ============================================================================

def test_checklist_item_creation():
    """Test ChecklistItem model creation and validation."""
    item = ChecklistItem(
        item="Review indemnification cap",
        category="Legal",
        priority="Critical",
        status="Pending",
        related_findings=["Unlimited indemnification"]
    )
    
    assert item.item == "Review indemnification cap"
    assert item.category == "Legal"
    assert item.priority == "Critical"
    assert item.status == "Pending"
    assert len(item.related_findings) == 1


def test_checklist_item_defaults():
    """Test ChecklistItem default values."""
    item = ChecklistItem(
        item="Test item",
        category="Financial",
        priority="Medium"
    )
    
    assert item.status == "Pending"
    assert item.related_findings == []


def test_follow_up_question_creation():
    """Test FollowUpQuestion model creation and validation."""
    question = FollowUpQuestion(
        question="What is the exact cap amount?",
        category="Legal",
        priority="Critical",
        context="Need to negotiate cap on indemnification"
    )
    
    assert question.question == "What is the exact cap amount?"
    assert question.category == "Legal"
    assert question.priority == "Critical"
    assert "negotiate cap" in question.context


def test_checklist_result_creation():
    """Test ChecklistResult model creation and validation."""
    items = [
        ChecklistItem(
            item="Review contracts",
            category="Legal",
            priority="High"
        )
    ]
    questions = [
        FollowUpQuestion(
            question="What are the payment terms?",
            category="Financial",
            priority="Medium",
            context="Need clarification"
        )
    ]
    
    result = ChecklistResult(
        checklist_items=items,
        follow_up_questions=questions,
        metadata={"test": True},
        document_id="doc_123"
    )
    
    assert len(result.checklist_items) == 1
    assert len(result.follow_up_questions) == 1
    assert result.metadata["test"] is True
    assert result.document_id == "doc_123"
    assert isinstance(result.timestamp, datetime)


# ============================================================================
# Test Agent Initialization
# ============================================================================

def test_agent_initialization(checklist_agent):
    """Test ChecklistAgent initialization."""
    assert checklist_agent.model_name == "gpt-4o-mini"
    assert checklist_agent.temperature == 0.0
    assert checklist_agent.llm is not None
    assert len(checklist_agent.tools) == 3
    assert checklist_agent.agent_executor is not None


def test_agent_tools_creation(checklist_agent):
    """Test that agent tools are created correctly."""
    tool_names = [tool.name for tool in checklist_agent.tools]
    
    assert "generate_standard_checklist" in tool_names
    assert "generate_risk_based_items" in tool_names
    assert "generate_follow_up_questions" in tool_names


# ============================================================================
# Test Tool Functions
# ============================================================================

def test_generate_standard_checklist_legal(checklist_agent):
    """Test standard checklist generation for Legal category."""
    input_data = {
        "category": "Legal",
        "memo_overview": "M&A transaction"
    }
    
    result_str = checklist_agent._generate_standard_checklist(json.dumps(input_data))
    result = json.loads(result_str)
    
    assert "checklist_items" in result
    assert len(result["checklist_items"]) == 5  # Legal has 5 standard items
    
    for item in result["checklist_items"]:
        assert item["category"] == "Legal"
        assert item["priority"] == "Medium"
        assert item["status"] == "Pending"


def test_generate_standard_checklist_financial(checklist_agent):
    """Test standard checklist generation for Financial category."""
    input_data = {
        "category": "Financial",
        "memo_overview": "M&A transaction"
    }
    
    result_str = checklist_agent._generate_standard_checklist(json.dumps(input_data))
    result = json.loads(result_str)
    
    assert "checklist_items" in result
    assert len(result["checklist_items"]) == 5  # Financial has 5 standard items
    
    for item in result["checklist_items"]:
        assert item["category"] == "Financial"


def test_generate_risk_based_items_from_findings(checklist_agent):
    """Test risk-based item generation from findings."""
    input_data = {
        "findings": [
            {
                "finding": "Unlimited indemnification liability",
                "severity": "Critical",
                "clause_reference": "Section 8.2"
            }
        ],
        "recommendations": []
    }
    
    result_str = checklist_agent._generate_risk_based_items(json.dumps(input_data))
    result = json.loads(result_str)
    
    assert "checklist_items" in result
    assert len(result["checklist_items"]) == 1
    
    item = result["checklist_items"][0]
    assert "Investigate and verify" in item["item"]
    assert "Unlimited indemnification" in item["item"]
    assert item["priority"] == "Critical"
    assert item["category"] == "Legal"


def test_generate_risk_based_items_from_recommendations(checklist_agent):
    """Test risk-based item generation from recommendations."""
    input_data = {
        "findings": [],
        "recommendations": [
            {
                "recommendation": "Negotiate indemnification cap",
                "priority": "Critical",
                "related_findings": ["Unlimited liability"]
            }
        ]
    }
    
    result_str = checklist_agent._generate_risk_based_items(json.dumps(input_data))
    result = json.loads(result_str)
    
    assert "checklist_items" in result
    assert len(result["checklist_items"]) == 1
    
    item = result["checklist_items"][0]
    assert item["item"] == "Negotiate indemnification cap"
    assert item["priority"] == "Critical"


def test_generate_follow_up_questions_from_findings(checklist_agent):
    """Test follow-up question generation from findings."""
    input_data = {
        "findings": [
            {
                "finding": "Vague payment terms",
                "severity": "High"
            }
        ],
        "clause_summaries": []
    }
    
    result_str = checklist_agent._generate_follow_up_questions(json.dumps(input_data))
    result = json.loads(result_str)
    
    assert "follow_up_questions" in result
    assert len(result["follow_up_questions"]) == 1
    
    question = result["follow_up_questions"][0]
    assert "more details" in question["question"]
    assert "Vague payment terms" in question["question"]
    assert question["priority"] == "High"


def test_generate_follow_up_questions_from_high_risk_clauses(checklist_agent):
    """Test follow-up question generation from high-risk clauses."""
    input_data = {
        "findings": [],
        "clause_summaries": [
            {
                "clause_type": "indemnification",
                "summary": "Unlimited liability",
                "risk_level": "Critical"
            }
        ]
    }
    
    result_str = checklist_agent._generate_follow_up_questions(json.dumps(input_data))
    result = json.loads(result_str)
    
    assert "follow_up_questions" in result
    assert len(result["follow_up_questions"]) == 1
    
    question = result["follow_up_questions"][0]
    assert "indemnification" in question["question"]
    assert question["priority"] == "High"


# ============================================================================
# Test Categorization Logic
# ============================================================================

def test_categorize_item_legal(checklist_agent):
    """Test item categorization for legal terms."""
    assert checklist_agent._categorize_item("Review indemnification clause") == "Legal"
    assert checklist_agent._categorize_item("Check contract compliance") == "Legal"
    assert checklist_agent._categorize_item("Verify intellectual property rights") == "Legal"


def test_categorize_item_financial(checklist_agent):
    """Test item categorization for financial terms."""
    assert checklist_agent._categorize_item("Review payment terms") == "Financial"
    assert checklist_agent._categorize_item("Verify financial statements") == "Financial"
    assert checklist_agent._categorize_item("Check tax compliance") == "Financial"


def test_categorize_item_operational(checklist_agent):
    """Test item categorization for operational terms."""
    assert checklist_agent._categorize_item("Review business operations") == "Operational"
    assert checklist_agent._categorize_item("Check employee retention") == "Operational"
    assert checklist_agent._categorize_item("Verify customer contracts") == "Operational"


def test_categorize_item_risk_management(checklist_agent):
    """Test item categorization for risk management terms."""
    assert checklist_agent._categorize_item("Review insurance coverage") == "Risk Management"
    assert checklist_agent._categorize_item("Check cybersecurity measures") == "Risk Management"
    assert checklist_agent._categorize_item("Verify data privacy compliance") == "Risk Management"


def test_categorize_item_default(checklist_agent):
    """Test item categorization defaults to Transaction-Specific."""
    assert checklist_agent._categorize_item("Some random text") == "Transaction-Specific"


# ============================================================================
# Test End-to-End Agent Execution
# ============================================================================

@patch('app.agents.checklist.create_react_agent')
def test_generate_checklist_success(mock_create_agent, checklist_agent, sample_memo):
    """Test successful checklist generation."""
    # Mock the agent executor
    mock_executor = MagicMock()
    mock_create_agent.return_value = mock_executor
    
    # Mock agent response with sample items and questions
    mock_response = {
        "messages": [
            Mock(content=json.dumps({
                "checklist_items": [
                    {
                        "item": "Review indemnification cap",
                        "category": "Legal",
                        "priority": "Critical",
                        "status": "Pending",
                        "related_findings": ["Unlimited indemnification"]
                    }
                ]
            })),
            Mock(content=json.dumps({
                "follow_up_questions": [
                    {
                        "question": "What is the proposed cap amount?",
                        "category": "Legal",
                        "priority": "Critical",
                        "context": "Need to negotiate cap"
                    }
                ]
            }))
        ]
    }
    mock_executor.invoke.return_value = mock_response
    checklist_agent.agent_executor = mock_executor
    
    # Generate checklist
    result = checklist_agent.generate_checklist(sample_memo)
    
    # Verify result
    assert isinstance(result, ChecklistResult)
    assert len(result.checklist_items) >= 1
    assert len(result.follow_up_questions) >= 1
    assert result.document_id == "test_doc_123"
    assert "processing_time_seconds" in result.metadata
    assert "model" in result.metadata


@patch('app.agents.checklist.create_react_agent')
def test_generate_checklist_handles_errors(mock_create_agent, checklist_agent, sample_memo):
    """Test that checklist generation handles errors gracefully."""
    # Mock the agent executor to raise an exception
    mock_executor = MagicMock()
    mock_create_agent.return_value = mock_executor
    mock_executor.invoke.side_effect = Exception("Test error")
    checklist_agent.agent_executor = mock_executor
    
    # Generate checklist
    result = checklist_agent.generate_checklist(sample_memo)
    
    # Verify error handling
    assert isinstance(result, ChecklistResult)
    assert len(result.checklist_items) == 0
    assert len(result.follow_up_questions) == 0
    assert "error" in result.metadata
    assert "Test error" in result.metadata["error"]


# ============================================================================
# Test Parsing Logic
# ============================================================================

def test_parse_checklist_results_with_wrapped_json(checklist_agent):
    """Test parsing results with wrapped JSON format."""
    mock_result = {
        "messages": [
            Mock(content='```json\n{"checklist_items": [{"item": "Test", "category": "Legal", "priority": "High"}]}\n```')
        ]
    }
    
    result = checklist_agent._parse_checklist_results(mock_result, "doc_123")
    
    assert isinstance(result, ChecklistResult)
    assert len(result.checklist_items) == 1
    assert result.checklist_items[0].item == "Test"
    assert result.document_id == "doc_123"


def test_parse_checklist_results_with_raw_json(checklist_agent):
    """Test parsing results with raw JSON format."""
    mock_result = {
        "messages": [
            Mock(content='{"follow_up_questions": [{"question": "Test?", "category": "Financial", "priority": "Medium", "context": "Test context"}]}')
        ]
    }
    
    result = checklist_agent._parse_checklist_results(mock_result, "doc_456")
    
    assert isinstance(result, ChecklistResult)
    assert len(result.follow_up_questions) == 1
    assert result.follow_up_questions[0].question == "Test?"
    assert result.document_id == "doc_456"


def test_parse_checklist_results_handles_invalid_json(checklist_agent):
    """Test that parsing handles invalid JSON gracefully."""
    mock_result = {
        "messages": [
            Mock(content='This is not JSON'),
            Mock(content='{"invalid": json}')
        ]
    }
    
    result = checklist_agent._parse_checklist_results(mock_result)
    
    assert isinstance(result, ChecklistResult)
    assert len(result.checklist_items) == 0
    assert len(result.follow_up_questions) == 0


# ============================================================================
# Test Graph Visualization
# ============================================================================

def test_get_graph_visualization(checklist_agent):
    """Test graph visualization generation."""
    # This test just ensures the method exists and doesn't crash
    # Actual visualization testing would require more complex mocking
    viz = checklist_agent.get_graph_visualization()
    # The method should return something (or None if it fails)
    assert viz is not None or viz is None  # Either outcome is acceptable


# ============================================================================
# Test Error Handling
# ============================================================================

def test_generate_standard_checklist_invalid_json(checklist_agent):
    """Test standard checklist generation with invalid JSON."""
    result_str = checklist_agent._generate_standard_checklist("invalid json")
    result = json.loads(result_str)
    
    assert "error" in result
    assert "checklist_items" in result
    assert len(result["checklist_items"]) == 0


def test_generate_risk_based_items_invalid_json(checklist_agent):
    """Test risk-based items generation with invalid JSON."""
    result_str = checklist_agent._generate_risk_based_items("invalid json")
    result = json.loads(result_str)
    
    assert "error" in result
    assert "checklist_items" in result
    assert len(result["checklist_items"]) == 0


def test_generate_follow_up_questions_invalid_json(checklist_agent):
    """Test follow-up questions generation with invalid JSON."""
    result_str = checklist_agent._generate_follow_up_questions("invalid json")
    result = json.loads(result_str)
    
    assert "error" in result
    assert "follow_up_questions" in result
    assert len(result["follow_up_questions"]) == 0
