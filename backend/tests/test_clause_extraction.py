"""
Tests for the Clause Extraction Agent.

This module contains unit and integration tests for the ClauseExtractionAgent,
including tests for initialization, tools, graph execution, and end-to-end extraction.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from langchain_core.documents import Document

from app.agents.clause_extraction import ClauseExtractionAgent
from app.models.agent import ClauseExtractionResult, ExtractedClause, RedFlag
from app.rag.vector_store import VectorStore


class TestClauseExtractionAgent:
    """Test suite for ClauseExtractionAgent."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = Mock(spec=VectorStore)
        mock_store.as_retriever = Mock()
        return mock_store
    
    @pytest.fixture
    def agent(self, mock_vector_store):
        """Create an agent instance with mocked dependencies."""
        with patch('app.agents.clause_extraction.ChatOpenAI'):
            with patch('app.agents.clause_extraction.get_naive_retriever'):
                agent = ClauseExtractionAgent(
                    vector_store=mock_vector_store,
                    model_name="gpt-4o-mini",
                    temperature=0.0,
                    top_k=10,
                )
                return agent
    
    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent is not None
        assert agent.top_k == 10
        assert len(agent.tools) == 3
        assert agent.agent_executor is not None
        
        # Check tool names
        tool_names = [tool.name for tool in agent.tools]
        assert "search_document" in tool_names
        assert "extract_clause" in tool_names
        assert "detect_red_flags" in tool_names
    
    def test_clause_types_defined(self, agent):
        """Test that M&A clause types are properly defined."""
        assert len(agent.CLAUSE_TYPES) > 0
        assert "payment_terms" in agent.CLAUSE_TYPES
        assert "warranties" in agent.CLAUSE_TYPES
        assert "indemnification" in agent.CLAUSE_TYPES
        assert "termination" in agent.CLAUSE_TYPES
        assert "confidentiality" in agent.CLAUSE_TYPES
        assert "non_compete" in agent.CLAUSE_TYPES
        assert "dispute_resolution" in agent.CLAUSE_TYPES
    
    def test_red_flag_patterns_defined(self, agent):
        """Test that red flag patterns are properly defined."""
        assert len(agent.RED_FLAG_PATTERNS) == 4
        assert "Critical" in agent.RED_FLAG_PATTERNS
        assert "High" in agent.RED_FLAG_PATTERNS
        assert "Medium" in agent.RED_FLAG_PATTERNS
        assert "Low" in agent.RED_FLAG_PATTERNS
        
        # Check that each severity has patterns
        for severity, patterns in agent.RED_FLAG_PATTERNS.items():
            assert len(patterns) > 0
    
    def test_search_document_tool_success(self, agent):
        """Test search_document tool with successful retrieval."""
        # Mock retriever
        mock_docs = [
            Document(
                page_content="The purchase price shall be $10,000,000.",
                metadata={"chunk_id": "chunk_1", "page": 5}
            ),
            Document(
                page_content="Payment shall be made in cash at closing.",
                metadata={"chunk_id": "chunk_2", "page": 5}
            ),
        ]
        # Mock the _ensure_retriever method to return a mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke = Mock(return_value=mock_docs)
        agent._ensure_retriever = Mock(return_value=mock_retriever)
        
        result = agent._search_document_tool("payment terms")
        
        assert result is not None
        assert "chunk_1" in result
        assert "chunk_2" in result
        assert "$10,000,000" in result
    
    def test_search_document_tool_no_results(self, agent):
        """Test search_document tool with no results."""
        # Mock the _ensure_retriever method to return a mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke = Mock(return_value=[])
        agent._ensure_retriever = Mock(return_value=mock_retriever)
        
        result = agent._search_document_tool("nonexistent clause")
        
        assert "No relevant documents found" in result
    
    def test_search_document_tool_error_handling(self, agent):
        """Test search_document tool error handling."""
        # Mock the _ensure_retriever method to return a mock retriever that raises error
        mock_retriever = Mock()
        mock_retriever.invoke = Mock(side_effect=Exception("Retrieval error"))
        agent._ensure_retriever = Mock(return_value=mock_retriever)
        
        result = agent._search_document_tool("payment terms")
        
        assert "Error:" in result
    
    def test_extract_clause_tool_valid_input(self, agent):
        """Test extract_clause tool with valid JSON input."""
        import json
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "clause_text": "The purchase price shall be $10,000,000.",
            "key_terms": ["purchase price", "$10,000,000"],
            "obligations": ["payment at closing"],
            "location": {"section": "2.1", "page": 5},
            "confidence": 0.95
        })
        agent.llm.invoke = Mock(return_value=mock_response)
        
        input_data = json.dumps({
            "text": "The purchase price shall be $10,000,000.",
            "clause_type": "payment_terms"
        })
        
        result = agent._extract_clause_tool(input_data)
        
        assert result is not None
        assert "purchase price" in result.lower()
    
    def test_extract_clause_tool_invalid_json(self, agent):
        """Test extract_clause tool with invalid JSON input."""
        result = agent._extract_clause_tool("not valid json")
        
        assert "Error:" in result
    
    def test_detect_red_flags_tool_success(self, agent):
        """Test detect_red_flags tool with successful detection."""
        import json
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps([
            {
                "description": "No cap on indemnification liability",
                "severity": "Critical",
                "recommendation": "Negotiate a cap",
                "category": "unfavorable_terms"
            }
        ])
        agent.llm.invoke = Mock(return_value=mock_response)
        
        result = agent._detect_red_flags_tool(
            "The seller shall indemnify the buyer for all losses."
        )
        
        assert result is not None
        assert "indemnification" in result.lower()
    
    def test_detect_red_flags_tool_error_handling(self, agent):
        """Test detect_red_flags tool error handling."""
        agent.llm.invoke = Mock(side_effect=Exception("LLM error"))
        
        result = agent._detect_red_flags_tool("Some clause text")
        
        assert "Error:" in result
    
    
    @patch('app.agents.clause_extraction.ChatOpenAI')
    @patch('app.agents.clause_extraction.get_naive_retriever')
    def test_extract_clauses_success(self, mock_retriever, mock_llm, mock_vector_store):
        """Test successful end-to-end clause extraction."""
        # Create agent
        agent = ClauseExtractionAgent(
            vector_store=mock_vector_store,
            model_name="gpt-4o-mini",
        )
        
        # Mock agent executor
        from langchain_core.messages import AIMessage
        mock_result = {
            "messages": [AIMessage(content="Analysis complete")]
        }
        agent.agent_executor.invoke = Mock(return_value=mock_result)
        
        # Execute
        result = agent.extract_clauses(document_id="doc_1")
        
        # Verify
        assert isinstance(result, ClauseExtractionResult)
        assert result.document_id == "doc_1"
        assert "processing_time_seconds" in result.metadata
        assert "model" in result.metadata
        assert result.metadata["model"] == "gpt-4o-mini"
    
    @patch('app.agents.clause_extraction.ChatOpenAI')
    @patch('app.agents.clause_extraction.get_naive_retriever')
    def test_extract_clauses_error_handling(self, mock_retriever, mock_llm, mock_vector_store):
        """Test error handling in extract_clauses."""
        # Create agent
        agent = ClauseExtractionAgent(
            vector_store=mock_vector_store,
            model_name="gpt-4o-mini",
        )
        
        # Mock agent executor to raise error
        agent.agent_executor.invoke = Mock(side_effect=Exception("Agent execution error"))
        
        # Execute
        result = agent.extract_clauses(document_id="doc_1")
        
        # Verify error handling
        assert isinstance(result, ClauseExtractionResult)
        assert result.document_id == "doc_1"
        assert "error" in result.metadata
        assert len(result.clauses) == 0
        assert len(result.red_flags) == 0
    
    def test_get_graph_visualization_success(self, agent):
        """Test graph visualization generation."""
        # Mock the entire IPython.display module
        mock_display_module = MagicMock()
        mock_image_class = MagicMock()
        mock_display_module.Image = mock_image_class
        
        with patch.dict('sys.modules', {'IPython': MagicMock(), 'IPython.display': mock_display_module}):
            mock_graph = Mock()
            mock_graph.draw_mermaid_png = Mock(return_value=b"fake_png_data")
            agent.agent_executor.get_graph = Mock(return_value=mock_graph)
            
            result = agent.get_graph_visualization()
            
            assert result is not None
            mock_image_class.assert_called_once_with(b"fake_png_data")
    
    def test_get_graph_visualization_no_ipython(self, agent):
        """Test graph visualization when IPython is not available."""
        # Patch the import itself to raise ImportError
        import sys
        with patch.dict(sys.modules, {'IPython': None, 'IPython.display': None}):
            result = agent.get_graph_visualization()
            
            assert result is None
    
    def test_get_graph_visualization_error(self, agent):
        """Test graph visualization error handling."""
        agent.agent_executor.get_graph = Mock(side_effect=Exception("Visualization error"))
        
        result = agent.get_graph_visualization()
        
        assert result is None


class TestIntegration:
    """Integration tests for the clause extraction agent."""
    
    @pytest.mark.integration
    @patch('app.agents.clause_extraction.ChatOpenAI')
    @patch('app.agents.clause_extraction.get_naive_retriever')
    def test_full_extraction_workflow(self, mock_retriever, mock_llm):
        """Test complete extraction workflow with mocked LLM."""
        # This test would require actual vector store and documents
        # For now, it's a placeholder for future integration testing
        pass
