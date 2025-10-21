"""
Clause Extraction Agent using LangGraph and ReAct pattern.

This agent extracts key clauses and detects red flags from M&A documents
using a ReAct (Reasoning and Acting) pattern with LangGraph orchestration.
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import operator

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.models.agent import (
    ExtractedClause,
    RedFlag,
    ClauseExtractionResult,
)
from app.rag.vector_store import VectorStore
from app.rag.retrievers import get_naive_retriever
from app.core.config import settings

logger = logging.getLogger(__name__)


# Define the agent state
class AgentState(TypedDict):
    """State for the clause extraction agent."""
    messages: Annotated[List[Any], operator.add]
    document_id: str
    session_id: Optional[str]
    chunks_analyzed: int
    clauses: List[Dict[str, Any]]
    red_flags: List[Dict[str, Any]]
    next_action: str


class ClauseExtractionAgent:
    """
    Agent for extracting clauses and detecting red flags in M&A documents.
    
    Uses LangGraph with ReAct pattern for reasoning and action.
    Integrates with Vector Similarity retriever for context retrieval.
    
    Attributes:
        llm: Language model for reasoning and extraction
        vector_store: Vector store for document retrieval
        retriever: Retriever for finding relevant chunks
        graph: LangGraph StateGraph for agent orchestration
    """
    
    # M&A Clause types to extract
    CLAUSE_TYPES = [
        "payment_terms",
        "warranties",
        "indemnification",
        "termination",
        "confidentiality",
        "non_compete",
        "dispute_resolution",
    ]
    
    # Red flag patterns and their severity
    RED_FLAG_PATTERNS = {
        "Critical": [
            "missing material warranties",
            "unlimited indemnification",
            "no cap on liability",
            "extremely broad non-compete (>5 years or global)",
        ],
        "High": [
            "vague payment terms",
            "short survival periods (<12 months)",
            "weak confidentiality protections",
            "unfavorable dispute resolution venue",
        ],
        "Medium": [
            "missing standard representations",
            "ambiguous termination conditions",
            "incomplete disclosure schedules",
            "unusual escrow terms",
        ],
        "Low": [
            "minor drafting inconsistencies",
            "non-standard but acceptable terms",
            "clarification needed",
        ],
    }
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_k: int = 10,
    ):
        """
        Initialize the Clause Extraction Agent.
        
        Args:
            vector_store: Vector store instance (creates new if None)
            model_name: OpenAI model to use (default: gpt-4o-mini)
            temperature: LLM temperature (default: 0.0 for consistency)
            top_k: Number of chunks to retrieve (default: 10)
        """
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
        
        # Initialize retriever
        self.retriever = get_naive_retriever(
            vector_store=self.vector_store,
            top_k=top_k,
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create LangGraph
        self.graph = self._create_graph()
        
        logger.info(
            f"ClauseExtractionAgent initialized with model={model_name}, "
            f"temperature={temperature}, top_k={top_k}"
        )
    
    def _create_tools(self) -> List[Tool]:
        """
        Create tools for the ReAct agent.
        
        Returns:
            List of LangChain Tool objects
        """
        tools = [
            Tool(
                name="search_document",
                description=(
                    "Search the document for relevant clauses using semantic similarity. "
                    "Input should be a search query describing what clause type to find "
                    "(e.g., 'payment terms', 'indemnification clauses', 'termination conditions')."
                ),
                func=self._search_document_tool,
            ),
            Tool(
                name="extract_clause",
                description=(
                    "Extract a specific clause from provided text. "
                    "Input should be JSON with 'text' and 'clause_type' fields. "
                    "Returns structured clause information."
                ),
                func=self._extract_clause_tool,
            ),
            Tool(
                name="detect_red_flags",
                description=(
                    "Analyze text for potential red flags or issues. "
                    "Input should be the clause text to analyze. "
                    "Returns list of detected red flags with severity levels."
                ),
                func=self._detect_red_flags_tool,
            ),
        ]
        return tools
    
    def _search_document_tool(self, query: str) -> str:
        """
        Tool for searching document chunks.
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        try:
            # Use invoke method for LangChain retrievers
            docs = self.retriever.invoke(query)
            
            if not docs:
                return "No relevant documents found."
            
            results = []
            for idx, doc in enumerate(docs[:5], 1):  # Limit to top 5 for context
                chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx}")
                page = doc.metadata.get("page", "unknown")
                results.append(
                    f"[{idx}] (Chunk: {chunk_id}, Page: {page})\n{doc.page_content[:300]}..."
                )
            
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Error in search_document_tool: {e}")
            return f"Error searching documents: {str(e)}"
    
    def _extract_clause_tool(self, input_str: str) -> str:
        """
        Tool for extracting structured clause information.
        
        Args:
            input_str: JSON string with text and clause_type
            
        Returns:
            Extracted clause information
        """
        try:
            import json
            input_data = json.loads(input_str)
            text = input_data.get("text", "")
            clause_type = input_data.get("clause_type", "unknown")
            
            # Use LLM to extract structured information
            prompt = f"""Extract the following information from this {clause_type} clause:
1. The exact clause text
2. Key terms and conditions
3. Any specific obligations or rights
4. Location indicators (section, page if mentioned)
5. Confidence level (0.0-1.0) based on clarity

Clause text:
{text}

Respond in JSON format with keys: clause_text, key_terms, obligations, location, confidence"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Error in extract_clause_tool: {e}")
            return f"Error extracting clause: {str(e)}"
    
    def _detect_red_flags_tool(self, text: str) -> str:
        """
        Tool for detecting red flags in clause text.
        
        Args:
            text: Clause text to analyze
            
        Returns:
            Detected red flags with severity
        """
        try:
            # Build prompt with red flag patterns
            patterns_str = "\n".join([
                f"{severity}: {', '.join(patterns)}"
                for severity, patterns in self.RED_FLAG_PATTERNS.items()
            ])
            
            prompt = f"""Analyze this legal clause for potential red flags or issues.

Known red flag patterns by severity:
{patterns_str}

Clause text:
{text}

For each red flag found, provide:
1. Description of the issue
2. Severity level (Critical, High, Medium, Low)
3. Specific recommendation
4. Category (e.g., 'unfavorable_terms', 'missing_protection', 'ambiguous_language')

Respond in JSON format as a list of red flags."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Error in detect_red_flags_tool: {e}")
            return f"Error detecting red flags: {str(e)}"
    
    def _create_graph(self) -> StateGraph:
        """
        Create LangGraph StateGraph for agent orchestration.
        
        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", ToolNode(self.tools))
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.set_entry_point("reason")
        workflow.add_edge("reason", "act")
        workflow.add_edge("act", "observe")
        
        # Conditional edge from observe
        workflow.add_conditional_edges(
            "observe",
            self._should_continue,
            {
                "continue": "reason",
                "finalize": "finalize",
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _reason_node(self, state: AgentState) -> AgentState:
        """
        Reasoning node - decides what action to take next.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with reasoning
        """
        messages = state["messages"]
        
        # Create reasoning prompt
        system_prompt = f"""You are a legal document analysis agent specializing in M&A agreements.

Your task is to extract clauses and detect red flags. You have access to these tools:
1. search_document - Search for relevant clauses
2. extract_clause - Extract structured clause information
3. detect_red_flags - Identify potential issues

Clause types to find: {', '.join(self.CLAUSE_TYPES)}

Analyze the document systematically. For each clause type:
1. Search for relevant sections
2. Extract the clause details
3. Check for red flags

Current progress:
- Clauses found: {len(state.get('clauses', []))}
- Red flags found: {len(state.get('red_flags', []))}
- Chunks analyzed: {state.get('chunks_analyzed', 0)}

Decide your next action based on what's been done and what remains."""
        
        # Get LLM response with tools
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(
            [SystemMessage(content=system_prompt)] + messages
        )
        
        return {"messages": [response]}
    
    def _observe_node(self, state: AgentState) -> AgentState:
        """
        Observation node - processes tool results.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with observations
        """
        import json
        import re
        
        messages = state["messages"]
        
        # Get current state
        clauses = state.get("clauses", [])
        red_flags = state.get("red_flags", [])
        
        # Look through recent messages for tool results
        # ToolNode adds ToolMessage objects to the messages list
        for msg in reversed(messages[-5:]):  # Check last 5 messages
            if hasattr(msg, "content"):
                content = str(msg.content)
                
                # Try to extract JSON from the content
                json_match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        parsed_data = json.loads(json_match.group(1))
                        
                        # Check if it's a clause extraction result
                        if isinstance(parsed_data, dict) and "clause_text" in parsed_data:
                            # Avoid duplicates
                            if parsed_data not in clauses:
                                clauses.append(parsed_data)
                                logger.info(f"Extracted clause: {parsed_data.get('clause_type', 'unknown')}")
                        
                        # Check if it's red flag detection result
                        elif isinstance(parsed_data, list):
                            for item in parsed_data:
                                if isinstance(item, dict) and ("severity" in item or "severity_level" in item):
                                    # Avoid duplicates
                                    if item not in red_flags:
                                        red_flags.append(item)
                                        logger.info(f"Detected red flag: {item.get('severity', 'unknown')}")
                    
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse JSON: {e}")
        
        observation = f"Processed tool results. Clauses: {len(clauses)}, Red flags: {len(red_flags)}"
        
        return {
            "messages": [AIMessage(content=observation)],
            "chunks_analyzed": state.get("chunks_analyzed", 0) + 1,
            "clauses": clauses,
            "red_flags": red_flags,
        }
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        """
        Finalization node - prepares final results.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with final results
        """
        return {
            "messages": [AIMessage(content="Analysis complete")],
            "next_action": "complete",
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Decide whether to continue the ReAct loop or finalize.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" or "finalize"
        """
        messages = state["messages"]
        chunks_analyzed = state.get("chunks_analyzed", 0)
        clauses_found = len(state.get("clauses", []))
        
        # Stop conditions
        max_iterations = 10
        min_clauses = 3  # Try to find at least 3 clauses
        
        # Stop if we've hit max iterations
        if chunks_analyzed >= max_iterations:
            logger.info(f"Stopping: max iterations reached ({chunks_analyzed})")
            return "finalize"
        
        # Stop if we have enough clauses
        if clauses_found >= min_clauses and chunks_analyzed >= 5:
            logger.info(f"Stopping: found {clauses_found} clauses after {chunks_analyzed} iterations")
            return "finalize"
        
        # Check if last message has tool calls
        last_message = messages[-1] if messages else None
        if last_message and hasattr(last_message, "tool_calls"):
            if not last_message.tool_calls:
                logger.info("Stopping: no more tool calls")
                return "finalize"
            # Continue if there are tool calls
            return "continue"
        
        # Default to continue
        return "continue"
    
    def extract_clauses(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> ClauseExtractionResult:
        """
        Extract clauses and detect red flags from a document.
        
        This is the main entry point for the agent. It executes the ReAct
        loop to systematically analyze the document.
        
        Args:
            document_id: ID of the document to analyze
            session_id: Optional session ID for retrieval
            
        Returns:
            ClauseExtractionResult with extracted clauses and red flags
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting clause extraction for document_id={document_id}")
        
        try:
            # Initialize state
            initial_state: AgentState = {
                "messages": [
                    HumanMessage(
                        content=f"Analyze document {document_id} for M&A clauses and red flags."
                    )
                ],
                "document_id": document_id,
                "session_id": session_id,
                "chunks_analyzed": 0,
                "clauses": [],
                "red_flags": [],
                "next_action": "start",
            }
            
            # Execute graph
            final_state = self.graph.invoke(initial_state)
            
            # Extract results from final state
            clause_dicts = final_state.get("clauses", [])
            red_flag_dicts = final_state.get("red_flags", [])
            
            # Convert to Pydantic models
            clauses = []
            for c in clause_dicts:
                try:
                    clause = ExtractedClause(
                        clause_text=c.get("clause_text", ""),
                        clause_type=c.get("clause_type", "unknown"),
                        location=c.get("location", {}),
                        confidence=float(c.get("confidence", 0.5)),
                        source_chunk_ids=c.get("source_chunk_ids", [])
                    )
                    clauses.append(clause)
                except Exception as e:
                    logger.warning(f"Failed to parse clause: {e}")
            
            red_flags = []
            for r in red_flag_dicts:
                try:
                    red_flag = RedFlag(
                        description=r.get("description", ""),
                        severity=r.get("severity", r.get("severity_level", "Low")),
                        clause_reference=r.get("clause_reference"),
                        recommendation=r.get("recommendation", ""),
                        category=r.get("category", "unknown")
                    )
                    red_flags.append(red_flag)
                except Exception as e:
                    logger.warning(f"Failed to parse red flag: {e}")
            
            # Create result
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result = ClauseExtractionResult(
                clauses=clauses,
                red_flags=red_flags,
                metadata={
                    "processing_time_seconds": processing_time,
                    "model": "gpt-4o-mini",
                    "total_chunks_analyzed": final_state.get("chunks_analyzed", 0),
                    "retriever_type": "naive",
                    "top_k": self.top_k,
                },
                timestamp=datetime.utcnow(),
                document_id=document_id,
            )
            
            logger.info(
                f"Clause extraction complete: {len(clauses)} clauses, "
                f"{len(red_flags)} red flags in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in extract_clauses: {e}", exc_info=True)
            # Return empty result on error
            return ClauseExtractionResult(
                clauses=[],
                red_flags=[],
                metadata={
                    "error": str(e),
                    "processing_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                },
                timestamp=datetime.utcnow(),
                document_id=document_id,
            )
    
    def get_graph_visualization(self) -> Any:
        """
        Get the LangGraph visualization for display in notebooks.
        
        Returns:
            Graph visualization object that can be rendered
            
        Example:
            >>> agent = ClauseExtractionAgent()
            >>> viz = agent.get_graph_visualization()
            >>> # In Jupyter: display(viz)
        """
        try:
            from IPython.display import Image, display
            return Image(self.graph.get_graph().draw_mermaid_png())
        except ImportError:
            logger.warning("IPython not available for visualization")
            return None
        except Exception as e:
            logger.error(f"Error generating graph visualization: {e}")
            return None
