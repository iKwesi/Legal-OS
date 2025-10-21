"""
Clause Extraction Agent using LangGraph and ReAct pattern.

This agent extracts key clauses and detects red flags from M&A documents
using LangGraph's create_react_agent for automatic tool looping.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.models.agent import (
    ExtractedClause,
    RedFlag,
    ClauseExtractionResult,
)
from app.rag.vector_store import VectorStore
from app.rag.retrievers import get_naive_retriever
from app.core.config import settings

logger = logging.getLogger(__name__)


class ClauseExtractionAgent:
    """
    Agent for extracting clauses and detecting red flags in M&A documents.
    
    Uses LangGraph's create_react_agent for automatic ReAct looping.
    """
    
    CLAUSE_TYPES = [
        "payment_terms",
        "warranties",
        "indemnification",
        "termination",
        "confidentiality",
        "non_compete",
        "dispute_resolution",
    ]
    
    RED_FLAG_PATTERNS = {
        "Critical": ["missing material warranties", "unlimited indemnification", "no cap on liability", "extremely broad non-compete (>5 years or global)"],
        "High": ["vague payment terms", "short survival periods (<12 months)", "weak confidentiality protections", "unfavorable dispute resolution venue"],
        "Medium": ["missing standard representations", "ambiguous termination conditions", "incomplete disclosure schedules", "unusual escrow terms"],
        "Low": ["minor drafting inconsistencies", "non-standard but acceptable terms", "clarification needed"],
    }
    
    def __init__(self, vector_store: Optional[VectorStore] = None, model_name: str = "gpt-4o-mini", temperature: float = 0.0, top_k: int = 10):
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=settings.openai_api_key)
        self.retriever = None  # Lazy initialization - created when first needed
        self.tools = self._create_tools()
        self.agent_executor = create_react_agent(model=self.llm, tools=self.tools)
        logger.info(f"ClauseExtractionAgent initialized with create_react_agent (lazy retriever)")
    
    def _ensure_retriever(self):
        """
        Ensure retriever is initialized (lazy initialization pattern).
        
        Creates the retriever on first use, allowing the agent to be initialized
        before the vector store is populated with documents.
        
        Returns:
            Initialized retriever
        """
        if self.retriever is None:
            self.retriever = get_naive_retriever(vector_store=self.vector_store, top_k=self.top_k)
            logger.info(f"Retriever created lazily with top_k={self.top_k}")
        return self.retriever
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search_document",
                description="Search for clauses using semantic similarity. Input: search query string (e.g., 'payment terms'). Returns top 5 relevant chunks.",
                func=self._search_document_tool
            ),
            Tool(
                name="extract_clause",
                description='Extract clause details. Input must be valid JSON string like: {"text": "clause text here", "clause_type": "payment_terms"}. Returns structured clause info.',
                func=self._extract_clause_tool
            ),
            Tool(
                name="detect_red_flags",
                description="Detect red flags in a clause. Input: the clause text as a plain string. Returns JSON list of red flags with severity levels.",
                func=self._detect_red_flags_tool
            ),
        ]
    
    def _search_document_tool(self, query: str) -> str:
        try:
            retriever = self._ensure_retriever()  # Lazy initialization
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            results = []
            for idx, doc in enumerate(docs[:5], 1):
                chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx}")
                page = doc.metadata.get("page", "unknown")
                results.append(f"[{idx}] (Chunk: {chunk_id}, Page: {page})\n{doc.page_content[:300]}...")
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Error in search_document_tool: {e}")
            return f"Error: {str(e)}"
    
    def _extract_clause_tool(self, input_str: str) -> str:
        try:
            input_data = json.loads(input_str)
            text = input_data.get("text", "")
            clause_type = input_data.get("clause_type", "unknown")
            prompt = f"""Extract from this {clause_type} clause: clause_text, clause_type, key_terms, obligations, location, confidence (0.0-1.0).

Clause: {text}

Respond in JSON format."""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error in extract_clause_tool: {e}")
            return f"Error: {str(e)}"
    
    def _detect_red_flags_tool(self, text: str) -> str:
        try:
            patterns_str = "\n".join([f"{sev}: {', '.join(pats)}" for sev, pats in self.RED_FLAG_PATTERNS.items()])
            prompt = f"""Analyze for red flags. Patterns:\n{patterns_str}\n\nClause: {text}\n\nRespond as JSON list with: description, severity, recommendation, category, clause_reference"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error in detect_red_flags_tool: {e}")
            return f"Error: {str(e)}"
    
    def extract_clauses(self, document_id: str, session_id: Optional[str] = None) -> ClauseExtractionResult:
        start_time = datetime.utcnow()
        logger.info(f"Starting clause extraction for document_id={document_id}")
        
        try:
            instruction = f"""Analyze M&A document {document_id}. Extract clauses for: payment_terms, indemnification, warranties (DEMO - just these 3).

For EACH type: 1) search_document 2) extract_clause 3) detect_red_flags

STOP after analyzing these 3 types."""
            
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=instruction)]},
                config={"recursion_limit": 50}
            )
            
            messages = result.get("messages", [])
            logger.info(f"Agent completed with {len(messages)} messages")
            
            # Parse results
            clauses_data = []
            red_flags_data = []
            
            for msg in messages:
                if hasattr(msg, "content"):
                    content = str(msg.content)
                    json_matches = re.findall(r'```json\s*(\{.*?\}|\[.*?\])\s*```', content, re.DOTALL)
                    for json_str in json_matches:
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and "clause_text" in parsed:
                                if parsed not in clauses_data:
                                    clauses_data.append(parsed)
                                    print(f"✅ Found clause: {parsed.get('clause_type')}")
                            elif isinstance(parsed, list):
                                for item in parsed:
                                    if isinstance(item, dict) and ("severity" in item or "severity_level" in item):
                                        if item not in red_flags_data:
                                            red_flags_data.append(item)
                                            print(f"✅ Found red flag: {item.get('severity', item.get('severity_level'))}")
                        except json.JSONDecodeError:
                            continue
            
            # Convert to Pydantic with validation
            clauses = []
            for c in clauses_data:
                try:
                    # Ensure location is a dict
                    location = c.get("location", {})
                    if not isinstance(location, dict):
                        location = {}
                    
                    clause = ExtractedClause(
                        clause_text=c.get("clause_text", ""),
                        clause_type=c.get("clause_type", "unknown"),
                        location=location,
                        confidence=float(c.get("confidence", 0.5)),
                        source_chunk_ids=c.get("source_chunk_ids", [])
                    )
                    clauses.append(clause)
                except Exception as e:
                    logger.warning(f"Failed to parse clause: {e}")
                    print(f"⚠️  Skipped clause due to validation error: {e}")
            
            red_flags = []
            for r in red_flags_data:
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
                    print(f"⚠️  Skipped red flag due to validation error: {e}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ClauseExtractionResult(
                clauses=clauses,
                red_flags=red_flags,
                metadata={"processing_time_seconds": processing_time, "model": "gpt-4o-mini", "total_messages": len(messages), "retriever_type": "naive", "top_k": self.top_k},
                timestamp=datetime.utcnow(),
                document_id=document_id,
            )
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return ClauseExtractionResult(clauses=[], red_flags=[], metadata={"error": str(e), "processing_time_seconds": (datetime.utcnow() - start_time).total_seconds()}, timestamp=datetime.utcnow(), document_id=document_id)
    
    def get_graph_visualization(self) -> Any:
        try:
            from IPython.display import Image
            return Image(self.agent_executor.get_graph().draw_mermaid_png())
        except:
            return None
