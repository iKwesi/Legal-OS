"""
Summary Agent using LangGraph and ReAct pattern.

This agent generates a comprehensive M&A diligence memo from risk scoring results,
using LangGraph's create_react_agent for automatic tool looping.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
import json
import re

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.models.agent import (
    RiskScoringResult,
    ScoredClause,
    KeyFinding,
    Recommendation,
    ExecutiveSummary,
    ClauseSummary,
    DiligenceMemo,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class SummaryAgent:
    """
    Agent for generating M&A diligence memos from risk scoring results.
    
    Uses LangGraph's create_react_agent for automatic ReAct looping.
    Generates comprehensive summaries with executive summary, clause analysis,
    key findings, recommendations, and overall assessment.
    """
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        """
        Initialize the Summary Agent.
        
        Args:
            retriever: Optional retriever for additional context (Vector Similarity)
            model_name: Name of the OpenAI model to use
            temperature: Temperature for LLM responses (0.0 for deterministic)
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key
        )
        self.tools = self._create_tools()
        self.agent_executor = create_react_agent(model=self.llm, tools=self.tools)
        logger.info("SummaryAgent initialized with create_react_agent")
    
    def _create_tools(self) -> List[Tool]:
        """Create the tools available to the agent."""
        return [
            Tool(
                name="retrieve_context",
                description=(
                    "Retrieve additional document context using vector search. "
                    "Input must be valid JSON string like: "
                    '{{"query": "indemnification terms", "k": 3}}. '
                    "Returns relevant document chunks for context."
                ),
                func=self._retrieve_context_tool
            ),
            Tool(
                name="generate_executive_summary",
                description=(
                    "Generate executive summary from risk data. "
                    "Input must be valid JSON string like: "
                    '{{"overall_risk_score": 65, "overall_risk_category": "High", '
                    '"scored_clauses": [...], "document_id": "doc_123"}}. '
                    "Returns JSON with overview, critical_findings, primary_recommendations, overall_risk_assessment."
                ),
                func=self._generate_executive_summary_tool
            ),
            Tool(
                name="generate_clause_summary",
                description=(
                    "Generate summary for a specific clause type. "
                    "Input must be valid JSON string like: "
                    '{{"clause_type": "indemnification", "clause_text": "...", '
                    '"risk_score": 75, "risk_category": "High", "risk_factors": [...]}}. '
                    "Returns JSON with clause_type, summary, risk_level, key_points."
                ),
                func=self._generate_clause_summary_tool
            ),
            Tool(
                name="extract_key_findings",
                description=(
                    "Extract key findings from scored clauses. "
                    "Input must be valid JSON string like: "
                    '{{"scored_clauses": [...]}}. '
                    "Returns JSON array of findings with finding, severity, clause_reference, impact."
                ),
                func=self._extract_key_findings_tool
            ),
            Tool(
                name="generate_recommendations",
                description=(
                    "Generate actionable recommendations from findings. "
                    "Input must be valid JSON string like: "
                    '{{"key_findings": [...], "scored_clauses": [...]}}. '
                    "Returns JSON array of recommendations with recommendation, priority, rationale, related_findings."
                ),
                func=self._generate_recommendations_tool
            ),
            Tool(
                name="generate_overall_assessment",
                description=(
                    "Generate final overall assessment and recommendation. "
                    "Input must be valid JSON string like: "
                    '{{"overall_risk_score": 65, "overall_risk_category": "High", '
                    '"key_findings_count": 5, "critical_findings_count": 2}}. '
                    "Returns assessment text with final recommendation (Proceed/Proceed with Caution/Do Not Proceed)."
                ),
                func=self._generate_overall_assessment_tool
            ),
        ]
    
    def _retrieve_context_tool(self, input_str: str) -> str:
        """
        Retrieve additional document context using vector search.
        
        Args:
            input_str: JSON string with query and k
            
        Returns:
            JSON string with retrieved chunks
        """
        try:
            if not self.retriever:
                return json.dumps({"error": "No retriever configured", "chunks": []})
            
            input_data = json.loads(input_str)
            query = input_data.get("query", "")
            k = input_data.get("k", 3)
            
            # Retrieve relevant chunks
            chunks = self.retriever.invoke(query, k=k)
            
            result = {
                "chunks": [
                    {
                        "content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunks
                ]
            }
            
            return json.dumps(result)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input to retrieve_context_tool: {e}")
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            logger.error(f"Error in retrieve_context_tool: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_executive_summary_tool(self, input_str: str) -> str:
        """
        Generate executive summary from risk data.
        
        Args:
            input_str: JSON string with risk scoring data
            
        Returns:
            JSON string with executive summary
        """
        try:
            input_data = json.loads(input_str)
            overall_risk_score = input_data.get("overall_risk_score", 0)
            overall_risk_category = input_data.get("overall_risk_category", "Low")
            scored_clauses = input_data.get("scored_clauses", [])
            document_id = input_data.get("document_id", "unknown")
            
            # Build context for LLM
            clauses_summary = "\n".join([
                f"- {sc.get('clause_type', 'unknown')}: Risk {sc.get('risk_score', 0)}/100 ({sc.get('risk_category', 'Low')})"
                for sc in scored_clauses
            ])
            
            prompt = f"""Generate an executive summary for this M&A document analysis.

Document ID: {document_id}
Overall Risk Score: {overall_risk_score}/100 ({overall_risk_category})

Scored Clauses:
{clauses_summary}

Provide a concise executive summary in JSON format:
{{
    "overview": "Brief 2-3 sentence overview of the document and transaction",
    "critical_findings": ["Top 3-5 most critical findings"],
    "primary_recommendations": ["Top 3-5 most important recommendations"],
    "overall_risk_assessment": "Overall risk level with brief explanation"
}}

Focus on the highest risk items and most actionable recommendations."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON input to generate_executive_summary_tool: {e}")
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            logger.debug(f"Error in generate_executive_summary_tool: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_clause_summary_tool(self, input_str: str) -> str:
        """
        Generate summary for a specific clause type.
        
        Args:
            input_str: JSON string with clause data
            
        Returns:
            JSON string with clause summary
        """
        try:
            input_data = json.loads(input_str)
            clause_type = input_data.get("clause_type", "unknown")
            clause_text = input_data.get("clause_text", "")
            risk_score = input_data.get("risk_score", 0)
            risk_category = input_data.get("risk_category", "Low")
            risk_factors = input_data.get("risk_factors", [])
            
            factors_text = "\n".join([f"- {f}" for f in risk_factors]) if risk_factors else "None identified"
            
            prompt = f"""Generate a concise summary for this M&A clause.

Clause Type: {clause_type}
Risk Score: {risk_score}/100 ({risk_category})
Risk Factors:
{factors_text}

Clause Text:
{clause_text[:500]}...

Provide a summary in JSON format:
{{
    "clause_type": "{clause_type}",
    "summary": "2-3 sentence summary of the clause provisions",
    "risk_level": "{risk_category}",
    "key_points": ["3-5 key points or notable provisions"]
}}

Be specific and actionable."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input to generate_clause_summary_tool: {e}")
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            logger.error(f"Error in generate_clause_summary_tool: {e}")
            return json.dumps({"error": str(e)})
    
    def _extract_key_findings_tool(self, input_str: str) -> str:
        """
        Extract key findings from scored clauses.
        
        Args:
            input_str: JSON string with scored clauses
            
        Returns:
            JSON string with key findings array
        """
        try:
            input_data = json.loads(input_str)
            scored_clauses = input_data.get("scored_clauses", [])
            
            # Build context for LLM
            clauses_context = "\n\n".join([
                f"Clause: {sc.get('clause_type', 'unknown')}\n"
                f"Risk: {sc.get('risk_score', 0)}/100 ({sc.get('risk_category', 'Low')})\n"
                f"Factors: {', '.join(sc.get('risk_factors', []))}\n"
                f"Text: {sc.get('clause_text', '')[:200]}..."
                for sc in scored_clauses
            ])
            
            prompt = f"""Extract key findings from these scored clauses.

Clauses:
{clauses_context}

Identify the most important findings (focus on High and Critical risk items).
Provide findings in JSON array format:
[
    {{
        "finding": "Clear description of the finding",
        "severity": "Low|Medium|High|Critical",
        "clause_reference": "Clause type or section reference",
        "impact": "Business impact description"
    }}
]

Focus on findings that require attention or action."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON input to extract_key_findings_tool: {e}")
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            logger.debug(f"Error in extract_key_findings_tool: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_recommendations_tool(self, input_str: str) -> str:
        """
        Generate actionable recommendations from findings.
        
        Args:
            input_str: JSON string with key findings
            
        Returns:
            JSON string with recommendations array
        """
        try:
            input_data = json.loads(input_str)
            key_findings = input_data.get("key_findings", [])
            
            findings_context = "\n".join([
                f"- {f.get('finding', 'unknown')} (Severity: {f.get('severity', 'Low')})"
                for f in key_findings
            ])
            
            prompt = f"""Generate actionable recommendations based on these findings.

Key Findings:
{findings_context}

Provide recommendations in JSON array format:
[
    {{
        "recommendation": "Specific actionable recommendation",
        "priority": "Low|Medium|High|Critical",
        "rationale": "Why this recommendation is important",
        "related_findings": ["List of related finding descriptions"]
    }}
]

Focus on practical, actionable steps. Prioritize by severity and business impact."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input to generate_recommendations_tool: {e}")
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            logger.error(f"Error in generate_recommendations_tool: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_overall_assessment_tool(self, input_str: str) -> str:
        """
        Generate final overall assessment and recommendation.
        
        Args:
            input_str: JSON string with summary data
            
        Returns:
            Assessment text with final recommendation
        """
        try:
            input_data = json.loads(input_str)
            overall_risk_score = input_data.get("overall_risk_score", 0)
            overall_risk_category = input_data.get("overall_risk_category", "Low")
            key_findings_count = input_data.get("key_findings_count", 0)
            critical_findings_count = input_data.get("critical_findings_count", 0)
            
            prompt = f"""Generate a final overall assessment for this M&A transaction.

Overall Risk Score: {overall_risk_score}/100 ({overall_risk_category})
Total Key Findings: {key_findings_count}
Critical Findings: {critical_findings_count}

Provide a 2-3 paragraph assessment that includes:
1. Summary of the overall risk profile
2. Key considerations for decision-making
3. Final recommendation: "Proceed", "Proceed with Caution", or "Do Not Proceed"

Be clear, concise, and actionable."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input to generate_overall_assessment_tool: {e}")
            return f"Error: Invalid JSON input: {str(e)}"
        except Exception as e:
            logger.error(f"Error in generate_overall_assessment_tool: {e}")
            return f"Error: {str(e)}"
    
    def _parse_summary_results(
        self,
        messages: List[Any],
        risk_scoring_result: RiskScoringResult
    ) -> DiligenceMemo:
        """
        Parse summary results from agent messages.
        
        Args:
            messages: Messages from agent execution
            risk_scoring_result: Original risk scoring result
            
        Returns:
            Complete DiligenceMemo
        """
        # Extract all JSON blocks from messages (both wrapped and raw formats)
        all_json_data = []
        all_text_data = []
        
        for msg in messages:
            if hasattr(msg, "content"):
                content = str(msg.content)
                
                # Try to find JSON blocks with ```json``` wrapper
                json_matches = re.findall(r'```json\s*(\{.*?\}|\[.*?\])\s*```', content, re.DOTALL)
                for json_str in json_matches:
                    try:
                        parsed = json.loads(json_str)
                        all_json_data.append(parsed)
                    except json.JSONDecodeError:
                        continue
                
                # Also try to parse raw JSON (without wrapper) - common in ToolMessage
                if content.strip().startswith('{') or content.strip().startswith('['):
                    try:
                        parsed = json.loads(content.strip())
                        all_json_data.append(parsed)
                    except json.JSONDecodeError:
                        pass
                
                # Collect text that's not JSON for overall assessment
                clean_content = re.sub(r'```json.*?```', '', content, flags=re.DOTALL)
                if clean_content.strip() and len(clean_content) > 50:
                    all_text_data.append(clean_content.strip())
        
        # Parse executive summary
        executive_summary = None
        for data in all_json_data:
            if isinstance(data, dict) and "overview" in data and "critical_findings" in data:
                executive_summary = ExecutiveSummary(**data)
                break
        
        if not executive_summary:
            # Create default executive summary
            executive_summary = ExecutiveSummary(
                overview=f"M&A document analysis with overall risk score of {risk_scoring_result.overall_risk_score}/100",
                critical_findings=["Analysis in progress"],
                primary_recommendations=["Review detailed findings"],
                overall_risk_assessment=f"{risk_scoring_result.overall_risk_category} risk ({risk_scoring_result.overall_risk_score}/100)"
            )
        
        # Parse clause summaries
        clause_summaries = []
        for data in all_json_data:
            if isinstance(data, dict) and "clause_type" in data and "summary" in data:
                try:
                    clause_summaries.append(ClauseSummary(**data))
                except Exception as e:
                    logger.warning(f"Failed to parse clause summary: {e}")
        
        # Parse key findings
        key_findings = []
        for data in all_json_data:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "finding" in item and "severity" in item:
                        try:
                            key_findings.append(KeyFinding(**item))
                        except Exception as e:
                            logger.warning(f"Failed to parse key finding: {e}")
        
        # Parse recommendations
        recommendations = []
        for data in all_json_data:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "recommendation" in item and "priority" in item:
                        try:
                            recommendations.append(Recommendation(**item))
                        except Exception as e:
                            logger.warning(f"Failed to parse recommendation: {e}")
        
        # Get overall assessment from text data
        overall_assessment = ""
        for text in all_text_data:
            if any(keyword in text.lower() for keyword in ["proceed", "caution", "assessment", "recommendation"]):
                overall_assessment = text
                break
        
        if not overall_assessment:
            overall_assessment = f"Overall risk assessment: {risk_scoring_result.overall_risk_category} ({risk_scoring_result.overall_risk_score}/100). Detailed review recommended."
        
        return DiligenceMemo(
            executive_summary=executive_summary,
            clause_summaries=clause_summaries,
            key_findings=key_findings,
            recommendations=recommendations,
            overall_assessment=overall_assessment,
            metadata={},
            document_id=risk_scoring_result.document_id
        )
    
    def generate_summary(
        self,
        risk_scoring_result: RiskScoringResult,
        document_id: Optional[str] = None
    ) -> DiligenceMemo:
        """
        Generate a comprehensive M&A diligence memo from risk scoring results.
        
        Args:
            risk_scoring_result: Result from risk scoring agent
            document_id: Optional document ID (uses result's document_id if not provided)
            
        Returns:
            DiligenceMemo with complete analysis
        """
        start_time = datetime.now(UTC)
        doc_id = document_id or risk_scoring_result.document_id
        
        logger.info(f"Starting summary generation for document_id={doc_id}")
        logger.info(f"Processing {len(risk_scoring_result.scored_clauses)} scored clauses")
        
        if not risk_scoring_result.scored_clauses:
            logger.warning("No scored clauses to summarize")
            return DiligenceMemo(
                executive_summary=ExecutiveSummary(
                    overview="No clauses available for analysis",
                    critical_findings=[],
                    primary_recommendations=[],
                    overall_risk_assessment="Unable to assess - no data"
                ),
                clause_summaries=[],
                key_findings=[],
                recommendations=[],
                overall_assessment="No data available for assessment",
                metadata={
                    "processing_time_seconds": 0,
                    "model": "gpt-4o-mini",
                    "warning": "No scored clauses provided"
                },
                document_id=doc_id
            )
        
        try:
            # Prepare scored clauses data for tools
            scored_clauses_data = [
                {
                    "clause_type": sc.clause.clause_type,
                    "clause_text": sc.clause.clause_text,
                    "risk_score": sc.risk_score.score,
                    "risk_category": sc.risk_score.category,
                    "risk_factors": [f.factor_name for f in sc.risk_score.factors if f.detected]
                }
                for sc in risk_scoring_result.scored_clauses
            ]
            
            # Build instruction for agent
            instruction = f"""Generate a comprehensive M&A diligence memo for document {doc_id}.

Overall Risk: {risk_scoring_result.overall_risk_score}/100 ({risk_scoring_result.overall_risk_category})
Scored Clauses: {len(scored_clauses_data)}

Steps to complete:
1. generate_executive_summary - Create executive summary with overview, critical findings, and recommendations
2. For EACH of the {len(scored_clauses_data)} clauses: generate_clause_summary
3. extract_key_findings - Extract key findings from all scored clauses
4. generate_recommendations - Generate actionable recommendations
5. generate_overall_assessment - Create final assessment with Proceed/Caution/Do Not Proceed recommendation

STOP after completing all {len(scored_clauses_data)} clause summaries and generating the overall assessment."""
            
            # Execute agent
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=instruction)]},
                config={"recursion_limit": 50}
            )
            
            messages = result.get("messages", [])
            logger.info(f"Agent completed with {len(messages)} messages")
            
            # Parse results from messages
            memo = self._parse_summary_results(messages, risk_scoring_result)
            
            # Add metadata
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            memo.metadata = {
                "processing_time_seconds": processing_time,
                "model": "gpt-4o-mini",
                "total_clauses_analyzed": len(risk_scoring_result.scored_clauses),
                "total_messages": len(messages),
                "clause_summaries_generated": len(memo.clause_summaries),
                "key_findings_extracted": len(memo.key_findings),
                "recommendations_generated": len(memo.recommendations)
            }
            
            logger.info(f"✅ Summary generated: {len(memo.clause_summaries)} clause summaries, "
                       f"{len(memo.key_findings)} findings, {len(memo.recommendations)} recommendations")
            
            return memo
            
        except Exception as e:
            logger.error(f"Error in summary generation: {e}", exc_info=True)
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            return DiligenceMemo(
                executive_summary=ExecutiveSummary(
                    overview="Error occurred during analysis",
                    critical_findings=[],
                    primary_recommendations=[],
                    overall_risk_assessment="Unable to complete assessment"
                ),
                clause_summaries=[],
                key_findings=[],
                recommendations=[],
                overall_assessment=f"Error: {str(e)}",
                metadata={
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                    "model": "gpt-4o-mini"
                },
                document_id=doc_id
            )
    
    def get_graph_visualization(self) -> Any:
        """
        Get a visualization of the agent's graph structure.
        
        Returns:
            IPython Image object for display in notebooks, or None if unavailable
        """
        try:
            from IPython.display import Image
            return Image(self.agent_executor.get_graph().draw_mermaid_png())
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}")
            return None
