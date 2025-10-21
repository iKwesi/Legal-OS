"""
Risk Scoring Agent using LangGraph and ReAct pattern.

This agent assigns risk scores to extracted clauses based on defined rules
and heuristics, using LangGraph's create_react_agent for automatic tool looping.
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
    ExtractedClause,
    ClauseExtractionResult,
    RiskFactor,
    RiskScore,
    ScoredClause,
    RiskScoringResult,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class RiskScoringAgent:
    """
    Agent for scoring risk levels of extracted clauses from M&A documents.
    
    Uses LangGraph's create_react_agent for automatic ReAct looping.
    Applies defined risk scoring rules and heuristics to assign numerical
    risk scores (0-100) and risk categories (Low/Medium/High/Critical).
    """
    
    # Risk scoring rules by clause type
    RISK_RULES = {
        "payment_terms": {
            "unusual_payment_structure": {"impact": 20, "description": "Non-standard payment structure"},
            "delayed_payment_terms": {"impact": 15, "description": "Payment terms delayed beyond standard"},
            "inadequate_escrow": {"impact": 25, "description": "Insufficient escrow arrangements"},
            "vague_earnout_provisions": {"impact": 20, "description": "Unclear earnout terms"},
            "no_payment_security": {"impact": 30, "description": "No security for payment obligations"},
        },
        "warranties": {
            "missing_material_warranties": {"impact": 40, "description": "Critical warranties missing"},
            "short_survival_period": {"impact": 30, "description": "Survival period less than 12 months"},
            "broad_carveouts": {"impact": 25, "description": "Excessive warranty exceptions"},
            "weak_disclosure_requirements": {"impact": 20, "description": "Inadequate disclosure obligations"},
        },
        "indemnification": {
            "unlimited_liability": {"impact": 50, "description": "No cap on indemnification liability"},
            "no_cap_on_indemnification": {"impact": 40, "description": "Missing indemnification cap"},
            "low_basket_threshold": {"impact": 15, "description": "Basket threshold too low"},
            "broad_indemnification_scope": {"impact": 25, "description": "Overly broad indemnification obligations"},
            "short_survival_period": {"impact": 20, "description": "Indemnification survival period too short"},
        },
        "termination": {
            "unfavorable_termination_rights": {"impact": 25, "description": "Unbalanced termination rights"},
            "high_termination_fees": {"impact": 20, "description": "Excessive termination penalties"},
            "vague_termination_conditions": {"impact": 15, "description": "Unclear termination triggers"},
            "unbalanced_termination_rights": {"impact": 30, "description": "One-sided termination provisions"},
        },
        "confidentiality": {
            "weak_confidentiality_terms": {"impact": 20, "description": "Insufficient confidentiality protections"},
            "short_confidentiality_period": {"impact": 15, "description": "Confidentiality period too short"},
            "broad_disclosure_exceptions": {"impact": 25, "description": "Too many disclosure exceptions"},
            "no_return_destruction_clause": {"impact": 10, "description": "Missing return/destruction obligations"},
        },
        "non_compete": {
            "overly_broad_scope": {"impact": 35, "description": "Non-compete scope exceeds 5 years or is global"},
            "global_geographic_scope": {"impact": 30, "description": "Unreasonable geographic restrictions"},
            "vague_restricted_activities": {"impact": 20, "description": "Unclear activity restrictions"},
            "no_reasonable_limitations": {"impact": 25, "description": "Missing reasonable scope limitations"},
        },
        "dispute_resolution": {
            "unfavorable_venue": {"impact": 15, "description": "Disadvantageous dispute resolution venue"},
            "no_arbitration_clause": {"impact": 10, "description": "Missing arbitration provisions"},
            "unclear_governing_law": {"impact": 20, "description": "Ambiguous governing law"},
            "no_attorney_fees_provision": {"impact": 5, "description": "Missing attorney fees clause"},
        },
    }
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the Risk Scoring Agent.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature for LLM responses (0.0 for deterministic)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key
        )
        self.tools = self._create_tools()
        self.agent_executor = create_react_agent(model=self.llm, tools=self.tools)
        logger.info(f"RiskScoringAgent initialized with create_react_agent")
    
    def _create_tools(self) -> List[Tool]:
        """Create the tools available to the agent."""
        return [
            Tool(
                name="analyze_clause_characteristics",
                description=(
                    "Analyze a clause to identify risk characteristics. "
                    "Input must be valid JSON string like: "
                    '{{"clause_text": "text here", "clause_type": "payment_terms"}}. '
                    "Returns JSON with identified risk factors."
                ),
                func=self._analyze_clause_tool
            ),
            Tool(
                name="calculate_risk_score",
                description=(
                    "Calculate numerical risk score based on identified factors. "
                    "Input must be valid JSON string like: "
                    '{{"clause_type": "indemnification", "factors": ["unlimited_liability", "no_cap"]}}. '
                    "Returns JSON with score (0-100) and category."
                ),
                func=self._calculate_risk_tool
            ),
            Tool(
                name="generate_risk_justification",
                description=(
                    "Generate detailed justification for a risk score. "
                    "Input must be valid JSON string like: "
                    '{{"clause_type": "warranties", "score": 65, "factors": ["missing_material_warranties"]}}. '
                    "Returns justification text explaining the risk assessment."
                ),
                func=self._generate_justification_tool
            ),
        ]
    
    def _analyze_clause_tool(self, input_str: str) -> str:
        """
        Analyze a clause to identify risk characteristics.
        
        Args:
            input_str: JSON string with clause_text and clause_type
            
        Returns:
            JSON string with identified risk factors
        """
        try:
            input_data = json.loads(input_str)
            clause_text = input_data.get("clause_text", "")
            clause_type = input_data.get("clause_type", "unknown")
            
            # Get applicable risk rules for this clause type
            rules = self.RISK_RULES.get(clause_type, {})
            if not rules:
                return json.dumps({
                    "error": f"No risk rules defined for clause type: {clause_type}",
                    "identified_factors": []
                })
            
            # Build prompt to identify risk factors
            rules_description = "\n".join([
                f"- {factor}: {info['description']} (impact: +{info['impact']})"
                for factor, info in rules.items()
            ])
            
            prompt = f"""Analyze this {clause_type} clause for risk factors.

Clause text:
{clause_text}

Possible risk factors for {clause_type}:
{rules_description}

Identify which risk factors are present in this clause. For each factor found, explain why it applies.

Respond in JSON format:
{{
    "identified_factors": [
        {{
            "factor_name": "factor_name_here",
            "detected": true,
            "evidence": "explanation of why this factor applies"
        }}
    ]
}}"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input to analyze_clause_tool: {e}")
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            logger.error(f"Error in analyze_clause_tool: {e}")
            return json.dumps({"error": str(e)})
    
    def _calculate_risk_tool(self, input_str: str) -> str:
        """
        Calculate numerical risk score based on identified factors.
        
        Args:
            input_str: JSON string with clause_type and list of factor names
            
        Returns:
            JSON string with score and category
        """
        try:
            input_data = json.loads(input_str)
            clause_type = input_data.get("clause_type", "unknown")
            factors = input_data.get("factors", [])
            
            # Get risk rules for this clause type
            rules = self.RISK_RULES.get(clause_type, {})
            
            # Calculate total score
            total_score = 0
            applied_factors = []
            
            for factor_name in factors:
                if factor_name in rules:
                    impact = rules[factor_name]["impact"]
                    total_score += impact
                    applied_factors.append({
                        "factor_name": factor_name,
                        "impact": impact,
                        "description": rules[factor_name]["description"]
                    })
            
            # Cap at 100
            total_score = min(total_score, 100)
            
            # Determine category
            if total_score <= 25:
                category = "Low"
            elif total_score <= 50:
                category = "Medium"
            elif total_score <= 75:
                category = "High"
            else:
                category = "Critical"
            
            result = {
                "score": total_score,
                "category": category,
                "applied_factors": applied_factors,
                "calculation": f"Sum of factor impacts: {' + '.join([str(f['impact']) for f in applied_factors])} = {total_score}"
            }
            
            return json.dumps(result)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input to calculate_risk_tool: {e}")
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            logger.error(f"Error in calculate_risk_tool: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_justification_tool(self, input_str: str) -> str:
        """
        Generate detailed justification for a risk score.
        
        Args:
            input_str: JSON string with clause_type, score, and factors
            
        Returns:
            Justification text
        """
        try:
            input_data = json.loads(input_str)
            clause_type = input_data.get("clause_type", "unknown")
            score = input_data.get("score", 0)
            factors = input_data.get("factors", [])
            
            prompt = f"""Generate a concise justification for this risk assessment:

Clause Type: {clause_type}
Risk Score: {score}/100
Risk Factors Identified: {', '.join(factors)}

Provide a 2-3 sentence justification explaining why this risk score was assigned and what the key concerns are.
Be specific and actionable."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input to generate_justification_tool: {e}")
            return f"Error: Invalid JSON input: {str(e)}"
        except Exception as e:
            logger.error(f"Error in generate_justification_tool: {e}")
            return f"Error: {str(e)}"
    
    def _parse_scoring_results(
        self,
        messages: List[Any],
        original_clauses: List[ExtractedClause]
    ) -> List[ScoredClause]:
        """
        Parse scoring results from agent messages.
        
        Args:
            messages: Messages from agent execution
            original_clauses: Original extracted clauses
            
        Returns:
            List of scored clauses
        """
        scored_clauses = []
        
        # Extract all JSON blocks from messages (both wrapped and raw formats)
        all_json_data = []
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
                        # Not valid JSON, skip
                        pass
        
        # Try to match scores to clauses
        for clause in original_clauses:
            # Look for scoring data for this clause
            score_data = None
            justification = ""
            factors = []
            
            for data in all_json_data:
                if isinstance(data, dict):
                    # Check if this is a score calculation result
                    if "score" in data and "category" in data:
                        score_data = data
                    # Check if this has factors
                    if "identified_factors" in data:
                        for factor in data["identified_factors"]:
                            if factor.get("detected"):
                                factors.append(RiskFactor(
                                    factor_name=factor.get("factor_name", "unknown"),
                                    description=factor.get("evidence", ""),
                                    score_impact=0,  # Will be filled from rules
                                    detected=True
                                ))
            
            # If we found score data, create scored clause
            if score_data:
                # Get justification from messages
                for msg in messages:
                    if hasattr(msg, "content"):
                        content = str(msg.content)
                        # Look for justification text (not in JSON blocks)
                        if clause.clause_type in content.lower() and len(content) < 500:
                            # Remove JSON blocks
                            clean_content = re.sub(r'```json.*?```', '', content, flags=re.DOTALL)
                            if clean_content.strip() and "risk" in clean_content.lower():
                                justification = clean_content.strip()
                                break
                
                if not justification:
                    justification = f"Risk score of {score_data['score']} assigned based on identified factors."
                
                risk_score = RiskScore(
                    score=score_data.get("score", 0),
                    category=score_data.get("category", "Low"),
                    factors=factors,
                    justification=justification
                )
                
                scored_clause = ScoredClause(
                    clause=clause,
                    risk_score=risk_score
                )
                scored_clauses.append(scored_clause)
                logger.info(f"✅ Scored {clause.clause_type}: {risk_score.score} ({risk_score.category})")
        
        return scored_clauses
    
    def score_risks(
        self,
        clause_extraction_result: ClauseExtractionResult,
        document_id: Optional[str] = None
    ) -> RiskScoringResult:
        """
        Score risks for all extracted clauses.
        
        Args:
            clause_extraction_result: Result from clause extraction agent
            document_id: Optional document ID (uses result's document_id if not provided)
            
        Returns:
            RiskScoringResult with scored clauses and overall risk assessment
        """
        start_time = datetime.now(UTC)
        doc_id = document_id or clause_extraction_result.document_id
        
        logger.info(f"Starting risk scoring for document_id={doc_id}")
        logger.info(f"Scoring {len(clause_extraction_result.clauses)} clauses")
        
        if not clause_extraction_result.clauses:
            logger.warning("No clauses to score")
            return RiskScoringResult(
                scored_clauses=[],
                overall_risk_score=0,
                overall_risk_category="Low",
                metadata={
                    "processing_time_seconds": 0,
                    "model": "gpt-4o-mini",
                    "total_clauses_scored": 0,
                    "warning": "No clauses provided for scoring"
                },
                timestamp=datetime.now(UTC),
                document_id=doc_id
            )
        
        try:
            # Build instruction for agent
            clauses_summary = "\n".join([
                f"{i+1}. {c.clause_type}: {c.clause_text[:100]}..."
                for i, c in enumerate(clause_extraction_result.clauses)
            ])
            
            instruction = f"""Score risk for these {len(clause_extraction_result.clauses)} clauses from document {doc_id}.

Clauses to score:
{clauses_summary}

For EACH clause:
1. analyze_clause_characteristics - identify risk factors
2. calculate_risk_score - compute numerical score
3. generate_risk_justification - explain the score

STOP after scoring all {len(clause_extraction_result.clauses)} clauses."""
            
            # Execute agent
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=instruction)]},
                config={"recursion_limit": 50}
            )
            
            messages = result.get("messages", [])
            logger.info(f"Agent completed with {len(messages)} messages")
            
            # Parse results from messages
            scored_clauses = self._parse_scoring_results(
                messages,
                clause_extraction_result.clauses
            )
            
            # Calculate overall risk
            if scored_clauses:
                overall_score = sum(sc.risk_score.score for sc in scored_clauses) // len(scored_clauses)
            else:
                overall_score = 0
            
            # Determine overall category
            if overall_score <= 25:
                overall_category = "Low"
            elif overall_score <= 50:
                overall_category = "Medium"
            elif overall_score <= 75:
                overall_category = "High"
            else:
                overall_category = "Critical"
            
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            
            return RiskScoringResult(
                scored_clauses=scored_clauses,
                overall_risk_score=overall_score,
                overall_risk_category=overall_category,
                metadata={
                    "processing_time_seconds": processing_time,
                    "model": "gpt-4o-mini",
                    "total_clauses_scored": len(scored_clauses),
                    "total_messages": len(messages)
                },
                timestamp=datetime.now(UTC),
                document_id=doc_id
            )
            
        except Exception as e:
            logger.error(f"Error in risk scoring: {e}", exc_info=True)
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            return RiskScoringResult(
                scored_clauses=[],
                overall_risk_score=0,
                overall_risk_category="Low",
                metadata={
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                    "model": "gpt-4o-mini"
                },
                timestamp=datetime.now(UTC),
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
