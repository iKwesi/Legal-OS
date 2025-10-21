"""
Checklist Agent for generating M&A due diligence checklists and follow-up questions.

This agent analyzes the final DiligenceMemo and generates:
1. Structured checklist items categorized by type (Legal, Financial, Operational, etc.)
2. Follow-up questions for additional due diligence
3. Priority assignments based on identified risks

The agent uses LangGraph's create_react_agent for autonomous tool looping,
following the pattern established in Stories 3.1 and 3.2.
"""

import json
import logging
import re
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.agent import DiligenceMemo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models (Task 5)
# ============================================================================

class ChecklistItem(BaseModel):
    """
    Individual checklist item for M&A due diligence.
    
    Attributes:
        item: Description of the checklist item
        category: Category (Legal, Financial, Operational, Risk Management, Transaction-Specific)
        priority: Priority level (Low, Medium, High, Critical)
        status: Current status (Pending, In Progress, Complete, N/A)
        related_findings: References to related findings from the memo
    """
    item: str = Field(
        description="Description of the checklist item"
    )
    category: str = Field(
        description="Category: Legal, Financial, Operational, Risk Management, or Transaction-Specific"
    )
    priority: str = Field(
        description="Priority level: Low, Medium, High, or Critical"
    )
    status: str = Field(
        default="Pending",
        description="Current status: Pending, In Progress, Complete, or N/A"
    )
    related_findings: List[str] = Field(
        default_factory=list,
        description="References to related findings from the diligence memo"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "item": "Review and validate all financial statements for accuracy",
                    "category": "Financial",
                    "priority": "High",
                    "status": "Pending",
                    "related_findings": ["Financial statement discrepancies noted"]
                }
            ]
        }
    }


class FollowUpQuestion(BaseModel):
    """
    Follow-up question for additional due diligence.
    
    Attributes:
        question: The follow-up question to ask
        category: Category (Legal, Financial, Operational, Risk Management, Transaction-Specific)
        priority: Priority level (Low, Medium, High, Critical)
        context: Context explaining why this question is important
    """
    question: str = Field(
        description="The follow-up question to ask"
    )
    category: str = Field(
        description="Category: Legal, Financial, Operational, Risk Management, or Transaction-Specific"
    )
    priority: str = Field(
        description="Priority level: Low, Medium, High, or Critical"
    )
    context: str = Field(
        description="Context explaining why this question is important"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the exact cap amount for indemnification liability?",
                    "category": "Legal",
                    "priority": "Critical",
                    "context": "The memo identified unlimited indemnification as a critical risk"
                }
            ]
        }
    }


class ChecklistResult(BaseModel):
    """
    Complete result from the checklist agent.
    
    Attributes:
        checklist_items: List of all checklist items
        follow_up_questions: List of all follow-up questions
        metadata: Additional metadata about the generation process
        timestamp: When the checklist was generated
        document_id: ID of the document that was analyzed
    """
    checklist_items: List[ChecklistItem] = Field(
        default_factory=list,
        description="List of all generated checklist items"
    )
    follow_up_questions: List[FollowUpQuestion] = Field(
        default_factory=list,
        description="List of all follow-up questions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., processing time, model used)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the checklist was generated"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="ID of the document that was analyzed"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "checklist_items": [
                        {
                            "item": "Review indemnification cap",
                            "category": "Legal",
                            "priority": "Critical",
                            "status": "Pending",
                            "related_findings": ["Unlimited indemnification"]
                        }
                    ],
                    "follow_up_questions": [
                        {
                            "question": "What is the proposed cap?",
                            "category": "Legal",
                            "priority": "Critical",
                            "context": "Need to negotiate cap"
                        }
                    ],
                    "metadata": {"processing_time_seconds": 30.5},
                    "timestamp": "2025-10-21T00:00:00Z",
                    "document_id": "doc_abc123"
                }
            ]
        }
    }


# ============================================================================
# Checklist Agent (Tasks 1, 2, 3, 4, 6, 7, 8)
# ============================================================================

class ChecklistAgent:
    """
    Agent for generating M&A due diligence checklists and follow-up questions.
    
    This agent analyzes a DiligenceMemo and generates:
    - Standard M&A due diligence checklist items
    - Risk-based checklist items based on identified issues
    - Follow-up questions for additional investigation
    - Priority assignments for all items
    
    The agent uses LangGraph's create_react_agent for autonomous operation,
    following the ReAct pattern (Reason -> Act -> Observe).
    """
    
    # Standard M&A Due Diligence Categories (Task 2)
    CHECKLIST_CATEGORIES = {
        "Legal": [
            "Contract review and validation",
            "Compliance verification",
            "Litigation and disputes review",
            "Intellectual property rights verification",
            "Regulatory approvals status"
        ],
        "Financial": [
            "Financial statement verification",
            "Tax compliance review",
            "Debt and liabilities assessment",
            "Working capital analysis",
            "Valuation verification"
        ],
        "Operational": [
            "Business operations review",
            "Key personnel retention plans",
            "Customer and supplier contracts review",
            "Technology and systems assessment",
            "Integration planning"
        ],
        "Risk Management": [
            "Insurance coverage review",
            "Cybersecurity assessment",
            "Environmental compliance check",
            "Data privacy compliance verification",
            "Business continuity plans review"
        ]
    }
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the Checklist Agent.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature for LLM responses (0.0 for deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key
        )
        
        # Create tools (Task 3)
        self.tools = self._create_tools()
        
        # Create ReAct agent using LangGraph (Task 4)
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools
        )
        
        logger.info(f"ChecklistAgent initialized with model: {model_name}")
    
    def _create_tools(self) -> List[Tool]:
        """
        Create the tools for the checklist agent (Task 3).
        
        Returns:
            List of Tool objects for the agent
        """
        return [
            Tool(
                name="generate_standard_checklist",
                description=(
                    "Generate standard M&A due diligence checklist items for a given category. "
                    "Input must be a valid JSON string like: "
                    '{\"category\": \"Legal\", \"memo_overview\": \"Brief overview of the transaction\"}'
                ),
                func=self._generate_standard_checklist
            ),
            Tool(
                name="generate_risk_based_items",
                description=(
                    "Generate checklist items based on identified risks and findings. "
                    "Input must be a valid JSON string like: "
                    '{\"findings\": [{\"finding\": \"Issue description\", \"severity\": \"High\"}], '
                    '\"recommendations\": [{\"recommendation\": \"Action to take\", \"priority\": \"Critical\"}]}'
                ),
                func=self._generate_risk_based_items
            ),
            Tool(
                name="generate_follow_up_questions",
                description=(
                    "Generate follow-up questions based on the analysis. "
                    "Input must be a valid JSON string like: "
                    '{\"findings\": [{\"finding\": \"Issue\", \"severity\": \"High\"}], '
                    '\"clause_summaries\": [{\"clause_type\": \"indemnification\", \"summary\": \"Details\"}]}'
                ),
                func=self._generate_follow_up_questions
            )
        ]
    
    def _generate_standard_checklist(self, input_str: str) -> str:
        """
        Generate standard M&A due diligence checklist items (Task 6).
        
        Args:
            input_str: JSON string with category and memo overview
            
        Returns:
            JSON string with checklist items
        """
        try:
            input_data = json.loads(input_str)
            category = input_data.get("category", "Legal")
            
            # Get standard items for this category
            standard_items = self.CHECKLIST_CATEGORIES.get(category, [])
            
            # Create checklist items with Medium priority (standard items)
            items = []
            for item_text in standard_items:
                items.append({
                    "item": item_text,
                    "category": category,
                    "priority": "Medium",
                    "status": "Pending",
                    "related_findings": []
                })
            
            result = {"checklist_items": items}
            logger.info(f"Generated {len(items)} standard checklist items for category: {category}")
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error generating standard checklist: {e}")
            return json.dumps({"error": str(e), "checklist_items": []})
    
    def _generate_risk_based_items(self, input_str: str) -> str:
        """
        Generate risk-based checklist items (Task 6).
        
        Args:
            input_str: JSON string with findings and recommendations
            
        Returns:
            JSON string with risk-based checklist items
        """
        try:
            input_data = json.loads(input_str)
            findings = input_data.get("findings", [])
            recommendations = input_data.get("recommendations", [])
            
            items = []
            
            # Generate items from findings
            for finding in findings:
                finding_text = finding.get("finding", "")
                severity = finding.get("severity", "Medium")
                clause_ref = finding.get("clause_reference", "")
                
                # Map severity to priority
                priority_map = {
                    "Critical": "Critical",
                    "High": "High",
                    "Medium": "Medium",
                    "Low": "Low"
                }
                priority = priority_map.get(severity, "Medium")
                
                # Determine category based on finding content
                category = self._categorize_item(finding_text)
                
                # Create investigation item
                item_text = f"Investigate and verify: {finding_text}"
                if clause_ref:
                    item_text += f" (Reference: {clause_ref})"
                
                items.append({
                    "item": item_text,
                    "category": category,
                    "priority": priority,
                    "status": "Pending",
                    "related_findings": [finding_text]
                })
            
            # Generate items from recommendations
            for rec in recommendations:
                rec_text = rec.get("recommendation", "")
                priority = rec.get("priority", "Medium")
                related = rec.get("related_findings", [])
                
                category = self._categorize_item(rec_text)
                
                items.append({
                    "item": rec_text,
                    "category": category,
                    "priority": priority,
                    "status": "Pending",
                    "related_findings": related
                })
            
            result = {"checklist_items": items}
            logger.info(f"Generated {len(items)} risk-based checklist items")
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error generating risk-based items: {e}")
            return json.dumps({"error": str(e), "checklist_items": []})
    
    def _generate_follow_up_questions(self, input_str: str) -> str:
        """
        Generate follow-up questions for additional due diligence (Task 6).
        
        Args:
            input_str: JSON string with findings and clause summaries
            
        Returns:
            JSON string with follow-up questions
        """
        try:
            input_data = json.loads(input_str)
            findings = input_data.get("findings", [])
            clause_summaries = input_data.get("clause_summaries", [])
            
            questions = []
            
            # Generate questions from findings
            for finding in findings:
                finding_text = finding.get("finding", "")
                severity = finding.get("severity", "Medium")
                
                # Map severity to priority
                priority_map = {
                    "Critical": "Critical",
                    "High": "High",
                    "Medium": "Medium",
                    "Low": "Low"
                }
                priority = priority_map.get(severity, "Medium")
                
                category = self._categorize_item(finding_text)
                
                # Generate clarifying question
                question_text = f"Can you provide more details about: {finding_text}?"
                context = f"This finding was identified with {severity} severity and requires clarification"
                
                questions.append({
                    "question": question_text,
                    "category": category,
                    "priority": priority,
                    "context": context
                })
            
            # Generate questions from clause summaries with high risk
            for clause_summary in clause_summaries:
                risk_level = clause_summary.get("risk_level", "Low")
                if risk_level in ["High", "Critical"]:
                    clause_type = clause_summary.get("clause_type", "")
                    summary = clause_summary.get("summary", "")
                    
                    category = self._categorize_item(clause_type)
                    priority = "High" if risk_level == "Critical" else "Medium"
                    
                    question_text = f"What are the specific terms and conditions for {clause_type.replace('_', ' ')}?"
                    context = f"The {clause_type.replace('_', ' ')} clause has {risk_level} risk level and requires detailed review"
                    
                    questions.append({
                        "question": question_text,
                        "category": category,
                        "priority": priority,
                        "context": context
                    })
            
            result = {"follow_up_questions": questions}
            logger.info(f"Generated {len(questions)} follow-up questions")
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return json.dumps({"error": str(e), "follow_up_questions": []})
    
    def _categorize_item(self, text: str) -> str:
        """
        Categorize an item based on its content (Task 6).
        
        Args:
            text: Text to categorize
            
        Returns:
            Category name (Legal, Financial, Operational, Risk Management, Transaction-Specific)
        """
        text_lower = text.lower()
        
        # Check more specific categories first to avoid keyword overlap
        
        # Financial keywords (check first for tax, financial statements, etc.)
        if any(keyword in text_lower for keyword in [
            "financial statement", "tax compliance", "payment", "price", "debt", 
            "valuation", "working capital", "escrow", "earnout", "revenue", "profit", "loss"
        ]):
            return "Financial"
        
        # Operational keywords (check before Legal to catch customer/supplier contracts)
        if any(keyword in text_lower for keyword in [
            "operational", "operations", "personnel", "employee", "customer contract",
            "supplier contract", "customer and supplier", "technology", "system", 
            "integration", "business process", "retention"
        ]):
            return "Operational"
        
        # Risk Management keywords (check before Legal to catch specific compliance types)
        if any(keyword in text_lower for keyword in [
            "insurance", "cybersecurity", "security", "environmental compliance",
            "data privacy compliance", "privacy", "business continuity", "disaster recovery"
        ]):
            return "Risk Management"
        
        # Legal keywords (more general, check last)
        if any(keyword in text_lower for keyword in [
            "contract review", "contract compliance", "legal", "litigation", "dispute", 
            "regulatory approval", "intellectual property", "ip", "indemnification", 
            "warranty", "representation", "termination", "confidentiality", "non-compete", 
            "governing law", "arbitration", "compliance verification"
        ]):
            return "Legal"
        
        # Default to Transaction-Specific
        return "Transaction-Specific"
    
    def generate_checklist(self, memo: DiligenceMemo) -> ChecklistResult:
        """
        Generate complete checklist and follow-up questions from a DiligenceMemo (Task 8).
        
        This is the main entry point for the agent. It analyzes the memo and generates:
        1. Standard M&A due diligence checklist items
        2. Risk-based checklist items from findings and recommendations
        3. Follow-up questions for additional investigation
        
        Args:
            memo: DiligenceMemo from the Summary Agent
            
        Returns:
            ChecklistResult with all checklist items and follow-up questions
        """
        start_time = datetime.now(UTC)
        logger.info("Starting checklist generation")
        
        try:
            # Prepare memo data for the agent
            memo_data = {
                "overview": memo.executive_summary.overview,
                "overall_risk": memo.executive_summary.overall_risk_assessment,
                "critical_findings": memo.executive_summary.critical_findings,
                "findings": [
                    {
                        "finding": f.finding,
                        "severity": f.severity,
                        "clause_reference": f.clause_reference,
                        "impact": f.impact
                    }
                    for f in memo.key_findings
                ],
                "recommendations": [
                    {
                        "recommendation": r.recommendation,
                        "priority": r.priority,
                        "rationale": r.rationale,
                        "related_findings": r.related_findings
                    }
                    for r in memo.recommendations
                ],
                "clause_summaries": [
                    {
                        "clause_type": cs.clause_type,
                        "summary": cs.summary,
                        "risk_level": cs.risk_level,
                        "key_points": cs.key_points
                    }
                    for cs in memo.clause_summaries
                ]
            }
            
            # Create prompt for the agent (Task 4)
            prompt = f"""You are an M&A due diligence checklist generator. Analyze the following diligence memo and generate:
1. Standard M&A due diligence checklist items for all relevant categories (Legal, Financial, Operational, Risk Management)
2. Risk-based checklist items based on the identified findings and recommendations
3. Follow-up questions for additional investigation

Diligence Memo Summary:
{json.dumps(memo_data, indent=2)}

Instructions:
- Use the generate_standard_checklist tool for each category (Legal, Financial, Operational, Risk Management)
- Use the generate_risk_based_items tool to create items from findings and recommendations
- Use the generate_follow_up_questions tool to create questions from findings and high-risk clauses
- After generating all items and questions, respond with "COMPLETE" to stop

Begin generating the checklist now."""
            
            # Execute the agent (Task 4)
            logger.info("Invoking ReAct agent for checklist generation")
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": 50}
            )
            
            # Parse results from agent messages (Task 8)
            checklist_result = self._parse_checklist_results(result, memo.document_id)
            
            # Calculate processing time
            end_time = datetime.now(UTC)
            processing_time = (end_time - start_time).total_seconds()
            
            # Add metadata
            checklist_result.metadata.update({
                "processing_time_seconds": processing_time,
                "model": self.model_name,
                "total_items": len(checklist_result.checklist_items),
                "total_questions": len(checklist_result.follow_up_questions),
                "memo_findings_count": len(memo.key_findings),
                "memo_recommendations_count": len(memo.recommendations)
            })
            
            logger.info(
                f"Checklist generation complete: {len(checklist_result.checklist_items)} items, "
                f"{len(checklist_result.follow_up_questions)} questions in {processing_time:.2f}s"
            )
            
            return checklist_result
            
        except Exception as e:
            logger.error(f"Error generating checklist: {e}")
            # Return empty result with error in metadata
            return ChecklistResult(
                checklist_items=[],
                follow_up_questions=[],
                metadata={
                    "error": str(e),
                    "processing_time_seconds": (datetime.now(UTC) - start_time).total_seconds()
                },
                document_id=memo.document_id
            )
    
    def _parse_checklist_results(
        self,
        agent_result: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> ChecklistResult:
        """
        Parse checklist results from agent messages (Task 8).
        
        Args:
            agent_result: Result from agent execution
            document_id: Optional document ID
            
        Returns:
            ChecklistResult with parsed items and questions
        """
        all_items = []
        all_questions = []
        
        messages = agent_result.get("messages", [])
        
        for msg in messages:
            content = str(msg.content)
            
            # Try to extract JSON from wrapped format: ```json {...} ```
            json_matches = re.findall(
                r'```json\s*(\{.*?\}|\[.*?\])\s*```',
                content,
                re.DOTALL
            )
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    
                    # Extract checklist items
                    if "checklist_items" in data:
                        for item_data in data["checklist_items"]:
                            try:
                                item = ChecklistItem(**item_data)
                                all_items.append(item)
                            except Exception as e:
                                logger.warning(f"Failed to parse checklist item: {e}")
                    
                    # Extract follow-up questions
                    if "follow_up_questions" in data:
                        for q_data in data["follow_up_questions"]:
                            try:
                                question = FollowUpQuestion(**q_data)
                                all_questions.append(question)
                            except Exception as e:
                                logger.warning(f"Failed to parse follow-up question: {e}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from message: {e}")
            
            # Try raw JSON format (common in ToolMessage)
            if content.strip().startswith('{') or content.strip().startswith('['):
                try:
                    data = json.loads(content.strip())
                    
                    # Extract checklist items
                    if "checklist_items" in data:
                        for item_data in data["checklist_items"]:
                            try:
                                item = ChecklistItem(**item_data)
                                all_items.append(item)
                            except Exception as e:
                                logger.warning(f"Failed to parse checklist item: {e}")
                    
                    # Extract follow-up questions
                    if "follow_up_questions" in data:
                        for q_data in data["follow_up_questions"]:
                            try:
                                question = FollowUpQuestion(**q_data)
                                all_questions.append(question)
                            except Exception as e:
                                logger.warning(f"Failed to parse follow-up question: {e}")
                
                except json.JSONDecodeError:
                    pass  # Not valid JSON, skip
        
        logger.info(f"Parsed {len(all_items)} checklist items and {len(all_questions)} questions from agent messages")
        
        return ChecklistResult(
            checklist_items=all_items,
            follow_up_questions=all_questions,
            metadata={},
            document_id=document_id
        )
    
    def get_graph_visualization(self) -> Any:
        """
        Get the LangGraph visualization for this agent (Task 7).
        
        Returns:
            Graph visualization object that can be displayed in notebooks
        """
        try:
            return self.agent_executor.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.error(f"Error generating graph visualization: {e}")
            return None
