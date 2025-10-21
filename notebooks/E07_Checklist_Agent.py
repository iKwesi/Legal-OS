# %% [markdown]
# # E07: Checklist Agent Demonstration
# 
# This notebook demonstrates the Checklist Agent, which generates M&A due diligence
# checklists and follow-up questions based on the DiligenceMemo from the Summary Agent.
#
# **Agent Capabilities:**
# - Generate standard M&A due diligence checklist items
# - Create risk-based checklist items from findings
# - Generate follow-up questions for additional investigation
# - Categorize items by type (Legal, Financial, Operational, Risk Management)
# - Assign priorities based on severity

# %%
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path("../backend").resolve()
sys.path.insert(0, str(backend_path))

# %%
from app.agents.checklist import ChecklistAgent, ChecklistItem, FollowUpQuestion, ChecklistResult
from app.models.agent import (
    DiligenceMemo,
    ExecutiveSummary,
    KeyFinding,
    Recommendation,
    ClauseSummary
)

# %% [markdown]
# ## 1. Initialize the Checklist Agent

# %%
agent = ChecklistAgent(model_name="gpt-4o-mini", temperature=0.0)
print(f"✓ ChecklistAgent initialized with model: {agent.model_name}")
print(f"✓ Number of tools: {len(agent.tools)}")
print(f"✓ Tool names: {[tool.name for tool in agent.tools]}")

# %% [markdown]
# ## 2. Create Sample DiligenceMemo
#
# We'll create a sample memo with findings and recommendations to test the agent.

# %%
sample_memo = DiligenceMemo(
    executive_summary=ExecutiveSummary(
        overview="Asset purchase agreement for $10M acquisition of manufacturing company",
        critical_findings=[
            "Unlimited indemnification liability without cap",
            "Short 12-month warranty survival period",
            "Broad non-compete restrictions (5 years, global scope)"
        ],
        primary_recommendations=[
            "Negotiate indemnification cap at 1-2x purchase price",
            "Extend warranty survival to 24 months minimum",
            "Narrow non-compete scope to relevant markets only"
        ],
        overall_risk_assessment="High risk (68/100) - Several critical issues require immediate negotiation"
    ),
    clause_summaries=[
        ClauseSummary(
            clause_type="indemnification",
            summary="Seller provides unlimited indemnification for breaches of warranties",
            risk_level="Critical",
            key_points=[
                "No cap on liability",
                "12-month survival period",
                "Broad scope of indemnification"
            ]
        ),
        ClauseSummary(
            clause_type="payment_terms",
            summary="$10M cash at closing with 10% escrow for 18 months",
            risk_level="Low",
            key_points=["Standard payment structure", "Reasonable escrow terms"]
        ),
        ClauseSummary(
            clause_type="non_compete",
            summary="Seller restricted from competing for 5 years globally",
            risk_level="High",
            key_points=["Overly broad geographic scope", "Long duration"]
        )
    ],
    key_findings=[
        KeyFinding(
            finding="Unlimited indemnification liability without cap",
            severity="Critical",
            clause_reference="Section 8.2 - Indemnification",
            impact="Exposes buyer to unlimited financial liability for warranty breaches"
        ),
        KeyFinding(
            finding="Short warranty survival period of 12 months",
            severity="High",
            clause_reference="Section 7.1 - Warranties",
            impact="Limited time to discover and claim breaches of representations"
        ),
        KeyFinding(
            finding="Overly broad non-compete restrictions",
            severity="High",
            clause_reference="Section 9.3 - Non-Compete",
            impact="May be unenforceable and creates uncertainty"
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
            rationale="12 months is insufficient for discovering material breaches in manufacturing operations",
            related_findings=["Short warranty survival period"]
        ),
        Recommendation(
            recommendation="Narrow non-compete scope to relevant geographic markets and reduce duration to 3 years",
            priority="High",
            rationale="Current scope may be unenforceable and creates legal uncertainty",
            related_findings=["Overly broad non-compete restrictions"]
        )
    ],
    overall_assessment="Proceed with Caution - Address critical indemnification and warranty issues before closing",
    metadata={"demo": True},
    document_id="demo_doc_001"
)

print("✓ Sample DiligenceMemo created")
print(f"  - Findings: {len(sample_memo.key_findings)}")
print(f"  - Recommendations: {len(sample_memo.recommendations)}")
print(f"  - Clause Summaries: {len(sample_memo.clause_summaries)}")

# %% [markdown]
# ## 3. Generate Checklist
#
# Now let's use the agent to generate a comprehensive checklist and follow-up questions.

# %%
print("Generating checklist... (this may take 30-60 seconds)")
result = agent.generate_checklist(sample_memo)

print(f"\n✓ Checklist generation complete!")
print(f"  - Total checklist items: {len(result.checklist_items)}")
print(f"  - Total follow-up questions: {len(result.follow_up_questions)}")
print(f"  - Processing time: {result.metadata.get('processing_time_seconds', 0):.2f}s")

# %% [markdown]
# ## 4. Analyze Checklist Items by Category

# %%
from collections import defaultdict

items_by_category = defaultdict(list)
for item in result.checklist_items:
    items_by_category[item.category].append(item)

print("\n=== CHECKLIST ITEMS BY CATEGORY ===\n")
for category in sorted(items_by_category.keys()):
    items = items_by_category[category]
    print(f"{category}: {len(items)} items")
    for item in items[:3]:  # Show first 3 items per category
        print(f"  [{item.priority}] {item.item[:80]}...")
    if len(items) > 3:
        print(f"  ... and {len(items) - 3} more items")
    print()

# %% [markdown]
# ## 5. Analyze Checklist Items by Priority

# %%
items_by_priority = defaultdict(list)
for item in result.checklist_items:
    items_by_priority[item.priority].append(item)

print("\n=== CHECKLIST ITEMS BY PRIORITY ===\n")
for priority in ["Critical", "High", "Medium", "Low"]:
    if priority in items_by_priority:
        items = items_by_priority[priority]
        print(f"{priority}: {len(items)} items")
        for item in items[:2]:  # Show first 2 items per priority
            print(f"  [{item.category}] {item.item[:80]}...")
        if len(items) > 2:
            print(f"  ... and {len(items) - 2} more items")
        print()

# %% [markdown]
# ## 6. Review Follow-Up Questions

# %%
questions_by_category = defaultdict(list)
for question in result.follow_up_questions:
    questions_by_category[question.category].append(question)

print("\n=== FOLLOW-UP QUESTIONS BY CATEGORY ===\n")
for category in sorted(questions_by_category.keys()):
    questions = questions_by_category[category]
    print(f"{category}: {len(questions)} questions")
    for q in questions[:2]:  # Show first 2 questions per category
        print(f"  [{q.priority}] {q.question[:80]}...")
        print(f"      Context: {q.context[:60]}...")
    if len(questions) > 2:
        print(f"  ... and {len(questions) - 2} more questions")
    print()

# %% [markdown]
# ## 7. Display Critical Items
#
# Let's focus on the most critical items that require immediate attention.

# %%
critical_items = [item for item in result.checklist_items if item.priority == "Critical"]

print("\n=== CRITICAL PRIORITY ITEMS ===\n")
for i, item in enumerate(critical_items, 1):
    print(f"{i}. [{item.category}] {item.item}")
    if item.related_findings:
        print(f"   Related findings: {', '.join(item.related_findings)}")
    print()

# %% [markdown]
# ## 8. Visualize Agent Graph
#
# Display the LangGraph structure of the Checklist Agent.

# %%
try:
    from IPython.display import Image, display
    
    viz = agent.get_graph_visualization()
    if viz:
        print("✓ Agent graph visualization:")
        display(Image(viz))
    else:
        print("⚠ Graph visualization not available")
except Exception as e:
    print(f"⚠ Could not display graph: {e}")

# %% [markdown]
# ## 9. Export Results
#
# Export the checklist to JSON for further processing.

# %%
import json

# Convert to dict for JSON serialization
checklist_dict = {
    "document_id": result.document_id,
    "timestamp": result.timestamp.isoformat(),
    "metadata": result.metadata,
    "checklist_items": [
        {
            "item": item.item,
            "category": item.category,
            "priority": item.priority,
            "status": item.status,
            "related_findings": item.related_findings
        }
        for item in result.checklist_items
    ],
    "follow_up_questions": [
        {
            "question": q.question,
            "category": q.category,
            "priority": q.priority,
            "context": q.context
        }
        for q in result.follow_up_questions
    ]
}

# Save to file
output_file = backend_path / "checklist_results.json"
with open(output_file, "w") as f:
    json.dump(checklist_dict, f, indent=2)

print(f"✓ Checklist exported to: {output_file}")
print(f"  - Total items: {len(checklist_dict['checklist_items'])}")
print(f"  - Total questions: {len(checklist_dict['follow_up_questions'])}")

# %% [markdown]
# ## 10. Summary Statistics

# %%
print("\n=== CHECKLIST GENERATION SUMMARY ===\n")
print(f"Document ID: {result.document_id}")
print(f"Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Processing Time: {result.metadata.get('processing_time_seconds', 0):.2f}s")
print(f"Model Used: {result.metadata.get('model', 'N/A')}")
print()
print("Checklist Items:")
print(f"  - Total: {result.metadata.get('total_items', 0)}")
print(f"  - Critical: {len([i for i in result.checklist_items if i.priority == 'Critical'])}")
print(f"  - High: {len([i for i in result.checklist_items if i.priority == 'High'])}")
print(f"  - Medium: {len([i for i in result.checklist_items if i.priority == 'Medium'])}")
print(f"  - Low: {len([i for i in result.checklist_items if i.priority == 'Low'])}")
print()
print("Follow-Up Questions:")
print(f"  - Total: {result.metadata.get('total_questions', 0)}")
print(f"  - Critical: {len([q for q in result.follow_up_questions if q.priority == 'Critical'])}")
print(f"  - High: {len([q for q in result.follow_up_questions if q.priority == 'High'])}")
print(f"  - Medium: {len([q for q in result.follow_up_questions if q.priority == 'Medium'])}")
print()
print("Categories Covered:")
categories = set(item.category for item in result.checklist_items)
for cat in sorted(categories):
    count = len([i for i in result.checklist_items if i.category == cat])
    print(f"  - {cat}: {count} items")

# %% [markdown]
# ## Conclusion
#
# The Checklist Agent successfully:
# - ✅ Generated standard M&A due diligence checklist items
# - ✅ Created risk-based items from findings and recommendations
# - ✅ Generated follow-up questions for additional investigation
# - ✅ Categorized all items appropriately
# - ✅ Assigned priorities based on severity
# - ✅ Provided actionable, structured output
#
# The agent is ready for integration into the full M&A analysis pipeline!

# %%
