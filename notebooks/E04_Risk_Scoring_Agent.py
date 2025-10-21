# %% [markdown]
# # E04: Risk Scoring Agent Demo
# 
# This notebook demonstrates the Risk Scoring Agent which assigns risk scores
# to extracted clauses based on defined rules and heuristics.
#
# **Prerequisites:**
# - Story 3.1 (Clause Extraction Agent) completed
# - Story 3.2 (Risk Scoring Agent) completed
# - Vector store populated with documents
# - OpenAI API key configured

# %% [markdown]
# ## Setup and Imports

# %%
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from uuid import uuid4

# Add backend to path
notebook_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
backend_path = notebook_dir.parent / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables
env_path = backend_path / ".env"
if env_path.exists():
    load_dotenv(env_path)

from app.agents.clause_extraction import ClauseExtractionAgent
from app.agents.risk_scoring import RiskScoringAgent
from app.pipelines.ingestion_pipeline import IngestionPipeline

print("✅ Imports successful")

# %% [markdown]
# ## Step 1: Ingest Document
#
# First, we need to ingest a document into the vector store.

# %%
# Select document
data_dir = notebook_dir.parent / "data"
freedom_doc = data_dir / "Freedom_Final_Asset_Agreement.pdf"

if not freedom_doc.exists():
    sample_docs = list(data_dir.glob("*.pdf"))
    if not sample_docs:
        raise FileNotFoundError(f"No PDF documents found in {data_dir}")
    sample_doc = sample_docs[0]
    print(f"⚠️  Freedom document not found, using {sample_doc.name} instead")
else:
    sample_doc = freedom_doc

print(f"📄 Ingesting document: {sample_doc.name}")

# Create ingestion pipeline with in-memory vector store
pipeline = IngestionPipeline(use_memory=True)
session_id = str(uuid4())

# Ingest document
result = pipeline.ingest_document(
    file_path=str(sample_doc),
    session_id=session_id
)

vector_store = pipeline.vector_store
document_id = result["document_id"]

print(f"✅ Ingestion complete!")
print(f"   Document ID: {document_id}")
print(f"   Chunks: {result['chunk_count']}")

# %% [markdown]
# ## Step 2: Initialize Agents
#
# Initialize both the Clause Extraction Agent and Risk Scoring Agent.

# %%
# Initialize Clause Extraction Agent
clause_agent = ClauseExtractionAgent(vector_store=vector_store)
print(f"✅ Clause Extraction Agent initialized")

# Initialize Risk Scoring Agent
risk_agent = RiskScoringAgent()
print(f"✅ Risk Scoring Agent initialized")

# %% [markdown]
# ## Step 3: Extract Clauses from Document
#
# Use the Clause Extraction Agent to extract clauses from the ingested document.

# %%
# Extract clauses from the document
print(f"📄 Extracting clauses from: {document_id}")
print("-" * 60)

clause_result = clause_agent.extract_clauses(document_id=document_id)

print(f"\n✅ Extraction complete!")
print(f"   - Clauses extracted: {len(clause_result.clauses)}")
print(f"   - Red flags detected: {len(clause_result.red_flags)}")
print(f"   - Processing time: {clause_result.metadata.get('processing_time_seconds', 0):.2f}s")

# %% [markdown]
# ## Step 4: Display Extracted Clauses

# %%
print("\n📋 Extracted Clauses:")
print("=" * 60)

for i, clause in enumerate(clause_result.clauses, 1):
    print(f"\n{i}. {clause.clause_type.upper()}")
    print(f"   Location: {clause.location}")
    print(f"   Confidence: {clause.confidence:.2%}")
    print(f"   Text: {clause.clause_text[:200]}...")

# %% [markdown]
# ## Step 5: Score Risks for Extracted Clauses
#
# Now we'll use the Risk Scoring Agent to assign risk scores to each clause.

# %%
print("\n🎯 Scoring risks for extracted clauses...")
print("-" * 60)

risk_result = risk_agent.score_risks(
    clause_extraction_result=clause_result,
    document_id=document_id
)

print(f"\n✅ Risk scoring complete!")
print(f"   - Clauses scored: {len(risk_result.scored_clauses)}")
print(f"   - Overall risk score: {risk_result.overall_risk_score}/100")
print(f"   - Overall risk category: {risk_result.overall_risk_category}")
print(f"   - Processing time: {risk_result.metadata.get('processing_time_seconds', 0):.2f}s")

# %% [markdown]
# ## Step 6: Display Risk Scores

# %%
print("\n📊 Risk Scoring Results:")
print("=" * 60)

for i, scored_clause in enumerate(risk_result.scored_clauses, 1):
    clause = scored_clause.clause
    risk = scored_clause.risk_score
    
    # Color code based on risk category
    category_emoji = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🟠",
        "Critical": "🔴"
    }
    
    print(f"\n{i}. {clause.clause_type.upper()}")
    print(f"   {category_emoji.get(risk.category, '⚪')} Risk Score: {risk.score}/100 ({risk.category})")
    print(f"   Clause: {clause.clause_text[:150]}...")
    print(f"   Justification: {risk.justification}")
    
    if risk.factors:
        print(f"   Risk Factors Detected:")
        for factor in risk.factors:
            if factor.detected:
                print(f"      - {factor.factor_name}: {factor.description}")

# %% [markdown]
# ## Step 7: Overall Risk Assessment

# %%
print("\n📈 Overall Document Risk Assessment:")
print("=" * 60)

category_emoji = {
    "Low": "🟢",
    "Medium": "🟡",
    "High": "🟠",
    "Critical": "🔴"
}

print(f"\n{category_emoji.get(risk_result.overall_risk_category, '⚪')} Overall Risk: {risk_result.overall_risk_score}/100")
print(f"   Category: {risk_result.overall_risk_category}")
print(f"   Document: {risk_result.document_id}")

# Risk distribution
risk_distribution = {}
for scored_clause in risk_result.scored_clauses:
    category = scored_clause.risk_score.category
    risk_distribution[category] = risk_distribution.get(category, 0) + 1

print(f"\n   Risk Distribution:")
for category in ["Low", "Medium", "High", "Critical"]:
    count = risk_distribution.get(category, 0)
    if count > 0:
        emoji = category_emoji.get(category, '⚪')
        print(f"      {emoji} {category}: {count} clause(s)")

# %% [markdown]
# ## Step 8: Visualize Agent Graph (Optional)
#
# Display the LangGraph visualization of the Risk Scoring Agent's workflow.

# %%
try:
    print("\n🔍 Generating agent graph visualization...")
    graph_viz = risk_agent.get_graph_visualization()
    
    if graph_viz:
        display(graph_viz)
        print("✅ Graph visualization displayed above")
    else:
        print("⚠️  Graph visualization not available (IPython may not be installed)")
except Exception as e:
    print(f"⚠️  Could not generate visualization: {e}")

# %% [markdown]
# ## Step 9: Detailed Risk Analysis
#
# Let's analyze the highest risk clauses in detail.

# %%
print("\n⚠️  Highest Risk Clauses:")
print("=" * 60)

# Sort by risk score
sorted_clauses = sorted(
    risk_result.scored_clauses,
    key=lambda x: x.risk_score.score,
    reverse=True
)

# Show top 3 highest risk clauses
for i, scored_clause in enumerate(sorted_clauses[:3], 1):
    clause = scored_clause.clause
    risk = scored_clause.risk_score
    
    category_emoji = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🟠",
        "Critical": "🔴"
    }
    
    print(f"\n{i}. {category_emoji.get(risk.category, '⚪')} {clause.clause_type.upper()}")
    print(f"   Risk Score: {risk.score}/100 ({risk.category})")
    print(f"   Location: {clause.location}")
    print(f"   Clause Text:")
    print(f"   {clause.clause_text}")
    print(f"\n   Risk Assessment:")
    print(f"   {risk.justification}")
    
    if risk.factors:
        print(f"\n   Detected Risk Factors:")
        for factor in risk.factors:
            if factor.detected:
                print(f"      • {factor.factor_name}")
                print(f"        {factor.description}")

# %% [markdown]
# ## Step 10: Risk Scoring Rules Reference
#
# Display the risk scoring rules used by the agent.

# %%
print("\n📚 Risk Scoring Rules Reference:")
print("=" * 60)

print("\nRisk Categories:")
print("  🟢 Low (0-25): Standard terms, minimal concerns")
print("  🟡 Medium (26-50): Some concerns, review recommended")
print("  🟠 High (51-75): Significant concerns, negotiation needed")
print("  🔴 Critical (76-100): Major concerns, immediate attention required")

print("\n\nRisk Factors by Clause Type:")
for clause_type, rules in risk_agent.RISK_RULES.items():
    print(f"\n{clause_type.upper().replace('_', ' ')}:")
    for factor_name, factor_info in rules.items():
        print(f"  • {factor_name.replace('_', ' ').title()}: +{factor_info['impact']} points")
        print(f"    {factor_info['description']}")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
# 1. ✅ Extracting clauses from a document using the Clause Extraction Agent
# 2. ✅ Scoring risks for extracted clauses using the Risk Scoring Agent
# 3. ✅ Analyzing risk scores and categories
# 4. ✅ Identifying highest risk clauses
# 5. ✅ Understanding the risk scoring methodology
#
# The Risk Scoring Agent successfully:
# - Applied defined risk scoring rules and heuristics
# - Assigned numerical scores (0-100) to each clause
# - Categorized risks as Low, Medium, High, or Critical
# - Provided justifications for risk assessments
# - Calculated an overall document risk score
#
# **Next Steps:**
# - Story 3.3: Summary Agent (generate executive summaries)
# - Story 3.4: Provenance Agent (track information sources)
# - Story 3.5: Checklist Agent (generate compliance checklists)

# %%
print("\n" + "=" * 60)
print("✅ Risk Scoring Agent Demo Complete!")
print("=" * 60)
