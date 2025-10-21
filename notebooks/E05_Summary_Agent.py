# %% [markdown]
# # E05: Summary Agent - M&A Diligence Memo Generation
#
# This notebook demonstrates the Summary Agent which generates comprehensive
# M&A diligence memos from risk scoring results.
#
# **Pipeline Flow:**
# 1. Load and process M&A document
# 2. Extract clauses (Clause Extraction Agent)
# 3. Score risks (Risk Scoring Agent)
# 4. Generate diligence memo (Summary Agent) ← **This notebook**
#
# **Summary Agent Features:**
# - Executive summary with critical findings
# - Clause-by-clause analysis
# - Key findings extraction
# - Actionable recommendations
# - Overall assessment (Proceed/Caution/Do Not Proceed)
# - Markdown output for reports

# %%
import sys
from pathlib import Path

# Add backend to path
backend_path = Path("../backend").resolve()
sys.path.insert(0, str(backend_path))

# %%
from app.agents.clause_extraction import ClauseExtractionAgent
from app.agents.risk_scoring import RiskScoringAgent
from app.agents.summary import SummaryAgent
from app.rag.vector_store import VectorStore
from app.pipelines.ingestion_pipeline import IngestionPipeline

print("✅ All imports successful")

# %% [markdown]
# ## Step 1: Load and Process Document
#
# We'll use the Freedom Final Asset Agreement as our test document.

# %%
# Initialize components
vector_store = VectorStore()
ingestion_pipeline = IngestionPipeline(vector_store=vector_store)

# Process document
document_path = "../data/Freedom_Final_Asset_Agreement.pdf"
session_id = "summary_demo"

print(f"📄 Processing document: {document_path}")
result = ingestion_pipeline.ingest_document(
    file_path=document_path,
    session_id=session_id
)

print(f"✅ Document processed:")
print(f"   - Chunks created: {result.get('chunks_created', 'N/A')}")
print(f"   - Session ID: {session_id}")

# %% [markdown]
# ## Step 2: Extract Clauses
#
# Run the Clause Extraction Agent to identify M&A clauses.

# %%
# Initialize Clause Extraction Agent
clause_agent = ClauseExtractionAgent(vector_store=vector_store)

print("🔍 Extracting clauses...")
clause_result = clause_agent.extract_clauses(
    session_id=session_id,
    document_id="freedom_asset_agreement"
)

print(f"\n✅ Clause Extraction Complete:")
print(f"   - Clauses extracted: {len(clause_result.clauses)}")
print(f"   - Red flags detected: {len(clause_result.red_flags)}")
print(f"   - Processing time: {clause_result.metadata.get('processing_time_seconds', 0):.1f}s")

# Display extracted clauses
print("\n📋 Extracted Clauses:")
for i, clause in enumerate(clause_result.clauses, 1):
    print(f"\n{i}. {clause.clause_type.upper()}")
    print(f"   Text: {clause.clause_text[:100]}...")
    print(f"   Confidence: {clause.confidence:.2f}")

# %% [markdown]
# ## Step 3: Score Risks
#
# Run the Risk Scoring Agent to assess risk levels.

# %%
# Initialize Risk Scoring Agent
risk_agent = RiskScoringAgent()

print("⚠️  Scoring risks...")
risk_result = risk_agent.score_risks(
    clause_extraction_result=clause_result,
    document_id="freedom_asset_agreement"
)

print(f"\n✅ Risk Scoring Complete:")
print(f"   - Clauses scored: {len(risk_result.scored_clauses)}")
print(f"   - Overall risk: {risk_result.overall_risk_score}/100 ({risk_result.overall_risk_category})")
print(f"   - Processing time: {risk_result.metadata.get('processing_time_seconds', 0):.1f}s")

# Display risk scores
print("\n📊 Risk Scores by Clause:")
for scored_clause in risk_result.scored_clauses:
    clause = scored_clause.clause
    risk = scored_clause.risk_score
    print(f"\n• {clause.clause_type.upper()}: {risk.score}/100 ({risk.category})")
    print(f"  Justification: {risk.justification[:100]}...")

# %% [markdown]
# ## Step 4: Generate Summary (Diligence Memo)
#
# Run the Summary Agent to create a comprehensive M&A diligence memo.

# %%
# Initialize Summary Agent with retriever for additional context
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
summary_agent = SummaryAgent(retriever=retriever)

print("📝 Generating M&A diligence memo...")
memo = summary_agent.generate_summary(
    risk_scoring_result=risk_result,
    document_id="freedom_asset_agreement"
)

print(f"\n✅ Summary Generation Complete:")
print(f"   - Clause summaries: {len(memo.clause_summaries)}")
print(f"   - Key findings: {len(memo.key_findings)}")
print(f"   - Recommendations: {len(memo.recommendations)}")
print(f"   - Processing time: {memo.metadata.get('processing_time_seconds', 0):.1f}s")

# %% [markdown]
# ## Step 5: Display Executive Summary

# %%
print("=" * 80)
print("EXECUTIVE SUMMARY")
print("=" * 80)
print(f"\n{memo.executive_summary.overview}\n")

print(f"Overall Risk Assessment: {memo.executive_summary.overall_risk_assessment}\n")

print("Critical Findings:")
for i, finding in enumerate(memo.executive_summary.critical_findings, 1):
    print(f"  {i}. {finding}")

print("\nPrimary Recommendations:")
for i, rec in enumerate(memo.executive_summary.primary_recommendations, 1):
    print(f"  {i}. {rec}")

# %% [markdown]
# ## Step 6: Display Clause-by-Clause Analysis

# %%
print("\n" + "=" * 80)
print("CLAUSE-BY-CLAUSE ANALYSIS")
print("=" * 80)

for clause_summary in memo.clause_summaries:
    print(f"\n### {clause_summary.clause_type.replace('_', ' ').title()}")
    print(f"Risk Level: {clause_summary.risk_level}")
    print(f"\nSummary: {clause_summary.summary}")
    
    if clause_summary.key_points:
        print("\nKey Points:")
        for point in clause_summary.key_points:
            print(f"  • {point}")

# %% [markdown]
# ## Step 7: Display Key Findings

# %%
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

for finding in memo.key_findings:
    print(f"\n### {finding.finding}")
    print(f"Severity: {finding.severity}")
    if finding.clause_reference:
        print(f"Reference: {finding.clause_reference}")
    print(f"Impact: {finding.impact}")

# %% [markdown]
# ## Step 8: Display Recommendations

# %%
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

for rec in memo.recommendations:
    print(f"\n### {rec.recommendation}")
    print(f"Priority: {rec.priority}")
    print(f"Rationale: {rec.rationale}")
    if rec.related_findings:
        print(f"Related Findings: {', '.join(rec.related_findings)}")

# %% [markdown]
# ## Step 9: Display Overall Assessment

# %%
print("\n" + "=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)
print(f"\n{memo.overall_assessment}\n")

# %% [markdown]
# ## Step 10: Export as Markdown
#
# The DiligenceMemo has a built-in `to_markdown()` method for generating
# formatted reports.

# %%
# Generate Markdown
markdown_output = memo.to_markdown()

# Save to file
output_path = "../backend/diligence_memo_freedom.md"
with open(output_path, "w") as f:
    f.write(markdown_output)

print(f"✅ Markdown memo saved to: {output_path}")
print(f"   File size: {len(markdown_output)} characters")

# Display first 1000 characters
print("\n" + "=" * 80)
print("MARKDOWN PREVIEW (first 1000 chars)")
print("=" * 80)
print(markdown_output[:1000])
print("\n... (truncated)")

# %% [markdown]
# ## Step 11: Visualize Agent Graph
#
# Display the LangGraph structure of the Summary Agent.

# %%
try:
    from IPython.display import Image, display
    
    print("📊 Generating LangGraph visualization...")
    graph_image = summary_agent.get_graph_visualization()
    
    if graph_image:
        print("✅ Graph visualization generated")
        display(graph_image)
    else:
        print("ℹ️  Graph visualization not available (IPython not installed)")
        print("   To view the graph, run this notebook in Jupyter:")
        print("   jupyter notebook E05_Summary_Agent.py")
except ImportError:
    print("ℹ️  Graph visualization skipped (IPython not available in script mode)")
    print("   To view the graph, run this notebook in Jupyter:")
    print("   jupyter notebook E05_Summary_Agent.py")
except Exception as e:
    print(f"ℹ️  Graph visualization not available: {e}")

# %% [markdown]
# ## Summary Statistics

# %%
print("\n" + "=" * 80)
print("PIPELINE STATISTICS")
print("=" * 80)

print(f"\n📄 Document: Freedom Final Asset Agreement")
print(f"   - Chunks: {result.get('chunks_created', 'N/A')}")

print(f"\n🔍 Clause Extraction:")
print(f"   - Clauses: {len(clause_result.clauses)}")
print(f"   - Red Flags: {len(clause_result.red_flags)}")
print(f"   - Time: {clause_result.metadata.get('processing_time_seconds', 0):.1f}s")

print(f"\n⚠️  Risk Scoring:")
print(f"   - Scored Clauses: {len(risk_result.scored_clauses)}")
print(f"   - Overall Risk: {risk_result.overall_risk_score}/100 ({risk_result.overall_risk_category})")
print(f"   - Time: {risk_result.metadata.get('processing_time_seconds', 0):.1f}s")

print(f"\n📝 Summary Generation:")
print(f"   - Clause Summaries: {len(memo.clause_summaries)}")
print(f"   - Key Findings: {len(memo.key_findings)}")
print(f"   - Recommendations: {len(memo.recommendations)}")
print(f"   - Time: {memo.metadata.get('processing_time_seconds', 0):.1f}s")

total_time = (
    clause_result.metadata.get('processing_time_seconds', 0) +
    risk_result.metadata.get('processing_time_seconds', 0) +
    memo.metadata.get('processing_time_seconds', 0)
)
print(f"\n⏱️  Total Pipeline Time: {total_time:.1f}s")

# %% [markdown]
# ## Conclusion
#
# The Summary Agent successfully:
# - ✅ Generated executive summary with critical findings
# - ✅ Created clause-by-clause analysis
# - ✅ Extracted key findings with severity levels
# - ✅ Provided actionable recommendations
# - ✅ Delivered overall assessment with final recommendation
# - ✅ Exported formatted Markdown report
#
# **Next Steps:**
# - Review the generated diligence memo
# - Customize summary templates for specific use cases
# - Integrate with frontend for user-facing reports
# - Add provenance tracking (Story 3.4)
# - Implement checklist generation (Story 3.5)

# %%
print("\n✅ E05: Summary Agent demonstration complete!")
print(f"📄 Full memo available at: {output_path}")
