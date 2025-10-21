# %% [markdown]
# # E08: Document Orchestration Demonstration
# 
# This notebook demonstrates the DocumentOrchestrator running the complete end-to-end
# M&A document analysis workflow with the Freedom_Final_Asset_Agreement.pdf.
#
# **Orchestrator Capabilities:**
# - Coordinates 6 specialized agents (Ingestion, Clause Extraction, Risk Scoring, Summary, Provenance, Checklist)
# - Manages state throughout the multi-agent workflow
# - Routes tasks based on workflow progression
# - Handles errors gracefully with loop prevention
# - Provides comprehensive results from all agents

# %%
import sys
import os
from pathlib import Path
import json

# Add backend to path
backend_path = Path("../backend").resolve()
sys.path.insert(0, str(backend_path))

# %%
from app.orchestration.pipeline import DocumentOrchestrator

# %% [markdown]
# ## 1. Initialize Orchestrator
#
# The orchestrator creates its own vector store and initializes all agents.
# With lazy retriever initialization, agents can be created before data is loaded.

# %%
print("Initializing DocumentOrchestrator...")
orchestrator = DocumentOrchestrator(
    model_name="gpt-4o-mini",
    temperature=0.0
)

print(f"✓ DocumentOrchestrator initialized")
print(f"  - Model: {orchestrator.model_name}")
print(f"  - Temperature: {orchestrator.temperature}")
print(f"  - Max iterations: {orchestrator.MAX_ITERATIONS}")
print(f"  - Agents: 6 (Ingestion, Clause Extraction, Risk Scoring, Summary, Provenance, Checklist)")

# %% [markdown]
# ## 2. Prepare Document for Processing
#
# We'll use the Freedom_Final_Asset_Agreement.pdf from the data directory.

# %%
# Path to sample M&A document
data_dir = Path("../data").resolve()
document_path = data_dir / "Freedom_Final_Asset_Agreement.pdf"

if document_path.exists():
    print(f"✓ Document found: {document_path.name}")
    print(f"  - Size: {document_path.stat().st_size / 1024:.1f} KB")
    print(f"  - Path: {document_path}")
else:
    print(f"⚠ Document not found: {document_path}")
    print("  Please ensure the document exists in the data/ directory")
    raise FileNotFoundError(f"Document not found: {document_path}")

# %% [markdown]
# ## 3. Run End-to-End Orchestration
#
# This will process the document through all 6 agents in sequence.
# **Note:** This may take 2-4 minutes depending on document size and API response times.

# %%
print("\n" + "=" * 80)
print("STARTING ORCHESTRATION")
print("=" * 80)
print(f"\nDocument: {document_path.name}")
print("This will run all 6 agents in sequence...")
print("Expected time: 2-4 minutes")
print("\nWorkflow:")
print("  1. Ingestion → Process PDF and create chunks")
print("  2. Clause Extraction → Extract clauses and detect red flags")
print("  3. Risk Scoring → Assign risk scores to clauses")
print("  4. Summary → Generate diligence memo")
print("  5. Provenance → Track source information")
print("  6. Checklist → Generate follow-up questions")
print("\nStarting...\n")

# Run orchestration
results = orchestrator.run_orchestration(
    document_path=str(document_path),
    document_id="freedom_asset_agreement"
)

print("\n" + "=" * 80)
print("ORCHESTRATION COMPLETE")
print("=" * 80)
print(f"\nStatus: {results['status']}")
print(f"Document ID: {results['document_id']}")
print(f"Completed Steps: {len(results['completed_steps'])}/6")

if results.get('errors'):
    print(f"\n⚠ Errors encountered: {len(results['errors'])}")
    for error in results['errors']:
        print(f"  - {error}")

# %% [markdown]
# ## 4. Workflow Execution Details

# %%
print("\n=== WORKFLOW EXECUTION DETAILS ===\n")

print(f"Started: {results.get('started_at', 'N/A')}")
print(f"Completed: {results.get('completed_at', 'N/A')}")
print()

print("Completed Steps:")
for i, step in enumerate(results['completed_steps'], 1):
    print(f"  {i}. {step}")
print()

print("Agent Processing Times:")
metadata = results['metadata']
for agent_name in ['ingestion', 'clause_extraction', 'risk_scoring', 'summary', 'provenance', 'checklist']:
    if agent_name in metadata:
        agent_meta = metadata[agent_name]
        time = agent_meta.get('processing_time', 0)
        print(f"  - {agent_name.replace('_', ' ').title()}: {time:.2f}s")

# %% [markdown]
# ## 5. Ingestion Results

# %%
ingestion_meta = metadata.get('ingestion', {})

print("\n=== INGESTION RESULTS ===\n")
print(f"Total Chunks Created: {ingestion_meta.get('total_chunks', 0)}")
print(f"Processing Time: {ingestion_meta.get('processing_time', 0):.2f}s")

# %% [markdown]
# ## 6. Clause Extraction Results

# %%
extracted_clauses = results.get('extracted_clauses', [])
clause_meta = metadata.get('clause_extraction', {})

print("\n=== CLAUSE EXTRACTION RESULTS ===\n")
print(f"Total Clauses Extracted: {clause_meta.get('total_clauses', 0)}")
print(f"Total Red Flags Detected: {clause_meta.get('total_red_flags', 0)}")
print(f"Processing Time: {clause_meta.get('processing_time', 0):.2f}s")
print()

if extracted_clauses:
    print("Extracted Clauses:")
    for i, clause in enumerate(extracted_clauses, 1):
        print(f"\n{i}. {clause.clause_type.replace('_', ' ').title()}")
        print(f"   Text: {clause.clause_text[:150]}...")
        print(f"   Confidence: {clause.confidence:.2f}")
        if clause.location:
            print(f"   Location: {clause.location}")

# %% [markdown]
# ## 7. Risk Scoring Results

# %%
scored_clauses = results.get('scored_clauses', [])
risk_meta = metadata.get('risk_scoring', {})

print("\n=== RISK SCORING RESULTS ===\n")
print(f"Total Clauses Scored: {risk_meta.get('total_scored_clauses', 0)}")
print(f"Overall Risk Score: {risk_meta.get('overall_risk_score', 0)}/100")
print(f"Overall Risk Category: {risk_meta.get('overall_risk_category', 'N/A')}")
print(f"Processing Time: {risk_meta.get('processing_time', 0):.2f}s")
print()

if scored_clauses:
    print("Risk Distribution:")
    risk_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
    for sc in scored_clauses:
        risk_counts[sc.risk_score.category] += 1
    
    for category in ['Critical', 'High', 'Medium', 'Low']:
        count = risk_counts[category]
        if count > 0:
            print(f"  - {category}: {count} clauses")
    
    print("\nScored Clauses:")
    sorted_clauses = sorted(scored_clauses, key=lambda x: x.risk_score.score, reverse=True)
    for i, sc in enumerate(sorted_clauses, 1):
        print(f"\n{i}. {sc.clause.clause_type.replace('_', ' ').title()}")
        print(f"   Risk Score: {sc.risk_score.score}/100 ({sc.risk_score.category})")
        print(f"   Justification: {sc.risk_score.justification[:120]}...")

# %% [markdown]
# ## 8. Summary (Diligence Memo)

# %%
summary = results.get('summary')
summary_meta = metadata.get('summary', {})

print("\n=== DILIGENCE MEMO SUMMARY ===\n")
print(f"Total Findings: {summary_meta.get('total_findings', 0)}")
print(f"Total Recommendations: {summary_meta.get('total_recommendations', 0)}")
print(f"Processing Time: {summary_meta.get('processing_time', 0):.2f}s")

if summary:
    print("\n--- Executive Summary ---")
    print(f"\nOverview: {summary.executive_summary.overview}")
    print(f"\nOverall Risk: {summary.executive_summary.overall_risk_assessment}")
    
    print("\nCritical Findings:")
    for i, finding in enumerate(summary.executive_summary.critical_findings, 1):
        print(f"  {i}. {finding}")
    
    print("\nPrimary Recommendations:")
    for i, rec in enumerate(summary.executive_summary.primary_recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n--- Overall Assessment ---")
    print(summary.overall_assessment)

# %% [markdown]
# ## 9. Provenance Results

# %%
provenance_data = results.get('provenance_data', {})
provenance_meta = metadata.get('provenance', {})

print("\n=== PROVENANCE TRACKING RESULTS ===\n")
print(f"Total Items Tracked: {provenance_meta.get('total_items_tracked', 0)}")
print()

if provenance_data:
    print("Source Tracking (first 3 items):")
    for i, (clause_id, prov_info) in enumerate(list(provenance_data.items())[:3], 1):
        print(f"\n{i}. Clause: {prov_info.get('clause_type', 'N/A').replace('_', ' ').title()}")
        sources = prov_info.get('sources', [])
        print(f"   Sources: {len(sources)} references")
        if sources:
            source = sources[0]
            print(f"   - Document: {source.get('document_id', 'N/A')}")
            print(f"   - Chunk: {source.get('chunk_id', 'N/A')}")
            print(f"   - Confidence: {source.get('confidence', 0):.2f}")
            print(f"   - Snippet: {source.get('text_snippet', 'N/A')[:80]}...")

# %% [markdown]
# ## 10. Checklist Results

# %%
checklist = results.get('checklist', [])
checklist_meta = metadata.get('checklist', {})

print("\n=== CHECKLIST RESULTS ===\n")
print(f"Total Questions Generated: {checklist_meta.get('total_questions', 0)}")
print(f"Processing Time: {checklist_meta.get('processing_time', 0):.2f}s")
print()

if checklist:
    # Group by priority
    by_priority = {'Critical': [], 'High': [], 'Medium': [], 'Low': []}
    for item in checklist:
        priority = item.get('priority', 'Medium')
        by_priority[priority].append(item)
    
    print("Questions by Priority:")
    for priority in ['Critical', 'High', 'Medium', 'Low']:
        count = len(by_priority[priority])
        if count > 0:
            print(f"\n{priority} Priority: {count} questions")
            for i, item in enumerate(by_priority[priority][:3], 1):
                question = item.get('question', item.get('item', 'N/A'))
                category = item.get('category', 'N/A')
                print(f"  {i}. [{category}] {question[:80]}...")

# %% [markdown]
# ## 11. Performance Statistics

# %%
print("\n" + "=" * 80)
print("ORCHESTRATION PERFORMANCE SUMMARY")
print("=" * 80)

# Calculate total processing time
total_time = sum(metadata[agent].get('processing_time', 0) 
                 for agent in ['ingestion', 'clause_extraction', 'risk_scoring', 
                              'summary', 'provenance', 'checklist']
                 if agent in metadata)

print(f"\nTotal Processing Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Number of Agents: 6")
if total_time > 0:
    print(f"Average Time per Agent: {total_time/6:.2f}s")
print()

print("Workflow Efficiency:")
print(f"  - Steps Completed: {len(results['completed_steps'])}/6")
if len(results['completed_steps']) > 0:
    print(f"  - Success Rate: {(len(results['completed_steps'])/6)*100:.1f}%")
print(f"  - Errors: {len(results.get('errors', []))}")
print()

print("Output Summary:")
print(f"  - Document Chunks: {metadata.get('ingestion', {}).get('total_chunks', 0)}")
print(f"  - Clauses Extracted: {metadata.get('clause_extraction', {}).get('total_clauses', 0)}")
print(f"  - Red Flags: {metadata.get('clause_extraction', {}).get('total_red_flags', 0)}")
print(f"  - Risk Score: {metadata.get('risk_scoring', {}).get('overall_risk_score', 0)}/100 ({metadata.get('risk_scoring', {}).get('overall_risk_category', 'N/A')})")
print(f"  - Findings: {metadata.get('summary', {}).get('total_findings', 0)}")
print(f"  - Recommendations: {metadata.get('summary', {}).get('total_recommendations', 0)}")
print(f"  - Checklist Items: {len(checklist)}")
print(f"  - Provenance Items: {len(provenance_data)}")

# %% [markdown]
# ## 12. Export Complete Results

# %%
# Save results to JSON
output_file = backend_path / "orchestration_results.json"

export_data = {
    "document_id": results['document_id'],
    "status": results['status'],
    "started_at": results.get('started_at'),
    "completed_at": results.get('completed_at'),
    "completed_steps": results.get('completed_steps', []),
    "errors": results.get('errors', []),
    "metadata": metadata,
    "summary": {
        "total_clauses": len(results.get('extracted_clauses', [])),
        "total_scored_clauses": len(results.get('scored_clauses', [])),
        "overall_risk_score": metadata.get('risk_scoring', {}).get('overall_risk_score', 0),
        "overall_risk_category": metadata.get('risk_scoring', {}).get('overall_risk_category', 'N/A'),
        "total_findings": metadata.get('summary', {}).get('total_findings', 0),
        "total_recommendations": metadata.get('summary', {}).get('total_recommendations', 0),
        "total_checklist_items": len(checklist),
        "total_provenance_items": len(provenance_data)
    }
}

with open(output_file, "w") as f:
    json.dump(export_data, f, indent=2)

print(f"\n✓ Results exported to: {output_file}")

# %% [markdown]
# ## Conclusion
#
# The DocumentOrchestrator successfully:
# - ✅ Coordinated all 6 specialized agents in sequence
# - ✅ Maintained state throughout the multi-agent workflow
# - ✅ Routed tasks based on workflow progression
# - ✅ Handled errors gracefully with loop prevention
# - ✅ Provided comprehensive results from all agents
# - ✅ Tracked processing times and metadata for each step
# - ✅ Generated complete end-to-end M&A document analysis
#
# The orchestrator is ready for production use in the Legal-OS system!

# %%
