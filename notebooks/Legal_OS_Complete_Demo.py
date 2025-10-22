# %% [markdown]
# # Legal-OS: Complete M&A Document Analysis Pipeline
#
# This comprehensive notebook demonstrates the entire Legal-OS system for M&A document analysis.
# It combines all validation notebooks (E01-E08) into a single, sequential workflow.
#
# ## System Overview
#
# Legal-OS is an AI-powered M&A due diligence system that:
# - Ingests and processes legal documents
# - Extracts and analyzes M&A clauses
# - Scores risks and identifies red flags
# - Generates comprehensive diligence memos
# - Tracks source provenance
# - Creates actionable checklists
#
# ## Notebook Structure
#
# 1. **Setup & Configuration** - Environment and imports
# 2. **Pipeline Foundation** - Document ingestion and RAG
# 3. **Specialized Agents** - Clause extraction, risk scoring, summary, checklist
# 4. **End-to-End Orchestration** - Complete workflow demonstration
# 5. **Results & Export** - Analysis and export
#
# ## Prerequisites
#
# - Environment variables configured (OPENAI_API_KEY)
# - Sample M&A documents in `data/` directory
# - Backend dependencies installed (`cd backend && uv sync`)
#
# ## Note
#
# This notebook uses **in-memory Qdrant** - no Docker required!

# %% [markdown]
# ---
# # Part 1: Setup & Configuration
# ---

# %% Setup and Imports
import os
import sys
from pathlib import Path
import pandas as pd
import json
import time
from dotenv import load_dotenv
from uuid import uuid4
from collections import defaultdict

# Try to import IPython for visualization
try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("⚠️  IPython not available - visualization will be skipped")

# Add backend to Python path
# Handle both script and notebook execution
try:
    backend_path = Path(__file__).parent.parent / "backend"
except NameError:
    # __file__ not defined in Jupyter notebooks
    backend_path = Path.cwd() / "backend"
    if not backend_path.exists():
        backend_path = Path.cwd().parent / "backend"

if backend_path.exists():
    sys.path.insert(0, str(backend_path))
else:
    print(f"⚠️  Warning: Backend path not found. Tried: {backend_path}")

# Load environment variables
env_path = backend_path / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Verify required environment variables
required_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

print("✅ Environment setup complete")
print(f"   OPENAI_API_KEY: {'*' * 20}{os.getenv('OPENAI_API_KEY', '')[-4:]}")
print(f"   Using in-memory Qdrant (no Docker required)")

# %% [markdown]
# ---
# # Part 2: Pipeline Foundation
# ---

# %% [markdown]
# ## Document Ingestion

# %% Document Ingestion
from app.pipelines.ingestion_pipeline import IngestionPipeline

# Select document
# Handle both script and notebook execution
try:
    data_dir = Path(__file__).parent.parent / "data"
except NameError:
    # __file__ not defined in Jupyter notebooks
    data_dir = Path.cwd() / "data"
    if not data_dir.exists():
        data_dir = Path.cwd().parent / "data"

freedom_doc = data_dir / "Freedom_Final_Asset_Agreement.pdf"

if not freedom_doc.exists():
    sample_docs = list(data_dir.glob("*.pdf"))
    if not sample_docs:
        raise FileNotFoundError(f"No PDF documents found in {data_dir}")
    sample_doc = sample_docs[0]
    print(f"⚠️  Freedom document not found, using {sample_doc.name}")
else:
    sample_doc = freedom_doc

print(f"📄 Ingesting: {sample_doc.name}")

# Create pipeline and ingest
pipeline = IngestionPipeline(use_memory=True)
session_id = str(uuid4())
result = pipeline.ingest_document(file_path=str(sample_doc), session_id=session_id)

vector_store = pipeline.vector_store
document_id = result["document_id"]

print(f"✅ Ingestion Complete!")
print(f"   Document ID: {document_id}")
print(f"   Chunks: {result['chunk_count']}")

# %% [markdown]
# ## RAG Pipeline Test

# %% RAG Test
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

test_question = "What are the key parties involved in this agreement?"
print(f"❓ Question: {test_question}\n")

retriever = vector_store.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

start_time = time.time()
response = rag_chain.invoke({"input": test_question})
elapsed_time = time.time() - start_time

print(f"✅ RAG Query Complete! ({elapsed_time:.2f}s)")
print(f"💡 Answer: {response.get('answer', '')}")

# %% [markdown]
# ---
# # Part 3: Specialized Agents
# ---

# %% [markdown]
# ## Clause Extraction Agent

# %% Clause Extraction
from app.agents.clause_extraction import ClauseExtractionAgent

print("🤖 Clause Extraction Agent")

clause_agent = ClauseExtractionAgent(
    vector_store=vector_store,
    model_name="gpt-4o-mini",
    temperature=0.0,
    top_k=10
)

print(f"🔍 Extracting clauses...")
start_time = time.time()
clause_result = clause_agent.extract_clauses(document_id=document_id)
elapsed_time = time.time() - start_time

print(f"✅ Complete! ({elapsed_time:.2f}s)")
print(f"   Clauses: {len(clause_result.clauses)}")
print(f"   Red Flags: {len(clause_result.red_flags)}")

if clause_result.clauses:
    print("\n📝 Sample Clauses:")
    for i, clause in enumerate(clause_result.clauses[:3], 1):
        print(f"   {i}. {clause.clause_type}: {clause.clause_text[:100]}...")

# %% [markdown]
# ## Risk Scoring Agent

# %% Risk Scoring
from app.agents.risk_scoring import RiskScoringAgent

print("🎯 Risk Scoring Agent")

risk_agent = RiskScoringAgent()

print(f"🎯 Scoring risks...")
start_time = time.time()
risk_result = risk_agent.score_risks(
    clause_extraction_result=clause_result,
    document_id=document_id
)
elapsed_time = time.time() - start_time

print(f"✅ Complete! ({elapsed_time:.2f}s)")
print(f"   Overall Risk: {risk_result.overall_risk_score}/100 ({risk_result.overall_risk_category})")

# Risk distribution
risk_dist = {}
for sc in risk_result.scored_clauses:
    risk_dist[sc.risk_score.category] = risk_dist.get(sc.risk_score.category, 0) + 1

print("\n📊 Risk Distribution:")
for cat in ["Critical", "High", "Medium", "Low"]:
    if cat in risk_dist:
        print(f"   {cat}: {risk_dist[cat]}")

# %% [markdown]
# ## Summary Agent

# %% Summary
from app.agents.summary import SummaryAgent

print("📝 Summary Agent")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
summary_agent = SummaryAgent(retriever=retriever)

print(f"📝 Generating diligence memo...")
start_time = time.time()
memo = summary_agent.generate_summary(
    risk_scoring_result=risk_result,
    document_id=document_id
)
elapsed_time = time.time() - start_time

print(f"✅ Complete! ({elapsed_time:.2f}s)")
print(f"   Findings: {len(memo.key_findings)}")
print(f"   Recommendations: {len(memo.recommendations)}")

print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY")
print("=" * 80)
print(f"\n{memo.executive_summary.overview}\n")
print(f"Risk: {memo.executive_summary.overall_risk_assessment}\n")

print("Critical Findings:")
for i, finding in enumerate(memo.executive_summary.critical_findings, 1):
    print(f"  {i}. {finding}")

print("\nRecommendations:")
for i, rec in enumerate(memo.executive_summary.primary_recommendations, 1):
    print(f"  {i}. {rec}")

# Export memo
memo_md = memo.to_markdown()
output_path = backend_path / "diligence_memo.md"
with open(output_path, "w") as f:
    f.write(memo_md)
print(f"\n💾 Memo saved to: {output_path.name}")

# %% [markdown]
# ## Source Tracker

# %% Source Tracking
from app.utils.source_tracker import SourceTracker

print("🔍 Source Tracker")

tracker = SourceTracker(document_id=document_id)

# Get sample chunks for source tracking
sample_chunks = vector_store.as_retriever(search_kwargs={"k": 3}).invoke("payment terms")

# Create source references
source_refs = []
for i, chunk in enumerate(sample_chunks[:3], 1):
    page = chunk.metadata.get("page") if hasattr(chunk, 'metadata') else None
    source_ref = tracker.create_source_reference(
        chunk_id=f"chunk_{i}",
        text_snippet=chunk.page_content[:150],
        page=page,
        confidence=0.95
    )
    source_refs.append(source_ref)

print(f"✅ Created {len(source_refs)} source references")

# Create metadata and links
source_metadata = tracker.create_source_metadata(
    sources=source_refs,
    extraction_method="llm_extraction"
)

links = tracker.generate_links(source_metadata)
print(f"🔗 Generated {len(links)} frontend links")

# Export
export_data = {
    "document_id": document_id,
    "source_references": len(source_refs),
    "frontend_links": len(links)
}

results_file = backend_path / "source_tracker_results.json"
with open(results_file, 'w') as f:
    json.dump(export_data, f, indent=2)
print(f"💾 Results saved to: {results_file.name}")

# %% [markdown]
# ## Checklist Agent

# %% Checklist
from app.agents.checklist import ChecklistAgent

print("📋 Checklist Agent")

checklist_agent = ChecklistAgent(model_name="gpt-4o-mini", temperature=0.0)

print(f"📋 Generating checklist...")
start_time = time.time()
checklist_result = checklist_agent.generate_checklist(memo)
elapsed_time = time.time() - start_time

print(f"✅ Complete! ({elapsed_time:.2f}s)")
print(f"   Checklist items: {len(checklist_result.checklist_items)}")
print(f"   Follow-up questions: {len(checklist_result.follow_up_questions)}")

# Group by priority
items_by_priority = defaultdict(list)
for item in checklist_result.checklist_items:
    items_by_priority[item.priority].append(item)

print("\n📊 Items by Priority:")
for priority in ["Critical", "High", "Medium", "Low"]:
    if priority in items_by_priority:
        print(f"   {priority}: {len(items_by_priority[priority])}")

# Export
checklist_dict = {
    "document_id": checklist_result.document_id,
    "total_items": len(checklist_result.checklist_items),
    "total_questions": len(checklist_result.follow_up_questions)
}

output_file = backend_path / "checklist_results.json"
with open(output_file, "w") as f:
    json.dump(checklist_dict, f, indent=2)
print(f"💾 Checklist saved to: {output_file.name}")

# %% [markdown]
# ---
# # Part 4: End-to-End Orchestration
# ---

# %% Orchestration
from app.orchestration.pipeline import DocumentOrchestrator

print("\n" + "=" * 80)
print("END-TO-END ORCHESTRATION")
print("=" * 80)

orchestrator = DocumentOrchestrator(model_name="gpt-4o-mini", temperature=0.0)

print(f"\n🚀 Running complete pipeline...")
print(f"   Document: {sample_doc.name}")
print(f"   Expected time: 2-4 minutes\n")

start_time = time.time()
orch_results = orchestrator.run_orchestration(
    document_path=str(sample_doc),
    document_id="freedom_asset_agreement"
)
total_time = time.time() - start_time

print("\n" + "=" * 80)
print("ORCHESTRATION COMPLETE")
print("=" * 80)
print(f"\nStatus: {orch_results['status']}")
print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Completed Steps: {len(orch_results['completed_steps'])}/6")

if orch_results.get('errors'):
    print(f"\n⚠️  Errors: {len(orch_results['errors'])}")

# %% [markdown]
# ## Orchestration Results

# %% Results Analysis
metadata = orch_results['metadata']

print("\n📊 PIPELINE STATISTICS")
print("=" * 80)

print(f"\nIngestion:")
print(f"  Chunks: {metadata.get('ingestion', {}).get('total_chunks', 0)}")

print(f"\nClause Extraction:")
print(f"  Clauses: {metadata.get('clause_extraction', {}).get('total_clauses', 0)}")
print(f"  Red Flags: {metadata.get('clause_extraction', {}).get('total_red_flags', 0)}")

print(f"\nRisk Scoring:")
print(f"  Overall Risk: {metadata.get('risk_scoring', {}).get('overall_risk_score', 0)}/100")
print(f"  Category: {metadata.get('risk_scoring', {}).get('overall_risk_category', 'N/A')}")

print(f"\nSummary:")
print(f"  Findings: {metadata.get('summary', {}).get('total_findings', 0)}")
print(f"  Recommendations: {metadata.get('summary', {}).get('total_recommendations', 0)}")

print(f"\nChecklist:")
print
