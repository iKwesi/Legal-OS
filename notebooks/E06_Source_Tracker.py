# %% [markdown]
# # E06: Source Tracker - Epic 3 Validation
#
# ## Purpose
# This notebook validates the Source Tracker (formerly Provenance Tracker) implementation from Story 3.4:
# - **Architecture**: Utility class for tracking source information
# - **Functionality**: Source reference creation, metadata management, link generation
# - **Integration**: Works with all agents to maintain source attribution
#
# ## What This Notebook Demonstrates
# 1. **SourceTracker Initialization**: Creating the utility class
# 2. **Source Reference Creation**: Tracking document chunks with metadata
# 3. **Source Metadata Management**: Aggregating sources with confidence scores
# 4. **Link Generation**: Creating frontend-renderable links
# 5. **Integration**: Embedding source info in extracted clauses
# 6. **Provenance Chain**: Tracking and querying source relationships
#
# ## Prerequisites
# - Environment variables configured (OPENAI_API_KEY)
# - Sample M&A documents in `data/` directory
# - Backend dependencies installed (`cd backend && uv sync`)
#
# ## Setup Instructions
# 1. Set environment variables in `backend/.env` (or export OPENAI_API_KEY)
# 2. Install dependencies: `cd backend && uv sync`
# 3. Run this notebook from the project root or backend directory
#
# ## Note
# This notebook uses **in-memory Qdrant** - no Docker required!
# Data is ephemeral and will be lost when the notebook restarts.

# %% Setup and Imports
import os
import sys
from pathlib import Path
import json
from dotenv import load_dotenv

# Add backend to Python path if running from notebooks directory
backend_path = Path(__file__).parent.parent / "backend"
if backend_path.exists():
    sys.path.insert(0, str(backend_path))

# Load environment variables
env_path = backend_path / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print("⚠️  Warning: .env file not found. Using system environment variables.")

# Verify required environment variables
required_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

print("✅ Environment setup complete")
print(f"   OPENAI_API_KEY: {'*' * 20}{os.getenv('OPENAI_API_KEY', '')[-4:]}")
print(f"   Using in-memory Qdrant (no Docker required)")

# %% [markdown]
# ## Cell 2: Document Ingestion
#
# Use the existing IngestionPipeline from Story 1.2 to load and process
# the Freedom M&A document. This provides the chunks we'll track sources for.

# %% Document Ingestion
from app.pipelines.ingestion_pipeline import IngestionPipeline
from uuid import uuid4

# Select the Freedom document
data_dir = Path(__file__).parent.parent / "data"
freedom_doc = data_dir / "Freedom_Final_Asset_Agreement.pdf"

if not freedom_doc.exists():
    # Fallback to any available document
    sample_docs = list(data_dir.glob("*.pdf"))
    if not sample_docs:
        raise FileNotFoundError(f"No PDF documents found in {data_dir}")
    sample_doc = sample_docs[0]
    print(f"⚠️  Freedom document not found, using {sample_doc.name} instead")
else:
    sample_doc = freedom_doc

print(f"📄 Ingesting M&A document: {sample_doc.name}")
print(f"   File size: {sample_doc.stat().st_size / 1024:.2f} KB")
print(f"   Using IngestionPipeline from Story 1.2\n")

try:
    # Create ingestion pipeline with in-memory vector store
    pipeline = IngestionPipeline(use_memory=True)
    
    # Generate session ID for this ingestion
    session_id = str(uuid4())
    
    # Ingest document
    result = pipeline.ingest_document(
        file_path=str(sample_doc),
        session_id=session_id
    )
    
    # Get the vector store
    vector_store = pipeline.vector_store
    
    # Store document_id for later use
    document_id = result["document_id"]
    
    print("✅ Ingestion Complete!")
    print(f"   Document ID: {document_id}")
    print(f"   File Name: {result['file_name']}")
    print(f"   Total chunks: {result['chunk_count']}")
    print(f"   Status: {result['status']}")
    
    # Get some sample chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    sample_chunks = retriever.invoke("purchase price payment terms")
    
    print("\n📝 Sample Chunks (first 3):")
    for i, chunk in enumerate(sample_chunks[:3], 1):
        preview = chunk.page_content[:150] + "..." if len(chunk.page_content) > 150 else chunk.page_content
        print(f"\n   Chunk {i}:")
        print(f"   {preview}")
        if hasattr(chunk, 'metadata'):
            print(f"   Metadata: {chunk.metadata}")
    
except Exception as e:
    print(f"❌ Ingestion failed: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

# %% [markdown]
# ## Cell 3: Initialize SourceTracker
#
# Create an instance of the SourceTracker utility class.
# This is NOT an AI agent - it's a helper utility for tracking sources.

# %% SourceTracker Initialization
from app.utils.source_tracker import SourceTracker

print("🔍 Initializing SourceTracker...")

try:
    # Create tracker for this document
    tracker = SourceTracker(document_id=document_id)
    
    print("\n✅ SourceTracker Initialized Successfully!")
    print(f"   Document ID: {tracker.document_id}")
    print(f"   Type: Utility Class (not an AI agent)")
    print(f"   Purpose: Track source information for findings")
    
    # Display available methods
    print("\n🔧 Available Methods:")
    methods = [
        ("create_source_reference", "Create a reference to a source location"),
        ("create_source_metadata", "Create metadata from source references"),
        ("track_item", "Track source metadata for an item"),
        ("get_source_metadata", "Retrieve source metadata for an item"),
        ("generate_link", "Generate frontend-renderable link"),
        ("generate_links", "Generate multiple links from metadata"),
        ("trace_chain", "Trace full source chain for an item"),
        ("embed_provenance_in_clause", "Add source info to extracted clause"),
        ("embed_provenance_in_finding", "Add source info to key finding"),
        ("embed_provenance_in_recommendation", "Add source info to recommendation"),
        ("get_sources_for_item", "Get all sources for an item"),
        ("get_items_from_source", "Get all items from a source"),
        ("generate_citation", "Generate citation string"),
    ]
    
    for i, (method, description) in enumerate(methods, 1):
        print(f"   {i}. {method}(): {description}")
    
except Exception as e:
    print(f"❌ SourceTracker initialization failed: {str(e)}")
    raise

# %% [markdown]
# ## Cell 4: Create Source References
#
# Demonstrate creating source references from document chunks.
# This shows how we track where information came from.

# %% Source Reference Creation
print("📍 Creating Source References from Document Chunks\n")

try:
    # Create source references from our sample chunks
    source_refs = []
    
    for i, chunk in enumerate(sample_chunks[:3], 1):
        # Extract metadata if available
        page = chunk.metadata.get("page") if hasattr(chunk, 'metadata') else None
        section = chunk.metadata.get("section") if hasattr(chunk, 'metadata') else None
        chunk_id = f"chunk_{i}"
        
        # Create source reference
        source_ref = tracker.create_source_reference(
            chunk_id=chunk_id,
            text_snippet=chunk.page_content[:150],
            page=page,
            section=section,
            confidence=0.95
        )
        
        source_refs.append(source_ref)
        
        print(f"Source Reference {i}:")
        print(f"   Document ID: {source_ref.document_id}")
        print(f"   Chunk ID: {source_ref.chunk_id}")
        print(f"   Page: {source_ref.page}")
        print(f"   Section: {source_ref.section}")
        print(f"   Confidence: {source_ref.confidence}")
        print(f"   Text Snippet: {source_ref.text_snippet[:100]}...")
        print()
    
    print(f"✅ Created {len(source_refs)} source references")
    
except Exception as e:
    print(f"❌ Source reference creation failed: {str(e)}")
    import traceback
    traceback.print_exc()

# %% [markdown]
# ## Cell 5: Create Source Metadata and Generate Links
#
# Demonstrate creating source metadata from references and generating
# frontend-renderable links for navigation.

# %% Source Metadata and Link Generation
print("🔗 Creating Source Metadata and Generating Links\n")

try:
    # Create source metadata from references
    source_metadata = tracker.create_source_metadata(
        sources=source_refs,
        extraction_method="llm_extraction",
        confidence=None  # Will calculate average
    )
    
    print("Source Metadata:")
    print(f"   Number of sources: {len(source_metadata.sources)}")
    print(f"   Overall confidence: {source_metadata.confidence:.2f}")
    print(f"   Extraction method: {source_metadata.extraction_method}")
    print(f"   Timestamp: {source_metadata.timestamp}")
    
    # Generate frontend links
    links = tracker.generate_links(source_metadata)
    
    print(f"\n🔗 Generated {len(links)} Frontend Links:")
    for i, link in enumerate(links, 1):
        print(f"\n   Link {i}:")
        print(f"   ID: {link.link_id}")
        print(f"   Text: {link.link_text}")
        print(f"   URL: {link.link_url}")
        print(f"   Tooltip: {link.tooltip[:80]}...")
    
    # Generate citations
    short_citation = tracker.generate_citation(source_metadata, format="short")
    full_citation = tracker.generate_citation(source_metadata, format="full")
    
    print(f"\n📚 Citations:")
    print(f"   Short: {short_citation}")
    print(f"   Full: {full_citation}")
    
    print("\n✅ Source metadata and links created successfully!")
    
except Exception as e:
    print(f"❌ Metadata/link generation failed: {str(e)}")
    import traceback
    traceback.print_exc()

# %% [markdown]
# ## Cell 6: Integration with Clause Extraction Agent
#
# Demonstrate how SourceTracker integrates with the Clause Extraction Agent
# to embed source information in extracted clauses.

# %% Integration with Clause Extraction
from app.agents.clause_extraction import ClauseExtractionAgent
from app.models.agent import ExtractedClause

print("🤖 Demonstrating Integration with Clause Extraction Agent\n")

try:
    # Create clause extraction agent
    clause_agent = ClauseExtractionAgent(
        vector_store=vector_store,
        model_name="gpt-4o-mini",
        temperature=0.0,
        top_k=10
    )
    
    print("✅ Clause Extraction Agent initialized")
    
    # Extract clauses
    print("\n🔍 Extracting clauses from document...")
    extraction_result = clause_agent.extract_clauses(document_id=document_id)
    
    print(f"✅ Extracted {len(extraction_result.clauses)} clauses")
    
    # Prepare chunk metadata for source tracking
    chunk_metadata = {}
    for i, chunk in enumerate(sample_chunks, 1):
        chunk_id = f"chunk_{i}"
        chunk_metadata[chunk_id] = {
            "page": chunk.metadata.get("page") if hasattr(chunk, 'metadata') else None,
            "section": chunk.metadata.get("section") if hasattr(chunk, 'metadata') else None,
            "text": chunk.page_content[:200]
        }
    
    # Embed source tracking in clauses
    print("\n📍 Embedding source information in clauses...")
    enriched_clauses = []
    
    for clause in extraction_result.clauses[:3]:  # Process first 3 clauses
        # Simulate chunk IDs (in real scenario, these come from extraction)
        clause.source_chunk_ids = [f"chunk_{i+1}" for i in range(min(2, len(sample_chunks)))]
        
        # Embed provenance
        enriched_clause = tracker.embed_provenance_in_clause(clause, chunk_metadata)
        enriched_clauses.append(enriched_clause)
        
        print(f"\n   Clause: {enriched_clause.clause_type}")
        print(f"   Text: {enriched_clause.clause_text[:100]}...")
        
        if enriched_clause.provenance:
            print(f"   ✅ Source tracking embedded:")
            print(f"      Sources: {len(enriched_clause.provenance.sources)}")
            print(f"      Confidence: {enriched_clause.provenance.confidence:.2f}")
            print(f"      Method: {enriched_clause.provenance.extraction_method}")
            
            # Show first source
            if enriched_clause.provenance.sources:
                src = enriched_clause.provenance.sources[0]
                print(f"      First source: Page {src.page}, Section {src.section}, Chunk {src.chunk_id}")
    
    print(f"\n✅ Embedded source tracking in {len(enriched_clauses)} clauses")
    
except Exception as e:
    print(f"❌ Integration demo failed: {str(e)}")
    import traceback
    traceback.print_exc()

# %% [markdown]
# ## Cell 7: Demonstrate Provenance Chain Tracking
#
# Show how to track items, query sources, and trace provenance chains.

# %% Provenance Chain Tracking
print("🔗 Demonstrating Provenance Chain Tracking\n")

try:
    # Track some items
    print("1️⃣ Tracking Items:")
    for i, clause in enumerate(enriched_clauses, 1):
        if clause.provenance:
            item_id = f"clause_{clause.clause_type}_{i}"
            tracker.track_item(item_id, clause.provenance)
            print(f"   Tracked: {item_id}")
    
    # Query: Get sources for a specific item
    print("\n2️⃣ Query: Get Sources for Item")
    if enriched_clauses and enriched_clauses[0].provenance:
        item_id = f"clause_{enriched_clauses[0].clause_type}_1"
        sources = tracker.get_sources_for_item(item_id)
        print(f"   Item: {item_id}")
        print(f"   Sources found: {len(sources)}")
        for src in sources:
            print(f"      - Chunk {src.chunk_id}, Page {src.page}, Section {src.section}")
    
    # Query: Get items from a specific source
    print("\n3️⃣ Query: Get Items from Source")
    if source_refs:
        chunk_id = source_refs[0].chunk_id
        items = tracker.get_items_from_source(chunk_id)
        print(f"   Source: {chunk_id}")
        print(f"   Items found: {len(items)}")
        for item_id in items:
            print(f"      - {item_id}")
    
    # Trace provenance chain
    print("\n4️⃣ Trace Provenance Chain")
    if enriched_clauses and enriched_clauses[0].provenance:
        item_id = f"clause_{enriched_clauses[0].clause_type}_1"
        chain = tracker.trace_chain(item_id)
        print(f"   Item: {item_id}")
        print(f"   Chain length: {len(chain)}")
        for i, metadata in enumerate(chain, 1):
            print(f"      Level {i}: {len(metadata.sources)} sources, confidence={metadata.confidence:.2f}")
    
    print("\n✅ Provenance chain tracking demonstrated successfully!")
    
except Exception as e:
    print(f"❌ Provenance tracking demo failed: {str(e)}")
    import traceback
    traceback.print_exc()

# %% [markdown]
# ## Cell 8: Export Results
#
# Export the source tracking results to JSON for inspection.

# %% Export Results
print("💾 Exporting Source Tracking Results\n")

try:
    # Prepare export data
    export_data = {
        "document_id": document_id,
        "tracker_info": {
            "total_items_tracked": len(tracker._source_chain),
            "total_source_references": len(source_refs)
        },
        "source_references": [
            {
                "chunk_id": ref.chunk_id,
                "document_id": ref.document_id,
                "page": ref.page,
                "section": ref.section,
                "text_snippet": ref.text_snippet[:100],
                "confidence": ref.confidence
            }
            for ref in source_refs
        ],
        "frontend_links": [
            {
                "link_id": link.link_id,
                "link_text": link.link_text,
                "link_url": link.link_url,
                "tooltip": link.tooltip[:100]
            }
            for link in links
        ],
        "enriched_clauses": [
            {
                "clause_type": clause.clause_type,
                "clause_text": clause.clause_text[:200],
                "has_provenance": clause.provenance is not None,
                "source_count": len(clause.provenance.sources) if clause.provenance else 0
            }
            for clause in enriched_clauses
        ]
    }
    
    # Export to JSON
    results_file = Path(__file__).parent.parent / "backend" / "source_tracker_results.json"
    with open(results_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"✅ Results exported to: {results_file.name}")
    print(f"   Total source references: {len(source_refs)}")
    print(f"   Total frontend links: {len(links)}")
    print(f"   Total enriched clauses: {len(enriched_clauses)}")
    
except Exception as e:
    print(f"❌ Export failed: {str(e)}")
    import traceback
    traceback.print_exc()

# %% [markdown]
# ## Summary
#
# This notebook has validated the SourceTracker implementation:
#
# ✅ **SourceTracker Initialization**: Successfully created utility class
# ✅ **Source Reference Creation**: Tracked document chunks with metadata
# ✅ **Source Metadata Management**: Aggregated sources with confidence scores
# ✅ **Link Generation**: Created frontend-renderable links for navigation
# ✅ **Integration**: Embedded source info in extracted clauses
# ✅ **Provenance Chain**: Tracked and queried source relationships
#
# ### Key Features Demonstrated
# - **Source Tracking**: Document ID, page, section, chunk ID, text snippets
# - **Confidence Scoring**: Automatic averaging of source confidences
# - **Frontend Links**: Clickable links with tooltips for source navigation
# - **Bidirectional Queries**: item→sources and source→items lookups
# - **Citation Generation**: Short and full citation formats
# - **Agent Integration**: Embedding source info in clause extraction outputs
#
# ### Architecture Notes
# - **NOT an AI Agent**: SourceTracker is a utility class, not a ReAct agent
# - **Supporting Role**: Provides helper methods for other agents to use
# - **Location**: `backend/app/utils/` (not in `agents/` directory)
# - **Naming**: "Source" is clearer than "Provenance" (technical term for origin/source)
#
# ### Integration Points
# - **Clause Extraction Agent**: Use `embed_provenance_in_clause()`
# - **Risk Scoring Agent**: Maintain provenance from input clauses
# - **Summary Agent**: Use `embed_provenance_in_finding()` and `embed_provenance_in_recommendation()`
# - **Frontend**: Use SourceLink objects for clickable source navigation
#
# ### Next Steps
# - Epic 3: Implement Checklist Agent (Story 3.5)
# - Epic 4: Agent orchestration with LangGraph
# - Integration: Connect agents to API endpoints with source tracking
#
# ### Troubleshooting
# - **OpenAI API Error**: Check OPENAI_API_KEY in `.env` file
# - **Import Errors**: Ensure backend dependencies installed (`cd backend && uv sync`)
# - **No Sources**: Check that chunks have metadata (page, section)
