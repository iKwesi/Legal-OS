# %% [markdown]
# # E01: Pipeline Foundation - Epic 1 Validation
#
# ## Purpose
# This notebook validates the implementations from Epic 1:
# - **Story 1.2**: Ingestion Agent & RAG Pipeline
# - **Story 1.3**: Synthetic Golden Dataset (SGD) Generation
#
# ## What This Notebook Demonstrates
# 1. **Document Ingestion**: Loading and processing legal documents into in-memory vector store
# 2. **RAG Pipeline**: Querying documents using Naive Retrieval
# 3. **SGD Generation**: Creating synthetic test datasets with RAGAS
#
# ## Prerequisites
# - Environment variables configured (OPENAI_API_KEY)
# - Sample documents in `data/` directory
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
import pandas as pd
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
# ## Cell 2: Document Ingestion Test
#
# This cell demonstrates the Ingestion Agent from Story 1.2:
# - Loads a sample legal document from the `data/` directory
# - Processes it using RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
# - Stores embeddings in **in-memory Qdrant vector store**
# - Displays ingestion results

# %% Document Ingestion
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Select a sample document
data_dir = Path(__file__).parent.parent / "data"
sample_docs = list(data_dir.glob("*.pdf"))

if not sample_docs:
    raise FileNotFoundError(f"No PDF documents found in {data_dir}")

# Use the first available document
sample_doc = sample_docs[0]
print(f"📄 Ingesting document: {sample_doc.name}")
print(f"   File size: {sample_doc.stat().st_size / 1024:.2f} KB")

try:
    # Load document
    loader = PyMuPDFLoader(str(sample_doc))
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create in-memory Qdrant client
    qdrant_client = QdrantClient(location=":memory:")
    
    # Create collection
    collection_name = "legal_documents"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    # Add documents to vector store
    vector_store.add_documents(chunks)
    
    print("\n✅ Ingestion Complete!")
    print(f"   Document: {sample_doc.name}")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Collection: {collection_name} (in-memory)")
    
    # Display sample chunks
    print("\n📝 Sample Chunks (first 2):")
    for i, chunk in enumerate(chunks[:2], 1):
        preview = chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
        print(f"\n   Chunk {i}:")
        print(f"   {preview}")
    
except Exception as e:
    print(f"❌ Ingestion failed: {str(e)}")
    raise

# %% [markdown]
# ## Cell 3: RAG Pipeline Test
#
# This cell demonstrates the RAG pipeline from Story 1.2:
# - Uses Naive Retrieval (basic similarity search)
# - Queries the in-memory vector store
# - Displays retrieved context and generated answer
# - Shows performance metrics

# %% RAG Pipeline Test
import time
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Define a test question relevant to legal documents
test_question = "What are the key parties involved in this agreement?"

print(f"❓ Question: {test_question}\n")

try:
    print("🔧 Creating RAG chain with Naive Retrieval...")
    
    # Create retriever from vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    # Create prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Execute query and measure time
    start_time = time.time()
    response = rag_chain.invoke({"input": test_question})
    elapsed_time = time.time() - start_time
    
    print("✅ RAG Query Complete!\n")
    print(f"⏱️  Response Time: {elapsed_time:.2f} seconds\n")
    
    # Display retrieved context
    if 'context' in response:
        contexts = response['context']
        print(f"📚 Retrieved Contexts ({len(contexts)} chunks):")
        for i, ctx in enumerate(contexts[:3], 1):  # Show first 3
            preview = ctx.page_content[:200] + "..." if len(ctx.page_content) > 200 else ctx.page_content
            print(f"\n   Context {i}:")
            print(f"   {preview}")
    
    # Display answer
    answer = response.get('answer', '')
    print(f"\n💡 Generated Answer:")
    print(f"   {answer}")
    
except Exception as e:
    print(f"❌ RAG query failed: {str(e)}")
    raise

# %% [markdown]
# ## Cell 4: Synthetic Golden Dataset (SGD) Generation
#
# This cell demonstrates SGD generation from Story 1.3:
# - Runs the SGD generation script with a small test size
# - Loads the generated CSV from `golden_dataset/`
# - Displays sample questions and metadata
# - Shows summary statistics

# %% SGD Generation and Display
import subprocess

# Run SGD generation with small test size to minimize API costs
print("🔬 Generating Synthetic Golden Dataset (test size: 3)...")
print("   This may take a few minutes due to API calls...\n")

try:
    # Run the SGD generation script
    result = subprocess.run(
        [sys.executable, "-m", "app.scripts.generate_sgd", "--test-size", "3"],
        cwd=str(backend_path),
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )
    
    if result.returncode != 0:
        print(f"⚠️  SGD generation had issues:")
        print(result.stderr)
    else:
        print("✅ SGD generation complete!")
        print(result.stdout)
    
except subprocess.TimeoutExpired:
    print("⚠️  SGD generation timed out (>5 minutes)")
except Exception as e:
    print(f"⚠️  SGD generation error: {str(e)}")

# Load and display the generated SGD
golden_dataset_dir = Path(__file__).parent.parent / "golden_dataset"
csv_files = sorted(golden_dataset_dir.glob("sgd_benchmark*.csv"))

if not csv_files:
    print("\n❌ No SGD CSV files found in golden_dataset/")
else:
    # Use the most recent file
    latest_csv = csv_files[-1]
    print(f"\n📊 Loading SGD from: {latest_csv.name}")
    
    try:
        df = pd.read_csv(latest_csv)
        
        print(f"\n✅ SGD Loaded Successfully!")
        print(f"   Total questions: {len(df)}")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        
        # Show summary statistics
        if 'synthesizer_name' in df.columns:
            print(f"\n📈 Question Type Distribution:")
            synthesizer_counts = df['synthesizer_name'].value_counts()
            for synthesizer, count in synthesizer_counts.items():
                print(f"   {synthesizer}: {count}")
        
        # Display sample rows
        print(f"\n📝 Sample Questions (first 3 rows):")
        pd.set_option('display.max_colwidth', 100)
        pd.set_option('display.width', 120)
        
        for idx, row in df.head(3).iterrows():
            print(f"\n   Question {idx + 1}:")
            print(f"   User Input: {row['user_input'][:150]}...")
            
            if 'reference' in row and pd.notna(row['reference']):
                print(f"   Reference: {str(row['reference'])[:150]}...")
            
            if 'synthesizer_name' in row:
                print(f"   Synthesizer: {row['synthesizer_name']}")
        
        # Display full dataframe info
        print(f"\n📋 Full Dataset Info:")
        print(df.info())
        
    except Exception as e:
        print(f"❌ Error loading SGD CSV: {str(e)}")
        raise

# %% [markdown]
# ## Summary
#
# This notebook has validated the Epic 1 implementations:
#
# ✅ **Document Ingestion**: Successfully loaded and chunked legal documents into Qdrant
# ✅ **RAG Pipeline**: Retrieved relevant context and generated answers using Naive Retrieval
# ✅ **SGD Generation**: Created synthetic test dataset with RAGAS
#
# ### Next Steps
# - Epic 2: Advanced RAG techniques (BM25, Multi-Query, Reranking)
# - Epic 3: Agent implementation (Clause Extraction, Risk Scoring, etc.)
# - Epic 4: Agent orchestration with LangGraph
#
# ### Troubleshooting
# - **OpenAI API Error**: Check OPENAI_API_KEY in `.env` file or environment variables
# - **Import Errors**: Ensure backend dependencies are installed (`cd backend && uv sync`)
# - **Memory Issues**: If processing large documents, consider reducing chunk size or using fewer documents
