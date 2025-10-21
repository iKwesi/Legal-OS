# Legal-OS Backend

FastAPI-based backend for Legal-OS with RAG pipeline and multi-agent orchestration.

## 🏗️ Architecture

The backend is organized into several key modules:

- **`app/agents/`** - Specialized AI agents for legal document analysis
- **`app/api/`** - REST API endpoints
- **`app/core/`** - Configuration and utilities
- **`app/models/`** - Pydantic data models
- **`app/orchestration/`** - ReAct-based agent orchestration
- **`app/pipelines/`** - Document ingestion pipeline
- **`app/rag/`** - RAG components (retrievers, embeddings)

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start Qdrant (required)
docker-compose up -d qdrant

# Run development server
uv run uvicorn app.main:app --reload
```

Access the API:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Docker

```bash
# From project root
docker-compose up -d backend
```

## 📚 API Endpoints

### Document Management

#### Upload Document
```http
POST /api/v1/upload
Content-Type: multipart/form-data

file: <PDF file>
```

**Response:**
```json
{
  "document_id": "uuid-string",
  "filename": "document.pdf",
  "status": "processed",
  "chunks_created": 42,
  "message": "Document uploaded and processed successfully"
}
```

#### List Documents
```http
GET /api/v1/documents
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "uuid-string",
      "filename": "document.pdf",
      "upload_date": "2025-01-20T10:30:00Z",
      "chunks": 42
    }
  ]
}
```

#### Get Document Details
```http
GET /api/v1/documents/{document_id}
```

**Response:**
```json
{
  "document_id": "uuid-string",
  "filename": "document.pdf",
  "upload_date": "2025-01-20T10:30:00Z",
  "chunks": 42,
  "metadata": {
    "pages": 15,
    "size_bytes": 524288
  }
}
```

### Query & Chat

#### Query Documents
```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "What are the key terms of this agreement?",
  "document_id": "uuid-string",  // optional
  "top_k": 5  // optional, default: 5
}
```

**Response:**
```json
{
  "query": "What are the key terms of this agreement?",
  "answer": "The key terms include...",
  "sources": [
    {
      "content": "Relevant text chunk...",
      "metadata": {
        "document_id": "uuid-string",
        "page": 3,
        "chunk_id": 12
      },
      "score": 0.89
    }
  ],
  "retrieval_method": "base"
}
```

#### Chat Interface
```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "Tell me about the termination clauses",
  "document_id": "uuid-string",  // optional
  "conversation_id": "uuid-string",  // optional, for multi-turn
  "top_k": 5  // optional
}
```

**Response:**
```json
{
  "response": "The termination clauses state...",
  "conversation_id": "uuid-string",
  "sources": [
    {
      "content": "Relevant text...",
      "metadata": {...},
      "score": 0.92
    }
  ]
}
```

### Agent Operations

#### Clause Extraction
```http
POST /api/v1/agents/clause-extraction
Content-Type: application/json

{
  "document_id": "uuid-string",
  "clause_types": ["termination", "indemnification", "confidentiality"]  // optional
}
```

**Response:**
```json
{
  "document_id": "uuid-string",
  "clauses": [
    {
      "type": "termination",
      "content": "Either party may terminate...",
      "location": {
        "page": 8,
        "section": "Article 12"
      },
      "confidence": 0.95
    }
  ],
  "total_clauses": 15
}
```

#### Risk Scoring
```http
POST /api/v1/agents/risk-scoring
Content-Type: application/json

{
  "document_id": "uuid-string",
  "focus_areas": ["financial", "legal", "operational"]  // optional
}
```

**Response:**
```json
{
  "document_id": "uuid-string",
  "overall_risk_score": 6.5,
  "risk_level": "medium",
  "risks": [
    {
      "category": "financial",
      "description": "Unlimited liability clause",
      "severity": "high",
      "score": 8.0,
      "location": {
        "page": 5,
        "section": "Article 7"
      },
      "mitigation": "Consider negotiating a liability cap"
    }
  ]
}
```

#### Summary Generation
```http
POST /api/v1/agents/summary
Content-Type: application/json

{
  "document_id": "uuid-string",
  "summary_type": "executive"  // options: executive, detailed, technical
}
```

**Response:**
```json
{
  "document_id": "uuid-string",
  "summary_type": "executive",
  "summary": "This Asset Purchase Agreement...",
  "key_points": [
    "Purchase price: $10M",
    "Closing date: 30 days from signing",
    "Key conditions: regulatory approval required"
  ],
  "word_count": 250
}
```

#### Checklist Generation
```http
POST /api/v1/agents/checklist
Content-Type: application/json

{
  "document_id": "uuid-string",
  "checklist_type": "due_diligence"  // options: due_diligence, compliance, review
}
```

**Response:**
```json
{
  "document_id": "uuid-string",
  "checklist_type": "due_diligence",
  "items": [
    {
      "category": "Financial",
      "item": "Verify purchase price calculation",
      "priority": "high",
      "status": "pending",
      "notes": "Review Schedule A for detailed breakdown"
    }
  ],
  "total_items": 25,
  "high_priority": 8
}
```

### Orchestration

#### Full Analysis
```http
POST /api/v1/orchestrate
Content-Type: application/json

{
  "document_id": "uuid-string",
  "query": "Perform a comprehensive due diligence analysis",
  "agents": ["clause_extraction", "risk_scoring", "summary", "checklist"]  // optional
}
```

**Response:**
```json
{
  "document_id": "uuid-string",
  "orchestration_id": "uuid-string",
  "status": "completed",
  "results": {
    "clause_extraction": {...},
    "risk_scoring": {...},
    "summary": {...},
    "checklist": {...}
  },
  "execution_time_seconds": 45.2,
  "agent_sequence": ["clause_extraction", "risk_scoring", "summary", "checklist"]
}
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "qdrant": "connected",
    "openai": "available"
  }
}
```

## 🧪 Testing

### Run All Tests
```bash
uv run pytest -v
```

### Run Specific Test Categories
```bash
# Ingestion pipeline
uv run pytest tests/test_ingestion.py -v

# RAG components
uv run pytest tests/test_rag.py -v

# Agents
uv run pytest tests/test_clause_extraction.py -v
uv run pytest tests/test_risk_scoring.py -v
uv run pytest tests/test_summary.py -v
uv run pytest tests/test_checklist.py -v

# API endpoints
uv run pytest tests/test_api.py -v

# Orchestration
uv run pytest tests/test_orchestration.py -v
```

### Run with Coverage
```bash
uv run pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## 🔧 Configuration

### Environment Variables

See `.env.example` for all available configuration options:

```bash
# Required
OPENAI_API_KEY=sk-...

# LLM Configuration
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2000

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Qdrant Configuration
QDRANT_HOST=localhost  # Use 'qdrant' in Docker
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=legal_documents

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.6
```

### Configuration File

The `app/core/config.py` module provides centralized configuration management using Pydantic settings.

## 🤖 Agent System

### Available Agents

1. **Clause Extraction Agent** (`app/agents/clause_extraction.py`)
   - Identifies and extracts specific clause types
   - Supports custom clause definitions
   - Provides confidence scores

2. **Risk Scoring Agent** (`app/agents/risk_scoring.py`)
   - Analyzes potential risks in documents
   - Categorizes risks by type and severity
   - Suggests mitigation strategies

3. **Summary Agent** (`app/agents/summary.py`)
   - Generates executive summaries
   - Extracts key points and terms
   - Supports multiple summary styles

4. **Checklist Agent** (`app/agents/checklist.py`)
   - Creates due diligence checklists
   - Prioritizes action items
   - Tracks completion status

5. **Provenance Agent** (`app/agents/source_tracker.py`)
   - Tracks source attribution
   - Maintains citation accuracy
   - Links claims to source documents

### Agent Orchestration

The `app/orchestration/` module implements ReAct-based orchestration using LangGraph:

- **Sequential Execution**: Agents run in logical order
- **Conditional Branching**: Dynamic agent selection based on results
- **State Management**: Maintains context across agent calls
- **Error Handling**: Graceful degradation on agent failures

## 📊 RAG Pipeline

### Retrieval Strategies

1. **Base Retriever** - Simple vector similarity search
2. **Contextual Compression** - Filters and compresses retrieved chunks
3. **Multi-Query** - Generates multiple query variations
4. **Ensemble** - Combines multiple retrieval methods
5. **Reranking** - Uses Cohere Rerank API for improved relevance

### Swappable Architecture

The RAG system supports runtime retriever selection:

```python
from app.rag.shared import get_retriever

# Get specific retriever
retriever = get_retriever(
    collection_name="legal_documents",
    retriever_type="rerank",
    top_k=5
)
```

## 🔍 Code Quality

### Linting
```bash
uv run ruff check app/
```

### Formatting
```bash
uv run black app/
```

### Type Checking
```bash
uv run mypy app/
```

## 📝 Development Guidelines

1. **Code Style**: Follow PEP 8, use Black for formatting
2. **Type Hints**: Use type hints for all function signatures
3. **Documentation**: Add docstrings to all public functions
4. **Testing**: Write tests for new features
5. **Error Handling**: Use proper exception handling
6. **Logging**: Use Python's logging module for debugging

## 🐛 Troubleshooting

### Common Issues

**Import errors:**
```bash
# Recreate virtual environment
rm -rf .venv
uv sync
```

**Qdrant connection errors:**
```bash
# Check Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant
```

**OpenAI API errors:**
- Verify API key in `.env`
- Check API quota and billing
- Ensure no extra whitespace in `.env`

**Test failures:**
- Ensure Qdrant is running
- Check environment variables are set
- Verify test data exists in `data/` directory

## 📖 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [RAGAS Documentation](https://docs.ragas.io/)
