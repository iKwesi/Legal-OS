# Legal-OS

AI-powered M&A due diligence system with multi-agent orchestration, RAG pipeline, and comprehensive RAGAS evaluation.

## 🎯 Overview

Legal-OS analyzes M&A legal documents using specialized AI agents to extract clauses, score risks, generate summaries, track sources, and create actionable checklists.

**Key Features:**
- 🤖 **5 Specialized AI Agents** - Clause extraction, risk scoring, summary, source tracking, checklist
- 🔍 **Advanced RAG Pipeline** - 5 retriever strategies with RAGAS evaluation
- 📊 **Comprehensive Evaluation** - 10 configurations tested (2 chunking × 5 retrievers)
- 💬 **Interactive Chat** - Ask questions about uploaded documents
- 📈 **LangSmith Tracing** - Optional monitoring and debugging
- 🎨 **Modern UI** - Next.js frontend with Shadcn/ui components

## 🏗️ System Architecture

```mermaid
flowchart TD
    USER["👤 User"] --> FRONTEND["🎨 Next.js Frontend<br/>React + TypeScript"]
    
    FRONTEND --> API["⚡ FastAPI Backend<br/>REST API"]
    
    API --> UPLOAD["📤 Document Upload<br/>PDF Processing"]
    API --> CHAT["💬 Interactive Chat<br/>RAG Queries"]
    API --> ORCHESTRATE["🎯 Full Analysis<br/>Multi-Agent Pipeline"]
    
    UPLOAD --> PROCESSING["📄 Document Processing<br/>Chunking + Embeddings"]
    PROCESSING --> VECTOR["💾 Qdrant Vector Store<br/>Semantic Search"]
    
    CHAT --> RAG["🔍 RAG Pipeline<br/>5 Retriever Strategies"]
    RAG --> VECTOR
    
    ORCHESTRATE --> AGENTS["🤖 LangGraph Orchestration<br/>4 AI Agents"]
    AGENTS --> AGENT1["1️⃣ Clause Extraction Agent<br/>LLM-Powered"]
    AGENTS --> AGENT2["2️⃣ Risk Scoring Agent<br/>LLM-Powered"]
    AGENTS --> AGENT3["3️⃣ Summary Agent<br/>LLM-Powered"]
    AGENTS --> AGENT4["4️⃣ Checklist Agent<br/>LLM-Powered"]
    
    AGENT1 --> RAG
    AGENT2 --> AGENT1
    AGENT3 --> AGENT2
    AGENT4 --> AGENT3
    
    AGENT1 --> TRACKER["📍 Source Tracker<br/>Provenance Utility"]
    AGENT2 --> TRACKER
    AGENT3 --> TRACKER
    AGENT4 --> TRACKER
    
    TRACKER --> RESULTS["📊 Analysis Results<br/>Memo + Checklist + Risk Scores<br/>with Source Attribution"]
    
    RESULTS --> FRONTEND
    
    AGENTS -.->|LLM Calls| OPENAI["🤖 OpenAI API<br/>gpt-4o-mini"]
    RAG -.->|Optional| COHERE["⭐ Cohere Rerank"]
    AGENTS -.->|Tracing| LANGSMITH["📊 LangSmith"]
    
    %% Styling
    classDef frontend fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000000,font-weight:bold
    classDef backend fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000,font-weight:bold
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000000,font-weight:bold
    classDef agents fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000000,font-weight:bold
    classDef utility fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000000,font-weight:bold
    classDef external fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000000,font-weight:bold
    classDef output fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000000,font-weight:bold
    
    class USER,FRONTEND frontend
    class API,UPLOAD,CHAT,ORCHESTRATE backend
    class PROCESSING,VECTOR,RAG processing
    class AGENTS,AGENT1,AGENT2,AGENT3,AGENT4 agents
    class TRACKER utility
    class OPENAI,COHERE,LANGSMITH external
    class RESULTS output
```

### Architecture Highlights

**🔄 Multi-Agent Orchestration (LangGraph)**
- Supervisor agent coordinates 5 specialized agents using ReAct pattern
- Sequential workflow: Ingestion → Clause Extraction → Risk Scoring → Summary → Provenance → Checklist
- State management with checkpointing for reliability

**🔍 Advanced RAG Pipeline**
- Swappable retriever architecture (5 strategies)
- Dual chunking strategies (Naive vs Semantic)
- RAGAS evaluation framework with 10 configurations
- Best-in-class retriever selection based on metrics

**🎯 Production-Ready Design**
- FastAPI backend with async support
- In-memory Qdrant for development
- Docker Compose for production deployment
- Comprehensive test coverage with pytest

**📊 Observability**
- LangSmith integration for tracing
- Detailed logging throughout pipeline
- Performance metrics tracking
- Error handling with fallback logic

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager
- **Node.js 20+** (for frontend)
- **OpenAI API Key** (required)
- **Cohere API Key** (optional - for reranking)
- **LangSmith API Key** (optional - for tracing)

### Setup (3 Steps)

```bash
# 1. Install backend dependencies
cd backend
uv sync

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# 3. Run the system
uv run python main.py
```

Backend runs on: http://localhost:8000
API docs: http://localhost:8000/docs

### Frontend Setup (Optional)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: http://localhost:3000

## 📊 Jupyter Notebooks

### Comprehensive Demo (Recommended)

```bash
# Open in Jupyter
cd notebooks
jupyter notebook combined_notebooks_v2.ipynb

# Or run as Python script
cd backend
uv run python ../notebooks/combined_notebooks.py
```

**Features:**
- Complete pipeline demonstration
- RAGAS evaluation (10 configurations)
- Ranked comparison table with medals 🥇🥈🥉
- Agent graph visualizations
- LangSmith tracing integration
- ~20-25 minutes runtime

### Individual Notebooks

- `E01_Pipeline_Foundation.py` - Document ingestion & RAG
- `E02_Evaluation_LangChain.py` - Basic RAG evaluation
- `E03_Clause_Extraction_Agent.py` - Clause extraction demo
- `E04_Risk_Scoring_Agent.py` - Risk scoring demo
- `E05_Summary_Agent.py` - Summary generation demo
- `E06_Source_Tracker.py` - Source tracking demo
- `E07_Checklist_Agent.py` - Checklist generation demo
- `E08_Orchestration_Demo.py` - End-to-end orchestration

## 🛠️ Technology Stack

### Backend
- **Python 3.13** with **uv** package manager
- **FastAPI** - Web framework
- **LangChain 0.3.x** - Agent framework (stable)
- **LangGraph** - Agent orchestration
- **RAGAS** - RAG evaluation
- **Qdrant** - Vector database (in-memory for dev)
- **OpenAI** - LLM (gpt-4o-mini) & embeddings (text-embedding-3-small)
- **Cohere** - Reranking (optional)

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Shadcn/ui** - UI components

## 📁 Project Structure

```
Legal-OS/
├── backend/
│   ├── app/
│   │   ├── agents/          # 5 AI agents
│   │   ├── api/v1/          # REST API endpoints
│   │   ├── orchestration/   # Agent coordination
│   │   ├── pipelines/       # Document ingestion
│   │   └── rag/             # RAG components & evaluation
│   ├── tests/               # Pytest tests
│   ├── data/                # Sample PDFs
│   ├── golden_dataset/      # RAGAS test data
│   └── pyproject.toml       # Dependencies (uv)
├── frontend/
│   ├── app/                 # Next.js pages
│   ├── components/          # React components
│   └── lib/                 # API client
├── notebooks/               # Jupyter notebooks
│   ├── combined_notebooks_v2.ipynb  # Comprehensive demo
│   └── E01-E08 notebooks    # Individual demos
└── docs/                    # Documentation
```

## 🔧 Configuration

### Required: `backend/.env`

```bash
# OpenAI (Required)
OPENAI_API_KEY=sk-...

# Optional - LangSmith Tracing
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Legal-OS-Evaluation

# Optional - Cohere Reranking
COHERE_API_KEY=...

# LLM Settings
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0

# RAG Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
```

## 📊 RAGAS Evaluation

The system evaluates 10 retriever configurations:

**Chunking Strategies (2):**
- Naive (RecursiveCharacterTextSplitter)
- Semantic (Semantic-based splitting)

**Retrievers (5):**
1. Vector Similarity
2. BM25 Keyword
3. Multi-Query
4. Ensemble (Vector + BM25 + Multi-Query)
5. Cohere Reranking (if API key provided)

**Results:**
- Ranked table (best to worst)
- RAGAS metrics: Precision, Recall, Faithfulness, Relevancy
- Performance vs accuracy analysis
- Best retriever recommendation

## 🤖 AI Agents

### 1. Clause Extraction Agent
Extracts M&A clauses (payment terms, warranties, indemnification, etc.) and detects red flags.

### 2. Risk Scoring Agent
Assigns risk scores (0-100) to clauses based on defined rules and categorizes as Low/Medium/High/Critical.

### 3. Summary Agent
Generates comprehensive M&A diligence memos with executive summary, findings, and recommendations.

### 4. Source Tracker
Tracks provenance of all findings with document references and page numbers.

### 5. Checklist Agent
Creates due diligence checklists and follow-up questions based on analysis.

## 🧪 Testing

```bash
# Run all tests
cd backend
uv run pytest -v

# Run specific tests
uv run pytest tests/test_agents/ -v
uv run pytest tests/test_rag.py -v

# With coverage
uv run pytest --cov=app --cov-report=html
```

## 🐳 Docker (Optional - For Production)

```bash
# Build and start all services
docker-compose up --build

# Access services
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# Qdrant: http://localhost:6333
```

**Note:** Docker is optional for development. The system works with in-memory Qdrant.

## 📖 API Endpoints

### Document Operations
- `POST /api/v1/upload` - Upload PDF document
- `POST /api/v1/query` - Query with RAG
- `POST /api/v1/chat` - Interactive chat

### Agent Operations
- `POST /api/v1/agents/clause-extraction` - Extract clauses
- `POST /api/v1/agents/risk-scoring` - Score risks
- `POST /api/v1/agents/summary` - Generate memo
- `POST /api/v1/agents/checklist` - Create checklist

### Orchestration
- `POST /api/v1/orchestrate` - Run complete analysis pipeline

Full API docs: http://localhost:8000/docs

## 🎓 Learning Resources

### Notebooks
Start with `notebooks/combined_notebooks_v2.ipynb` for a complete demonstration of:
- Document ingestion
- RAG pipeline
- All 5 agents
- RAGAS evaluation
- End-to-end orchestration

### Documentation
- [Product Requirements](docs/prd/) - System specifications
- [Architecture](docs/architecture/) - Technical design
- [User Stories](docs/stories/) - Development stories

## 🐛 Troubleshooting

### Import Errors
```bash
# Ensure backend venv is activated
cd backend
uv sync

# In VS Code/Cursor: Select interpreter
# Cmd+Shift+P → "Python: Select Interpreter"
# Choose: backend/.venv/bin/python
```

### OpenAI API Errors
- Verify `OPENAI_API_KEY` in `backend/.env`
- Check API key has credits
- Ensure no extra spaces in `.env`

### Notebook Kernel Issues
- Restart kernel: Click "Restart" button
- Select correct kernel: backend/.venv
- Reload VS Code if kernel not visible

### JSON Parsing Errors in Agents
- Normal for LLM-based agents
- Agents have fallback logic
- Check logs for details

## 📝 Development Notes

### Using uv (Package Manager)
```bash
# Add package
uv add package-name

# Remove package
uv remove package-name

# Sync dependencies
uv sync

# Run command in venv
uv run python script.py
```

### LangChain Version
This project uses **LangChain 0.3.x** (stable) for compatibility with `langchain-cohere`.

### In-Memory vs Docker Qdrant
- **Development**: Uses in-memory Qdrant (no Docker needed)
- **Production**: Use Docker Qdrant for persistence

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run tests: `cd backend && uv run pytest`
4. Format code: `cd backend && uv run black .`
5. Submit PR

## 📄 License

[Your License]

---

**Built with:** LangChain • LangGraph • FastAPI • Next.js • Qdrant • OpenAI
