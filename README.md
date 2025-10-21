# Legal-OS

AI-powered legal document processing system with RAG pipeline and multi-agent orchestration for M&A due diligence analysis.

## 🎯 Features

- **📄 Document Ingestion**: Upload and process legal documents (PDF) with intelligent chunking
- **🤖 Multi-Agent System**: Specialized AI agents for different analysis tasks:
  - Clause Extraction Agent
  - Risk Scoring Agent
  - Summary Agent
  - Provenance/Source Tracking Agent
  - Checklist Agent
- **🔍 RAG Pipeline**: Advanced retrieval-augmented generation with multiple retriever strategies
- **💬 Interactive Chat**: Ask questions about uploaded documents with context-aware responses
- **📊 Comprehensive Analysis**: Generate detailed reports, risk assessments, and checklists
- **🎨 Modern UI**: Clean, responsive interface built with Next.js and Shadcn/ui

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended for easiest setup)
- **Python 3.11+** (for local development)
- **Node.js 20+** (for local development)
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager
- **OpenAI API Key** (required for LLM operations)

### Running with Docker (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd Legal-OS

# 2. Set up environment variables
cp backend/.env.example backend/.env
# Edit backend/.env and add your OPENAI_API_KEY

# 3. Build and start all services
make build
make start

# 4. View logs (optional)
make logs
```

Access the services:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### Local Development (Fast Iteration)

For faster development without Docker rebuilds:

```bash
# 1. Install all dependencies
make install-all

# 2. Set up environment variables
cp backend/.env.example backend/.env
# Edit backend/.env and add your OPENAI_API_KEY

# 3. Start Qdrant in Docker (required)
make qdrant

# 4. Start backend and frontend locally
make dev-all
```

This runs the backend and frontend with hot-reload enabled for rapid development.

## 📁 Project Structure

```
Legal-OS/
├── backend/              # Python/FastAPI backend
│   ├── app/
│   │   ├── agents/      # AI agents (clause extraction, risk scoring, etc.)
│   │   ├── api/         # REST API endpoints
│   │   │   └── v1/
│   │   │       └── endpoints/  # Chat, query, upload endpoints
│   │   ├── core/        # Configuration and utilities
│   │   ├── models/      # Pydantic data models
│   │   ├── orchestration/  # ReAct-based agent orchestration
│   │   ├── pipelines/   # Document ingestion pipeline
│   │   ├── rag/         # RAG components (retrievers, embeddings)
│   │   └── main.py      # FastAPI application entry point
│   ├── tests/           # Pytest test suite
│   ├── pyproject.toml   # Python dependencies (uv)
│   └── Dockerfile
├── frontend/            # Next.js/React frontend
│   ├── app/            # Next.js App Router pages
│   │   ├── chat/       # Chat interface
│   │   ├── checklist/  # Checklist view
│   │   └── results/    # Analysis results
│   ├── components/     # React components
│   │   ├── chat/       # Chat UI components
│   │   ├── upload/     # Document upload components
│   │   └── ui/         # Shadcn/ui components
│   ├── lib/            # Utilities and API client
│   └── Dockerfile
├── data/               # Source documents (PDFs)
├── golden_dataset/     # Synthetic golden dataset for evaluation
├── notebooks/          # Jupyter notebooks for testing/demos
├── docs/               # Project documentation
│   ├── prd/           # Product requirements
│   ├── architecture/  # Architecture documentation
│   └── stories/       # User stories
├── docker-compose.yml  # Service orchestration
└── Makefile           # Development commands
```

## 🛠️ Technology Stack

### Backend
- **Python 3.11/3.12** - Programming language
- **FastAPI** - High-performance web framework
- **LangChain & LangGraph** - Agent framework and orchestration
- **RAGAS** - RAG evaluation framework
- **Qdrant** - Vector database for embeddings
- **OpenAI API** - LLM and embeddings (gpt-4o-mini, text-embedding-3-small)
- **uv** - Fast Python package manager
- **Pytest** - Testing framework

### Frontend
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Shadcn/ui** - Accessible UI components (Radix + Tailwind)
- **Zustand** - Lightweight state management

### Infrastructure
- **Docker & Docker Compose** - Containerization and orchestration
- **Qdrant** - Vector database

## 🔧 Environment Setup

### Backend Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure:

```bash
# Required
OPENAI_API_KEY=sk-...                    # Your OpenAI API key

# LLM Configuration
LLM_MODEL=gpt-4o-mini                    # Model for agents
LLM_TEMPERATURE=0.0                      # Temperature (0.0 = deterministic)
LLM_MAX_TOKENS=2000                      # Max tokens per response

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small   # Embedding model
EMBEDDING_DIMENSIONS=1536                # Embedding dimensions

# Qdrant Configuration
QDRANT_HOST=qdrant                       # Use 'localhost' for local dev
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=legal_documents

# RAG Configuration
CHUNK_SIZE=1000                          # Document chunk size
CHUNK_OVERLAP=200                        # Overlap between chunks
RETRIEVAL_TOP_K=5                        # Number of chunks to retrieve
SIMILARITY_THRESHOLD=0.6                 # Minimum similarity score
```

### Frontend Environment Variables

The frontend uses `NEXT_PUBLIC_API_URL` which is set in `docker-compose.yml`. For local development, it defaults to `http://localhost:8000`.

## 📚 API Endpoints

### Document Management
- `POST /api/v1/upload` - Upload and process a legal document
- `GET /api/v1/documents` - List all uploaded documents
- `GET /api/v1/documents/{doc_id}` - Get document details

### Analysis & Querying
- `POST /api/v1/query` - Query documents with RAG retrieval
- `POST /api/v1/chat` - Interactive chat with document context
- `POST /api/v1/orchestrate` - Run full multi-agent analysis

### Agent Operations
- `POST /api/v1/agents/clause-extraction` - Extract clauses from document
- `POST /api/v1/agents/risk-scoring` - Analyze risks in document
- `POST /api/v1/agents/summary` - Generate document summary
- `POST /api/v1/agents/checklist` - Generate due diligence checklist

### Health & Status
- `GET /health` - Health check endpoint

Full API documentation available at: http://localhost:8000/docs

## 🧪 Development Workflow

### 1. Make Changes
Edit backend or frontend code. Changes are hot-reloaded in local dev mode.

### 2. Run Tests

```bash
# Backend tests (local)
cd backend && uv run pytest -v

# Backend tests (Docker)
make test

# Run specific test file
cd backend && uv run pytest tests/test_ingestion.py -v
```

### 3. Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Backend linting only
make backend-lint

# Frontend linting only
make frontend-lint
```

### 4. Test in Docker

```bash
# Rebuild and restart services
make build
make restart

# View logs
make logs

# View specific service logs
make logs-backend
make logs-frontend
```

## 📋 Available Make Commands

### Local Development (Fast)
```bash
make install-all      # Install all dependencies
make qdrant           # Start Qdrant in Docker
make backend          # Start backend locally (port 8000)
make frontend         # Start frontend locally (port 3000)
make dev-all          # Start both backend & frontend
make stop-local       # Stop local processes
make clean-local      # Clean caches
```

### Docker/Production
```bash
make build            # Build Docker containers
make start            # Start all services
make stop             # Stop all services
make restart          # Restart all services
make clean            # Remove containers and volumes
make logs             # View all logs
```

### Individual Services
```bash
make restart-backend  # Restart backend only
make restart-frontend # Restart frontend only
make logs-backend     # Backend logs only
make logs-frontend    # Frontend logs only
```

### Testing & Quality
```bash
make test             # Run backend tests
make format           # Format all code
make lint             # Lint all code
```

Run `make help` to see all available commands.

## 🔍 Testing

### Backend Tests

```bash
# Run all tests
cd backend && uv run pytest -v

# Run specific test categories
uv run pytest tests/test_ingestion.py -v      # Ingestion pipeline
uv run pytest tests/test_rag.py -v            # RAG components
uv run pytest tests/test_agents/ -v           # Agent tests
uv run pytest tests/test_api.py -v            # API endpoints

# Run with coverage
uv run pytest --cov=app --cov-report=html
```

### Frontend Tests

```bash
cd frontend
npm run test          # Run tests
npm run test:watch    # Watch mode
```

## 🎓 Jupyter Notebooks

Interactive notebooks for testing and demonstration:

```bash
# Start Jupyter
cd notebooks
jupyter notebook

# Or use the Python scripts directly
python E01_Pipeline_Foundation.py
python E08_Orchestration_Demo.py
```

Available notebooks:
- `E01_Pipeline_Foundation.py` - Document ingestion pipeline
- `E02_Evaluation_LangChain.py` - RAG evaluation
- `E03_Clause_Extraction_Agent.py` - Clause extraction demo
- `E04_Risk_Scoring_Agent.py` - Risk scoring demo
- `E05_Summary_Agent.py` - Summary generation demo
- `E06_Source_Tracker.py` - Provenance tracking demo
- `E07_Checklist_Agent.py` - Checklist generation demo
- `E08_Orchestration_Demo.py` - Full multi-agent orchestration

## 🐛 Troubleshooting

### Docker Issues

**Build fails:**
```bash
# Clean and rebuild
make clean
make build
```

**Port already in use:**
```bash
# Check running containers
docker ps

# Stop conflicting services
docker stop <container_id>

# Or change ports in docker-compose.yml
```

**Services won't start:**
```bash
# Check logs
make logs

# Restart specific service
make restart-backend
```

### Backend Issues

**Import errors in IDE:**
```bash
# Recreate virtual environment
cd backend
rm -rf .venv
uv sync

# Select interpreter in VS Code/Cursor:
# Cmd/Ctrl + Shift + P → "Python: Select Interpreter"
# Choose: backend/.venv/bin/python
```

**OpenAI API errors:**
- Verify `OPENAI_API_KEY` is set in `backend/.env`
- Check API key is valid and has credits
- Ensure no extra spaces in `.env` file

**Qdrant connection errors:**
```bash
# Ensure Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
make restart-qdrant

# Check Qdrant logs
make logs-qdrant
```

### Frontend Issues

**Module not found:**
```bash
# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**API connection errors:**
- Verify backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` environment variable
- Ensure CORS is configured correctly in backend

### General Issues

**Hot reload not working:**
- Restart the development server
- Check file watchers aren't exhausted (increase limit on Linux)

**Tests failing:**
- Ensure all services are running (`make start`)
- Check environment variables are set
- Verify OpenAI API key is valid

## 📖 Documentation

- **[Product Requirements](docs/prd/)** - Detailed product specifications
- **[Architecture](docs/architecture/)** - System architecture and design
- **[User Stories](docs/stories/)** - Development user stories
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when running)

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `make test`
4. Format code: `make format`
5. Lint code: `make lint`
6. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- [LangChain](https://www.langchain.com/) - Agent framework
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Next.js](https://nextjs.org/) - Frontend framework
- [Qdrant](https://qdrant.tech/) - Vector database
- [Shadcn/ui](https://ui.shadcn.com/) - UI components
