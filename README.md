# Legal-OS

AI-powered legal document processing system with RAG pipeline and multi-agent orchestration.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.13+ (for local development)
- Node.js 20+ (for local development)
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Running with Docker (Recommended)

```bash
# Build all services
make build

# Start all services
make start

# View logs
make logs

# Stop services
make stop
```

Access the services:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant**: http://localhost:6333

### Local Development

#### Backend Setup

```bash
cd backend

# Create virtual environment and install dependencies
uv venv
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Run tests
uv run pytest

# Run development server
uv run uvicorn app.main:app --reload
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## Project Structure

```
Legal-OS/
├── backend/              # Python/FastAPI backend
│   ├── app/
│   │   ├── api/         # API routes
│   │   ├── agents/      # AI agents
│   │   ├── core/        # Core utilities
│   │   ├── models/      # Data models
│   │   ├── orchestration/  # Agent orchestration
│   │   ├── rag/         # RAG pipeline
│   │   └── main.py      # FastAPI entry point
│   ├── tests/           # Backend tests
│   ├── pyproject.toml   # Python dependencies
│   └── Dockerfile
├── frontend/            # Next.js frontend
│   ├── app/            # Next.js App Router
│   ├── components/     # React components
│   ├── lib/            # Utilities
│   └── Dockerfile
├── data/               # Source documents
├── golden_dataset/     # Synthetic golden dataset
├── notebooks/          # Jupyter notebooks
├── docs/               # Documentation
├── docker-compose.yml  # Service orchestration
└── Makefile           # Development commands
```

## Technology Stack

### Backend
- **Python 3.13** - Programming language
- **FastAPI** - Web framework
- **LangChain & LangGraph** - Agent framework
- **RAGAS** - RAG evaluation
- **Qdrant** - Vector database
- **uv** - Package manager

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Shadcn/ui** - UI components

## Available Commands

```bash
make help       # Show all available commands
make build      # Build Docker containers
make start      # Start all services
make stop       # Stop all services
make restart    # Restart all services
make logs       # View service logs
make clean      # Clean up containers and volumes
make test       # Run tests
make format     # Format code
make lint       # Lint code
```

## Development Workflow

1. **Make changes** to backend or frontend code
2. **Run tests** locally: `cd backend && uv run pytest`
3. **Test in Docker**: `make build && make start`
4. **View logs**: `make logs`
5. **Stop services**: `make stop`

## IDE Setup

### VS Code / Cursor

The project includes a local Python virtual environment in `backend/.venv` for IDE support. After running `uv sync` in the backend directory, your IDE should automatically detect the virtual environment and provide proper IntelliSense.

If imports are not resolved:
1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Select "Python: Select Interpreter"
3. Choose the interpreter from `backend/.venv/bin/python`

## Troubleshooting

### Docker build fails
- Ensure Docker daemon is running
- Try: `make clean && make build`

### Import errors in IDE
- Run `cd backend && uv sync` to create local venv
- Select the correct Python interpreter in your IDE

### Port already in use
- Check if services are already running: `docker ps`
- Stop conflicting services or change ports in `docker-compose.yml`

## License

[Your License Here]
