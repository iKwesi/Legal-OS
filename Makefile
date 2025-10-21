.PHONY: help
.PHONY: backend frontend dev-all qdrant install-all stop-local clean-local
.PHONY: backend-install backend-test backend-lint backend-format
.PHONY: frontend-install frontend-build frontend-start frontend-lint
.PHONY: start stop build clean restart logs test format lint
.PHONY: restart-qdrant restart-backend restart-frontend logs-qdrant logs-backend logs-frontend
.PHONY: notebooks test-agents test-api test-rag test-ingestion generate-dataset

# Default target
help:
	@echo "Legal-OS Development Commands"
	@echo "=============================="
	@echo ""
	@echo "🚀 LOCAL DEVELOPMENT (Docker-free, fast iteration):"
	@echo "  make install-all      - Install all dependencies (backend + frontend)"
	@echo "  make qdrant           - Start Qdrant in Docker (required for local dev)"
	@echo "  make backend          - Start backend locally (port 8000)"
	@echo "  make frontend         - Start frontend locally (port 3000)"
	@echo "  make dev-all          - Start both backend & frontend locally"
	@echo "  make stop-local       - Stop local development processes"
	@echo "  make clean-local      - Clean local caches and build artifacts"
	@echo ""
	@echo "📦 PRODUCTION/DOCKER (Full stack, prod-like testing):"
	@echo "  make start            - Start all services in Docker"
	@echo "  make stop             - Stop all Docker services"
	@echo "  make build            - Build all Docker containers"
	@echo "  make clean            - Stop services and remove volumes"
	@echo "  make restart          - Restart all Docker services"
	@echo ""
	@echo "🔧 INDIVIDUAL SERVICE MANAGEMENT:"
	@echo "  make restart-qdrant   - Restart only Qdrant (Docker)"
	@echo "  make restart-backend  - Restart only backend (Docker)"
	@echo "  make restart-frontend - Restart only frontend (Docker)"
	@echo ""
	@echo "📊 LOGS & MONITORING:"
	@echo "  make logs             - Show logs from all Docker services"
	@echo "  make logs-qdrant      - Show Qdrant logs (Docker)"
	@echo "  make logs-backend     - Show backend logs (Docker)"
	@echo "  make logs-frontend    - Show frontend logs (Docker)"
	@echo ""
	@echo "🧪 TESTING & QUALITY:"
	@echo "  make test             - Run all backend tests"
	@echo "  make test-ingestion   - Run ingestion pipeline tests"
	@echo "  make test-rag         - Run RAG component tests"
	@echo "  make test-agents      - Run agent tests"
	@echo "  make test-api         - Run API endpoint tests"
	@echo "  make backend-test     - Run backend tests locally"
	@echo "  make format           - Format all code"
	@echo "  make lint             - Lint all code"
	@echo ""
	@echo "📓 NOTEBOOKS & DATA:"
	@echo "  make notebooks        - Start Jupyter notebook server"
	@echo "  make generate-dataset - Generate synthetic golden dataset"
	@echo ""
	@echo "💡 QUICK START:"
	@echo "  Local Dev:  make install-all && make qdrant && make dev-all"
	@echo "  Docker:     make build && make start"
	@echo ""

# ============================================================================
# LOCAL DEVELOPMENT COMMANDS (Docker-free)
# ============================================================================

# Install all dependencies
install-all: backend-install frontend-install
	@echo "✅ All dependencies installed!"

# Backend dependency installation
backend-install:
	@echo "📦 Installing backend dependencies with uv..."
	cd backend && uv sync
	@echo "✅ Backend dependencies installed!"

# Frontend dependency installation
frontend-install:
	@echo "📦 Installing frontend dependencies with npm..."
	cd frontend && npm install
	@echo "✅ Frontend dependencies installed!"

# Start Qdrant only (required for local development)
qdrant:
	@echo "🚀 Starting Qdrant in Docker..."
	docker-compose up -d qdrant
	@echo "✅ Qdrant started at http://localhost:6333"
	@echo "📊 Dashboard: http://localhost:6333/dashboard"

# Start backend locally
backend:
	@echo "🚀 Starting backend locally on port 8000..."
	@echo "📍 Backend will be available at http://localhost:8000"
	@echo "📍 API docs at http://localhost:8000/docs"
	@echo ""
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend locally
frontend:
	@echo "🚀 Starting frontend locally on port 3000..."
	@echo "📍 Frontend will be available at http://localhost:3000"
	@echo ""
	cd frontend && npm run dev

# Start both backend and frontend locally
dev-all:
	@echo "🚀 Starting local development environment..."
	@echo "📍 Backend:  http://localhost:8000"
	@echo "📍 Frontend: http://localhost:3000"
	@echo "📍 Qdrant:   http://localhost:6333"
	@echo ""
	@echo "💡 Tip: Run 'make qdrant' first if you haven't already"
	@echo "💡 Press Ctrl+C to stop both services"
	@echo ""
	@trap 'kill 0' EXIT; \
	(cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) & \
	(cd frontend && npm run dev)

# Stop local development processes
stop-local:
	@echo "🛑 Stopping local development processes..."
	@-pkill -f "uvicorn app.main:app" 2>/dev/null || true
	@-pkill -f "next dev" 2>/dev/null || true
	@echo "✅ Local processes stopped"

# Clean local caches and build artifacts
clean-local:
	@echo "🧹 Cleaning local caches and build artifacts..."
	@rm -rf backend/__pycache__
	@rm -rf backend/app/__pycache__
	@rm -rf backend/.pytest_cache
	@rm -rf backend/.ruff_cache
	@rm -rf frontend/.next
	@rm -rf frontend/node_modules/.cache
	@echo "✅ Local caches cleaned"

# Backend testing locally
backend-test:
	@echo "🧪 Running backend tests locally..."
	cd backend && uv run pytest tests/ -v

# Backend linting
backend-lint:
	@echo "🔍 Linting backend code..."
	cd backend && uv run ruff check app/

# Backend formatting
backend-format:
	@echo "✨ Formatting backend code..."
	cd backend && uv run black app/

# Frontend build
frontend-build:
	@echo "🏗️  Building frontend for production..."
	cd frontend && npm run build

# Frontend production start
frontend-start:
	@echo "🚀 Starting frontend production server..."
	cd frontend && npm run start

# Frontend linting
frontend-lint:
	@echo "🔍 Linting frontend code..."
	cd frontend && npm run lint

# ============================================================================
# PRODUCTION/DOCKER COMMANDS (Full stack)
# ============================================================================

# Start all services in Docker
start:
	docker-compose up -d
	@echo "✅ Services started in Docker. Access:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  Qdrant:   http://localhost:6333"

# Stop all Docker services
stop:
	docker-compose down

# Build all Docker containers
build:
	docker-compose build

# Clean up Docker containers and volumes
clean:
	docker-compose down -v
	@echo "✅ Cleaned up Docker containers and volumes"

# Restart all Docker services
restart: stop start

# Show Docker logs
logs:
	docker-compose logs -f

# Individual Docker service restarts
restart-qdrant:
	@echo "🔄 Restarting Qdrant..."
	docker-compose restart qdrant

restart-backend:
	@echo "🔄 Restarting backend..."
	docker-compose restart backend

restart-frontend:
	@echo "🔄 Restarting frontend..."
	docker-compose restart frontend

# Individual Docker service logs
logs-qdrant:
	docker-compose logs -f qdrant

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

# ============================================================================
# TESTING & QUALITY COMMANDS
# ============================================================================

# Run all tests (Docker-based)
test:
	@echo "🧪 Running all backend tests in Docker..."
	docker-compose run --rm backend uv run pytest tests/ -v

# Run ingestion pipeline tests
test-ingestion:
	@echo "🧪 Running ingestion pipeline tests..."
	cd backend && uv run pytest tests/test_ingestion.py -v

# Run RAG component tests
test-rag:
	@echo "🧪 Running RAG component tests..."
	cd backend && uv run pytest tests/test_rag.py tests/test_swappable_retrievers.py -v

# Run agent tests
test-agents:
	@echo "🧪 Running agent tests..."
	cd backend && uv run pytest tests/test_clause_extraction.py tests/test_risk_scoring.py tests/test_summary.py tests/test_checklist.py tests/test_source_tracker.py -v

# Run API endpoint tests
test-api:
	@echo "🧪 Running API endpoint tests..."
	@echo "⚠️  Note: Ensure services are running (make start) before running API tests"
	cd backend && uv run pytest tests/test_api.py tests/test_api_orchestration.py -v

# Format all code
format: backend-format
	@echo "✨ Formatting frontend code..."
	@cd frontend && npm run format 2>/dev/null || echo "⚠️  Add format script to package.json if needed"
	@echo "✅ All code formatted!"

# Lint all code
lint: backend-lint
	@echo "🔍 Linting frontend code..."
	@cd frontend && npm run lint
	@echo "✅ All code linted!"

# ============================================================================
# NOTEBOOKS & DATA COMMANDS
# ============================================================================

# Start Jupyter notebook server
notebooks:
	@echo "📓 Starting Jupyter notebook server..."
	@echo "📍 Notebooks will be available at http://localhost:8888"
	@echo ""
	@echo "Available notebooks:"
	@echo "  E01_Pipeline_Foundation.py - Document ingestion pipeline"
	@echo "  E02_Evaluation_LangChain.py - RAG evaluation"
	@echo "  E03_Clause_Extraction_Agent.py - Clause extraction demo"
	@echo "  E04_Risk_Scoring_Agent.py - Risk scoring demo"
	@echo "  E05_Summary_Agent.py - Summary generation demo"
	@echo "  E06_Source_Tracker.py - Provenance tracking demo"
	@echo "  E07_Checklist_Agent.py - Checklist generation demo"
	@echo "  E08_Orchestration_Demo.py - Full multi-agent orchestration"
	@echo ""
	cd notebooks && jupyter notebook

# Generate synthetic golden dataset
generate-dataset:
	@echo "📊 Generating synthetic golden dataset..."
	@echo "⚠️  Note: Ensure Qdrant is running and documents are ingested"
	cd backend && uv run pytest tests/test_sgd_generation.py -v
	@echo "✅ Dataset generated in backend/golden_dataset/"
