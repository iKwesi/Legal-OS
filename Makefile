.PHONY: help start stop build clean restart logs test format lint
.PHONY: restart-qdrant restart-backend restart-frontend logs-qdrant logs-backend logs-frontend

# Default target
help:
	@echo "Legal-OS Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Service Management:"
	@echo "  make start            - Start all services"
	@echo "  make stop             - Stop all services"
	@echo "  make build            - Build all Docker containers"
	@echo "  make clean            - Stop services and remove volumes"
	@echo "  make restart          - Restart all services"
	@echo "  make restart-qdrant   - Restart only Qdrant"
	@echo "  make restart-backend  - Restart only backend"
	@echo "  make restart-frontend - Restart only frontend"
	@echo ""
	@echo "Logs & Monitoring:"
	@echo "  make logs             - Show logs from all services"
	@echo "  make logs-qdrant      - Show Qdrant logs"
	@echo "  make logs-backend     - Show backend logs"
	@echo "  make logs-frontend    - Show frontend logs"
	@echo ""
	@echo "Development:"
	@echo "  make test             - Run tests"
	@echo "  make format           - Format code"
	@echo "  make lint             - Lint code"
	@echo ""

# Start all services
start:
	docker-compose up -d
	@echo "Services started. Access:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo "  Qdrant:   http://localhost:6333"

# Stop all services
stop:
	docker-compose down

# Build all containers
build:
	docker-compose build

# Clean up containers and volumes
clean:
	docker-compose down -v
	@echo "Cleaned up containers and volumes"

# Restart all services
restart: stop start

# Show logs
logs:
	docker-compose logs -f

# Individual service restarts
restart-qdrant:
	@echo "Restarting Qdrant..."
	docker-compose restart qdrant

restart-backend:
	@echo "Restarting backend..."
	docker-compose restart backend

restart-frontend:
	@echo "Restarting frontend..."
	docker-compose restart frontend

# Individual service logs
logs-qdrant:
	docker-compose logs -f qdrant

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

# Run tests
test:
	@echo "Running backend tests in Docker..."
	docker-compose run --rm backend uv run pytest tests/test_ingestion.py tests/test_rag.py -v
	@echo ""
	@echo "Note: API and main tests require running services (use 'make start' first)"
	@echo "To run all tests including API tests, ensure services are running and use:"
	@echo "  docker-compose exec backend uv run pytest -v"

# Format code
format:
	@echo "Formatting backend code..."
	cd backend && uv run black app/
	@echo "Formatting frontend code..."
	cd frontend && npm run format || echo "Add format script to package.json"

# Lint code
lint:
	@echo "Linting backend code..."
	cd backend && uv run ruff check app/
	@echo "Linting frontend code..."
	cd frontend && npm run lint
