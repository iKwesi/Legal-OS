.PHONY: help start stop build clean restart logs test format lint

# Default target
help:
	@echo "Legal-OS Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  make start      - Start all services with docker-compose up"
	@echo "  make stop       - Stop all services with docker-compose down"
	@echo "  make build      - Build all Docker containers"
	@echo "  make clean      - Stop services and remove volumes"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - Show logs from all services"
	@echo "  make test       - Run tests for backend and frontend"
	@echo "  make format     - Format code (backend: black, frontend: prettier)"
	@echo "  make lint       - Lint code (backend: ruff, frontend: eslint)"
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

# Run tests
test:
	@echo "Running backend tests..."
	cd backend && uv run pytest
	@echo "Running frontend tests..."
	cd frontend && npm test

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
