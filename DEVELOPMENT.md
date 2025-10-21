# Development Guide

## Quick Start

### Local Development (Recommended for Fast Iteration)

```bash
# First time setup
make install-all          # Install all dependencies
make qdrant              # Start Qdrant (required)

# Start development
make dev-all             # Start both backend & frontend

# Or start individually
make backend             # Backend only (port 8000)
make frontend            # Frontend only (port 3000)

# Stop services
make stop-local          # Stop local processes
```

### Docker/Production Testing

```bash
# Build and start everything
make build
make start

# Stop everything
make stop

# Clean up
make clean
```

## Available Commands

### 🚀 Local Development (Docker-free)

| Command | Description |
|---------|-------------|
| `make install-all` | Install all dependencies (backend + frontend) |
| `make qdrant` | Start Qdrant in Docker (required for local dev) |
| `make backend` | Start backend locally (port 8000) |
| `make frontend` | Start frontend locally (port 3000) |
| `make dev-all` | Start both backend & frontend locally |
| `make stop-local` | Stop local development processes |
| `make clean-local` | Clean local caches and build artifacts |

### 📦 Production/Docker

| Command | Description |
|---------|-------------|
| `make start` | Start all services in Docker |
| `make stop` | Stop all Docker services |
| `make build` | Build all Docker containers |
| `make clean` | Stop services and remove volumes |
| `make restart` | Restart all Docker services |

### 🔧 Individual Service Management

| Command | Description |
|---------|-------------|
| `make restart-qdrant` | Restart only Qdrant (Docker) |
| `make restart-backend` | Restart only backend (Docker) |
| `make restart-frontend` | Restart only frontend (Docker) |

### 📊 Logs & Monitoring

| Command | Description |
|---------|-------------|
| `make logs` | Show logs from all Docker services |
| `make logs-qdrant` | Show Qdrant logs (Docker) |
| `make logs-backend` | Show backend logs (Docker) |
| `make logs-frontend` | Show frontend logs (Docker) |

### 🧪 Testing & Quality

| Command | Description |
|---------|-------------|
| `make test` | Run backend tests (Docker) |
| `make backend-test` | Run backend tests locally |
| `make format` | Format all code |
| `make lint` | Lint all code |
| `make backend-lint` | Lint backend only |
| `make frontend-lint` | Lint frontend only |

## Service URLs

When running locally or in Docker:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant**: http://localhost:6333

## Development Workflow

### Typical Local Development Session

```bash
# 1. Start Qdrant (only needed once)
make qdrant

# 2. Start development servers
make dev-all

# 3. Make changes to code (auto-reload enabled)

# 4. Run tests when needed
make backend-test

# 5. Stop when done
make stop-local
```

### Testing with Docker (Production-like)

```bash
# 1. Build containers
make build

# 2. Start all services
make start

# 3. View logs
make logs

# 4. Run tests
make test

# 5. Stop and clean up
make stop
make clean
```

## Troubleshooting

### Port Already in Use

If you get port conflicts:

```bash
# Stop local processes
make stop-local

# Or stop Docker services
make stop
```

### Clean Start

For a completely fresh start:

```bash
# Clean local
make clean-local

# Clean Docker
make clean

# Reinstall dependencies
make install-all
```

### Dependencies Not Found

```bash
# Reinstall everything
make install-all

# Or individually
make backend-install
make frontend-install
```

## Tips

- **Local development** is faster for iteration (no Docker overhead)
- **Docker** is better for testing the full stack and production-like behavior
- Use `make help` anytime to see all available commands
- Press `Ctrl+C` to stop `make dev-all` (stops both services)
- Qdrant must be running for backend to work (use `make qdrant`)
