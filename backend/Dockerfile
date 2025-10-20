# Use Python 3.13 image for better security and compatibility
FROM python:3.13

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies using uv
# The full python:3.13 image has all necessary build tools
RUN uv sync --no-cache

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application using uv run
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
