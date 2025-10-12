# Production Dockerfile for Seraph MCP
# SDD-compliant: stdio MCP protocol server
FROM python:3.11-slim AS builder

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies (production only)
RUN uv sync --frozen --no-dev

# ============================================================================
# Runtime stage
# ============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app fastmcp.json ./
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app .env.example ./

# Switch to non-root user
USER app

# No exposed ports (stdio MCP protocol, not HTTP)
# MCP communication happens over stdin/stdout

# No healthcheck (stdio protocol has no HTTP endpoint)

# Start the MCP server via fastmcp CLI
CMD ["fastmcp", "run", "fastmcp.json"]
