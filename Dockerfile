FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better caching
COPY src/pyproject.toml src/uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
COPY LICENSE ./

WORKDIR /app/src

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

# Run
CMD ["uv", "run", "voice-agent"]
