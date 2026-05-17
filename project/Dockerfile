# =============================================================================
# IsoCortex — Multi-Stage Dockerfile
# =============================================================================
# Optimized multi-stage build for the IsoCortex API server.
# SRS Section 14.1: Docker Images
#
# Images:
#   - isocortex-api: Full API server with embedding model (< 2.5 GB)
#
# Build:  docker build -t isocortex-api .
# Run:    docker run -p 8900:8900 -v isocortex-data:/home/isocortex/.isocortex isocortex-api
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — Compile C++ extensions and install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Build dependencies for C++ extensions (pybind11 / HNSW)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY cli/ cli/

# Install PyTorch CPU-only first (separate layer for caching)
RUN pip install --no-cache-dir --extra-index-url \
    https://download.pytorch.org/whl/cpu \
    torch

# Install project with all dependencies
RUN pip install --no-cache-dir .


# ---------------------------------------------------------------------------
# Stage 2: Runtime — Minimal production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r isocortex && useradd -r -g isocortex -d /app -s /sbin/nologin isocortex

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --from=builder /build/src/ /app/src/
COPY --from=builder /build/cli/ /app/cli/

# Copy pyproject.toml for version info
COPY --from=builder /build/pyproject.toml /app/pyproject.toml

# Create data directory with correct ownership
RUN mkdir -p /home/isocortex/.isocortex/{indices,models} && chown -R isocortex:isocortex /app /home/isocortex

# Environment variables (SRS Section 14.3)
ENV ISOCORTEX_HOST=0.0.0.0 \
    ISOCORTEX_PORT=8900 \
    ISOCORTEX_WORKERS=1 \
    ISOCORTEX_LOG_LEVEL=INFO \
    ISOCORTEX_DATA_DIR=/home/isocortex/.isocortex \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Expose API port
EXPOSE 8900

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8900/health')" || exit 1

# Run as non-root user
USER isocortex

# Start server (SRS Section 14.1)
CMD ["uvicorn", "isocortex.api:app", "--host", "0.0.0.0", "--port", "8900"]
