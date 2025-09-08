# Multi-stage Dockerfile for HailHero MVP Flask Application
# Stage 1: Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt requirements_enhanced.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements_enhanced.txt

# Stage 2: Production stage
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FLASK_APP=/app/src/mvp/app.py \
    FLASK_ENV=production \
    PYTHONPATH=/app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p /app/src/mvp /app/data /app/logs /app/uploads && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser specs/ ./specs/

# Create necessary directories and files
RUN mkdir -p /app/specs/001-hail-hero-hail/data && \
    touch /app/specs/001-hail-hero-hail/data/leads.jsonl && \
    chown -R appuser:appuser /app/specs/001-hail-hero-hail/data

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Expose port
EXPOSE 5000

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for any dependent services (Redis, etc.)\n\
echo "Starting HailHero MVP application..."\n\
\n\
# Initialize data directory if empty\n\
if [ ! -s "/app/specs/001-hail-hero-hail/data/leads.jsonl" ]; then\n\
    echo "Initializing empty leads database..."\n\
    touch /app/specs/001-hail-hero-hail/data/leads.jsonl\n\
fi\n\
\n\
# Start the application\n\
exec python -m flask run --host=0.0.0.0 --port=5000' > /app/start.sh && \
chmod +x /app/start.sh

# Start the application
CMD ["/app/start.sh"]