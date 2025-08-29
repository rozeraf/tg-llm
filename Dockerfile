# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/logs && \
    chown -R botuser:botuser /app

# Copy application code
COPY --chown=botuser:botuser main.py .
COPY --chown=botuser:botuser config/ ./config/
COPY --chown=botuser:botuser scripts/ ./scripts/

# Copy health check script
COPY --chown=botuser:botuser <<EOF /app/healthcheck.py
#!/usr/bin/env python3
import sys
import os
import psycopg2
from urllib.parse import urlparse

def check_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        conn.close()
        return True
    except:
        return False

def check_bot():
    # Add bot-specific health checks here
    return os.path.exists("/app/main.py")

if __name__ == "__main__":
    if not check_db():
        print("Database health check failed")
        sys.exit(1)
    
    if not check_bot():
        print("Bot health check failed")
        sys.exit(1)
    
    print("All health checks passed")
    sys.exit(0)
EOF

RUN chmod +x /app/healthcheck.py

# Switch to non-root user
USER botuser

# Expose health check port (optional)
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TZ=UTC

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Command to run the application
CMD ["python", "main.py"]