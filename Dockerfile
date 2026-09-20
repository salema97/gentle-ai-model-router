# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PORT=8377 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies (curl for healthchecks and asset fetching)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies first for layer caching
COPY pyproject.toml README.md ./
RUN uv pip install --system ".[onnx]" transformers

# Copy application source code
COPY src/ ./src/

# Install the application in editable / system mode
RUN uv pip install --system --no-deps -e .

# Prepare directories
RUN mkdir -p data models/modernbert-router/v14

# Download runtime model artifacts and registry if not present
RUN if [ ! -s data/router.db ]; then \
        echo "Downloading router.db from release v0.2.0..." && \
        curl -fSL -o data/router.db https://github.com/salema97/gentle-ai-model-router/releases/download/v0.2.0/router.db ; \
    fi && \
    if [ ! -s models/modernbert-router/v14/model.quant.onnx ]; then \
        echo "Downloading ModernBERT v14 ONNX artifacts from release v0.2.0..." && \
        curl -fSL -o models/modernbert-router/v14/model.quant.onnx https://github.com/salema97/gentle-ai-model-router/releases/download/v0.2.0/model.quant.onnx && \
        curl -fSL -o models/modernbert-router/v14/config.json https://github.com/salema97/gentle-ai-model-router/releases/download/v0.2.0/config.json && \
        curl -fSL -o models/modernbert-router/v14/tokenizer.json https://github.com/salema97/gentle-ai-model-router/releases/download/v0.2.0/tokenizer.json && \
        curl -fSL -o models/modernbert-router/v14/metrics.json https://github.com/salema97/gentle-ai-model-router/releases/download/v0.2.0/metrics.json ; \
    fi

EXPOSE 8377

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8377/health || exit 1

CMD ["router", "serve", "--host", "0.0.0.0", "--port", "8377", "--ranker", "models/modernbert-router/v14/model.quant.onnx"]
