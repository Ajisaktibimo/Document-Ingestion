FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies (Cached)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy dependency files and source needed for optional extras build
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY README.md ./README.md

# 2. Install core dependencies (Cached unless requirements change)
RUN uv pip install --system --no-cache -r requirements.txt

# 3. Use ARG for parser selection (Only invalidates cache if YOU change the parser)
ARG DOCUMENT_PARSER=docling
ARG DEVICE=cpu
ARG DENSE_EMBEDDING_CPU_MODEL=nomic-ai/nomic-embed-text-v1.5
ARG DENSE_EMBEDDING_BACKEND=onnx

# 4. Integrated Parser Installation
# We 'lock' the system torch version first to prevent uv from re-downloading it
RUN pip freeze | grep -E "torch|nvidia" > /tmp/constraints.txt && \
    echo "Integrated Build: Installing extra [$DOCUMENT_PARSER] on Device: [$DEVICE]..." && \
    if [ "$DOCUMENT_PARSER" = "paddle" ] && [ "$DEVICE" = "gpu" ]; then \
    uv pip install --system --no-cache -c /tmp/constraints.txt ".[paddle-gpu]"; \
    else \
    uv pip install --system --no-cache -c /tmp/constraints.txt ".[$DOCUMENT_PARSER]"; \
    fi
# Skip megablocks install during build (GPU driver not available)
# Will be installed at runtime if needed via post-build script

# 5. Pre-download dense ONNX embedding model into /opt/dense_cache at build time.
#    /opt is outside /app so the .:/app bind-mount in docker-compose cannot
#    shadow this layer at runtime — rebuilding always produces a working image.
COPY scripts/warm_dense_embedder.py /tmp/warm_dense_embedder.py
RUN DENSE_EMBEDDING_CPU_MODEL=${DENSE_EMBEDDING_CPU_MODEL} \
    DENSE_EMBEDDING_BACKEND=${DENSE_EMBEDDING_BACKEND} \
    python /tmp/warm_dense_embedder.py

# 6. Pre-download sparse ONNX embedding model into /opt/sparse_cache at build time.
#    Same reasoning as step 5 — outside /app so the bind-mount cannot shadow it.
COPY scripts/warm_sparse_embedder.py /tmp/warm_sparse_embedder.py
RUN python /tmp/warm_sparse_embedder.py

# 8. Pre-download RapidOCR ONNX models at build time so the container is
#    offline-capable after first build. Copied separately from app source so
#    this layer stays cached even when source changes.
COPY scripts/warm_rapidocr.py /tmp/warm_rapidocr.py
RUN python /tmp/warm_rapidocr.py || true

# 9. Copy application source (Done LAST to protect the heavy layers above)
COPY . .

# Install as an editable package so scripts (api, mcp) are registered
RUN uv pip install --system --no-cache -e .

ENV PYTHONPATH=/app:/app/src
