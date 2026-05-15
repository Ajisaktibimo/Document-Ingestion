"""
Build-time warmup script: pre-downloads the fastembed sparse ONNX model
so the container is self-contained after `docker build`.

Downloads to /opt/sparse_cache — intentionally outside /app so the
  .:/app  bind-mount in docker-compose cannot shadow it at runtime.
"""
import os
import sys

CACHE_DIR = "/opt/sparse_cache"
MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"

try:
    from fastembed import SparseTextEmbedding
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[build] Pre-downloading sparse ONNX model: {MODEL} → {CACHE_DIR}", flush=True)
    SparseTextEmbedding(model_name=MODEL, cache_dir=CACHE_DIR)
    print("[build] Sparse embedding ONNX model ready.", flush=True)
except Exception as e:
    print(f"[build] Sparse embedding warmup failed: {e}", file=sys.stderr)
    sys.exit(1)
