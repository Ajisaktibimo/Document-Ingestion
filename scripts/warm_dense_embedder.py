"""
Build-time warmup script: pre-downloads the fastembed ONNX embedding model
so the container is self-contained after `docker build`.

Downloads to /opt/dense_cache — intentionally outside /app so the
  .:/app  bind-mount in docker-compose cannot shadow it at runtime.

Skipped automatically when DENSE_EMBEDDING_BACKEND != onnx (e.g. torch/openvino
for GPU builds), since those backends use the HuggingFace cache instead.
"""
import os
import sys

BACKEND = os.environ.get("DENSE_EMBEDDING_BACKEND", "onnx")
CACHE_DIR = "/opt/dense_cache"
MODEL = os.environ.get("DENSE_EMBEDDING_CPU_MODEL", "nomic-ai/nomic-embed-text-v1.5")

if BACKEND != "onnx":
    print(
        f"[build] DENSE_EMBEDDING_BACKEND={BACKEND} — fastembed warmup not needed, skipping.",
        flush=True,
    )
    sys.exit(0)

try:
    from fastembed import TextEmbedding
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[build] Pre-downloading fastembed ONNX model: {MODEL} → {CACHE_DIR}", flush=True)
    TextEmbedding(model_name=MODEL, cache_dir=CACHE_DIR)
    print("[build] Dense embedding ONNX model ready.", flush=True)
except Exception as e:
    print(f"[build] Dense embedding warmup failed: {e}", file=sys.stderr)
    sys.exit(1)
