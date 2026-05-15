"""
Build-time warmup script: pre-downloads RapidOCR ONNX models so the
container is offline-capable after `docker build`.

Docling auto-selects rapidocr with onnxruntime as its OCR backend.
Running inference on a blank image forces the model download and caches
the model files inside the image layer.
"""
import sys
import numpy as np

for pkg in ("rapidocr_onnxruntime", "rapidocr"):
    try:
        import importlib
        m = importlib.import_module(pkg)
        ocr = m.RapidOCR()
        ocr(np.zeros((50, 200, 3), dtype="uint8"))
        print(f"[build] RapidOCR models pre-cached via {pkg}", flush=True)
        break
    except Exception as e:
        print(f"[build] {pkg} unavailable: {e}", file=sys.stderr)
else:
    print("[build] RapidOCR warmup skipped — neither package found.", file=sys.stderr)
