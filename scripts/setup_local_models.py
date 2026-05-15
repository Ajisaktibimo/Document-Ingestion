import os
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_sentence_transformer(model_id: str, sub_dir: str):
    target_path = MODELS_DIR / sub_dir
    if target_path.exists():
        logger.info(f"Model {model_id} already exists in {target_path}")
        return str(target_path)
    
    logger.info(f"Downloading {model_id} to {target_path}...")
    snapshot_download(repo_id=model_id, local_dir=str(target_path), local_dir_use_symlinks=False)
    return str(target_path)

def setup_mineru_config(models_dir: Path):
    """Generates a magic-pdf.json for local execution."""
    config_path = models_dir / "magic-pdf.json"
    
    # Typical MinerU model structure
    # You would normally download these from MinerU's model zoo
    # For now, we point to the models directory
    config = {
        "models-dir": str(models_dir.absolute()),
        "device-mode": "cpu", # or "cuda"
        "formula-config": {
            "mfr-model": "mfr_v1.0",
            "det-model": "mfd_v1.0"
        },
        "layout-config": {
            "model": "layout_v1.1"
        }
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Generated MinerU local config at {config_path}")
    logger.info("IMPORTANT: You still need to manually download MinerU weights into the models directory.")

def main():
    # 1. Download Dense Embedder
    dense_path = download_sentence_transformer("nomic-ai/nomic-embed-text-v2-moe", "nomic-embed")
    
    # 2. Download Reranker
    rerank_path = download_sentence_transformer("BAAI/bge-reranker-v2-m3", "bge-reranker")
    
    # 3. Setup MinerU Config
    setup_mineru_config(MODELS_DIR)
    
    logger.info("\n--- Truly Local Setup Complete ---")
    logger.info("To run in offline mode, update your .env file:")
    logger.info(f"DENSE_MODEL_PATH={dense_path}")
    logger.info(f"RERANKER_MODEL_PATH={rerank_path}")
    logger.info("And ensures your internet is disconnected.")

if __name__ == "__main__":
    main()
