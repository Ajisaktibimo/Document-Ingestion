import asyncio
import sys
import argparse
import logging
from docai.ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def main():
    parser = argparse.ArgumentParser(description="Ingest a document into the system (Track A)")
    parser.add_argument("file_path", help="Path to the PDF or image file")
    parser.add_argument("--doc-class", default="general", help="Document classification category")
    
    args = parser.parse_args()
    
    pipeline = IngestionPipeline()
    doc_id = await pipeline.ingest_document(args.file_path, args.doc_class)
    
    if doc_id:
        print(f"\n✅ Successfully ingested document. ID: {doc_id}")
    else:
        print("\n❌ Ingestion failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
