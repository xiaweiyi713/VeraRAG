#!/usr/bin/env python3
"""Build search indexes from documents.

Usage:
    # Index VeraBench corpus with BM25
    python experiments/build_index.py --source data/verabench/corpus.jsonl --type bm25

    # Index a directory of documents
    python experiments/build_index.py --source data/raw/ --type bm25 --dir

    # Index with hybrid retriever (BM25 + Dense)
    python experiments/build_index.py --source data/verabench/corpus.jsonl --type hybrid

    # Custom chunk size
    python experiments/build_index.py --source data/verabench/corpus.jsonl --type bm25 --chunk-size 256

    # Save index to disk
    python experiments/build_index.py --source data/verabench/corpus.jsonl --type bm25 --output data/indexes/verabench
"""

import argparse
import logging
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_index")


def main():
    parser = argparse.ArgumentParser(description="Build search indexes from documents")
    parser.add_argument("--source", required=True, help="Source file or directory path")
    parser.add_argument("--type", default="bm25", choices=["bm25", "dense", "faiss", "hybrid"],
                       help="Retriever type (default: bm25)")
    parser.add_argument("--dir", action="store_true", help="Source is a directory")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap in characters")
    parser.add_argument("--chunk-strategy", default="fixed",
                       choices=["fixed", "sentence", "paragraph", "heading"],
                       help="Chunking strategy")
    parser.add_argument("--output", help="Output directory for saved index")
    parser.add_argument("--query", help="Test query after indexing")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results for test query")
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_strategy=args.chunk_strategy,
    )

    logger.info(f"Loading from: {args.source}")
    chunks, retriever = pipeline.ingest_and_index(
        source_path=args.source,
        retriever_type=args.type,
        is_directory=args.dir,
    )

    print(f"\nIndexed {len(chunks)} chunks from {args.source}")
    print(f"  Retriever: {args.type}")
    print(f"  Chunk strategy: {args.chunk_strategy} (size={args.chunk_size}, overlap={args.chunk_overlap})")

    # Show sample chunks
    if chunks:
        print(f"\n  Sample chunks:")
        for c in chunks[:3]:
            preview = c.text[:80].replace("\n", " ")
            print(f"    [{c.chunk_id}] {preview}...")

    # Test query
    if args.query:
        print(f"\n  Query: {args.query}")
        results = retriever.retrieve(args.query, top_k=args.top_k)
        print(f"  Results ({len(results)}):")
        for r in results:
            preview = r.content[:100].replace("\n", " ")
            print(f"    [{r.doc_id}] score={r.score:.4f} | {preview}...")

    # Save index
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        retriever.save_index(str(output_path / "index"))
        print(f"\n  Index saved to {output_path}")

        # Save chunk metadata
        meta_path = output_path / "chunks.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title,
                  "text_len": len(c.text), "source": c.metadata.get("source", "")}
                 for c in chunks],
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Chunk metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
