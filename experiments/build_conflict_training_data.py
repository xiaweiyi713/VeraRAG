#!/usr/bin/env python3
"""Build pairwise conflict-detection training data from VeraBench."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.conflict_pairs import (  # noqa: E402
    build_conflict_pair_examples,
    summarize_conflict_pair_examples,
    write_conflict_pair_dataset,
)
from src.benchmark.loader import VERABENCH_VERSION, VeraBenchLoader  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VeraBench conflict-pair training data")
    parser.add_argument("--data-dir", help="Optional VeraBench data directory")
    parser.add_argument("--output-dir", default="outputs/conflict_pairs", help="Directory for train/val/test JSONL")
    parser.add_argument(
        "--max-negative-per-question",
        type=int,
        default=4,
        help="Maximum non-conflict evidence pairs sampled per question. Use -1 for all.",
    )
    parser.add_argument(
        "--max-hard-negative-per-question",
        type=int,
        default=2,
        help="Maximum cross-question topical hard negatives per question.",
    )
    parser.add_argument(
        "--max-weak-positive-per-question",
        type=int,
        default=2,
        help="Maximum heuristic conflict positives per question. Use 0 to disable.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    max_negative = None if args.max_negative_per_question < 0 else args.max_negative_per_question
    loader = VeraBenchLoader(args.data_dir)
    benchmark = loader.load()
    examples = build_conflict_pair_examples(
        benchmark=benchmark,
        max_negative_per_question=max_negative,
        max_hard_negative_per_question=args.max_hard_negative_per_question,
        max_weak_positive_per_question=args.max_weak_positive_per_question,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    paths = write_conflict_pair_dataset(examples, args.output_dir)
    summary = summarize_conflict_pair_examples(examples)
    metadata_path = paths["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["benchmark"] = {
        "version": VERABENCH_VERSION,
        "fingerprints": {
            "corpus_sha256": _sha256(loader.data_dir / "corpus.jsonl"),
            "questions_sha256": _sha256(loader.data_dir / "questions.jsonl"),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = metadata
    print(json.dumps({
        "summary": summary,
        "paths": {split: str(path) for split, path in paths.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
