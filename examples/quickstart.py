#!/usr/bin/env python3
"""Minimal VeraRAG quickstart.

Default mode needs no API key. It validates that the package can load bundled
VeraBench data and run a small demo evaluation. Use --run-query when a real LLM
provider is configured through environment variables.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verarag import VeraBenchEvaluator, VeraRAG, load_verabench  # noqa: E402


def _build_index_documents(limit: int = 8) -> list[dict]:
    bench = load_verabench()
    docs = list(bench.corpus.values())[:limit]
    return [
        {
            "id": doc.doc_id,
            "title": doc.title,
            "content": doc.content,
            "metadata": {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "date": doc.date,
                "url": doc.url,
            },
        }
        for doc in docs
    ]


def run_demo(max_questions: int) -> None:
    bench = load_verabench()
    stats = bench.stats()
    print("VeraBench loaded")
    print(f"  documents: {stats['total_documents']}")
    print(f"  questions: {stats['total_questions']}")
    print(f"  types: {stats['questions_by_type']}")

    evaluator = VeraBenchEvaluator(benchmark=bench)
    report = evaluator.evaluate(max_questions=max_questions)
    print("Demo evaluation")
    print(f"  completed: {report.completed}/{report.total_questions}")
    print(f"  answer_f1: {report.overall_answer_f1:.3f}")
    print(f"  behavior_accuracy: {report.behavior_accuracy:.3f}")


def run_query(question: str) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or omit --run-query for no-key demo mode.")

    pipeline = VeraRAG({
        "llm": {
            "provider": "openai",
            "model": os.getenv("VERARAG_MODEL", "gpt-4o-mini"),
            "api_key": "${OPENAI_API_KEY}",
            "temperature": 0.0,
            "max_tokens": 800,
        },
        "pipeline": {
            "max_retrieval_rounds": 2,
            "enable_conflict_graph": True,
            "enable_uncertainty": True,
            "enable_verification": True,
            "enable_repair": True,
        },
    })
    pipeline.index_documents(_build_index_documents())

    result = pipeline.query(question)
    print("Pipeline answer")
    print(f"  answer: {result.answer}")
    print(f"  confidence: {result.confidence:.3f}")
    print(f"  evidence: {result.metadata.get('num_evidence', len(result.evidence))}")
    print(f"  conflicts: {result.metadata.get('num_conflicts', 0)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VeraRAG quickstart")
    parser.add_argument(
        "--max-demo-questions",
        type=int,
        default=3,
        help="Number of VeraBench questions to score in no-key demo mode.",
    )
    parser.add_argument(
        "--run-query",
        action="store_true",
        help="Run a real pipeline query. Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--question",
        default="VeraBench 中的冲突型问题主要考察什么能力？",
        help="Question used with --run-query.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_query:
        run_query(args.question)
    else:
        run_demo(args.max_demo_questions)


if __name__ == "__main__":
    main()
