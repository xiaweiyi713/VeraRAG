#!/usr/bin/env python3
"""Recompute VeraBench answer and aggregate metrics from a saved report."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.benchmark.evaluator import QuestionResult, VeraBenchEvaluator  # noqa: E402
from src.benchmark.loader import VERABENCH_VERSION, VeraBenchLoader  # noqa: E402
from src.evaluation.answer_metrics import AnswerMetrics  # noqa: E402


def _load_rows(report: dict[str, Any]) -> list[QuestionResult]:
    valid = {item.name for item in fields(QuestionResult)}
    rows = report.get("question_results") or report.get("results") or []
    return [
        QuestionResult(**{key: value for key, value in row.items() if key in valid})
        for row in rows
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_metadata(loader: VeraBenchLoader) -> dict[str, Any]:
    return {
        "version": VERABENCH_VERSION,
        "fingerprints": {
            "corpus_sha256": _sha256(loader.data_dir / "corpus.jsonl"),
            "questions_sha256": _sha256(loader.data_dir / "questions.jsonl"),
        },
    }


def _benchmark_identity(metadata: dict[str, Any]) -> tuple[str, str, str]:
    fingerprints = metadata.get("fingerprints") or {}
    return (
        str(metadata.get("version") or ""),
        str(fingerprints.get("corpus_sha256") or ""),
        str(fingerprints.get("questions_sha256") or ""),
    )


def rescore_report(
    report: dict[str, Any],
    *,
    data_dir: str | None = None,
    allow_benchmark_mismatch: bool = False,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    loader = VeraBenchLoader(data_dir)
    benchmark = loader.load()
    target_benchmark = _benchmark_metadata(loader)
    source_metadata = dict(report.get("metadata") or {})
    source_benchmark = dict(source_metadata.get("benchmark") or {})
    source_identity = _benchmark_identity(source_benchmark)
    target_identity = _benchmark_identity(target_benchmark)
    if not all(source_identity) and not allow_unverified:
        raise ValueError(
            "Source report is missing benchmark version/fingerprints; "
            "pass --allow-unverified only for explicitly labeled legacy reports"
        )
    benchmark_mismatch = all(source_identity) and source_identity != target_identity
    if benchmark_mismatch and not allow_benchmark_mismatch:
        raise ValueError(
            "Source report benchmark does not match the rescore benchmark; "
            "pass --allow-benchmark-mismatch only for explicitly labeled "
            "historical sensitivity analysis"
        )

    rows = _load_rows(report)
    question_ids = [row.question_id for row in rows]
    duplicates = sorted(
        question_id
        for question_id, count in Counter(question_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(
            "Source report contains duplicate question IDs: "
            + ", ".join(duplicates)
        )
    known_ids = {question.id for question in benchmark.questions}
    unknown_ids = sorted(set(question_ids) - known_ids)
    if unknown_ids:
        raise ValueError(
            "Source report contains question IDs absent from rescore benchmark: "
            + ", ".join(unknown_ids)
        )

    evaluator = VeraBenchEvaluator(benchmark=benchmark)
    rescored = evaluator.rescore_results(rows).to_dict()
    metadata = source_metadata
    metric_versions = dict(metadata.get("metric_versions") or {})
    metric_versions["answer"] = AnswerMetrics.VERSION
    metric_versions["behavior"] = VeraBenchEvaluator.BEHAVIOR_METRIC_VERSION
    metric_versions["conflict"] = VeraBenchEvaluator.CONFLICT_METRIC_VERSION
    metadata["metric_versions"] = metric_versions
    metadata["rescored_offline"] = True
    metadata["rescored_against_benchmark"] = target_benchmark
    if benchmark_mismatch:
        metadata["source_benchmark"] = source_benchmark
        metadata["benchmark_mismatch_allowed"] = True
    if not all(source_identity):
        metadata["benchmark_verification"] = "source_metadata_missing"
    rescored["metadata"] = metadata
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Saved run_verabench.py report JSON")
    parser.add_argument("--output", required=True, help="Path for the rescored report")
    parser.add_argument("--data-dir", default=None, help="Optional VeraBench data directory")
    parser.add_argument(
        "--allow-benchmark-mismatch",
        action="store_true",
        help=(
            "Allow explicitly labeled historical sensitivity rescoring against "
            "a different benchmark fingerprint."
        ),
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow a legacy source report missing benchmark fingerprints.",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        report = json.load(handle)
    try:
        rescored = rescore_report(
            report,
            data_dir=args.data_dir,
            allow_benchmark_mismatch=args.allow_benchmark_mismatch,
            allow_unverified=args.allow_unverified,
        )
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rescored, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Rescored {rescored['completed']} questions with "
        f"{AnswerMetrics.VERSION}: {output}"
    )


if __name__ == "__main__":
    main()
