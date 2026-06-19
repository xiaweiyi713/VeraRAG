#!/usr/bin/env python3
"""Merge compatible VeraBench partition reports and recompute aggregates."""

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.rescore_verabench import rescore_report  # noqa: E402
from src.benchmark.loader import load_verabench  # noqa: E402

_COMPATIBILITY_PATHS = (
    ("benchmark", "version"),
    ("benchmark", "fingerprints", "corpus_sha256"),
    ("benchmark", "fingerprints", "questions_sha256"),
    ("run_signature", "implementation_sha256"),
    ("run_signature", "config_sha256"),
    ("metric_versions", "answer"),
    ("provider",),
    ("model",),
)


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("question_results") or report.get("results") or []
    if not isinstance(rows, list):
        raise ValueError("Report question_results must be a list")
    return rows


def _validate_compatibility(reports: list[dict[str, Any]]) -> None:
    reference = reports[0].get("metadata") or {}
    for index, report in enumerate(reports[1:], start=2):
        metadata = report.get("metadata") or {}
        mismatches = [
            ".".join(path)
            for path in _COMPATIBILITY_PATHS
            if _nested_value(metadata, path) != _nested_value(reference, path)
        ]
        if mismatches:
            raise ValueError(
                f"Report {index} is incompatible with report 1: "
                f"{', '.join(mismatches)}"
            )


def merge_reports(
    reports: list[dict[str, Any]],
    *,
    labels: list[str] | None = None,
    data_dir: str | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Merge disjoint reports, validate provenance, and rescore all rows."""
    if len(reports) < 2:
        raise ValueError("At least two reports are required")
    if labels is not None and len(labels) != len(reports):
        raise ValueError("labels must match the number of reports")

    _validate_compatibility(reports)
    benchmark = load_verabench(data_dir)
    benchmark_ids = {question.id for question in benchmark.questions}

    merged_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for report_index, report in enumerate(reports, start=1):
        for row in _rows(report):
            question_id = row.get("question_id")
            if not question_id:
                raise ValueError(
                    f"Report {report_index} contains a row without question_id"
                )
            if question_id in seen:
                raise ValueError(f"Duplicate question_id across reports: {question_id}")
            if question_id not in benchmark_ids:
                raise ValueError(f"Unknown VeraBench question_id: {question_id}")
            seen.add(question_id)
            merged_rows.append(row)

    if require_complete and seen != benchmark_ids:
        missing = sorted(benchmark_ids - seen)
        raise ValueError(
            f"Merged reports are incomplete: {len(seen)}/{len(benchmark_ids)} "
            f"questions; missing {', '.join(missing[:10])}"
        )

    report_labels = labels or [
        f"report-{index}" for index in range(1, len(reports) + 1)
    ]
    metadata = deepcopy(reports[0].get("metadata") or {})
    metadata["mode"] = "partitioned_pipeline"
    metadata["timestamp"] = datetime.now(timezone.utc).astimezone().isoformat()
    metadata["question_types"] = sorted({
        str(row.get("question_type", "unknown"))
        for row in merged_rows
    })
    metadata["max_questions"] = len(merged_rows)
    metadata["partition_reports"] = [
        {
            "label": report_labels[index],
            "completed": report.get("completed"),
            "errors": report.get("errors"),
            "question_types": (report.get("metadata") or {}).get("question_types"),
            "timestamp": (report.get("metadata") or {}).get("timestamp"),
            "run_signature": (report.get("metadata") or {}).get("run_signature"),
        }
        for index, report in enumerate(reports)
    ]
    run_signature = deepcopy(metadata.get("run_signature") or {})
    run_signature["question_types"] = metadata["question_types"]
    run_signature["max_questions"] = len(merged_rows)
    run_signature["partitioned"] = True
    metadata["run_signature"] = run_signature

    merged_rows.sort(key=lambda row: str(row["question_id"]))
    return rescore_report(
        {
            "metadata": metadata,
            "question_results": merged_rows,
        },
        data_dir=data_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Partition report JSON files")
    parser.add_argument("--output", required=True, help="Merged report JSON path")
    parser.add_argument("--data-dir", default=None, help="Optional VeraBench data directory")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require the merged reports to cover every benchmark question",
    )
    args = parser.parse_args()

    reports = []
    for input_path in args.inputs:
        with open(input_path, encoding="utf-8") as handle:
            reports.append(json.load(handle))
    merged = merge_reports(
        reports,
        labels=[str(Path(path)) for path in args.inputs],
        data_dir=args.data_dir,
        require_complete=args.require_complete,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Merged {len(reports)} reports into {merged['completed']} questions: "
        f"{output}"
    )


if __name__ == "__main__":
    main()
