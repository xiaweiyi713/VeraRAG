#!/usr/bin/env python3
"""Filter a saved VeraBench report to a fixed question-id set and rescore it."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.rescore_verabench import rescore_report  # noqa: E402
from experiments.run_verabench import _read_question_ids_file  # noqa: E402


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def filter_report(
    report: dict[str, Any],
    question_ids: list[str],
    *,
    data_dir: str | None = None,
    allow_benchmark_mismatch: bool = False,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Return a rescored report containing exactly ``question_ids`` rows."""
    if not question_ids:
        raise ValueError("question id list must not be empty")
    requested = _dedupe_preserving_order(question_ids)
    rows = report.get("question_results") or report.get("results") or []
    duplicates = sorted(
        question_id
        for question_id, count in Counter(
            str(row.get("question_id") or "") for row in rows
        ).items()
        if question_id and count > 1
    )
    if duplicates:
        raise ValueError(
            "source report contains duplicate question IDs: "
            + ", ".join(duplicates)
        )
    by_id = {
        str(row.get("question_id") or ""): row
        for row in rows
        if str(row.get("question_id") or "")
    }
    missing = [question_id for question_id in requested if question_id not in by_id]
    if missing:
        raise ValueError(
            "source report is missing requested question IDs: "
            + ", ".join(missing)
        )

    filtered = dict(report)
    filtered["question_results"] = [by_id[question_id] for question_id in requested]
    filtered.pop("results", None)
    metadata = dict(filtered.get("metadata") or {})
    metadata["question_ids"] = requested
    metadata["filtered_offline"] = True
    filtered["metadata"] = metadata
    rescored = rescore_report(
        filtered,
        data_dir=data_dir,
        allow_benchmark_mismatch=allow_benchmark_mismatch,
        allow_unverified=allow_unverified,
    )
    rescored_metadata = dict(rescored.get("metadata") or {})
    rescored_metadata["question_ids"] = requested
    rescored_metadata["filtered_offline"] = True
    rescored["metadata"] = rescored_metadata
    return rescored


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Saved run_verabench.py report JSON")
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--ids", nargs="+", help="Question ids to keep")
    ids.add_argument("--ids-file", help="Question-id file to keep")
    parser.add_argument("--output", required=True, help="Path for filtered report JSON")
    parser.add_argument("--data-dir", default=None, help="Optional VeraBench data directory")
    parser.add_argument(
        "--allow-benchmark-mismatch",
        action="store_true",
        help="Allow explicitly labeled historical sensitivity filtering.",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow a legacy source report missing benchmark fingerprints.",
    )
    args = parser.parse_args(argv)

    question_ids = args.ids or _read_question_ids_file(args.ids_file)
    with open(args.input, encoding="utf-8") as handle:
        report = json.load(handle)
    try:
        filtered = filter_report(
            report,
            question_ids,
            data_dir=args.data_dir,
            allow_benchmark_mismatch=args.allow_benchmark_mismatch,
            allow_unverified=args.allow_unverified,
        )
    except ValueError as error:
        parser.error(str(error))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Filtered {filtered['completed']} questions: {output}")


if __name__ == "__main__":
    main()
