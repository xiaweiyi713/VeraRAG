#!/usr/bin/env python3
"""Summarize VeraBench checkpoint and report progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({
                    "question_id": f"malformed-line-{line_number}",
                    "error": f"Malformed checkpoint JSON: {exc}",
                })
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                rows.append({
                    "question_id": f"malformed-line-{line_number}",
                    "error": "Checkpoint line is not a JSON object",
                })
    return rows


def _error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("error")]


def summarize_status(
    *,
    checkpoint_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a compact progress summary for a VeraBench run."""
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint is not None:
        checkpoint_rows = _read_jsonl(checkpoint)
    checkpoint_errors = _error_rows(checkpoint_rows)

    report_summary: dict[str, Any] | None = None
    report = Path(report_path) if report_path else None
    if report is not None and report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        report_rows = data.get("question_results") or data.get("results") or []
        report_errors = [
            row for row in report_rows
            if isinstance(row, dict) and row.get("error")
        ]
        report_summary = {
            "path": str(report),
            "exists": True,
            "total_questions": data.get("total_questions"),
            "completed": data.get("completed"),
            "errors": data.get("errors"),
            "row_errors": len(report_errors),
            "error_question_ids": [
                row.get("question_id") or row.get("id") or ""
                for row in report_errors
            ],
        }
    elif report is not None:
        report_summary = {"path": str(report), "exists": False}

    return {
        "checkpoint": (
            {
                "path": str(checkpoint),
                "exists": checkpoint.exists(),
                "rows": len(checkpoint_rows),
                "errors": len(checkpoint_errors),
                "last_question_id": (
                    checkpoint_rows[-1].get("question_id")
                    or checkpoint_rows[-1].get("id")
                    if checkpoint_rows else None
                ),
                "last_question_type": (
                    checkpoint_rows[-1].get("question_type")
                    or checkpoint_rows[-1].get("type")
                    if checkpoint_rows else None
                ),
                "error_question_ids": [
                    row.get("question_id") or row.get("id") or ""
                    for row in checkpoint_errors
                ],
            }
            if checkpoint is not None
            else None
        ),
        "report": report_summary,
    }


def render_text(summary: dict[str, Any]) -> str:
    """Render a terse human-readable status summary."""
    lines: list[str] = []
    checkpoint = summary.get("checkpoint")
    if checkpoint:
        lines.append(
            "checkpoint: "
            f"rows={checkpoint['rows']} errors={checkpoint['errors']} "
            f"last={checkpoint['last_question_id'] or 'n/a'}"
        )
        if checkpoint["error_question_ids"]:
            lines.append(
                "checkpoint_error_ids: "
                + ", ".join(checkpoint["error_question_ids"])
            )
    report = summary.get("report")
    if report:
        if report.get("exists"):
            lines.append(
                "report: "
                f"completed={report.get('completed')} "
                f"total={report.get('total_questions')} "
                f"errors={report.get('errors')} "
                f"row_errors={report.get('row_errors')}"
            )
            if report.get("error_question_ids"):
                lines.append(
                    "report_error_ids: "
                    + ", ".join(report["error_question_ids"])
                )
        else:
            lines.append("report: missing")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Checkpoint JSONL path")
    parser.add_argument("--report", help="Final VeraBench report JSON path")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args(argv)
    if not args.checkpoint and not args.report:
        parser.error("at least one of --checkpoint or --report is required")
    summary = summarize_status(
        checkpoint_path=args.checkpoint,
        report_path=args.report,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_text(summary))


if __name__ == "__main__":
    main()
