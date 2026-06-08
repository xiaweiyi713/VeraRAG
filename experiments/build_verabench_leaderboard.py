#!/usr/bin/env python3
"""Build a reproducible VeraBench leaderboard from saved report JSON files."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_METRICS = [
    "behavior_accuracy",
    "overall_answer_f1",
    "overall_evidence_recall",
    "overall_conflict_f1",
    "ece",
]


@dataclass
class LeaderboardRow:
    run: str
    path: str
    provider: str
    model: str
    git_commit: str
    timestamp: str
    config_path: str
    total_questions: int
    completed: int
    errors: int
    behavior_accuracy: float
    overall_answer_f1: float
    overall_evidence_recall: float
    overall_evidence_precision: float
    overall_conflict_f1: float
    ece: float
    brier_score: float
    avg_confidence: float
    avg_latency: float
    by_type: dict[str, dict[str, float]]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _float(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    return float(value) if value is not None else 0.0


def _int(report: dict[str, Any], key: str) -> int:
    value = report.get(key)
    return int(value) if value is not None else 0


def load_row(path: str | Path, run_name: str | None = None) -> LeaderboardRow:
    report_path = Path(path)
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    metadata = report.get("metadata") or {}
    model = str(metadata.get("model") or "")
    provider = str(metadata.get("provider") or "")
    default_run = report_path.stem

    return LeaderboardRow(
        run=run_name or str(metadata.get("run_name") or default_run),
        path=str(report_path),
        provider=provider,
        model=model,
        git_commit=str(metadata.get("git_commit") or ""),
        timestamp=str(metadata.get("timestamp") or ""),
        config_path=str(metadata.get("config_path") or ""),
        total_questions=_int(report, "total_questions"),
        completed=_int(report, "completed"),
        errors=_int(report, "errors"),
        behavior_accuracy=_float(report, "behavior_accuracy"),
        overall_answer_f1=_float(report, "overall_answer_f1"),
        overall_evidence_recall=_float(report, "overall_evidence_recall"),
        overall_evidence_precision=_float(report, "overall_evidence_precision"),
        overall_conflict_f1=_float(report, "overall_conflict_f1"),
        ece=_float(report, "ece"),
        brier_score=_float(report, "brier_score"),
        avg_confidence=_float(report, "avg_confidence"),
        avg_latency=_float(report, "avg_latency"),
        by_type=report.get("by_type") or {},
    )


def build_rows(paths: list[str | Path], sort_metric: str = "behavior_accuracy") -> list[LeaderboardRow]:
    rows = [load_row(path) for path in paths]
    reverse = sort_metric not in {"ece", "brier_score", "avg_latency", "errors"}
    rows.sort(key=lambda row: getattr(row, sort_metric), reverse=reverse)
    return rows


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _model_label(row: LeaderboardRow) -> str:
    if row.provider and row.model:
        return f"{row.provider}/{row.model}"
    return row.model or row.provider or "-"


def render_markdown(
    rows: list[LeaderboardRow],
    title: str = "VeraBench Results",
    source_command: str | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "This file is generated from saved `run_verabench.py --output` JSON reports.",
        "Commit raw result JSON files only when intentionally publishing a reproducibility artifact; otherwise keep them in ignored `results/` and regenerate this summary.",
        "",
    ]
    if source_command:
        lines.extend([
            "Generation command:",
            "",
            f"```bash\n{source_command}\n```",
            "",
        ])

    lines.extend([
        "## Leaderboard",
        "",
        "| Rank | Run | Model | Questions | Errors | Behavior Acc | Answer F1 | Evidence Recall | Conflict F1 | ECE | Avg Latency | Commit |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.run} | {_model_label(row)} | {row.completed}/{row.total_questions} | {row.errors} | "
            f"{_fmt(row.behavior_accuracy)} | {_fmt(row.overall_answer_f1)} | "
            f"{_fmt(row.overall_evidence_recall)} | {_fmt(row.overall_conflict_f1)} | "
            f"{_fmt(row.ece)} | {row.avg_latency:.1f}s | {row.git_commit or '-'} |"
        )

    if rows:
        best = rows[0]
        lines.extend([
            "",
            f"## Best Run By Type: {best.run} ({_model_label(best)})",
            "",
            "| Type | Count | Answer F1 | Evidence Recall | Behavior Acc |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for qtype, metrics in sorted(best.by_type.items()):
            lines.append(
                "| "
                f"{qtype} | {int(metrics.get('count', 0))} | "
                f"{_fmt(float(metrics.get('answer_f1', 0.0)))} | "
                f"{_fmt(float(metrics.get('evidence_recall', 0.0)))} | "
                f"{_fmt(float(metrics.get('behavior_accuracy', 0.0)))} |"
            )

        lines.extend([
            "",
            "## Reproducibility Metadata",
            "",
            "| Run | Provider | Model | Config | Timestamp | Result Path |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for row in rows:
            lines.append(
                "| "
                f"{row.run} | {row.provider or '-'} | {row.model or '-'} | "
                f"{row.config_path or '-'} | {row.timestamp or '-'} | `{row.path}` |"
            )

    lines.append("")
    return "\n".join(lines)


def render_json(rows: list[LeaderboardRow]) -> str:
    payload = {
        "schema_version": 1,
        "metrics": DEFAULT_METRICS,
        "runs": [row.to_json() for row in rows],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VeraBench leaderboard from report JSON files")
    parser.add_argument("reports", nargs="+", help="Saved run_verabench.py JSON report(s)")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", help="Output path. Defaults to stdout.")
    parser.add_argument(
        "--sort-metric",
        default="behavior_accuracy",
        choices=[
            "behavior_accuracy",
            "overall_answer_f1",
            "overall_evidence_recall",
            "overall_conflict_f1",
            "ece",
            "brier_score",
            "avg_latency",
            "errors",
        ],
        help="Metric used for sorting. Lower is better for ece, brier_score, avg_latency, and errors.",
    )
    parser.add_argument("--title", default="VeraBench Results")
    args = parser.parse_args()

    rows = build_rows(args.reports, sort_metric=args.sort_metric)
    if args.format == "json":
        rendered = render_json(rows)
    else:
        command = "python experiments/build_verabench_leaderboard.py " + " ".join(args.reports)
        if args.output:
            command += f" --output {args.output}"
        rendered = render_markdown(rows, title=args.title, source_command=command)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
