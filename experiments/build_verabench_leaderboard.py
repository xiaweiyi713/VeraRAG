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
    mode: str
    provider: str
    model: str
    git_commit: str
    timestamp: str
    config_path: str
    benchmark_version: str
    benchmark_corpus_sha256: str
    benchmark_questions_sha256: str
    implementation_sha256: str
    config_sha256: str
    metric_versions: dict[str, str]
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
    confidence_intervals: dict[str, Any]
    dependency_robust_confidence_intervals: dict[str, Any]
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
    benchmark = metadata.get("benchmark") or {}
    fingerprints = benchmark.get("fingerprints") or {}
    run_signature = metadata.get("run_signature") or {}
    model = str(metadata.get("model") or "")
    provider = str(metadata.get("provider") or "")
    default_run = report_path.stem

    return LeaderboardRow(
        run=run_name or str(metadata.get("run_name") or default_run),
        path=str(report_path),
        mode=str(metadata.get("mode") or ""),
        provider=provider,
        model=model,
        git_commit=str(metadata.get("git_commit") or ""),
        timestamp=str(metadata.get("timestamp") or ""),
        config_path=str(metadata.get("config_path") or ""),
        benchmark_version=str(benchmark.get("version") or ""),
        benchmark_corpus_sha256=str(fingerprints.get("corpus_sha256") or ""),
        benchmark_questions_sha256=str(fingerprints.get("questions_sha256") or ""),
        implementation_sha256=str(run_signature.get("implementation_sha256") or ""),
        config_sha256=str(run_signature.get("config_sha256") or ""),
        metric_versions={
            str(key): str(value)
            for key, value in (metadata.get("metric_versions") or {}).items()
        },
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
        confidence_intervals=report.get("confidence_intervals") or {},
        dependency_robust_confidence_intervals=report.get(
            "dependency_robust_confidence_intervals"
        ) or {},
        by_type=report.get("by_type") or {},
    )


def _benchmark_identity(row: LeaderboardRow) -> tuple[Any, ...]:
    return (
        row.benchmark_version,
        row.benchmark_corpus_sha256,
        row.benchmark_questions_sha256,
        tuple(sorted(row.metric_versions.items())),
    )


def _validate_rankable_rows(
    rows: list[LeaderboardRow],
    *,
    allow_demo: bool,
    allow_incomplete: bool,
    allow_mixed_benchmarks: bool,
    allow_unverified: bool,
) -> None:
    for row in rows:
        if row.mode == "demo" and not allow_demo:
            raise ValueError(
                f"{row.path}: demo reports are plumbing checks, not leaderboard runs; "
                "pass --allow-demo only for diagnostic output"
            )
        if (
            (row.completed != row.total_questions or row.errors)
            and not allow_incomplete
        ):
            raise ValueError(
                f"{row.path}: incomplete run "
                f"({row.completed}/{row.total_questions}, errors={row.errors}); "
                "pass --allow-incomplete only for diagnostic output"
            )

        missing = [
            label
            for label, value in (
                ("metadata.mode", row.mode),
                ("metadata.benchmark.version", row.benchmark_version),
                (
                    "metadata.benchmark.fingerprints.corpus_sha256",
                    row.benchmark_corpus_sha256,
                ),
                (
                    "metadata.benchmark.fingerprints.questions_sha256",
                    row.benchmark_questions_sha256,
                ),
                (
                    "metadata.run_signature.implementation_sha256",
                    row.implementation_sha256,
                ),
                ("metadata.metric_versions", row.metric_versions),
            )
            if not value
        ]
        if row.mode != "demo" and not row.config_sha256:
            missing.append("metadata.run_signature.config_sha256")
        if missing and not allow_unverified:
            raise ValueError(
                f"{row.path}: missing reproducibility metadata: {', '.join(missing)}; "
                "pass --allow-unverified only for explicitly labeled legacy reports"
            )

    identities = {
        _benchmark_identity(row)
        for row in rows
        if all(_benchmark_identity(row))
    }
    if len(identities) > 1 and not allow_mixed_benchmarks:
        raise ValueError(
            "Reports use different VeraBench fingerprints or metric versions; "
            "pass --allow-mixed-benchmarks only for explicitly labeled historical tables"
        )


def build_rows(
    paths: list[str | Path],
    sort_metric: str = "behavior_accuracy",
    *,
    allow_demo: bool = False,
    allow_incomplete: bool = False,
    allow_mixed_benchmarks: bool = False,
    allow_unverified: bool = False,
) -> list[LeaderboardRow]:
    rows = [load_row(path) for path in paths]
    _validate_rankable_rows(
        rows,
        allow_demo=allow_demo,
        allow_incomplete=allow_incomplete,
        allow_mixed_benchmarks=allow_mixed_benchmarks,
        allow_unverified=allow_unverified,
    )
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
        rows_with_intervals = [
            row
            for row in rows
            if (row.confidence_intervals.get("metrics") or {})
        ]
        if rows_with_intervals:
            lines.extend([
                "",
                "## Statistical Uncertainty",
                "",
                "| Run | Confidence | Answer F1 CI | Behavior Acc CI | Conflict F1 CI | Method |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ])
            for row in rows_with_intervals:
                intervals = row.confidence_intervals
                metrics = intervals.get("metrics") or {}

                def interval(name: str) -> str:
                    values = metrics.get(name)
                    if not values:
                        return "-"
                    return (
                        f"[{float(values['lower']):.3f}, "
                        f"{float(values['upper']):.3f}]"
                    )

                lines.append(
                    f"| {row.run} | "
                    f"{float(intervals.get('confidence_level', 0.95)) * 100:.1f}% | "
                    f"{interval('answer_f1')} | "
                    f"{interval('behavior_accuracy')} | "
                    f"{interval('conflict_micro_f1')} | "
                    f"{intervals.get('method', '-')} |"
                )

        rows_with_dependency_intervals = [
            row
            for row in rows
            if (
                row.dependency_robust_confidence_intervals.get("metrics")
                or {}
            )
        ]
        if rows_with_dependency_intervals:
            lines.extend([
                "",
                "## Shared-Evidence Dependency Sensitivity",
                "",
                "| Run | Clusters | Answer F1 CI | Behavior Acc CI | Conflict F1 CI | Method |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ])
            for row in rows_with_dependency_intervals:
                intervals = row.dependency_robust_confidence_intervals
                metrics = intervals.get("metrics") or {}

                def dependency_interval(name: str) -> str:
                    values = metrics.get(name)
                    if not values:
                        return "-"
                    return (
                        f"[{float(values['lower']):.3f}, "
                        f"{float(values['upper']):.3f}]"
                    )

                lines.append(
                    f"| {row.run} | {int(intervals.get('clusters', 0))} | "
                    f"{dependency_interval('answer_f1')} | "
                    f"{dependency_interval('behavior_accuracy')} | "
                    f"{dependency_interval('conflict_micro_f1')} | "
                    f"{intervals.get('method', '-')} |"
                )

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
            "| Run | Mode | Benchmark | Questions SHA | Provider | Model | Config | Timestamp | Result Path |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in rows:
            lines.append(
                "| "
                f"{row.run} | {row.mode or '-'} | {row.benchmark_version or '-'} | "
                f"{row.benchmark_questions_sha256[:12] or '-'} | "
                f"{row.provider or '-'} | {row.model or '-'} | "
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
    parser.add_argument(
        "--allow-demo",
        action="store_true",
        help="Allow demo identity reports. Intended only for diagnostic output.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow partial or errored runs. Intended only for diagnostic output.",
    )
    parser.add_argument(
        "--allow-mixed-benchmarks",
        action="store_true",
        help="Allow different benchmark fingerprints or metric versions.",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow legacy reports that lack current reproducibility metadata.",
    )
    args = parser.parse_args()

    try:
        rows = build_rows(
            args.reports,
            sort_metric=args.sort_metric,
            allow_demo=args.allow_demo,
            allow_incomplete=args.allow_incomplete,
            allow_mixed_benchmarks=args.allow_mixed_benchmarks,
            allow_unverified=args.allow_unverified,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
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
