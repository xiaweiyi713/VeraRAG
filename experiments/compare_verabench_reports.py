#!/usr/bin/env python3
"""Compare two compatible VeraBench reports with paired statistical inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.evaluation.statistics import paired_bootstrap_comparison  # noqa: E402

_IDENTITY_PATHS = (
    ("benchmark", "version"),
    ("benchmark", "fingerprints", "corpus_sha256"),
    ("benchmark", "fingerprints", "questions_sha256"),
    ("metric_versions", "answer"),
    ("metric_versions", "behavior"),
    ("metric_versions", "conflict"),
)


def _nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _rows(report: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = report.get("question_results") or report.get("results") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: report has no per-question results")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"{label}: result row is missing question_id")
        if question_id in by_id:
            raise ValueError(f"{label}: duplicate question_id {question_id}")
        if row.get("error"):
            raise ValueError(
                f"{label}: question {question_id} has an error; "
                "paired publication comparisons require complete runs"
            )
        by_id[question_id] = row
    return [by_id[question_id] for question_id in sorted(by_id)]


def _validate_metadata(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_demo: bool,
    allow_unverified: bool,
) -> None:
    baseline_metadata = baseline.get("metadata") or {}
    candidate_metadata = candidate.get("metadata") or {}
    for label, metadata in (
        ("baseline", baseline_metadata),
        ("candidate", candidate_metadata),
    ):
        mode = str(metadata.get("mode") or "")
        if not mode and not allow_unverified:
            raise ValueError(f"{label}: missing reproducibility metadata: mode")
        if mode == "demo" and not allow_demo:
            raise ValueError(
                f"{label}: demo reports cannot be used for model comparison"
            )
        missing = [
            ".".join(path)
            for path in _IDENTITY_PATHS
            if _nested(metadata, path) in (None, "")
        ]
        if missing and not allow_unverified:
            raise ValueError(
                f"{label}: missing reproducibility metadata: {', '.join(missing)}"
            )

    mismatches = [
        ".".join(path)
        for path in _IDENTITY_PATHS
        if _nested(baseline_metadata, path)
        != _nested(candidate_metadata, path)
    ]
    if mismatches:
        raise ValueError(
            "Reports are not statistically comparable: "
            + ", ".join(mismatches)
        )


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    confidence_level: float = 0.95,
    resamples: int = 5000,
    seed: int = 1729,
    allow_demo: bool = False,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Validate, align, and statistically compare two saved reports."""
    _validate_metadata(
        baseline,
        candidate,
        allow_demo=allow_demo,
        allow_unverified=allow_unverified,
    )
    baseline_rows = _rows(baseline, baseline_label)
    candidate_rows = _rows(candidate, candidate_label)
    for label, report, rows in (
        (baseline_label, baseline, baseline_rows),
        (candidate_label, candidate, candidate_rows),
    ):
        total = int(report.get("total_questions") or 0)
        completed = int(report.get("completed") or 0)
        errors = int(report.get("errors") or 0)
        if total != completed or errors or len(rows) != completed:
            raise ValueError(
                f"{label}: incomplete or inconsistent report "
                f"(rows={len(rows)}, completed={completed}, total={total}, "
                f"errors={errors})"
            )
    baseline_ids = [str(row["question_id"]) for row in baseline_rows]
    candidate_ids = [str(row["question_id"]) for row in candidate_rows]
    if baseline_ids != candidate_ids:
        missing_from_candidate = sorted(set(baseline_ids) - set(candidate_ids))
        extra_in_candidate = sorted(set(candidate_ids) - set(baseline_ids))
        raise ValueError(
            "Reports cover different question IDs: "
            f"missing_from_candidate={missing_from_candidate[:10]}, "
            f"extra_in_candidate={extra_in_candidate[:10]}"
        )

    comparison = paired_bootstrap_comparison(
        baseline_rows,
        candidate_rows,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    baseline_metadata = baseline.get("metadata") or {}
    candidate_metadata = candidate.get("metadata") or {}
    return {
        "schema_version": 1,
        "baseline": {
            "label": baseline_label,
            "provider": baseline_metadata.get("provider", ""),
            "model": baseline_metadata.get("model", ""),
            "git_commit": baseline_metadata.get("git_commit", ""),
            "config_path": baseline_metadata.get("config_path", ""),
        },
        "candidate": {
            "label": candidate_label,
            "provider": candidate_metadata.get("provider", ""),
            "model": candidate_metadata.get("model", ""),
            "git_commit": candidate_metadata.get("git_commit", ""),
            "config_path": candidate_metadata.get("config_path", ""),
        },
        "benchmark": baseline_metadata.get("benchmark") or {},
        "metric_versions": baseline_metadata.get("metric_versions") or {},
        "comparison": comparison,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    baseline = payload["baseline"]
    candidate = payload["candidate"]
    comparison = payload["comparison"]
    confidence = float(comparison["confidence_level"]) * 100
    lines = [
        "# VeraBench Paired Comparison",
        "",
        f"- Baseline: `{baseline['label']}`",
        f"- Candidate: `{candidate['label']}`",
        f"- Paired questions: {comparison['questions']}",
        (
            f"- Inference: {comparison['method']}, "
            f"{comparison['resamples']} resamples, seed {comparison['seed']}"
        ),
        "",
        "## Metric Deltas",
        "",
        (
            "| Metric | Direction | Baseline | Candidate | "
            "Delta (candidate - baseline) | "
            f"{confidence:.1f}% delta CI | P(candidate better) |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in sorted(comparison["metrics"].items()):
        lines.append(
            f"| {name} | {metric.get('direction', 'higher_is_better')} | "
            f"{metric['baseline']:.4f} | "
            f"{metric['candidate']:.4f} | "
            f"{metric['delta_candidate_minus_baseline']:+.4f} | "
            f"[{metric['delta_lower']:+.4f}, {metric['delta_upper']:+.4f}] | "
            f"{metric['probability_candidate_better']:.3f} |"
        )

    dependency_robust = comparison.get("dependency_robust", {})
    if dependency_robust:
        lines.extend([
            "",
            "## Shared-Evidence Dependency Sensitivity",
            "",
            (
                f"{dependency_robust['clusters']} evidence-connected clusters; "
                f"{dependency_robust['resamples']} paired cluster resamples."
            ),
            "",
            (
                "| Metric | Delta (candidate - baseline) | "
                f"{confidence:.1f}% cluster-robust delta CI |"
            ),
            "| --- | ---: | ---: |",
        ])
        for name, metric in sorted(dependency_robust["metrics"].items()):
            lines.append(
                f"| {name} | "
                f"{metric['delta_candidate_minus_baseline']:+.4f} | "
                f"[{metric['delta_lower']:+.4f}, "
                f"{metric['delta_upper']:+.4f}] |"
            )

    mcnemar = comparison["behavior_mcnemar_exact"]
    lines.extend([
        "",
        "## Behavior McNemar Test",
        "",
        (
            f"Baseline-only correct: {mcnemar['baseline_only_correct']}; "
            f"candidate-only correct: {mcnemar['candidate_only_correct']}; "
            f"two-sided exact p={mcnemar['two_sided_exact_p']:.6f}."
        ),
        "",
        (
            "Intervals quantify uncertainty on this fixed benchmark sample; "
            "they do not establish external validity beyond VeraBench."
        ),
        "",
    ])
    return "\n".join(lines)


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline VeraBench report JSON")
    parser.add_argument("candidate", help="Candidate VeraBench report JSON")
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--confidence-level", type=_probability, default=0.95)
    parser.add_argument("--resamples", type=_positive_int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-demo",
        action="store_true",
        help="Allow demo identity reports for diagnostics only.",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow legacy reports missing current reproducibility metadata.",
    )
    args = parser.parse_args()

    try:
        with open(args.baseline, encoding="utf-8") as handle:
            baseline = json.load(handle)
        with open(args.candidate, encoding="utf-8") as handle:
            candidate = json.load(handle)
        payload = compare_reports(
            baseline,
            candidate,
            baseline_label=args.baseline_label or Path(args.baseline).stem,
            candidate_label=args.candidate_label or Path(args.candidate).stem,
            confidence_level=args.confidence_level,
            resamples=args.resamples,
            seed=args.seed,
            allow_demo=args.allow_demo,
            allow_unverified=args.allow_unverified,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    rendered = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else render_markdown(payload)
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
