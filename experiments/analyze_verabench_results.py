#!/usr/bin/env python3
"""Analyze an existing VeraBench JSON report.

This is intentionally offline: it reads a saved ``run_verabench.py --output``
report and summarizes behavior mismatches, weak evidence retrieval, and conflict
failures without calling any LLM.

Usage:
    python experiments/analyze_verabench_results.py results/verabench_full_v3.json
    python experiments/analyze_verabench_results.py results/verabench_full_v3.json --json
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.benchmark.loader import (  # noqa: E402
    evidence_dependency_groups,
    load_verabench,
)
from src.evaluation.statistics import (  # noqa: E402
    dependency_cluster_bootstrap_confidence_intervals,
    stratified_bootstrap_confidence_intervals,
)


def _load_report(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("VeraBench report must be a JSON object")
    return data


def _question_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report.get("question_results") or report.get("results") or []


def _behavior_confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = row.get("expected_behavior") or "unknown"
        actual = row.get("actual_behavior") or "unknown"
        confusion.setdefault(expected, {})
        confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
    return confusion


def _count_by_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.get("question_type", "unknown")] += 1
    return dict(sorted(counts.items()))


def _build_summary(rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    behavior_failures = [
        r for r in rows
        if r.get("actual_behavior") != r.get("expected_behavior")
    ]
    low_evidence_recall = [
        r for r in rows
        if float(r.get("evidence_recall") or 0.0) < 0.5
        and r.get("expected_behavior") != "abstain"
    ]
    conflict_failures = [
        r for r in rows
        if (
            int(r.get("conflict_false_positives") or 0) > 0
            or int(r.get("conflict_false_negatives") or 0) > 0
            or (
                r.get("question_type") == "conflict"
                and r.get("actual_behavior") != r.get("expected_behavior")
            )
        )
    ]

    def severity_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row.get("answer_f1") or 0.0),
            float(row.get("evidence_recall") or 0.0),
            float(row.get("conflict_detection_f1") or 0.0),
            -float(row.get("latency_seconds") or 0.0),
        )

    ranked_failures = sorted(behavior_failures, key=severity_key)

    return {
        "behavior_failure_count": len(behavior_failures),
        "behavior_failures_by_type": _count_by_type(behavior_failures),
        "low_evidence_recall_count": len(low_evidence_recall),
        "low_evidence_recall_by_type": _count_by_type(low_evidence_recall),
        "conflict_failure_count": len(conflict_failures),
        "conflict_failures_by_type": _count_by_type(conflict_failures),
        "top_behavior_failures": ranked_failures[:max_examples],
    }


def _calibration_bins(rows: list[dict[str, Any]], n_bins: int = 10) -> list[dict[str, Any]]:
    bins = []
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == n_bins - 1:
            bucket = [
                r for r in rows
                if lower <= float(r.get("confidence") or 0.0) <= upper
            ]
        else:
            bucket = [
                r for r in rows
                if lower <= float(r.get("confidence") or 0.0) < upper
            ]

        if bucket:
            avg_conf = sum(float(r.get("confidence") or 0.0) for r in bucket) / len(bucket)
            acc = sum(1 for r in bucket if r.get("correct")) / len(bucket)
        else:
            avg_conf = (lower + upper) / 2
            acc = 0.0
        bins.append({
            "bin": i + 1,
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "count": len(bucket),
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(abs(avg_conf - acc), 4),
        })
    return bins


def _correctness(row: dict[str, Any]) -> bool | None:
    value = row.get("correct")
    if isinstance(value, bool):
        return value
    expected = row.get("expected_behavior")
    actual = row.get("actual_behavior")
    if isinstance(expected, str) and isinstance(actual, str):
        return expected == actual
    return None


def _confidence_value(row: dict[str, Any]) -> float | None:
    value = row.get("confidence")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _round_optional(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _confidence_auc(pairs: list[tuple[float, bool]]) -> float | None:
    positives = [confidence for confidence, correct in pairs if correct]
    negatives = [confidence for confidence, correct in pairs if not correct]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total


def _risk_coverage(
    pairs: list[tuple[float, bool]],
    *,
    targets: tuple[float, ...] = (0.8, 0.9, 0.95),
) -> dict[str, Any]:
    if not pairs:
        return {
            "available": False,
            "reason": "no rows with valid confidence and correctness",
        }
    ranked = sorted(pairs, key=lambda item: item[0], reverse=True)
    points: list[dict[str, float | int]] = []
    correct_count = 0
    total = len(ranked)
    best_by_accuracy = {target: 0.0 for target in targets}
    for idx, (confidence, correct) in enumerate(ranked, start=1):
        correct_count += int(correct)
        coverage = idx / total
        accuracy = correct_count / idx
        risk = 1.0 - accuracy
        points.append(
            {
                "coverage": round(coverage, 6),
                "accuracy": round(accuracy, 6),
                "risk": round(risk, 6),
                "threshold": round(confidence, 6),
                "kept": idx,
            }
        )
        for target in targets:
            if accuracy >= target:
                best_by_accuracy[target] = coverage

    aurc = sum(float(point["risk"]) for point in points) / len(points)
    return {
        "available": True,
        "aurc": round(aurc, 6),
        "coverage_at_accuracy": {
            f"{target:.2f}": round(coverage, 6)
            for target, coverage in best_by_accuracy.items()
        },
        "points": points,
    }


def _write_risk_coverage_csv(risk_coverage: dict[str, Any], path: Path) -> None:
    points = _require_risk_coverage_points(risk_coverage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kept", "coverage", "accuracy", "risk", "threshold"],
        )
        writer.writeheader()
        writer.writerows(points)


def _write_risk_coverage_svg(risk_coverage: dict[str, Any], path: Path) -> None:
    points = _require_risk_coverage_points(risk_coverage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_risk_coverage_svg(risk_coverage, points), encoding="utf-8")


def _require_risk_coverage_points(risk_coverage: dict[str, Any]) -> list[dict[str, Any]]:
    if risk_coverage.get("available") is False:
        raise ValueError(f"risk-coverage unavailable: {risk_coverage.get('reason', 'missing data')}")
    points = risk_coverage.get("points") or []
    if not points:
        raise ValueError("risk-coverage curve has no points")
    return points


def _risk_coverage_svg(risk_coverage: dict[str, Any], points: list[dict[str, Any]]) -> str:
    width = 760
    height = 460
    left = 72
    right = 32
    top = 44
    bottom = 74
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_coord(coverage: float) -> float:
        return left + coverage * plot_width

    def y_coord(risk: float) -> float:
        return top + (1.0 - risk) * plot_height

    polyline = " ".join(
        f"{x_coord(float(point['coverage'])):.2f},{y_coord(float(point['risk'])):.2f}"
        for point in points
    )
    coverage_labels = risk_coverage.get("coverage_at_accuracy") or {}
    coverage_text = ", ".join(
        f"acc>={target}: {float(coverage):.3f}"
        for target, coverage in sorted(coverage_labels.items())
    )
    if not coverage_text:
        coverage_text = "coverage targets unavailable"

    grid_lines = []
    tick_labels = []
    for idx in range(6):
        value = idx / 5
        x = x_coord(value)
        y = y_coord(value)
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" class="grid"/>'
        )
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="grid"/>'
        )
        tick_labels.append(
            f'<text x="{x:.2f}" y="{height - bottom + 24}" text-anchor="middle">{value:.1f}</text>'
        )
        tick_labels.append(
            f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">VeraBench Risk-Coverage Curve</title>
  <desc id="desc">Selective prediction curve showing risk as coverage increases by confidence threshold.</desc>
  <style>
    .bg {{ fill: #ffffff; }}
    .axis {{ stroke: #1f2937; stroke-width: 1.4; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .curve {{ fill: none; stroke: #2563eb; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
    .label {{ fill: #111827; font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .small {{ fill: #4b5563; font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .title {{ fill: #111827; font: 700 18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  </style>
  <rect class="bg" width="100%" height="100%"/>
  <text x="{left}" y="26" class="title">Risk-Coverage Curve</text>
  <text x="{left}" y="{height - 18}" class="small">AURC={float(risk_coverage.get('aurc') or 0.0):.4f}; {coverage_text}</text>
  {''.join(grid_lines)}
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis"/>
  <polyline points="{polyline}" class="curve"/>
  {''.join(tick_labels)}
  <text x="{left + plot_width / 2:.2f}" y="{height - 34}" text-anchor="middle" class="label">Coverage</text>
  <text x="22" y="{top + plot_height / 2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 22 {top + plot_height / 2:.2f})">Risk</text>
</svg>
"""


def _confidence_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[float, bool]] = []
    for row in rows:
        confidence = _confidence_value(row)
        correct = _correctness(row)
        if confidence is not None and correct is not None:
            pairs.append((confidence, correct))

    if not pairs:
        return {
            "available": False,
            "reason": "no rows with valid confidence and correctness",
        }

    confidences = [confidence for confidence, _ in pairs]
    correct_confidences = [confidence for confidence, correct in pairs if correct]
    incorrect_confidences = [
        confidence for confidence, correct in pairs if not correct
    ]
    accuracy = sum(int(correct) for _, correct in pairs) / len(pairs)
    mean_confidence = sum(confidences) / len(confidences)
    std_confidence = _std(confidences)
    auc = _confidence_auc(pairs)

    flags = []
    if std_confidence is not None and std_confidence < 0.03:
        flags.append("near_constant_confidence")
    if mean_confidence < accuracy - 0.10:
        flags.append("underconfident")
    if mean_confidence > accuracy + 0.10:
        flags.append("overconfident")
    if auc is not None and auc <= 0.60:
        flags.append("weak_correctness_discrimination")
    if not flags:
        flags.append("no_obvious_confidence_pathology")

    return {
        "available": True,
        "rows": len(pairs),
        "accuracy": round(accuracy, 6),
        "mean_confidence": round(mean_confidence, 6),
        "confidence_std": _round_optional(std_confidence),
        "confidence_min": round(min(confidences), 6),
        "confidence_p25": _round_optional(_quantile(confidences, 0.25)),
        "confidence_median": _round_optional(_quantile(confidences, 0.5)),
        "confidence_p75": _round_optional(_quantile(confidences, 0.75)),
        "confidence_max": round(max(confidences), 6),
        "correct_mean_confidence": _round_optional(_mean(correct_confidences)),
        "incorrect_mean_confidence": _round_optional(_mean(incorrect_confidences)),
        "confidence_auroc": _round_optional(auc),
        "diagnostic_flags": flags,
    }


def _confidence_slice_summary(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    max_errors: int = 5,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float, bool]]] = defaultdict(list)
    for row in rows:
        confidence = _confidence_value(row)
        correct = _correctness(row)
        if confidence is None or correct is None:
            continue
        group = str(row.get(group_field) or "unknown")
        grouped[group].append((row, confidence, correct))

    summaries: dict[str, dict[str, Any]] = {}
    for group, items in sorted(grouped.items()):
        pairs = [(confidence, correct) for _, confidence, correct in items]
        confidences = [confidence for _, confidence, _ in items]
        correct_confidences = [
            confidence for _, confidence, correct in items if correct
        ]
        incorrect_items = [
            (row, confidence)
            for row, confidence, correct in items
            if not correct
        ]
        incorrect_confidences = [confidence for _, confidence in incorrect_items]
        accuracy = sum(int(correct) for _, _, correct in items) / len(items)
        high_confidence_errors = [
            {
                "question_id": row.get("question_id"),
                "question_type": row.get("question_type"),
                "expected_behavior": row.get("expected_behavior"),
                "actual_behavior": row.get("actual_behavior"),
                "confidence": round(confidence, 6),
                "answer_f1": row.get("answer_f1"),
                "evidence_recall": row.get("evidence_recall"),
            }
            for row, confidence in sorted(
                incorrect_items,
                key=lambda item: item[1],
                reverse=True,
            )[:max_errors]
        ]
        summaries[group] = {
            "rows": len(items),
            "accuracy": round(accuracy, 6),
            "mean_confidence": round(sum(confidences) / len(confidences), 6),
            "confidence_std": _round_optional(_std(confidences)),
            "correct_mean_confidence": _round_optional(_mean(correct_confidences)),
            "incorrect_mean_confidence": _round_optional(_mean(incorrect_confidences)),
            "confidence_auroc": _round_optional(_confidence_auc(pairs)),
            "high_confidence_errors": high_confidence_errors,
        }
    return summaries


def _confidence_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not any(
        _confidence_value(row) is not None and _correctness(row) is not None
        for row in rows
    ):
        return {
            "available": False,
            "reason": "no rows with valid confidence and correctness",
        }
    return {
        "available": True,
        "by_actual_behavior": _confidence_slice_summary(
            rows,
            group_field="actual_behavior",
        ),
        "by_expected_behavior": _confidence_slice_summary(
            rows,
            group_field="expected_behavior",
        ),
        "by_question_type": _confidence_slice_summary(
            rows,
            group_field="question_type",
        ),
    }


def _runtime_confidence_calibration_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for row in rows:
        diagnostics = row.get("diagnostics") or {}
        if not isinstance(diagnostics, dict):
            continue
        calibration = diagnostics.get("confidence_calibration")
        if not isinstance(calibration, dict) or not calibration:
            output_metadata = diagnostics.get("output_metadata") or {}
            if isinstance(output_metadata, dict):
                calibration = output_metadata.get("confidence_calibration")
        if isinstance(calibration, dict) and calibration:
            summaries.append(calibration)

    if not summaries:
        return {
            "available": False,
            "reason": "no runtime confidence calibration diagnostics",
        }

    stages: dict[str, int] = defaultdict(int)
    predicted_behaviors: dict[str, int] = defaultdict(int)
    cap_reasons: dict[str, int] = defaultdict(int)
    enabled_rows = 0
    failure_mode_cap_rows = 0
    for summary in summaries:
        if summary.get("enabled") is True:
            enabled_rows += 1
        stages[str(summary.get("stage") or "unknown")] += 1
        predicted_behaviors[str(summary.get("predicted_behavior") or "unknown")] += 1
        caps = summary.get("failure_mode_caps") or []
        if isinstance(caps, list) and caps:
            failure_mode_cap_rows += 1
            for cap in caps:
                reason = "unknown"
                if isinstance(cap, dict):
                    reason = str(cap.get("reason") or "unknown")
                cap_reasons[reason] += 1

    return {
        "available": True,
        "rows": len(summaries),
        "enabled_rows": enabled_rows,
        "disabled_rows": len(summaries) - enabled_rows,
        "stages": dict(sorted(stages.items())),
        "predicted_behaviors": dict(sorted(predicted_behaviors.items())),
        "failure_mode_cap_rows": failure_mode_cap_rows,
        "failure_mode_caps": dict(sorted(cap_reasons.items())),
    }


def _conflict_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    has_counts = any(
        "predicted_conflicts" in r or "gold_conflicts" in r
        for r in rows
    )
    if not has_counts:
        return {
            "available": False,
            "reason": "Per-question conflict TP/FP/FN fields are not present in this report. Re-run run_verabench.py with the current evaluator to generate them.",
        }

    total_gold = sum(int(r.get("gold_conflicts") or 0) for r in rows)
    total_pred = sum(int(r.get("predicted_conflicts") or 0) for r in rows)
    total_tp = sum(int(r.get("conflict_true_positives") or 0) for r in rows)
    total_fp = sum(int(r.get("conflict_false_positives") or 0) for r in rows)
    total_fn = sum(int(r.get("conflict_false_negatives") or 0) for r in rows)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    if total_fp > total_fn * 2:
        dominant_failure = "over_detection"
    elif total_fn > total_fp * 2:
        dominant_failure = "under_detection"
    elif total_fp or total_fn:
        dominant_failure = "mixed"
    else:
        dominant_failure = "none"

    return {
        "available": True,
        "gold_conflicts": total_gold,
        "predicted_conflicts": total_pred,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            4,
        ),
        "dominant_failure": dominant_failure,
    }


def _premise_refutation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    has_fields = any(
        "premise_refutation_expected" in r or "premise_refutation_detected" in r
        for r in rows
    )
    if not has_fields:
        return {
            "available": False,
            "reason": "Per-question premise_refutation fields are not present in this report. Re-run run_verabench.py with the current evaluator to generate them.",
        }

    expected = sum(1 for r in rows if r.get("premise_refutation_expected"))
    detected = sum(1 for r in rows if r.get("premise_refutation_detected"))
    tp = sum(
        1 for r in rows
        if r.get("premise_refutation_expected") and r.get("premise_refutation_detected")
    )
    fp = sum(
        1 for r in rows
        if not r.get("premise_refutation_expected") and r.get("premise_refutation_detected")
    )
    fn = sum(
        1 for r in rows
        if r.get("premise_refutation_expected") and not r.get("premise_refutation_detected")
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "available": True,
        "expected": expected,
        "detected": detected,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def _derive_dependency_intervals(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {}
    if all(row.get("dependency_group") for row in rows):
        return dependency_cluster_bootstrap_confidence_intervals(rows)

    benchmark = load_verabench()
    groups = evidence_dependency_groups(benchmark.questions)
    question_ids = [str(row.get("question_id") or "") for row in rows]
    if not all(question_id in groups for question_id in question_ids):
        return {}
    enriched = [
        {**row, "dependency_group": groups[question_id]}
        for row, question_id in zip(rows, question_ids, strict=True)
    ]
    intervals = dependency_cluster_bootstrap_confidence_intervals(enriched)
    intervals["dependency_mapping_source"] = (
        "current-verabench-shared-gold-document-components"
    )
    return intervals


def analyze(report: dict[str, Any], max_examples: int = 10) -> dict[str, Any]:
    rows = _question_results(report)
    embedded_summary = report.get("failure_summary") or {}
    summary = embedded_summary or _build_summary(rows, max_examples=max_examples)
    confusion = report.get("behavior_confusion") or _behavior_confusion(rows)
    calibration_bins = report.get("calibration_bins") or _calibration_bins(rows)
    confidence_diagnostics = _confidence_diagnostics(rows)
    confidence_slices = _confidence_slices(rows)
    runtime_confidence_calibration = _runtime_confidence_calibration_summary(rows)
    risk_coverage = _risk_coverage(
        [
            (confidence, correct)
            for row in rows
            if (confidence := _confidence_value(row)) is not None
            and (correct := _correctness(row)) is not None
        ]
    )
    conflict_summary = report.get("conflict_summary") or _conflict_summary(rows)
    premise_refutation_summary = (
        report.get("premise_refutation_summary")
        or _premise_refutation_summary(rows)
    )
    confidence_intervals = (
        report.get("confidence_intervals")
        or stratified_bootstrap_confidence_intervals(rows)
    )
    dependency_intervals = report.get(
        "dependency_robust_confidence_intervals"
    ) or _derive_dependency_intervals(rows)
    return {
        "report": {
            "total_questions": report.get("total_questions", len(rows)),
            "completed": report.get("completed"),
            "errors": report.get("errors"),
            "behavior_accuracy": report.get("behavior_accuracy"),
            "answer_f1": report.get("overall_answer_f1"),
            "evidence_recall": report.get("overall_evidence_recall"),
            "conflict_f1": report.get("overall_conflict_f1"),
            "ece": report.get("ece"),
            "brier_score": report.get("brier_score"),
        },
        "metadata": report.get("metadata", {}),
        "calibration_bins": calibration_bins,
        "confidence_diagnostics": confidence_diagnostics,
        "confidence_slices": confidence_slices,
        "runtime_confidence_calibration": runtime_confidence_calibration,
        "risk_coverage": risk_coverage,
        "confidence_intervals": confidence_intervals,
        "dependency_robust_confidence_intervals": dependency_intervals,
        "conflict_summary": conflict_summary,
        "premise_refutation_summary": premise_refutation_summary,
        "behavior_confusion": confusion,
        "failure_summary": summary,
    }


def _print_table(analysis: dict[str, Any], max_examples: int) -> None:
    report = analysis["report"]
    metadata = analysis["metadata"]
    summary = analysis["failure_summary"]

    print("VeraBench Result Analysis")
    print("=" * 72)
    print(f"Questions:       {report.get('total_questions')}")
    print(f"Behavior Acc:    {float(report.get('behavior_accuracy') or 0.0):.4f}")
    print(f"Answer F1:       {float(report.get('answer_f1') or 0.0):.4f}")
    print(f"Evidence Recall: {float(report.get('evidence_recall') or 0.0):.4f}")
    print(f"Conflict F1:     {float(report.get('conflict_f1') or 0.0):.4f}")
    print(f"ECE / Brier:     {float(report.get('ece') or 0.0):.4f} / {float(report.get('brier_score') or 0.0):.4f}")
    if metadata:
        provider = metadata.get("provider", "")
        model = metadata.get("model", "")
        commit = metadata.get("git_commit", "")
        print(f"Run:             {provider}/{model} @ {commit}")

    intervals = analysis.get("confidence_intervals", {})
    interval_metrics = intervals.get("metrics", {})
    if interval_metrics:
        confidence = float(intervals.get("confidence_level", 0.95)) * 100
        print(f"\nStratified Bootstrap {confidence:.1f}% Intervals")
        print("-" * 72)
        for key in (
            "answer_f1",
            "evidence_recall",
            "behavior_accuracy",
            "conflict_micro_f1",
            "ece",
            "brier_score",
        ):
            values = interval_metrics.get(key)
            if values:
                print(
                    f"{key:<24s} "
                    f"[{float(values['lower']):.4f}, "
                    f"{float(values['upper']):.4f}]"
                )

    dependency_intervals = analysis.get(
        "dependency_robust_confidence_intervals",
        {},
    )
    dependency_metrics = dependency_intervals.get("metrics", {})
    if dependency_metrics:
        confidence = (
            float(dependency_intervals.get("confidence_level", 0.95)) * 100
        )
        clusters = int(dependency_intervals.get("clusters", 0))
        print(
            f"\nEvidence-Cluster Bootstrap {confidence:.1f}% Intervals "
            f"({clusters} clusters)"
        )
        print("-" * 72)
        for key in (
            "answer_f1",
            "evidence_recall",
            "behavior_accuracy",
            "conflict_micro_f1",
            "ece",
            "brier_score",
        ):
            values = dependency_metrics.get(key)
            if values:
                print(
                    f"{key:<24s} "
                    f"[{float(values['lower']):.4f}, "
                    f"{float(values['upper']):.4f}]"
                )

    print("\nFailure Summary")
    print("-" * 72)
    print(f"Behavior failures:   {summary.get('behavior_failure_count', 0)}")
    print(f"Low evidence recall: {summary.get('low_evidence_recall_count', 0)}")
    print(f"Conflict failures:   {summary.get('conflict_failure_count', 0)}")
    by_type = summary.get("behavior_failures_by_type", {})
    if by_type:
        print("Behavior failures by type:")
        for qtype, count in sorted(by_type.items()):
            print(f"  {qtype:<18s} {count}")

    conflict = analysis.get("conflict_summary", {})
    if conflict:
        print("\nConflict Summary")
        print("-" * 72)
        if conflict.get("available") is False:
            print(f"Unavailable: {conflict.get('reason', 'missing per-question conflict counts')}")
        else:
            print(f"Gold / Predicted: {conflict.get('gold_conflicts', 0)} / {conflict.get('predicted_conflicts', 0)}")
            print(f"TP / FP / FN:     {conflict.get('true_positives', 0)} / {conflict.get('false_positives', 0)} / {conflict.get('false_negatives', 0)}")
            print(f"Precision/Recall: {float(conflict.get('precision') or 0.0):.4f} / {float(conflict.get('recall') or 0.0):.4f}")
            print(f"Dominant failure: {conflict.get('dominant_failure', 'unknown')}")

    premise = analysis.get("premise_refutation_summary", {})
    if premise:
        print("\nPremise Refutation Summary")
        print("-" * 72)
        if premise.get("available") is False:
            print(f"Unavailable: {premise.get('reason', 'missing per-question premise-refutation fields')}")
        else:
            print(f"Expected / Detected: {premise.get('expected', 0)} / {premise.get('detected', 0)}")
            print(f"TP / FP / FN:         {premise.get('true_positives', 0)} / {premise.get('false_positives', 0)} / {premise.get('false_negatives', 0)}")
            print(f"Precision/Recall:     {float(premise.get('precision') or 0.0):.4f} / {float(premise.get('recall') or 0.0):.4f}")

    bins = analysis.get("calibration_bins", [])
    if bins:
        print("\nCalibration Bins")
        print("-" * 72)
        for b in bins:
            if b.get("count", 0) > 0:
                print(
                    f"{b['lower']:.1f}-{b['upper']:.1f} "
                    f"n={b['count']:<3d} conf={b['avg_confidence']:.3f} "
                    f"acc={b['accuracy']:.3f} gap={b['gap']:.3f}"
                )

    confidence = analysis.get("confidence_diagnostics", {})
    if confidence:
        print("\nConfidence Diagnostics")
        print("-" * 72)
        if confidence.get("available") is False:
            print(f"Unavailable: {confidence.get('reason', 'missing confidence/correctness rows')}")
        else:
            print(
                f"Mean / std:       "
                f"{float(confidence.get('mean_confidence') or 0.0):.4f} / "
                f"{float(confidence.get('confidence_std') or 0.0):.4f}"
            )
            print(
                f"Correct / wrong:  "
                f"{float(confidence.get('correct_mean_confidence') or 0.0):.4f} / "
                f"{float(confidence.get('incorrect_mean_confidence') or 0.0):.4f}"
            )
            auc = confidence.get("confidence_auroc")
            print(f"AUROC:            {float(auc):.4f}" if auc is not None else "AUROC:            unavailable")
            print(
                "Flags:            "
                + ", ".join(confidence.get("diagnostic_flags", []))
            )

    confidence_slices = analysis.get("confidence_slices", {})
    if confidence_slices:
        print("\nConfidence by Actual Behavior")
        print("-" * 72)
        if confidence_slices.get("available") is False:
            print(
                "Unavailable: "
                f"{confidence_slices.get('reason', 'missing confidence/correctness rows')}"
            )
        else:
            by_behavior = confidence_slices.get("by_actual_behavior") or {}
            for behavior, values in sorted(by_behavior.items()):
                wrong_mean = values.get("incorrect_mean_confidence")
                wrong_text = (
                    f"{float(wrong_mean):.4f}" if wrong_mean is not None else "n/a"
                )
                auc = values.get("confidence_auroc")
                auc_text = f"{float(auc):.4f}" if auc is not None else "n/a"
                print(
                    f"{behavior:<26s} "
                    f"n={int(values.get('rows', 0)):<3d} "
                    f"acc={float(values.get('accuracy') or 0.0):.3f} "
                    f"conf={float(values.get('mean_confidence') or 0.0):.3f} "
                    f"wrong_conf={wrong_text} "
                    f"auc={auc_text}"
                )

    runtime_calibration = analysis.get("runtime_confidence_calibration", {})
    if runtime_calibration:
        print("\nRuntime Confidence Calibration")
        print("-" * 72)
        if runtime_calibration.get("available") is False:
            print(
                "Unavailable: "
                f"{runtime_calibration.get('reason', 'missing runtime calibration diagnostics')}"
            )
        else:
            print(
                f"Rows enabled:     {runtime_calibration.get('enabled_rows', 0)} / "
                f"{runtime_calibration.get('rows', 0)}"
            )
            behaviors = runtime_calibration.get("predicted_behaviors") or {}
            if behaviors:
                formatted = ", ".join(
                    f"{behavior}:{count}" for behavior, count in sorted(behaviors.items())
                )
                print(f"Behaviors:        {formatted}")
            caps = runtime_calibration.get("failure_mode_caps") or {}
            if caps:
                formatted = ", ".join(
                    f"{reason}:{count}" for reason, count in sorted(caps.items())
                )
                print(f"Failure caps:     {formatted}")
            else:
                print("Failure caps:     none")

    risk_coverage = analysis.get("risk_coverage", {})
    if risk_coverage:
        print("\nRisk-Coverage")
        print("-" * 72)
        if risk_coverage.get("available") is False:
            print(f"Unavailable: {risk_coverage.get('reason', 'missing confidence/correctness rows')}")
        else:
            print(f"AURC: {float(risk_coverage.get('aurc') or 0.0):.4f}")
            for target, coverage in sorted(
                (risk_coverage.get("coverage_at_accuracy") or {}).items()
            ):
                print(f"coverage@accuracy>={target}: {float(coverage):.4f}")

    print("\nBehavior Confusion")
    print("-" * 72)
    for expected, actual_counts in sorted(analysis["behavior_confusion"].items()):
        actual = ", ".join(f"{k}:{v}" for k, v in sorted(actual_counts.items()))
        print(f"{expected:<26s} -> {actual}")

    examples = summary.get("top_behavior_failures", [])[:max_examples]
    if examples:
        print("\nTop Behavior Failures")
        print("-" * 72)
        for row in examples:
            question = (row.get("question") or "").replace("\n", " ").strip()
            print(
                f"{row.get('question_id')} [{row.get('question_type')}] "
                f"{row.get('expected_behavior')} -> {row.get('actual_behavior')} "
                f"F1={float(row.get('answer_f1') or 0.0):.3f} "
                f"EvR={float(row.get('evidence_recall') or 0.0):.3f}"
            )
            print(f"  {question[:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a saved VeraBench report JSON")
    parser.add_argument("report", help="Path to a run_verabench.py JSON report")
    parser.add_argument("--max-examples", type=int, default=10, help="Number of failure examples")
    parser.add_argument("--risk-coverage-svg", help="Optional path to write a risk-coverage SVG curve")
    parser.add_argument("--risk-coverage-csv", help="Optional path to write risk-coverage curve points as CSV")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        raise SystemExit(f"Report not found: {path}")

    analysis = analyze(_load_report(str(path)), max_examples=args.max_examples)
    try:
        if args.risk_coverage_svg:
            _write_risk_coverage_svg(analysis["risk_coverage"], Path(args.risk_coverage_svg))
        if args.risk_coverage_csv:
            _write_risk_coverage_csv(analysis["risk_coverage"], Path(args.risk_coverage_csv))
    except ValueError as exc:
        raise SystemExit(f"Cannot write risk-coverage artifact: {exc}") from exc

    if args.json:
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
    else:
        _print_table(analysis, max_examples=args.max_examples)
        if args.risk_coverage_svg:
            print(f"\nWrote risk-coverage SVG: {args.risk_coverage_svg}")
        if args.risk_coverage_csv:
            print(f"Wrote risk-coverage CSV: {args.risk_coverage_csv}")


if __name__ == "__main__":
    main()
