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
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_report(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        if r.get("question_type") == "conflict"
        and (
            r.get("actual_behavior") != r.get("expected_behavior")
            or float(r.get("conflict_detection_f1") or 0.0) < 0.5
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
        "dominant_failure": dominant_failure,
    }


def analyze(report: dict[str, Any], max_examples: int = 10) -> dict[str, Any]:
    rows = _question_results(report)
    embedded_summary = report.get("failure_summary") or {}
    summary = embedded_summary or _build_summary(rows, max_examples=max_examples)
    confusion = report.get("behavior_confusion") or _behavior_confusion(rows)
    calibration_bins = report.get("calibration_bins") or _calibration_bins(rows)
    conflict_summary = report.get("conflict_summary") or _conflict_summary(rows)
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
        "conflict_summary": conflict_summary,
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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        raise SystemExit(f"Report not found: {path}")

    analysis = analyze(_load_report(str(path)), max_examples=args.max_examples)
    if args.json:
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
    else:
        _print_table(analysis, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
