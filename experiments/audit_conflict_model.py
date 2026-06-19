#!/usr/bin/env python3
"""Fail-closed promotion audit for learned conflict detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

POLICY_VERSION = "conflict-model-promotion-v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_metrics(path: Path) -> tuple[dict[str, float], int, bool]:
    tp = fp = tn = fn = 0
    rows = 0
    decisions_consistent = True
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            label = int(row["label"])
            predicted = int(row["predicted"])
            probability = float(row["probability"])
            threshold = float(row["threshold"])
            decisions_consistent &= predicted == int(probability >= threshold)
            if label == 1 and predicted == 1:
                tp += 1
            elif label == 0 and predicted == 1:
                fp += 1
            elif label == 0 and predicted == 0:
                tn += 1
            else:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }, rows, decisions_consistent


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    observed: Any,
    required: Any,
) -> None:
    checks.append({
        "id": check_id,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    })


def _load_run(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "training_metadata.json"
    metrics_path = run_dir / "training_metrics.json"
    metadata = _read_json(metadata_path)
    metrics = _read_json(metrics_path)
    artifact = (metrics.get("prediction_artifacts") or {}).get("test") or {}
    prediction_path = run_dir / str(artifact.get("path") or "")
    artifact_hash_valid = (
        bool(artifact)
        and prediction_path.is_file()
        and artifact.get("sha256") == _sha256(prediction_path)
    )
    selected = (
        ((metrics.get("splits") or {}).get("test") or {})
        .get("selected_threshold")
        or {}
    )
    recomputed_metrics: dict[str, float] = {}
    prediction_rows = 0
    decisions_consistent = False
    if artifact_hash_valid:
        recomputed_metrics, prediction_rows, decisions_consistent = (
            _prediction_metrics(prediction_path)
        )
    metrics_consistent = bool(selected) and all(
        recomputed_metrics.get(name) == selected.get(name)
        for name in ("precision", "recall", "f1")
    )
    artifact_valid = (
        artifact_hash_valid
        and prediction_rows == int(artifact.get("rows") or 0)
        and prediction_rows > 0
        and decisions_consistent
        and metrics_consistent
    )
    uncertainty = (
        ((metrics.get("splits") or {}).get("test") or {})
        .get("dependency_robust_confidence_intervals")
        or {}
    )
    f1_interval = (uncertainty.get("metrics") or {}).get("f1") or {}
    manifest = metadata.get("dataset_manifest") or {}
    manifest_contents = manifest.get("contents") or {}
    benchmark_fingerprints = (
        (manifest_contents.get("benchmark") or {}).get("fingerprints") or {}
    )
    return {
        "path": str(run_dir),
        "seed": metadata.get("seed"),
        "balance_train": metadata.get("balance_train"),
        "split_integrity": (
            (metadata.get("summary") or {}).get("split_integrity") or {}
        ),
        "manifest_sha256": manifest.get("sha256"),
        "manifest_schema": manifest_contents.get("schema_version"),
        "training_benchmark_fingerprints": benchmark_fingerprints,
        "metrics_schema": metrics.get("schema_version"),
        "test_f1": selected.get("f1"),
        "test_precision": selected.get("precision"),
        "test_recall": selected.get("recall"),
        "test_f1_cluster_lower": f1_interval.get("lower"),
        "test_dependency_clusters": uncertainty.get("clusters"),
        "prediction_artifact_valid": artifact_valid,
        "prediction_metrics_recomputed": recomputed_metrics,
    }


def audit_conflict_model(
    run_dirs: list[Path],
    ablation: dict[str, Any],
    *,
    min_seeds: int = 3,
    min_test_f1_mean: float = 0.70,
    min_test_f1_worst: float = 0.50,
    min_cluster_f1_lower: float = 0.30,
    min_ablation_questions: int = 10,
    min_ablation_gold_conflicts: int = 10,
    min_ablation_f1_delta: float = 0.02,
    max_ablation_fp_delta: int = 0,
    allow_internal_heldout: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable promotion decision and every gate result."""
    runs = [_load_run(path.resolve()) for path in run_dirs]
    checks: list[dict[str, Any]] = []

    seeds = [run["seed"] for run in runs]
    _check(
        checks,
        "minimum_distinct_seeds",
        len(set(seeds)) >= min_seeds and len(runs) >= min_seeds,
        observed={"runs": len(runs), "distinct_seeds": len(set(seeds)), "seeds": seeds},
        required={"runs": min_seeds, "distinct_seeds": min_seeds},
    )
    _check(
        checks,
        "verified_dependency_splits",
        bool(runs) and all(
            run["split_integrity"].get("status") == "verified"
            for run in runs
        ),
        observed=[run["split_integrity"].get("status") for run in runs],
        required="verified for every run",
    )
    manifests = [run["manifest_sha256"] for run in runs]
    _check(
        checks,
        "identical_dataset_manifest",
        bool(runs)
        and all(manifests)
        and len(set(manifests)) == 1
        and all(run["manifest_schema"] == "conflict-pairs-v2" for run in runs),
        observed={
            "manifest_sha256": manifests,
            "schema_versions": [run["manifest_schema"] for run in runs],
        },
        required="one conflict-pairs-v2 manifest shared by every run",
    )
    _check(
        checks,
        "auditable_test_predictions",
        bool(runs) and all(run["prediction_artifact_valid"] for run in runs),
        observed=[run["prediction_artifact_valid"] for run in runs],
        required="hash-verified test_predictions.jsonl for every run",
    )
    _check(
        checks,
        "metrics_schema",
        bool(runs) and all(
            run["metrics_schema"] == "conflict-training-metrics-v2"
            for run in runs
        ),
        observed=[run["metrics_schema"] for run in runs],
        required="conflict-training-metrics-v2",
    )

    test_f1 = [
        float(run["test_f1"])
        for run in runs
        if isinstance(run["test_f1"], (int, float))
    ]
    mean_f1 = statistics.mean(test_f1) if len(test_f1) == len(runs) and runs else 0.0
    worst_f1 = min(test_f1) if len(test_f1) == len(runs) and runs else 0.0
    _check(
        checks,
        "multi_seed_test_f1",
        mean_f1 >= min_test_f1_mean and worst_f1 >= min_test_f1_worst,
        observed={"mean": round(mean_f1, 6), "worst": round(worst_f1, 6)},
        required={"mean_at_least": min_test_f1_mean, "worst_at_least": min_test_f1_worst},
    )
    cluster_lowers = [
        float(run["test_f1_cluster_lower"])
        for run in runs
        if isinstance(run["test_f1_cluster_lower"], (int, float))
    ]
    worst_cluster_lower = (
        min(cluster_lowers)
        if len(cluster_lowers) == len(runs) and runs
        else 0.0
    )
    _check(
        checks,
        "dependency_robust_f1_lower_bound",
        worst_cluster_lower >= min_cluster_f1_lower,
        observed=round(worst_cluster_lower, 6),
        required={"worst_seed_lower_bound_at_least": min_cluster_f1_lower},
    )

    scope = ablation.get("evaluation_scope") or {}
    independent = bool(scope.get("independent_test"))
    valid_scope = independent or (
        allow_internal_heldout and scope.get("split") == "test"
    )
    _check(
        checks,
        "independent_evaluation_scope",
        valid_scope,
        observed={
            "independent_test": independent,
            "split": scope.get("split"),
            "evaluation_id": scope.get("evaluation_id"),
        },
        required=(
            "independent external test set"
            if not allow_internal_heldout
            else "independent external test set or explicit internal test split"
        ),
    )
    external_fingerprints = (
        ((scope.get("dataset") or {}).get("fingerprints")) or {}
    )
    training_question_hashes = {
        (run["training_benchmark_fingerprints"] or {}).get("questions_sha256")
        for run in runs
    }
    training_question_hashes.discard(None)
    external_question_hash = external_fingerprints.get("questions_sha256")
    _check(
        checks,
        "independent_dataset_fingerprint",
        (
            not independent
            and allow_internal_heldout
        )
        or (
            bool(external_question_hash)
            and bool(training_question_hashes)
            and external_question_hash not in training_question_hashes
        ),
        observed={
            "external_questions_sha256": external_question_hash,
            "training_questions_sha256": sorted(training_question_hashes),
        },
        required=(
            "external questions fingerprint must differ from every training benchmark"
        ),
    )
    variants = {
        str(variant.get("name")): variant
        for variant in ablation.get("variants") or []
        if isinstance(variant, dict)
    }
    rules = (variants.get("rules") or {}).get("summary") or {}
    learned_variant = variants.get("rules_plus_learned") or {}
    learned = learned_variant.get("summary") or {}
    questions = int(learned.get("questions") or 0)
    gold_conflicts = int(learned.get("gold_conflicts") or 0)
    _check(
        checks,
        "minimum_ablation_sample",
        questions >= min_ablation_questions
        and gold_conflicts >= min_ablation_gold_conflicts,
        observed={"questions": questions, "gold_conflicts": gold_conflicts},
        required={
            "questions_at_least": min_ablation_questions,
            "gold_conflicts_at_least": min_ablation_gold_conflicts,
        },
    )
    _check(
        checks,
        "learned_model_loaded",
        bool(learned_variant.get("learned_available")),
        observed=bool(learned_variant.get("learned_available")),
        required=True,
    )
    f1_delta = float(learned.get("f1") or 0.0) - float(rules.get("f1") or 0.0)
    recall_delta = (
        float(learned.get("recall") or 0.0)
        - float(rules.get("recall") or 0.0)
    )
    fp_delta = (
        int(learned.get("false_positives") or 0)
        - int(rules.get("false_positives") or 0)
    )
    _check(
        checks,
        "heldout_ablation_improvement",
        f1_delta >= min_ablation_f1_delta
        and recall_delta >= 0.0
        and fp_delta <= max_ablation_fp_delta,
        observed={
            "f1_delta": round(f1_delta, 6),
            "recall_delta": round(recall_delta, 6),
            "false_positive_delta": fp_delta,
        },
        required={
            "f1_delta_at_least": min_ablation_f1_delta,
            "recall_delta_at_least": 0.0,
            "false_positive_delta_at_most": max_ablation_fp_delta,
        },
    )

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "decision": "promote" if passed else "reject",
        "passed": passed,
        "policy": {
            "min_seeds": min_seeds,
            "min_test_f1_mean": min_test_f1_mean,
            "min_test_f1_worst": min_test_f1_worst,
            "min_cluster_f1_lower": min_cluster_f1_lower,
            "min_ablation_questions": min_ablation_questions,
            "min_ablation_gold_conflicts": min_ablation_gold_conflicts,
            "min_ablation_f1_delta": min_ablation_f1_delta,
            "max_ablation_fp_delta": max_ablation_fp_delta,
            "allow_internal_heldout": allow_internal_heldout,
        },
        "runs": runs,
        "checks": checks,
        "failed_checks": [
            check["id"] for check in checks if not check["passed"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Training output directories from distinct random seeds.",
    )
    parser.add_argument(
        "--ablation",
        required=True,
        help="Gold-evidence rules vs rules+learned comparison JSON.",
    )
    parser.add_argument("--output", help="Optional audit JSON output path.")
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-test-f1-mean", type=float, default=0.70)
    parser.add_argument("--min-test-f1-worst", type=float, default=0.50)
    parser.add_argument("--min-cluster-f1-lower", type=float, default=0.30)
    parser.add_argument("--min-ablation-questions", type=int, default=10)
    parser.add_argument("--min-ablation-gold-conflicts", type=int, default=10)
    parser.add_argument("--min-ablation-f1-delta", type=float, default=0.02)
    parser.add_argument("--max-ablation-fp-delta", type=int, default=0)
    parser.add_argument(
        "--allow-internal-heldout",
        action="store_true",
        help="Development-only override for a VeraBench internal test split.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit zero after writing the decision.",
    )
    args = parser.parse_args()

    payload = audit_conflict_model(
        [Path(path) for path in args.runs],
        _read_json(Path(args.ablation)),
        min_seeds=args.min_seeds,
        min_test_f1_mean=args.min_test_f1_mean,
        min_test_f1_worst=args.min_test_f1_worst,
        min_cluster_f1_lower=args.min_cluster_f1_lower,
        min_ablation_questions=args.min_ablation_questions,
        min_ablation_gold_conflicts=args.min_ablation_gold_conflicts,
        min_ablation_f1_delta=args.min_ablation_f1_delta,
        max_ablation_fp_delta=args.max_ablation_fp_delta,
        allow_internal_heldout=args.allow_internal_heldout,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    if not payload["passed"] and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
