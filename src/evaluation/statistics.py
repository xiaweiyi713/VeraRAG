"""Deterministic statistical inference for VeraBench reports."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

BOOTSTRAP_VERSION = "stratified-question-bootstrap-v1"
DEPENDENCY_BOOTSTRAP_VERSION = "evidence-cluster-bootstrap-v1"
PAIR_DEPENDENCY_BOOTSTRAP_VERSION = "binary-pair-dependency-bootstrap-v1"
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 1729

_LOWER_IS_BETTER = {"ece", "brier_score", "latency_seconds"}
_MEAN_FIELD_METRICS = {
    "answer_f1": "answer_f1",
    "evidence_recall": "evidence_recall",
    "evidence_precision": "evidence_precision",
    "latency_seconds": "latency_seconds",
    "citation_precision": "citation_precision",
    "citation_recall": "citation_recall",
    "citation_f1": "citation_f1",
    "supporting_fact_precision": "supporting_fact_precision",
    "supporting_fact_recall": "supporting_fact_recall",
    "supporting_fact_f1": "supporting_fact_f1",
}


def _get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _has_field(row: Any, name: str) -> bool:
    if isinstance(row, Mapping):
        return name in row
    return hasattr(row, name)


def _has_any_field(rows: Sequence[Any], name: str) -> bool:
    return any(_has_field(row, name) for row in rows)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def _ece(rows: Sequence[Any], n_bins: int = 10) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    score = 0.0
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if index == n_bins - 1:
            bucket = [
                row for row in rows
                if lower <= float(_get(row, "confidence", 0.0)) <= upper
            ]
        else:
            bucket = [
                row for row in rows
                if lower <= float(_get(row, "confidence", 0.0)) < upper
            ]
        if not bucket:
            continue
        confidence = _mean([
            float(_get(row, "confidence", 0.0))
            for row in bucket
        ])
        accuracy = _mean([
            1.0 if bool(_get(row, "correct", False)) else 0.0
            for row in bucket
        ])
        score += len(bucket) / total * abs(confidence - accuracy)
    return score


def metric_estimates(rows: Sequence[Any]) -> dict[str, float]:
    """Compute VeraBench aggregate metrics from per-question rows."""
    if not rows:
        return {}

    conflict_tp = sum(int(_get(row, "conflict_true_positives", 0)) for row in rows)
    conflict_fp = sum(int(_get(row, "conflict_false_positives", 0)) for row in rows)
    conflict_fn = sum(int(_get(row, "conflict_false_negatives", 0)) for row in rows)
    premise_tp = sum(
        1
        for row in rows
        if bool(_get(row, "premise_refutation_expected", False))
        and bool(_get(row, "premise_refutation_detected", False))
    )
    premise_fp = sum(
        1
        for row in rows
        if not bool(_get(row, "premise_refutation_expected", False))
        and bool(_get(row, "premise_refutation_detected", False))
    )
    premise_fn = sum(
        1
        for row in rows
        if bool(_get(row, "premise_refutation_expected", False))
        and not bool(_get(row, "premise_refutation_detected", False))
    )

    estimates = {
        "behavior_accuracy": _mean([
            1.0
            if _get(row, "actual_behavior", "") == _get(row, "expected_behavior", "")
            else 0.0
            for row in rows
        ]),
        "correctness_accuracy": _mean([
            1.0 if bool(_get(row, "correct", False)) else 0.0
            for row in rows
        ]),
        "ece": _ece(rows),
        "brier_score": _mean([
            (
                float(_get(row, "confidence", 0.0))
                - (1.0 if bool(_get(row, "correct", False)) else 0.0)
            ) ** 2
            for row in rows
        ]),
    }
    for metric, field in _MEAN_FIELD_METRICS.items():
        if _has_any_field(rows, field):
            estimates[metric] = _mean([
                float(_get(row, field, 0.0))
                for row in rows
            ])
    if conflict_tp + conflict_fp + conflict_fn:
        estimates["conflict_micro_f1"] = _binary_f1(
            conflict_tp,
            conflict_fp,
            conflict_fn,
        )
    if premise_tp + premise_fp + premise_fn:
        estimates["premise_refutation_f1"] = _binary_f1(
            premise_tp,
            premise_fp,
            premise_fn,
        )
    return estimates


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _strata(rows: Sequence[Any]) -> list[list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[str(_get(row, "question_type", "unknown"))].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _sample_strata(strata: Sequence[Sequence[Any]], rng: random.Random) -> list[Any]:
    sampled: list[Any] = []
    for group in strata:
        sampled.extend(group[rng.randrange(len(group))] for _ in range(len(group)))
    return sampled


def stratified_bootstrap_confidence_intervals(
    rows: Sequence[Any],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Estimate percentile confidence intervals with question-type stratification."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not rows:
        return {
            "method": BOOTSTRAP_VERSION,
            "confidence_level": confidence_level,
            "resamples": resamples,
            "seed": seed,
            "stratified_by": "question_type",
            "effective_resamples_by_metric": {},
            "metrics": {},
        }

    point = metric_estimates(rows)
    distributions: dict[str, list[float]] = {name: [] for name in point}
    rng = random.Random(seed)
    strata = _strata(rows)
    for _ in range(resamples):
        sampled = metric_estimates(_sample_strata(strata, rng))
        for name in distributions:
            if name in sampled:
                distributions[name].append(sampled[name])

    alpha = (1.0 - confidence_level) / 2.0
    return {
        "method": BOOTSTRAP_VERSION,
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "stratified_by": "question_type",
        "effective_resamples_by_metric": {
            name: len(values)
            for name, values in distributions.items()
        },
        "metrics": {
            name: {
                "estimate": round(point[name], 6),
                "lower": round(_percentile(values, alpha), 6),
                "upper": round(_percentile(values, 1.0 - alpha), 6),
            }
            for name, values in distributions.items()
        },
    }


def dependency_cluster_bootstrap_confidence_intervals(
    rows: Sequence[Any],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    cluster_field: str = "dependency_group",
) -> dict[str, Any]:
    """Bootstrap shared-evidence clusters as the independent sampling units."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        cluster = str(_get(row, cluster_field, "") or "")
        if not cluster:
            raise ValueError(
                f"all rows require a non-empty {cluster_field} for cluster bootstrap"
            )
        grouped[cluster].append(row)
    clusters = [grouped[key] for key in sorted(grouped)]
    if not rows:
        return {
            "method": DEPENDENCY_BOOTSTRAP_VERSION,
            "confidence_level": confidence_level,
            "resamples": resamples,
            "seed": seed,
            "clustered_by": cluster_field,
            "clusters": 0,
            "cluster_sizes": [],
            "effective_resamples_by_metric": {},
            "metrics": {},
        }

    point = metric_estimates(rows)
    distributions: dict[str, list[float]] = {name: [] for name in point}
    rng = random.Random(seed)
    for _ in range(resamples):
        sampled: list[Any] = []
        for _ in range(len(clusters)):
            sampled.extend(clusters[rng.randrange(len(clusters))])
        estimates = metric_estimates(sampled)
        for name in distributions:
            if name in estimates:
                distributions[name].append(estimates[name])

    alpha = (1.0 - confidence_level) / 2.0
    return {
        "method": DEPENDENCY_BOOTSTRAP_VERSION,
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "clustered_by": cluster_field,
        "clusters": len(clusters),
        "cluster_sizes": sorted((len(cluster) for cluster in clusters), reverse=True),
        "effective_resamples_by_metric": {
            name: len(values)
            for name, values in distributions.items()
        },
        "metrics": {
            name: {
                "estimate": round(point[name], 6),
                "lower": round(_percentile(values, alpha), 6),
                "upper": round(_percentile(values, 1.0 - alpha), 6),
            }
            for name, values in distributions.items()
        },
    }


def _binary_classification_estimates(
    rows: Sequence[Any],
    *,
    label_field: str,
    prediction_field: str,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        label = int(_get(row, label_field, 0))
        predicted = int(_get(row, prediction_field, 0))
        if label == 1 and predicted == 1:
            tp += 1
        elif label == 0 and predicted == 1:
            fp += 1
        elif label == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    estimates: dict[str, float] = {}
    total = tp + fp + tn + fn
    if total:
        estimates["accuracy"] = (tp + tn) / total
    if tp + fp:
        estimates["precision"] = tp / (tp + fp)
    if tp + fn:
        estimates["recall"] = tp / (tp + fn)
    if 2 * tp + fp + fn:
        estimates["f1"] = _binary_f1(tp, fp, fn)
    return estimates


def binary_pair_dependency_bootstrap_confidence_intervals(
    rows: Sequence[Any],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    cluster_field: str = "evaluation_dependency_component",
    label_field: str = "label",
    prediction_field: str = "predicted",
) -> dict[str, Any]:
    """Bootstrap pair-classifier metrics over dependency-connected units."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        cluster = str(_get(row, cluster_field, "") or "")
        if not cluster:
            raise ValueError(
                f"all rows require a non-empty {cluster_field} for pair bootstrap"
            )
        grouped[cluster].append(row)
    clusters = [grouped[key] for key in sorted(grouped)]
    if not rows:
        return {
            "method": PAIR_DEPENDENCY_BOOTSTRAP_VERSION,
            "confidence_level": confidence_level,
            "resamples": resamples,
            "seed": seed,
            "clustered_by": cluster_field,
            "clusters": 0,
            "cluster_sizes": [],
            "effective_resamples_by_metric": {},
            "metrics": {},
        }

    point = _binary_classification_estimates(
        rows,
        label_field=label_field,
        prediction_field=prediction_field,
    )
    distributions: dict[str, list[float]] = {name: [] for name in point}
    rng = random.Random(seed)
    for _ in range(resamples):
        sampled: list[Any] = []
        for _ in range(len(clusters)):
            sampled.extend(clusters[rng.randrange(len(clusters))])
        estimates = _binary_classification_estimates(
            sampled,
            label_field=label_field,
            prediction_field=prediction_field,
        )
        for name in distributions:
            if name in estimates:
                distributions[name].append(estimates[name])

    alpha = (1.0 - confidence_level) / 2.0
    return {
        "method": PAIR_DEPENDENCY_BOOTSTRAP_VERSION,
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "clustered_by": cluster_field,
        "clusters": len(clusters),
        "cluster_sizes": sorted((len(cluster) for cluster in clusters), reverse=True),
        "effective_resamples_by_metric": {
            name: len(values)
            for name, values in distributions.items()
        },
        "metrics": {
            name: {
                "estimate": round(point[name], 6),
                "lower": round(_percentile(values, alpha), 6),
                "upper": round(_percentile(values, 1.0 - alpha), 6),
            }
            for name, values in distributions.items()
        },
    }


def _paired_outcomes(
    baseline_rows: Sequence[Any],
    candidate_rows: Sequence[Any],
    metric: str,
) -> dict[str, int] | None:
    def value(row: Any) -> float:
        if metric == "behavior_accuracy":
            return float(
                _get(row, "actual_behavior", "")
                == _get(row, "expected_behavior", "")
            )
        if metric == "correctness_accuracy":
            return float(bool(_get(row, "correct", False)))
        if metric == "brier_score":
            confidence = float(_get(row, "confidence", 0.0))
            label = 1.0 if bool(_get(row, "correct", False)) else 0.0
            return (confidence - label) ** 2
        field = _MEAN_FIELD_METRICS.get(metric)
        if field is None:
            return 0.0
        return float(_get(row, field, 0.0))

    if metric not in {
        *_MEAN_FIELD_METRICS,
        "behavior_accuracy",
        "correctness_accuracy",
        "brier_score",
    }:
        return None

    higher_is_better = metric not in _LOWER_IS_BETTER
    wins = ties = losses = 0
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        delta = value(candidate) - value(baseline)
        if abs(delta) <= 1e-12:
            ties += 1
        elif (delta > 0) == higher_is_better:
            wins += 1
        else:
            losses += 1
    return {"candidate_wins": wins, "ties": ties, "candidate_losses": losses}


def _mcnemar_exact(baseline_rows: Sequence[Any], candidate_rows: Sequence[Any]) -> dict[str, Any]:
    baseline_only = 0
    candidate_only = 0
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        baseline_ok = (
            _get(baseline, "actual_behavior", "")
            == _get(baseline, "expected_behavior", "")
        )
        candidate_ok = (
            _get(candidate, "actual_behavior", "")
            == _get(candidate, "expected_behavior", "")
        )
        if baseline_ok and not candidate_ok:
            baseline_only += 1
        elif candidate_ok and not baseline_ok:
            candidate_only += 1
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(baseline_only, candidate_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p": round(p_value, 6),
    }


def _summarize_paired_metrics(
    baseline_rows: Sequence[Any],
    candidate_rows: Sequence[Any],
    point_a: Mapping[str, float],
    point_b: Mapping[str, float],
    distributions: Mapping[str, Sequence[float]],
    confidence_level: float,
) -> dict[str, Any]:
    alpha = (1.0 - confidence_level) / 2.0
    metrics: dict[str, Any] = {}
    for name in sorted(distributions):
        deltas = distributions[name]
        if not deltas:
            continue
        direction = "lower_is_better" if name in _LOWER_IS_BETTER else "higher_is_better"
        if direction == "higher_is_better":
            probability_better = sum(delta > 0 for delta in deltas) / len(deltas)
        else:
            probability_better = sum(delta < 0 for delta in deltas) / len(deltas)
        metrics[name] = {
            "baseline": round(point_a[name], 6),
            "candidate": round(point_b[name], 6),
            "delta_candidate_minus_baseline": round(
                point_b[name] - point_a[name],
                6,
            ),
            "delta_lower": round(_percentile(deltas, alpha), 6),
            "delta_upper": round(_percentile(deltas, 1.0 - alpha), 6),
            "direction": direction,
            "probability_candidate_better": round(probability_better, 6),
            "paired_outcomes": _paired_outcomes(
                baseline_rows,
                candidate_rows,
                name,
            ),
        }
    return metrics


def _paired_dependency_cluster_comparison(
    baseline_rows: Sequence[Any],
    candidate_rows: Sequence[Any],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
    cluster_field: str = "dependency_group",
) -> dict[str, Any] | None:
    pair_clusters: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        baseline_cluster = str(_get(baseline, cluster_field, "") or "")
        candidate_cluster = str(_get(candidate, cluster_field, "") or "")
        if not baseline_cluster or not candidate_cluster:
            return None
        if baseline_cluster != candidate_cluster:
            raise ValueError("paired rows disagree on dependency_group")
        pair_clusters[baseline_cluster].append((baseline, candidate))

    point_a = metric_estimates(baseline_rows)
    point_b = metric_estimates(candidate_rows)
    metric_names = sorted(set(point_a) & set(point_b))
    distributions: dict[str, list[float]] = {
        name: [] for name in metric_names
    }
    clusters = [pair_clusters[key] for key in sorted(pair_clusters)]
    rng = random.Random(seed)
    for _ in range(resamples):
        sampled_a: list[Any] = []
        sampled_b: list[Any] = []
        for _ in range(len(clusters)):
            for baseline, candidate in clusters[rng.randrange(len(clusters))]:
                sampled_a.append(baseline)
                sampled_b.append(candidate)
        estimates_a = metric_estimates(sampled_a)
        estimates_b = metric_estimates(sampled_b)
        for name in metric_names:
            if name in estimates_a and name in estimates_b:
                distributions[name].append(estimates_b[name] - estimates_a[name])

    return {
        "method": "paired-evidence-cluster-bootstrap-v1",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "clustered_by": cluster_field,
        "clusters": len(clusters),
        "cluster_sizes": sorted((len(cluster) for cluster in clusters), reverse=True),
        "metrics": _summarize_paired_metrics(
            baseline_rows,
            candidate_rows,
            point_a,
            point_b,
            distributions,
            confidence_level,
        ),
        "effective_resamples_by_metric": {
            name: len(values)
            for name, values in distributions.items()
        },
    }


def paired_bootstrap_comparison(
    baseline_rows: Sequence[Any],
    candidate_rows: Sequence[Any],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    resamples: int = 5000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare aligned runs using paired, question-type-stratified bootstrap."""
    if len(baseline_rows) != len(candidate_rows) or not baseline_rows:
        raise ValueError("paired comparison requires equal non-empty row sets")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    baseline_ids = [str(_get(row, "question_id", "")) for row in baseline_rows]
    candidate_ids = [str(_get(row, "question_id", "")) for row in candidate_rows]
    if baseline_ids != candidate_ids:
        raise ValueError("paired rows must be aligned by question_id")

    point_a = metric_estimates(baseline_rows)
    point_b = metric_estimates(candidate_rows)
    metric_names = sorted(set(point_a) & set(point_b))
    distributions: dict[str, list[float]] = {
        name: [] for name in metric_names
    }

    pair_strata: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        question_type = str(_get(baseline, "question_type", "unknown"))
        if question_type != str(_get(candidate, "question_type", "unknown")):
            raise ValueError("paired rows disagree on question_type")
        pair_strata[question_type].append((baseline, candidate))

    rng = random.Random(seed)
    for _ in range(resamples):
        sampled_a: list[Any] = []
        sampled_b: list[Any] = []
        for key in sorted(pair_strata):
            group = pair_strata[key]
            for _ in range(len(group)):
                baseline, candidate = group[rng.randrange(len(group))]
                sampled_a.append(baseline)
                sampled_b.append(candidate)
        sampled_estimates_a = metric_estimates(sampled_a)
        sampled_estimates_b = metric_estimates(sampled_b)
        for name in metric_names:
            if name in sampled_estimates_a and name in sampled_estimates_b:
                distributions[name].append(
                    sampled_estimates_b[name] - sampled_estimates_a[name]
                )

    result = {
        "method": "paired-stratified-question-bootstrap-v1",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "stratified_by": "question_type",
        "questions": len(baseline_rows),
        "effective_resamples_by_metric": {
            name: len(values)
            for name, values in distributions.items()
        },
        "metrics": _summarize_paired_metrics(
            baseline_rows,
            candidate_rows,
            point_a,
            point_b,
            distributions,
            confidence_level,
        ),
        "behavior_mcnemar_exact": _mcnemar_exact(
            baseline_rows,
            candidate_rows,
        ),
    }
    dependency_robust = _paired_dependency_cluster_comparison(
        baseline_rows,
        candidate_rows,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    if dependency_robust is not None:
        result["dependency_robust"] = dependency_robust
    return result
