#!/usr/bin/env python3
"""Post-hoc confidence calibration for saved VeraBench reports."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.calibration_curve import compute_brier_score, compute_calibration  # noqa: E402

Method = Literal["platt", "temperature"]
GroupFallback = Literal["smoothed_constant", "global"]


@dataclass(frozen=True)
class CalibrationModel:
    method: Method
    a: float = 1.0
    b: float = 0.0
    temperature: float = 1.0

    def transform(self, confidences: np.ndarray) -> np.ndarray:
        logits = _logit(confidences)
        if self.method == "temperature":
            return _sigmoid(logits / self.temperature)
        return _sigmoid(self.a * logits + self.b)


@dataclass(frozen=True)
class CalibrationDataset:
    confidences: np.ndarray
    labels: np.ndarray
    calibration_indices: list[int]
    holdout_indices: list[int]


@dataclass(frozen=True)
class GroupCalibrationResult:
    calibrated: np.ndarray
    row_diagnostics: list[dict[str, Any]]
    groups: dict[str, Any]


def calibrate_report(
    report: dict[str, Any],
    *,
    method: Method = "platt",
    correctness_field: str = "correct",
    group_field: str | None = None,
    min_group_rows: int = 8,
    group_fallback: GroupFallback = "smoothed_constant",
    calibration_fraction: float = 0.5,
    seed: int = 1729,
    bins: int = 10,
    max_iter: int = 5000,
    lr: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit post-hoc calibration on a deterministic split and apply to a report."""
    _validate_method(method)
    _validate_group_options(group_field, min_group_rows, group_fallback)
    rows = _extract_rows(report)
    dataset = _build_dataset(
        rows,
        correctness_field=correctness_field,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    model = fit_calibration_model(
        dataset.confidences[dataset.calibration_indices],
        dataset.labels[dataset.calibration_indices],
        method=method,
        max_iter=max_iter,
        lr=lr,
    )
    row_diagnostics: list[dict[str, Any]] | None = None
    group_summaries: dict[str, Any] | None = None
    if group_field:
        group_result = _calibrate_by_group(
            rows,
            dataset=dataset,
            global_model=model,
            method=method,
            group_field=group_field,
            min_group_rows=min_group_rows,
            group_fallback=group_fallback,
            bins=bins,
            max_iter=max_iter,
            lr=lr,
        )
        calibrated = group_result.calibrated
        row_diagnostics = group_result.row_diagnostics
        group_summaries = group_result.groups
    else:
        calibrated = model.transform(dataset.confidences)
    summary = _build_summary(
        model=model,
        dataset=dataset,
        original=dataset.confidences,
        calibrated=calibrated,
        correctness_field=correctness_field,
        group_field=group_field,
        min_group_rows=min_group_rows,
        group_fallback=group_fallback,
        group_summaries=group_summaries,
        calibration_fraction=calibration_fraction,
        seed=seed,
        bins=bins,
    )
    output = _apply_calibration(
        report,
        calibrated,
        dataset,
        summary,
        row_diagnostics=row_diagnostics,
        bins=bins,
    )
    return output, summary


def fit_calibration_model(
    confidences: np.ndarray,
    labels: np.ndarray,
    *,
    method: Method,
    max_iter: int = 5000,
    lr: float = 0.05,
) -> CalibrationModel:
    """Fit a calibration model from probabilities and binary labels."""
    _validate_method(method)
    _validate_training_vectors(confidences, labels)
    if method == "temperature":
        return CalibrationModel(method="temperature", temperature=_fit_temperature(confidences, labels))
    a, b = _fit_platt(confidences, labels, max_iter=max_iter, lr=lr)
    return CalibrationModel(method="platt", a=a, b=b)


def _extract_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("question_results") or report.get("results") or []
    if not rows:
        raise ValueError("report contains no question_results/results rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("report rows must be JSON objects")
    return rows


def _build_dataset(
    rows: list[dict[str, Any]],
    *,
    correctness_field: str,
    calibration_fraction: float,
    seed: int,
) -> CalibrationDataset:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    confidences = np.array(
        [_coerce_confidence(row.get("confidence"), idx) for idx, row in enumerate(rows)],
        dtype=float,
    )
    labels = np.array(
        [_coerce_correctness(row, correctness_field, idx) for idx, row in enumerate(rows)],
        dtype=float,
    )
    calibration_indices, holdout_indices = _stratified_hash_split(
        rows,
        labels,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    _validate_training_vectors(confidences[calibration_indices], labels[calibration_indices])
    if not holdout_indices:
        raise ValueError("holdout split is empty")
    return CalibrationDataset(confidences, labels, calibration_indices, holdout_indices)


def _stratified_hash_split(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    grouped: dict[float, list[int]] = {0.0: [], 1.0: []}
    for idx, label in enumerate(labels):
        grouped[float(label)].append(idx)
    if not grouped[0.0] or not grouped[1.0]:
        raise ValueError("post-hoc calibration requires both correct and incorrect rows")

    calibration: set[int] = set()
    for indices in grouped.values():
        ranked = sorted(
            indices,
            key=lambda idx: _stable_hash_score(_row_split_key(rows[idx], idx), seed),
        )
        take = round(len(ranked) * calibration_fraction)
        take = max(1, min(len(ranked) - 1, take)) if len(ranked) > 1 else 1
        calibration.update(ranked[:take])

    all_indices = set(range(len(rows)))
    holdout = sorted(all_indices - calibration)
    return sorted(calibration), holdout


def _row_split_key(row: dict[str, Any], idx: int) -> str:
    return str(row.get("question_id") or row.get("id") or idx)


def _stable_hash_score(key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest, 16)


def _coerce_confidence(value: Any, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"row {row_index} confidence must be a number in [0, 1]")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"row {row_index} confidence must be finite and in [0, 1]")
    return confidence


def _coerce_correctness(row: dict[str, Any], correctness_field: str, row_index: int) -> float:
    if correctness_field not in row:
        raise ValueError(f"row {row_index} missing correctness field {correctness_field!r}")
    value = row[correctness_field]
    if not isinstance(value, bool):
        raise ValueError(f"row {row_index} correctness field {correctness_field!r} must be boolean")
    return 1.0 if value else 0.0


def _validate_training_vectors(confidences: np.ndarray, labels: np.ndarray) -> None:
    if confidences.shape != labels.shape:
        raise ValueError("confidences and labels must have the same shape")
    if confidences.size < 2:
        raise ValueError("calibration requires at least two training rows")
    if not np.all(np.isfinite(confidences)) or not np.all((confidences >= 0.0) & (confidences <= 1.0)):
        raise ValueError("confidences must be finite probabilities in [0, 1]")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("labels must be binary")
    if len(set(labels.tolist())) < 2:
        raise ValueError("calibration training split requires both classes")


def _fit_platt(
    confidences: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int,
    lr: float,
) -> tuple[float, float]:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    x = _logit(confidences)
    positive_rate = float(np.clip(labels.mean(), 1e-4, 1.0 - 1e-4))
    a = 1.0
    b = float(math.log(positive_rate / (1.0 - positive_rate)))
    l2 = 1e-3
    for _ in range(max_iter):
        probs = _sigmoid(a * x + b)
        error = probs - labels
        grad_a = float(np.mean(error * x) + l2 * (a - 1.0))
        grad_b = float(np.mean(error) + l2 * b)
        a -= lr * grad_a
        b -= lr * grad_b
        if abs(grad_a) + abs(grad_b) < 1e-8:
            break
    return float(a), float(b)


def _fit_temperature(confidences: np.ndarray, labels: np.ndarray) -> float:
    logits = _logit(confidences)

    def nll(temperature: float) -> float:
        probs = np.clip(_sigmoid(logits / temperature), 1e-9, 1.0 - 1e-9)
        return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))

    best_temperature = 1.0
    best_loss = nll(best_temperature)
    for temperature in np.geomspace(0.05, 20.0, num=160):
        loss = nll(float(temperature))
        if loss < best_loss:
            best_temperature = float(temperature)
            best_loss = loss
    return best_temperature


def _build_summary(
    *,
    model: CalibrationModel,
    dataset: CalibrationDataset,
    original: np.ndarray,
    calibrated: np.ndarray,
    correctness_field: str,
    group_field: str | None,
    min_group_rows: int,
    group_fallback: GroupFallback,
    group_summaries: dict[str, Any] | None,
    calibration_fraction: float,
    seed: int,
    bins: int,
) -> dict[str, Any]:
    labels = dataset.labels
    split_masks = {
        "all": list(range(len(labels))),
        "calibration": dataset.calibration_indices,
        "holdout": dataset.holdout_indices,
    }
    metrics = {
        name: {
            "rows": len(indices),
            "before": _metric_summary(original[indices], labels[indices], bins),
            "after": _metric_summary(calibrated[indices], labels[indices], bins),
        }
        for name, indices in split_masks.items()
    }
    payload: dict[str, Any] = {
        "method": model.method,
        "correctness_field": correctness_field,
        "calibration_fraction": calibration_fraction,
        "seed": seed,
        "rows": len(labels),
        "calibration_rows": len(dataset.calibration_indices),
        "holdout_rows": len(dataset.holdout_indices),
        "metrics": metrics,
    }
    if group_field:
        payload["group_field"] = group_field
        payload["min_group_rows"] = min_group_rows
        payload["group_fallback"] = group_fallback
        payload["groups"] = group_summaries or {}
    if model.method == "temperature":
        payload["temperature"] = round(model.temperature, 8)
    else:
        payload["platt"] = {"a": round(model.a, 8), "b": round(model.b, 8)}
    return payload


def _calibrate_by_group(
    rows: list[dict[str, Any]],
    *,
    dataset: CalibrationDataset,
    global_model: CalibrationModel,
    method: Method,
    group_field: str,
    min_group_rows: int,
    group_fallback: GroupFallback,
    bins: int,
    max_iter: int,
    lr: float,
) -> GroupCalibrationResult:
    group_indices = _group_row_indices(rows, group_field)
    calibrated = np.empty_like(dataset.confidences, dtype=float)
    row_diagnostics: list[dict[str, Any]] = [{} for _ in rows]
    group_summaries: dict[str, Any] = {}
    calibration_set = set(dataset.calibration_indices)
    holdout_set = set(dataset.holdout_indices)
    global_calibration_labels = dataset.labels[dataset.calibration_indices]

    for group_value, indices in sorted(group_indices.items()):
        calibration_indices = [idx for idx in indices if idx in calibration_set]
        holdout_indices = [idx for idx in indices if idx in holdout_set]
        labels = dataset.labels[calibration_indices]
        fallback_reason = _group_fallback_reason(labels, min_group_rows)
        fitted_model: CalibrationModel | None = None
        constant_probability: float | None = None

        if fallback_reason is None:
            fitted_model = fit_calibration_model(
                dataset.confidences[calibration_indices],
                labels,
                method=method,
                max_iter=max_iter,
                lr=lr,
            )
            group_calibrated = fitted_model.transform(dataset.confidences[indices])
            mode = f"group_{method}"
            model_scope = "group"
        elif group_fallback == "global":
            group_calibrated = global_model.transform(dataset.confidences[indices])
            mode = "global_fallback"
            model_scope = "fallback"
        else:
            constant_probability = _smoothed_positive_rate(labels, global_calibration_labels)
            group_calibrated = np.full(len(indices), constant_probability, dtype=float)
            mode = "smoothed_constant"
            model_scope = "fallback"

        calibrated[indices] = group_calibrated
        for offset, row_idx in enumerate(indices):
            row_diagnostics[row_idx] = {
                "group_field": group_field,
                "group_value": group_value,
                "model_scope": model_scope,
                "mode": mode,
            }
            if fallback_reason:
                row_diagnostics[row_idx]["fallback_reason"] = fallback_reason
            if constant_probability is not None:
                row_diagnostics[row_idx]["constant_probability"] = round(constant_probability, 8)
            row_diagnostics[row_idx]["group_confidence"] = round(float(group_calibrated[offset]), 6)

        group_summary: dict[str, Any] = {
            "rows": len(indices),
            "calibration_rows": len(calibration_indices),
            "holdout_rows": len(holdout_indices),
            "mode": mode,
            "metrics": {
                "all": {
                    "before": _metric_summary(dataset.confidences[indices], dataset.labels[indices], bins),
                    "after": _metric_summary(calibrated[indices], dataset.labels[indices], bins),
                },
            },
        }
        if calibration_indices:
            group_summary["metrics"]["calibration"] = {
                "before": _metric_summary(dataset.confidences[calibration_indices], dataset.labels[calibration_indices], bins),
                "after": _metric_summary(calibrated[calibration_indices], dataset.labels[calibration_indices], bins),
            }
        if holdout_indices:
            group_summary["metrics"]["holdout"] = {
                "before": _metric_summary(dataset.confidences[holdout_indices], dataset.labels[holdout_indices], bins),
                "after": _metric_summary(calibrated[holdout_indices], dataset.labels[holdout_indices], bins),
            }
        if fallback_reason:
            group_summary["fallback_reason"] = fallback_reason
        if fitted_model:
            if fitted_model.method == "temperature":
                group_summary["temperature"] = round(fitted_model.temperature, 8)
            else:
                group_summary["platt"] = {"a": round(fitted_model.a, 8), "b": round(fitted_model.b, 8)}
        if constant_probability is not None:
            group_summary["constant_probability"] = round(constant_probability, 8)
        group_summaries[group_value] = group_summary

    return GroupCalibrationResult(calibrated=calibrated, row_diagnostics=row_diagnostics, groups=group_summaries)


def _group_row_indices(rows: list[dict[str, Any]], group_field: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        if group_field not in row:
            raise ValueError(f"row {idx} missing group field {group_field!r}")
        group_value = _coerce_group_value(row[group_field], group_field, idx)
        groups.setdefault(group_value, []).append(idx)
    return groups


def _coerce_group_value(value: Any, group_field: str, row_index: int) -> str:
    if value is None or isinstance(value, dict | list):
        raise ValueError(f"row {row_index} group field {group_field!r} must be a scalar value")
    text = str(value).strip()
    if not text:
        raise ValueError(f"row {row_index} group field {group_field!r} must be non-empty")
    return text


def _group_fallback_reason(labels: np.ndarray, min_group_rows: int) -> str | None:
    if labels.size < min_group_rows:
        return f"calibration rows {labels.size} below min_group_rows {min_group_rows}"
    if len(set(labels.tolist())) < 2:
        return "calibration split has only one correctness class"
    return None


def _smoothed_positive_rate(labels: np.ndarray, fallback_labels: np.ndarray) -> float:
    if labels.size == 0:
        labels = fallback_labels
    return float((labels.sum() + 1.0) / (labels.size + 2.0))


def _metric_summary(confidences: np.ndarray, labels: np.ndarray, bins: int) -> dict[str, Any]:
    calibration_bins, ece = compute_calibration(confidences, labels, n_bins=bins)
    return {
        "avg_confidence": round(float(confidences.mean()), 6),
        "accuracy": round(float(labels.mean()), 6),
        "ece": round(float(ece), 6),
        "brier_score": round(compute_brier_score(confidences, labels), 6),
        "calibration_bins": calibration_bins,
    }


def _apply_calibration(
    report: dict[str, Any],
    calibrated: np.ndarray,
    dataset: CalibrationDataset,
    summary: dict[str, Any],
    *,
    row_diagnostics: list[dict[str, Any]] | None = None,
    bins: int,
) -> dict[str, Any]:
    output = copy.deepcopy(report)
    rows = _extract_rows(output)
    calibration_set = set(dataset.calibration_indices)
    for idx, row in enumerate(rows):
        original_confidence = dataset.confidences[idx]
        row["confidence"] = round(float(calibrated[idx]), 6)
        diagnostics = row.setdefault("diagnostics", {})
        if isinstance(diagnostics, dict):
            diagnostics["confidence_calibration"] = {
                "original_confidence": round(float(original_confidence), 6),
                "split": "calibration" if idx in calibration_set else "holdout",
                "method": summary["method"],
            }
            if row_diagnostics:
                diagnostics["confidence_calibration"].update(row_diagnostics[idx])

    labels = dataset.labels
    output["avg_confidence"] = round(float(calibrated.mean()), 6)
    output["ece"] = round(float(compute_calibration(calibrated, labels, n_bins=bins)[1]), 6)
    output["brier_score"] = round(compute_brier_score(calibrated, labels), 6)
    output["calibration_bins"] = compute_calibration(calibrated, labels, n_bins=bins)[0]
    metadata = output.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["posthoc_confidence_calibration"] = summary
    else:
        output["metadata"] = {"posthoc_confidence_calibration": summary}
    return output


def _logit(confidences: np.ndarray) -> np.ndarray:
    clipped = np.clip(confidences.astype(float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _validate_method(method: str) -> None:
    if method not in {"platt", "temperature"}:
        raise ValueError("method must be 'platt' or 'temperature'")


def _validate_group_options(group_field: str | None, min_group_rows: int, group_fallback: str) -> None:
    if group_field is not None and not group_field.strip():
        raise ValueError("group_field must be non-empty when provided")
    if min_group_rows < 2:
        raise ValueError("min_group_rows must be at least 2")
    if group_fallback not in {"smoothed_constant", "global"}:
        raise ValueError("group_fallback must be 'smoothed_constant' or 'global'")


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("input report must be a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a saved VeraBench report JSON.")
    parser.add_argument("--output", required=True, help="Path to write the calibrated report JSON.")
    parser.add_argument("--method", choices=["platt", "temperature"], default="platt")
    parser.add_argument("--correctness-field", default="correct")
    parser.add_argument("--group-field", help="Optional row field for separate behavior-level calibration.")
    parser.add_argument("--min-group-rows", type=int, default=8)
    parser.add_argument("--group-fallback", choices=["smoothed_constant", "global"], default="smoothed_constant")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--summary-output", help="Optional path to write calibration summary JSON.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    args = parser.parse_args(argv)

    try:
        report = _load_json(args.input)
        calibrated_report, summary = calibrate_report(
            report,
            method=args.method,
            correctness_field=args.correctness_field,
            group_field=args.group_field,
            min_group_rows=args.min_group_rows,
            group_fallback=args.group_fallback,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
            bins=args.bins,
            max_iter=args.max_iter,
            lr=args.lr,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"calibration input error: {exc}\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibrated_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        holdout = summary["metrics"]["holdout"]
        print(
            "Holdout ECE: "
            f"{holdout['before']['ece']:.4f} -> {holdout['after']['ece']:.4f}; "
            "Brier: "
            f"{holdout['before']['brier_score']:.4f} -> {holdout['after']['brier_score']:.4f}"
        )
        print(f"Wrote calibrated report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
