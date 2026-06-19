#!/usr/bin/env python3
"""Train a CrossEncoder conflict detector from VeraBench pair data.

This script is intentionally optional-dependency friendly: use
``--dry-run`` to validate data and environment without importing or training
sentence-transformers.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistics import (  # noqa: E402
    binary_pair_dependency_bootstrap_confidence_intervals,
)


def _load_jsonl(
    path: str | Path,
    *,
    required: bool = True,
) -> list[dict[str, Any]]:
    if not Path(path).exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _balanced_limit(rows: list[dict[str, Any]], max_samples: int | None = None) -> list[dict[str, Any]]:
    if max_samples is None or len(rows) <= max_samples:
        return rows
    positives = [row for row in rows if int(row.get("label", 0)) == 1]
    negatives = [row for row in rows if int(row.get("label", 0)) == 0]
    positive_target = min(len(positives), max(1, max_samples // 2))
    negative_target = max_samples - positive_target
    selected = positives[:positive_target] + negatives[:negative_target]
    if len(selected) < max_samples:
        remainder = [
            row for row in rows
            if row not in selected
        ][: max_samples - len(selected)]
        selected.extend(remainder)
    return selected


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(1 for row in rows if int(row.get("label", 0)) == 1)
    return {
        "total": len(rows),
        "positive": positives,
        "negative": len(rows) - positives,
        "positive_rate": round(positives / len(rows), 4) if rows else 0.0,
    }


def _validate_split_integrity(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    required_fields = {
        "split",
        "question_id",
        "paired_question_id",
        "dependency_group",
        "paired_dependency_group",
    }
    missing_rows = sum(
        not required_fields.issubset(row)
        for rows in rows_by_split.values()
        for row in rows
    )
    if missing_rows:
        if not allow_unverified:
            raise ValueError(
                f"{missing_rows} rows lack dependency-aware split provenance; "
                "regenerate the dataset or pass --allow-unverified-splits only "
                "for legacy diagnostics"
            )
        return {
            "status": "unverified",
            "missing_provenance_rows": missing_rows,
        }

    group_splits: dict[str, set[str]] = {}
    question_splits: dict[str, set[str]] = {}
    text_splits: dict[str, set[str]] = {}
    declared_split_mismatches = 0
    for split, rows in rows_by_split.items():
        for row in rows:
            declared_split_mismatches += int(row["split"] != split)
            for group_id in (
                str(row["dependency_group"]),
                str(row["paired_dependency_group"]),
            ):
                group_splits.setdefault(group_id, set()).add(split)
            for question_id in (
                str(row["question_id"]),
                str(row["paired_question_id"]),
            ):
                question_splits.setdefault(question_id, set()).add(split)
            for text in (
                str(row.get("text_a", "")).strip(),
                str(row.get("text_b", "")).strip(),
            ):
                if text:
                    text_splits.setdefault(text, set()).add(split)

    dependency_group_overlap = sum(
        len(splits) > 1
        for splits in group_splits.values()
    )
    question_overlap = sum(
        len(splits) > 1
        for splits in question_splits.values()
    )
    text_overlap = sum(
        len(splits) > 1
        for splits in text_splits.values()
    )
    if (
        declared_split_mismatches
        or dependency_group_overlap
        or question_overlap
        or text_overlap
    ):
        raise ValueError(
            "cross-split leakage detected: "
            f"declared_split_mismatches={declared_split_mismatches}, "
            f"dependency_groups={dependency_group_overlap}, "
            f"questions={question_overlap}, exact_texts={text_overlap}"
        )
    return {
        "status": "verified",
        "strategy": "shared-evidence-component-stratified-v1",
        "declared_split_mismatches": 0,
        "dependency_group_overlap": 0,
        "question_overlap": 0,
        "cross_split_text_overlap": 0,
    }


def _balance_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [row for row in rows if int(row.get("label", 0)) == 1]
    negatives = [row for row in rows if int(row.get("label", 0)) == 0]
    if not positives or not negatives or len(positives) >= len(negatives):
        return rows

    repeats, remainder = divmod(len(negatives), len(positives))
    balanced = negatives + positives * repeats + positives[:remainder]
    return sorted(
        balanced,
        key=lambda row: (
            str(row.get("question_id", "")),
            str(row.get("id", "")),
            int(row.get("label", 0)),
        ),
    )


def _score_to_probability(score: Any) -> float:
    if hasattr(score, "tolist"):
        score = score.tolist()
    while isinstance(score, list):
        if not score:
            return 0.0
        score = score[0]
    value = float(score)
    if math.isnan(value):
        return 0.0
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + math.exp(-value))


def _classification_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for label, probability in zip(labels, probabilities, strict=True):
        predicted = 1 if probability >= threshold else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 6),
        "total": total,
        "positive": sum(labels),
        "negative": total - sum(labels),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _best_threshold(labels: list[int], probabilities: list[float]) -> float:
    if not labels:
        return 0.5
    candidates = sorted({0.0, 0.5, 0.7, 1.0, *probabilities})

    def key(threshold: float) -> tuple[float, float, float]:
        metrics = _classification_metrics(labels, probabilities, threshold)
        return (
            float(metrics["f1"]),
            float(metrics["recall"]),
            -abs(threshold - 0.5),
        )

    return max(candidates, key=key)


def _score_rows(model: Any, rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    if not rows:
        return [], []
    scores = model.predict([[row["text_a"], row["text_b"]] for row in rows])
    probabilities = [_score_to_probability(score) for score in scores]
    labels = [int(row["label"]) for row in rows]
    return labels, probabilities


def _pair_dependency_components(
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(group_id: str) -> str:
        parent.setdefault(group_id, group_id)
        if parent[group_id] != group_id:
            parent[group_id] = find(parent[group_id])
        return parent[group_id]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        keep, merge = sorted([left_root, right_root])
        parent[merge] = keep

    for row in rows:
        left = str(row.get("dependency_group", "") or "")
        right = str(row.get("paired_dependency_group", "") or "")
        if not left or not right:
            continue
        union(left, right)

    groups_by_root: dict[str, list[str]] = {}
    for group_id in parent:
        groups_by_root.setdefault(find(group_id), []).append(group_id)
    component_by_group: dict[str, str] = {}
    for groups in groups_by_root.values():
        digest = hashlib.sha256(
            "\n".join(sorted(groups)).encode("utf-8")
        ).hexdigest()[:16]
        component_id = f"pair-component-{digest}"
        for group_id in groups:
            component_by_group[group_id] = component_id
    return component_by_group


def _prediction_rows(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    threshold: float,
) -> list[dict[str, Any]]:
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must have the same length")
    component_by_group = _pair_dependency_components(rows)
    predictions = []
    for index, (row, probability) in enumerate(
        zip(rows, probabilities, strict=True)
    ):
        dependency_group = str(row.get("dependency_group", "") or "")
        predictions.append({
            "id": str(row.get("id") or f"row-{index}"),
            "question_id": str(row.get("question_id", "")),
            "paired_question_id": str(row.get("paired_question_id", "")),
            "dependency_group": dependency_group,
            "paired_dependency_group": str(
                row.get("paired_dependency_group", "")
            ),
            "evaluation_dependency_component": component_by_group.get(
                dependency_group,
                "",
            ),
            "label": int(row["label"]),
            "probability": round(probability, 8),
            "threshold": round(threshold, 8),
            "predicted": int(probability >= threshold),
        })
    return predictions


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _cross_encoder_init_kwargs(device: str | None = None) -> dict[str, Any]:
    """Build kwargs for binary conflict-score fine-tuning.

    NLI CrossEncoder checkpoints often ship with a 3-way classifier head
    (contradiction/entailment/neutral). VeraRAG trains a single conflict score,
    so the head is intentionally reinitialized while the encoder body is reused.
    """
    kwargs: dict[str, Any] = {
        "num_labels": 1,
        "config_kwargs": {
            "num_labels": 1,
            "id2label": {0: "CONFLICT"},
            "label2id": {"CONFLICT": 0},
        },
        "model_kwargs": {"ignore_mismatched_sizes": True},
    }
    if device:
        kwargs["device"] = device
    return kwargs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def _save_trained_model(model: Any, output_dir: str | Path) -> None:
    # Generated model cards may query Hugging Face metadata and stall on an
    # offline training host. VeraRAG writes richer local metadata below.
    model.save(str(output_dir), create_model_card=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VeraBench conflict CrossEncoder")
    parser.add_argument("--train", default="outputs/conflict_pairs/train.jsonl")
    parser.add_argument("--val", default="outputs/conflict_pairs/val.jsonl")
    parser.add_argument(
        "--test",
        help="Test JSONL; defaults to test.jsonl beside the selected train file.",
    )
    parser.add_argument("--output-dir", default="outputs/conflict_cross_encoder")
    parser.add_argument("--model", default="cross-encoder/nli-distilroberta-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--no-balance-train",
        action="store_true",
        help="Disable positive oversampling in the training loader.",
    )
    parser.add_argument(
        "--allow-unverified-splits",
        action="store_true",
        help="Allow legacy rows missing split provenance; detected leakage still fails.",
    )
    parser.add_argument("--max-samples", type=int, help="Limit train/val rows for smoke runs")
    parser.add_argument("--device", help="Optional torch device, e.g. cuda, cpu")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print summary without training")
    args = parser.parse_args()

    test_path = args.test or str(Path(args.train).with_name("test.jsonl"))
    dataset_metadata_path = Path(args.train).with_name("metadata.json")
    train_rows = _balanced_limit(_load_jsonl(args.train), args.max_samples)
    val_rows = _balanced_limit(_load_jsonl(args.val), args.max_samples)
    test_rows = _balanced_limit(
        _load_jsonl(test_path, required=False),
        args.max_samples,
    )
    try:
        split_integrity = _validate_split_integrity(
            {"train": train_rows, "val": val_rows, "test": test_rows},
            allow_unverified=args.allow_unverified_splits,
        )
    except ValueError as exc:
        raise SystemExit(f"Split integrity validation failed: {exc}") from exc
    train_loader_rows = train_rows if args.no_balance_train else _balance_training_rows(train_rows)
    summary = {
        "train": _summarize(train_rows),
        "train_loader": _summarize(train_loader_rows),
        "val": _summarize(val_rows),
        "test": _summarize(test_rows),
        "split_integrity": split_integrity,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return
    if not train_rows:
        raise SystemExit(f"No training rows found: {args.train}")
    _set_seed(args.seed)

    try:
        import torch
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Training requires optional dependencies. Install them with "
            "`pip install 'verarag[train]'` or use the Windows GPU conda env."
        ) from exc

    try:
        from sentence_transformers.cross_encoder.evaluation import (
            CrossEncoderClassificationEvaluator,
        )

        def make_evaluator() -> Any:
            return CrossEncoderClassificationEvaluator(
                sentence_pairs=[[row["text_a"], row["text_b"]] for row in val_rows],
                labels=[int(row["label"]) for row in val_rows],
                name="verabench-conflict-val",
            )
    except ImportError:
        from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

        def make_evaluator() -> Any:
            return CEBinaryClassificationEvaluator(
                sentence_pairs=[[row["text_a"], row["text_b"]] for row in val_rows],
                labels=[int(row["label"]) for row in val_rows],
                name="verabench-conflict-val",
            )

    train_examples = [
        InputExample(texts=[row["text_a"], row["text_b"]], label=float(row["label"]))
        for row in train_loader_rows
    ]
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        generator=generator,
    )
    evaluator = make_evaluator() if val_rows else None

    try:
        model = CrossEncoder(args.model, **_cross_encoder_init_kwargs(args.device))
    except Exception as exc:
        raise SystemExit(
            f"Failed to load base model '{args.model}'. If the remote GPU host cannot "
            "reach Hugging Face, pass --model /path/to/a/pre-cached/local/model or "
            "pre-download the model before starting training."
        ) from exc
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=None,
        save_best_model=False,
        show_progress_bar=True,
    )
    _save_trained_model(model, args.output_dir)

    metadata = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "balance_train": not args.no_balance_train,
        "train": args.train,
        "val": args.val,
        "test": test_path,
        "dataset_fingerprints": {
            "train_sha256": _sha256(args.train),
            "val_sha256": _sha256(args.val),
            "test_sha256": _sha256(test_path) if Path(test_path).exists() else None,
        },
        "dataset_manifest": (
            {
                "path": str(dataset_metadata_path),
                "sha256": _sha256(dataset_metadata_path),
                "contents": json.loads(
                    dataset_metadata_path.read_text(encoding="utf-8")
                ),
            }
            if dataset_metadata_path.exists()
            else None
        ),
        "summary": summary,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metric_rows = {"val": val_rows, "test": test_rows}
    scored: dict[str, dict[str, Any]] = {}
    val_labels, val_probabilities = _score_rows(model, val_rows)
    selected_threshold = _best_threshold(val_labels, val_probabilities)
    if val_rows:
        scored["val"] = {
            "default_threshold": _classification_metrics(
                val_labels,
                val_probabilities,
                args.threshold,
            ),
            "selected_threshold": _classification_metrics(
                val_labels,
                val_probabilities,
                selected_threshold,
            ),
        }
    prediction_artifacts: dict[str, dict[str, Any]] = {}
    for split_name, rows in metric_rows.items():
        if not rows:
            continue
        if split_name == "val":
            labels = val_labels
            probabilities = val_probabilities
        else:
            labels, probabilities = _score_rows(model, rows)
            scored[split_name] = {
                "default_threshold": _classification_metrics(
                    labels,
                    probabilities,
                    args.threshold,
                ),
                "selected_threshold": _classification_metrics(
                    labels,
                    probabilities,
                    selected_threshold,
                ),
            }
        predictions = _prediction_rows(
            rows,
            probabilities,
            selected_threshold,
        )
        prediction_path = Path(args.output_dir, f"{split_name}_predictions.jsonl")
        _write_jsonl(prediction_path, predictions)
        prediction_artifacts[split_name] = {
            "path": prediction_path.name,
            "sha256": _sha256(prediction_path),
            "rows": len(predictions),
        }
        if all(
            prediction["evaluation_dependency_component"]
            for prediction in predictions
        ):
            uncertainty = binary_pair_dependency_bootstrap_confidence_intervals(
                predictions,
                seed=args.seed,
            )
        else:
            uncertainty = {
                "status": "unavailable",
                "reason": "missing dependency-aware split provenance",
            }
        scored[split_name]["dependency_robust_confidence_intervals"] = uncertainty
    metrics_payload = {
        "schema_version": "conflict-training-metrics-v2",
        "default_threshold": args.threshold,
        "selected_threshold": round(selected_threshold, 6),
        "prediction_artifacts": prediction_artifacts,
        "splits": scored,
    }
    Path(args.output_dir, "training_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.output_dir, "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
