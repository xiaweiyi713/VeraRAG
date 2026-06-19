"""Tests for the fail-closed conflict-model promotion policy."""

import hashlib
import json

from experiments.audit_conflict_model import audit_conflict_model


def _write_run(
    root,
    *,
    seed,
    test_f1,
    cluster_lower,
    manifest="manifest-sha",
):
    run_dir = root / f"seed-{seed}"
    run_dir.mkdir()
    predicted = int(test_f1 >= 0.5)
    metric_value = float(predicted)
    predictions = run_dir / "test_predictions.jsonl"
    predictions.write_text(
        json.dumps({
            "id": "row-1",
            "label": 1,
            "predicted": predicted,
            "probability": 0.9 if predicted else 0.1,
            "threshold": 0.5,
            "evaluation_dependency_component": "component-1",
        }) + "\n",
        encoding="utf-8",
    )
    prediction_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
    (run_dir / "training_metadata.json").write_text(
        json.dumps({
            "seed": seed,
            "balance_train": True,
            "summary": {"split_integrity": {"status": "verified"}},
            "dataset_manifest": {
                "sha256": manifest,
                "contents": {
                    "schema_version": "conflict-pairs-v2",
                    "benchmark": {
                        "fingerprints": {
                            "questions_sha256": "training-questions",
                            "corpus_sha256": "training-corpus",
                        }
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    (run_dir / "training_metrics.json").write_text(
        json.dumps({
            "schema_version": "conflict-training-metrics-v2",
            "prediction_artifacts": {
                "test": {
                    "path": predictions.name,
                    "sha256": prediction_hash,
                    "rows": 1,
                }
            },
            "splits": {
                "test": {
                    "selected_threshold": {
                        "f1": metric_value,
                        "precision": metric_value,
                        "recall": metric_value,
                    },
                    "dependency_robust_confidence_intervals": {
                        "clusters": 10,
                        "metrics": {"f1": {"lower": cluster_lower}},
                    },
                }
            },
        }),
        encoding="utf-8",
    )
    return run_dir


def _ablation(*, independent=True, learned_f1=0.85, rules_f1=0.75):
    return {
        "evaluation_scope": {
            "split": "all" if independent else "test",
            "independent_test": independent,
            "evaluation_id": "external-conflicts-v1" if independent else None,
            "dataset": {
                "fingerprints": (
                    {
                        "questions_sha256": "external-questions",
                        "corpus_sha256": "external-corpus",
                    }
                    if independent
                    else None
                )
            },
        },
        "variants": [
            {
                "name": "rules",
                "learned_available": False,
                "summary": {
                    "questions": 20,
                    "gold_conflicts": 20,
                    "false_positives": 2,
                    "recall": 0.75,
                    "f1": rules_f1,
                },
            },
            {
                "name": "rules_plus_learned",
                "learned_available": True,
                "summary": {
                    "questions": 20,
                    "gold_conflicts": 20,
                    "false_positives": 2,
                    "recall": 0.85,
                    "f1": learned_f1,
                },
            },
        ],
    }


def test_promotion_audit_accepts_only_reproducible_external_gain(tmp_path):
    runs = [
        _write_run(tmp_path, seed=seed, test_f1=1.0, cluster_lower=0.6)
        for seed in (13, 17, 23)
    ]

    result = audit_conflict_model(runs, _ablation())

    assert result["decision"] == "promote"
    assert result["passed"] is True
    assert result["failed_checks"] == []


def test_promotion_audit_rejects_small_internal_no_gain_result(tmp_path):
    runs = [
        _write_run(tmp_path, seed=seed, test_f1=0.0, cluster_lower=0.0)
        for seed in (13, 17, 23)
    ]
    ablation = _ablation(
        independent=False,
        learned_f1=0.8571,
        rules_f1=0.8571,
    )
    for variant in ablation["variants"]:
        variant["summary"]["questions"] = 3
        variant["summary"]["gold_conflicts"] = 3

    result = audit_conflict_model(runs, ablation)

    assert result["decision"] == "reject"
    assert {
        "multi_seed_test_f1",
        "dependency_robust_f1_lower_bound",
        "independent_evaluation_scope",
        "minimum_ablation_sample",
        "heldout_ablation_improvement",
    }.issubset(result["failed_checks"])


def test_promotion_audit_rejects_tampered_predictions(tmp_path):
    runs = [
        _write_run(tmp_path, seed=seed, test_f1=0.8, cluster_lower=0.6)
        for seed in (13, 17, 23)
    ]
    (runs[0] / "test_predictions.jsonl").write_text("tampered\n", encoding="utf-8")

    result = audit_conflict_model(runs, _ablation())

    assert result["decision"] == "reject"
    assert "auditable_test_predictions" in result["failed_checks"]
