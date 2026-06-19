"""Tests for post-hoc VeraBench confidence calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.calibrate_verabench_confidence import calibrate_report, main


def _report() -> dict:
    rows = [
        ("V001", 0.18, False),
        ("V002", 0.22, False),
        ("V003", 0.25, False),
        ("V004", 0.32, True),
        ("V005", 0.36, True),
        ("V006", 0.40, True),
        ("V007", 0.44, True),
        ("V008", 0.48, True),
        ("V009", 0.52, True),
        ("V010", 0.56, True),
    ]
    return {
        "total_questions": len(rows),
        "completed": len(rows),
        "avg_confidence": 0.373,
        "ece": 0.0,
        "brier_score": 0.0,
        "metadata": {"benchmark_version": "1.1.2"},
        "question_results": [
            {
                "question_id": question_id,
                "question_type": "factual",
                "actual_behavior": "answer_with_citation" if question_id <= "V006" else "abstain",
                "confidence": confidence,
                "correct": correct,
            }
            for question_id, confidence, correct in rows
        ],
    }


def test_calibrate_report_writes_platt_metadata_and_preserves_original_confidence():
    calibrated, summary = calibrate_report(
        _report(),
        method="platt",
        calibration_fraction=0.5,
        seed=7,
        bins=5,
    )

    metadata = calibrated["metadata"]["posthoc_confidence_calibration"]
    assert metadata["method"] == "platt"
    assert metadata["metrics"]["all"]["after"]["ece"] <= metadata["metrics"]["all"]["before"]["ece"]
    assert summary == metadata
    assert calibrated["avg_confidence"] != pytest.approx(_report()["avg_confidence"])

    row = calibrated["question_results"][0]
    calibration_diag = row["diagnostics"]["confidence_calibration"]
    assert calibration_diag["original_confidence"] == 0.18
    assert calibration_diag["split"] in {"calibration", "holdout"}
    assert 0.0 <= row["confidence"] <= 1.0


def test_calibrate_report_temperature_method_records_temperature():
    calibrated, summary = calibrate_report(
        _report(),
        method="temperature",
        calibration_fraction=0.5,
        seed=7,
        bins=5,
    )

    assert summary["method"] == "temperature"
    assert summary["temperature"] > 0
    assert calibrated["metadata"]["posthoc_confidence_calibration"]["temperature"] == summary["temperature"]


def test_calibrate_report_rejects_missing_or_single_class_correctness():
    report = _report()
    del report["question_results"][0]["correct"]
    with pytest.raises(ValueError, match="missing correctness field"):
        calibrate_report(report)

    single_class = _report()
    for row in single_class["question_results"]:
        row["correct"] = True
    with pytest.raises(ValueError, match="both correct and incorrect"):
        calibrate_report(single_class)


def test_calibrate_report_supports_behavior_group_calibration_with_fallbacks():
    calibrated, summary = calibrate_report(
        _report(),
        method="platt",
        group_field="actual_behavior",
        min_group_rows=2,
        calibration_fraction=0.5,
        seed=7,
        bins=5,
    )

    assert summary["group_field"] == "actual_behavior"
    assert summary["group_fallback"] == "smoothed_constant"
    assert set(summary["groups"]) == {"answer_with_citation", "abstain"}
    assert summary["groups"]["answer_with_citation"]["mode"] == "group_platt"
    assert summary["groups"]["abstain"]["mode"] == "smoothed_constant"
    assert "fallback_reason" in summary["groups"]["abstain"]

    answer_row = next(
        row for row in calibrated["question_results"] if row["actual_behavior"] == "answer_with_citation"
    )
    answer_diag = answer_row["diagnostics"]["confidence_calibration"]
    assert answer_diag["group_field"] == "actual_behavior"
    assert answer_diag["group_value"] == "answer_with_citation"
    assert answer_diag["model_scope"] == "group"

    abstain_row = next(row for row in calibrated["question_results"] if row["actual_behavior"] == "abstain")
    abstain_diag = abstain_row["diagnostics"]["confidence_calibration"]
    assert abstain_diag["model_scope"] == "fallback"
    assert abstain_diag["mode"] == "smoothed_constant"
    assert "fallback_reason" in abstain_diag


def test_calibrate_report_rejects_missing_group_field():
    report = _report()
    del report["question_results"][0]["actual_behavior"]

    with pytest.raises(ValueError, match="missing group field"):
        calibrate_report(report, group_field="actual_behavior")


def test_calibrate_report_cli_writes_report_and_summary(tmp_path: Path):
    input_path = tmp_path / "report.json"
    output_path = tmp_path / "calibrated.json"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_report()), encoding="utf-8")

    exit_code = main([
        "--input", str(input_path),
        "--output", str(output_path),
        "--summary-output", str(summary_path),
        "--method", "platt",
        "--group-field", "actual_behavior",
        "--min-group-rows", "2",
        "--bins", "5",
        "--json",
    ])

    assert exit_code == 0
    calibrated = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert calibrated["metadata"]["posthoc_confidence_calibration"] == summary
    assert summary["group_field"] == "actual_behavior"
    assert summary["holdout_rows"] > 0
