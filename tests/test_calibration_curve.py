"""Tests for calibration curve generation."""

import json

import numpy as np
import pytest

from experiments.calibration_curve import (
    compute_brier_score,
    compute_calibration,
    load_confidence_rows,
    main,
)


def test_load_confidence_rows_accepts_current_report_shape(tmp_path):
    report = {
        "question_results": [
            {"question_id": "V001", "confidence": 0.8, "correct": True},
            {"question_id": "V002", "confidence": 0.2, "correct": False},
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    predicted, actual = load_confidence_rows(path)

    assert predicted.tolist() == [0.8, 0.2]
    assert actual.tolist() == [1.0, 0.0]


def test_load_confidence_rows_uses_requested_boolean_correctness_field(tmp_path):
    report = [
        {"confidence": 0.7, "correct": False, "premise_refutation_correct": True},
        {"confidence": 0.4, "correct": True, "premise_refutation_correct": False},
    ]
    path = tmp_path / "premise.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    _, actual = load_confidence_rows(path, correctness_field="premise_refutation_correct")

    assert actual.tolist() == [1.0, 0.0]


def test_load_confidence_rows_rejects_ambiguous_or_invalid_rows(tmp_path):
    path = tmp_path / "bad.json"

    path.write_text(json.dumps({"question_results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no result rows"):
        load_confidence_rows(path)

    path.write_text(json.dumps([{"confidence": 0.5}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing correctness field"):
        load_confidence_rows(path)

    path.write_text(json.dumps([{"confidence": 1.2, "correct": True}]), encoding="utf-8")
    with pytest.raises(ValueError, match="finite value"):
        load_confidence_rows(path)

    path.write_text(json.dumps([{"confidence": 0.6, "correct": 1}]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be boolean"):
        load_confidence_rows(path)


def test_compute_calibration_and_brier_validate_inputs():
    predicted = np.array([0.9, 0.1])
    actual = np.array([1.0, 0.0])

    bins, ece = compute_calibration(predicted, actual, n_bins=2)

    assert ece < 0.11
    assert sum(row["count"] for row in bins) == 2
    assert compute_brier_score(predicted, actual) == pytest.approx(0.01)

    with pytest.raises(ValueError, match="same length"):
        compute_calibration(np.array([0.5]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="positive"):
        compute_calibration(predicted, actual, n_bins=0)


def test_calibration_curve_cli_writes_svg_and_json_summary(tmp_path, capsys):
    report = {
        "question_results": [
            {"confidence": 0.9, "correct": True},
            {"confidence": 0.4, "correct": False},
        ]
    }
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "calibration.svg"
    json_path = tmp_path / "calibration.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    exit_code = main(
        [
            "--input",
            str(report_path),
            "--output",
            str(svg_path),
            "--json-output",
            str(json_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")
    assert payload["rows"] == 2
    assert payload["correctness_field"] == "correct"
    assert summary["expected_calibration_error"] == payload["expected_calibration_error"]
