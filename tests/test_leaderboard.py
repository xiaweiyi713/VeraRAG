"""Tests for VeraBench leaderboard generation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.build_verabench_leaderboard import (
    build_rows,
    render_json,
    render_markdown,
)


def _write_report(
    path: Path,
    behavior_accuracy: float,
    model: str,
    *,
    mode: str = "pipeline",
    benchmark_version: str = "1.1.2",
    questions_sha256: str = "questions-a",
    completed: int = 2,
    errors: int = 0,
    with_intervals: bool = False,
) -> None:
    report = {
        "total_questions": 2,
        "completed": completed,
        "errors": errors,
        "overall_answer_f1": 0.5,
        "overall_evidence_recall": 0.75,
        "overall_evidence_precision": 0.25,
        "overall_conflict_f1": 0.2,
        "behavior_accuracy": behavior_accuracy,
        "ece": 0.1,
        "brier_score": 0.2,
        "avg_confidence": 0.7,
        "avg_latency": 1.5,
        "metadata": {
            "mode": mode,
            "provider": "test",
            "model": model,
            "git_commit": "abc123",
            "config_path": "configs/test.yaml",
            "timestamp": "2026-06-06T00:00:00+0800",
            "benchmark": {
                "version": benchmark_version,
                "fingerprints": {
                    "corpus_sha256": "corpus-a",
                    "questions_sha256": questions_sha256,
                },
            },
            "run_signature": {
                "implementation_sha256": "implementation-a",
                "config_sha256": "config-a",
            },
            "metric_versions": {
                "answer": "soft-f1-v2",
                "behavior": "behavior-v2",
                "conflict": "gold-evidence-pair-micro-f1-v2",
            },
        },
        "by_type": {
            "single_evidence": {
                "count": 2,
                "answer_f1": 0.5,
                "evidence_recall": 0.75,
                "behavior_accuracy": behavior_accuracy,
            }
        },
    }
    if with_intervals:
        report["confidence_intervals"] = {
            "method": "stratified-question-bootstrap-v1",
            "confidence_level": 0.95,
            "resamples": 2000,
            "seed": 1729,
            "metrics": {
                "answer_f1": {"estimate": 0.5, "lower": 0.4, "upper": 0.6},
                "behavior_accuracy": {
                    "estimate": behavior_accuracy,
                    "lower": 0.7,
                    "upper": 0.9,
                },
                "conflict_micro_f1": {
                    "estimate": 0.2,
                    "lower": 0.1,
                    "upper": 0.3,
                },
            },
        }
        report["dependency_robust_confidence_intervals"] = {
            "method": "evidence-cluster-bootstrap-v1",
            "confidence_level": 0.95,
            "resamples": 2000,
            "seed": 1729,
            "clusters": 2,
            "metrics": report["confidence_intervals"]["metrics"],
        }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_build_rows_sorts_by_behavior_accuracy(tmp_path):
    low = tmp_path / "low.json"
    high = tmp_path / "high.json"
    _write_report(low, 0.4, "low-model")
    _write_report(high, 0.8, "high-model")

    rows = build_rows([low, high])

    assert [row.model for row in rows] == ["high-model", "low-model"]
    assert rows[0].completed == 2
    assert rows[0].git_commit == "abc123"


def test_render_markdown_includes_leaderboard_and_by_type(tmp_path):
    report = tmp_path / "run.json"
    _write_report(report, 0.8, "model-a", with_intervals=True)

    markdown = render_markdown(build_rows([report]), source_command="verarag-leaderboard run.json")

    assert "# VeraBench Results" in markdown
    assert "| 1 | run | test/model-a | 2/2 | 0 | 0.800 | 0.500 | 0.750 | 0.200 | 0.100 | 1.5s | abc123 |" in markdown
    assert "## Statistical Uncertainty" in markdown
    assert "## Shared-Evidence Dependency Sensitivity" in markdown
    assert "| run | 95.0% | [0.400, 0.600] | [0.700, 0.900] | [0.100, 0.300] |" in markdown
    assert "| single_evidence | 2 | 0.500 | 0.750 | 0.800 |" in markdown


def test_render_json_is_machine_readable(tmp_path):
    report = tmp_path / "run.json"
    _write_report(report, 0.8, "model-a")

    payload = json.loads(render_json(build_rows([report])))

    assert payload["schema_version"] == 1
    assert payload["runs"][0]["model"] == "model-a"
    assert payload["runs"][0]["behavior_accuracy"] == 0.8
    assert payload["runs"][0]["benchmark_version"] == "1.1.2"


def test_leaderboard_cli_writes_markdown(tmp_path):
    report = tmp_path / "run.json"
    output = tmp_path / "leaderboard.md"
    _write_report(report, 0.8, "model-a")

    subprocess.run(
        [
            sys.executable,
            "experiments/build_verabench_leaderboard.py",
            str(report),
            "--output",
            str(output),
        ],
        check=True,
    )

    assert output.exists()
    assert "test/model-a" in output.read_text(encoding="utf-8")


def test_leaderboard_cli_reports_missing_input_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/build_verabench_leaderboard.py",
            str(tmp_path / "missing.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "No such file or directory" in result.stderr
    assert "Traceback" not in result.stderr


def test_leaderboard_rejects_demo_reports_by_default(tmp_path):
    report = tmp_path / "demo.json"
    _write_report(report, 1.0, "gold-identity", mode="demo")

    with pytest.raises(ValueError, match="demo reports"):
        build_rows([report])

    rows = build_rows([report], allow_demo=True)
    assert rows[0].mode == "demo"


def test_leaderboard_rejects_incomplete_reports_by_default(tmp_path):
    report = tmp_path / "partial.json"
    _write_report(report, 0.8, "model-a", completed=1, errors=1)

    with pytest.raises(ValueError, match="incomplete run"):
        build_rows([report])

    assert build_rows([report], allow_incomplete=True)[0].completed == 1


def test_leaderboard_rejects_mixed_benchmark_fingerprints(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_report(first, 0.8, "model-a", questions_sha256="questions-a")
    _write_report(second, 0.7, "model-b", questions_sha256="questions-b")

    with pytest.raises(ValueError, match="different VeraBench"):
        build_rows([first, second])

    rows = build_rows([first, second], allow_mixed_benchmarks=True)
    assert len(rows) == 2


def test_leaderboard_rejects_unverified_legacy_report(tmp_path):
    report = tmp_path / "legacy.json"
    _write_report(report, 0.8, "model-a")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["metadata"].pop("benchmark")
    payload["metadata"].pop("run_signature")
    payload["metadata"].pop("metric_versions")
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing reproducibility metadata"):
        build_rows([report])

    assert build_rows([report], allow_unverified=True)[0].model == "model-a"
