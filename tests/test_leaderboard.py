"""Tests for VeraBench leaderboard generation."""

import json
import subprocess
import sys
from pathlib import Path

from experiments.build_verabench_leaderboard import (
    build_rows,
    render_json,
    render_markdown,
)


def _write_report(path: Path, behavior_accuracy: float, model: str) -> None:
    report = {
        "total_questions": 2,
        "completed": 2,
        "errors": 0,
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
            "provider": "test",
            "model": model,
            "git_commit": "abc123",
            "config_path": "configs/test.yaml",
            "timestamp": "2026-06-06T00:00:00+0800",
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
    _write_report(report, 0.8, "model-a")

    markdown = render_markdown(build_rows([report]), source_command="verarag-leaderboard run.json")

    assert "# VeraBench Results" in markdown
    assert "| 1 | run | test/model-a | 2/2 | 0 | 0.800 | 0.500 | 0.750 | 0.200 | 0.100 | 1.5s | abc123 |" in markdown
    assert "| single_evidence | 2 | 0.500 | 0.750 | 0.800 |" in markdown


def test_render_json_is_machine_readable(tmp_path):
    report = tmp_path / "run.json"
    _write_report(report, 0.8, "model-a")

    payload = json.loads(render_json(build_rows([report])))

    assert payload["schema_version"] == 1
    assert payload["runs"][0]["model"] == "model-a"
    assert payload["runs"][0]["behavior_accuracy"] == 0.8


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
