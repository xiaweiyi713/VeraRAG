"""Tests for paired VeraBench report comparison."""

import json
import subprocess
import sys

import pytest

from experiments.compare_verabench_reports import (
    compare_reports,
    render_markdown,
)


def _report(answer_scores, *, questions_hash="questions-a", mode="pipeline"):
    rows = []
    for index, score in enumerate(answer_scores, start=1):
        correct = score >= 0.5
        rows.append({
            "question_id": f"T{index:03d}",
            "question_type": "single_evidence",
            "answer_f1": score,
            "evidence_recall": score,
            "evidence_precision": score,
            "expected_behavior": "answer_with_citation",
            "actual_behavior": (
                "answer_with_citation" if correct else "abstain"
            ),
            "correct": correct,
            "confidence": score,
            "conflict_true_positives": 0,
            "conflict_false_positives": 0,
            "conflict_false_negatives": 0,
            "dependency_group": f"component-{(index + 1) // 2}",
        })
    return {
        "total_questions": len(rows),
        "completed": len(rows),
        "errors": 0,
        "question_results": rows,
        "metadata": {
            "mode": mode,
            "provider": "test",
            "model": "model",
            "git_commit": "abc",
            "config_path": "configs/test.yaml",
            "benchmark": {
                "version": "1.1.2",
                "fingerprints": {
                    "corpus_sha256": "corpus-a",
                    "questions_sha256": questions_hash,
                },
            },
            "metric_versions": {
                "answer": "soft-f1-v2",
                "behavior": "behavior-v2",
                "conflict": "gold-evidence-pair-micro-f1-v2",
            },
        },
    }


def test_compare_reports_returns_paired_delta_and_markdown():
    payload = compare_reports(
        _report([0.2, 0.3, 0.4, 0.1]),
        _report([0.8, 0.9, 1.0, 0.7]),
        baseline_label="old",
        candidate_label="new",
        resamples=200,
        seed=3,
    )

    answer = payload["comparison"]["metrics"]["answer_f1"]
    assert answer["delta_candidate_minus_baseline"] == 0.6
    assert answer["probability_candidate_better"] == 1.0
    assert payload["comparison"]["behavior_mcnemar_exact"][
        "candidate_only_correct"
    ] == 4
    markdown = render_markdown(payload)
    assert "# VeraBench Paired Comparison" in markdown
    assert "| answer_f1 |" in markdown
    assert "Shared-Evidence Dependency Sensitivity" in markdown
    assert "two-sided exact p=0.125000" in markdown


def test_compare_reports_rejects_mismatched_benchmark():
    with pytest.raises(ValueError, match="not statistically comparable"):
        compare_reports(
            _report([0.5], questions_hash="questions-a"),
            _report([0.5], questions_hash="questions-b"),
            resamples=10,
        )


def test_compare_reports_rejects_different_question_coverage():
    with pytest.raises(ValueError, match="different question IDs"):
        compare_reports(
            _report([0.5, 0.5]),
            _report([0.5]),
            resamples=10,
        )


def test_compare_reports_rejects_incomplete_top_level_counts():
    baseline = _report([0.5, 0.5])
    baseline["completed"] = 1

    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        compare_reports(
            baseline,
            _report([0.5, 0.5]),
            resamples=10,
        )


def test_compare_reports_rejects_demo_by_default():
    with pytest.raises(ValueError, match="demo reports"):
        compare_reports(
            _report([1.0], mode="demo"),
            _report([1.0], mode="demo"),
            resamples=10,
        )


def test_compare_reports_cli_writes_json(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(_report([0.2, 0.3])), encoding="utf-8")
    candidate.write_text(json.dumps(_report([0.8, 0.9])), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "experiments/compare_verabench_reports.py",
            str(baseline),
            str(candidate),
            "--resamples",
            "100",
            "--format",
            "json",
            "--output",
            str(output),
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["comparison"]["questions"] == 2
    assert payload["comparison"]["metrics"]["answer_f1"][
        "probability_candidate_better"
    ] == 1.0
