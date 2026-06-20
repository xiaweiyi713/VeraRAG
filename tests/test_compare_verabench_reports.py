"""Tests for paired VeraBench report comparison."""

import json
import subprocess
import sys

import pytest

from experiments.compare_verabench_reports import (
    compare_reports,
    render_markdown,
)


def _report(
    answer_scores,
    *,
    questions_hash="questions-a",
    mode="pipeline",
    citation_scores=None,
    supporting_scores=None,
    latency_scores=None,
):
    rows = []
    for index, score in enumerate(answer_scores, start=1):
        correct = score >= 0.5
        row = {
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
        }
        if latency_scores is not None:
            row["latency_seconds"] = latency_scores[index - 1]
        if citation_scores is not None:
            citation_score = citation_scores[index - 1]
            row.update({
                "citation_precision": citation_score,
                "citation_recall": citation_score,
                "citation_f1": citation_score,
            })
        if supporting_scores is not None:
            supporting_score = supporting_scores[index - 1]
            row.update({
                "supporting_fact_precision": supporting_score,
                "supporting_fact_recall": supporting_score,
                "supporting_fact_f1": supporting_score,
            })
        rows.append(row)
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


def test_compare_reports_includes_citation_and_supporting_fact_metrics():
    payload = compare_reports(
        _report(
            [0.5, 0.5],
            citation_scores=[0.25, 0.25],
            supporting_scores=[0.5, 0.5],
        ),
        _report(
            [0.5, 0.5],
            citation_scores=[0.75, 0.75],
            supporting_scores=[1.0, 1.0],
        ),
        resamples=100,
        seed=3,
    )

    assert payload["comparison"]["metrics"]["citation_f1"][
        "delta_candidate_minus_baseline"
    ] == 0.5
    assert payload["comparison"]["metrics"]["supporting_fact_f1"][
        "delta_candidate_minus_baseline"
    ] == 0.5
    markdown = render_markdown(payload)
    assert "| citation_f1 |" in markdown
    assert "| supporting_fact_f1 |" in markdown


def test_compare_reports_includes_latency_as_lower_is_better():
    payload = compare_reports(
        _report([0.5, 0.5], latency_scores=[10.0, 20.0]),
        _report([0.5, 0.5], latency_scores=[5.0, 8.0]),
        resamples=100,
        seed=3,
    )

    latency = payload["comparison"]["metrics"]["latency_seconds"]
    assert latency["baseline"] == 15.0
    assert latency["candidate"] == 6.5
    assert latency["delta_candidate_minus_baseline"] == -8.5
    assert latency["direction"] == "lower_is_better"
    assert latency["probability_candidate_better"] == 1.0
    assert latency["paired_outcomes"] == {
        "candidate_wins": 2,
        "ties": 0,
        "candidate_losses": 0,
    }
    markdown = render_markdown(payload)
    assert "| latency_seconds |" in markdown
    assert "| latency_seconds | lower_is_better |" in markdown


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
