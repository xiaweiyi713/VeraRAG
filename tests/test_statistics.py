"""Tests for VeraBench statistical inference."""

from types import SimpleNamespace

import pytest

from src.evaluation.statistics import (
    binary_pair_dependency_bootstrap_confidence_intervals,
    dependency_cluster_bootstrap_confidence_intervals,
    metric_estimates,
    paired_bootstrap_comparison,
    stratified_bootstrap_confidence_intervals,
)


def _row(
    question_id: str,
    *,
    question_type: str,
    answer_f1: float,
    evidence_recall: float,
    behavior_correct: bool,
    correct: bool,
    confidence: float,
    conflict_tp: int = 0,
    conflict_fp: int = 0,
    conflict_fn: int = 0,
    dependency_group: str = "",
    citation_f1: float | None = None,
    citation_precision: float | None = None,
    citation_recall: float | None = None,
    supporting_fact_f1: float | None = None,
    supporting_fact_precision: float | None = None,
    supporting_fact_recall: float | None = None,
):
    expected = "answer_with_citation"
    actual = expected if behavior_correct else "abstain"
    row = SimpleNamespace(
        question_id=question_id,
        question_type=question_type,
        answer_f1=answer_f1,
        evidence_recall=evidence_recall,
        evidence_precision=evidence_recall,
        expected_behavior=expected,
        actual_behavior=actual,
        correct=correct,
        confidence=confidence,
        conflict_true_positives=conflict_tp,
        conflict_false_positives=conflict_fp,
        conflict_false_negatives=conflict_fn,
        premise_refutation_expected=False,
        premise_refutation_detected=False,
        dependency_group=dependency_group,
    )
    optional_metrics = {
        "citation_f1": citation_f1,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "supporting_fact_f1": supporting_fact_f1,
        "supporting_fact_precision": supporting_fact_precision,
        "supporting_fact_recall": supporting_fact_recall,
    }
    for name, value in optional_metrics.items():
        if value is not None:
            setattr(row, name, value)
    return row


def test_metric_estimates_include_micro_conflict_and_calibration():
    rows = [
        _row(
            "T1",
            question_type="conflict",
            answer_f1=1.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=1.0,
            conflict_tp=1,
        ),
        _row(
            "T2",
            question_type="single_evidence",
            answer_f1=0.0,
            evidence_recall=0.0,
            behavior_correct=False,
            correct=False,
            confidence=1.0,
            conflict_fp=1,
        ),
    ]

    estimates = metric_estimates(rows)

    assert estimates["answer_f1"] == 0.5
    assert estimates["behavior_accuracy"] == 0.5
    assert estimates["conflict_micro_f1"] == pytest.approx(2 / 3)
    assert estimates["ece"] == 0.5
    assert estimates["brier_score"] == 0.5


def test_metric_estimates_include_citation_and_supporting_fact_when_present():
    rows = [
        _row(
            "T1",
            question_type="single_evidence",
            answer_f1=1.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.9,
            citation_precision=1.0,
            citation_recall=0.5,
            citation_f1=2 / 3,
            supporting_fact_precision=1.0,
            supporting_fact_recall=1.0,
            supporting_fact_f1=1.0,
        ),
        _row(
            "T2",
            question_type="single_evidence",
            answer_f1=0.0,
            evidence_recall=0.0,
            behavior_correct=False,
            correct=False,
            confidence=0.1,
            citation_precision=0.0,
            citation_recall=0.0,
            citation_f1=0.0,
            supporting_fact_precision=0.5,
            supporting_fact_recall=1.0,
            supporting_fact_f1=2 / 3,
        ),
    ]

    estimates = metric_estimates(rows)

    assert estimates["citation_precision"] == 0.5
    assert estimates["citation_recall"] == 0.25
    assert estimates["citation_f1"] == pytest.approx(1 / 3)
    assert estimates["supporting_fact_precision"] == 0.75
    assert estimates["supporting_fact_recall"] == 1.0
    assert estimates["supporting_fact_f1"] == pytest.approx(5 / 6)


def test_metric_estimates_omit_citation_and_supporting_fact_when_absent():
    rows = [
        _row(
            "T1",
            question_type="single_evidence",
            answer_f1=1.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.9,
        )
    ]

    estimates = metric_estimates(rows)

    assert "citation_f1" not in estimates
    assert "supporting_fact_f1" not in estimates


def test_bootstrap_is_deterministic_and_stratified():
    rows = [
        _row(
            "C1",
            question_type="conflict",
            answer_f1=1.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.9,
            conflict_tp=1,
        ),
        _row(
            "C2",
            question_type="conflict",
            answer_f1=0.0,
            evidence_recall=0.0,
            behavior_correct=False,
            correct=False,
            confidence=0.1,
            conflict_fn=1,
        ),
        _row(
            "S1",
            question_type="single_evidence",
            answer_f1=0.5,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.8,
        ),
    ]

    first = stratified_bootstrap_confidence_intervals(
        rows,
        resamples=200,
        seed=7,
    )
    second = stratified_bootstrap_confidence_intervals(
        rows,
        resamples=200,
        seed=7,
    )

    assert first == second
    assert first["stratified_by"] == "question_type"
    assert first["metrics"]["conflict_micro_f1"]["estimate"] == pytest.approx(2 / 3)
    assert 0.0 <= first["metrics"]["answer_f1"]["lower"] <= 0.5
    assert 0.5 <= first["metrics"]["answer_f1"]["upper"] <= 1.0


def test_dependency_cluster_bootstrap_resamples_shared_evidence_units():
    rows = [
        _row(
            f"A{index}",
            question_type="single_evidence",
            answer_f1=1.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.9,
            dependency_group="component-a",
        )
        for index in range(3)
    ] + [
        _row(
            "B1",
            question_type="single_evidence",
            answer_f1=0.0,
            evidence_recall=0.0,
            behavior_correct=False,
            correct=False,
            confidence=0.1,
            dependency_group="component-b",
        )
    ]

    first = dependency_cluster_bootstrap_confidence_intervals(
        rows,
        resamples=200,
        seed=11,
    )
    second = dependency_cluster_bootstrap_confidence_intervals(
        rows,
        resamples=200,
        seed=11,
    )

    assert first == second
    assert first["clusters"] == 2
    assert first["cluster_sizes"] == [3, 1]
    assert first["metrics"]["answer_f1"]["estimate"] == 0.75
    assert first["metrics"]["answer_f1"]["lower"] == 0.0
    assert first["metrics"]["answer_f1"]["upper"] == 1.0


def test_binary_pair_bootstrap_uses_dependency_components():
    rows = [
        {
            "label": 1,
            "predicted": 1,
            "evaluation_dependency_component": "component-a",
        },
        {
            "label": 0,
            "predicted": 1,
            "evaluation_dependency_component": "component-a",
        },
        {
            "label": 0,
            "predicted": 0,
            "evaluation_dependency_component": "component-b",
        },
    ]

    result = binary_pair_dependency_bootstrap_confidence_intervals(
        rows,
        resamples=200,
        seed=19,
    )

    assert result["method"] == "binary-pair-dependency-bootstrap-v1"
    assert result["clusters"] == 2
    assert result["cluster_sizes"] == [2, 1]
    assert result["metrics"]["precision"]["estimate"] == 0.5
    assert result["metrics"]["recall"]["estimate"] == 1.0
    assert result["metrics"]["f1"]["estimate"] == pytest.approx(2 / 3)
    assert result["effective_resamples_by_metric"]["accuracy"] == 200
    assert 0 < result["effective_resamples_by_metric"]["f1"] < 200


def test_binary_pair_bootstrap_requires_dependency_component():
    with pytest.raises(ValueError, match="evaluation_dependency_component"):
        binary_pair_dependency_bootstrap_confidence_intervals(
            [{"label": 1, "predicted": 1}],
            resamples=10,
        )


def test_bootstrap_reports_effective_resamples_for_sparse_metrics():
    rows = [
        _row(
            "C1",
            question_type="conflict",
            answer_f1=1.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.9,
            conflict_tp=1,
            dependency_group="component-a",
        ),
        _row(
            "C2",
            question_type="conflict",
            answer_f1=0.0,
            evidence_recall=0.0,
            behavior_correct=False,
            correct=False,
            confidence=0.1,
            dependency_group="component-b",
        ),
    ]

    stratified = stratified_bootstrap_confidence_intervals(
        rows,
        resamples=200,
        seed=17,
    )
    clustered = dependency_cluster_bootstrap_confidence_intervals(
        rows,
        resamples=200,
        seed=17,
    )

    for result in (stratified, clustered):
        effective = result["effective_resamples_by_metric"]
        assert effective["answer_f1"] == 200
        assert 0 < effective["conflict_micro_f1"] < 200
        assert result["metrics"]["conflict_micro_f1"]["estimate"] == 1.0


def test_paired_bootstrap_detects_uniform_candidate_improvement():
    baseline = [
        _row(
            f"T{index}",
            question_type="single_evidence",
            answer_f1=0.2,
            evidence_recall=0.5,
            behavior_correct=False,
            correct=False,
            confidence=0.2,
        )
        for index in range(4)
    ]
    candidate = [
        _row(
            f"T{index}",
            question_type="single_evidence",
            answer_f1=0.8,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.8,
        )
        for index in range(4)
    ]

    comparison = paired_bootstrap_comparison(
        baseline,
        candidate,
        resamples=200,
        seed=9,
    )

    answer = comparison["metrics"]["answer_f1"]
    assert answer["delta_candidate_minus_baseline"] == 0.6
    assert answer["delta_lower"] == 0.6
    assert answer["delta_upper"] == 0.6
    assert answer["probability_candidate_better"] == 1.0
    assert answer["paired_outcomes"] == {
        "candidate_wins": 4,
        "ties": 0,
        "candidate_losses": 0,
    }
    assert comparison["behavior_mcnemar_exact"]["candidate_only_correct"] == 4
    assert comparison["behavior_mcnemar_exact"]["two_sided_exact_p"] == 0.125


def test_paired_bootstrap_compares_citation_and_supporting_fact_metrics():
    baseline = [
        _row(
            f"T{index}",
            question_type="single_evidence",
            answer_f1=0.5,
            evidence_recall=0.5,
            behavior_correct=True,
            correct=True,
            confidence=0.5,
            citation_precision=0.25,
            citation_recall=0.5,
            citation_f1=1 / 3,
            supporting_fact_precision=0.5,
            supporting_fact_recall=0.5,
            supporting_fact_f1=0.5,
        )
        for index in range(4)
    ]
    candidate = [
        _row(
            f"T{index}",
            question_type="single_evidence",
            answer_f1=0.5,
            evidence_recall=0.5,
            behavior_correct=True,
            correct=True,
            confidence=0.5,
            citation_precision=0.75,
            citation_recall=1.0,
            citation_f1=6 / 7,
            supporting_fact_precision=1.0,
            supporting_fact_recall=1.0,
            supporting_fact_f1=1.0,
        )
        for index in range(4)
    ]

    comparison = paired_bootstrap_comparison(
        baseline,
        candidate,
        resamples=50,
        seed=9,
    )

    citation = comparison["metrics"]["citation_f1"]
    supporting = comparison["metrics"]["supporting_fact_f1"]
    assert citation["delta_candidate_minus_baseline"] == pytest.approx(0.52381)
    assert citation["probability_candidate_better"] == 1.0
    assert citation["paired_outcomes"] == {
        "candidate_wins": 4,
        "ties": 0,
        "candidate_losses": 0,
    }
    assert supporting["delta_candidate_minus_baseline"] == 0.5


def test_paired_bootstrap_requires_aligned_ids():
    baseline = [
        _row(
            "T1",
            question_type="single_evidence",
            answer_f1=0.2,
            evidence_recall=0.5,
            behavior_correct=False,
            correct=False,
            confidence=0.2,
        )
    ]
    candidate = [
        _row(
            "T2",
            question_type="single_evidence",
            answer_f1=0.8,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.8,
        )
    ]

    with pytest.raises(ValueError, match="aligned by question_id"):
        paired_bootstrap_comparison(baseline, candidate, resamples=10)


def test_paired_bootstrap_includes_dependency_robust_sensitivity():
    baseline = [
        _row(
            "T1",
            question_type="single_evidence",
            answer_f1=0.0,
            evidence_recall=0.0,
            behavior_correct=False,
            correct=False,
            confidence=0.1,
            dependency_group="component-a",
        ),
        _row(
            "T2",
            question_type="single_evidence",
            answer_f1=0.5,
            evidence_recall=0.5,
            behavior_correct=True,
            correct=True,
            confidence=0.5,
            dependency_group="component-b",
        ),
    ]
    candidate = [
        _row(
            "T1",
            question_type="single_evidence",
            answer_f1=1.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.9,
            dependency_group="component-a",
        ),
        _row(
            "T2",
            question_type="single_evidence",
            answer_f1=0.5,
            evidence_recall=0.5,
            behavior_correct=True,
            correct=True,
            confidence=0.5,
            dependency_group="component-b",
        ),
    ]

    comparison = paired_bootstrap_comparison(
        baseline,
        candidate,
        resamples=200,
        seed=12,
    )

    robust = comparison["dependency_robust"]
    assert robust["clusters"] == 2
    assert robust["metrics"]["answer_f1"]["delta_candidate_minus_baseline"] == 0.5


def test_paired_bootstrap_skips_undefined_sparse_metric_replicates():
    baseline = [
        _row(
            "C1",
            question_type="conflict",
            answer_f1=0.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=False,
            confidence=0.2,
            conflict_fn=1,
            dependency_group="component-a",
        ),
        _row(
            "C2",
            question_type="conflict",
            answer_f1=0.0,
            evidence_recall=0.0,
            behavior_correct=False,
            correct=False,
            confidence=0.1,
            dependency_group="component-b",
        ),
    ]
    candidate = [
        _row(
            "C1",
            question_type="conflict",
            answer_f1=1.0,
            evidence_recall=1.0,
            behavior_correct=True,
            correct=True,
            confidence=0.9,
            conflict_tp=1,
            dependency_group="component-a",
        ),
        _row(
            "C2",
            question_type="conflict",
            answer_f1=0.0,
            evidence_recall=0.0,
            behavior_correct=False,
            correct=False,
            confidence=0.1,
            dependency_group="component-b",
        ),
    ]

    comparison = paired_bootstrap_comparison(
        baseline,
        candidate,
        resamples=200,
        seed=23,
    )

    assert comparison["effective_resamples_by_metric"]["answer_f1"] == 200
    assert 0 < comparison["effective_resamples_by_metric"]["conflict_micro_f1"] < 200
    robust = comparison["dependency_robust"]
    assert 0 < robust["effective_resamples_by_metric"]["conflict_micro_f1"] < 200
