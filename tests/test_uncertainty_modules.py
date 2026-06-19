"""Tests for uncertainty modules: estimator, calibrator, controller."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_structures import (
    ConflictEdge,
    ConflictType,
    Evidence,
    EvidenceConflictGraph,
    SubQuestion,
    UncertaintyBreakdown,
)

# --- UncertaintyEstimator Tests ---

class TestUncertaintyEstimator:
    def _make_subquestions(self, coverage=0.8):
        return [
            SubQuestion(id="sq1", question="子问题1", required_evidence_type="factual",
                       dependency_ids=[], requires_counter_evidence=False,
                       status="completed", coverage_score=coverage),
        ]

    def _make_evidence(self, count=3):
        return [
            Evidence(evidence_id=f"E{i+1}", source="paper", title=f"T{i}",
                    text_span=f"证据{i}", credibility_score=0.8, relevance_score=0.7)
            for i in range(count)
        ]

    def test_estimate_returns_breakdown(self):
        from src.uncertainty.estimator import UncertaintyEstimator
        estimator = UncertaintyEstimator()
        result = estimator.estimate(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )
        assert isinstance(result, UncertaintyBreakdown)
        assert 0 <= result.retrieval_uncertainty <= 1
        assert 0 <= result.evidence_conflict <= 1
        assert 0 <= result.overall <= 1

    def test_more_evidence_less_retrieval_uncertainty(self):
        from src.uncertainty.estimator import UncertaintyEstimator
        estimator = UncertaintyEstimator()
        few = estimator.estimate(self._make_subquestions(), self._make_evidence(1), EvidenceConflictGraph())
        many = estimator.estimate(self._make_subquestions(), self._make_evidence(10), EvidenceConflictGraph())
        assert few.retrieval_uncertainty >= many.retrieval_uncertainty

    def test_conflicts_increase_uncertainty(self):
        from src.uncertainty.estimator import UncertaintyEstimator
        estimator = UncertaintyEstimator()
        graph_clean = EvidenceConflictGraph()
        graph_conflict = EvidenceConflictGraph()
        graph_conflict.add_edge(ConflictEdge(source_id="E1", target_id="E2",
                                              conflict_type=ConflictType.NUMERIC_CONFLICT,
                                              severity="high", confidence=0.9))
        clean = estimator.estimate(self._make_subquestions(), self._make_evidence(), graph_clean)
        conflict = estimator.estimate(self._make_subquestions(), self._make_evidence(), graph_conflict)
        assert conflict.evidence_conflict >= clean.evidence_conflict

    def test_estimate_for_answer(self):
        from src.uncertainty.estimator import UncertaintyEstimator
        estimator = UncertaintyEstimator()
        base = UncertaintyBreakdown(
            retrieval_uncertainty=0.2, evidence_conflict=0.1,
            reasoning_gap=0.15, source_reliability=0.1, verification_uncertainty=0.05,
        )
        result = estimator.estimate_for_answer(
            answer_confidence=0.9, verification_confidence=0.85, base_uncertainty=base,
        )
        assert isinstance(result, UncertaintyBreakdown)

    def test_custom_weights(self):
        from src.uncertainty.estimator import UncertaintyEstimator
        estimator = UncertaintyEstimator(config={"weights": {"retrieval": 1.0, "conflict": 0, "reasoning": 0, "source": 0, "verification": 0}})
        result = estimator.estimate(self._make_subquestions(), self._make_evidence(), EvidenceConflictGraph())
        assert isinstance(result, UncertaintyBreakdown)


# --- ConfidenceCalibrator Tests ---

class TestConfidenceCalibrator:
    def test_add_sample(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        cal.add_sample(0.8, True)
        cal.add_sample(0.3, False)
        assert len(cal.calibration_data) == 2

    def test_compute_metrics(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        for conf, correct in [(0.9, True), (0.8, True), (0.3, False), (0.7, True), (0.2, False)]:
            cal.add_sample(conf, correct)
        metrics = cal.compute_calibration_metrics(
            confidences=[0.9, 0.8, 0.3, 0.7, 0.2],
            correct=[True, True, False, True, False],
        )
        assert "ece" in metrics
        assert "brier_score" in metrics
        assert metrics["ece"] >= 0
        assert 0 <= metrics["brier_score"] <= 1

    def test_calibrate_confidence(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        unc = UncertaintyBreakdown(
            retrieval_uncertainty=0.2, evidence_conflict=0.1,
            reasoning_gap=0.15, source_reliability=0.1, verification_uncertainty=0.05,
        )
        calibrated = cal.calibrate_confidence(raw_confidence=0.8, uncertainty=unc)
        assert 0 <= calibrated <= 1

    def test_calibrate_confidence_preserves_strong_low_uncertainty_signal(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()
        low_uncertainty = UncertaintyBreakdown(
            retrieval_uncertainty=0.05,
            evidence_conflict=0.0,
            reasoning_gap=0.05,
            source_reliability=0.05,
            verification_uncertainty=0.0,
        )
        high_uncertainty = UncertaintyBreakdown(
            retrieval_uncertainty=0.8,
            evidence_conflict=0.8,
            reasoning_gap=0.7,
            source_reliability=0.6,
            verification_uncertainty=0.8,
        )

        strong = cal.calibrate_confidence(raw_confidence=0.8, uncertainty=low_uncertainty)
        weak = cal.calibrate_confidence(raw_confidence=0.8, uncertainty=high_uncertainty)

        assert strong > 0.75
        assert weak < strong
        assert weak < 0.65

    def test_empty_metrics(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        metrics = cal.compute_calibration_metrics(confidences=[], correct=[])
        assert metrics["ece"] == 0

    def test_ece_includes_confidence_one_in_final_bin(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()

        perfect = cal.compute_calibration_metrics(
            confidences=[1.0],
            correct=[True],
        )
        overconfident = cal.compute_calibration_metrics(
            confidences=[1.0],
            correct=[False],
        )

        assert perfect["ece"] == 0.0
        assert overconfident["ece"] == 1.0

    def test_temperature_calibration_handles_probability_boundaries(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()

        temperature = cal.calibrate_temperature(
            predictions=np.array([0.0, 1.0, 0.8]),
            labels=np.array([0, 1, 1]),
            max_iter=5,
        )

        assert np.isfinite(temperature)
        assert 0.1 <= temperature <= 10.0

    def test_temperature_calibration_accepts_logits(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()

        temperature = cal.calibrate_temperature(
            predictions=np.array([-2.0, 0.0, 3.0]),
            labels=np.array([0, 0, 1]),
            max_iter=3,
        )

        assert np.isfinite(temperature)

    @pytest.mark.parametrize(
        ("predictions", "labels", "message"),
        [
            (np.array([]), np.array([]), "at least one sample"),
            (np.array([0.1, 0.2]), np.array([1]), "same shape"),
            (np.array([np.nan]), np.array([1]), "finite"),
            (np.array([0.8]), np.array([0.5]), "binary"),
        ],
    )
    def test_temperature_calibration_rejects_invalid_inputs(
        self, predictions, labels, message
    ):
        from src.uncertainty.calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()

        with pytest.raises(ValueError, match=message):
            cal.calibrate_temperature(predictions, labels)

    def test_temperature_calibration_rejects_invalid_optimizer_settings(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()

        with pytest.raises(ValueError, match="max_iter"):
            cal.calibrate_temperature(
                np.array([0.8]), np.array([1]), max_iter=-1
            )

        with pytest.raises(ValueError, match="lr"):
            cal.calibrate_temperature(
                np.array([0.8]), np.array([1]), lr=0
            )

    def test_calibration_metrics_reject_invalid_inputs(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()

        with pytest.raises(ValueError, match="same length"):
            cal.compute_calibration_metrics([0.9], [])

        with pytest.raises(ValueError, match="same length"):
            cal.compute_calibration_metrics([], [True])

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            cal.compute_calibration_metrics([1.2], [True])

        with pytest.raises(ValueError, match="booleans"):
            cal.compute_calibration_metrics([0.8], [1])

        with pytest.raises(ValueError, match="finite"):
            cal.calibrate_temperature(np.array([0.8]), np.array([np.nan]))


# --- UncertaintyController Tests ---

class TestUncertaintyController:
    def _make_subquestions(self):
        return [SubQuestion(id="sq1", question="Q", required_evidence_type="factual",
                           dependency_ids=[], requires_counter_evidence=False,
                           status="completed", coverage_score=0.8)]

    def _make_evidence(self, count=5):
        return [Evidence(evidence_id=f"E{i+1}", source="paper", title=f"T{i}",
                        text_span=f"证据{i}", credibility_score=0.8)
                for i in range(count)]

    class _FakeEstimator:
        def __init__(self, breakdown):
            self.breakdown = breakdown

        def estimate(self, *args, **kwargs):
            return self.breakdown

    def _controller_with_breakdown(self, breakdown, config=None):
        from src.uncertainty.controller import UncertaintyController

        ctrl = UncertaintyController(config=config)
        ctrl.estimator = self._FakeEstimator(breakdown)
        return ctrl

    def test_assess_proceed_with_good_evidence(self):
        from src.uncertainty.controller import Action, UncertaintyController
        ctrl = UncertaintyController()
        decision = ctrl.assess(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(10),
            conflict_graph=EvidenceConflictGraph(),
            current_round=3, max_rounds=5,
        )
        assert isinstance(decision.action, Action)
        assert decision.confidence >= 0

    def test_assess_abstains_when_overall_uncertainty_exceeds_threshold(self):
        from src.uncertainty.controller import Action

        ctrl = self._controller_with_breakdown(
            UncertaintyBreakdown(
                retrieval_uncertainty=1.0,
                evidence_conflict=1.0,
                reasoning_gap=1.0,
                source_reliability=1.0,
                verification_uncertainty=1.0,
            )
        )

        decision = ctrl.assess(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )

        assert decision.action == Action.ABSTAIN
        assert decision.should_stop is True

    def test_assess_lowers_confidence_when_max_rounds_exhausted_with_high_uncertainty(self):
        from src.uncertainty.controller import Action

        ctrl = self._controller_with_breakdown(
            UncertaintyBreakdown(
                retrieval_uncertainty=0.5,
                evidence_conflict=0.5,
                reasoning_gap=0.5,
                source_reliability=0.5,
                verification_uncertainty=0.5,
            ),
            config={"high_threshold": 0.4, "abstain_threshold": 0.9},
        )

        decision = ctrl.assess(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
            current_round=2,
            max_rounds=2,
        )

        assert decision.action == Action.LOWER_CONFIDENCE
        assert decision.should_stop is True

    def test_assess_proceeds_when_max_rounds_exhausted_with_acceptable_uncertainty(self):
        from src.uncertainty.controller import Action

        ctrl = self._controller_with_breakdown(
            UncertaintyBreakdown(
                retrieval_uncertainty=0.1,
                evidence_conflict=0.1,
                reasoning_gap=0.1,
                source_reliability=0.1,
                verification_uncertainty=0.1,
            )
        )

        decision = ctrl.assess(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
            current_round=3,
            max_rounds=3,
        )

        assert decision.action == Action.PROCEED
        assert decision.should_stop is True

    def test_assess_resolves_conflicts_before_more_retrieval(self):
        from src.uncertainty.controller import Action

        ctrl = self._controller_with_breakdown(
            UncertaintyBreakdown(
                retrieval_uncertainty=0.9,
                evidence_conflict=0.8,
                reasoning_gap=0.0,
                source_reliability=0.0,
                verification_uncertainty=0.0,
            )
        )

        decision = ctrl.assess(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )

        assert decision.action == Action.RESOLVE_CONFLICTS

    def test_assess_continues_retrieval_for_moderate_overall_uncertainty(self):
        from src.uncertainty.controller import Action

        ctrl = self._controller_with_breakdown(
            UncertaintyBreakdown(
                retrieval_uncertainty=0.4,
                evidence_conflict=0.4,
                reasoning_gap=0.4,
                source_reliability=0.4,
                verification_uncertainty=0.4,
            ),
            config={"high_threshold": 0.8, "abstain_threshold": 0.9},
        )

        decision = ctrl.assess(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )

        assert decision.action == Action.CONTINUE_RETRIEVAL
        assert decision.should_stop is False

    def test_assess_uses_high_conflict_threshold_as_legacy_high_threshold(self):
        from src.uncertainty.controller import Action

        ctrl = self._controller_with_breakdown(
            UncertaintyBreakdown(
                retrieval_uncertainty=0.0,
                evidence_conflict=0.55,
                reasoning_gap=0.0,
                source_reliability=0.0,
                verification_uncertainty=0.0,
            ),
            config={"high_conflict_threshold": 0.5},
        )

        decision = ctrl.assess(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )

        assert decision.action == Action.RESOLVE_CONFLICTS

    def test_assess_continue_with_low_coverage(self):
        from src.uncertainty.controller import Action, UncertaintyController
        ctrl = UncertaintyController(config={"acceptable_threshold": 0.1})
        sq = [SubQuestion(id="sq1", question="Q", required_evidence_type="factual",
                         dependency_ids=[], requires_counter_evidence=False,
                         status="pending", coverage_score=0.1)]
        decision = ctrl.assess(
            subquestions=sq, evidence_pool=self._make_evidence(1),
            conflict_graph=EvidenceConflictGraph(),
            current_round=1, max_rounds=5,
        )
        assert decision.action == Action.CONTINUE_RETRIEVAL

    def test_assess_for_repair_with_unsupported_claims(self):
        from src.uncertainty.controller import Action, UncertaintyController
        ctrl = UncertaintyController()
        decision = ctrl.assess_for_repair(
            verification_uncertainty=0.6, has_unsupported_claims=True, has_ignored_conflicts=False,
        )
        assert decision.action == Action.REPAIR_ANSWER

    def test_assess_for_repair_prioritizes_ignored_conflicts(self):
        from src.uncertainty.controller import Action, UncertaintyController

        ctrl = UncertaintyController()

        decision = ctrl.assess_for_repair(
            verification_uncertainty=0.1,
            has_unsupported_claims=True,
            has_ignored_conflicts=True,
        )

        assert decision.action == Action.REPAIR_ANSWER
        assert "Conflicts" in decision.reason
        assert decision.confidence == 0.5

    def test_assess_for_repair_lowers_confidence_on_high_verification_uncertainty(self):
        from src.uncertainty.controller import Action, UncertaintyController

        ctrl = UncertaintyController()

        decision = ctrl.assess_for_repair(
            verification_uncertainty=0.8,
            has_unsupported_claims=False,
            has_ignored_conflicts=False,
        )

        assert decision.action == Action.LOWER_CONFIDENCE
        assert decision.should_stop is True

    def test_assess_for_repair_proceeds_when_verification_uncertainty_is_low(self):
        from src.uncertainty.controller import Action, UncertaintyController

        ctrl = UncertaintyController()

        decision = ctrl.assess_for_repair(
            verification_uncertainty=0.2,
            has_unsupported_claims=False,
            has_ignored_conflicts=False,
        )

        assert decision.action == Action.PROCEED
        assert decision.should_stop is True

    @pytest.mark.parametrize(
        "config",
        [
            {"acceptable_threshold": -0.1},
            {"high_threshold": 1.2},
            {"abstain_threshold": "0.7"},
            {"acceptable_threshold": 0.8, "high_threshold": 0.6},
            {"high_threshold": 0.8, "abstain_threshold": 0.7},
        ],
    )
    def test_controller_rejects_invalid_threshold_config(self, config):
        from src.uncertainty.controller import UncertaintyController

        with pytest.raises(ValueError):
            UncertaintyController(config=config)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"reasoning_completeness": 1.2},
            {"reasoning_completeness": True},
            {"current_round": -1},
            {"max_rounds": -1},
            {"current_round": 1.5},
            {"max_rounds": 1.5},
        ],
    )
    def test_assess_rejects_invalid_public_inputs(self, kwargs):
        from src.uncertainty.controller import UncertaintyController

        ctrl = UncertaintyController()
        params = {
            "subquestions": self._make_subquestions(),
            "evidence_pool": self._make_evidence(),
            "conflict_graph": EvidenceConflictGraph(),
        }
        params.update(kwargs)

        with pytest.raises(ValueError):
            ctrl.assess(**params)

    @pytest.mark.parametrize("verification_uncertainty", [1.1, "0.1"])
    def test_assess_for_repair_rejects_invalid_verification_uncertainty(
        self, verification_uncertainty
    ):
        from src.uncertainty.controller import UncertaintyController

        ctrl = UncertaintyController()

        with pytest.raises(ValueError, match="verification_uncertainty"):
            ctrl.assess_for_repair(
                verification_uncertainty=verification_uncertainty,
                has_unsupported_claims=False,
                has_ignored_conflicts=False,
            )

    def test_get_uncertainty_breakdown(self):
        from src.uncertainty.controller import UncertaintyController
        ctrl = UncertaintyController()
        breakdown = ctrl.get_uncertainty_breakdown(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )
        assert isinstance(breakdown, UncertaintyBreakdown)
