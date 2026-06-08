"""Tests for uncertainty modules: estimator, calibrator, controller."""

import sys
from pathlib import Path

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

    def test_empty_metrics(self):
        from src.uncertainty.calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        metrics = cal.compute_calibration_metrics(confidences=[], correct=[])
        assert metrics["ece"] == 0


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

    def test_get_uncertainty_breakdown(self):
        from src.uncertainty.controller import UncertaintyController
        ctrl = UncertaintyController()
        breakdown = ctrl.get_uncertainty_breakdown(
            subquestions=self._make_subquestions(),
            evidence_pool=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )
        assert isinstance(breakdown, UncertaintyBreakdown)
