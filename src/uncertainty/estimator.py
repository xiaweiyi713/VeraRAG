"""Uncertainty Estimator for VeraRAG."""

from typing import Dict, Any, Optional, List

from ..utils.data_structures import (
    UncertaintyBreakdown,
    SubQuestion,
    Evidence,
    EvidenceConflictGraph
)


class UncertaintyEstimator:
    """
    Estimates uncertainty across multiple dimensions.

    Uncertainty sources:
    1. Retrieval uncertainty: How well retrieval covered the question
    2. Evidence conflict: How much conflict exists in evidence
    3. Reasoning gap: How complete the reasoning chain is
    4. Source reliability: How reliable the sources are
    5. Verification uncertainty: How uncertain verification is
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.weights = self.config.get("weights", {
            "retrieval": 0.25,
            "conflict": 0.30,
            "reasoning": 0.20,
            "source": 0.15,
            "verification": 0.10
        })

    def estimate(
        self,
        subquestions: List[SubQuestion],
        evidence_pool: List[Evidence],
        conflict_graph: EvidenceConflictGraph,
        reasoning_completeness: float = 0.8
    ) -> UncertaintyBreakdown:
        """
        Estimate overall uncertainty.

        Args:
            subquestions: Decomposed sub-questions
            evidence_pool: Retrieved evidence
            conflict_graph: Evidence conflict relationships
            reasoning_completeness: How complete reasoning is (0-1)

        Returns:
            UncertaintyBreakdown with uncertainty estimates
        """
        breakdown = UncertaintyBreakdown()

        # Retrieval uncertainty
        breakdown.retrieval_uncertainty = self._estimate_retrieval_uncertainty(
            subquestions, evidence_pool
        )

        # Evidence conflict uncertainty
        breakdown.evidence_conflict = self._estimate_conflict_uncertainty(
            conflict_graph
        )

        # Reasoning gap uncertainty
        breakdown.reasoning_gap = 1.0 - reasoning_completeness

        # Source reliability uncertainty
        breakdown.source_reliability = self._estimate_source_uncertainty(
            evidence_pool
        )

        # Verification uncertainty (will be set by verifier)
        breakdown.verification_uncertainty = 0.0

        return breakdown

    def _estimate_retrieval_uncertainty(
        self,
        subquestions: List[SubQuestion],
        evidence_pool: List[Evidence]
    ) -> float:
        """
        Estimate how well retrieval covered the information needs.

        High uncertainty = poor coverage
        """
        if not subquestions:
            return 0.5

        # Average coverage score
        coverage_scores = [sq.coverage_score for sq in subquestions]

        if not coverage_scores:
            return 0.5

        avg_coverage = sum(coverage_scores) / len(coverage_scores)

        # Uncertainty is inverse of coverage
        return 1.0 - avg_coverage

    def _estimate_conflict_uncertainty(
        self,
        conflict_graph: EvidenceConflictGraph
    ) -> float:
        """
        Estimate uncertainty due to evidence conflicts.

        High conflict = high uncertainty
        """
        # Use conflict graph's conflict score
        base_conflict = conflict_graph.get_conflict_score()

        # Also consider confidence of conflicts
        conflicts = conflict_graph.get_conflicts()

        if not conflicts:
            return 0.0

        # Weight by confidence
        weighted_conflict = sum(c.confidence for c in conflicts) / len(conflicts)

        # Combine base score and weighted confidence
        return (base_conflict + weighted_conflict) / 2

    def _estimate_source_uncertainty(
        self,
        evidence_pool: List[Evidence]
    ) -> float:
        """
        Estimate uncertainty due to source reliability issues.

        Low credibility sources = high uncertainty
        """
        if not evidence_pool:
            return 1.0

        # Average credibility score
        avg_credibility = sum(ev.credibility_score for ev in evidence_pool) / len(evidence_pool)

        # Uncertainty is inverse of credibility
        return 1.0 - avg_credibility

    def estimate_for_answer(
        self,
        answer_confidence: float,
        verification_confidence: float,
        base_uncertainty: UncertaintyBreakdown
    ) -> UncertaintyBreakdown:
        """
        Update uncertainty estimates for a specific answer.

        Args:
            answer_confidence: Confidence in the answer
            verification_confidence: Confidence from verification
            base_uncertainty: Base uncertainty breakdown

        Returns:
            Updated uncertainty breakdown
        """
        # Verification uncertainty
        base_uncertainty.verification_uncertainty = 1.0 - verification_confidence

        # Adjust based on answer confidence
        # If answer confidence is high but uncertainty is high, that's problematic
        if answer_confidence > 0.8 and base_uncertainty.overall > 0.5:
            # Increase verification uncertainty
            base_uncertainty.verification_uncertainty = min(
                1.0,
                base_uncertainty.verification_uncertainty + 0.2
            )

        return base_uncertainty
