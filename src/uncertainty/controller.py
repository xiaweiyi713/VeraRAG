"""Uncertainty Controller for VeraRAG."""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..utils.data_structures import (
    UncertaintyBreakdown,
    SubQuestion,
    Evidence,
    EvidenceConflictGraph
)
from .estimator import UncertaintyEstimator
from .calibrator import ConfidenceCalibrator


class Action(Enum):
    """Actions the uncertainty controller can recommend."""
    CONTINUE_RETRIEVAL = "continue_retrieval"
    RESOLVE_CONFLICTS = "resolve_conflicts"
    REPAIR_ANSWER = "repair_answer"
    ABSTAIN = "abstain"
    PROCEED = "proceed"
    LOWER_CONFIDENCE = "lower_confidence"


@dataclass
class ControlDecision:
    """Decision from the uncertainty controller."""
    action: Action
    reason: str
    confidence: float
    should_stop: bool = False


class UncertaintyController:
    """
    Controls the system behavior based on uncertainty estimates.

    Decides when to:
    1. Continue retrieving
    2. Resolve conflicts
    3. Repair the answer
    4. Abstain from answering
    5. Proceed with current answer
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.estimator = UncertaintyEstimator(config)
        self.calibrator = ConfidenceCalibrator(config)

        # Thresholds
        self.acceptable_threshold = self.config.get("acceptable_threshold", 0.3)
        self.high_threshold = self.config.get("high_threshold", 0.6)
        self.abstain_threshold = self.config.get("abstain_threshold", 0.7)

    def assess(
        self,
        subquestions: List[SubQuestion],
        evidence_pool: List[Evidence],
        conflict_graph: EvidenceConflictGraph,
        reasoning_completeness: float = 0.8,
        max_rounds: int = 5,
        current_round: int = 0
    ) -> ControlDecision:
        """
        Assess current state and recommend action.

        Args:
            subquestions: Current sub-questions
            evidence_pool: Current evidence pool
            conflict_graph: Evidence conflict relationships
            reasoning_completeness: How complete reasoning is
            max_rounds: Maximum retrieval rounds allowed
            current_round: Current retrieval round

        Returns:
            ControlDecision with recommended action
        """
        # Estimate uncertainty
        uncertainty = self.estimator.estimate(
            subquestions, evidence_pool, conflict_graph, reasoning_completeness
        )

        overall_uncertainty = uncertainty.overall

        # Check if we should abstain
        if overall_uncertainty > self.abstain_threshold:
            return ControlDecision(
                action=Action.ABSTAIN,
                reason=f"Overall uncertainty ({overall_uncertainty:.2f}) exceeds abstain threshold ({self.abstain_threshold})",
                confidence=1.0 - overall_uncertainty,
                should_stop=True
            )

        # Check if we've hit max rounds
        if current_round >= max_rounds:
            if overall_uncertainty > self.high_threshold:
                # High uncertainty but out of rounds
                return ControlDecision(
                    action=Action.LOWER_CONFIDENCE,
                    reason=f"Max rounds reached with high uncertainty ({overall_uncertainty:.2f})",
                    confidence=1.0 - overall_uncertainty,
                    should_stop=True
                )
            else:
                # Acceptable uncertainty, proceed
                return ControlDecision(
                    action=Action.PROCEED,
                    reason=f"Max rounds reached with acceptable uncertainty ({overall_uncertainty:.2f})",
                    confidence=self.calibrator.calibrate_confidence(
                        1.0 - overall_uncertainty, uncertainty
                    ),
                    should_stop=True
                )

        # High conflict -> resolve conflicts first
        if uncertainty.evidence_conflict > self.high_threshold:
            return ControlDecision(
                action=Action.RESOLVE_CONFLICTS,
                reason=f"High evidence conflict ({uncertainty.evidence_conflict:.2f})",
                confidence=1.0 - uncertainty.evidence_conflict,
                should_stop=False
            )

        # High retrieval uncertainty -> continue retrieval
        if uncertainty.retrieval_uncertainty > self.high_threshold:
            return ControlDecision(
                action=Action.CONTINUE_RETRIEVAL,
                reason=f"High retrieval uncertainty ({uncertainty.retrieval_uncertainty:.2f})",
                confidence=1.0 - uncertainty.retrieval_uncertainty,
                should_stop=False
            )

        # Moderate uncertainty -> continue with targeted retrieval
        if overall_uncertainty > self.acceptable_threshold:
            return ControlDecision(
                action=Action.CONTINUE_RETRIEVAL,
                reason=f"Uncertainty ({overall_uncertainty:.2f}) above acceptable threshold ({self.acceptable_threshold})",
                confidence=1.0 - overall_uncertainty,
                should_stop=False
            )

        # Low uncertainty -> proceed
        return ControlDecision(
            action=Action.PROCEED,
            reason=f"Acceptable uncertainty ({overall_uncertainty:.2f})",
            confidence=self.calibrator.calibrate_confidence(
                1.0 - overall_uncertainty, uncertainty
            ),
            should_stop=True
        )

    def assess_for_repair(
        self,
        verification_uncertainty: float,
        has_unsupported_claims: bool,
        has_ignored_conflicts: bool
    ) -> ControlDecision:
        """
        Assess whether repair is needed after verification.

        Args:
            verification_uncertainty: Uncertainty from verification
            has_unsupported_claims: Whether there are unsupported claims
            has_ignored_conflicts: Whether conflicts were ignored

        Returns:
            ControlDecision with recommended action
        """
        if has_ignored_conflicts:
            return ControlDecision(
                action=Action.REPAIR_ANSWER,
                reason="Conflicts were ignored in the answer",
                confidence=0.5,
                should_stop=False
            )

        if has_unsupported_claims:
            return ControlDecision(
                action=Action.REPAIR_ANSWER,
                reason="Some claims lack sufficient evidence",
                confidence=0.6,
                should_stop=False
            )

        if verification_uncertainty > self.high_threshold:
            return ControlDecision(
                action=Action.LOWER_CONFIDENCE,
                reason=f"High verification uncertainty ({verification_uncertainty:.2f})",
                confidence=1.0 - verification_uncertainty,
                should_stop=True
            )

        return ControlDecision(
            action=Action.PROCEED,
            reason="Verification passed with acceptable uncertainty",
            confidence=1.0 - verification_uncertainty,
            should_stop=True
        )

    def get_uncertainty_breakdown(
        self,
        subquestions: List[SubQuestion],
        evidence_pool: List[Evidence],
        conflict_graph: EvidenceConflictGraph
    ) -> UncertaintyBreakdown:
        """Get detailed uncertainty breakdown."""
        return self.estimator.estimate(subquestions, evidence_pool, conflict_graph)
