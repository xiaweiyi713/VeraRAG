"""Repair Agent for VeraRAG."""

import json
from typing import Dict, Any, Optional, List

from .base import BaseAgent
from ..utils.data_structures import (
    AnswerClaim,
    VerificationReport,
    Evidence
)


class RepairAgent(BaseAgent):
    """
    Repair agent that fixes issues identified by the verifier.

    Handles:
    1. Removing unsupported claims
    2. Downgrading overconfident language
    3. Adding conflict explanations
    4. Attaching missing citations
    5. Flagging claims needing more research
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client: Optional[Any] = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert editor for fact-checked content.
Your job is to repair answers to ensure they are well-supported by evidence.
Be conservative: if evidence is weak, say so explicitly.
Output ONLY valid JSON, no other text."""

    def repair_answer(
        self,
        answer: str,
        claims: List[AnswerClaim],
        verification_report: VerificationReport,
        evidence: List[Evidence]
    ) -> tuple[str, List[AnswerClaim]]:
        """
        Repair the answer based on verification issues.

        Args:
            answer: The original answer
            claims: Claims made in the answer
            verification_report: Verification findings
            evidence: Available evidence

        Returns:
            Tuple of (repaired_answer, repaired_claims)
        """
        if not verification_report.has_critical_issues():
            return answer, claims

        # Repair each claim
        repaired_claims = []
        claim_map = {c.claim: c for c in claims}

        for verification in verification_report.claim_verifications:
            claim_text = verification["claim"]
            original_claim = claim_map.get(claim_text)

            if original_claim is None:
                continue

            repaired_claim = self._repair_claim(
                original_claim,
                verification,
                evidence
            )
            repaired_claims.append(repaired_claim)

        # Generate repaired answer text
        repaired_answer = self._generate_repaired_answer(
            answer,
            repaired_claims,
            verification_report
        )

        return repaired_answer, repaired_claims

    def _repair_claim(
        self,
        claim: AnswerClaim,
        verification: Dict[str, Any],
        evidence: List[Evidence]
    ) -> AnswerClaim:
        """Repair a single claim based on verification."""
        status = verification["status"]
        confidence = verification.get("confidence", claim.confidence)

        if status == "REFUTED":
            # Refuted claims should be removed or heavily qualified
            return AnswerClaim(
                claim=self._downgrade_claim(claim.claim),
                supporting_evidence=claim.supporting_evidence,
                conflicting_evidence=claim.conflicting_evidence,
                confidence=confidence * 0.3,
                verification_status=status
            )

        elif status == "NOT_ENOUGH_INFO":
            # Unsupported claims should be downgraded
            return AnswerClaim(
                claim=self._add_uncertainty_hedge(claim.claim),
                supporting_evidence=claim.supporting_evidence,
                conflicting_evidence=claim.conflicting_evidence,
                confidence=confidence * 0.6,
                verification_status=status
            )

        else:  # SUPPORTED
            return AnswerClaim(
                claim=claim.claim,
                supporting_evidence=claim.supporting_evidence,
                conflicting_evidence=claim.conflicting_evidence,
                confidence=confidence,
                verification_status=status
            )

    def _downgrade_claim(self, claim_text: str) -> str:
        """Downgrade a claim to be more tentative."""
        # Add uncertainty language
        hedges = [
            "It is possible that ",
            "Some evidence suggests ",
            "There is limited indication that ",
            "It has been suggested that "
        ]

        # Select appropriate hedge based on claim
        claim_lower = claim_text.lower()

        if any(word in claim_lower for word in ["all", "every", "always", "never"]):
            return claim_text.replace("all", "some").replace("every", "some").replace("always", "may sometimes").replace("never", "rarely")

        # Add hedge if not already present
        if not any(hedge in claim_text for hedge in hedges):
            return hedges[0] + claim_text[0].lower() + claim_text[1:]

        return claim_text

    def _add_uncertainty_hedge(self, claim_text: str) -> str:
        """Add uncertainty hedge to a claim."""
        hedges = [
            "The available evidence is limited, but ",
            "Based on the current evidence, it appears that ",
            "While not conclusively established, "
        ]

        for hedge in hedges:
            if hedge not in claim_text:
                return hedge + claim_text[0].lower() + claim_text[1:]

        return claim_text

    def _generate_repaired_answer(
        self,
        original_answer: str,
        repaired_claims: List[AnswerClaim],
        verification_report: VerificationReport
    ) -> str:
        """Generate repaired answer text."""
        # For now, simple concatenation of repaired claims
        # Can be enhanced to maintain narrative flow

        if not repaired_claims:
            return "Unable to provide a reliable answer due to insufficient evidence."

        answer_parts = []

        # Group claims by confidence
        high_conf = [c for c in repaired_claims if c.confidence >= 0.7]
        med_conf = [c for c in repaired_claims if 0.4 <= c.confidence < 0.7]
        low_conf = [c for c in repaired_claims if c.confidence < 0.4]

        if high_conf:
            answer_parts.append("Based on the available evidence:")
            for c in high_conf:
                answer_parts.append(f"- {c.claim}")

        if med_conf:
            answer_parts.append("There is some indication that:")
            for c in med_conf:
                answer_parts.append(f"- {c.claim}")

        if low_conf:
            answer_parts.append("The following claims are uncertain due to limited evidence:")
            for c in low_conf:
                answer_parts.append(f"- {c.claim}")

        # Add conflict acknowledgment
        if verification_report.ignored_conflicts:
            answer_parts.append("\nNote: There are conflicting perspectives on some aspects of this question.")

        return "\n".join(answer_parts)

    def run(self, *args, **kwargs) -> Any:
        """Run the repair agent."""
        return self.repair_answer(*args, **kwargs)
