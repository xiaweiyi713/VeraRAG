"""Repair Agent for VeraRAG."""

from typing import Any

from ..utils.data_structures import (
    AnswerClaim,
    Evidence,
    VerificationReport,
    VerificationStatus,
)
from .base import BaseAgent


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

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any | None = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert editor for fact-checked content.
Your job is to repair answers to ensure they are well-supported by evidence.
Be conservative: if evidence is weak, say so explicitly.
Output ONLY valid JSON, no other text."""

    def repair_answer(
        self,
        answer: str,
        claims: list[AnswerClaim],
        verification_report: VerificationReport,
        evidence: list[Evidence]
    ) -> tuple[str, list[AnswerClaim]]:
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
        verification: dict[str, Any],
        evidence: list[Evidence]
    ) -> AnswerClaim:
        """Repair a single claim based on verification."""
        status = self._normalize_status(verification["status"])
        confidence = self._normalize_confidence(
            verification.get("confidence", claim.confidence)
        )

        if status is VerificationStatus.REFUTED:
            # Refuted claims should be removed or heavily qualified
            return AnswerClaim(
                claim=self._downgrade_claim(claim.claim),
                supporting_evidence=claim.supporting_evidence,
                conflicting_evidence=claim.conflicting_evidence,
                confidence=confidence * 0.3,
                verification_status=status
            )

        elif status is VerificationStatus.NOT_ENOUGH_INFO:
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
        """Downgrade a refuted claim to be more tentative (Chinese)."""
        hedge = "（注：该说法证据不足，甚至存在反证）"
        if hedge in claim_text:
            return claim_text
        return f"{claim_text}{hedge}"

    def _add_uncertainty_hedge(self, claim_text: str) -> str:
        """Add an uncertainty hedge to an under-supported claim (Chinese)."""
        hedge = "现有证据有限，"
        if claim_text.startswith(hedge):
            return claim_text
        return hedge + claim_text

    def _normalize_status(self, status: Any) -> VerificationStatus:
        """Normalize verifier status values from enum, enum-name, or wire value."""
        if isinstance(status, VerificationStatus):
            return status
        if not isinstance(status, str):
            raise ValueError("verification status must be a VerificationStatus or string")

        normalized = status.strip().lower()
        by_name = {
            "supported": VerificationStatus.SUPPORTED,
            "refuted": VerificationStatus.REFUTED,
            "not_enough_info": VerificationStatus.NOT_ENOUGH_INFO,
            "not enough info": VerificationStatus.NOT_ENOUGH_INFO,
            "nei": VerificationStatus.NOT_ENOUGH_INFO,
        }
        if normalized in by_name:
            return by_name[normalized]

        raise ValueError(f"Unknown verification status: {status}")

    def _normalize_confidence(self, confidence: Any) -> float:
        """Normalize and clamp verifier confidence into [0, 1]."""
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ValueError("verification confidence must be a number in [0, 1]")
        return max(0.0, min(1.0, float(confidence)))

    def _generate_repaired_answer(
        self,
        original_answer: str,
        repaired_claims: list[AnswerClaim],
        verification_report: VerificationReport
    ) -> str:
        """Repair the answer text while PRESERVING the reasoning agent's behavior.

        The reasoning agent already decides the high-level behavior (正常作答 /
        拒答 / 纠正前提 / 标注冲突) and phrases the Chinese answer accordingly.
        Rebuilding the answer from claims here would destroy that framing (and
        risk injecting spurious "conflict" notes), so we keep the original text
        and only append a conservative caveat when some claims are unsupported.
        """
        answer = (original_answer or "").strip()
        if not answer:
            # 仅当推理未给出任何答案时，才回退为明确拒答
            return "根据现有证据，信息不足，无法给出可靠回答。"

        has_unsupported = any(
            getattr(c, "verification_status", None)
            in (VerificationStatus.NOT_ENOUGH_INFO, VerificationStatus.REFUTED)
            for c in repaired_claims
        )
        caveat = "（注：上述部分论断证据有限，请谨慎对待。）"
        if has_unsupported and caveat not in answer:
            answer = f"{answer}\n{caveat}"
        return answer

    def run(self, *args, **kwargs) -> Any:
        """Run the repair agent."""
        return self.repair_answer(*args, **kwargs)
