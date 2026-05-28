"""Verifier Agent for VeraRAG."""

import json
from typing import Any

from ..utils.data_structures import (
    AnswerClaim,
    Evidence,
    EvidenceConflictGraph,
    VerificationReport,
    VerificationStatus,
)
from .base import BaseAgent


class VerifierAgent(BaseAgent):
    """
    Verifier agent that checks answer claims against evidence.

    Performs claim-level verification to ensure:
    1. Each claim has supporting evidence
    2. No claims contradict the evidence
    3. Numerical claims match evidence
    4. Entities are consistent
    5. Conflicts are acknowledged
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_client: Any | None = None,
        use_nli: bool = True
    ):
        super().__init__(config, llm_client)
        self.use_nli = use_nli
        self.nli_model: Any = None
        self.system_prompt = """You are a fact-checking expert.
Your job is to verify claims against evidence with high precision.
Output ONLY valid JSON, no other text."""

    def verify_answer(
        self,
        answer: str,
        claims: list[AnswerClaim],
        evidence: list[Evidence],
        conflict_graph: EvidenceConflictGraph
    ) -> VerificationReport:
        """
        Verify the answer against evidence.

        Args:
            answer: The generated answer
            claims: Claims made in the answer
            evidence: Retrieved evidence
            conflict_graph: Evidence conflict relationships

        Returns:
            VerificationReport with findings
        """
        report = VerificationReport()

        # Create evidence lookup
        evidence_map = {ev.evidence_id: ev for ev in evidence}

        # Verify each claim
        for claim in claims:
            verification = self._verify_claim(claim, evidence_map, conflict_graph)

            report.claim_verifications.append(verification)

            # Track issues
            if verification["status"] == "NOT_ENOUGH_INFO":
                report.missing_evidence_for.append(claim.claim)
            elif verification["confidence"] < 0.5:
                report.overconfident_claims.append(claim.claim)

        # Check for ignored conflicts
        report.ignored_conflicts = self._check_ignored_conflicts(
            claims, conflict_graph
        )

        # Determine overall status
        supported_count = sum(
            1 for v in report.claim_verifications
            if v["status"] == "SUPPORTED"
        )
        refuted_count = sum(
            1 for v in report.claim_verifications
            if v["status"] == "REFUTED"
        )

        if refuted_count > 0:
            report.overall_status = VerificationStatus.REFUTED
        elif supported_count == len(claims):
            report.overall_status = VerificationStatus.SUPPORTED
        else:
            report.overall_status = VerificationStatus.NOT_ENOUGH_INFO

        # Compile issues
        if report.missing_evidence_for:
            report.issues.append({
                "type": "unsupported_claim",
                "description": f"{len(report.missing_evidence_for)} claims lack sufficient evidence"
            })

        if report.ignored_conflicts:
            report.issues.append({
                "type": "conflict_ignored",
                "description": f"{len(report.ignored_conflicts)} conflicts were not acknowledged"
            })

        return report

    def _verify_claim(
        self,
        claim: AnswerClaim,
        evidence_map: dict[str, Evidence],
        conflict_graph: EvidenceConflictGraph
    ) -> dict[str, Any]:
        """Verify a single claim."""
        # Get supporting evidence
        supporting_texts = []
        for ev_id in claim.supporting_evidence:
            if ev_id in evidence_map:
                ev = evidence_map[ev_id]
                supporting_texts.append(f"{ev.title}: {ev.text_span}")

        # Get conflicting evidence
        conflicting_texts = []
        for ev_id in claim.conflicting_evidence:
            if ev_id in evidence_map:
                ev = evidence_map[ev_id]
                conflicting_texts.append(f"{ev.title}: {ev.text_span}")

        # Use NLI if available
        if self.use_nli and supporting_texts:
            nli_result = self._nli_verify(claim.claim, supporting_texts)
            if nli_result:
                return nli_result

        # Fall back to LLM verification
        return self._llm_verify_claim(claim.claim, supporting_texts, conflicting_texts)

    def _nli_verify(
        self,
        claim: str,
        evidence_texts: list[str]
    ) -> dict[str, Any] | None:
        """Use NLI model for verification."""
        try:
            if self.nli_model is None:
                from sentence_transformers import CrossEncoder
                self.nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

            # Score against each evidence
            scores = []
            for ev_text in evidence_texts:
                score = self.nli_model.predict([[claim, ev_text]])[0]
                scores.append(score)

            # NLI models typically output logits for classes
            # Assuming: [contradiction, entailment, neutral] or similar
            avg_score = sum(scores) / len(scores)

            # Interpret score
            if avg_score > 2.0:  # Entailment
                return {
                    "claim": claim,
                    "status": "SUPPORTED",
                    "confidence": min(0.95, 0.5 + (avg_score - 2.0) * 0.2),
                    "method": "nli"
                }
            elif avg_score < 1.0:  # Contradiction
                return {
                    "claim": claim,
                    "status": "REFUTED",
                    "confidence": min(0.95, 0.5 + (1.0 - avg_score) * 0.5),
                    "method": "nli"
                }
            else:  # Neutral
                return {
                    "claim": claim,
                    "status": "NOT_ENOUGH_INFO",
                    "confidence": 0.5,
                    "method": "nli"
                }

        except Exception:
            return None

    def _llm_verify_claim(
        self,
        claim: str,
        supporting_texts: list[str],
        conflicting_texts: list[str]
    ) -> dict[str, Any]:
        """Use LLM for claim verification."""
        prompt = f"""Verify the following claim against the evidence.

Claim: "{claim}"

Supporting Evidence:
{json.dumps(supporting_texts, indent=2) if supporting_texts else "None"}

Conflicting Evidence:
{json.dumps(conflicting_texts, indent=2) if conflicting_texts else "None"}

Determine if the claim is SUPPORTED, REFUTED, or NOT_ENOUGH_INFO based on the evidence.

Output JSON:
{{
    "status": "SUPPORTED|REFUTED|NOT_ENOUGH_INFO",
    "confidence": <0.0-1.0>,
    "supporting_span": "relevant text from evidence",
    "missing_info": "what information is missing (if any)",
    "rationale": "explanation"
}}
"""

        try:
            response = self._call_llm(
                prompt,
                system_prompt=self.system_prompt,
                response_format="json"
            )
            data = json.loads(response)

            return {
                "claim": claim,
                "status": data.get("status", "NOT_ENOUGH_INFO"),
                "confidence": data.get("confidence", 0.5),
                "supporting_span": data.get("supporting_span", ""),
                "missing_info": data.get("missing_info", ""),
                "rationale": data.get("rationale", ""),
                "method": "llm"
            }

        except Exception:
            return {
                "claim": claim,
                "status": "NOT_ENOUGH_INFO",
                "confidence": 0.3,
                "method": "fallback"
            }

    def _check_ignored_conflicts(
        self,
        claims: list[AnswerClaim],
        conflict_graph: EvidenceConflictGraph
    ) -> list[dict[str, Any]]:
        """Check if any conflicts were ignored in the answer."""
        ignored = []

        # Get all conflicts from graph
        conflicts = conflict_graph.get_conflicts()

        for conflict in conflicts:
            # Check if any claim acknowledges this conflict
            acknowledged = any(
                conflict.source_id in claim.supporting_evidence or
                conflict.target_id in claim.conflicting_evidence or
                conflict.source_id in claim.conflicting_evidence or
                conflict.target_id in claim.supporting_evidence
                for claim in claims
            )

            if not acknowledged:
                ignored.append({
                    "conflict_type": conflict.conflict_type.value,
                    "source_id": conflict.source_id,
                    "target_id": conflict.target_id,
                    "confidence": conflict.confidence
                })

        return ignored

    def run(self, *args, **kwargs) -> Any:
        """Run the verifier agent."""
        return self.verify_answer(*args, **kwargs)
