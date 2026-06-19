"""Verifier Agent for VeraRAG."""

import json
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.data_structures import (
    AnswerClaim,
    Evidence,
    EvidenceConflictGraph,
    VerificationReport,
    VerificationStatus,
)
from ..utils.model_cache import load_optional_model_once
from .base import BaseAgent

logger = logging.getLogger(__name__)


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
        verification_config = (config or {}).get("verification", {})
        self.nli_model_name = verification_config.get(
            "nli_model",
            "cross-encoder/nli-distilroberta-base",
        )
        self.nli_local_files_only = verification_config.get(
            "nli_local_files_only",
            False,
        )
        self.nli_threshold = float(verification_config.get("nli_threshold", 0.7))
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
            verification = self._normalize_verification(verification, claim.claim)

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
        if self.nli_model is None:
            self.nli_model = self._load_nli_model()
        if self.nli_model is None:
            return None

        try:
            # NLI expects premise=evidence and hypothesis=claim.
            raw_scores = self.nli_model.predict(
                [(ev_text, claim) for ev_text in evidence_texts],
                show_progress_bar=False,
            )
            probabilities = self._nli_probabilities(raw_scores)
            if probabilities is None:
                return None
            contradiction_index, entailment_index, neutral_index = self._nli_label_indices()
            contradiction = float(probabilities[:, contradiction_index].max())
            entailment = float(probabilities[:, entailment_index].max())
            neutral = float(probabilities[:, neutral_index].max())

            if entailment >= self.nli_threshold and entailment >= contradiction:
                return {
                    "claim": claim,
                    "status": "SUPPORTED",
                    "confidence": round(entailment, 4),
                    "method": "nli",
                }
            if contradiction >= self.nli_threshold:
                return {
                    "claim": claim,
                    "status": "REFUTED",
                    "confidence": round(contradiction, 4),
                    "method": "nli",
                }
            return {
                "claim": claim,
                "status": "NOT_ENOUGH_INFO",
                "confidence": round(neutral, 4),
                "method": "nli",
            }
        except Exception as exc:
            logger.debug("NLI verification failed: %s", exc)
            return None

    def _load_nli_model(self) -> Any | None:
        model_name = self.nli_model_name

        def factory() -> Any:
            from sentence_transformers import CrossEncoder

            if self.nli_local_files_only:
                return CrossEncoder(model_name, local_files_only=True)
            return CrossEncoder(model_name)

        model, error = load_optional_model_once(
            "cross_encoder",
            model_name,
            factory,
            local_files_only=self.nli_local_files_only,
        )
        if error is not None:
            logger.debug("Verifier NLI unavailable; using LLM fallback: %s", error)
        return model

    def _nli_label_indices(self) -> tuple[int, int, int]:
        """Return contradiction, entailment, and neutral label indices."""
        config = getattr(getattr(self.nli_model, "model", None), "config", None)
        raw_labels = getattr(config, "id2label", {}) or {}
        labels = {
            int(index): str(label).lower()
            for index, label in raw_labels.items()
        }

        def find(marker: str, fallback: int) -> int:
            return next(
                (index for index, label in labels.items() if marker in label),
                fallback,
            )

        return (
            find("contrad", 0),
            find("entail", 1),
            find("neutral", 2),
        )

    @staticmethod
    def _nli_probabilities(raw_scores: Any) -> NDArray[np.float64] | None:
        scores = np.asarray(raw_scores, dtype=float)
        if scores.ndim == 1:
            if scores.shape[0] != 3:
                return None
            scores = scores.reshape(1, 3)
        if scores.ndim != 2 or scores.shape[1] != 3:
            return None
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probabilities: NDArray[np.float64] = np.asarray(
            exp_scores / exp_scores.sum(axis=1, keepdims=True),
            dtype=np.float64,
        )
        return probabilities

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
                "status": self._normalize_status(data.get("status", "NOT_ENOUGH_INFO")),
                "confidence": self._normalize_confidence(data.get("confidence", 0.5)),
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

    def _normalize_verification(
        self,
        verification: dict[str, Any],
        claim: str,
    ) -> dict[str, Any]:
        """Normalize verifier output before it influences report decisions."""
        normalized = dict(verification)
        normalized["claim"] = str(normalized.get("claim", claim) or claim)
        normalized["status"] = self._normalize_status(
            normalized.get("status", "NOT_ENOUGH_INFO")
        )
        normalized["confidence"] = self._normalize_confidence(
            normalized.get("confidence", 0.5)
        )
        return normalized

    def _normalize_status(self, status: Any) -> str:
        """Normalize status values to the verifier's string wire contract."""
        if isinstance(status, VerificationStatus):
            return status.name
        if not isinstance(status, str):
            return VerificationStatus.NOT_ENOUGH_INFO.name

        normalized = status.strip().lower()
        by_name = {
            "supported": VerificationStatus.SUPPORTED.name,
            "refuted": VerificationStatus.REFUTED.name,
            "not_enough_info": VerificationStatus.NOT_ENOUGH_INFO.name,
            "not enough info": VerificationStatus.NOT_ENOUGH_INFO.name,
            "nei": VerificationStatus.NOT_ENOUGH_INFO.name,
        }
        return by_name.get(normalized, VerificationStatus.NOT_ENOUGH_INFO.name)

    def _normalize_confidence(self, confidence: Any) -> float:
        """Normalize and clamp verifier confidence into [0, 1]."""
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            return 0.0
        if not np.isfinite(confidence):
            return 0.0
        return max(0.0, min(1.0, float(confidence)))

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
