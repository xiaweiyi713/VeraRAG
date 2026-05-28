"""Reasoning Agent for VeraRAG."""

import json
from typing import Any

from ..utils.data_structures import (
    AnswerClaim,
    Evidence,
    EvidenceConflictGraph,
    ReasoningStep,
    SubQuestion,
)
from .base import BaseAgent


class ReasoningAgent(BaseAgent):
    """
    Reasoning agent that generates answers based on evidence.

    Takes into account:
    - The reasoning plan from decomposition
    - Retrieved evidence
    - Evidence conflict graph
    - Uncertainty information
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any | None = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert reasoning agent for complex knowledge tasks.
Your goal is to generate accurate, well-supported answers based on evidence.
Always acknowledge uncertainty and conflicts in the evidence.
Output ONLY valid JSON, no other text."""

    def reason(
        self,
        question: str,
        subquestions: list[SubQuestion],
        evidence: list[Evidence],
        conflict_graph: EvidenceConflictGraph,
        reasoning_plan: list[str]
    ) -> tuple[str, list[AnswerClaim], list[ReasoningStep]]:
        """
        Generate a reasoned answer with claims and reasoning chain.

        Args:
            question: The original question
            subquestions: Decomposed sub-questions
            evidence: Retrieved evidence
            conflict_graph: Evidence conflict relationships
            reasoning_plan: Planned reasoning steps

        Returns:
            Tuple of (answer, answer_claims, reasoning_chain)
        """
        # Prepare evidence context
        evidence_context = self._prepare_evidence_context(evidence)

        # Prepare conflict context
        conflict_context = self._prepare_conflict_context(conflict_graph)

        prompt = f"""Generate a comprehensive answer to the following question based on the evidence.

Question: "{question}"

Sub-questions:
{json.dumps([sq.to_dict() for sq in subquestions], indent=2)}

Evidence:
{evidence_context}

Conflicts in Evidence:
{conflict_context}

Reasoning Plan:
{json.dumps(reasoning_plan, indent=2)}

Generate a JSON response with this structure:
{{
    "answer": "Your comprehensive answer. Acknowledge uncertainty and conflicts.",
    "answer_claims": [
        {{
            "claim": "Specific claim made in the answer",
            "supporting_evidence": ["E1", "E3"],
            "conflicting_evidence": ["E5"],
            "confidence": 0.8,
            "claim_type": "factual",
            "verifiable": true,
            "support_type": "direct"
        }}
    ],
    "reasoning_chain": [
        {{
            "step": 1,
            "description": "First, I established...",
            "evidence_ids": ["E1", "E2"],
            "confidence": 0.9
        }}
    ]
}}

Guidelines:
1. Base your answer ONLY on the provided evidence
2. If evidence conflicts, acknowledge the conflict explicitly
3. If evidence is insufficient, state this clearly
4. Cite specific evidence for each claim
5. Assign confidence scores based on evidence strength
6. Include a clear reasoning chain
7. For each answer_claim, classify:
   - claim_type: "factual" (directly from evidence), "inference" (derived), or "prediction" (forward-looking)
   - verifiable: true if the claim can be checked against evidence
   - support_type: "direct" (evidence explicitly states it), "indirect" (requires inference), or "none" (unsupported)
"""

        response = self._call_llm(
            prompt,
            system_prompt=self.system_prompt,
            response_format="json",
            max_tokens=3000
        )

        try:
            data = json.loads(response)

            answer = data.get("answer", "")
            claims = [
                AnswerClaim(
                    claim=c.get("claim", ""),
                    supporting_evidence=c.get("supporting_evidence", []),
                    conflicting_evidence=c.get("conflicting_evidence", []),
                    confidence=c.get("confidence", 0.5),
                    claim_type=c.get("claim_type", "factual"),
                    verifiable=c.get("verifiable", True),
                    support_type=c.get("support_type", "none"),
                )
                for c in data.get("answer_claims", [])
            ]

            reasoning = [
                ReasoningStep(
                    step=r.get("step", i+1),
                    description=r.get("description", ""),
                    evidence_ids=r.get("evidence_ids", []),
                    confidence=r.get("confidence", 0.5)
                )
                for i, r in enumerate(data.get("reasoning_chain", []))
            ]

            return answer, claims, reasoning

        except (json.JSONDecodeError, KeyError):
            # Fallback: generate simple answer
            return self._fallback_answer(question, evidence)

    def _prepare_evidence_context(self, evidence: list[Evidence]) -> str:
        """Prepare evidence context for the prompt."""
        if not evidence:
            return "No evidence available."

        contexts = []
        for ev in evidence[:20]:  # Limit to prevent token overflow
            contexts.append(f"[{ev.evidence_id}] {ev.title}\n{ev.text_span}\n")

        return "\n".join(contexts)

    def _prepare_conflict_context(self, conflict_graph: EvidenceConflictGraph) -> str:
        """Prepare conflict context for the prompt."""
        conflicts = conflict_graph.get_conflicts()

        if not conflicts:
            return "No significant conflicts detected."

        contexts = []
        for c in conflicts[:10]:
            contexts.append(
                f"- {c.conflict_type.value}: {c.source_id} ↔ {c.target_id} "
                f"(confidence: {c.confidence:.2f})"
            )

        return "\n".join(contexts)

    def _fallback_answer(
        self,
        question: str,
        evidence: list[Evidence]
    ) -> tuple[str, list[AnswerClaim], list[ReasoningStep]]:
        """Generate a fallback answer if LLM fails."""
        if not evidence:
            return (
                "I don't have enough evidence to answer this question.",
                [],
                []
            )

        # Simple synthesis from top evidence
        top_ev = evidence[:3]
        answer = "Based on the available evidence: " + \
                 " ".join([ev.text_span[:200] for ev in top_ev])

        claim = AnswerClaim(
            claim=answer,
            supporting_evidence=[ev.evidence_id for ev in top_ev],
            confidence=0.5
        )

        step = ReasoningStep(
            step=1,
            description="Synthesized from retrieved evidence",
            evidence_ids=[ev.evidence_id for ev in top_ev],
            confidence=0.5
        )

        return answer, [claim], [step]

    def run(self, *args, **kwargs) -> Any:
        """Run the reasoning agent."""
        return self.reason(*args, **kwargs)
