"""
Self-RAG Baseline.

Implements a Self-RAG style approach:
1. Retrieve documents
2. Generate answer
3. Self-critique: ask the LLM to judge whether the answer is supported
4. Optionally revise the answer

This adds a verification/review step on top of basic RAG.
"""

import os
import sys
from pathlib import Path
from typing import Any

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from src.agents.base import LLMClient
from src.retriever.bm25 import BM25Retriever
from src.utils.data_structures import (
    AnswerClaim,
    Evidence,
    ReasoningStep,
    UncertaintyBreakdown,
    VerificationReport,
    VerificationStatus,
    VeraRAGOutput,
)


class SelfRAG:
    """
    Self-RAG: retrieve, generate, self-critique, optionally revise.

    Adds a self-verification loop on top of Vanilla RAG.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.name = "SelfRAG"

        # LLM
        llm_config = self.config.get("llm", {})
        self.llm_client = LLMClient(
            provider=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o"),
            api_key=os.getenv(llm_config.get("api_key_env", "OPENAI_API_KEY")),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2000),
        )

        # Sparse retriever (BM25)
        self.retriever = BM25Retriever()

        self.top_k = self.config.get("retriever", {}).get("top_k", 5)
        self.max_revisions = self.config.get("pipeline", {}).get("max_revisions", 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """Build BM25 index."""
        self.retriever.index_documents(documents)

    def query(self, question: str, **_kwargs) -> VeraRAGOutput:
        """Answer a single question with self-critique."""
        import json as json_mod
        import time
        import uuid

        start = time.time()

        # 1. Retrieve
        results = self.retriever.retrieve(question, top_k=self.top_k)

        context_parts = []
        evidence_list: list[Evidence] = []
        for i, r in enumerate(results):
            context_parts.append(f"[{i + 1}] (score={r.score:.3f}) {r.content}")
            evidence_list.append(
                Evidence(
                    evidence_id=f"E{uuid.uuid4().hex[:8]}",
                    source="retrieved",
                    title=r.title,
                    text_span=r.content,
                    relevance_score=min(1.0, r.score),
                )
            )

        context_str = "\n".join(context_parts) if context_parts else "No relevant documents found."

        # 2. Generate initial answer
        gen_prompt = (
            "Answer the following question based ONLY on the provided context.\n"
            "If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        answer = self.llm_client.generate(gen_prompt)

        # 3. Self-critique loop
        verification_status = VerificationStatus.NOT_ENOUGH_INFO
        critique_issues: list[dict[str, Any]] = []

        for _rev in range(self.max_revisions):
            critique_prompt = (
                "You are a critical reviewer. Given the question, context, and answer below, "
                "evaluate whether the answer is fully supported by the context.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context_str}\n\n"
                f"Answer: {answer}\n\n"
                "Respond in JSON with:\n"
                '{"verdict": "supported" | "refuted" | "not_enough_info",\n'
                ' "issues": ["issue1", "issue2", ...],\n'
                ' "confidence": 0.0-1.0}\n\n'
                "JSON:"
            )

            try:
                critique_raw = self.llm_client.generate(
                    critique_prompt, response_format="json"
                )
                critique = json_mod.loads(critique_raw)
            except (json_mod.JSONDecodeError, Exception):
                # If critique fails, keep current answer
                break

            verdict = critique.get("verdict", "not_enough_info")
            if verdict == "supported":
                verification_status = VerificationStatus.SUPPORTED
                break
            elif verdict == "refuted":
                verification_status = VerificationStatus.REFUTED

            critique_issues = [
                {"issue": iss, "severity": "medium"}
                for iss in critique.get("issues", [])
            ]

            # Revise answer
            if critique_issues:
                revise_prompt = (
                    "Revise the following answer to address the identified issues.\n"
                    "Use ONLY information from the context.\n\n"
                    f"Question: {question}\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Original Answer: {answer}\n\n"
                    f"Issues to fix: {critique.get('issues', [])}\n\n"
                    "Revised Answer:"
                )
                answer = self.llm_client.generate(revise_prompt)

        # 4. Package output
        elapsed = time.time() - start

        verification_report = VerificationReport(
            claim_verifications=[
                {
                    "claim": answer,
                    "status": verification_status.value,
                    "issues": critique_issues,
                }
            ],
            overall_status=verification_status,
            issues=critique_issues,
        )

        confidence = 0.6
        if verification_status == VerificationStatus.SUPPORTED:
            confidence = 0.75
        elif verification_status == VerificationStatus.REFUTED:
            confidence = 0.3

        return VeraRAGOutput(
            question=question,
            answer=answer,
            answer_claims=[
                AnswerClaim(
                    claim=answer,
                    supporting_evidence=[e.evidence_id for e in evidence_list],
                    confidence=confidence,
                    verification_status=verification_status,
                )
            ],
            evidence=evidence_list,
            reasoning_chain=[
                ReasoningStep(
                    step=1,
                    description="Retrieve + generate + self-critique",
                    evidence_ids=[e.evidence_id for e in evidence_list],
                    confidence=confidence,
                )
            ],
            conflict_report={"nodes": [], "edges": [], "conflict_score": 0.0},
            verification_report=verification_report,
            confidence=confidence,
            uncertainty=UncertaintyBreakdown(
                retrieval_uncertainty=0.2,
                evidence_conflict=0.0,
                reasoning_gap=0.2,
                source_reliability=0.2,
                verification_uncertainty=0.2 if verification_status == VerificationStatus.SUPPORTED else 0.4,
            ),
            metadata={
                "baseline": self.name,
                "num_evidence": len(evidence_list),
                "elapsed_time": elapsed,
                "retrieval_rounds": 1,
                "verification_status": verification_status.value,
            },
        )

    def batch_query(self, questions: list[str], **kwargs) -> list[VeraRAGOutput]:
        """Answer multiple questions."""
        return [self.query(q, **kwargs) for q in questions]
