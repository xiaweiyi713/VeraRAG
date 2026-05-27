"""
Vanilla RAG Baseline.

A simple retrieve-then-generate pipeline:
1. Retrieve top-k documents via BM25
2. Concatenate retrieved documents as context
3. Generate answer with a single LLM call

This is the most basic RAG baseline with no advanced features.
"""

import os
import sys
from pathlib import Path
from typing import Any

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from src.agents.base import LLMClient
from src.retriever.bm25 import BM25Retriever
from src.evaluation.answer_metrics import AnswerMetrics
from src.utils.data_structures import (
    AnswerClaim,
    Evidence,
    ReasoningStep,
    UncertaintyBreakdown,
    VeraRAGOutput,
)


class VanillaRAG:
    """
    Vanilla RAG: retrieve with BM25, then generate with LLM.

    No decomposition, no conflict detection, no verification.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.name = "VanillaRAG"

        # LLM
        llm_config = self.config.get("llm", {})
        self.llm_client = LLMClient(
            provider=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o"),
            api_key=os.getenv(llm_config.get("api_key_env", "OPENAI_API_KEY")),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2000),
        )

        # Sparse retriever only (BM25)
        self.retriever = BM25Retriever()

        self.top_k = self.config.get("retriever", {}).get("top_k", 5)

    # ------------------------------------------------------------------
    # Public API (same surface as VeraRAG for drop-in benchmarking)
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """Build BM25 index."""
        self.retriever.index_documents(documents)

    def query(self, question: str, **_kwargs) -> VeraRAGOutput:
        """Answer a single question."""
        import time
        import uuid

        start = time.time()

        # 1. Retrieve
        results = self.retriever.retrieve(question, top_k=self.top_k)

        # 2. Build prompt
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

        prompt = (
            "Answer the following question based ONLY on the provided context.\n"
            "If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # 3. Generate
        answer = self.llm_client.generate(prompt)

        # 4. Package output
        elapsed = time.time() - start
        return VeraRAGOutput(
            question=question,
            answer=answer,
            answer_claims=[
                AnswerClaim(
                    claim=answer,
                    supporting_evidence=[e.evidence_id for e in evidence_list],
                    confidence=0.5,
                )
            ],
            evidence=evidence_list,
            reasoning_chain=[
                ReasoningStep(
                    step=1,
                    description="Single-step retrieval + generation",
                    evidence_ids=[e.evidence_id for e in evidence_list],
                    confidence=0.5,
                )
            ],
            conflict_report={"nodes": [], "edges": [], "conflict_score": 0.0},
            confidence=0.5,
            uncertainty=UncertaintyBreakdown(
                retrieval_uncertainty=0.3,
                evidence_conflict=0.0,
                reasoning_gap=0.3,
                source_reliability=0.2,
                verification_uncertainty=0.4,
            ),
            metadata={
                "baseline": self.name,
                "num_evidence": len(evidence_list),
                "elapsed_time": elapsed,
                "retrieval_rounds": 1,
            },
        )

    def batch_query(self, questions: list[str], **kwargs) -> list[VeraRAGOutput]:
        """Answer multiple questions."""
        return [self.query(q, **kwargs) for q in questions]
