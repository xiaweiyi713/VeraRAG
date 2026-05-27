"""
Hybrid RAG Baseline.

Uses both sparse (BM25) and dense retrieval with Reciprocal Rank Fusion,
then generates with a single LLM call.

Stronger than Vanilla RAG due to hybrid retrieval, but still no
decomposition, conflict detection, or verification.
"""

import os
import sys
from pathlib import Path
from typing import Any

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from src.agents.base import LLMClient
from src.retriever.hybrid import HybridRetriever
from src.utils.data_structures import (
    AnswerClaim,
    Evidence,
    ReasoningStep,
    UncertaintyBreakdown,
    VeraRAGOutput,
)


class HybridRAG:
    """
    Hybrid RAG: BM25 + dense retrieval with RRF fusion, then generate.

    Better retrieval quality than Vanilla, but single-pass generation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.name = "HybridRAG"

        # LLM
        llm_config = self.config.get("llm", {})
        self.llm_client = LLMClient(
            provider=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o"),
            api_key=os.getenv(llm_config.get("api_key_env", "OPENAI_API_KEY")),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2000),
        )

        # Hybrid retriever (BM25 + dense)
        retriever_config = self.config.get("retriever", {})
        self.retriever = HybridRetriever(
            config=retriever_config,
            sparse_weight=retriever_config.get("sparse_weight", 0.3),
            dense_weight=retriever_config.get("dense_weight", 0.7),
        )

        self.top_k = retriever_config.get("top_k", 10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """Build hybrid index."""
        self.retriever.index_documents(documents)

    def query(self, question: str, **_kwargs) -> VeraRAGOutput:
        """Answer a single question."""
        import time
        import uuid

        start = time.time()

        # 1. Retrieve (hybrid)
        results = self.retriever.retrieve(question, top_k=self.top_k)

        # 2. Build prompt
        context_parts = []
        evidence_list: list[Evidence] = []
        for i, r in enumerate(results):
            context_parts.append(f"[{i + 1}] (score={r.score:.4f}) {r.content}")
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
                    confidence=0.55,
                )
            ],
            evidence=evidence_list,
            reasoning_chain=[
                ReasoningStep(
                    step=1,
                    description="Hybrid retrieval (BM25+dense) + single-pass generation",
                    evidence_ids=[e.evidence_id for e in evidence_list],
                    confidence=0.55,
                )
            ],
            conflict_report={"nodes": [], "edges": [], "conflict_score": 0.0},
            confidence=0.55,
            uncertainty=UncertaintyBreakdown(
                retrieval_uncertainty=0.2,
                evidence_conflict=0.0,
                reasoning_gap=0.25,
                source_reliability=0.2,
                verification_uncertainty=0.35,
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
