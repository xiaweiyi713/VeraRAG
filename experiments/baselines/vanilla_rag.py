"""Vanilla RAG Baseline.

Simple retrieve-then-generate pipeline:
1. BM25 retrieval (single round)
2. LLM generates answer from top-k evidence

No conflict detection, no uncertainty estimation, no verification, no repair.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.retriever.bm25 import BM25Retriever


class VanillaRAG:
    """Vanilla RAG: single-round BM25 retrieval + LLM generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.retriever = BM25Retriever()
        self.top_k = self.config.get("retriever", {}).get("top_k", 5)
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            from src.agents.base import LLMClient
            llm_cfg = self.config.get("llm", {})
            self.llm = LLMClient(
                provider=llm_cfg.get("provider", "openai"),
                model=llm_cfg.get("model", "gpt-4o"),
                api_key=llm_cfg.get("api_key", ""),
                base_url=llm_cfg.get("base_url", ""),
            )
        return self.llm

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for retrieval."""
        self.retriever.index_documents(documents)

    def query(self, question: str) -> Dict[str, Any]:
        """Run vanilla RAG: retrieve + generate."""
        # 1. Retrieve
        results = self.retriever.retrieve(question, top_k=self.top_k)

        # 2. Build context
        evidence_list = []
        context_parts = []
        for i, r in enumerate(results):
            ev = {
                "evidence_id": f"E{i+1}",
                "source": r.metadata.get("source", "unknown"),
                "title": r.title,
                "text_span": r.content[:500],
                "combined_score": round(r.score, 3),
            }
            evidence_list.append(ev)
            context_parts.append(f"[{i+1}] {r.title}: {r.content[:300]}")

        context = "\n\n".join(context_parts) if context_parts else "无相关证据。"

        # 3. Generate answer
        prompt = f"""基于以下证据回答问题。直接给出答案，不需要引用证据编号。

证据：
{context}

问题：{question}

答案："""

        try:
            llm = self._get_llm()
            answer = llm.generate(prompt)
        except Exception:
            answer = f"基于{len(results)}条证据的分析：{context[:200]}"

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence_list,
            "confidence": 0.5,
            "answer_claims": [],
            "conflict_report": {"conflicts": [], "conflict_score": 0.0},
            "uncertainty": {
                "retrieval_uncertainty": 0.3,
                "evidence_conflict": 0.1,
                "reasoning_gap": 0.3,
                "source_reliability": 0.15,
                "verification_uncertainty": 0.2,
                "overall_uncertainty": 0.3,
            },
            "metadata": {"baseline": "vanilla_rag"},
        }


class MockVanillaRAG(VanillaRAG):
    """Vanilla RAG with mock LLM (for testing without API keys)."""

    def _get_llm(self):
        return None

    def query(self, question: str) -> Dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=self.top_k)
        evidence_list = []
        for i, r in enumerate(results):
            evidence_list.append({
                "evidence_id": f"E{i+1}",
                "source": r.metadata.get("source", "unknown"),
                "title": r.title,
                "text_span": r.content[:500],
                "combined_score": round(r.score, 3),
            })

        parts = [f"根据{len(results)}条证据检索分析："]
        for ev in evidence_list[:3]:
            parts.append(ev["text_span"][:100])
        answer = "".join(parts)

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence_list,
            "confidence": 0.5,
            "answer_claims": [],
            "conflict_report": {"conflicts": [], "conflict_score": 0.0},
            "uncertainty": {
                "retrieval_uncertainty": 0.3,
                "evidence_conflict": 0.1,
                "reasoning_gap": 0.3,
                "source_reliability": 0.15,
                "verification_uncertainty": 0.2,
                "overall_uncertainty": 0.3,
            },
            "metadata": {"baseline": "vanilla_rag"},
        }
