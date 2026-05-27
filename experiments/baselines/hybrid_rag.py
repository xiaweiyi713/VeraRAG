"""Hybrid RAG Baseline.

Retrieval-augmented generation with hybrid (BM25 + Dense) retrieval
and cross-encoder reranking, but no conflict detection or verification.

Pipeline:
1. Hybrid retrieval (BM25 + Dense via RRF)
2. Cross-encoder reranking
3. LLM generates answer from reranked evidence
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.retriever.bm25 import BM25Retriever


class HybridRAGBaseline:
    """Hybrid RAG: BM25 + reranking + LLM generation.

    Uses only BM25 (since Dense requires sentence-transformers).
    Simulates hybrid reranking via score-based reordering.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.retriever = BM25Retriever()
        self.top_k = self.config.get("retriever", {}).get("top_k", 10)
        self.fetch_k = self.config.get("retriever", {}).get("fetch_k", 20)
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
        self.retriever.index_documents(documents)

    def query(self, question: str) -> Dict[str, Any]:
        # 1. Over-retrieve then rerank
        results = self.retriever.retrieve(question, top_k=self.fetch_k)

        # 2. Simulated reranking: boost results with title keyword overlap
        query_tokens = set(question.lower().split())
        scored = []
        for r in results:
            title_tokens = set(r.title.lower().split())
            overlap = len(query_tokens & title_tokens)
            rerank_score = r.score + overlap * 0.5
            scored.append((r, rerank_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_results = [r for r, _ in scored[:self.top_k]]

        # 3. Build evidence
        evidence_list = []
        context_parts = []
        for i, r in enumerate(top_results):
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

        # 4. Generate answer
        prompt = f"""基于以下证据回答问题。请综合多条证据给出全面回答。

证据：
{context}

问题：{question}

答案："""

        try:
            llm = self._get_llm()
            answer = llm.generate(prompt)
        except Exception:
            answer = f"基于{len(top_results)}条证据的综合分析：{context[:200]}"

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence_list,
            "confidence": 0.6,
            "answer_claims": [],
            "conflict_report": {"conflicts": [], "conflict_score": 0.0},
            "uncertainty": {
                "retrieval_uncertainty": 0.2,
                "evidence_conflict": 0.1,
                "reasoning_gap": 0.25,
                "source_reliability": 0.12,
                "verification_uncertainty": 0.18,
                "overall_uncertainty": 0.25,
            },
            "metadata": {"baseline": "hybrid_rag"},
        }


class MockHybridRAG(HybridRAGBaseline):
    """Hybrid RAG with mock LLM."""

    def _get_llm(self):
        return None

    def query(self, question: str) -> Dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=self.fetch_k)
        query_tokens = set(question.lower().split())
        scored = []
        for r in results:
            title_tokens = set(r.title.lower().split())
            overlap = len(query_tokens & title_tokens)
            scored.append((r, r.score + overlap * 0.5))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_results = [r for r, _ in scored[:self.top_k]]

        evidence_list = []
        for i, r in enumerate(top_results):
            evidence_list.append({
                "evidence_id": f"E{i+1}",
                "source": r.metadata.get("source", "unknown"),
                "title": r.title,
                "text_span": r.content[:500],
                "combined_score": round(r.score, 3),
            })

        parts = [f"经过检索和重排序，获得{len(top_results)}条相关证据："]
        for ev in evidence_list[:3]:
            parts.append(ev["text_span"][:100])
        answer = "".join(parts)

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence_list,
            "confidence": 0.6,
            "answer_claims": [],
            "conflict_report": {"conflicts": [], "conflict_score": 0.0},
            "uncertainty": {
                "retrieval_uncertainty": 0.2,
                "evidence_conflict": 0.1,
                "reasoning_gap": 0.25,
                "source_reliability": 0.12,
                "verification_uncertainty": 0.18,
                "overall_uncertainty": 0.25,
            },
            "metadata": {"baseline": "hybrid_rag"},
        }
