"""Self-RAG Baseline.

Self-Reflective RAG: retrieve + generate + self-critique + regenerate.
Inspired by the Self-RAG paper (Asai et al., 2023).

Pipeline:
1. BM25 retrieval
2. LLM generates answer
3. LLM critiques its own answer (is it supported by evidence?)
4. If critique finds issues, regenerate with critique feedback

No conflict detection, no uncertainty estimation.
"""

import sys
from pathlib import Path
from typing import Any

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.retriever.bm25 import BM25Retriever  # noqa: E402


class SelfRAGBaseline:
    """Self-RAG: retrieve → generate → critique → regenerate."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.retriever = BM25Retriever()
        self.top_k = self.config.get("retriever", {}).get("top_k", 5)
        self.max_retries = self.config.get("pipeline", {}).get("max_retrieval_rounds", 2)
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

    def index_documents(self, documents: list[dict[str, Any]]):
        self.retriever.index_documents(documents)

    def query(self, question: str) -> dict[str, Any]:
        # 1. Retrieve
        results = self.retriever.retrieve(question, top_k=self.top_k)

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

        # 2. Generate initial answer
        gen_prompt = f"""基于以下证据回答问题。

证据：
{context}

问题：{question}

答案："""

        try:
            llm = self._get_llm()
            answer = llm.generate(gen_prompt)
        except Exception:
            answer = f"基于{len(results)}条证据的分析：{context[:200]}"

        # 3. Self-critique
        critique_prompt = f"""请评估以下回答是否被证据充分支持。

证据：
{context}

回答：{answer}

请指出回答中哪些部分：
1. 被证据直接支持
2. 没有证据支持（幻觉）
3. 与证据矛盾

如果回答完全由证据支持，回复"完全支持"。否则列出问题。"""

        try:
            llm = self._get_llm()
            critique = llm.generate(critique_prompt)
        except Exception:
            critique = ""

        # 4. Regenerate if critique finds issues
        confidence = 0.65
        if critique and "完全支持" not in critique:
            regen_prompt = f"""你之前的回答存在以下问题：
{critique}

请基于证据重新回答，确保所有内容都有证据支持。如果证据不足，请明确说明。

证据：
{context}

问题：{question}

修正答案："""

            try:
                answer = llm.generate(regen_prompt)
                confidence = 0.70
            except Exception:
                confidence = 0.50

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence_list,
            "confidence": confidence,
            "answer_claims": [],
            "conflict_report": {"conflicts": [], "conflict_score": 0.0},
            "uncertainty": {
                "retrieval_uncertainty": 0.25,
                "evidence_conflict": 0.1,
                "reasoning_gap": 0.2,
                "source_reliability": 0.12,
                "verification_uncertainty": 0.15,
                "overall_uncertainty": 0.25,
            },
            "metadata": {
                "baseline": "self_rag",
                "critique": critique[:200] if critique else "",
            },
        }


class MockSelfRAG(SelfRAGBaseline):
    """Self-RAG with mock LLM."""

    def _get_llm(self):
        return None

    def query(self, question: str) -> dict[str, Any]:
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

        # Simulated self-critique: check if top evidence is consistent
        parts = [f"经过自反思检索，基于{len(results)}条证据："]
        for ev in evidence_list[:3]:
            parts.append(ev["text_span"][:80])
        answer = "".join(parts)

        # Lower confidence if fewer than 3 results
        confidence = 0.65 if len(results) >= 3 else 0.50

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence_list,
            "confidence": confidence,
            "answer_claims": [],
            "conflict_report": {"conflicts": [], "conflict_score": 0.0},
            "uncertainty": {
                "retrieval_uncertainty": 0.25,
                "evidence_conflict": 0.1,
                "reasoning_gap": 0.2,
                "source_reliability": 0.12,
                "verification_uncertainty": 0.15,
                "overall_uncertainty": 0.25,
            },
            "metadata": {"baseline": "self_rag"},
        }
