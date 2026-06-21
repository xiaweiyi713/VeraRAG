"""Reasoning Agent for VeraRAG."""

import json
import re
from itertools import pairwise
from typing import Any

from ..utils.data_structures import (
    AnswerClaim,
    Evidence,
    EvidenceConflictGraph,
    ReasoningStep,
    SubQuestion,
)
from .base import BaseAgent

_CITATION_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")


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
        reasoning_config = self.config.get("reasoning", {})
        enforce_answer_citations = reasoning_config.get("enforce_answer_citations", True)
        if not isinstance(enforce_answer_citations, bool):
            raise ValueError("reasoning.enforce_answer_citations must be a boolean")
        self.enforce_answer_citations = enforce_answer_citations
        claim_slot_selection = reasoning_config.get("claim_slot_selection_enabled", False)
        if not isinstance(claim_slot_selection, bool):
            raise ValueError("reasoning.claim_slot_selection_enabled must be a boolean")
        self.claim_slot_selection_enabled = claim_slot_selection
        self.claim_slot_max_evidence = self._positive_config_int(
            reasoning_config.get("claim_slot_max_evidence", 6),
            "reasoning.claim_slot_max_evidence",
        )
        self.system_prompt = """你是面向复杂知识任务的推理专家，目标是基于给定证据生成准确、有据可依的回答。
核心准则：宁可拒答，也不臆造。证据不足时必须明确拒答；问题前提与证据矛盾时必须纠正前提；证据相互冲突时必须如实标注冲突。
所有回答用简体中文。只输出合法 JSON，不要输出其它内容。"""

    @staticmethod
    def _positive_config_int(value: Any, field: str) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive integer") from exc
        if numeric < 1:
            raise ValueError(f"{field} must be a positive integer")
        return numeric

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
        evidence_context = self._prepare_evidence_context(
            evidence,
            question=question,
            subquestions=subquestions,
            conflict_graph=conflict_graph,
        )

        # Prepare conflict context
        conflict_context = self._prepare_conflict_context(conflict_graph, evidence)

        prompt = f"""请基于下列证据回答问题。用简体中文作答。

问题："{question}"

子问题：
{json.dumps([sq.to_dict() for sq in subquestions], ensure_ascii=False, indent=2)}

证据：
{evidence_context}

证据中检测到的冲突：
{conflict_context}

推理计划：
{json.dumps(reasoning_plan, ensure_ascii=False, indent=2)}

【第一步：先判断证据与问题的关系，据此决定回答方式（这一步至关重要）】
总原则：**默认应当作答（情形 D）**。情形 A/B/C 是需要特殊处理的少数情况，只有明确符合其条件时才偏离默认。尤其对开放式、多部分的问题（如"……现状如何/有哪些进展/经历了哪些节点/中美各有何成就"），即使证据不完整，也要就能查到的部分作答，而不是拒答。
- 情形 A —— 证据中确实查无相关信息（拒答是少数情况，需谨慎）：**仅当**证据中几乎没有与问题相关的内容、或问题问的是一个证据从未提及的具体数据/事实时，才以"根据现有证据无法回答此问题"或"信息不足，无法确定"开头拒答，并说明缺少什么。**关键限制：如果证据只是"部分覆盖"问题——例如多部分问题里你能答上其中一部分、或证据给了相关背景/趋势但缺确切数字——绝不要拒答，而应就证据支持的部分尽量作答（走情形 D），并简要注明哪一部分缺乏证据。**绝不可编造、也不可用无关证据强行作答。此时 behavior 填 "abstain"。
- 情形 B —— 断言型问题且前提不成立：若问题是在请你确认某个断言（例如以"对吗/是吗/是不是"结尾，或陈述了一个具体说法/数据），而证据并不支持该断言——无论是直接反驳它，还是证据中根本查不到相关依据——都必须**以"该说法不准确"或"这个前提有误"开头**，再依据证据说明（用"实际上……"陈述；若证据中确实查无此说法，则说明"该说法缺乏证据支持"）。此时 behavior 填 "correct_premise"。（即：断言型问题不要轻易拒答，应纠正或指出其缺乏依据。）
- 情形 C —— 证据相互冲突：若多条证据（尤其新旧来源、不同机构）对同一问题给出相互矛盾/不一致的说法，则 answer 必须明确写出"证据存在冲突/不一致"，分别呈现不同来源的说法与各自出处；即使你能判断哪一方更可信，也要**先点明这种不一致**，再说明以哪条为准。此时 behavior 填 "conflict_note"。
- 情形 D —— 正常作答：证据充分且无上述问题时，正常综合作答，并为每个论断引用具体证据编号。此时 behavior 填 "answer"。

【第二步：输出 JSON】（answer 与 claim 文本均用简体中文）
{{
    "behavior": "answer | abstain | correct_premise | conflict_note",
    "answer": "你的中文回答（按上面的情形给出对应措辞）",
    "answer_claims": [
        {{
            "claim": "回答中的具体论断",
            "supporting_evidence": ["D001_c0"],
            "conflicting_evidence": [],
            "confidence": 0.8,
            "claim_type": "factual",
            "verifiable": true,
            "support_type": "direct"
        }}
    ],
    "reasoning_chain": [
        {{"step": 1, "description": "首先，我确认了……", "evidence_ids": ["D001_c0"], "confidence": 0.9}}
    ]
}}

要求：
1. 仅依据上面提供的证据作答，不要使用证据之外的知识
2. 拒答(A)、纠正前提(B)、标注冲突(C) 三种情形必须严格按上述措辞明确表达，不要含糊带过；只要“证据中检测到的冲突”不是 No significant conflicts detected，answer 必须包含“冲突”“不一致”“错误”或“不准确”等明确字样
3. supporting_evidence/evidence_ids 使用证据条目方括号内的编号（如 D001_c0）
4. confidence 依据证据强度赋值；拒答时各论断 confidence 应较低
5. 每个 answer_claim 标注：claim_type（factual 直接来自证据 / inference 推断得出 / prediction 前瞻预测）、verifiable（能否对照证据核验）、support_type（direct 证据明确陈述 / indirect 需推断 / none 无支撑）
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

            answer, claims, reasoning = self._ensure_conflict_acknowledged(
                answer,
                claims,
                reasoning,
                conflict_graph,
                evidence,
            )
            answer = self._ensure_answer_citations(answer, claims, evidence)
            return answer, claims, reasoning

        except (json.JSONDecodeError, KeyError):
            # Fallback: generate simple answer
            return self._fallback_answer(question, evidence)

    def _prepare_evidence_context(
        self,
        evidence: list[Evidence],
        *,
        question: str = "",
        subquestions: list[SubQuestion] | None = None,
        conflict_graph: EvidenceConflictGraph | None = None,
    ) -> str:
        """Prepare evidence context for the prompt."""
        if not evidence:
            return "No evidence available."

        if self.claim_slot_selection_enabled:
            evidence = self._select_claim_slot_evidence(
                evidence,
                question=question,
                subquestions=subquestions or [],
                conflict_graph=conflict_graph or EvidenceConflictGraph(),
            )

        contexts = []
        for ev in evidence[:20]:  # Limit to prevent token overflow
            contexts.append(f"[{ev.evidence_id}] {ev.title}\n{ev.text_span}\n")

        return "\n".join(contexts)

    def _select_claim_slot_evidence(
        self,
        evidence: list[Evidence],
        *,
        question: str,
        subquestions: list[SubQuestion],
        conflict_graph: EvidenceConflictGraph,
    ) -> list[Evidence]:
        """Compress answer evidence to the strongest claim-level slots."""
        if len(evidence) <= self.claim_slot_max_evidence:
            return evidence

        conflict_ids = self._conflict_evidence_ids(evidence, conflict_graph)
        query_terms = self._slot_terms(
            " ".join([question, *(sq.question for sq in subquestions)])
        )
        scored = [
            (
                self._claim_slot_score(
                    item,
                    index=index,
                    total=len(evidence),
                    query_terms=query_terms,
                    conflict_ids=conflict_ids,
                ),
                index,
                item,
            )
            for index, item in enumerate(evidence)
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        selected = [item for _score, _index, item in scored[: self.claim_slot_max_evidence]]

        selected_ids = {item.evidence_id for item in selected}
        for item in evidence:
            if item.evidence_id in conflict_ids and item.evidence_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.evidence_id)

        selected.sort(key=lambda item: next(
            index for index, original in enumerate(evidence)
            if original.evidence_id == item.evidence_id
        ))
        return selected

    def _claim_slot_score(
        self,
        evidence: Evidence,
        *,
        index: int,
        total: int,
        query_terms: set[str],
        conflict_ids: set[str],
    ) -> float:
        quality = evidence.combined_score * 0.45 + evidence.relevance_score * 0.25
        lexical = self._slot_overlap_score(query_terms, f"{evidence.title} {evidence.text_span}")
        rank_bonus = 0.20 * (1.0 - index / max(total, 1))
        conflict_bonus = 0.18 if evidence.evidence_id in conflict_ids else 0.0
        claim_bonus = min(0.08, len(evidence.claims) * 0.02)
        return quality + lexical * 0.25 + rank_bonus + conflict_bonus + claim_bonus

    @staticmethod
    def _slot_terms(text: str) -> set[str]:
        lowered = text.lower()
        terms = set(re.findall(r"[a-z0-9_]+", lowered))
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
        terms.update(
            "".join(pair)
            for pair in pairwise(cjk_chars)
        )
        return {term for term in terms if len(term) >= 2}

    def _slot_overlap_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        evidence_terms = self._slot_terms(text)
        if not evidence_terms:
            return 0.0
        return len(query_terms & evidence_terms) / len(query_terms)

    def _conflict_evidence_ids(
        self,
        evidence: list[Evidence],
        conflict_graph: EvidenceConflictGraph,
    ) -> set[str]:
        if not conflict_graph.get_conflicts():
            return set()
        claim_lookup = self._claim_lookup(evidence, conflict_graph)
        evidence_ids = {item.evidence_id for item in evidence}
        selected = set()
        for conflict in conflict_graph.get_conflicts():
            for claim_id in (conflict.source_id, conflict.target_id):
                mapped_id = claim_lookup.get(claim_id, {}).get("evidence_id", claim_id)
                if mapped_id in evidence_ids:
                    selected.add(mapped_id)
        return selected

    def _prepare_conflict_context(
        self,
        conflict_graph: EvidenceConflictGraph,
        evidence: list[Evidence],
    ) -> str:
        """Prepare conflict context for the prompt."""
        conflicts = conflict_graph.get_conflicts()

        if not conflicts:
            return "No significant conflicts detected."

        claim_lookup = self._claim_lookup(evidence, conflict_graph)
        contexts = []
        for c in conflicts[:10]:
            contexts.append(
                f"- {c.conflict_type.value} (confidence: {c.confidence:.2f}): "
                f"{self._format_conflict_side(c.source_id, claim_lookup)} vs "
                f"{self._format_conflict_side(c.target_id, claim_lookup)}"
            )

        return "\n".join(contexts)

    @staticmethod
    def _claim_lookup(
        evidence: list[Evidence],
        conflict_graph: EvidenceConflictGraph,
    ) -> dict[str, dict[str, str]]:
        lookup = {}
        for ev in evidence:
            for claim in ev.claims:
                lookup[claim.claim_id] = {
                    "evidence_id": ev.evidence_id,
                    "title": ev.title,
                    "claim": claim.claim,
                }
        for node_id, node in conflict_graph.nodes.items():
            if node_id not in lookup:
                lookup[node_id] = {
                    "evidence_id": node.evidence_ids[0] if node.evidence_ids else node_id,
                    "title": "",
                    "claim": node.content,
                }
        return lookup

    @staticmethod
    def _format_conflict_side(claim_id: str, claim_lookup: dict[str, dict[str, str]]) -> str:
        item = claim_lookup.get(claim_id)
        if not item:
            return claim_id
        title = f" {item['title']}" if item["title"] else ""
        claim = item["claim"][:160]
        return f"[{item['evidence_id']}]{title}: {claim}"

    def _ensure_conflict_acknowledged(
        self,
        answer: str,
        claims: list[AnswerClaim],
        reasoning: list[ReasoningStep],
        conflict_graph: EvidenceConflictGraph,
        evidence: list[Evidence],
    ) -> tuple[str, list[AnswerClaim], list[ReasoningStep]]:
        conflicts = conflict_graph.get_conflicts()
        if not conflicts:
            return answer, claims, reasoning

        acknowledgement_keywords = ("冲突", "矛盾", "不一致", "争议", "错误", "不准确", "不实")
        if any(keyword in answer for keyword in acknowledgement_keywords):
            return answer, claims, reasoning

        claim_lookup = self._claim_lookup(evidence, conflict_graph)
        first = conflicts[0]
        source = claim_lookup.get(first.source_id, {})
        target = claim_lookup.get(first.target_id, {})
        source_ev = source.get("evidence_id", first.source_id)
        target_ev = target.get("evidence_id", first.target_id)
        source_claim = source.get("claim", first.source_id)[:80]
        target_claim = target.get("claim", first.target_id)[:80]
        note = (
            f"证据存在冲突：{source_ev} 提到“{source_claim}”，"
            f"而 {target_ev} 提到“{target_claim}”。"
        )
        answer = f"{note}综合判断，{answer}"

        conflict_evidence = list(dict.fromkeys([source_ev, target_ev]))
        for claim in claims:
            claim.conflicting_evidence = list(dict.fromkeys([
                *claim.conflicting_evidence,
                *conflict_evidence,
            ]))
        reasoning.insert(
            0,
            ReasoningStep(
                step=1,
                description="先检查证据冲突并在回答中显式标注不一致信息。",
                evidence_ids=conflict_evidence,
                confidence=first.confidence,
            ),
        )
        for index, step in enumerate(reasoning, start=1):
            step.step = index
        return answer, claims, reasoning

    def _ensure_answer_citations(
        self,
        answer: str,
        claims: list[AnswerClaim],
        evidence: list[Evidence],
    ) -> str:
        """Attach missing bracketed evidence IDs from supported claims."""
        if not self.enforce_answer_citations or not answer or not claims:
            return answer

        available_ids = {item.evidence_id for item in evidence}
        supporting_ids: list[str] = []
        for claim in claims:
            for evidence_id in claim.supporting_evidence:
                if evidence_id not in available_ids or evidence_id in supporting_ids:
                    continue
                supporting_ids.append(evidence_id)

        if not supporting_ids:
            return answer

        existing_ids = set(_CITATION_PATTERN.findall(answer))
        missing_ids = [
            evidence_id for evidence_id in supporting_ids
            if evidence_id not in existing_ids
        ]
        if not missing_ids:
            return answer

        citation_footer = "引用证据：" + " ".join(f"[{evidence_id}]" for evidence_id in missing_ids)
        if answer.rstrip().endswith(citation_footer):
            return answer
        return f"{answer.rstrip()}\n{citation_footer}"

    def _fallback_answer(
        self,
        question: str,
        evidence: list[Evidence]
    ) -> tuple[str, list[AnswerClaim], list[ReasoningStep]]:
        """Generate a fallback answer if LLM fails."""
        if not evidence:
            return (
                "根据现有证据，信息不足，无法回答此问题。",
                [],
                []
            )

        # Simple synthesis from top evidence
        top_ev = evidence[:3]
        answer = "根据现有证据：" + \
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
