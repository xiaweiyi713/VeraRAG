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
        self.system_prompt = """你是面向复杂知识任务的推理专家，目标是基于给定证据生成准确、有据可依的回答。
核心准则：宁可拒答，也不臆造。证据不足时必须明确拒答；问题前提与证据矛盾时必须纠正前提；证据相互冲突时必须如实标注冲突。
所有回答用简体中文。只输出合法 JSON，不要输出其它内容。"""

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
2. 拒答(A)、纠正前提(B)、标注冲突(C) 三种情形必须严格按上述措辞明确表达，不要含糊带过
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
