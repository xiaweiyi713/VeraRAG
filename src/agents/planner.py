"""Decomposition and Planning Agent for VeraRAG."""

import json
from typing import Any

from ..utils.data_structures import SubQuestion, TaskAnalysis, UncertaintyBreakdown
from .base import BaseAgent


class DecompositionPlanner(BaseAgent):
    """
    Decomposes complex questions into sub-questions and creates a reasoning plan.

    For complex multi-hop questions, breaks them down into:
    1. Sub-questions that can be answered independently
    2. Dependency relationships between sub-questions
    3. Evidence types required for each sub-question
    4. Whether counter-evidence should be sought
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any | None = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert at breaking down complex questions into clear, answerable sub-questions.
Your goal is to create a structured plan for answering complex questions.
Output ONLY valid JSON, no other text."""

    def decompose(
        self,
        question: str,
        task_analysis: TaskAnalysis,
        max_subquestions: int = 10
    ) -> list[SubQuestion]:
        """
        Decompose a complex question into sub-questions.

        Args:
            question: The user's question
            task_analysis: Analysis of the question
            max_subquestions: Maximum number of sub-questions to generate

        Returns:
            List of SubQuestion objects
        """
        # For simple questions, no decomposition needed
        if task_analysis.complexity.value == "low" or task_analysis.estimated_hops == 1:
            return [
                SubQuestion(
                    id="sq0",
                    question=question,
                    required_evidence_type="general",
                    dependency_ids=[],
                    requires_counter_evidence=task_analysis.requires_conflict_check
                )
            ]

        # Use LLM for complex decomposition
        return self._llm_decompose(question, task_analysis, max_subquestions)

    def _llm_decompose(
        self,
        question: str,
        task_analysis: TaskAnalysis,
        max_subquestions: int
    ) -> list[SubQuestion]:
        """Use LLM to decompose the question."""
        prompt = f"""Break down the following complex question into sub-questions.

Original Question: "{question}"

Context:
- Task Type: {task_analysis.task_type.value}
- Complexity: {task_analysis.complexity.value}
- Estimated Hops: {task_analysis.estimated_hops}
- Keywords: {', '.join(task_analysis.keywords)}

Create a JSON array of sub-questions. Each sub-question should:
1. Be specific and answerable
2. Identify what type of evidence is needed
3. Note dependencies on other sub-questions (if any)
4. Indicate if counter-evidence should be sought

Output format:
{{
    "subquestions": [
        {{
            "question": "...",
            "required_evidence_type": "empirical_study|benchmark_or_study|statistical_data|expert_opinion|general",
            "dependency_ids": ["id_of_dependency"],
            "requires_counter_evidence": true|false
        }}
    ],
    "reasoning_plan": ["step1", "step2", "step3"]
}}

Limit to {max_subquestions} sub-questions.
"""

        response = self._call_llm(
            prompt,
            system_prompt=self.system_prompt,
            response_format="json"
        )

        try:
            data = json.loads(response)
            subquestions = []

            for i, sq_data in enumerate(data.get("subquestions", [])):
                sq = SubQuestion(
                    id=f"sq{i}",
                    question=sq_data.get("question", ""),
                    required_evidence_type=sq_data.get("required_evidence_type", "general"),
                    dependency_ids=sq_data.get("dependency_ids", []),
                    requires_counter_evidence=sq_data.get("requires_counter_evidence", False),
                    status="pending"
                )
                subquestions.append(sq)

            return subquestions

        except (json.JSONDecodeError, KeyError):
            # Fallback: create simple decomposition
            return self._fallback_decompose(question, task_analysis)

    def _fallback_decompose(
        self,
        question: str,
        task_analysis: TaskAnalysis
    ) -> list[SubQuestion]:
        """Fallback decomposition strategy."""
        # Simple approach: create sub-questions for each keyword pair
        keywords = task_analysis.keywords[:5]
        subquestions = []

        for i, keyword in enumerate(keywords):
            sq = SubQuestion(
                id=f"sq{i}",
                question=f"What information is available about '{keyword}' in the context of this question?",
                required_evidence_type="general",
                dependency_ids=[],
                requires_counter_evidence=task_analysis.requires_conflict_check
            )
            subquestions.append(sq)

        # Add synthesis question
        subquestions.append(SubQuestion(
            id=f"sq{len(subquestions)}",
            question=question,
            required_evidence_type="synthesis",
            dependency_ids=[sq.id for sq in subquestions],
            requires_counter_evidence=False
        ))

        return subquestions

    def get_reasoning_plan(
        self,
        question: str,
        subquestions: list[SubQuestion]
    ) -> list[str]:
        """
        Generate a reasoning plan for answering the question.

        Args:
            question: The original question
            subquestions: List of sub-questions

        Returns:
            List of reasoning steps
        """
        prompt = f"""Given the original question and its sub-questions, create a reasoning plan.

Original Question: "{question}"

Sub-questions:
{json.dumps([sq.to_dict() for sq in subquestions], indent=2)}

Output a JSON array of reasoning steps:
{{
    "reasoning_plan": [
        "First, establish...",
        "Then, compare...",
        "Finally, synthesize..."
    ]
}}
"""

        try:
            response = self._call_llm(
                prompt,
                system_prompt=self.system_prompt,
                response_format="json"
            )
            data = json.loads(response)
            return list(data.get("reasoning_plan", []))
        except Exception:
            # Fallback reasoning plan
            return [
                "Gather evidence for each sub-question",
                "Identify relationships and dependencies",
                "Resolve any conflicts in the evidence",
                "Synthesize findings into a coherent answer",
                "Verify that the answer is supported by evidence"
            ]

    def refine_plan(
        self,
        subquestions: list[SubQuestion],
        uncertainty_report: dict[str, Any]
    ) -> list[SubQuestion]:
        """
        Refine the plan based on uncertainty feedback.

        Args:
            subquestions: Current sub-questions
            uncertainty_report: Report with keys: "uncertainty" (UncertaintyBreakdown),
                                optional "conflicts" (list), optional "max_subquestions" (int)

        Returns:
            Refined list of sub-questions
        """
        uncertainty: UncertaintyBreakdown = uncertainty_report.get(
            "uncertainty", UncertaintyBreakdown()
        )
        conflicts = uncertainty_report.get("conflicts", [])
        max_sq = uncertainty_report.get("max_subquestions", 10)

        refined = list(subquestions)

        # Mark high-coverage questions as resolved
        for sq in refined:
            if sq.coverage_score >= 0.8:
                sq.status = "resolved"

        # Handle high retrieval uncertainty
        if uncertainty.retrieval_uncertainty > 0.5:
            for sq in refined:
                if sq.coverage_score < 0.5 and sq.status != "resolved":
                    sq.status = "in_progress"

        # Handle high conflict uncertainty -> add resolution sub-questions
        if uncertainty.evidence_conflict > 0.5 and conflicts and len(refined) < max_sq:
            conflict_types = set()
            for c in conflicts:
                ctype = c.get("type", "general") if isinstance(c, dict) else "general"
                conflict_types.add(ctype)

            for ctype in conflict_types:
                if len(refined) >= max_sq:
                    break
                resolve_sq = SubQuestion(
                    id=f"sq_resolve_{ctype}",
                    question=f"Resolve conflicting evidence about {ctype} relationships",
                    required_evidence_type="general",
                    dependency_ids=[sq.id for sq in refined if sq.status != "resolved"],
                    requires_counter_evidence=False,
                    status="pending",
                    coverage_score=0.0
                )
                refined.append(resolve_sq)

        # Handle high source reliability uncertainty
        if uncertainty.source_reliability > 0.5 and len(refined) < max_sq:
            source_sq = SubQuestion(
                id="sq_source_verify",
                question="Find high-credibility sources to verify key claims",
                required_evidence_type="empirical_study",
                dependency_ids=[sq.id for sq in refined[:3]],
                requires_counter_evidence=False,
                status="pending",
                coverage_score=0.0
            )
            refined.append(source_sq)

        return refined[:max_sq]

    def run(
        self,
        question: str,
        task_analysis: TaskAnalysis,
        max_subquestions: int = 10
    ) -> list[SubQuestion]:
        """Run the decomposition planner."""
        return self.decompose(question, task_analysis, max_subquestions)
