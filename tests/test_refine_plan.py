"""Tests for plan refinement."""

import sys
import unittest

sys.path.insert(0, 'src')

from src.agents.planner import DecompositionPlanner
from src.utils.data_structures import (
    Complexity,
    SubQuestion,
    TaskAnalysis,
    TaskType,
    UncertaintyBreakdown,
)


class MockLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt, **kwargs):
        return self.response


class TestRefinePlan(unittest.TestCase):
    """Test plan refinement based on uncertainty feedback."""

    def setUp(self):
        self.planner = DecompositionPlanner()

    def test_high_retrieval_uncertainty_adds_queries(self):
        """检索不确定性高时，为低覆盖度子问题提升优先级"""
        subquestions = [
            SubQuestion(id="sq0", question="What is RAG?", coverage_score=0.2),
            SubQuestion(id="sq1", question="How does RAG work?", coverage_score=0.9),
        ]
        uncertainty = UncertaintyBreakdown(retrieval_uncertainty=0.8)

        refined = self.planner.refine_plan(subquestions, {"uncertainty": uncertainty})

        # sq0 应被标记为 in_progress（低覆盖度 + 高检索不确定性）
        sq0 = next(sq for sq in refined if sq.id == "sq0")
        self.assertEqual(sq0.status, "in_progress")
        # sq1 覆盖度高，保持 resolved
        sq1 = next(sq for sq in refined if sq.id == "sq1")
        self.assertEqual(sq1.status, "resolved")

    def test_high_conflict_uncertainty_adds_resolution_question(self):
        """冲突不确定性高时，添加冲突解决子问题"""
        subquestions = [
            SubQuestion(id="sq0", question="What caused the crash?", coverage_score=0.7),
        ]
        uncertainty = UncertaintyBreakdown(evidence_conflict=0.8)

        refined = self.planner.refine_plan(subquestions, {
            "uncertainty": uncertainty,
            "conflicts": [{"type": "temporal"}]
        })

        # 应新增冲突解决子问题
        new_sqs = [sq for sq in refined if "conflict" in sq.question.lower()]
        self.assertGreater(len(new_sqs), 0)
        self.assertIn("conflict", new_sqs[0].question.lower())

    def test_low_uncertainty_returns_unchanged(self):
        """低不确定性时保持原样"""
        subquestions = [
            SubQuestion(id="sq0", question="What is X?", coverage_score=0.7),
        ]
        uncertainty = UncertaintyBreakdown()

        refined = self.planner.refine_plan(subquestions, {"uncertainty": uncertainty})

        self.assertEqual(len(refined), len(subquestions))

    def test_respects_max_subquestions_limit(self):
        """不超过最大子问题数量限制"""
        subquestions = [
            SubQuestion(id=f"sq{i}", question=f"Question {i}", coverage_score=0.3)
            for i in range(8)
        ]
        uncertainty = UncertaintyBreakdown(
            retrieval_uncertainty=0.8,
            evidence_conflict=0.8
        )

        refined = self.planner.refine_plan(subquestions, {
            "uncertainty": uncertainty,
            "conflicts": [{"type": "temporal"}],
            "max_subquestions": 10
        })

        self.assertLessEqual(len(refined), 10)

    def test_llm_decompose_normalizes_empty_questions_and_invalid_dependencies(self):
        planner = DecompositionPlanner(llm_client=MockLLM(
            """
            {
              "subquestions": [
                {"question": "  ", "dependency_ids": ["missing"]},
                {"question": "Find revenue", "dependency_ids": ["missing"]},
                {"question": "Compare growth", "dependency_ids": ["sq1", "missing"]},
                "bad item"
              ]
            }
            """
        ))
        analysis = TaskAnalysis(
            task_type=TaskType.MULTI_HOP_QA,
            complexity=Complexity.HIGH,
            estimated_hops=3,
            keywords=["revenue", "growth"],
        )

        subquestions = planner.decompose("Compare company revenue growth", analysis, max_subquestions=3)

        self.assertEqual([sq.id for sq in subquestions], ["sq0", "sq1"])
        self.assertEqual(subquestions[0].question, "Find revenue")
        self.assertEqual(subquestions[0].dependency_ids, [])
        self.assertEqual(subquestions[1].dependency_ids, ["sq0"])

    def test_llm_decompose_falls_back_when_response_has_no_usable_questions(self):
        planner = DecompositionPlanner(llm_client=MockLLM('{"subquestions": []}'))
        analysis = TaskAnalysis(
            task_type=TaskType.MULTI_HOP_QA,
            complexity=Complexity.HIGH,
            requires_conflict_check=True,
            estimated_hops=3,
            keywords=[],
        )

        subquestions = planner.decompose("What happened?", analysis, max_subquestions=3)

        self.assertEqual(len(subquestions), 1)
        self.assertEqual(subquestions[0].id, "sq0")
        self.assertEqual(subquestions[0].question, "What happened?")
        self.assertTrue(subquestions[0].requires_counter_evidence)

    def test_fallback_decompose_respects_max_and_remaps_dependencies(self):
        analysis = TaskAnalysis(
            task_type=TaskType.COMPARATIVE_ANALYSIS,
            complexity=Complexity.HIGH,
            requires_conflict_check=True,
            estimated_hops=4,
            keywords=["a", "b", "c", "d"],
        )

        subquestions = self.planner._fallback_decompose(
            "Compare a and b",
            analysis,
            max_subquestions=3,
        )

        self.assertEqual([sq.id for sq in subquestions], ["sq0", "sq1", "sq2"])
        self.assertLessEqual(len(subquestions), 3)
        for sq in subquestions:
            self.assertTrue(set(sq.dependency_ids) <= {"sq0", "sq1"})

    def test_reasoning_plan_filters_empty_steps_and_falls_back_on_bad_shape(self):
        planner = DecompositionPlanner(llm_client=MockLLM(
            '{"reasoning_plan": [" First step ", "", 3]}'
        ))
        steps = planner.get_reasoning_plan("Q", [SubQuestion(id="sq0", question="Q")])

        self.assertEqual(steps, ["First step", "3"])

        planner_bad = DecompositionPlanner(llm_client=MockLLM('{"reasoning_plan": ""}'))
        fallback = planner_bad.get_reasoning_plan("Q", [SubQuestion(id="sq0", question="Q")])

        self.assertIn("Gather evidence", fallback[0])

    def test_refine_plan_normalizes_ids_after_adding_followups(self):
        subquestions = [
            SubQuestion(id="old_a", question="Check claim A", coverage_score=0.2),
            SubQuestion(id="old_b", question="Check claim B", coverage_score=0.2),
        ]
        uncertainty = UncertaintyBreakdown(evidence_conflict=0.9, source_reliability=0.9)

        refined = self.planner.refine_plan(subquestions, {
            "uncertainty": uncertainty,
            "conflicts": [{"type": "numeric"}, {"type": "temporal"}],
            "max_subquestions": 4,
        })

        self.assertEqual([sq.id for sq in refined], [f"sq{i}" for i in range(len(refined))])
        for index, sq in enumerate(refined):
            previous_ids = {f"sq{i}" for i in range(index)}
            self.assertTrue(set(sq.dependency_ids) <= previous_ids)

    def test_refine_plan_tolerates_bad_max_subquestions(self):
        subquestions = [
            SubQuestion(id="old_a", question="Check claim A", coverage_score=0.2),
        ]

        refined = self.planner.refine_plan(
            subquestions,
            {
                "uncertainty": UncertaintyBreakdown(evidence_conflict=0.9),
                "conflicts": [{"type": "numeric"}],
                "max_subquestions": "bad-limit",
            },
        )

        self.assertEqual(refined[0].id, "sq0")
        self.assertGreaterEqual(len(refined), 1)


if __name__ == "__main__":
    unittest.main()
