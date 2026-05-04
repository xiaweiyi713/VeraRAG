"""Tests for plan refinement."""

import sys
sys.path.insert(0, 'src')

import unittest
from src.agents.planner import DecompositionPlanner
from src.utils.data_structures import SubQuestion, UncertaintyBreakdown


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
        new_sqs = [sq for sq in refined if sq.id.startswith("sq_resolve")]
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


if __name__ == "__main__":
    unittest.main()
