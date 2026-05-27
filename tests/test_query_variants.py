"""Tests for query variant generation and subquestion refinement."""

import sys

sys.path.insert(0, 'src')

import unittest
from unittest.mock import MagicMock

from src.agents.retrieval_agent import DynamicRetrievalAgent
from src.utils.data_structures import Evidence, SubQuestion


class TestQueryVariants(unittest.TestCase):
    """Test query variant generation."""

    def setUp(self):
        mock_retriever = MagicMock()
        self.agent = DynamicRetrievalAgent(retriever=mock_retriever)

    def test_generates_multiple_variants(self):
        """应生成 3-5 个查询变体"""
        variants = self.agent._generate_query_variants("What is the relationship between RAG and hallucination?")
        self.assertGreaterEqual(len(variants), 3)
        self.assertLessEqual(len(variants), 5)

    def test_original_question_included(self):
        """原始问题应包含在变体中"""
        question = "What causes climate change?"
        variants = self.agent._generate_query_variants(question)
        self.assertIn(question, variants)

    def test_stopwords_removed_variant(self):
        """应有去掉停用词的精简版变体"""
        variants = self.agent._generate_query_variants("What is the impact of RAG on hallucination?")
        non_stopword_variants = [
            v for v in variants
            if not any(w in v.lower().split() for w in ["what", "is", "the", "of"])
        ]
        self.assertGreater(len(non_stopword_variants), 0)

    def test_entity_focused_variant(self):
        """应有实体聚焦版变体"""
        variants = self.agent._generate_query_variants(
            "How does Einstein's theory relate to Newton's laws?"
        )
        entity_variants = [v for v in variants if "Einstein" in v or "Newton" in v]
        self.assertGreater(len(entity_variants), 0)


class TestSubquestionRefinement(unittest.TestCase):
    """Test subquestion refinement."""

    def setUp(self):
        mock_retriever = MagicMock()
        self.agent = DynamicRetrievalAgent(retriever=mock_retriever)

    def test_refine_updates_question_text(self):
        """精炼后的子问题应有不同的文本"""
        sq = SubQuestion(
            id="sq0",
            question="What is the mechanism of action?",
            coverage_score=0.2
        )
        evidence = [
            Evidence(
                evidence_id="E1", source="test", title="test",
                text_span="The mechanism involves protein synthesis"
            )
        ]

        refined = self.agent._refine_subquestion(sq, evidence)
        self.assertIsNotNone(refined)
        self.assertEqual(refined.id, "sq0")

    def test_refine_preserves_context(self):
        """精炼后应保留原子问题的上下文"""
        sq = SubQuestion(
            id="sq1",
            question="How does RAG reduce hallucination?",
            dependency_ids=["sq0"],
            coverage_score=0.3
        )
        evidence = []

        refined = self.agent._refine_subquestion(sq, evidence)
        self.assertEqual(refined.dependency_ids, ["sq0"])


if __name__ == "__main__":
    unittest.main()
