"""Tests for query variant generation and subquestion refinement."""

import sys

sys.path.insert(0, 'src')

import unittest
from unittest.mock import MagicMock

from src.agents.retrieval_agent import DynamicRetrievalAgent
from src.retriever.base import RetrievalResult
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

    def test_result_to_evidence_preserves_temporal_metadata(self):
        result = RetrievalResult(
            doc_id="D011_c0",
            content="截至2023年末，公司员工总数增至41,000人。",
            title="星辰科技2023年度财务报告",
            score=0.9,
            metadata={
                "source": "report",
                "date": "2024-03-28",
                "author": "星辰科技投资者关系部",
                "url": "https://startech.com/ir/2023",
            },
        )

        evidence = self.agent._result_to_evidence(result, "D011_c0")

        self.assertEqual(evidence.date, "2024-03-28")
        self.assertEqual(evidence.author, "星辰科技投资者关系部")

    def test_counter_evidence_uses_at_least_one_result_for_small_top_k(self):
        class RecordingRetriever:
            def __init__(self):
                self.calls = []

            def retrieve(self, query, top_k=10):
                self.calls.append((query, top_k))
                return []

        retriever = RecordingRetriever()
        agent = DynamicRetrievalAgent(retriever=retriever)

        agent.retrieve_for_subquestion(
            SubQuestion(
                id="sq0",
                question="Claim X is true",
                requires_counter_evidence=True,
            ),
            top_k=1,
        )

        counter_calls = [
            (query, top_k)
            for query, top_k in retriever.calls
            if "不实" in query or "false" in query or "最新" in query
        ]
        self.assertGreater(len(counter_calls), 0)
        self.assertTrue(all(top_k == 1 for _, top_k in counter_calls))

    def test_counter_evidence_query_generation_covers_challenge_temporal_and_alternative_paths(self):
        queries = self.agent._generate_counter_evidence_queries("欧盟AI法案已经通过了吗？")

        self.assertEqual(len(queries), 8)
        self.assertTrue(any("不实" in query for query in queries))
        self.assertTrue(any(query.startswith("最新 ") for query in queries))
        self.assertTrue(any("不同观点" in query for query in queries))

    def test_current_attribute_refresh_forces_second_pass_and_latest_report_query(self):
        agent = DynamicRetrievalAgent(
            retriever=MagicMock(),
            config={"retriever": {"targeted_second_pass_enabled": True}},
        )
        subquestion = SubQuestion(
            id="sq0",
            question="星辰科技是哪一年成立的？目前有多少员工？",
            required_evidence_type="general",
        )

        self.assertTrue(agent._should_run_targeted_second_pass(subquestion, coverage=1.0))

        refined = agent._refine_subquestion(
            subquestion,
            [
                Evidence(
                    evidence_id="D010_c0",
                    source="report",
                    title="星辰科技2022年度财务报告",
                    text_span="截至2022年末，公司员工总数为32,000人。",
                )
            ],
        )

        self.assertIn("星辰科技", refined.question)
        self.assertIn("年度财务报告", refined.question)
        self.assertIn("员工总数", refined.question)
        self.assertIn("截至2023年末", refined.question)

    def test_current_role_refresh_targets_new_officer_evidence(self):
        agent = DynamicRetrievalAgent(
            retriever=MagicMock(),
            config={"retriever": {"targeted_second_pass_enabled": True}},
        )
        subquestion = SubQuestion(
            id="sq0",
            question="星辰科技目前的CTO是谁？",
            required_evidence_type="temporal",
        )

        self.assertTrue(agent._should_run_targeted_second_pass(subquestion, coverage=1.0))

        refined = agent._refine_subquestion(
            subquestion,
            [
                Evidence(
                    evidence_id="D011_c0",
                    source="report",
                    title="星辰科技2023年度财务报告",
                    text_span="2023年11月，公司CTO张伟宣布因个人原因离职。",
                )
            ],
        )

        self.assertIn("星辰科技", refined.question)
        self.assertIn("CTO", refined.question)
        self.assertIn("现任", refined.question)
        self.assertIn("新任", refined.question)
        self.assertIn("加入", refined.question)


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

    def test_refine_returns_same_object_when_no_content_words(self):
        sq = SubQuestion(id="sq0", question="is it?", coverage_score=0.1)

        refined = self.agent._refine_subquestion(sq, [])

        self.assertIs(refined, sq)

    def test_refine_returns_same_object_when_all_content_words_are_covered(self):
        sq = SubQuestion(id="sq0", question="alpha beta", coverage_score=0.4)
        evidence = [
            Evidence(
                evidence_id="E1",
                source="test",
                title="alpha",
                text_span="beta",
            )
        ]

        refined = self.agent._refine_subquestion(sq, evidence)

        self.assertIs(refined, sq)


class TestDynamicRetrieve(unittest.TestCase):
    class RecordingRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=10):
            self.calls.append((query, top_k))
            return [
                RetrievalResult(
                    doc_id="",
                    content="alpha evidence only",
                    title="alpha source",
                    score=0.7,
                    metadata={"source": "fixture"},
                )
            ]

    class CoveringRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=10):
            self.calls.append((query, top_k))
            token = "beta" if "beta" in query else "alpha"
            return [
                RetrievalResult(
                    doc_id=f"{token}-{index}",
                    content=f"{token} supporting evidence",
                    title=f"{token} source",
                    score=1.0 - (index * 0.01),
                    metadata={"source": "fixture"},
                )
                for index in range(3)
            ]

    def test_dynamic_retrieve_writes_refined_subquestion_back_for_next_round(self):
        retriever = self.RecordingRetriever()
        agent = DynamicRetrievalAgent(retriever=retriever)
        subquestions = [
            SubQuestion(id="sq0", question="alpha beta", coverage_score=0.1)
        ]

        evidence = agent.dynamic_retrieve(
            subquestions,
            [],
            max_rounds=2,
            budget_per_round=1,
        )

        self.assertTrue(
            any(
                query.startswith("Find specific information about beta")
                for query, _top_k in retriever.calls
            )
        )
        self.assertNotEqual(subquestions[0].question, "alpha beta")
        self.assertIn("beta", subquestions[0].question)
        self.assertTrue(evidence[0].evidence_id.startswith("E"))
        self.assertGreaterEqual(min(top_k for _query, top_k in retriever.calls), 1)

    def test_dynamic_retrieve_returns_existing_pool_when_all_resolved(self):
        agent = DynamicRetrievalAgent(retriever=self.RecordingRetriever())
        pool = [
            Evidence(evidence_id="E1", source="test", title="done", text_span="done")
        ]

        result = agent.dynamic_retrieve(
            [SubQuestion(id="sq0", question="done", status="resolved")],
            pool,
        )

        self.assertIs(result, pool)

    def test_dynamic_retrieve_resolves_questions_and_breaks_when_done(self):
        retriever = self.CoveringRetriever()
        agent = DynamicRetrievalAgent(retriever=retriever)
        subquestions = [
            SubQuestion(id="sq0", question="alpha", coverage_score=0.1),
            SubQuestion(id="sq1", question="beta", coverage_score=0.2),
        ]

        result = agent.dynamic_retrieve(
            subquestions,
            [],
            max_rounds=3,
            budget_per_round=6,
        )

        self.assertEqual([sq.status for sq in subquestions], ["resolved", "resolved"])
        self.assertEqual([sq.coverage_score for sq in subquestions], [1.0, 1.0])
        self.assertEqual(len(result), 6)

    def test_run_delegates_to_dynamic_retrieve(self):
        agent = DynamicRetrievalAgent(retriever=self.CoveringRetriever())
        subquestion = SubQuestion(id="sq0", question="alpha", coverage_score=0.1)

        result = agent.run([subquestion], [], max_rounds=1, budget_per_round=3)

        self.assertEqual(subquestion.status, "resolved")
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
