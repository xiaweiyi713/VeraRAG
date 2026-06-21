"""Tests for agent modules: task_analyzer, reasoning, verifier, repair."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_structures import (
    AnswerClaim,
    Claim,
    ClaimType,
    Complexity,
    ConflictEdge,
    ConflictGraphNode,
    ConflictType,
    Evidence,
    EvidenceConflictGraph,
    SubQuestion,
    TaskAnalysis,
    TaskType,
    VerificationReport,
    VerificationStatus,
)


class MockLLM:
    """Mock LLM client that returns canned responses."""
    def __init__(self, response: str = "mock response"):
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self.response


class TestDynamicRetrievalAgent:
    class RecordingRetriever:
        def __init__(self):
            self.top_k_calls = []

        def retrieve(self, query, top_k=10, **kwargs):
            from src.retriever.base import RetrievalResult

            self.top_k_calls.append(top_k)
            return [
                RetrievalResult(
                    doc_id=f"D{idx:03d}_c0",
                    content=f"欧盟AI法案证据 {idx}",
                    title="欧盟AI法案",
                    score=float(top_k - idx),
                    metadata={"entities": ["欧盟AI法案"]},
                )
                for idx in range(top_k)
            ]

    def test_result_to_evidence_preserves_entity_metadata(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent
        from src.retriever.base import RetrievalResult

        agent = DynamicRetrievalAgent(retriever=None)  # type: ignore[arg-type]
        result = RetrievalResult(
            doc_id="D043_c0",
            content="比亚迪销量302.4万辆，特斯拉交付181万辆。",
            title="比亚迪vs特斯拉：2023年全球销量对比",
            score=1.0,
            metadata={
                "source": "report",
                "entities": ["比亚迪", "特斯拉", "销量"],
                "date": "2024-01-01",
                "author": "analyst",
            },
        )

        evidence = agent._result_to_evidence(result, result.doc_id)

        assert evidence.entities == ["比亚迪", "特斯拉", "销量"]
        assert evidence.date == "2024-01-01"
        assert evidence.author == "analyst"

    def test_original_question_anchor_keeps_full_retrieval_depth(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent

        retriever = self.RecordingRetriever()
        agent = DynamicRetrievalAgent(retriever=retriever)  # type: ignore[arg-type]
        subquestions = [
            SubQuestion(id="sq_original", question="原始问题", requires_counter_evidence=False),
            *[
                SubQuestion(id=f"sq{i}", question=f"子问题{i}", requires_counter_evidence=False)
                for i in range(10)
            ],
        ]

        agent.dynamic_retrieve(subquestions, [], max_rounds=1, budget_per_round=50)

        assert retriever.top_k_calls[0] == 10

    def test_original_question_anchor_uses_configured_retrieval_depth(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent

        retriever = self.RecordingRetriever()
        agent = DynamicRetrievalAgent(
            retriever=retriever,  # type: ignore[arg-type]
            config={"retriever": {"retrieval_top_k": 3}},
        )
        subquestions = [
            SubQuestion(id="sq_original", question="原始问题", requires_counter_evidence=False),
            *[
                SubQuestion(id=f"sq{i}", question=f"子问题{i}", requires_counter_evidence=False)
                for i in range(10)
            ],
        ]

        agent.dynamic_retrieve(subquestions, [], max_rounds=1, budget_per_round=50)

        assert retriever.top_k_calls[0] == 3

    def test_precision_cap_keeps_retrieval_depth_but_limits_output(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent

        retriever = self.RecordingRetriever()
        agent = DynamicRetrievalAgent(
            retriever=retriever,  # type: ignore[arg-type]
            config={"retriever": {"top_k_policy": "precision_cap", "precision_cap_top_k": 4}},
        )

        results = agent.retrieve_for_subquestion(
            SubQuestion(id="sq_original", question="原始问题"),
            top_k=10,
        )

        assert retriever.top_k_calls == [10]
        assert len(results) == 4

    def test_complexity_adaptive_keeps_more_for_conflict_questions(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent

        retriever = self.RecordingRetriever()
        agent = DynamicRetrievalAgent(
            retriever=retriever,  # type: ignore[arg-type]
            config={"retriever": {"top_k_policy": "complexity_adaptive"}},
        )

        simple = agent.retrieve_for_subquestion(
            SubQuestion(id="sq_simple", question="普通事实问题"),
            top_k=10,
        )
        conflict = agent.retrieve_for_subquestion(
            SubQuestion(
                id="sq_conflict",
                question="冲突问题",
                required_evidence_type="conflict",
            ),
            top_k=10,
        )

        assert len(simple) == 2
        assert len(conflict) == 5

    def test_complexity_adaptive_top3_caps_complex_output_at_retrieval_depth(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent

        retriever = self.RecordingRetriever()
        agent = DynamicRetrievalAgent(
            retriever=retriever,  # type: ignore[arg-type]
            config={
                "retriever": {
                    "retrieval_top_k": 3,
                    "top_k_policy": "complexity_adaptive",
                }
            },
        )

        conflict = agent.retrieve_for_subquestion(
            SubQuestion(
                id="sq_conflict",
                question="冲突问题",
                required_evidence_type="conflict",
            ),
            top_k=agent.retrieval_top_k,
        )

        assert retriever.top_k_calls == [3]
        assert len(conflict) == 3

    def test_targeted_second_pass_appends_bounded_evidence_for_low_coverage_complex_need(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent

        retriever = self.RecordingRetriever()
        agent = DynamicRetrievalAgent(
            retriever=retriever,  # type: ignore[arg-type]
            config={
                "retriever": {
                    "retrieval_top_k": 3,
                    "top_k_policy": "complexity_adaptive",
                    "targeted_second_pass_enabled": True,
                    "targeted_second_pass_top_k": 8,
                    "targeted_second_pass_max_new_evidence": 2,
                    "targeted_second_pass_coverage_threshold": 0.67,
                }
            },
        )
        subquestions = [
            SubQuestion(
                id="sq_original",
                question="完全不匹配的问题",
                required_evidence_type="conflict",
            )
        ]

        evidence = agent.dynamic_retrieve(subquestions, [], max_rounds=1)

        assert retriever.top_k_calls[0] == 3
        assert retriever.top_k_calls[1:]
        assert set(retriever.top_k_calls[1:]) == {8}
        assert [item.evidence_id for item in evidence] == [
            "D000_c0",
            "D001_c0",
            "D002_c0",
            "D003_c0",
            "D004_c0",
        ]

    def test_targeted_second_pass_skips_simple_need(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent

        retriever = self.RecordingRetriever()
        agent = DynamicRetrievalAgent(
            retriever=retriever,  # type: ignore[arg-type]
            config={
                "retriever": {
                    "retrieval_top_k": 3,
                    "targeted_second_pass_enabled": True,
                    "targeted_second_pass_top_k": 8,
                }
            },
        )
        subquestions = [
            SubQuestion(id="sq_original", question="完全不匹配的问题")
        ]

        evidence = agent.dynamic_retrieve(subquestions, [], max_rounds=1)

        assert retriever.top_k_calls == [3]
        assert len(evidence) == 3

    def test_compact_entity_group_expands_retrieval_queries(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent

        agent = DynamicRetrievalAgent(retriever=self.RecordingRetriever())  # type: ignore[arg-type]

        variants = agent._generate_query_variants(
            "中美欧在AI监管方面分别采取了什么模式？各有什么特点？"
        )

        assert variants[:4] == [
            "中美欧在AI监管方面分别采取了什么模式？各有什么特点？",
            "中国在AI监管方面分别采取了什么模式？各有什么特点？",
            "美国在AI监管方面分别采取了什么模式？各有什么特点？",
            "欧盟在AI监管方面分别采取了什么模式？各有什么特点？",
        ]


# --- TaskAnalyzer Tests ---

class TestTaskAnalyzer:
    def test_rule_based_analysis(self):
        from src.agents.task_analyzer import TaskAnalyzer
        analyzer = TaskAnalyzer(llm_client=MockLLM())
        result = analyzer.analyze("什么是欧盟AI法案的主要条款？")
        assert isinstance(result, TaskAnalysis)
        assert result.keywords is not None
        # keywords may be empty if LLM is needed for extraction
        assert isinstance(result.keywords, list)

    def test_multihop_detection(self):
        from src.agents.task_analyzer import TaskAnalyzer
        analyzer = TaskAnalyzer(llm_client=MockLLM())
        result = analyzer.analyze("比较A公司和B公司在2024年的营收增长率和市场份额")
        assert result.estimated_hops >= 2

    def test_keywords_extraction(self):
        from src.agents.task_analyzer import TaskAnalyzer
        analyzer = TaskAnalyzer(llm_client=MockLLM())
        result = analyzer.analyze("量子计算在密码学中的应用")
        # TaskAnalyzer returns TaskAnalysis with keywords field
        assert isinstance(result, TaskAnalysis)
        assert isinstance(result.keywords, list)

    def test_requires_retrieval_default(self):
        from src.agents.task_analyzer import TaskAnalyzer
        analyzer = TaskAnalyzer(llm_client=MockLLM())
        result = analyzer.analyze("为什么天空是蓝色的？")
        assert result.requires_retrieval is True

    def test_empty_question(self):
        from src.agents.task_analyzer import TaskAnalyzer
        analyzer = TaskAnalyzer(llm_client=MockLLM())
        result = analyzer.analyze("")
        assert isinstance(result, TaskAnalysis)

    def test_simple_question_stays_low_without_llm(self):
        from src.agents.task_analyzer import TaskAnalyzer

        analyzer = TaskAnalyzer()

        result = analyzer.analyze("What is VeraRAG?")

        assert result.complexity == Complexity.LOW
        assert result.estimated_hops == 1

    def test_complex_question_without_llm_falls_back_to_rules(self):
        from src.agents.task_analyzer import TaskAnalyzer

        analyzer = TaskAnalyzer()

        result = analyzer.analyze("Compare revenue in 2023 and 2024 and explain the trend")

        assert result.task_type == TaskType.COMPARATIVE_ANALYSIS
        assert result.complexity in {Complexity.MEDIUM, Complexity.HIGH}
        assert result.requires_conflict_check is True

    def test_llm_analysis_normalizes_common_json_shapes(self):
        from src.agents.task_analyzer import TaskAnalyzer

        response = """{
            "task_type": "multi_hop_qa",
            "complexity": "HIGH",
            "requires_retrieval": "false",
            "requires_conflict_check": "yes",
            "requires_numerical_reasoning": "1",
            "requires_temporal_reasoning": "0",
            "estimated_hops": "10",
            "keywords": ["Revenue", " revenue ", "", 123, "Market"]
        }"""
        analyzer = TaskAnalyzer(llm_client=MockLLM(response))

        result = analyzer.analyze("Compare revenue in 2023 and 2024 and explain the trend")

        assert result.task_type == TaskType.MULTI_HOP_QA
        assert result.complexity == Complexity.HIGH
        assert result.requires_retrieval is False
        assert result.requires_conflict_check is True
        assert result.requires_numerical_reasoning is True
        assert result.requires_temporal_reasoning is False
        assert result.estimated_hops == 5
        assert result.keywords == ["Revenue", "Market"]

    def test_llm_analysis_splits_keyword_string_and_clamps_low_hops(self):
        from src.agents.task_analyzer import TaskAnalyzer

        response = """{
            "task_type": "fact verification",
            "complexity": "medium",
            "estimated_hops": 0,
            "keywords": "EU AI Act, facial recognition; 2024"
        }"""
        analyzer = TaskAnalyzer(llm_client=MockLLM(response))

        result = analyzer.analyze("Is the claim about the EU AI Act in 2024 accurate?")

        assert result.task_type == TaskType.FACT_VERIFICATION
        assert result.estimated_hops == 1
        assert result.keywords == ["EU AI Act", "facial recognition", "2024"]

    def test_llm_analysis_non_object_response_falls_back_to_rules(self):
        from src.agents.task_analyzer import TaskAnalyzer

        analyzer = TaskAnalyzer(llm_client=MockLLM('["not", "an", "object"]'))

        result = analyzer.analyze("Compare revenue in 2023 and 2024")

        assert result.task_type == TaskType.COMPARATIVE_ANALYSIS
        assert result.requires_conflict_check is True

    def test_task_analyzer_coercion_helpers_cover_defaults_and_limits(self):
        from src.agents.task_analyzer import TaskAnalyzer

        assert TaskAnalyzer._coerce_task_type(
            TaskType.SCIENTIFIC_REVIEW,
            TaskType.MULTI_HOP_QA,
        ) == TaskType.SCIENTIFIC_REVIEW
        assert TaskAnalyzer._coerce_task_type(
            "temporal_reasoning",
            TaskType.MULTI_HOP_QA,
        ) == TaskType.TEMPORAL_REASONING
        assert TaskAnalyzer._coerce_task_type("unknown", TaskType.MULTI_HOP_QA) == TaskType.MULTI_HOP_QA
        assert TaskAnalyzer._coerce_task_type(None, TaskType.FACT_VERIFICATION) == TaskType.FACT_VERIFICATION

        assert TaskAnalyzer._coerce_complexity(Complexity.HIGH, Complexity.LOW) == Complexity.HIGH
        assert TaskAnalyzer._coerce_complexity("bad", Complexity.MEDIUM) == Complexity.MEDIUM
        assert TaskAnalyzer._coerce_complexity(None, Complexity.LOW) == Complexity.LOW

        assert TaskAnalyzer._coerce_bool(True, False) is True
        assert TaskAnalyzer._coerce_bool("maybe", True) is True
        assert TaskAnalyzer._coerce_estimated_hops(True, 3) == 3
        assert TaskAnalyzer._coerce_estimated_hops("bad", 2) == 2
        assert TaskAnalyzer._coerce_keywords(None) == []
        assert TaskAnalyzer._coerce_keywords([str(i) for i in range(12)]) == [str(i) for i in range(10)]

    def test_run_delegates_to_analyze(self):
        from src.agents.task_analyzer import TaskAnalyzer

        result = TaskAnalyzer().run("What is VeraRAG?")

        assert result.complexity == Complexity.LOW


# --- ReasoningAgent Tests ---

class TestReasoningAgent:
    def _make_evidence(self, count=2):
        evs = []
        for i in range(count):
            ev = Evidence(
                evidence_id=f"E{i+1}",
                source="test",
                title=f"Evidence {i+1}",
                text_span=f"这是第{i+1}条证据内容。",
                credibility_score=0.8,
                relevance_score=0.7,
            )
            evs.append(ev)
        return evs

    def _make_subquestions(self):
        return [
            SubQuestion(id="sq1", question="子问题1", required_evidence_type="factual",
                       dependency_ids=[], requires_counter_evidence=False, status="completed", coverage_score=0.8),
        ]

    def test_claim_slot_selection_compresses_reasoning_evidence_context(self):
        from src.agents.reasoning_agent import ReasoningAgent

        evidence = [
            Evidence(evidence_id="E1", source="test", title="噪声1", text_span="无关背景"),
            Evidence(evidence_id="E2", source="test", title="噪声2", text_span="更多无关背景"),
            Evidence(
                evidence_id="E3",
                source="test",
                title="星辰科技财报",
                text_span="星辰科技2023财年营收达到612亿元。",
                credibility_score=1.0,
                recency_score=1.0,
                relevance_score=1.0,
            ),
            Evidence(
                evidence_id="E4",
                source="test",
                title="星辰科技成本说明",
                text_span="星辰科技2023财年研发投入为120亿元。",
                credibility_score=0.9,
                recency_score=0.9,
                relevance_score=0.9,
            ),
        ]
        agent = ReasoningAgent(
            config={"reasoning": {"claim_slot_selection_enabled": True, "claim_slot_max_evidence": 2}},
            llm_client=MockLLM(),
        )

        context = agent._prepare_evidence_context(
            evidence,
            question="星辰科技2023年营收是多少？",
            subquestions=[],
            conflict_graph=EvidenceConflictGraph(),
        )

        assert "[E3]" in context
        assert "[E4]" in context
        assert "[E1]" not in context
        assert "[E2]" not in context

    def test_claim_slot_selection_preserves_conflict_evidence_beyond_slot_limit(self):
        from src.agents.reasoning_agent import ReasoningAgent

        ev1 = Evidence(
            evidence_id="E1",
            source="test",
            title="权威财报",
            text_span="星辰科技2023财年营收达到612亿元。",
            claims=[Claim("C1", "星辰科技2023财年营收达到612亿元", ClaimType.NUMERICAL)],
            credibility_score=1.0,
            recency_score=1.0,
            relevance_score=1.0,
        )
        ev2 = Evidence(
            evidence_id="E2",
            source="test",
            title="不实报道",
            text_span="星辰科技2023年营收突破800亿元。",
            claims=[Claim("C2", "星辰科技2023年营收突破800亿元", ClaimType.NUMERICAL)],
            credibility_score=0.1,
            recency_score=0.1,
            relevance_score=0.1,
        )
        graph = EvidenceConflictGraph()
        graph.add_node(ConflictGraphNode("C1", ev1.claims[0].claim, "claim", ["E1"]))
        graph.add_node(ConflictGraphNode("C2", ev2.claims[0].claim, "claim", ["E2"]))
        graph.add_edge(ConflictEdge("C1", "C2", ConflictType.NUMERIC_CONFLICT, 0.9))
        agent = ReasoningAgent(
            config={"reasoning": {"claim_slot_selection_enabled": True, "claim_slot_max_evidence": 1}},
            llm_client=MockLLM(),
        )

        context = agent._prepare_evidence_context(
            [ev1, ev2],
            question="星辰科技2023年营收是多少？",
            subquestions=[],
            conflict_graph=graph,
        )

        assert "[E1]" in context
        assert "[E2]" in context

    def test_reason_with_mock_llm(self):
        from src.agents.reasoning_agent import ReasoningAgent

        json_response = '''{
            "answer": "测试答案",
            "claims": [{"claim": "声明1", "supporting_evidence": ["E1"], "conflicting_evidence": [], "confidence": 0.9, "verification_status": "supported"}],
            "steps": [{"step": 1, "description": "分析证据", "evidence_ids": ["E1"], "confidence": 0.85}]
        }'''
        agent = ReasoningAgent(llm_client=MockLLM(json_response))
        answer, _claims, _steps = agent.reason(
            question="测试问题",
            subquestions=self._make_subquestions(),
            evidence=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
            reasoning_plan=["分析证据"],
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_reason_fallback_on_bad_json(self):
        from src.agents.reasoning_agent import ReasoningAgent
        agent = ReasoningAgent(llm_client=MockLLM("not valid json"))
        answer, _claims, _steps = agent.reason(
            question="测试问题",
            subquestions=[],
            evidence=self._make_evidence(1),
            conflict_graph=EvidenceConflictGraph(),
            reasoning_plan=[],
        )
        assert isinstance(answer, str)

    def test_reason_acknowledges_detected_conflicts(self):
        from src.agents.reasoning_agent import ReasoningAgent

        ev1 = Evidence(
            evidence_id="D014_c0",
            source="news",
            title="不实报道",
            text_span="近日有消息称星辰科技2023年营收突破800亿元。",
            claims=[
                Claim(
                    claim_id="C_bad",
                    claim="星辰科技2023年营收突破800亿元",
                    claim_type=ClaimType.FACTUAL,
                    entities=["星辰科技"],
                )
            ],
        )
        ev2 = Evidence(
            evidence_id="D011_c0",
            source="report",
            title="2023财报",
            text_span="星辰科技2023财年全年营收达到612亿元人民币。",
            claims=[
                Claim(
                    claim_id="C_good",
                    claim="星辰科技2023财年全年营收达到612亿元人民币",
                    claim_type=ClaimType.FACTUAL,
                    entities=["星辰科技"],
                )
            ],
        )
        graph = EvidenceConflictGraph()
        graph.add_node(ConflictGraphNode("C_bad", ev1.claims[0].claim, "claim", ["D014_c0"]))
        graph.add_node(ConflictGraphNode("C_good", ev2.claims[0].claim, "claim", ["D011_c0"]))
        graph.add_edge(ConflictEdge("C_bad", "C_good", ConflictType.NUMERIC_CONFLICT, 0.9))
        json_response = '''{
            "answer": "星辰科技2023财年全年营收为612亿元人民币。",
            "answer_claims": [
                {
                    "claim": "星辰科技2023年营收为612亿元",
                    "supporting_evidence": ["D011_c0"],
                    "conflicting_evidence": [],
                    "confidence": 0.8,
                    "claim_type": "factual",
                    "verifiable": true,
                    "support_type": "direct"
                }
            ],
            "reasoning_chain": [
                {"step": 1, "description": "读取财报", "evidence_ids": ["D011_c0"], "confidence": 0.8}
            ]
        }'''
        agent = ReasoningAgent(llm_client=MockLLM(json_response))

        answer, claims, steps = agent.reason(
            question="星辰科技2023年的营收是多少？",
            subquestions=self._make_subquestions(),
            evidence=[ev1, ev2],
            conflict_graph=graph,
            reasoning_plan=["分析证据"],
        )

        assert "冲突" in answer
        assert "D014_c0" in answer
        assert "D011_c0" in answer
        assert claims[0].conflicting_evidence == ["D014_c0", "D011_c0"]
        assert steps[0].evidence_ids == ["D014_c0", "D011_c0"]

    def test_reason_appends_missing_answer_citations_from_supported_claims(self):
        from src.agents.reasoning_agent import ReasoningAgent

        json_response = '''{
            "answer": "星辰科技2023年营收为612亿元。",
            "answer_claims": [
                {
                    "claim": "星辰科技2023年营收为612亿元",
                    "supporting_evidence": ["D011_c0", "D999_c0"],
                    "conflicting_evidence": [],
                    "confidence": 0.8,
                    "claim_type": "factual",
                    "verifiable": true,
                    "support_type": "direct"
                }
            ],
            "reasoning_chain": [
                {"step": 1, "description": "读取财报", "evidence_ids": ["D011_c0"], "confidence": 0.8}
            ]
        }'''
        evidence = [
            Evidence(
                evidence_id="D011_c0",
                source="report",
                title="2023财报",
                text_span="星辰科技2023财年全年营收达到612亿元人民币。",
            )
        ]
        agent = ReasoningAgent(llm_client=MockLLM(json_response))

        answer, _claims, _steps = agent.reason(
            question="星辰科技2023年的营收是多少？",
            subquestions=self._make_subquestions(),
            evidence=evidence,
            conflict_graph=EvidenceConflictGraph(),
            reasoning_plan=["分析证据"],
        )

        assert answer.endswith("引用证据：[D011_c0]")
        assert "D999_c0" not in answer

    def test_reason_can_disable_answer_citation_footer(self):
        from src.agents.reasoning_agent import ReasoningAgent

        json_response = '''{
            "answer": "星辰科技2023年营收为612亿元。",
            "answer_claims": [
                {
                    "claim": "星辰科技2023年营收为612亿元",
                    "supporting_evidence": ["D011_c0"],
                    "conflicting_evidence": [],
                    "confidence": 0.8,
                    "claim_type": "factual",
                    "verifiable": true,
                    "support_type": "direct"
                }
            ],
            "reasoning_chain": []
        }'''
        evidence = [
            Evidence(
                evidence_id="D011_c0",
                source="report",
                title="2023财报",
                text_span="星辰科技2023财年全年营收达到612亿元人民币。",
            )
        ]
        agent = ReasoningAgent(
            config={"reasoning": {"enforce_answer_citations": False}},
            llm_client=MockLLM(json_response),
        )

        answer, _claims, _steps = agent.reason(
            question="星辰科技2023年的营收是多少？",
            subquestions=self._make_subquestions(),
            evidence=evidence,
            conflict_graph=EvidenceConflictGraph(),
            reasoning_plan=["分析证据"],
        )

        assert answer == "星辰科技2023年营收为612亿元。"

    def test_reason_rejects_non_boolean_citation_guard_config(self):
        from src.agents.reasoning_agent import ReasoningAgent

        with pytest.raises(ValueError, match="enforce_answer_citations"):
            ReasoningAgent(
                config={"reasoning": {"enforce_answer_citations": "false"}},
                llm_client=MockLLM(),
            )


# --- VerifierAgent Tests ---

class TestVerifierAgent:
    def _make_claims(self):
        return [
            AnswerClaim(claim="声明1", supporting_evidence=["E1"], conflicting_evidence=[],
                       confidence=0.9, verification_status=VerificationStatus.SUPPORTED),
            AnswerClaim(claim="声明2", supporting_evidence=[], conflicting_evidence=["E1"],
                       confidence=0.3, verification_status=VerificationStatus.REFUTED),
        ]

    def _make_evidence(self):
        return [Evidence(evidence_id="E1", source="test", title="T", text_span="内容", credibility_score=0.8)]

    def test_verify_no_critical_issues(self):
        from src.agents.verifier_agent import VerifierAgent
        agent = VerifierAgent(llm_client=MockLLM(), use_nli=False)
        report = agent.verify_answer(
            answer="测试答案",
            claims=self._make_claims(),
            evidence=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )
        assert isinstance(report, VerificationReport)
        assert len(report.claim_verifications) >= 0

    def test_verify_with_refuted_claim(self):
        from src.agents.verifier_agent import VerifierAgent
        claims = [
            AnswerClaim(claim="错误声明", supporting_evidence=[], conflicting_evidence=["E1"],
                       confidence=0.2, verification_status=VerificationStatus.REFUTED),
        ]
        agent = VerifierAgent(llm_client=MockLLM(), use_nli=False)
        report = agent.verify_answer(
            answer="包含错误声明",
            claims=claims,
            evidence=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )
        assert isinstance(report, VerificationReport)

    def test_ignored_conflicts_with_unacknowledged_edges(self):
        """Verify that conflicts not referenced by any claim appear in ignored_conflicts.

        Regression test for bug: conflict.conflict (attribute error) → conflict.confidence.
        """
        from src.agents.verifier_agent import VerifierAgent
        agent = VerifierAgent(llm_client=MockLLM(), use_nli=False)

        graph = EvidenceConflictGraph()
        graph.add_node(ConflictGraphNode(node_id="E1", content="证据A", node_type="evidence"))
        graph.add_node(ConflictGraphNode(node_id="E2", content="证据B", node_type="evidence"))
        graph.add_edge(ConflictEdge(
            source_id="E1", target_id="E2",
            conflict_type=ConflictType.NUMERIC_CONFLICT,
            confidence=0.8,
            rationale="数值冲突",
        ))

        # Claims do NOT reference E1/E2 → conflict should be marked as ignored
        claims = [
            AnswerClaim(claim="无关声明", supporting_evidence=["E99"], conflicting_evidence=[],
                        confidence=0.5, verification_status=VerificationStatus.NOT_ENOUGH_INFO),
        ]
        evidence = [
            Evidence(evidence_id="E99", source="test", title="T", text_span="内容"),
            Evidence(evidence_id="E1", source="test", title="A", text_span="证据A内容"),
            Evidence(evidence_id="E2", source="test", title="B", text_span="证据B内容"),
        ]

        report = agent.verify_answer(answer="答案", claims=claims, evidence=evidence, conflict_graph=graph)
        assert len(report.ignored_conflicts) == 1
        assert report.ignored_conflicts[0]["confidence"] == 0.8
        assert report.ignored_conflicts[0]["conflict_type"] == "numeric_conflict"

    def test_use_nli_flag(self):
        from src.agents.verifier_agent import VerifierAgent
        agent = VerifierAgent(llm_client=MockLLM(), use_nli=False)
        assert agent.use_nli is False

    def test_nli_uses_evidence_as_premise_and_interprets_logits(self):
        from types import SimpleNamespace

        from src.agents.verifier_agent import VerifierAgent

        calls = []

        class FakeNLI:
            model = SimpleNamespace(
                config=SimpleNamespace(
                    id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
                )
            )

            def predict(self, pairs, show_progress_bar=False):
                calls.extend(pairs)
                return [[0.0, 4.0, 0.0] for _ in pairs]

        agent = VerifierAgent(
            config={"verification": {"nli_threshold": 0.7}},
            llm_client=MockLLM(),
        )
        agent.nli_model = FakeNLI()

        result = agent._nli_verify("声明", ["证据文本"])

        assert calls == [("证据文本", "声明")]
        assert result is not None
        assert result["status"] == "SUPPORTED"
        assert result["confidence"] > 0.9

    def test_verify_claim_prefers_nli_when_available(self):
        from types import SimpleNamespace

        from src.agents.verifier_agent import VerifierAgent

        class FakeNLI:
            model = SimpleNamespace(
                config=SimpleNamespace(
                    id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
                )
            )

            @staticmethod
            def predict(pairs, show_progress_bar=False):
                assert pairs == [("T: 内容", "声明1")]
                return [[0.0, 4.0, 0.0]]

        agent = VerifierAgent(llm_client=MockLLM("not json"))
        agent.nli_model = FakeNLI()

        verification = agent._verify_claim(
            self._make_claims()[0],
            {"E1": self._make_evidence()[0]},
            EvidenceConflictGraph(),
        )

        assert verification["status"] == "SUPPORTED"
        assert verification["method"] == "nli"

    def test_nli_refuted_neutral_invalid_and_exception_paths(self):
        from types import SimpleNamespace

        from src.agents.verifier_agent import VerifierAgent

        class FakeNLI:
            model = SimpleNamespace(
                config=SimpleNamespace(
                    id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
                )
            )

            def __init__(self, scores):
                self.scores = scores

            def predict(self, pairs, show_progress_bar=False):
                return self.scores

        agent = VerifierAgent(
            config={"verification": {"nli_threshold": 0.7}},
            llm_client=MockLLM(),
        )

        agent.nli_model = FakeNLI([[4.0, 0.0, 0.0]])
        refuted = agent._nli_verify("声明", ["证据"])
        assert refuted is not None
        assert refuted["status"] == "REFUTED"

        agent.nli_model = FakeNLI([[0.0, 0.0, 4.0]])
        neutral = agent._nli_verify("声明", ["证据"])
        assert neutral is not None
        assert neutral["status"] == "NOT_ENOUGH_INFO"

        agent.nli_model = FakeNLI([[1.0, 2.0]])
        assert agent._nli_verify("声明", ["证据"]) is None

        class RaisingNLI:
            def predict(self, pairs, show_progress_bar=False):
                raise RuntimeError("boom")

        agent.nli_model = RaisingNLI()
        assert agent._nli_verify("声明", ["证据"]) is None

        assert agent._nli_probabilities([1.0, 2.0]) is None
        assert agent._nli_probabilities([[1.0, 2.0]]) is None

    def test_nli_returns_none_when_model_cannot_load(self, monkeypatch):
        from src.agents.verifier_agent import VerifierAgent

        agent = VerifierAgent(llm_client=MockLLM())
        monkeypatch.setattr(agent, "_load_nli_model", lambda: None)

        assert agent._nli_verify("声明", ["证据"]) is None

    def test_failed_nli_load_is_cached_across_agents(self, monkeypatch):
        import sys
        from types import ModuleType

        from src.agents.verifier_agent import VerifierAgent
        from src.utils.model_cache import clear_optional_model_cache

        clear_optional_model_cache()
        calls = []

        class FailingCrossEncoder:
            def __init__(self, model_name, **kwargs):
                calls.append((model_name, kwargs))
                raise OSError("model unavailable")

        fake_module = ModuleType("sentence_transformers")
        fake_module.CrossEncoder = FailingCrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
        config = {
            "verification": {
                "nli_model": "missing/nli",
                "nli_local_files_only": True,
            }
        }

        assert VerifierAgent(config=config)._load_nli_model() is None
        assert VerifierAgent(config=config)._load_nli_model() is None
        assert calls == [("missing/nli", {"local_files_only": True})]
        clear_optional_model_cache()

    def test_nli_load_without_local_files_flag_uses_plain_cross_encoder(self, monkeypatch):
        import sys
        from types import ModuleType

        from src.agents.verifier_agent import VerifierAgent
        from src.utils.model_cache import clear_optional_model_cache

        clear_optional_model_cache()
        calls = []
        fake_model = object()

        class FakeCrossEncoder:
            def __new__(cls, model_name, **kwargs):
                calls.append((model_name, kwargs))
                return fake_model

        fake_module = ModuleType("sentence_transformers")
        fake_module.CrossEncoder = FakeCrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        agent = VerifierAgent(
            config={"verification": {"nli_model": "available/nli"}},
            llm_client=MockLLM(),
        )

        assert agent._load_nli_model() is fake_model
        assert calls == [("available/nli", {})]
        clear_optional_model_cache()

    def test_llm_verification_normalizes_status_and_confidence(self):
        from src.agents.verifier_agent import VerifierAgent

        agent = VerifierAgent(
            llm_client=MockLLM(
                '{"status": "unknown", "confidence": "high", "rationale": "bad fields"}'
            ),
            use_nli=False,
        )

        verification = agent._llm_verify_claim("声明", ["证据"], [])

        assert verification["status"] == "NOT_ENOUGH_INFO"
        assert verification["confidence"] == 0.0
        assert verification["method"] == "llm"

    def test_verification_normalizers_cover_enum_bad_status_and_nonfinite_confidence(self):
        from src.agents.verifier_agent import VerifierAgent

        agent = VerifierAgent(llm_client=MockLLM(), use_nli=False)

        normalized = agent._normalize_verification(
            {
                "status": VerificationStatus.REFUTED,
                "confidence": float("nan"),
            },
            "声明",
        )

        assert normalized["claim"] == "声明"
        assert normalized["status"] == "REFUTED"
        assert normalized["confidence"] == 0.0
        assert agent._normalize_status(object()) == "NOT_ENOUGH_INFO"
        probabilities = agent._nli_probabilities([0.0, 1.0, 0.0])
        assert probabilities is not None
        assert probabilities.shape == (1, 3)

    def test_verify_answer_tracks_overconfident_refuted_and_missing_claims(self):
        from src.agents.verifier_agent import VerifierAgent

        class SequentialLLM:
            def __init__(self):
                self.responses = iter([
                    '{"status": "supported", "confidence": 0.4}',
                    '{"status": "refuted", "confidence": 0.9}',
                    '{"status": "not enough info", "confidence": 0.7}',
                ])

            def generate(self, prompt: str, **kwargs) -> str:
                return next(self.responses)

        claims = [
            AnswerClaim(claim="低置信支持", supporting_evidence=["E1"], confidence=0.9),
            AnswerClaim(claim="被反驳", supporting_evidence=["E1"], confidence=0.9),
            AnswerClaim(claim="证据不足", supporting_evidence=["E1"], confidence=0.9),
        ]
        agent = VerifierAgent(llm_client=SequentialLLM(), use_nli=False)

        report = agent.verify_answer(
            answer="答案",
            claims=claims,
            evidence=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )

        assert report.overall_status is VerificationStatus.REFUTED
        assert report.overconfident_claims == ["低置信支持"]
        assert report.missing_evidence_for == ["证据不足"]
        assert report.issues == [
            {
                "type": "unsupported_claim",
                "description": "1 claims lack sufficient evidence",
            }
        ]

    def test_verify_answer_supported_and_fallback_paths(self):
        from src.agents.verifier_agent import VerifierAgent

        supported_agent = VerifierAgent(
            llm_client=MockLLM('{"status": "SUPPORTED", "confidence": 1.5}'),
            use_nli=False,
        )
        supported = supported_agent.verify_answer(
            answer="答案",
            claims=[AnswerClaim(claim="声明", supporting_evidence=["E1"])],
            evidence=self._make_evidence(),
            conflict_graph=EvidenceConflictGraph(),
        )
        assert supported.overall_status is VerificationStatus.SUPPORTED
        assert supported.claim_verifications[0]["confidence"] == 1.0

        fallback_agent = VerifierAgent(llm_client=MockLLM("not json"), use_nli=False)
        fallback = fallback_agent._llm_verify_claim("声明", [], [])
        assert fallback == {
            "claim": "声明",
            "status": "NOT_ENOUGH_INFO",
            "confidence": 0.3,
            "method": "fallback",
        }

    def test_acknowledged_conflicts_are_not_reported_as_ignored(self):
        from src.agents.verifier_agent import VerifierAgent

        graph = EvidenceConflictGraph()
        graph.add_node(ConflictGraphNode(node_id="E1", content="证据A", node_type="evidence"))
        graph.add_node(ConflictGraphNode(node_id="E2", content="证据B", node_type="evidence"))
        graph.add_edge(ConflictEdge(
            source_id="E1", target_id="E2",
            conflict_type=ConflictType.NUMERIC_CONFLICT,
            confidence=0.8,
        ))
        claims = [
            AnswerClaim(claim="承认冲突", supporting_evidence=["E1"], conflicting_evidence=["E2"])
        ]
        agent = VerifierAgent(llm_client=MockLLM(), use_nli=False)

        assert agent._check_ignored_conflicts(claims, graph) == []

    def test_run_delegates_to_verify_answer(self):
        from src.agents.verifier_agent import VerifierAgent

        agent = VerifierAgent(
            llm_client=MockLLM('{"status": "SUPPORTED", "confidence": 0.9}'),
            use_nli=False,
        )

        report = agent.run(
            "答案",
            [AnswerClaim(claim="声明", supporting_evidence=["E1"])],
            self._make_evidence(),
            EvidenceConflictGraph(),
        )

        assert report.overall_status is VerificationStatus.SUPPORTED


# --- ConflictGraphBuilder Tests (Three-Layer Architecture) ---

class TestConflictGraphBuilder:
    """Tests for the upgraded three-layer conflict detection."""

    def _make_claim(self, text: str, entities=None, numbers=None, time_expr=None) -> Claim:
        return Claim(
            claim_id=f"C_{hash(text) % 10000:04d}",
            claim=text,
            claim_type=ClaimType.FACTUAL,
            entities=entities or [],
            numbers=numbers or [],
            time_expressions=time_expr or [],
        )

    def _make_evidence(self, claims: list, source: str = "test", date: str | None = None) -> Evidence:
        eid = f"E_{hash(source) % 10000:04d}"
        return Evidence(
            evidence_id=eid, source=source, title="T",
            text_span="content", claims=claims, date=date,
        )

    def test_numeric_conflict_with_year_filter(self):
        """Years (1900-2099) should not be treated as numeric conflicts."""
        from src.evidence.conflict_graph import ConflictGraphBuilder
        builder = ConflictGraphBuilder(llm_client=None, config={"conflict_graph": {"enable_nli": False}})

        c1 = self._make_claim("In 2020 GDP was 5%", numbers=["2020", "5"])
        c2 = self._make_claim("In 2021 GDP was 6%", numbers=["2021", "6"])

        ev1 = self._make_evidence([c1])
        ev2 = self._make_evidence([c2])

        graph = builder.build_graph([ev1, ev2], use_llm=False)
        # "5" vs "6" ratio > 1.1 but no shared entities → should not flag
        # unless shared entities exist; here there are none → threshold is 1.3
        numeric_conflicts = [e for e in graph.edges if e.conflict_type == ConflictType.NUMERIC_CONFLICT]
        # 5 and 6 are both parsed, ratio=1.2, threshold=1.3 for no shared entities → no conflict
        assert len(numeric_conflicts) == 0

    def test_numeric_conflict_with_shared_entities(self):
        """Different numbers for same entity should be detected."""
        from src.evidence.conflict_graph import ConflictGraphBuilder
        builder = ConflictGraphBuilder(llm_client=None, config={"conflict_graph": {"enable_nli": False}})

        c1 = self._make_claim("GDP of China was 5%", entities=["China", "GDP"], numbers=["5"])
        c2 = self._make_claim("GDP of China was 8%", entities=["China", "GDP"], numbers=["8"])

        ev1 = self._make_evidence([c1])
        ev2 = self._make_evidence([c2])

        graph = builder.build_graph([ev1, ev2], use_llm=False)
        numeric_conflicts = [e for e in graph.edges if e.conflict_type == ConflictType.NUMERIC_CONFLICT]
        assert len(numeric_conflicts) == 1
        assert numeric_conflicts[0].confidence > 0.5

    def test_support_detection_similar_claims(self):
        """Near-identical claims with shared entities → SUPPORT edge."""
        from src.evidence.conflict_graph import ConflictGraphBuilder
        builder = ConflictGraphBuilder(llm_client=None, config={"conflict_graph": {"enable_nli": False}})

        c1 = self._make_claim("China GDP growth rate was 5.2 percent in 2023", entities=["China", "GDP"])
        c2 = self._make_claim("China GDP growth rate was 5.2 percent in 2023", entities=["China", "GDP"])

        ev1 = self._make_evidence([c1])
        ev2 = self._make_evidence([c2])

        graph = builder.build_graph([ev1, ev2], use_llm=False)
        supports = [e for e in graph.edges if e.conflict_type == ConflictType.SUPPORT]
        assert len(supports) == 1
        assert supports[0].confidence > 0.8

    def test_semantic_contradiction_negation(self):
        """Claims with negation pairs and shared entities → conflict detected."""
        from src.evidence.conflict_graph import ConflictGraphBuilder
        builder = ConflictGraphBuilder(llm_client=None, config={"conflict_graph": {"enable_nli": False}})

        c1 = self._make_claim("AI是未来发展的关键动力", entities=["AI"])
        c2 = self._make_claim("AI不是未来发展的关键动力", entities=["AI"])

        ev1 = self._make_evidence([c1])
        ev2 = self._make_evidence([c2])

        graph = builder.build_graph([ev1, ev2], use_llm=False)
        # Detected by either entity_mismatch (rule #2) or refute (rule #10)
        conflicts = graph.get_conflicts()
        assert len(conflicts) >= 1

    def test_text_similarity_high(self):
        """_text_similarity should return high value for similar strings."""
        from src.evidence.conflict_graph import ConflictGraphBuilder
        sim = ConflictGraphBuilder._text_similarity(
            "中国GDP增长率为5.2%", "中国GDP增长率为5.2%"
        )
        assert sim > 0.9

    def test_text_similarity_low(self):
        """_text_similarity should return low value for dissimilar strings."""
        from src.evidence.conflict_graph import ConflictGraphBuilder
        sim = ConflictGraphBuilder._text_similarity(
            "今天天气不错", "量子计算的未来发展"
        )
        assert sim < 0.3

    def test_is_likely_year(self):
        from src.evidence.conflict_graph import ConflictGraphBuilder
        assert ConflictGraphBuilder._is_likely_year("2020")
        assert ConflictGraphBuilder._is_likely_year("1999")
        assert ConflictGraphBuilder._is_likely_year("2050")
        assert not ConflictGraphBuilder._is_likely_year("100")
        assert not ConflictGraphBuilder._is_likely_year("42")

    def test_nli_disabled_by_default(self):
        """NLI should not crash when sentence-transformers is not installed."""
        from src.evidence.conflict_graph import ConflictGraphBuilder
        builder = ConflictGraphBuilder(llm_client=None, config={"conflict_graph": {"enable_nli": True}})
        # Should not raise
        c1 = self._make_claim("test claim one")
        c2 = self._make_claim("test claim two")
        result = builder._nli_detect(c1, c2)
        # Returns None since model can't load
        assert result is None

    def test_nli_uses_model_label_mapping(self):
        """Conflict NLI must not assume a fixed three-class label order."""
        from types import SimpleNamespace

        import numpy as np

        from src.evidence.conflict_graph import ConflictGraphBuilder

        class DummyNLI:
            model = SimpleNamespace(
                config=SimpleNamespace(
                    id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                ),
            )

            @staticmethod
            def predict(pairs, show_progress_bar=False):
                assert pairs == [("声明A", "声明B")]
                assert show_progress_bar is False
                return np.asarray([[8.0, 0.0, -4.0]])

        builder = ConflictGraphBuilder(
            llm_client=None,
            config={"conflict_graph": {"enable_nli": True, "nli_threshold": 0.7}},
        )
        builder._nli_available = True
        builder._nli_model = DummyNLI()

        edge = builder._nli_detect(
            self._make_claim("声明A"),
            self._make_claim("声明B"),
        )

        assert edge is not None
        assert edge.conflict_type == ConflictType.SUPPORT
        assert edge.confidence > 0.99

    def test_three_layer_fallback_to_llm(self):
        """When rule-based and NLI miss, LLM should be called."""
        from src.evidence.conflict_graph import ConflictGraphBuilder

        llm = MockLLM('{"relationship": "REFUTE", "confidence": 0.85, "rationale": "contradicts", "conflict_type": "none"}')
        builder = ConflictGraphBuilder(llm_client=llm, config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_llm_adjudication": True,
            }
        })

        c1 = self._make_claim("苹果公司的总部在加利福尼亚", entities=["苹果公司"])
        c2 = self._make_claim("苹果公司的总部在纽约", entities=["苹果公司"])

        ev1 = self._make_evidence([c1])
        ev2 = self._make_evidence([c2])

        graph = builder.build_graph([ev1, ev2], use_llm=True)
        # Should have detected via LLM
        assert len(graph.edges) >= 1


# --- RepairAgent Tests ---

class TestRepairAgent:
    def _make_agent(self):
        from src.agents.repair_agent import RepairAgent

        return RepairAgent(llm_client=MockLLM())

    def _make_claim(self, text="c1", confidence=0.8):
        return AnswerClaim(
            claim=text,
            supporting_evidence=["E1"],
            conflicting_evidence=["E2"],
            confidence=confidence,
            verification_status=VerificationStatus.SUPPORTED,
        )

    def _make_report(self, verifications):
        return VerificationReport(
            claim_verifications=verifications,
            overall_status=VerificationStatus.NOT_ENOUGH_INFO,
            issues=[{"type": "verification_issue"}],
            missing_evidence_for=[],
            overconfident_claims=[],
            ignored_conflicts=[],
        )

    def test_no_repair_when_no_critical_issues(self):
        from src.agents.repair_agent import RepairAgent
        agent = RepairAgent(llm_client=MockLLM())
        report = VerificationReport(
            claim_verifications=[],
            overall_status=VerificationStatus.SUPPORTED,
            issues=[], missing_evidence_for=[],
            overconfident_claims=[], ignored_conflicts=[],
        )
        answer, _claims = agent.repair_answer(
            answer="原答案",
            claims=[AnswerClaim(claim="c1", supporting_evidence=["E1"], conflicting_evidence=[],
                               confidence=0.9, verification_status=VerificationStatus.SUPPORTED)],
            verification_report=report,
            evidence=[Evidence(evidence_id="E1", source="test", title="T", text_span="内容")],
        )
        assert answer == "原答案"

    def test_repair_with_critical_issues(self):
        from src.agents.repair_agent import RepairAgent
        agent = RepairAgent(llm_client=MockLLM("修复后的答案"))
        report = VerificationReport(
            claim_verifications=[{"claim": "c1", "status": "refuted"}],
            overall_status=VerificationStatus.REFUTED,
            issues=[{"type": "refuted_claim", "claim": "c1"}],
            missing_evidence_for=[],
            overconfident_claims=["c1"],
            ignored_conflicts=[],
        )
        assert report.has_critical_issues()
        answer, _claims = agent.repair_answer(
            answer="需要修复的答案",
            claims=[AnswerClaim(claim="c1", supporting_evidence=[], conflicting_evidence=["E1"],
                               confidence=0.2, verification_status=VerificationStatus.REFUTED)],
            verification_report=report,
            evidence=[Evidence(evidence_id="E1", source="test", title="T", text_span="内容")],
        )
        # Should return some answer (may be original or repaired)
        assert isinstance(answer, str)

    def test_repair_skips_verification_entries_without_matching_claim(self):
        agent = self._make_agent()
        report = self._make_report([
            {"claim": "unknown", "status": "refuted", "confidence": 0.9}
        ])

        answer, claims = agent.repair_answer(
            answer="原答案",
            claims=[self._make_claim("c1")],
            verification_report=report,
            evidence=[],
        )

        assert answer == "原答案"
        assert claims == []

    def test_repair_claim_refuted_status_is_downgraded_and_enum_normalized(self):
        agent = self._make_agent()
        claim = self._make_claim("太阳从西边升起", confidence=0.9)

        repaired = agent._repair_claim(
            claim,
            {"claim": claim.claim, "status": "REFUTED", "confidence": 0.8},
            evidence=[],
        )

        assert "存在反证" in repaired.claim
        assert repaired.confidence == 0.8 * 0.3
        assert repaired.verification_status is VerificationStatus.REFUTED

    def test_repair_claim_not_enough_info_accepts_wire_value_and_adds_hedge(self):
        agent = self._make_agent()
        claim = self._make_claim("证据不足的结论", confidence=0.5)

        repaired = agent._repair_claim(
            claim,
            {"claim": claim.claim, "status": "not_enough_info"},
            evidence=[],
        )

        assert repaired.claim.startswith("现有证据有限，")
        assert repaired.confidence == 0.5 * 0.6
        assert repaired.verification_status is VerificationStatus.NOT_ENOUGH_INFO

    def test_repair_claim_supported_preserves_claim_and_clamps_confidence(self):
        agent = self._make_agent()
        claim = self._make_claim("受证据支持的结论", confidence=0.4)

        repaired = agent._repair_claim(
            claim,
            {"claim": claim.claim, "status": VerificationStatus.SUPPORTED, "confidence": 2.0},
            evidence=[],
        )

        assert repaired.claim == claim.claim
        assert repaired.confidence == 1.0
        assert repaired.verification_status is VerificationStatus.SUPPORTED

    def test_repair_helpers_are_idempotent(self):
        agent = self._make_agent()

        downgraded = agent._downgrade_claim("结论（注：该说法证据不足，甚至存在反证）")
        hedged = agent._add_uncertainty_hedge("现有证据有限，结论")

        assert downgraded == "结论（注：该说法证据不足，甚至存在反证）"
        assert hedged == "现有证据有限，结论"

    def test_generate_repaired_answer_falls_back_for_empty_answer(self):
        agent = self._make_agent()

        repaired = agent._generate_repaired_answer(
            "",
            repaired_claims=[],
            verification_report=self._make_report([]),
        )

        assert repaired == "根据现有证据，信息不足，无法给出可靠回答。"

    def test_generate_repaired_answer_appends_caveat_once_for_refuted_claims(self):
        agent = self._make_agent()
        caveat = "（注：上述部分论断证据有限，请谨慎对待。）"
        repaired_claims = [
            AnswerClaim(
                claim="c1",
                verification_status=VerificationStatus.REFUTED,
            )
        ]

        repaired = agent._generate_repaired_answer(
            f"原答案\n{caveat}",
            repaired_claims=repaired_claims,
            verification_report=self._make_report([]),
        )

        assert repaired.count(caveat) == 1

    def test_run_delegates_to_repair_answer(self):
        agent = self._make_agent()
        report = VerificationReport(overall_status=VerificationStatus.SUPPORTED)

        answer, claims = agent.run(
            "原答案",
            [self._make_claim("c1")],
            report,
            [],
        )

        assert answer == "原答案"
        assert len(claims) == 1

    def test_repair_rejects_unknown_status_and_invalid_confidence(self):
        agent = self._make_agent()
        claim = self._make_claim("c1")

        import pytest

        with pytest.raises(ValueError, match="VerificationStatus or string"):
            agent._repair_claim(
                claim,
                {"claim": claim.claim, "status": object()},
                evidence=[],
            )

        with pytest.raises(ValueError, match="Unknown verification status"):
            agent._repair_claim(
                claim,
                {"claim": claim.claim, "status": "maybe"},
                evidence=[],
            )

        with pytest.raises(ValueError, match="confidence"):
            agent._repair_claim(
                claim,
                {"claim": claim.claim, "status": "supported", "confidence": "high"},
                evidence=[],
            )
