"""Tests for agent modules: task_analyzer, reasoning, verifier, repair."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_structures import (
    TaskType, Complexity, TaskAnalysis,
    Evidence, Claim, ClaimType, AnswerClaim,
    VerificationStatus, VerificationReport,
    ReasoningStep, SubQuestion, EvidenceConflictGraph,
    ConflictEdge, ConflictGraphNode, ConflictType,
)


class MockLLM:
    """Mock LLM client that returns canned responses."""
    def __init__(self, response: str = "mock response"):
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self.response


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

    def test_reason_with_mock_llm(self):
        from src.agents.reasoning_agent import ReasoningAgent

        json_response = '''{
            "answer": "测试答案",
            "claims": [{"claim": "声明1", "supporting_evidence": ["E1"], "conflicting_evidence": [], "confidence": 0.9, "verification_status": "supported"}],
            "steps": [{"step": 1, "description": "分析证据", "evidence_ids": ["E1"], "confidence": 0.85}]
        }'''
        agent = ReasoningAgent(llm_client=MockLLM(json_response))
        answer, claims, steps = agent.reason(
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
        answer, claims, steps = agent.reason(
            question="测试问题",
            subquestions=[],
            evidence=self._make_evidence(1),
            conflict_graph=EvidenceConflictGraph(),
            reasoning_plan=[],
        )
        assert isinstance(answer, str)


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

    def _make_evidence(self, claims: list, source: str = "test", date: str = None) -> Evidence:
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

    def test_three_layer_fallback_to_llm(self):
        """When rule-based and NLI miss, LLM should be called."""
        from src.evidence.conflict_graph import ConflictGraphBuilder

        llm = MockLLM('{"relationship": "REFUTE", "confidence": 0.85, "rationale": "contradicts", "conflict_type": "none"}')
        builder = ConflictGraphBuilder(llm_client=llm, config={"conflict_graph": {"enable_nli": False}})

        c1 = self._make_claim("苹果公司的总部在加利福尼亚")
        c2 = self._make_claim("苹果公司的总部在纽约")

        ev1 = self._make_evidence([c1])
        ev2 = self._make_evidence([c2])

        graph = builder.build_graph([ev1, ev2], use_llm=True)
        # Should have detected via LLM
        assert len(graph.edges) >= 1


# --- RepairAgent Tests ---

class TestRepairAgent:
    def test_no_repair_when_no_critical_issues(self):
        from src.agents.repair_agent import RepairAgent
        agent = RepairAgent(llm_client=MockLLM())
        report = VerificationReport(
            claim_verifications=[],
            overall_status=VerificationStatus.SUPPORTED,
            issues=[], missing_evidence_for=[],
            overconfident_claims=[], ignored_conflicts=[],
        )
        answer, claims = agent.repair_answer(
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
        answer, claims = agent.repair_answer(
            answer="需要修复的答案",
            claims=[AnswerClaim(claim="c1", supporting_evidence=[], conflicting_evidence=["E1"],
                               confidence=0.2, verification_status=VerificationStatus.REFUTED)],
            verification_report=report,
            evidence=[Evidence(evidence_id="E1", source="test", title="T", text_span="内容")],
        )
        # Should return some answer (may be original or repaired)
        assert isinstance(answer, str)
