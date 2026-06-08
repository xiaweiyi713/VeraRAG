"""Tests for enhanced conflict detection (8 conflict types)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from src.evidence.conflict_graph import ConflictGraphBuilder
from src.utils.data_structures import (
    Claim,
    ClaimType,
    ConflictEdge,
    ConflictType,
    Evidence,
)


def _make_claim(claim_id, text, entities=None, numbers=None, time_expressions=None):
    return Claim(
        claim_id=claim_id,
        claim=text,
        claim_type=ClaimType.FACTUAL,
        entities=entities or [],
        numbers=numbers or [],
        time_expressions=time_expressions or [],
    )


def _make_evidence(evidence_id, source="report", date=None):
    return Evidence(
        evidence_id=evidence_id,
        source=source,
        title="Test",
        text_span="",
        date=date,
    )


class TestNumericConflict(unittest.TestCase):
    def test_significant_numeric_difference(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "营收500亿元", numbers=["500"])
        c2 = _make_claim("C2", "营收800亿元", numbers=["800"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.NUMERIC_CONFLICT)
        self.assertEqual(edge.resolver_strategy, "flag_for_verification")

    def test_close_numbers_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "增长23%", numbers=["23"])
        c2 = _make_claim("C2", "增长24%", numbers=["24"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_severity_high_for_large_gap(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "人数500", numbers=["500"])
        c2 = _make_claim("C2", "人数60000", numbers=["60000"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.severity, "high")


class TestTemporalConflict(unittest.TestCase):
    def test_different_dates_same_entity(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "公司成立于2012年", entities=["StarTech"], time_expressions=["2012年"])
        c2 = _make_claim("C2", "公司成立于2010年", entities=["StarTech"], time_expressions=["2010年"])
        e1 = _make_evidence("E1", date="2012-01-01")
        e2 = _make_evidence("E2", date="2010-01-01")
        edge = builder._check_temporal_conflict(c1, c2, e1, e2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.TEMPORAL_CONFLICT)
        self.assertEqual(edge.resolver_strategy, "prefer_newer")

    def test_same_date_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "营收增长", entities=["StarTech"], time_expressions=["2023"])
        c2 = _make_claim("C2", "利润增长", entities=["StarTech"], time_expressions=["2023"])
        e1 = _make_evidence("E1", date="2024-01-01")
        e2 = _make_evidence("E2", date="2024-01-01")
        edge = builder._check_temporal_conflict(c1, c2, e1, e2)
        self.assertIsNone(edge)


class TestScopeConflict(unittest.TestCase):
    def test_global_vs_regional(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "全球AI市场规模达到5880亿美元", entities=["AI市场"])
        c2 = _make_claim("C2", "中国AI市场规模达到1200亿元", entities=["AI市场"])
        edge = builder._check_scope_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.SCOPE_CONFLICT)
        self.assertEqual(edge.resolver_strategy, "prefer_narrower")

    def test_same_scope_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "中国新能源汽车销量增长", entities=["新能源汽车"])
        c2 = _make_claim("C2", "中国动力电池装机量增长", entities=["动力电池"])
        edge = builder._check_scope_conflict(c1, c2)
        self.assertIsNone(edge)


class TestCausalConflict(unittest.TestCase):
    def test_causal_vs_no_causal(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "碳排放增加导致全球变暖", entities=["碳排放", "全球变暖"])
        c2 = _make_claim("C2", "碳排放与全球变暖无关", entities=["碳排放", "全球变暖"])
        edge = builder._check_causal_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.CAUSAL_CONFLICT)
        self.assertEqual(edge.severity, "high")

    def test_both_causal_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "AI发展推动了芯片需求", entities=["AI", "芯片"])
        c2 = _make_claim("C2", "大模型训练导致算力需求激增", entities=["大模型", "算力"])
        edge = builder._check_causal_conflict(c1, c2)
        self.assertIsNone(edge)


class TestGranularityConflict(unittest.TestCase):
    def test_quarterly_vs_annual(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "Q1营收178亿元", entities=["StarTech"])
        c2 = _make_claim("C2", "年度营收612亿元", entities=["StarTech"])
        edge = builder._check_granularity_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.GRANULARITY_CONFLICT)
        self.assertEqual(edge.severity, "low")

    def test_same_granularity_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "2023年营收612亿元", entities=["StarTech"])
        c2 = _make_claim("C2", "2024年营收预计增长", entities=["StarTech"])
        edge = builder._check_granularity_conflict(c1, c2)
        self.assertIsNone(edge)


class TestDefinitionalConflict(unittest.TestCase):
    def test_different_definitions(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "量子霸权是指在特定任务上超越经典计算机", entities=["量子霸权"])
        c2 = _make_claim("C2", "量子霸权指的是量子计算机全面取代经典计算机", entities=["量子霸权"])
        edge = builder._check_definitional_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.DEFINITIONAL_CONFLICT)

    def test_same_definition_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "RAG是指检索增强生成", entities=["RAG"])
        c2 = _make_claim("C2", "RAG技术发展迅速", entities=["RAG"])
        edge = builder._check_definitional_conflict(c1, c2)
        self.assertIsNone(edge)


class TestSourceReliabilityConflict(unittest.TestCase):
    def test_official_vs_blog_contradiction(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "营收达到612亿元（官方）", entities=["StarTech"])
        c2 = _make_claim("C2", "营收并非612亿元，实际为800亿", entities=["StarTech"])
        e1 = _make_evidence("E1", source="official")
        e2 = _make_evidence("E2", source="blog")
        edge = builder._check_source_reliability_conflict(c1, e1, c2, e2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.SOURCE_DISAGREEMENT)

    def test_similar_source_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "营收增长33%", entities=["StarTech"])
        c2 = _make_claim("C2", "利润增长50%", entities=["StarTech"])
        e1 = _make_evidence("E1", source="report")
        e2 = _make_evidence("E2", source="report")
        edge = builder._check_source_reliability_conflict(c1, e1, c2, e2)
        self.assertIsNone(edge)


class TestEntityConflictEnhanced(unittest.TestCase):
    def test_chinese_negation(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "该系统有自动化功能", entities=["系统"])
        c2 = _make_claim("C2", "该系统没有自动化功能", entities=["系统"])
        edge = builder._check_entity_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.ENTITY_MISMATCH)
        self.assertEqual(edge.severity, "high")

    def test_same_entity_values_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "成立于2012年", entities=["StarTech", "2012年"])
        c2 = _make_claim("C2", "创立于2012年", entities=["StarTech", "2012年"])
        edge = builder._check_entity_conflict(c1, c2)
        self.assertIsNone(edge)


class TestConflictEdgeAttributes(unittest.TestCase):
    def test_edge_has_severity(self):
        edge = ConflictEdge(
            source_id="C1", target_id="C2",
            conflict_type=ConflictType.NUMERIC_CONFLICT,
            confidence=0.8,
            severity="high",
            resolver_strategy="flag_for_verification",
        )
        self.assertEqual(edge.severity, "high")
        self.assertEqual(edge.resolver_strategy, "flag_for_verification")

    def test_edge_to_dict_includes_new_fields(self):
        edge = ConflictEdge(
            source_id="C1", target_id="C2",
            conflict_type=ConflictType.SCOPE_CONFLICT,
            confidence=0.6,
            severity="medium",
            rationale="scope mismatch",
            resolver_strategy="prefer_narrower",
        )
        d = edge.to_dict()
        self.assertIn("severity", d)
        self.assertIn("resolver_strategy", d)
        self.assertEqual(d["severity"], "medium")


if __name__ == "__main__":
    unittest.main()
