"""Tests for enhanced conflict detection (8 conflict types)."""

import os
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
        c1 = _make_claim("C1", "星辰科技营收500亿元", entities=["星辰科技"], numbers=["500"])
        c2 = _make_claim("C2", "星辰科技营收800亿元", entities=["星辰科技"], numbers=["800"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.NUMERIC_CONFLICT)
        self.assertEqual(edge.resolver_strategy, "flag_for_verification")

    def test_close_numbers_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技增长23%", entities=["星辰科技"], numbers=["23"])
        c2 = _make_claim("C2", "星辰科技增长24%", entities=["星辰科技"], numbers=["24"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_severity_high_for_large_gap(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技人数500", entities=["星辰科技"], numbers=["500"])
        c2 = _make_claim("C2", "星辰科技人数60000", entities=["星辰科技"], numbers=["60000"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.severity, "high")

    def test_unrelated_numbers_do_not_conflict_without_shared_fact_slot(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技成立于2012年", entities=["星辰科技"], numbers=["2012"])
        c2 = _make_claim("C2", "星辰科技员工超过60000人", entities=["星辰科技"], numbers=["60000"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_date_components_do_not_create_numeric_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "欧盟AI法案已于2024年3月通过",
            entities=["欧盟AI法案"],
            numbers=["2024年", "3月"],
        )
        c2 = _make_claim(
            "C2",
            "欧盟AI法案于2024年3月13日正式通过",
            entities=["欧盟AI法案"],
            numbers=["2024年", "3月", "13日"],
        )
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_date_range_component_does_not_conflict_with_quantity(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "2024年全球新能源汽车销量达到约2100万辆",
            entities=["新能源汽车"],
            numbers=["2024年", "2100万"],
        )
        c2 = _make_claim(
            "C2",
            "2024年1-9月，中国新能源汽车累计销量约728万辆",
            entities=["新能源汽车"],
            numbers=["2024年", "1", "9月", "728万"],
        )

        edge = builder._check_numerical_conflict(c1, c2)

        self.assertIsNone(edge)

    def test_qubit_count_does_not_conflict_with_runtime(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "谷歌使用53量子比特处理器完成采样任务",
            entities=["谷歌", "量子霸权"],
            numbers=["53量子比特"],
        )
        c2 = _make_claim(
            "C2",
            "IBM认为经典算法约2.5天可完成同一任务",
            entities=["谷歌", "量子霸权", "IBM"],
            numbers=["2.5天"],
        )

        edge = builder._check_numerical_conflict(c1, c2)

        self.assertIsNone(edge)

    def test_unit_values_still_create_numeric_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技营收500亿元", entities=["星辰科技"], numbers=["500亿元"])
        c2 = _make_claim("C2", "星辰科技营收800亿元", entities=["星辰科技"], numbers=["800亿元"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.NUMERIC_CONFLICT)

    def test_process_node_label_does_not_conflict_with_physical_length(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            '"3nm"和"5nm"等命名已不再代表实际的物理栅极长度',
            entities=["芯片制程"],
            numbers=["3nm", "5nm"],
        )
        c2 = _make_claim(
            "C2",
            '台积电的"3nm"工艺的实际栅极长度约为20nm以上',
            entities=["芯片制程"],
            numbers=["3nm", "20nm"],
        )

        edge = builder._check_numerical_conflict(c1, c2)

        self.assertIsNone(edge)

    def test_different_process_node_labels_are_not_numeric_conflicts(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "台积电在先进制程7nm及以下代工市场占比超过90%", entities=["台积电"], numbers=["7nm", "90%"])
        c2 = _make_claim("C2", "三星和英特尔正在追赶3nm及以下制程", entities=["台积电"], numbers=["3nm"])

        edge = builder._check_numerical_conflict(c1, c2)

        self.assertIsNone(edge)

    def test_climate_sensitivity_does_not_conflict_with_temperature_target(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "我们估计ECS可能为2.5°C",
            entities=["ECS", "气候敏感度"],
            numbers=["2.5°C"],
        )
        c2 = _make_claim(
            "C2",
            "以较低的气候敏感度计算，要实现1.5°C目标仍需大幅减排",
            entities=["ECS", "气候敏感度"],
            numbers=["1.5°C"],
        )
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_total_sales_do_not_conflict_with_export_sales(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "比亚迪在全球销售了约302.4万辆新能源汽车",
            entities=["比亚迪"],
            numbers=["302.4万"],
        )
        c2 = _make_claim(
            "C2",
            "在海外市场方面，比亚迪2023年出口约24.3万辆",
            entities=["比亚迪"],
            numbers=["24.3万"],
        )
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_emissions_components_do_not_conflict_with_total(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "全球化石燃料CO2排放为368亿吨",
            entities=["CO2"],
            numbers=["368亿吨"],
        )
        c2 = _make_claim(
            "C2",
            "土地利用变化相关排放为109亿吨",
            entities=["CO2"],
            numbers=["109亿吨"],
        )
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_manufacturer_sales_do_not_conflict_with_total_sales(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "2024年全球新能源汽车销量达到2100万辆",
            entities=["新能源汽车"],
            numbers=["2100万辆"],
        )
        c2 = _make_claim(
            "C2",
            "比亚迪是全球最大新能源汽车制造商，2024年销量约430万辆",
            entities=["新能源汽车"],
            numbers=["430万辆"],
        )
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_regional_sales_do_not_conflict_with_global_sales(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "2024年全球新能源汽车销量达到约2100万辆",
            entities=["新能源汽车"],
            numbers=["2100万"],
        )
        c2 = _make_claim(
            "C2",
            "2024年中国新能源汽车累计销量约728万辆",
            entities=["新能源汽车"],
            numbers=["728万"],
        )

        edge = builder._check_numerical_conflict(c1, c2)

        self.assertIsNone(edge)

    def test_decimal_chinese_magnitude_units_are_scaled(self):
        builder = ConflictGraphBuilder()

        numbers = builder._parse_numbers(["302.4万", "181万", "1.5亿"])

        self.assertEqual(numbers, [3024000.0, 1810000.0, 150000000.0])

    def test_invalid_number_token_does_not_shift_raw_unit_alignment(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "星辰科技营收500亿元",
            entities=["星辰科技"],
            numbers=["unknown", "500亿元"],
        )
        c2 = _make_claim(
            "C2",
            "星辰科技营收800亿元",
            entities=["星辰科技"],
            numbers=["800亿元"],
        )

        edge = builder._check_numerical_conflict(c1, c2)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.NUMERIC_CONFLICT)

    def test_non_string_number_tokens_are_handled_without_crashing(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "星辰科技2024年营收500亿元",
            entities=["星辰科技"],
            numbers=[2024, "500亿元"],
        )
        c2 = _make_claim(
            "C2",
            "星辰科技2024年营收800亿元",
            entities=["星辰科技"],
            numbers=["2024年", "800亿元"],
        )

        edge = builder._check_numerical_conflict(c1, c2)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.NUMERIC_CONFLICT)

    def test_same_evidence_comparison_title_allows_numeric_edge(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "compare_within_evidence": False,
            }
        })
        evidence = Evidence(
            evidence_id="E1",
            source="report",
            title="比亚迪vs特斯拉：2023年全球销量对比",
            text_span="2023年，比亚迪销量302.4万辆。特斯拉交付181万辆。",
            claims=[
                _make_claim(
                    "C1",
                    "2023年，比亚迪销量302.4万辆",
                    entities=["比亚迪", "特斯拉", "销量"],
                    numbers=["2023年", "302.4万"],
                ),
                _make_claim(
                    "C2",
                    "特斯拉交付181万辆",
                    entities=["比亚迪", "特斯拉", "销量"],
                    numbers=["181万"],
                ),
            ],
        )

        graph = builder.build_graph([evidence], use_llm=False)

        self.assertTrue(graph.get_conflicts())

    def test_percent_and_absolute_values_are_different_numeric_slots(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "比亚迪销量302.4万辆，同比增长61.9%",
            entities=["比亚迪"],
            numbers=["302.4万", "61.9%"],
        )
        c2 = _make_claim(
            "C2",
            "比亚迪年销量302.4万辆",
            entities=["比亚迪"],
            numbers=["302.4万"],
        )
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_revenue_from_different_years_is_different_fact_slot(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技2022财年全年营收为458亿元", entities=["星辰科技"], numbers=["2022", "458亿元"])
        c2 = _make_claim("C2", "星辰科技2023财年全年营收达到612亿元", entities=["星辰科技"], numbers=["2023", "612亿元"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_inherited_time_expressions_define_revenue_fact_slot(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "云服务业务营收185亿元",
            entities=["星辰科技"],
            numbers=["185亿元"],
            time_expressions=["2022年"],
        )
        c2 = _make_claim(
            "C2",
            "云服务营收262亿元",
            entities=["星辰科技"],
            numbers=["262亿元"],
            time_expressions=["2023年"],
        )
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_quarterly_and_annual_revenue_are_different_fact_slots(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技2024年Q1营收为178亿元", entities=["星辰科技"], numbers=["2024年", "1", "178亿元"])
        c2 = _make_claim("C2", "星辰科技2024财年全年营收达到700亿元", entities=["星辰科技"], numbers=["2024", "700亿元"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_segment_and_total_revenue_are_different_fact_slots(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技2023财年全年营收达到612亿元", entities=["星辰科技"], numbers=["2023", "612亿元"])
        c2 = _make_claim("C2", "云服务营收262亿元", entities=["星辰科技"], numbers=["262亿元"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNone(edge)

    def test_same_year_total_revenue_difference_still_conflicts(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技2023财年全年营收达到612亿元", entities=["星辰科技"], numbers=["2023", "612亿元"])
        c2 = _make_claim("C2", "星辰科技2023财年全年营收为500亿元", entities=["星辰科技"], numbers=["2023", "500亿元"])
        edge = builder._check_numerical_conflict(c1, c2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.NUMERIC_CONFLICT)


class TestSelfRefutingClaims(unittest.TestCase):
    def test_process_node_definition_creates_self_refutation(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        claim = _make_claim(
            "C1",
            '"3nm"和"5nm"等命名已不再代表实际的物理栅极长度',
            entities=["芯片制程"],
            numbers=["3nm", "5nm"],
        )
        evidence = Evidence("E1", "blog", "关于芯片制程的常见误解", "", claims=[claim])

        graph = builder.build_graph([evidence], use_llm=False)

        conflicts = graph.get_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].source_id, "C1")
        self.assertEqual(conflicts[0].target_id, "C1")
        self.assertEqual(conflicts[0].conflict_type, ConflictType.DEFINITIONAL_CONFLICT)

    def test_retrieval_count_monotonic_claim_creates_self_refutation(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        claim = _make_claim(
            "C1",
            "增加检索文档数量（top-k从3增至10）并不总是降低幻觉率，因为不相关文档会干扰生成",
            entities=["RAG"],
            numbers=["3", "10"],
        )
        evidence = Evidence("E1", "paper", "RAG系统幻觉率评估基准HAAG", "", claims=[claim])

        graph = builder.build_graph([evidence], use_llm=False)

        conflicts = graph.get_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].source_id, "C1")
        self.assertEqual(conflicts[0].target_id, "C1")
        self.assertEqual(conflicts[0].conflict_type, ConflictType.CAUSAL_CONFLICT)

    def test_global_emissions_self_refutation_requires_global_claim(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        regional_claim = _make_claim(
            "C1",
            "中国排放量较2022年增长2%",
            entities=["中国", "CO2"],
            numbers=["2022年", "2%"],
        )
        evidence = Evidence(
            "E1",
            "report",
            "全球碳计划2023年度报告：碳排放创新高",
            "",
            claims=[regional_claim],
        )

        graph = builder.build_graph([evidence], use_llm=False)

        self.assertEqual(graph.get_conflicts(), [])


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

    def test_different_dates_same_entity_different_fact_slot_no_conflict(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "星辰科技成立于2012年", entities=["星辰科技"], time_expressions=["2012年"])
        c2 = _make_claim("C2", "星辰科技员工超过60000人", entities=["星辰科技"], numbers=["60000"])
        e1 = _make_evidence("E1", date="2012-01-01")
        e2 = _make_evidence("E2", date="2024-01-01")
        edge = builder._check_temporal_conflict(c1, c2, e1, e2)
        self.assertIsNone(edge)

    def test_effective_year_does_not_conflict_with_passed_year(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim(
            "C1",
            "实际上，欧盟AI法案不仅已获通过，而且部分条款将于2025年开始生效",
            entities=["欧盟AI法案", "AI法案"],
            time_expressions=["2025年"],
        )
        c2 = _make_claim(
            "C2",
            "2024年3月13日，欧洲议会正式通过了《人工智能法案》",
            entities=["欧盟AI法案", "AI法案"],
            time_expressions=["2024年"],
        )
        e1 = _make_evidence("E1", date="2024-04-15")
        e2 = _make_evidence("E2", date="2024-03-13")

        edge = builder._check_temporal_conflict(c1, c2, e1, e2)

        self.assertIsNone(edge)


class TestScopeConflict(unittest.TestCase):
    def test_global_vs_regional(self):
        builder = ConflictGraphBuilder()
        c1 = _make_claim("C1", "全球AI监管框架逐步成型", entities=["AI监管框架"])
        c2 = _make_claim("C2", "中国AI监管框架逐步成型", entities=["AI监管框架"])
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

    def test_scope_conflict_disabled_in_dispatcher_by_default(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_support_detection": False,
            }
        })
        c1 = _make_claim("C1", "全球AI市场规模达到5880亿美元", entities=["AI市场"])
        c2 = _make_claim("C2", "中国AI市场规模达到1200亿元", entities=["AI市场"])
        e1 = _make_evidence("E1")
        e2 = _make_evidence("E2")

        edge = builder._detect_relationship(c1, e1, c2, e2, use_llm=False)

        self.assertIsNone(edge)

    def test_quantifier_scope_conflict_runs_without_broad_scope_rule(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_scope_conflict": False,
            }
        })
        c1 = _make_claim("C1", "欧盟AI法案禁止所有AI人脸识别", entities=["欧盟AI法案"], numbers=[])
        c2 = _make_claim("C2", "欧盟AI法案仅禁止实时远程生物识别", entities=["欧盟AI法案"], numbers=[])
        e1 = _make_evidence("E1")
        e2 = _make_evidence("E2")

        edge = builder._detect_relationship(c1, e1, c2, e2, use_llm=False)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.SCOPE_CONFLICT)


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

    def test_source_reliability_disabled_in_dispatcher_by_default(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_support_detection": False,
            }
        })
        c1 = _make_claim("C1", "星辰科技营收为612亿元", entities=["星辰科技"], numbers=["612亿元"])
        c2 = _make_claim("C2", "星辰科技营收错误为612亿元", entities=["星辰科技"], numbers=["612亿元"])
        e1 = _make_evidence("E1", source="official")
        e2 = _make_evidence("E2", source="blog")

        edge = builder._detect_relationship(c1, e1, c2, e2, use_llm=False)

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

    def test_status_contradiction_passed_vs_shelved(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        c1 = _make_claim("C1", "欧盟AI法案已正式通过", entities=["欧盟AI法案"])
        c2 = _make_claim("C2", "欧盟AI法案已无限期搁置", entities=["欧盟AI法案"])
        e1 = _make_evidence("E1")
        e2 = _make_evidence("E2")

        edge = builder._detect_relationship(c1, e1, c2, e2, use_llm=False)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.REFUTE)

    def test_contrast_context_does_not_create_status_contradiction(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        c1 = _make_claim("C1", "欧盟AI法案已无限期搁置", entities=["欧盟AI法案", "AI法案"])
        c2 = _make_claim(
            "C2",
            "与欧盟AI法案不同，美国目前尚未通过联邦层面的综合性AI立法",
            entities=["欧盟AI法案", "AI法案", "美国"],
        )
        e1 = _make_evidence("E1", source="blog")
        e2 = _make_evidence("E2", source="official")

        edge = builder._detect_relationship(c1, e1, c2, e2, use_llm=False)

        self.assertIsNone(edge)

    def test_passed_modifier_is_not_status_attribute(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        status = _make_claim("C1", "欧盟AI法案已正式通过", entities=["欧盟AI法案", "AI法案"])
        classification = _make_claim(
            "C2",
            "提案将AI系统分为三个风险等级，与最终通过的四个等级分类有所不同",
            entities=["欧盟AI法案", "AI法案"],
        )
        e1 = _make_evidence("E1", date="2024-03-13")
        e2 = _make_evidence("E2", date="2021-04-21")

        edge = builder._check_temporal_conflict(status, classification, e1, e2)

        self.assertIsNone(edge)

    def test_through_legal_framework_is_not_passed_status_attribute(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        claim = _make_claim(
            "C1",
            "欧盟采取立法先行模式，通过统一法律框架提供确定性",
            entities=["欧盟AI法案", "AI法案"],
        )

        self.assertNotIn("passed", builder._claim_attributes(claim))


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


class TestConflictGraphBuild(unittest.TestCase):
    def test_same_evidence_claims_are_not_compared_by_default(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        c1 = _make_claim("C1", "碳市场覆盖150个国家", entities=["碳市场"], numbers=["150个"])
        c2 = _make_claim("C2", "碳市场共有36个交易系统", entities=["碳市场"], numbers=["36个"])
        ev = Evidence(
            evidence_id="E1",
            source="report",
            title="碳市场报告",
            text_span="",
            claims=[c1, c2],
        )

        graph = builder.build_graph([ev], use_llm=False)

        self.assertEqual(graph.get_conflicts(), [])

    def test_explicit_same_evidence_counterclaims_are_compared(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        c1 = _make_claim("C1", "欧盟AI法案禁止所有AI人脸识别", entities=["欧盟AI法案"])
        c1.source_span = "reported_claim"
        c2 = _make_claim("C2", "欧盟AI法案仅禁止实时远程生物识别", entities=["欧盟AI法案"])
        c2.source_span = "corrective_claim"
        ev = Evidence(
            evidence_id="E1",
            source="blog",
            title="AI监管政策争议",
            text_span="",
            claims=[c1, c2],
        )

        graph = builder.build_graph([ev], use_llm=False)

        conflicts = graph.get_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.SCOPE_CONFLICT)

    def test_explicit_same_evidence_numeric_dispute_is_compared(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        c1 = _make_claim(
            "C1",
            "谷歌称经典超级计算机需要约10000年才能完成该计算任务",
            entities=["量子霸权", "谷歌", "IBM"],
            numbers=["10000年"],
        )
        c2 = _make_claim(
            "C2",
            "IBM提出质疑，认为经优化的经典算法可在2.5天内完成相同任务",
            entities=["量子霸权", "谷歌", "IBM"],
            numbers=["2.5"],
        )
        ev = Evidence(
            evidence_id="D030_c0",
            source="paper",
            title="谷歌量子计算里程碑：量子霸权论文",
            text_span="谷歌称经典超级计算机需要约10000年。IBM随后提出质疑，认为经典算法可在2.5天内完成。",
            claims=[c1, c2],
        )

        graph = builder.build_graph([ev], use_llm=False)

        conflicts = graph.get_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.NUMERIC_CONFLICT)

    def test_same_evidence_climate_sensitivity_numeric_contrast_is_compared(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        ecs = _make_claim(
            "C1",
            "我们估计ECS可能为2.5°C",
            entities=["ECS"],
            numbers=["2.5°C"],
        )
        ipcc = _make_claim(
            "C2",
            "略低于IPCC AR6的最佳估计值3.0°C",
            entities=["IPCC", "AR6"],
            numbers=["3.0°C"],
        )
        ev = Evidence(
            evidence_id="E1",
            source="paper",
            title="关于全球升温速度的学术争议",
            text_span="我们估计ECS可能为2.5°C，略低于IPCC AR6的最佳估计值3.0°C。",
            claims=[ecs, ipcc],
        )

        graph = builder.build_graph([ev], use_llm=False)

        conflicts = graph.get_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.NUMERIC_CONFLICT)

    def test_corrected_reported_claim_does_not_emit_duplicate_cross_evidence_edge(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        passed = _make_claim("C1", "欧盟AI法案已正式通过", entities=["欧盟AI法案", "AI法案"])
        reported = _make_claim("C2", "欧盟AI法案已无限期搁置", entities=["欧盟AI法案", "AI法案"])
        reported.source_span = "reported_claim"
        corrective = _make_claim("C3", "实际上，欧盟AI法案不仅已获通过", entities=["欧盟AI法案", "AI法案"])
        corrective.source_span = "corrective_claim"
        official = Evidence("E1", "official", "欧盟AI法案", "", claims=[passed])
        correction = Evidence(
            "E2",
            "blog",
            "AI监管政策争议",
            "部分自媒体声称该法案已无限期搁置，这是完全错误的。实际上，欧盟AI法案不仅已获通过。",
            claims=[reported, corrective],
        )

        graph = builder.build_graph([official, correction], use_llm=False)

        conflicts = graph.get_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual({conflicts[0].source_id, conflicts[0].target_id}, {"C2", "C3"})

    def test_iter_first_plasma_delay_is_self_refuting_temporal_conflict(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        claim = _make_claim(
            "C1",
            "ITER原计划2025年实现首次等离子体，但因延误和超支，预计推迟至2030年前后",
            entities=["ITER"],
            numbers=["2025年", "2030年"],
            time_expressions=["2025", "2030"],
        )
        ev = Evidence(
            evidence_id="E1",
            source="report",
            title="全球核聚变研究进展：从ITER到商业发电",
            text_span=claim.claim,
            claims=[claim],
        )

        graph = builder.build_graph([ev], use_llm=False)

        conflicts = graph.get_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.TEMPORAL_CONFLICT)
        self.assertEqual(conflicts[0].source_id, conflicts[0].target_id)

    def test_qubit_count_is_not_compared_with_duration_or_code_distance(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        qubits = _make_claim(
            "C1",
            "Willow拥有105个量子比特",
            entities=["Willow"],
            numbers=["105个"],
        )
        duration = _make_claim(
            "C2",
            "量子纠错是该领域30年来的核心目标",
            entities=["Willow"],
            numbers=["30年"],
        )
        code_distance = _make_claim(
            "C3",
            "实验使用105个物理量子比特构建距离为7的表面码",
            entities=["Willow"],
            numbers=["7"],
        )

        ev1 = Evidence("E1", "report", "Willow", "", claims=[qubits])
        ev2 = Evidence("E2", "report", "Willow", "", claims=[duration, code_distance])

        graph = builder.build_graph([ev1, ev2], use_llm=False)

        self.assertEqual(graph.get_conflicts(), [])

    def test_current_qubit_count_is_not_compared_with_future_requirement(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        current = _make_claim(
            "C1",
            "Willow拥有105个物理量子比特",
            entities=["Willow"],
            numbers=["105个"],
        )
        required = _make_claim(
            "C2",
            "实用量子纠错需要至少1000个逻辑量子比特",
            entities=["Willow"],
            numbers=["1000个"],
        )

        graph = builder.build_graph(
            [
                Evidence("E1", "report", "Willow", "", claims=[current]),
                Evidence("E2", "paper", "Willow", "", claims=[required]),
            ],
            use_llm=False,
        )

        self.assertEqual(graph.get_conflicts(), [])

    def test_explicit_same_evidence_process_node_misnomer_is_compared(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        c1 = _make_claim(
            "C1",
            "3nm和5nm等命名已不再代表实际的物理栅极长度",
            entities=["芯片制程", "台积电"],
            numbers=["3", "5"],
        )
        c2 = _make_claim(
            "C2",
            "台积电的3nm工艺的实际栅极长度约为20nm以上",
            entities=["芯片制程", "台积电"],
            numbers=["3", "20"],
        )
        ev = Evidence(
            evidence_id="D053_c0",
            source="blog",
            title="关于芯片制程的常见误解",
            text_span="关于芯片制程存在常见误解。3nm命名已不再代表实际栅极长度，实际长度约20nm以上。",
            claims=[c1, c2],
        )

        graph = builder.build_graph([ev], use_llm=False)

        self.assertTrue(graph.get_conflicts())

    def test_explicit_same_evidence_sales_scope_dispute_is_compared(self):
        builder = ConflictGraphBuilder(config={"conflict_graph": {"enable_nli": False}})
        c1 = _make_claim(
            "C1",
            "比亚迪在全球销售了约302.4万辆新能源汽车",
            entities=["比亚迪", "特斯拉", "销量"],
            numbers=["302.4万"],
        )
        c2 = _make_claim(
            "C2",
            "特斯拉全球交付约181万辆纯电动汽车",
            entities=["比亚迪", "特斯拉", "销量"],
            numbers=["181万"],
        )
        ev = Evidence(
            evidence_id="D043_c0",
            source="news",
            title="比亚迪vs特斯拉：2023年全球销量对比",
            text_span="比亚迪销售新能源汽车，特斯拉交付纯电动汽车。仅从纯电动汽车来看，特斯拉仍然领先。",
            claims=[c1, c2],
        )

        graph = builder.build_graph([ev], use_llm=False)

        self.assertTrue(graph.get_conflicts())


class TestLearnedConflictDetector(unittest.TestCase):
    class DummyModel:
        def __init__(self, score):
            self.score = score
            self.calls = []

        def predict(self, pairs, show_progress_bar=False):
            self.calls.append((pairs, show_progress_bar))
            return [self.score] * len(pairs)

    def test_score_to_probability_accepts_probability(self):
        self.assertEqual(ConflictGraphBuilder._score_to_probability([0.8]), 0.8)

    def test_score_to_probability_uses_positive_class_probability(self):
        self.assertEqual(ConflictGraphBuilder._score_to_probability([0.1, 0.9]), 0.9)
        self.assertEqual(ConflictGraphBuilder._score_to_probability([[0.2, 0.8]]), 0.8)

    def test_score_to_probability_applies_sigmoid_to_logit(self):
        prob = ConflictGraphBuilder._score_to_probability([2.0])
        self.assertAlmostEqual(prob, 0.8807970779778823)

    def test_score_to_probability_applies_softmax_to_two_class_logits(self):
        prob = ConflictGraphBuilder._score_to_probability([0.0, 2.0])
        self.assertAlmostEqual(prob, 0.8807970779778823)

    def test_score_to_probability_fails_closed_for_invalid_values(self):
        self.assertEqual(ConflictGraphBuilder._score_to_probability([]), 0.0)
        self.assertEqual(ConflictGraphBuilder._score_to_probability([0.1, 0.2, 0.7]), 0.0)
        self.assertEqual(ConflictGraphBuilder._score_to_probability(float("nan")), 0.0)
        self.assertEqual(ConflictGraphBuilder._score_to_probability(float("inf")), 0.0)
        self.assertEqual(ConflictGraphBuilder._score_to_probability("not-a-score"), 0.0)

    def test_score_to_probability_handles_extreme_logits_without_overflow(self):
        self.assertEqual(ConflictGraphBuilder._score_to_probability(1000.0), 1.0)
        self.assertEqual(ConflictGraphBuilder._score_to_probability(-1000.0), 0.0)

    def test_learned_detector_missing_path_degrades(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_learned_detector": True,
                "learned_model_path": "",
            }
        })
        assert builder._init_learned_detector() is False
        self.assertTrue(builder._learned_tried)
        self.assertFalse(builder._learned_available)

    def test_unset_env_placeholder_does_not_enable_learned_detector(self):
        old_value = os.environ.pop("VERARAG_MISSING_CONFLICT_MODEL", None)
        try:
            builder = ConflictGraphBuilder(config={
                "conflict_graph": {
                    "learned_model_path": "${VERARAG_MISSING_CONFLICT_MODEL}",
                }
            })
        finally:
            if old_value is not None:
                os.environ["VERARAG_MISSING_CONFLICT_MODEL"] = old_value

        self.assertEqual(builder.learned_model_path, "")
        self.assertFalse(builder.enable_learned_detector)

    def test_env_placeholder_can_enable_learned_detector(self):
        old_value = os.environ.get("VERARAG_TEST_CONFLICT_MODEL")
        os.environ["VERARAG_TEST_CONFLICT_MODEL"] = "~/models/conflict"
        try:
            builder = ConflictGraphBuilder(config={
                "conflict_graph": {
                    "learned_model_path": "${VERARAG_TEST_CONFLICT_MODEL}",
                }
            })
        finally:
            if old_value is None:
                os.environ.pop("VERARAG_TEST_CONFLICT_MODEL", None)
            else:
                os.environ["VERARAG_TEST_CONFLICT_MODEL"] = old_value

        self.assertTrue(builder.learned_model_path.endswith("/models/conflict"))
        self.assertTrue(builder.enable_learned_detector)

    def test_probability_threshold_config_accepts_strings(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "learned_threshold": "0.71",
                "learned_candidate_similarity": "0.19",
                "nli_threshold": "0.72",
                "text_similarity_threshold": "0.61",
                "min_conflict_similarity": "0.23",
                "unattributed_conflict_similarity": "0.56",
                "llm_adjudication_similarity": "0.36",
            }
        })

        self.assertEqual(builder.learned_threshold, 0.71)
        self.assertEqual(builder.learned_candidate_similarity, 0.19)
        self.assertEqual(builder.nli_threshold, 0.72)
        self.assertEqual(builder.text_similarity_threshold, 0.61)
        self.assertEqual(builder.min_conflict_similarity, 0.23)
        self.assertEqual(builder.unattributed_conflict_similarity, 0.56)
        self.assertEqual(builder.llm_adjudication_similarity, 0.36)

    def test_probability_threshold_config_bounds_invalid_values(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "learned_threshold": "not-a-number",
                "learned_candidate_similarity": True,
                "nli_threshold": float("nan"),
                "min_conflict_similarity": -1,
                "unattributed_conflict_similarity": 2,
            }
        })

        self.assertEqual(builder.learned_threshold, 0.7)
        self.assertEqual(builder.learned_candidate_similarity, 0.18)
        self.assertEqual(builder.nli_threshold, 0.7)
        self.assertEqual(builder.min_conflict_similarity, 0.0)
        self.assertEqual(builder.unattributed_conflict_similarity, 1.0)

    def test_learned_detector_returns_refute_edge(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "learned_threshold": 0.7,
            }
        })
        builder._learned_available = True
        builder._learned_model = self.DummyModel(0.91)
        c1 = _make_claim("C1", "该政策适用于所有企业", entities=["政策"])
        c2 = _make_claim("C2", "该政策不适用于所有企业", entities=["政策"])

        edge = builder._learned_conflict_detect(c1, c2)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.REFUTE)
        self.assertEqual(edge.confidence, 0.91)
        self.assertEqual(edge.severity, "high")

    def test_learned_detector_requires_context_by_default(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "learned_threshold": 0.7,
            }
        })
        builder._learned_available = True
        builder._learned_model = self.DummyModel(0.99)
        c1 = _make_claim("C1", "星辰科技成立于2012年", entities=["星辰科技"])
        c2 = _make_claim("C2", "星辰科技员工超过60000人", entities=["星辰科技"])

        edge = builder._learned_conflict_detect(c1, c2)

        self.assertIsNone(edge)
        self.assertEqual(builder._learned_model.calls, [])

    def test_learned_detector_rejects_shared_numbers_without_same_fact_slot(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "learned_threshold": 0.7,
            }
        })
        builder._learned_available = True
        builder._learned_model = self.DummyModel(0.99)
        c1 = _make_claim(
            "C1",
            "星辰科技2022财年全年营收为458亿元",
            entities=["星辰科技"],
            numbers=["2022", "458亿元"],
        )
        c2 = _make_claim(
            "C2",
            "公司于2018年在上海证券交易所科创板上市",
            entities=["星辰科技"],
            numbers=["2018年"],
        )

        edge = builder._learned_conflict_detect(c1, c2)

        self.assertIsNone(edge)
        self.assertEqual(builder._learned_model.calls, [])

    def test_learned_detector_allows_same_fact_slot(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "learned_threshold": 0.7,
            }
        })
        builder._learned_available = True
        builder._learned_model = self.DummyModel(0.91)
        c1 = _make_claim("C1", "星辰科技营收500亿元", entities=["星辰科技"], numbers=["500"])
        c2 = _make_claim("C2", "星辰科技营收800亿元", entities=["星辰科技"], numbers=["800"])

        edge = builder._learned_conflict_detect(c1, c2)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.REFUTE)

    def test_learned_detector_accepts_two_class_positive_score(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "learned_threshold": 0.7,
            }
        })
        builder._learned_available = True
        builder._learned_model = self.DummyModel([0.05, 0.95])
        c1 = _make_claim("C1", "星辰科技营收500亿元", entities=["星辰科技"], numbers=["500"])
        c2 = _make_claim("C2", "星辰科技营收800亿元", entities=["星辰科技"], numbers=["800"])

        edge = builder._learned_conflict_detect(c1, c2)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.confidence, 0.95)

    def test_learned_detector_accepts_string_threshold_config(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "learned_threshold": "0.7",
            }
        })
        builder._learned_available = True
        builder._learned_model = self.DummyModel(0.91)
        c1 = _make_claim("C1", "星辰科技营收500亿元", entities=["星辰科技"], numbers=["500"])
        c2 = _make_claim("C2", "星辰科技营收800亿元", entities=["星辰科技"], numbers=["800"])

        edge = builder._learned_conflict_detect(c1, c2)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.confidence, 0.91)

    def test_learned_detector_below_threshold_returns_none(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "learned_threshold": 0.7,
            }
        })
        builder._learned_available = True
        builder._learned_model = self.DummyModel(0.4)
        c1 = _make_claim("C1", "甲公司发布产品")
        c2 = _make_claim("C2", "乙公司发布产品")

        self.assertIsNone(builder._learned_conflict_detect(c1, c2))

    def test_learned_detector_rejects_nonfinite_cached_probability(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "learned_threshold": 0.7,
            }
        })
        builder._learned_available = True
        builder._learned_model = self.DummyModel(0.99)
        c1 = _make_claim("C1", "星辰科技营收500亿元", entities=["星辰科技"], numbers=["500"])
        c2 = _make_claim("C2", "星辰科技营收800亿元", entities=["星辰科技"], numbers=["800"])
        builder._learned_score_cache[(id(c1), id(c2))] = float("nan")

        edge = builder._learned_conflict_detect(c1, c2)

        self.assertIsNone(edge)
        self.assertEqual(builder._learned_model.calls, [])

    def test_build_graph_batches_learned_candidate_scores(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_learned_detector": True,
                "enable_support_detection": False,
                "learned_threshold": 0.7,
            }
        })
        builder._learned_available = True
        builder._learned_tried = True
        builder._learned_model = self.DummyModel(0.8)
        evidence = [
            Evidence(
                evidence_id=f"E{index}",
                source="report",
                title="Alpha 项目策略",
                text_span="",
                claims=[
                    _make_claim(
                        f"C{index}",
                        text,
                        entities=["Alpha 项目"],
                    )
                ],
            )
            for index, text in enumerate(
                [
                    "Alpha 项目策略采用 A 方案",
                    "Alpha 项目策略未采用 A 方案",
                    "Alpha 项目策略考虑 B 方案",
                ],
                start=1,
            )
        ]

        builder.build_graph(evidence, use_llm=False)

        self.assertEqual(len(builder._learned_model.calls), 1)
        pairs, show_progress_bar = builder._learned_model.calls[0]
        self.assertGreater(len(pairs), 1)
        self.assertFalse(show_progress_bar)

    def test_dispatcher_uses_learned_layer_before_llm(self):
        builder = ConflictGraphBuilder(
            llm_client=None,
            config={
                "conflict_graph": {
                    "enable_nli": False,
                    "enable_learned_detector": True,
                    "enable_support_detection": False,
                    "learned_threshold": 0.7,
                }
            },
        )
        builder._learned_available = True
        builder._learned_tried = True
        builder._learned_model = self.DummyModel(0.75)
        c1 = _make_claim("C1", "Alpha 项目采用 A 方案", entities=["Alpha 项目"])
        c2 = _make_claim("C2", "Alpha 项目未采用 A 方案", entities=["Alpha 项目"])
        e1 = _make_evidence("E1")
        e2 = _make_evidence("E2")

        edge = builder._detect_relationship(c1, e1, c2, e2, use_llm=False)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.REFUTE)

    def test_llm_adjudication_disabled_by_default(self):
        class DummyLLM:
            pass

        builder = ConflictGraphBuilder(llm_client=DummyLLM())
        c1 = _make_claim("C1", "星辰科技成立于2012年", entities=["星辰科技"])
        c2 = _make_claim("C2", "星辰科技员工超过60000人", entities=["星辰科技"])
        e1 = _make_evidence("E1")
        e2 = _make_evidence("E2")

        edge = builder._detect_relationship(c1, e1, c2, e2, use_llm=True)

        self.assertIsNone(edge)


class TestNLIConflictDetection(unittest.TestCase):
    class DummyConfig:
        def __init__(self, id2label):
            self.id2label = id2label

    class DummyBackbone:
        def __init__(self, id2label):
            self.config = TestNLIConflictDetection.DummyConfig(id2label)

    class DummyNLIModel:
        def __init__(self, scores, id2label):
            self.scores = scores
            self.model = TestNLIConflictDetection.DummyBackbone(id2label)

        def predict(self, pairs, show_progress_bar=False):
            return self.scores

    def _builder_with_nli(self, scores, id2label):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": True,
                "nli_threshold": 0.7,
            }
        })
        builder._nli_available = True
        builder._nli_model = self.DummyNLIModel(scores, id2label)
        return builder

    def test_nli_detection_uses_model_label_order(self):
        builder = self._builder_with_nli(
            scores=[[0.0, 0.0, 5.0]],
            id2label={0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},
        )

        edge = builder._nli_detect(
            _make_claim("C1", "Alpha 项目采用 A 方案"),
            _make_claim("C2", "Alpha 项目采用 A 方案"),
        )

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.SUPPORT)

    def test_nli_detection_rejects_ambiguous_label_mapping(self):
        builder = self._builder_with_nli(
            scores=[[5.0, 0.0, 0.0]],
            id2label={0: "CONTRADICTION", 1: "CONTRADICTION", 2: "NEUTRAL"},
        )

        edge = builder._nli_detect(
            _make_claim("C1", "Alpha 项目采用 A 方案"),
            _make_claim("C2", "Alpha 项目未采用 A 方案"),
        )

        self.assertIsNone(edge)

    def test_nli_detection_rejects_nonfinite_scores(self):
        builder = self._builder_with_nli(
            scores=[[float("inf"), 0.0, 0.0]],
            id2label={0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"},
        )

        edge = builder._nli_detect(
            _make_claim("C1", "Alpha 项目采用 A 方案"),
            _make_claim("C2", "Alpha 项目未采用 A 方案"),
        )

        self.assertIsNone(edge)

    def test_nli_detection_keeps_opposite_passed_status_contradiction(self):
        builder = self._builder_with_nli(
            scores=[[5.0, 0.0, 0.0]],
            id2label={0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"},
        )

        edge = builder._nli_detect(
            _make_claim("C1", "欧盟AI法案已无限期搁置，尚未通过"),
            _make_claim("C2", "2024年3月13日，欧洲议会正式通过了《人工智能法案》"),
        )

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.REFUTE)


class TestGraphUpdate(unittest.TestCase):
    def test_update_graph_connects_existing_claims_to_new_evidence(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_support_detection": False,
            }
        })
        old_evidence = Evidence(
            evidence_id="E1",
            source="report",
            title="Old",
            text_span="",
            claims=[
                _make_claim("C1", "星辰科技2024年营收500亿元", entities=["星辰科技"], numbers=["500亿元"]),
            ],
        )
        graph = builder.build_graph([old_evidence], use_llm=False)
        new_evidence = [
            Evidence(
                evidence_id="E2",
                source="report",
                title="New",
                text_span="",
                claims=[
                    _make_claim("C2", "星辰科技2024年营收800亿元", entities=["星辰科技"], numbers=["800亿元"]),
                ],
            )
        ]

        builder.update_graph(graph, new_evidence, use_llm=False)

        self.assertIn("C1", graph.nodes)
        self.assertIn("C2", graph.nodes)
        conflicts = graph.get_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.NUMERIC_CONFLICT)
        self.assertEqual({conflicts[0].source_id, conflicts[0].target_id}, {"C1", "C2"})

    def test_update_graph_does_not_duplicate_existing_edges(self):
        builder = ConflictGraphBuilder(config={
            "conflict_graph": {
                "enable_nli": False,
                "enable_support_detection": False,
            }
        })
        old_evidence = Evidence(
            evidence_id="E1",
            source="report",
            title="Old",
            text_span="",
            claims=[
                _make_claim("C1", "星辰科技2024年营收500亿元", entities=["星辰科技"], numbers=["500亿元"]),
            ],
        )
        new_evidence = [
            Evidence(
                evidence_id="E2",
                source="report",
                title="New",
                text_span="",
                claims=[
                    _make_claim("C2", "星辰科技2024年营收800亿元", entities=["星辰科技"], numbers=["800亿元"]),
                ],
            )
        ]
        graph = builder.build_graph([old_evidence], use_llm=False)

        builder.update_graph(graph, new_evidence, use_llm=False)
        builder.update_graph(graph, new_evidence, use_llm=False)

        self.assertEqual(len(graph.get_conflicts()), 1)


class TestLLMConflictAdjudication(unittest.TestCase):
    class DummyLLM:
        def __init__(self, response):
            self.response = response
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return self.response

    def test_llm_conflict_detection_normalizes_relationship_and_confidence(self):
        llm = self.DummyLLM(
            '{"relationship": " refute ", "confidence": 1.4, "rationale": "contradicts"}'
        )
        builder = ConflictGraphBuilder(llm_client=llm)
        c1 = _make_claim("C1", "Alpha 项目采用 A 方案")
        c2 = _make_claim("C2", "Alpha 项目未采用 A 方案")

        edge = builder._llm_conflict_detection(c1, c2)

        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.REFUTE)
        self.assertEqual(edge.confidence, 1.0)
        self.assertEqual(edge.severity, "high")
        self.assertEqual(llm.calls[0]["response_format"], "json")

    def test_llm_conflict_detection_rejects_unknown_relationship(self):
        llm = self.DummyLLM(
            '{"relationship": "MAYBE", "confidence": 0.9, "rationale": "unclear"}'
        )
        builder = ConflictGraphBuilder(llm_client=llm)

        edge = builder._llm_conflict_detection(
            _make_claim("C1", "声明 A"),
            _make_claim("C2", "声明 B"),
        )

        self.assertIsNone(edge)

    def test_llm_conflict_detection_rejects_invalid_or_nonfinite_confidence(self):
        for response in (
            '{"relationship": "REFUTE", "confidence": "high"}',
            '{"relationship": "REFUTE", "confidence": NaN}',
        ):
            builder = ConflictGraphBuilder(llm_client=self.DummyLLM(response))

            edge = builder._llm_conflict_detection(
                _make_claim("C1", "声明 A"),
                _make_claim("C2", "声明 B"),
            )

            self.assertIsNone(edge)


if __name__ == "__main__":
    unittest.main()
