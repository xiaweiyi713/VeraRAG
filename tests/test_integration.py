"""Integration test: full VeraRAG pipeline with Mock LLM."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_structures import (
    AnswerClaim,
    Claim,
    ClaimType,
    ConflictEdge,
    ConflictType,
    Evidence,
    EvidenceConflictGraph,
    ReasoningStep,
    VeraRAGOutput,
    VerificationReport,
    VerificationStatus,
)


class MockRetriever:
    """Mock retriever that uses BM25 internally to avoid sentence-transformers dep."""
    def __init__(self, config=None, **kwargs):
        self.docs = []
        self._bm25 = None

    def index_documents(self, documents):
        from src.retriever.bm25 import BM25Retriever
        self._bm25 = BM25Retriever()
        self._bm25.index_documents(documents)
        self.docs = documents

    def retrieve(self, query, top_k=10, **kwargs):
        return self._bm25.retrieve(query, top_k=top_k)


class MockLLMClient:
    """Mock LLM that returns structured responses based on prompt keywords."""

    def __init__(self, **kwargs):
        self.provider = kwargs.get("provider", "mock")
        self.model = kwargs.get("model", "mock")

    def generate(self, prompt: str, **kwargs) -> str:
        # Planner: must use unique markers not present in other prompts
        if "break down the following complex question" in prompt.lower() or "分解" in prompt or "decompos" in prompt.lower():
            return json.dumps({
                "subquestions": [
                    {"question": "量子计算是什么", "required_evidence_type": "general",
                     "dependency_ids": [], "requires_counter_evidence": False},
                    {"question": "量子计算的最新进展有哪些", "required_evidence_type": "general",
                     "dependency_ids": ["sq0"], "requires_counter_evidence": False},
                ],
                "reasoning_plan": ["查找量子计算基本概念", "汇总最新进展"]
            })
        if "分析" in prompt or "任务" in prompt or "task" in prompt.lower():
            return json.dumps({
                "task_type": "multi_hop_qa",
                "complexity": "high",
                "keywords": ["测试", "关键词"],
                "requires_retrieval": True,
                "requires_conflict_check": True,
                "estimated_hops": 2,
            })
        if "答案" in prompt or "answer" in prompt.lower() or "推理" in prompt or "reason" in prompt.lower():
            return json.dumps({
                "answer": "根据证据分析，这是一个测试答案。",
                "answer_claims": [
                    {"claim": "测试声明", "supporting_evidence": ["E1"],
                     "conflicting_evidence": [], "confidence": 0.85, "verification_status": "supported",
                     "claim_type": "factual", "verifiable": True, "support_type": "direct"}
                ],
                "reasoning_chain": [
                    {"step": 1, "description": "检索证据", "evidence_ids": ["E1"], "confidence": 0.8}
                ],
            })
        if "验证" in prompt or "verif" in prompt.lower():
            return json.dumps({
                "claim_verifications": [
                    {"claim": "声明1", "status": "supported"},
                ],
                "overall_status": "supported",
                "issues": [],
            })
        if "修复" in prompt or "repair" in prompt.lower():
            return "修复后的答案"
        return "默认回答"


@pytest.fixture
def pipeline_config():
    return {
        "llm": {"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
        "pipeline": {
            "max_retrieval_rounds": 2,
            "max_subquestions": 3,
            "enable_conflict_graph": True,
            "enable_uncertainty": True,
            "enable_verification": True,
            "enable_repair": True,
        },
        "retriever": {"top_k": 5},
    }


def _create_pipeline(config):
    """Create pipeline with mock LLM and mock retriever."""
    from src.pipeline.verarag import VeraRAG
    with patch("src.pipeline.verarag.HybridRetriever", MockRetriever):
        pipeline = VeraRAG(config)
    # Replace LLM client on pipeline and all agents
    mock_llm = MockLLMClient()
    pipeline.llm_client = mock_llm
    for agent in [pipeline.task_analyzer, pipeline.planner, pipeline.retrieval_agent,
                  pipeline.reasoning_agent, pipeline.verifier_agent, pipeline.repair_agent,
                  pipeline.evidence_extractor, pipeline.conflict_graph_builder]:
        if hasattr(agent, 'llm_client'):
            agent.llm_client = mock_llm
    return pipeline


def test_retrieval_top_k_policy_config_reaches_retrieval_agent(pipeline_config):
    config = {
        **pipeline_config,
        "retriever": {
            **pipeline_config["retriever"],
            "top_k_policy": "complexity_adaptive",
        },
    }

    pipeline = _create_pipeline(config)

    assert pipeline.retrieval_agent.top_k_policy == "complexity_adaptive"


class TestPipelineIntegration:
    def test_original_question_retrieval_anchor_is_added(self):
        from src.pipeline.verarag import VeraRAG
        from src.utils.data_structures import SubQuestion

        subquestions = [
            SubQuestion(id="sq0", question="该法案的适用范围是什么？"),
        ]

        anchored = VeraRAG._ensure_original_question_retrieval_anchor(
            "欧盟AI法案是否禁止所有人脸识别？",
            subquestions,
            requires_counter_evidence=True,
        )

        assert anchored[0].id == "sq_original"
        assert anchored[0].question == "欧盟AI法案是否禁止所有人脸识别？"
        assert anchored[0].requires_counter_evidence is True
        assert anchored[1:] == subquestions

    def test_original_question_retrieval_anchor_is_not_duplicated(self):
        from src.pipeline.verarag import VeraRAG
        from src.utils.data_structures import SubQuestion

        subquestions = [
            SubQuestion(id="sq0", question="欧盟AI法案是否禁止所有人脸识别？"),
        ]

        anchored = VeraRAG._ensure_original_question_retrieval_anchor(
            "欧盟AI法案是否禁止所有人脸识别？",
            subquestions,
            requires_counter_evidence=True,
        )

        assert anchored == subquestions

    def test_title_entity_anchors_filter_metric_phrases(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)

        anchors = pipeline._title_entity_anchors("全球新能源汽车销量突破2000万辆")

        assert "全球新能源汽车销量" not in anchors

    def test_title_entity_anchors_keep_named_subjects(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)

        company_anchors = pipeline._title_entity_anchors("星辰科技2023年度财务报告")
        policy_anchors = pipeline._title_entity_anchors("欧盟人工智能法案：从提案到立法")

        assert "星辰科技" in company_anchors
        assert "欧盟AI法案" in policy_anchors

    def test_retrieval_result_metadata_preserved_as_evidence_anchors(self, pipeline_config):
        from src.pipeline.verarag import VeraRAG
        from src.retriever.base import RetrievalResult

        pipeline = VeraRAG({
            **pipeline_config,
            "retriever": {"type": "bm25"},
        })
        result = RetrievalResult(
            doc_id="D017_c0",
            content="欧盟AI法案已于2024年3月13日通过。",
            title="欧盟AI法案",
            score=0.9,
            metadata={
                "source": "official",
                "date": "2024-03-13",
                "url": "https://example.test/eu-ai-act",
                "entities": ["欧盟AI法案"],
            },
        )

        evidence = pipeline._retrieval_result_to_evidence(result)

        assert evidence.entities == ["欧盟AI法案"]
        assert evidence.date == "2024-03-13"
        assert evidence.url == "https://example.test/eu-ai-act"

    def test_question_focus_filters_unrelated_conflict_edges(self, pipeline_config):
        pipeline = _create_pipeline({
            **pipeline_config,
            "pipeline": {
                **pipeline_config["pipeline"],
                "enable_verification": False,
                "enable_repair": False,
            },
        })
        passed = Claim(
            claim_id="C_passed",
            claim="欧盟AI法案已于2024年3月13日通过",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        shelved = Claim(
            claim_id="C_shelved",
            claim="欧盟AI法案已无限期搁置",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
            source_span="reported_claim",
        )
        all_face = Claim(
            claim_id="C_all_face",
            claim="欧盟AI法案禁止所有人脸识别",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        limited_face = Claim(
            claim_id="C_limited_face",
            claim="欧盟AI法案仅禁止实时远程生物识别",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        evidence_pool = [
            Evidence("E_passed", "official", "欧盟AI法案", "", claims=[passed], relevance_score=0.9),
            Evidence("E_blog", "blog", "争议报道", "", claims=[shelved], relevance_score=0.5),
            Evidence("E_scope", "report", "AI监管范围", "", claims=[all_face, limited_face], relevance_score=0.8),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_shelved", "C_passed", ConflictType.REFUTE, 0.9),
            ConflictEdge("C_all_face", "C_limited_face", ConflictType.SCOPE_CONFLICT, 0.85),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "欧盟AI法案是否已经通过？",
        )

        assert [(edge.source_id, edge.target_id) for edge in filtered.get_conflicts()] == [
            ("C_shelved", "C_passed")
        ]

    def test_question_fact_slot_filters_law_status_from_fine_question(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        passed = Claim(
            claim_id="C_passed",
            claim="欧盟AI法案已于2024年3月13日通过",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        shelved = Claim(
            claim_id="C_shelved",
            claim="欧盟AI法案已无限期搁置",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        evidence_pool = [
            Evidence("E_passed", "official", "欧盟AI法案", "", claims=[passed]),
            Evidence("E_shelved", "blog", "错误解读", "", claims=[shelved]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_passed", "C_shelved", ConflictType.REFUTE, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "欧盟AI法案将违规罚款上限设定为多少？",
        )

        assert filtered.get_conflicts() == []

    def test_question_fact_slot_filters_runtime_from_qubit_question(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        google_runtime = Claim(
            claim_id="C_google_runtime",
            claim="谷歌称经典超级计算机需要约10000年完成该计算任务",
            claim_type=ClaimType.FACTUAL,
            entities=["谷歌", "量子霸权"],
            numbers=["10000年"],
        )
        ibm_runtime = Claim(
            claim_id="C_ibm_runtime",
            claim="IBM认为经典算法只需2.5天完成相同计算任务",
            claim_type=ClaimType.FACTUAL,
            entities=["谷歌", "IBM", "量子霸权"],
            numbers=["2.5天"],
        )
        evidence = Evidence(
            "D030_c0",
            "paper",
            "谷歌量子计算里程碑",
            "",
            claims=[google_runtime, ibm_runtime],
        )
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge(
                "C_google_runtime",
                "C_ibm_runtime",
                ConflictType.NUMERIC_CONFLICT,
                0.9,
            ),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            [evidence],
            "谷歌Willow量子处理器有多少个量子比特？",
        )

        assert filtered.get_conflicts() == []

    def test_question_fact_slot_checks_both_sides_of_mixed_claim(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        mixed = Claim(
            claim_id="C_mixed",
            claim="谷歌53量子比特处理器在200秒内完成采样任务",
            claim_type=ClaimType.FACTUAL,
            entities=["谷歌", "Sycamore"],
            numbers=["53量子比特", "200秒"],
        )
        runtime = Claim(
            claim_id="C_runtime",
            claim="IBM认为经典算法可在2.5天内完成相同任务",
            claim_type=ClaimType.FACTUAL,
            entities=["谷歌", "IBM", "Sycamore"],
            numbers=["2.5天"],
        )
        evidence = Evidence(
            "D030_c0",
            "paper",
            "谷歌量子霸权论文",
            "",
            claims=[mixed, runtime],
        )
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_mixed", "C_runtime", ConflictType.NUMERIC_CONFLICT, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            [evidence],
            "谷歌Willow量子处理器有多少个量子比特？",
        )

        assert filtered.get_conflicts() == []

    def test_opposite_attribute_values_share_question_fact_slot(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        decline = Claim(
            claim_id="C_decline",
            claim="全球碳排放已经开始下降",
            claim_type=ClaimType.FACTUAL,
            entities=["全球碳排放"],
        )
        growth = Claim(
            claim_id="C_growth",
            claim="全球化石燃料CO2排放仍在增长并创历史新高",
            claim_type=ClaimType.FACTUAL,
            entities=["全球碳排放"],
        )
        evidence = Evidence(
            "E1",
            "report",
            "全球碳排放",
            "",
            claims=[decline, growth],
        )
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_decline", "C_growth", ConflictType.SCOPE_CONFLICT, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            [evidence],
            "全球碳排放已经在下降了，对吗？",
        )

        assert len(filtered.get_conflicts()) == 1

    def test_founder_question_filters_company_metric_conflicts(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        employees_official = Claim(
            claim_id="C_employees_official",
            claim="公司员工总数为41000人",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
            numbers=["41000人"],
        )
        employees_media = Claim(
            claim_id="C_employees_media",
            claim="该公司员工已超过6万人",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
            numbers=["6万人"],
        )
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge(
                "C_employees_official",
                "C_employees_media",
                ConflictType.NUMERIC_CONFLICT,
                0.8,
            ),
        ]
        evidence = Evidence(
            "E1",
            "report",
            "星辰科技",
            "",
            claims=[employees_official, employees_media],
        )

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            [evidence],
            "星辰科技的创始人是谁？",
        )

        assert filtered.get_conflicts() == []

    def test_strategy_question_filters_unrelated_runtime_dispute(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        google = Claim(
            claim_id="C_google",
            claim="谷歌称经典超算需要10000年完成采样",
            claim_type=ClaimType.FACTUAL,
            entities=["谷歌", "IBM"],
            numbers=["10000年"],
        )
        ibm = Claim(
            claim_id="C_ibm",
            claim="IBM认为经典算法只需2.5天",
            claim_type=ClaimType.FACTUAL,
            entities=["谷歌", "IBM"],
            numbers=["2.5天"],
        )
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_google", "C_ibm", ConflictType.NUMERIC_CONFLICT, 0.9),
        ]
        evidence = Evidence("E1", "paper", "量子计算", "", claims=[google, ibm])

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            [evidence],
            "比较谷歌和IBM在量子计算路线上的不同策略。",
        )

        assert filtered.get_conflicts() == []

    def test_self_refutation_edge_requires_premise_validation_question(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        claim = Claim(
            claim_id="C_emissions",
            claim="全球碳排放仍在增长并创历史新高",
            claim_type=ClaimType.FACTUAL,
            entities=["全球碳排放"],
        )
        evidence = Evidence("E1", "report", "全球碳排放", "", claims=[claim])
        edge = ConflictEdge(
            "C_emissions",
            "C_emissions",
            ConflictType.SCOPE_CONFLICT,
            0.72,
            rationale="Claim says global emissions are growing or at a record high",
        )

        neutral_graph = EvidenceConflictGraph()
        neutral_graph.edges = [edge]
        neutral = pipeline._filter_conflict_graph_for_question(
            neutral_graph,
            [evidence],
            "全球碳排放现状如何？",
        )
        assert neutral.get_conflicts() == []

        validation_graph = EvidenceConflictGraph()
        validation_graph.edges = [edge]
        validation = pipeline._filter_conflict_graph_for_question(
            validation_graph,
            [evidence],
            "全球碳排放已经在下降了，对吗？",
        )
        assert len(validation.get_conflicts()) == 1

    def test_point_in_time_question_filters_expected_version_evolution(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        proposal = Claim(
            claim_id="C_proposal",
            claim="提案将罚款上限设定为3000万欧元或全球年营业额6%",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
            numbers=["3000万欧元", "6%"],
        )
        final = Claim(
            claim_id="C_final",
            claim="最终法案将罚款上限设定为3500万欧元或全球年营业额7%",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
            numbers=["3500万欧元", "7%"],
        )
        evidence_pool = [
            Evidence(
                "D002_c0",
                "report",
                "欧盟AI法案早期提案版本要点",
                "2021年初始提案。",
                date="2021-04-21",
                claims=[proposal],
            ),
            Evidence(
                "D001_c0",
                "official",
                "欧盟人工智能法案：从提案到立法",
                "2024年最终通过。",
                date="2024-03-13",
                claims=[final],
            ),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_proposal", "C_final", ConflictType.NUMERIC_CONFLICT, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "欧盟AI法案将违规罚款上限设定为多少？",
        )

        assert filtered.get_conflicts() == []

        comparison = EvidenceConflictGraph()
        comparison.edges = [
            ConflictEdge("C_proposal", "C_final", ConflictType.NUMERIC_CONFLICT, 0.9),
        ]
        compared = pipeline._filter_conflict_graph_for_question(
            comparison,
            evidence_pool,
            "欧盟AI法案早期提案与最终版本的罚款上限相比有何变化？",
        )

        assert len(compared.get_conflicts()) == 1

    def test_status_question_drops_compatible_nli_conflict_but_premise_keeps_it(
        self,
        pipeline_config,
    ):
        pipeline = _create_pipeline(pipeline_config)
        reported = Claim(
            claim_id="C_reported",
            claim="欧盟AI法案已无限期搁置",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
            source_span="reported_claim",
        )
        corrective = Claim(
            claim_id="C_corrective",
            claim="实际上，欧盟AI法案不仅已获通过，而且部分条款将于2025年开始生效",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        official = Claim(
            claim_id="C_official",
            claim="2024年3月13日，欧洲议会正式通过了《人工智能法案》",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        evidence_pool = [
            Evidence(
                "D006_c0",
                "blog",
                "AI监管政策争议",
                "",
                claims=[reported, corrective],
            ),
            Evidence("D001_c0", "official", "欧盟AI法案", "", claims=[official]),
        ]
        edges = [
            ConflictEdge(
                "C_corrective",
                "C_official",
                ConflictType.REFUTE,
                0.89,
                rationale="NLI contradiction: 0.89",
            ),
            ConflictEdge(
                "C_reported",
                "C_corrective",
                ConflictType.REFUTE,
                0.75,
                rationale="Negation contradiction on shared entities: '通过' vs '搁置'",
            ),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = list(edges)

        status_filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "欧盟AI法案目前处于什么状态？是否已经通过？",
        )

        assert [edge.source_id for edge in status_filtered.get_conflicts()] == ["C_reported"]

        premise_graph = EvidenceConflictGraph()
        premise_graph.edges = list(edges)
        premise_filtered = pipeline._filter_conflict_graph_for_question(
            premise_graph,
            evidence_pool,
            "欧盟AI法案已被搁置，目前尚未通过，对吗？",
        )

        assert len(premise_filtered.get_conflicts()) == 2

    def test_reported_claim_conflict_dedupe_prefers_stronger_evidence(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        reported = Claim(
            claim_id="C_reported",
            claim="欧盟AI法案已无限期搁置",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
            source_span="reported_claim",
        )
        official = Claim(
            claim_id="C_official",
            claim="欧盟AI法案已于2024年3月13日通过",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        blog = Claim(
            claim_id="C_blog",
            claim="欧盟AI法案的通过标志监管框架成型",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案"],
        )
        evidence_pool = [
            Evidence("E_reported", "blog", "争议报道", "", claims=[reported], relevance_score=0.4),
            Evidence("E_official", "official", "欧盟AI法案", "", claims=[official], relevance_score=0.95),
            Evidence("E_blog", "blog", "政策评论", "", claims=[blog], relevance_score=0.7),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_reported", "C_blog", ConflictType.REFUTE, 0.95),
            ConflictEdge("C_reported", "C_official", ConflictType.REFUTE, 0.8),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "欧盟AI法案是否已经通过？",
        )

        conflicts = filtered.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].target_id == "C_official"

    def test_reported_claim_dedupe_prefers_newer_same_tier_evidence(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        reported = Claim(
            claim_id="C_reported",
            claim="星辰科技于2010年成立",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
            source_span="reported_claim",
        )
        old_report = Claim(
            claim_id="C_old_report",
            claim="星辰科技由李明远博士于2012年创立",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
        )
        newer_wiki = Claim(
            claim_id="C_newer_wiki",
            claim="星辰科技由李明远博士于2012年在北京创立",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
        )
        evidence_pool = [
            Evidence("E_reported", "news", "不实报道", "", claims=[reported], relevance_score=0.8),
            Evidence("E_old", "report", "2022财报", "", date="2023-03-30", claims=[old_report]),
            Evidence("E_new", "wiki", "公司介绍", "", date="2024-01-15", claims=[newer_wiki]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_reported", "C_old_report", ConflictType.TEMPORAL_CONFLICT, 0.7),
            ConflictEdge("C_reported", "C_newer_wiki", ConflictType.TEMPORAL_CONFLICT, 0.7),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "星辰科技2010年成立，对吗？",
        )

        conflicts = filtered.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].target_id == "C_newer_wiki"

    def test_question_focus_filters_by_question_entity_and_year(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        reported = Claim(
            claim_id="C_reported",
            claim="星辰科技2023年营收突破800亿元",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
            numbers=["2023年", "800亿元"],
            time_expressions=["2023"],
            source_span="reported_claim",
        )
        official_2023 = Claim(
            claim_id="C_official_2023",
            claim="星辰科技2023财年全年营收达到612亿元人民币",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
            numbers=["2023", "612亿元"],
            time_expressions=["2023"],
        )
        old_2022 = Claim(
            claim_id="C_old_2022",
            claim="星辰科技2022财年全年营收为458亿元人民币",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
            numbers=["2022", "458亿元"],
            time_expressions=["2022"],
        )
        industry = Claim(
            claim_id="C_industry",
            claim="2023年AI芯片市场营收增长超过200%",
            claim_type=ClaimType.FACTUAL,
            entities=["AI芯片市场"],
            numbers=["2023年", "200%"],
            time_expressions=["2023"],
        )
        evidence_pool = [
            Evidence("E_reported", "news", "不实报道", "", claims=[reported], relevance_score=0.8),
            Evidence("E_official", "report", "2023财报", "", claims=[official_2023], relevance_score=0.9),
            Evidence("E_old", "report", "2022财报", "", claims=[old_2022], relevance_score=0.7),
            Evidence("E_industry", "report", "行业报告", "", claims=[industry], relevance_score=0.9),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_reported", "C_old_2022", ConflictType.NUMERIC_CONFLICT, 0.9),
            ConflictEdge("C_reported", "C_industry", ConflictType.NUMERIC_CONFLICT, 0.95),
            ConflictEdge("C_reported", "C_official_2023", ConflictType.NUMERIC_CONFLICT, 0.7),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "星辰科技2023年的营收是多少？",
        )

        conflicts = filtered.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].target_id == "C_official_2023"

    def test_question_focus_does_not_match_only_weak_ai_entity(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        medical = Claim(
            claim_id="C_medical",
            claim="AI医疗诊断也面临诸多挑战",
            claim_type=ClaimType.FACTUAL,
            entities=["AI医疗"],
        )
        shelved = Claim(
            claim_id="C_shelved",
            claim="欧盟AI法案已无限期搁置",
            claim_type=ClaimType.FACTUAL,
            entities=["AI", "欧盟AI法案"],
        )
        passed = Claim(
            claim_id="C_passed",
            claim="欧盟AI法案已于2024年3月通过",
            claim_type=ClaimType.FACTUAL,
            entities=["AI", "欧盟AI法案"],
        )
        evidence_pool = [
            Evidence("E_medical", "paper", "AI医疗诊断", "", entities=["AI医疗"], claims=[medical]),
            Evidence("E_blog", "blog", "AI监管争议", "", entities=["AI", "欧盟AI法案"], claims=[shelved]),
            Evidence("E_official", "official", "欧盟AI法案", "", entities=["AI", "欧盟AI法案"], claims=[passed]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_shelved", "C_passed", ConflictType.REFUTE, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "中国AI医疗诊断技术已经被证明全面超越人类医生，对吗？",
        )

        assert filtered.get_conflicts() == []

    def test_question_focus_requires_strong_entity_when_ai_is_broad(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        us_policy = Claim(
            claim_id="C_us",
            claim="美国目前尚未通过联邦层面的综合性AI立法",
            claim_type=ClaimType.FACTUAL,
            entities=["美国", "AI"],
        )
        shelved = Claim(
            claim_id="C_shelved",
            claim="欧盟AI法案已无限期搁置",
            claim_type=ClaimType.FACTUAL,
            entities=["AI", "欧盟AI法案"],
        )
        passed = Claim(
            claim_id="C_passed",
            claim="欧盟AI法案已于2024年3月通过",
            claim_type=ClaimType.FACTUAL,
            entities=["AI", "欧盟AI法案"],
        )
        evidence_pool = [
            Evidence("E_us", "official", "美国AI行政命令", "", entities=["美国", "AI"], claims=[us_policy]),
            Evidence("E_blog", "blog", "AI监管争议", "", entities=["AI", "欧盟AI法案"], claims=[shelved]),
            Evidence("E_official", "official", "欧盟AI法案", "", entities=["AI", "欧盟AI法案"], claims=[passed]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_shelved", "C_passed", ConflictType.REFUTE, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "美国已经通过了联邦层面的综合性AI立法，对吗？",
        )

        assert filtered.get_conflicts() == []

    def test_extrapolation_questions_drop_same_evidence_numeric_edges(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        quantum = Claim(
            claim_id="C_quantum",
            claim="谷歌量子处理器用约200秒完成采样任务，IBM认为经典算法约2.5天可完成",
            claim_type=ClaimType.FACTUAL,
            entities=["量子霸权", "谷歌", "IBM"],
            numbers=["200秒", "2.5天"],
        )
        evidence_pool = [
            Evidence("E_quantum", "paper", "量子霸权争议", "", entities=["量子霸权", "谷歌", "IBM"], claims=[quantum]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_quantum", "C_quantum", ConflictType.NUMERIC_CONFLICT, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "既然谷歌2019年就实现了量子霸权，量子计算机现在应该已经取代经典计算机了，对吧？",
        )

        assert filtered.get_conflicts() == []

    def test_extrapolation_questions_drop_global_emissions_self_refutation(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        emissions = Claim(
            claim_id="C_emissions",
            claim="2023年全球化石燃料CO2排放量预计达到368亿吨，较2022年增长1.1%，创历史新高",
            claim_type=ClaimType.FACTUAL,
            entities=["CO2", "碳排放"],
            numbers=["2023年", "368亿", "2022年", "1.1%"],
        )
        evidence_pool = [
            Evidence("E_emissions", "report", "全球碳计划2023年度报告", "", entities=["CO2", "碳排放"], claims=[emissions]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge(
                "C_emissions",
                "C_emissions",
                ConflictType.SCOPE_CONFLICT,
                0.72,
                rationale="Claim says global emissions are growing or at a record high",
            ),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "既然全球碳排放还在创新高，减排政策是不是完全无效？",
        )

        assert filtered.get_conflicts() == []

    def test_implication_questions_drop_process_self_refutation(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        process = Claim(
            claim_id="C_process",
            claim='"3nm"和"5nm"等命名已不再代表实际的物理栅极长度',
            claim_type=ClaimType.FACTUAL,
            entities=["芯片制程", "台积电"],
            numbers=["3nm", "5nm"],
        )
        evidence_pool = [
            Evidence("E_process", "blog", "关于芯片制程的常见误解", "", entities=["芯片制程", "台积电"], claims=[process]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge(
                "C_process",
                "C_process",
                ConflictType.DEFINITIONAL_CONFLICT,
                0.72,
                rationale="Process-node naming is explicitly distinguished from physical dimensions",
            ),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "台积电的3nm工艺意味着晶体管栅极长度只有3纳米，对吗？",
        )

        assert filtered.get_conflicts() == []

    def test_advanced_packaging_self_refutation_requires_packaging_focus(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        packaging = Claim(
            claim_id="C_packaging",
            claim="随着摩尔定律放缓，先进封装成为延续芯片性能提升的关键路径",
            claim_type=ClaimType.FACTUAL,
            entities=["先进封装", "台积电"],
        )
        evidence_pool = [
            Evidence("E_packaging", "report", "半导体先进封装技术演进", "", entities=["先进封装", "台积电"], claims=[packaging]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge(
                "C_packaging",
                "C_packaging",
                ConflictType.DEFINITIONAL_CONFLICT,
                0.72,
                rationale="Claim frames advanced packaging as one improvement path, not a complete process replacement",
            ),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "台积电的3nm工艺意味着晶体管栅极长度只有3纳米，对吗？",
        )

        assert filtered.get_conflicts() == []

    def test_process_self_refutation_kept_for_matching_question_focus(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        process = Claim(
            claim_id="C_process",
            claim='"3nm"和"5nm"等命名已不再代表实际的物理栅极长度',
            claim_type=ClaimType.FACTUAL,
            entities=["芯片制程"],
            numbers=["3nm", "5nm"],
        )
        evidence_pool = [
            Evidence("E_process", "blog", "关于芯片制程的常见误解", "", entities=["芯片制程"], claims=[process]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_process", "C_process", ConflictType.DEFINITIONAL_CONFLICT, 0.72),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            '芯片制程中的"3nm"是否代表实际的3纳米物理尺寸？',
        )

        assert len(filtered.get_conflicts()) == 1

    def test_co2_entity_matches_carbon_emissions_question(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        emissions = Claim(
            claim_id="C_emissions",
            claim="2023年全球化石燃料CO2排放量预计达到368亿吨，较2022年增长1.1%，创历史新高",
            claim_type=ClaimType.FACTUAL,
            entities=["CO2", "化石燃料"],
            numbers=["2023年", "368亿", "2022年", "1.1%"],
        )
        evidence_pool = [
            Evidence("E_emissions", "report", "全球碳计划2023年度报告", "", entities=["CO2", "碳排放"], claims=[emissions]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge(
                "C_emissions",
                "C_emissions",
                ConflictType.SCOPE_CONFLICT,
                0.72,
                rationale="Claim says global emissions are growing or at a record high",
            ),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "全球碳排放已经在下降了，对吗？",
        )

        assert len(filtered.get_conflicts()) == 1

    def test_contrast_entity_does_not_satisfy_question_focus(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        us_law = Claim(
            claim_id="C_us_law",
            claim="与欧盟AI法案不同，美国目前尚未通过联邦层面的综合性AI立法",
            claim_type=ClaimType.FACTUAL,
            entities=["欧盟AI法案", "美国", "AI"],
        )
        evidence_pool = [
            Evidence("E_us", "official", "美国AI行政命令", "", entities=["欧盟AI法案", "美国", "AI"], claims=[us_law]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_us_law", "C_us_law", ConflictType.DEFINITIONAL_CONFLICT, 0.72),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "欧盟AI法案目前处于什么状态？是否已经通过？",
        )

        assert filtered.get_conflicts() == []

    def test_assertion_question_year_is_not_used_as_strict_filter(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        reported = Claim(
            claim_id="C_reported",
            claim="星辰科技于2010年成立",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
            numbers=["2010年"],
            time_expressions=["2010"],
            source_span="reported_claim",
        )
        corrected = Claim(
            claim_id="C_corrected",
            claim="星辰科技由李明远博士于2012年在北京创立",
            claim_type=ClaimType.FACTUAL,
            entities=["星辰科技"],
            numbers=["2012年"],
            time_expressions=["2012"],
        )
        evidence_pool = [
            Evidence("E_reported", "news", "不实报道", "", claims=[reported], relevance_score=0.8),
            Evidence("E_wiki", "wiki", "公司介绍", "", claims=[corrected], relevance_score=0.9),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_reported", "C_corrected", ConflictType.TEMPORAL_CONFLICT, 0.7),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "星辰科技2010年成立，对吗？",
        )

        assert len(filtered.get_conflicts()) == 1

    def test_question_focus_requires_at_least_one_question_entity_match(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        relevant = Claim(
            claim_id="C_relevant",
            claim="谷歌称量子霸权计算需要10000年",
            claim_type=ClaimType.FACTUAL,
            entities=["谷歌", "量子霸权"],
            numbers=["10000年"],
        )
        relevant_other = Claim(
            claim_id="C_relevant_other",
            claim="IBM认为该任务可在2.5天内完成",
            claim_type=ClaimType.FACTUAL,
            entities=["IBM", "量子霸权"],
            numbers=["2.5"],
        )
        unrelated = Claim(
            claim_id="C_unrelated",
            claim="3nm命名不代表实际栅极长度",
            claim_type=ClaimType.FACTUAL,
            entities=["芯片制程"],
            numbers=["3"],
        )
        unrelated_other = Claim(
            claim_id="C_unrelated_other",
            claim="3nm工艺实际栅极长度约20nm",
            claim_type=ClaimType.FACTUAL,
            entities=["芯片制程"],
            numbers=["20"],
        )
        evidence_pool = [
            Evidence("E_quantum", "paper", "量子霸权", "", claims=[relevant, relevant_other]),
            Evidence("E_chip", "blog", "芯片制程", "", claims=[unrelated, unrelated_other]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_relevant", "C_relevant_other", ConflictType.NUMERIC_CONFLICT, 0.9),
            ConflictEdge("C_unrelated", "C_unrelated_other", ConflictType.NUMERIC_CONFLICT, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            "谷歌2019年宣称实现量子霸权，IBM对此有何不同看法？",
        )

        assert [(edge.source_id, edge.target_id) for edge in filtered.get_conflicts()] == [
            ("C_relevant", "C_relevant_other")
        ]

    def test_question_focus_uses_evidence_entities_without_broad_substring_leak(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        focused = Claim(
            claim_id="C_focused",
            claim="3nm命名不代表实际3纳米尺寸",
            claim_type=ClaimType.FACTUAL,
            entities=["芯片制程", "纳米"],
        )
        focused_other = Claim(
            claim_id="C_focused_other",
            claim="3nm工艺实际栅极长度约20nm",
            claim_type=ClaimType.FACTUAL,
            entities=["芯片制程", "纳米"],
        )
        broad = Claim(
            claim_id="C_broad",
            claim="中芯国际已量产14nm工艺",
            claim_type=ClaimType.FACTUAL,
            entities=["中国芯片", "中芯国际"],
        )
        evidence_pool = [
            Evidence("E_focused", "blog", "芯片制程", "", entities=["芯片制程"], claims=[focused, focused_other]),
            Evidence("E_broad", "report", "中国芯片", "", entities=["中国芯片"], claims=[broad]),
        ]
        graph = EvidenceConflictGraph()
        graph.edges = [
            ConflictEdge("C_focused", "C_focused_other", ConflictType.NUMERIC_CONFLICT, 0.9),
            ConflictEdge("C_focused", "C_broad", ConflictType.NUMERIC_CONFLICT, 0.9),
        ]

        filtered = pipeline._filter_conflict_graph_for_question(
            graph,
            evidence_pool,
            '芯片制程中的"3nm"是否代表实际的3纳米物理尺寸？',
        )

        assert [(edge.source_id, edge.target_id) for edge in filtered.get_conflicts()] == [
            ("C_focused", "C_focused_other")
        ]

    def test_answerability_guard_abstains_on_exact_question_with_approximate_evidence(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        evidence = [
            Evidence(
                "D040_c0",
                "report",
                "2023年中国新能源汽车市场年终报告",
                "2023年中国新能源汽车销量949.5万辆。其中纯电动汽车销量667.6万辆，"
                "插电式混合动力销量268.6万辆，燃料电池汽车销量约6000辆。",
                relevance_score=0.95,
            ),
            Evidence(
                "D043_c0",
                "report",
                "比亚迪vs特斯拉：2023年全球销量对比",
                "比亚迪全球销售约302.4万辆新能源汽车，特斯拉全球交付约181万辆纯电动汽车。",
                relevance_score=0.4,
            ),
        ]

        answer, claims, steps, guard = pipeline._apply_answerability_guard(
            "2023年中国氢燃料电池汽车的具体销量是多少？",
            "2023年中国氢燃料电池汽车的具体销量约为6000辆。",
            [],
            [],
            evidence,
        )

        assert guard == {"action": "exact_value_gap_abstain"}
        assert answer.startswith("无法给出精确数字")
        assert "约6000辆" in answer
        assert "D040_c0" in answer
        assert claims[0].supporting_evidence[0] == "D040_c0"
        assert steps[0].evidence_ids[0] == "D040_c0"

    def test_answerability_guard_turns_assertion_abstention_into_premise_correction(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)

        answer, claims, steps, guard = pipeline._apply_answerability_guard(
            "既然IPCC说全球升温会在2020年代达到1.5度，那巴黎协定的目标是不是已经失败了？",
            "根据现有证据无法回答此问题。证据中提及IPCC报告指出全球升温将在2020年代达到1.5°C，"
            "但并未明确说明巴黎协定目标是否因此失败。",
            [],
            [],
            [],
        )

        assert guard == {"action": "premise_abstention_corrected"}
        assert answer.startswith("该说法不准确")
        assert "不能根据现有证据断定" in answer
        assert "巴黎协定的目标" in answer
        assert "无法回答此问题" not in answer
        assert claims[0].claim == "问题中的断言缺乏充分证据支持"
        assert steps[0].description.startswith("识别到这是前提/断言验证问题")

    def test_answerability_guard_corrects_premise_when_answer_repeats_unreliable_report(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        evidence = [
            Evidence(
                evidence_id="D014_c0",
                source="news",
                title="媒体对星辰科技的不实报道",
                text_span="另有报道称该公司目前员工已超过6万人，且星辰科技于2010年成立。",
                relevance_score=0.8,
            ),
            Evidence(
                evidence_id="D013_c0",
                source="report",
                title="星辰科技公司介绍与发展历程",
                text_span="星辰科技由李明远博士于2012年在北京创立。2019年开始研发AI专用芯片。",
                relevance_score=0.95,
            ),
            Evidence(
                evidence_id="D010_c0",
                source="report",
                title="星辰科技2023财年年报摘要",
                text_span="截至2023年末，星辰科技员工总数为41000人。",
                relevance_score=0.9,
            ),
        ]

        answer, claims, steps, guard = pipeline._apply_answerability_guard(
            "星辰科技2010年成立，是中国最早的AI芯片公司，目前员工超过6万人，对吗？",
            "根据现有证据：媒体报道称星辰科技于2010年成立，是国内最早的AI芯片公司，"
            "目前员工已超过6万人。以上部分数据需进一步核实，请以公司正式财报为准。",
            [],
            [],
            evidence,
        )

        assert guard == {"action": "premise_noncorrection_corrected"}
        assert answer.startswith("该说法不准确")
        assert "2012年" in answer
        assert "41000人" in answer
        assert "D014_c0" not in claims[0].supporting_evidence
        assert set(claims[0].supporting_evidence) == {"D013_c0", "D010_c0"}
        assert steps[0].description.startswith("识别到这是前提/断言验证问题")

    def test_point_in_time_value_guard_filters_historical_version_noise(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        evidence = [
            Evidence(
                evidence_id="D002_c0",
                source="report",
                title="欧盟AI法案早期提案版本要点",
                text_span="2021年初始提案规定罚款上限为3000万欧元或全球年营业额6%。",
                date="2021-04-21",
                relevance_score=0.7,
            ),
            Evidence(
                evidence_id="D001_c0",
                source="official",
                title="欧盟人工智能法案：从提案到立法",
                text_span="违规者最高可被处以3500万欧元或全球年营业额7%的罚款。",
                date="2024-03-13",
                relevance_score=0.95,
            ),
        ]

        answer, claims, steps, guard = pipeline._apply_point_in_time_value_guard(
            "欧盟AI法案将违规罚款上限设定为多少？",
            "证据存在冲突：早期提案是3000万欧元或6%，最终通过版本是3500万欧元或7%。\n"
            "引用证据：[D002_c0] [D001_c0]",
            [
                AnswerClaim(
                    claim="罚款上限存在早期和最终版本差异",
                    supporting_evidence=["D002_c0", "D001_c0"],
                )
            ],
            [],
            evidence,
            EvidenceConflictGraph(),
        )

        assert guard == {
            "action": "point_in_time_value_answer",
            "selected_evidence": "D001_c0",
        }
        assert "3500万欧元" in answer
        assert "[D001_c0]" in answer
        assert "D002_c0" not in answer
        assert claims[0].supporting_evidence == ["D001_c0"]
        assert steps[0].evidence_ids == ["D001_c0"]

    def test_concise_value_guard_compresses_simple_numeric_answer(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        evidence = [
            Evidence(
                evidence_id="D078_c0",
                source="report",
                title="嫦娥六号任务样本",
                text_span="嫦娥六号于2024年6月完成人类首次月球背面采样返回，带回约1935.3克月壤样本。",
                relevance_score=0.95,
            ),
            Evidence(
                evidence_id="D079_c0",
                source="report",
                title="嫦娥六号科研进展",
                text_span="后续研究将分析月球背面样本的矿物组成。",
                relevance_score=0.7,
            ),
        ]

        answer, claims, steps, guard = pipeline._apply_concise_value_answer_guard(
            "嫦娥六号带回的月壤样本大约有多少克？",
            "嫦娥六号带回的月壤样本大约为1935.3克。引用证据：[D078_c0]",
            [],
            [],
            evidence,
            EvidenceConflictGraph(),
        )

        assert guard == {
            "action": "concise_value_answer",
            "selected_evidence": "D078_c0",
        }
        assert answer == "约1935.3克。引用证据：[D078_c0]"
        assert claims[0].claim == "约1935.3克"
        assert claims[0].supporting_evidence == ["D078_c0"]
        assert steps[0].evidence_ids == ["D078_c0"]

    def test_concise_value_guard_handles_irrelevant_conflict_preamble_for_simple_value(
        self,
        pipeline_config,
    ):
        pipeline = _create_pipeline(pipeline_config)
        evidence = [
            Evidence(
                evidence_id="D073_c0",
                source="paper",
                title="RAG系统幻觉率评估基准HAAG",
                text_span="HAAG基准包含2000个问题-答案对，覆盖6个领域。",
                relevance_score=0.95,
            ),
            Evidence(
                evidence_id="D064_c0",
                source="paper",
                title="Agentic RAG综述",
                text_span="Agentic RAG成为复杂任务的新方向。",
                relevance_score=0.6,
            ),
        ]
        graph = EvidenceConflictGraph()
        graph.add_edge(ConflictEdge("C_haag", "C_rag", ConflictType.REFUTE, 0.7))

        answer, claims, steps, guard = pipeline._apply_concise_value_answer_guard(
            "RAG系统幻觉率评估基准HAAG包含多少个问题-答案对？",
            "证据存在冲突：D073_c0 与 D064_c0 内容不一致。综合判断，HAAG基准包含2000个问题-答案对。",
            [],
            [],
            evidence,
            graph,
        )

        assert guard == {
            "action": "concise_value_answer",
            "selected_evidence": "D073_c0",
        }
        assert answer == "2000个问题-答案对。引用证据：[D073_c0]"
        assert claims[0].supporting_evidence == ["D073_c0"]
        assert steps[0].evidence_ids == ["D073_c0"]

    def test_abstention_conflict_prefix_guard_strips_unrelated_preamble(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)

        answer, claims, steps, guard = pipeline._apply_abstention_conflict_prefix_guard(
            "全球禁止人类生殖细胞基因编辑的70个国家具体是哪些？",
            "证据存在冲突：D076_c0 提到无关大豆实验，而 D089_c0 提到递送系统优化。"
            "综合判断，根据现有证据无法回答此问题。证据仅提到已有超过70个国家禁止相关临床应用，"
            "但未列出具体国家名单。",
            [],
            [],
        )

        assert guard == {"action": "abstention_conflict_prefix_stripped"}
        assert answer.startswith("根据现有证据无法回答此问题")
        assert "证据存在冲突" not in answer
        assert claims == []
        assert steps[0].description.startswith("识别到答案主体为不可答")

    def test_final_confidence_uses_verification_evidence_and_conflicts(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        evidence = [
            Evidence(
                evidence_id="E1",
                source="report",
                title="量子计算进展",
                text_span="Willow 是 2024 年发布的量子处理器。",
                credibility_score=0.9,
                recency_score=0.9,
                relevance_score=0.95,
            )
        ]
        claims = [
            AnswerClaim(
                claim="Willow 是 2024 年发布的量子处理器。",
                supporting_evidence=["E1"],
                confidence=0.9,
                verification_status=VerificationStatus.SUPPORTED,
                support_type="direct",
            )
        ]
        steps = [ReasoningStep(step=1, description="读取证据", evidence_ids=["E1"], confidence=0.85)]
        supported_report = VerificationReport(
            claim_verifications=[
                {"claim": claims[0].claim, "status": "SUPPORTED", "confidence": 0.92}
            ],
            overall_status=VerificationStatus.SUPPORTED,
        )
        refuted_report = VerificationReport(
            claim_verifications=[
                {"claim": claims[0].claim, "status": "REFUTED", "confidence": 0.9}
            ],
            overall_status=VerificationStatus.REFUTED,
        )
        conflict_graph = EvidenceConflictGraph()
        conflict_graph.add_edge(
            ConflictEdge(
                source_id="E1",
                target_id="E2",
                conflict_type=ConflictType.NUMERIC_CONFLICT,
                severity="high",
                confidence=0.9,
            )
        )

        supported = pipeline._estimate_final_confidence(
            answer="Willow 是 2024 年发布的量子处理器。",
            answer_claims=claims,
            reasoning_chain=steps,
            evidence_pool=evidence,
            conflict_graph=EvidenceConflictGraph(),
            verification_report=supported_report,
            uncertainty=pipeline.uncertainty_controller.get_uncertainty_breakdown(
                [], evidence, EvidenceConflictGraph()
            ),
            answerability_guard=None,
        )
        refuted = pipeline._estimate_final_confidence(
            answer="Willow 是 2025 年发布的量子处理器。",
            answer_claims=claims,
            reasoning_chain=steps,
            evidence_pool=evidence,
            conflict_graph=conflict_graph,
            verification_report=refuted_report,
            uncertainty=pipeline.uncertainty_controller.get_uncertainty_breakdown(
                [], evidence, conflict_graph
            ),
            answerability_guard=None,
        )

        assert supported > 0.65
        assert refuted < 0.35
        assert supported - refuted > 0.35

    def test_final_confidence_scores_reasonable_abstention(self, pipeline_config):
        pipeline = _create_pipeline(pipeline_config)
        not_enough_report = VerificationReport(
            claim_verifications=[],
            overall_status=VerificationStatus.NOT_ENOUGH_INFO,
        )
        supported_report = VerificationReport(
            claim_verifications=[
                {"claim": "已有直接证据", "status": "SUPPORTED", "confidence": 0.95}
            ],
            overall_status=VerificationStatus.SUPPORTED,
        )
        high_uncertainty = pipeline.uncertainty_controller.get_uncertainty_breakdown(
            [], [], EvidenceConflictGraph()
        )
        supported_evidence = [
            Evidence(
                evidence_id="E1",
                source="report",
                title="直接证据",
                text_span="已有直接证据。",
                credibility_score=0.95,
                recency_score=0.9,
                relevance_score=0.95,
            )
        ]
        low_uncertainty = pipeline.uncertainty_controller.get_uncertainty_breakdown(
            [], supported_evidence, EvidenceConflictGraph()
        )

        justified = pipeline._estimate_final_confidence(
            answer="证据不足，无法回答。",
            answer_claims=[],
            reasoning_chain=[],
            evidence_pool=[],
            conflict_graph=EvidenceConflictGraph(),
            verification_report=not_enough_report,
            uncertainty=high_uncertainty,
            answerability_guard={"action": "abstain"},
        )
        unjustified = pipeline._estimate_final_confidence(
            answer="证据不足，无法回答。",
            answer_claims=[],
            reasoning_chain=[],
            evidence_pool=supported_evidence,
            conflict_graph=EvidenceConflictGraph(),
            verification_report=supported_report,
            uncertainty=low_uncertainty,
            answerability_guard=None,
        )

        assert justified > 0.65
        assert unjustified < justified
        assert unjustified < 0.55

    def test_full_pipeline_run(self, pipeline_config):
        """Test that the full pipeline runs end-to-end with mock LLM."""
        pipeline = _create_pipeline(pipeline_config)

        docs = [
            {"id": "D1", "text": "量子计算是一种利用量子力学原理进行信息处理的计算方式。"
                      "与传统计算机不同，量子计算机使用量子比特。"
                      "2024年谷歌发布了最新的量子处理器Willow。",
             "title": "量子计算技术综述"},
            {"id": "D2", "text": "人工智能在医疗影像诊断中表现出色。"
                      "研究显示AI诊断准确率达到95%。"
                      "但AI在复杂病例中仍存在局限性。",
             "title": "AI医疗诊断报告"},
            {"id": "D3", "text": "量子比特是量子计算的基本单元。"
                      "超导量子比特和离子阱是两种主要技术路线。"
                      "谷歌IBM等公司都在推进量子计算研究。",
             "title": "量子比特技术"},
            {"id": "D4", "text": "2024年量子计算领域取得多项突破。"
                      "微软发布了拓扑量子比特的研究成果。"
                      "IBM推出了超过1000量子比特的处理器。",
             "title": "2024量子计算进展"},
            {"id": "D5", "text": "量子计算的应用前景包括药物发现、密码学和优化问题。"
                      "不过目前的量子计算机还处于NISQ时代，存在噪声问题。",
             "title": "量子计算应用"},
        ]
        pipeline.index_documents(docs)

        output = pipeline.query("量子计算的最新进展是什么？")
        assert isinstance(output, VeraRAGOutput)
        assert output.question == "量子计算的最新进展是什么？"
        assert isinstance(output.answer, str)
        assert len(output.answer) > 0
        assert 0 <= output.confidence <= 1

    def test_pipeline_streaming(self, pipeline_config):
        """Test streaming callback receives events."""
        pipeline = _create_pipeline(pipeline_config)

        docs = [{"id": "D1", "text": "测试文档内容。" * 10, "title": "测试"}]
        pipeline.index_documents(docs)

        events = []
        def callback(event_type, data):
            events.append((event_type, data))

        output = pipeline.query_stream("测试问题", max_rounds=1, callback=callback)

        assert isinstance(output, VeraRAGOutput)
        assert len(events) > 0
        event_types = [e[0] for e in events]
        assert "complete" in event_types

    def test_pipeline_with_conflicts_disabled(self, pipeline_config):
        """Test pipeline runs with conflict detection disabled."""
        pipeline_config["pipeline"]["enable_conflict_graph"] = False
        pipeline = _create_pipeline(pipeline_config)

        docs = [{"id": "D1", "text": "测试内容", "title": "T"}]
        pipeline.index_documents(docs)

        output = pipeline.query("测试")
        assert isinstance(output, VeraRAGOutput)

    def test_pipeline_with_verification_disabled(self, pipeline_config):
        """Test pipeline skips verification when disabled."""
        pipeline_config["pipeline"]["enable_verification"] = False
        pipeline = _create_pipeline(pipeline_config)

        docs = [{"id": "D1", "text": "测试内容", "title": "T"}]
        pipeline.index_documents(docs)

        output = pipeline.query("测试")
        assert isinstance(output, VeraRAGOutput)

    def test_pipeline_minimal_config(self):
        """Test pipeline with minimal configuration."""
        pipeline = _create_pipeline({})

        docs = [{"id": "D1", "text": "测试内容", "title": "T"}]
        pipeline.index_documents(docs)

        output = pipeline.query("测试")
        assert isinstance(output, VeraRAGOutput)
