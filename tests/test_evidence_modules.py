"""Tests for evidence modules: extractor, evidence_scorer."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever.base import RetrievalResult
from src.utils.data_structures import (
    Claim,
    ClaimType,
    ConflictEdge,
    ConflictType,
    Evidence,
    EvidenceConflictGraph,
)


class MockLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        return '[{"claim": "测试声明", "claim_type": "factual", "confidence": 0.8}]'


class JsonLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self.response


# --- EvidenceNormalizer Tests ---

class TestEvidenceNormalizer:
    def _make_evidence(self, evidence_id, cred=0.5, rec=0.5, rel=0.5):
        return Evidence(
            evidence_id=evidence_id,
            source="fixture",
            title=f"Evidence {evidence_id}",
            text_span=f"text {evidence_id}",
            credibility_score=cred,
            recency_score=rec,
            relevance_score=rel,
        )

    def test_normalize_retrieval_results_preserves_metadata_and_clamps_relevance(self):
        from src.evidence.normalizer import EvidenceNormalizer

        normalizer = EvidenceNormalizer()
        results = [
            RetrievalResult(
                doc_id="D1",
                content="High score content",
                title="High",
                score=1.7,
                metadata={
                    "source": "Nature",
                    "date": "2026-06-18",
                    "author": "Researcher",
                    "url": "https://example.test/high",
                },
            ),
            RetrievalResult(
                doc_id="D2",
                content="Negative score content",
                title="Low",
                score=-0.25,
                metadata={"source": "unknown"},
            ),
        ]

        evidence = normalizer.normalize_retrieval_results(results)

        assert len(evidence) == 2
        assert evidence[0].source == "Nature"
        assert evidence[0].title == "High"
        assert evidence[0].text_span == "High score content"
        assert evidence[0].date == "2026-06-18"
        assert evidence[0].author == "Researcher"
        assert evidence[0].url == "https://example.test/high"
        assert evidence[0].relevance_score == 1.0
        assert evidence[0].credibility_score == 0.9
        assert evidence[1].relevance_score == 0.0

    def test_estimate_recency_accepts_formats_and_caps_future_dates(self):
        from src.evidence.normalizer import EvidenceNormalizer

        normalizer = EvidenceNormalizer()

        assert normalizer._estimate_recency(None) == 0.5
        assert normalizer._estimate_recency("not-a-date") == 0.5
        current_year = datetime.now().year
        assert normalizer._estimate_recency(f"{current_year}/06/18") == 1.0
        assert normalizer._estimate_recency(f"18-06-{current_year}") == 1.0
        assert normalizer._estimate_recency(datetime(current_year, 6, 18)) == 1.0
        assert normalizer._estimate_recency(f"{current_year + 9}-01-01") == 1.0

    def test_filter_low_quality_and_rank_by_combined_score(self):
        from src.evidence.normalizer import EvidenceNormalizer

        normalizer = EvidenceNormalizer()
        low = self._make_evidence("low", cred=0.1, rec=0.1, rel=0.1)
        mid = self._make_evidence("mid", cred=0.5, rec=0.5, rel=0.5)
        high = self._make_evidence("high", cred=0.9, rec=0.9, rel=0.9)

        filtered = normalizer.filter_low_quality([low, mid, high], min_score=0.4)
        ranked = normalizer.rank_by_quality([mid, high, low])

        assert [ev.evidence_id for ev in filtered] == ["mid", "high"]
        assert [ev.evidence_id for ev in ranked] == ["high", "mid", "low"]


# --- EvidenceExtractor Tests ---

class TestEvidenceExtractor:
    def test_extract_from_text(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())
        evidence = extractor.extract_from_text(
            text="量子计算是利用量子力学原理进行计算的技术。",
            source="test",
            title="量子计算",
        )
        assert isinstance(evidence, Evidence)
        assert evidence.source == "test"
        assert evidence.title == "量子计算"
        assert evidence.text_span != ""

    def test_extract_from_text_with_metadata(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())
        evidence = extractor.extract_from_text(
            text="测试内容",
            source="wiki",
            title="T",
            metadata={"author": "test_author"},
        )
        assert evidence.source == "wiki"

    def test_extract_claims_rule_based(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())
        evidence = extractor.extract_from_text(
            text="2024年全球AI市场规模达到500亿美元。比亚迪2024年营收增长30%。",
            source="report",
            title="市场报告",
        )
        # Rule-based extraction should detect claims
        assert isinstance(evidence, Evidence)

    def test_extract_claims_splits_chinese_sentences(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        claims = extractor._extract_claims(
            "欧盟AI法案已于2024年3月通过。部分条款将于2025年开始生效；该法案并未禁止所有人脸识别。"
        )

        assert [claim.claim for claim in claims] == [
            "欧盟AI法案已于2024年3月通过",
            "部分条款将于2025年开始生效",
            "该法案并未禁止所有人脸识别",
        ]
        assert claims[0].numbers == ["2024年", "3月"]
        assert claims[1].numbers == ["2025年"]

    def test_extract_embedded_reported_and_corrective_claims(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        claims = extractor._extract_claims(
            "有报道称欧盟AI法案禁止所有AI人脸识别，这也是不准确的——法案仅禁止实时远程生物识别。"
        )

        reported = [claim for claim in claims if claim.source_span == "reported_claim"]
        corrective = [claim for claim in claims if claim.source_span == "corrective_claim"]
        plain = [claim for claim in claims if claim.source_span is None]
        assert [claim.claim for claim in reported] == ["欧盟AI法案禁止所有AI人脸识别"]
        assert [claim.claim for claim in corrective] == ["欧盟AI法案仅禁止实时远程生物识别"]
        assert plain == []
        assert "欧盟AI法案" in reported[0].entities
        assert "欧盟AI法案" in corrective[0].entities

    def test_decimal_numbers_do_not_split_claims(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        claims = extractor._extract_claims(
            "IBM认为经优化的经典算法可在2.5天内完成相同任务。"
            "比亚迪在全球销售了约302.4万辆新能源汽车。"
        )

        assert [claim.claim for claim in claims] == [
            "IBM认为经优化的经典算法可在2.5天内完成相同任务",
            "比亚迪在全球销售了约302.4万辆新能源汽车",
        ]
        assert claims[0].numbers == ["2.5天"]
        assert claims[1].numbers == ["302.4万"]

    def test_comparative_estimate_sentence_splits_atomic_claims(self):
        from src.evidence.conflict_graph import ConflictGraphBuilder
        from src.evidence.extractor import EvidenceExtractor

        extractor = EvidenceExtractor(llm_client=MockLLM())
        evidence = extractor.extract_from_text(
            "基于古气候数据和能量平衡模型，我们估计ECS可能为2.5°C"
            "（90%置信区间：1.8-3.6°C），略低于IPCC AR6的最佳估计值3.0°C。",
            source="paper",
            title="气候敏感度估计",
        )

        claims = evidence.claims
        assert [claim.source_span for claim in claims] == [
            "comparative_subject_claim",
            "comparative_reference_claim",
        ]
        assert claims[0].numbers[0] == "2.5°C"
        assert claims[1].numbers == ["3.0°C"]

        graph = ConflictGraphBuilder(
            {"conflict_graph": {"enable_nli": False, "compare_within_evidence": False}}
        ).build_graph([evidence], use_llm=False)

        assert graph.get_conflicts()

    def test_extract_chinese_entities(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        entities = extractor._extract_entities("欧盟AI法案已通过。星辰科技2023年营收为612亿元。")

        assert "欧盟AI法案" in entities
        assert "星辰科技" in entities

    def test_extract_uppercase_digit_acronym_entity(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        entities = extractor._extract_entities("全球化石燃料CO2排放为368亿吨。")

        assert "CO2" in entities
        assert "CO" not in entities

    def test_extract_length_units_with_numbers(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        claim = extractor._make_claim('台积电的"3nm"工艺实际栅极长度约为20nm以上。')

        assert "3nm" in claim.numbers
        assert "20nm" in claim.numbers

    def test_extract_duration_and_qubit_units_with_numbers(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        claim = extractor._make_claim("谷歌53量子比特处理器完成任务，IBM称经典算法约2.5天可完成。")

        assert "53量子比特" in claim.numbers
        assert "2.5天" in claim.numbers

    def test_extract_chinese_entities_trims_predicate_suffix(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        entities = extractor._extract_entities("有报道称欧盟AI法案禁止所有AI人脸识别。")

        assert "欧盟AI法案" in entities
        assert "有报道称欧盟AI法案禁止所有AI" not in entities

    def test_normalize_ai_law_alias(self):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        entities = extractor._extract_entities("欧盟人工智能法案已通过。")

        assert "欧盟AI法案" in entities
        assert "AI法案" in entities

    @pytest.mark.parametrize(
        ("claim_text", "expected_type"),
        [
            ("比亚迪2024年营收增长30%。", ClaimType.NUMERICAL),
            ("欧盟AI法案于2024年3月通过。", ClaimType.TEMPORAL),
            ("由于需求下降，星辰科技营收减少。", ClaimType.CAUSAL),
            ("ECS估计值2.5°C低于IPCC AR6的3.0°C。", ClaimType.COMPARATIVE),
            ("量子霸权是指量子设备在特定任务上超过经典计算机。", ClaimType.DEFINITIONAL),
            ("该预测可能存在不确定性。", ClaimType.UNCERTAINTY),
        ],
    )
    def test_classify_chinese_claim_types(self, claim_text, expected_type):
        from src.evidence.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=MockLLM())

        assert extractor._classify_claim_type(claim_text) == expected_type

    def test_extract_claims_with_llm_accepts_list_json_and_sanitizes_fields(self):
        from src.evidence.extractor import EvidenceExtractor

        extractor = EvidenceExtractor(llm_client=JsonLLM("""
        [
          {
            "claim": "星辰科技营收增长30%",
            "claim_type": "numeric",
            "entities": "星辰科技",
            "numbers": [30, " "],
            "time_expressions": "2024",
            "verifiable": "false",
            "support_type": "DIRECT"
          },
          "bad row",
          {
            "claim": "ECS低于IPCC估计",
            "claim_type": "unknown",
            "entities": ["ECS", 123],
            "support_type": "unsupported"
          },
          {
            "claim": "第三条不应超过max_claims",
            "claim_type": "factual"
          }
        ]
        """))

        claims = extractor.extract_claims_with_llm("fallback text", max_claims=3)

        assert [claim.claim for claim in claims] == [
            "星辰科技营收增长30%",
            "ECS低于IPCC估计",
        ]
        assert claims[0].claim_type == ClaimType.NUMERICAL
        assert claims[0].entities == ["星辰科技"]
        assert claims[0].numbers == ["30"]
        assert claims[0].time_expressions == ["2024"]
        assert claims[0].verifiable is False
        assert claims[0].support_type == "direct"
        assert claims[1].claim_type == ClaimType.FACTUAL
        assert claims[1].entities == ["ECS", "123"]
        assert claims[1].support_type == "none"

    def test_extract_claims_with_llm_respects_zero_and_rejects_bad_max(self):
        from src.evidence.extractor import EvidenceExtractor

        extractor = EvidenceExtractor(llm_client=MockLLM())

        assert extractor.extract_claims_with_llm("不会调用 LLM", max_claims=0) == []
        with pytest.raises(ValueError, match="max_claims"):
            extractor.extract_claims_with_llm("bad", max_claims=-1)
        with pytest.raises(TypeError, match="max_claims"):
            extractor.extract_claims_with_llm("bad", max_claims=True)

    def test_extract_claims_with_llm_falls_back_for_non_list_payload(self):
        from src.evidence.extractor import EvidenceExtractor

        extractor = EvidenceExtractor(llm_client=JsonLLM('{"claims": "not-a-list"}'))

        claims = extractor.extract_claims_with_llm("欧盟AI法案已于2024年3月通过。")

        assert len(claims) == 1
        assert claims[0].claim == "欧盟AI法案已于2024年3月通过"

    def test_extract_claims_with_llm_skips_empty_claim_and_falls_back_on_bad_json(self):
        from src.evidence.extractor import EvidenceExtractor

        extractor = EvidenceExtractor(llm_client=JsonLLM('{"claims": [{"claim": "  "}]}'))
        assert extractor.extract_claims_with_llm("unused") == []

        fallback = EvidenceExtractor(llm_client=JsonLLM("not-json"))
        claims = fallback.extract_claims_with_llm("欧盟AI法案已于2024年3月通过。")

        assert [claim.claim for claim in claims] == ["欧盟AI法案已于2024年3月通过"]

    def test_extractor_helpers_cover_safe_defaults(self):
        from src.evidence.extractor import EvidenceExtractor

        extractor = EvidenceExtractor(llm_client=MockLLM())

        assert extractor._coerce_claim_type(ClaimType.CAUSAL) == ClaimType.CAUSAL
        assert extractor._coerce_claim_type(None) == ClaimType.FACTUAL
        assert extractor._coerce_string_list(None) == []
        assert extractor._coerce_string_list({"bad": "shape"}) == []
        assert extractor._coerce_bool("yes") is True
        assert extractor._coerce_bool("no") is False
        assert extractor._coerce_bool(object(), default=False) is False
        assert extractor._coerce_support_type(None) == "none"
        assert extractor._resolve_pronouns("该法案已通过", []) == "该法案已通过"
        assert extractor._select_resolution_anchor(["EU", "AI"]) == "EU"

    def test_comparative_numeric_extraction_rejects_ambiguous_shapes(self):
        from src.evidence.extractor import EvidenceExtractor

        extractor = EvidenceExtractor(llm_client=MockLLM())

        assert extractor._extract_comparative_numeric_claims("ECS为2.5°C。") == []
        assert extractor._extract_comparative_numeric_claims("ECS低于3.0°C。") == []
        assert extractor._extract_comparative_numeric_claims("2.5°C低于IPCC AR6的3.0°C。") == []
        assert extractor._extract_comparative_numeric_claims("ECS估计值2.5°C低于IPCC。") == []
        assert extractor._extract_comparative_numeric_claims("ECS 2.5°C低于IPCC AR6 3.0°C。") == []

    def test_run_delegates_to_extract_from_text(self):
        from src.evidence.extractor import EvidenceExtractor

        extractor = EvidenceExtractor(llm_client=MockLLM())
        evidence = extractor.run("欧盟AI法案已于2024年3月通过。", source="law", title="AI Act")

        assert evidence.source == "law"
        assert evidence.title == "AI Act"


# --- EvidenceScorer Tests ---

class TestEvidenceScorer:
    def _make_evidence(self, cred=0.8, rec=0.7, rel=0.6):
        return Evidence(
            evidence_id="E1", source="paper", title="T", text_span="内容",
            credibility_score=cred, recency_score=rec, relevance_score=rel,
        )

    def test_score_evidence(self):
        from src.evidence.evidence_scorer import EvidenceScorer
        scorer = EvidenceScorer()
        ev = self._make_evidence()
        score = scorer.score_evidence(ev)
        assert 0 <= score <= 1

    def test_high_credibility_scores_higher(self):
        from src.evidence.evidence_scorer import EvidenceScorer
        scorer = EvidenceScorer()
        ev_low = self._make_evidence(cred=0.3, rec=0.5, rel=0.5)
        ev_high = self._make_evidence(cred=0.9, rec=0.5, rel=0.5)
        assert scorer.score_evidence(ev_high) > scorer.score_evidence(ev_low)

    def test_score_evidence_list(self):
        from src.evidence.evidence_scorer import EvidenceScorer
        scorer = EvidenceScorer()
        evs = [self._make_evidence(cred=c) for c in [0.5, 0.8, 0.3]]
        scores = scorer.score_evidence_list(evs)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_rank_evidence(self):
        from src.evidence.evidence_scorer import EvidenceScorer
        scorer = EvidenceScorer()
        evs = [self._make_evidence(cred=c) for c in [0.3, 0.9, 0.5]]
        ranked = scorer.rank_evidence(evs)
        assert len(ranked) == 3
        # Should be sorted descending by score
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_filter_by_threshold(self):
        from src.evidence.evidence_scorer import EvidenceScorer
        scorer = EvidenceScorer()
        evs = [self._make_evidence(cred=c) for c in [0.9, 0.5, 0.1]]
        filtered = scorer.filter_by_threshold(evs, threshold=0.5)
        # At least the highest-credibility one should pass
        assert len(filtered) >= 1

    def test_custom_weights(self):
        from src.evidence.evidence_scorer import EvidenceScorer
        scorer = EvidenceScorer(config={"weights": {"credibility": 1.0, "recency": 0, "relevance": 0, "support": 0, "conflict": 0}})
        ev = self._make_evidence(cred=0.7, rec=0.1, rel=0.1)
        score = scorer.score_evidence(ev)
        assert score == pytest.approx(0.7, abs=0.05)

    def test_with_conflict_graph(self):
        from src.evidence.evidence_scorer import EvidenceScorer
        scorer = EvidenceScorer()
        ev = self._make_evidence()
        graph = EvidenceConflictGraph()
        graph.add_edge(ConflictEdge(source_id="E1", target_id="E2", conflict_type=ConflictType.NUMERIC_CONFLICT,
                                     severity="high", confidence=0.8))
        score = scorer.score_evidence(ev, conflict_graph=graph)
        assert 0 <= score <= 1

    def test_conflict_penalty_counts_target_evidence_edges(self):
        from src.evidence.evidence_scorer import EvidenceScorer

        scorer = EvidenceScorer(config={
            "weights": {
                "credibility": 0,
                "recency": 0,
                "relevance": 0,
                "support": 0,
                "conflict": 1,
            }
        })
        ev = self._make_evidence()
        graph = EvidenceConflictGraph()
        graph.add_edge(ConflictEdge(
            source_id="other",
            target_id="E1",
            conflict_type=ConflictType.REFUTE,
            confidence=0.8,
        ))

        assert scorer.score_evidence(ev, conflict_graph=graph) == 0.0
        assert scorer._calculate_conflict_penalty(ev, graph) == pytest.approx(0.8)

    def test_support_score_counts_target_claim_edges(self):
        from src.evidence.evidence_scorer import EvidenceScorer

        scorer = EvidenceScorer()
        ev = self._make_evidence()
        ev.claims = [
            Claim(
                claim_id="C1",
                claim="星辰科技营收增长",
                claim_type=ClaimType.FACTUAL,
            )
        ]
        graph = EvidenceConflictGraph()
        graph.add_edge(ConflictEdge(
            source_id="other",
            target_id="C1",
            conflict_type=ConflictType.SUPPORT,
            confidence=0.9,
        ))

        assert scorer._calculate_support_score(ev, graph) == 1.0

    def test_graph_scores_use_defaults_without_related_edges(self):
        from src.evidence.evidence_scorer import EvidenceScorer

        scorer = EvidenceScorer()
        ev = self._make_evidence()
        graph = EvidenceConflictGraph()
        graph.add_edge(ConflictEdge(
            source_id="other-a",
            target_id="other-b",
            conflict_type=ConflictType.REFUTE,
            confidence=0.9,
        ))

        assert scorer._calculate_support_score(ev, graph) == 0.5
        assert scorer._calculate_conflict_penalty(ev, graph) == 0.0
