"""Tests for evidence modules: extractor, evidence_scorer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_structures import (
    ConflictEdge,
    ConflictType,
    Evidence,
    EvidenceConflictGraph,
)


class MockLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        return '[{"claim": "测试声明", "claim_type": "factual", "confidence": 0.8}]'


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
