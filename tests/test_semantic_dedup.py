"""Tests for semantic deduplication."""

import sys
sys.path.insert(0, 'src')

import unittest
from src.evidence.normalizer import EvidenceNormalizer
from src.utils.data_structures import Evidence


class TestSemanticDedup(unittest.TestCase):
    """Test semantic deduplication."""

    def _make_evidence(self, eid, text, score=0.8):
        return Evidence(
            evidence_id=eid,
            source="test",
            title=f"Test {eid}",
            text_span=text,
            credibility_score=score,
            recency_score=0.8,
            relevance_score=0.8
        )

    def test_exact_dedup_still_works(self):
        """精确文本匹配去重仍应正常工作"""
        normalizer = EvidenceNormalizer()
        ev1 = self._make_evidence("E1", "RAG improves factuality in LLMs")
        ev2 = self._make_evidence("E2", "RAG improves factuality in LLMs")

        result = normalizer.deduplicate([ev1, ev2])
        self.assertEqual(len(result), 1)

    def test_different_text_not_deduped(self):
        """完全不同的文本不应被去重"""
        normalizer = EvidenceNormalizer()
        ev1 = self._make_evidence("E1", "RAG improves factuality in language models")
        ev2 = self._make_evidence("E2", "Climate change affects global temperature")

        result = normalizer.deduplicate([ev1, ev2])
        self.assertEqual(len(result), 2)

    def test_keeps_higher_score_on_dedup(self):
        """去重时保留综合分更高的证据"""
        normalizer = EvidenceNormalizer()
        ev1 = self._make_evidence("E1", "RAG improves factuality", score=0.9)
        ev2 = self._make_evidence("E2", "RAG improves factuality", score=0.5)

        result = normalizer.deduplicate([ev2, ev1])  # 低分在前
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].evidence_id, "E1")

    def test_hardcoded_year_fixed(self):
        """验证 current_year 不再硬编码为 2025"""
        import inspect
        from src.evidence.normalizer import EvidenceNormalizer
        source = inspect.getsource(EvidenceNormalizer._estimate_recency)
        self.assertNotIn("current_year = 2025", source)
        self.assertIn("datetime", source)


if __name__ == "__main__":
    unittest.main()
