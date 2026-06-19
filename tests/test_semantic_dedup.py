"""Tests for semantic deduplication."""

import sys
from types import ModuleType

import numpy as np

sys.path.insert(0, 'src')

import unittest
from unittest.mock import patch

from src.evidence.normalizer import EvidenceNormalizer
from src.utils.data_structures import Evidence
from src.utils.model_cache import clear_optional_model_cache


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

    def test_semantic_dedup_can_be_disabled(self):
        """离线配置可跳过 sentence-transformers 语义去重"""
        normalizer = EvidenceNormalizer({"evidence": {"semantic_dedup": False}})
        ev1 = self._make_evidence("E1", "RAG improves factuality", score=0.9)
        ev2 = self._make_evidence("E2", "RAG improves factuality", score=0.5)

        result = normalizer.deduplicate([ev2, ev1])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].evidence_id, "E1")

    def test_semantic_dedup_removes_high_similarity_evidence(self):
        normalizer = EvidenceNormalizer()
        ev1 = self._make_evidence("E1", "strong evidence", score=0.9)
        ev2 = self._make_evidence("E2", "near duplicate", score=0.6)
        ev3 = self._make_evidence("E3", "different evidence", score=0.8)

        class FakeModel:
            def encode(self, texts, normalize_embeddings):
                self.texts = texts
                self.normalize_embeddings = normalize_embeddings
                return np.array([
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.99, 0.01],
                ])

        model = FakeModel()
        normalizer._get_semantic_model = lambda: model

        result = normalizer.deduplicate([ev2, ev1, ev3], similarity_threshold=0.95)

        self.assertEqual([ev.evidence_id for ev in result], ["E1", "E3"])
        self.assertEqual(model.texts, ["strong evidence", "different evidence", "near duplicate"])
        self.assertTrue(model.normalize_embeddings)

    def test_semantic_dedup_failure_falls_back_to_exact_dedup(self):
        normalizer = EvidenceNormalizer()
        ev1 = self._make_evidence("E1", "RAG improves factuality", score=0.9)
        ev2 = self._make_evidence("E2", "RAG   improves factuality", score=0.6)

        class FailingModel:
            def encode(self, texts, normalize_embeddings):
                raise RuntimeError("embedding failed")

        normalizer._get_semantic_model = lambda: FailingModel()

        result = normalizer.deduplicate([ev2, ev1])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].evidence_id, "E1")

    def test_semantic_model_is_cached_across_normalizers(self):
        clear_optional_model_cache()
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, model_name, **kwargs):
                calls.append((model_name, kwargs))

        fake_module = ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        config = {
            "evidence": {
                "semantic_dedup_model": "test/multilingual-model",
                "semantic_dedup_local_files_only": True,
            }
        }
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            first = EvidenceNormalizer(config)._get_semantic_model()
            second = EvidenceNormalizer(config)._get_semantic_model()

        self.assertIs(first, second)
        self.assertEqual(
            calls,
            [("test/multilingual-model", {"local_files_only": True})],
        )
        clear_optional_model_cache()

    def test_failed_semantic_model_load_is_not_retried(self):
        clear_optional_model_cache()
        calls = []

        class FailingSentenceTransformer:
            def __init__(self, model_name, **kwargs):
                calls.append((model_name, kwargs))
                raise OSError("model unavailable")

        fake_module = ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FailingSentenceTransformer
        config = {
            "evidence": {
                "semantic_dedup_model": "missing/model",
                "semantic_dedup_local_files_only": True,
            }
        }
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            self.assertIsNone(EvidenceNormalizer(config)._get_semantic_model())
            self.assertIsNone(EvidenceNormalizer(config)._get_semantic_model())

        self.assertEqual(len(calls), 1)
        clear_optional_model_cache()

    def test_hardcoded_year_fixed(self):
        """验证 current_year 不再硬编码为 2025"""
        import inspect

        from src.evidence.normalizer import EvidenceNormalizer
        source = inspect.getsource(EvidenceNormalizer._estimate_recency)
        self.assertNotIn("current_year = 2025", source)
        self.assertIn("datetime", source)


if __name__ == "__main__":
    unittest.main()
