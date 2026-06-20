"""Tests for retriever modules: BM25, reranker."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever.base import BaseRetriever, RetrievalResult
from src.retriever.bm25 import BM25Retriever
from src.retriever.dense import DenseRetriever, FAISSRetriever
from src.retriever.hybrid import HybridRetriever


class TestBaseRetriever:
    def test_batch_retrieve_preserves_order_and_kwargs(self):
        retriever = _RecordingRetriever()

        batches = retriever.batch_retrieve(["alpha", "beta"], top_k=1, marker="kept")

        assert [[result.doc_id for result in results] for results in batches] == [
            ["alpha-0"],
            ["beta-0"],
        ]
        assert retriever.calls == [
            ("alpha", 1, {"marker": "kept"}),
            ("beta", 1, {"marker": "kept"}),
        ]

    @pytest.mark.parametrize("top_k", [-1, 1.5, True])
    def test_batch_retrieve_rejects_invalid_top_k(self, top_k):
        retriever = _RecordingRetriever()

        with pytest.raises((TypeError, ValueError), match="top_k"):
            retriever.batch_retrieve(["alpha"], top_k=top_k)

    def test_batch_retrieve_rejects_invalid_queries(self):
        retriever = _RecordingRetriever()

        with pytest.raises(TypeError, match="queries"):
            retriever.batch_retrieve(("alpha", "beta"))
        with pytest.raises(TypeError, match="query"):
            retriever.batch_retrieve(["alpha", None])

    def test_base_optional_index_methods_raise_clear_errors(self):
        retriever = _RecordingRetriever()

        with pytest.raises(NotImplementedError, match="save_index"):
            retriever.save_index("unused.pkl")
        with pytest.raises(NotImplementedError, match="load_index"):
            retriever.load_index("unused.pkl")

    def test_abstract_default_methods_raise_when_delegated(self):
        retriever = _DelegatingRetriever()

        with pytest.raises(NotImplementedError):
            retriever.retrieve("query")
        with pytest.raises(NotImplementedError):
            retriever.index_documents([])


class TestBM25Retriever:
    @pytest.fixture
    def indexed_retriever(self):
        retriever = BM25Retriever()
        docs = [
            {"id": "D1", "text": "量子计算是利用量子力学原理进行计算的技术", "title": "量子计算入门"},
            {"id": "D2", "text": "人工智能在医疗领域有广泛应用前景", "title": "AI医疗应用"},
            {"id": "D3", "text": "新能源汽车销量持续增长，比亚迪领跑", "title": "新能源汽车市场"},
            {"id": "D4", "text": "欧盟AI法案对人工智能发展产生重要影响", "title": "欧盟AI法案"},
            {"id": "D5", "text": "固态电池技术取得突破，能量密度大幅提升", "title": "固态电池技术"},
        ]
        retriever.index_documents(docs)
        return retriever

    def test_retrieve_returns_results(self, indexed_retriever):
        results = indexed_retriever.retrieve("量子计算", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_scores_positive(self, indexed_retriever):
        results = indexed_retriever.retrieve("人工智能", top_k=3)
        assert all(r.score > 0 for r in results)

    def test_retrieve_top_k_limits(self, indexed_retriever):
        results = indexed_retriever.retrieve("技术", top_k=2)
        assert len(results) <= 2

    def test_retrieve_no_match(self, indexed_retriever):
        results = indexed_retriever.retrieve("xyznomatch123", top_k=3)
        assert isinstance(results, list)

    def test_chinese_tokenization(self, indexed_retriever):
        results = indexed_retriever.retrieve("欧盟AI法案", top_k=3)
        assert len(results) > 0
        assert any("AI" in r.title or "法案" in r.title for r in results)

    def test_empty_query(self, indexed_retriever):
        results = indexed_retriever.retrieve("", top_k=3)
        assert isinstance(results, list)
        assert results == []

    def test_retrieve_rejects_invalid_limits_and_query_type(self, indexed_retriever):
        with pytest.raises(ValueError, match="top_k"):
            indexed_retriever.retrieve("量子计算", top_k=-1)
        with pytest.raises(TypeError, match="top_k"):
            indexed_retriever.retrieve("量子计算", top_k=True)
        with pytest.raises(TypeError, match="query"):
            indexed_retriever.retrieve(None, top_k=1)

    def test_no_index_returns_empty(self):
        retriever = BM25Retriever()
        results = retriever.retrieve("test", top_k=3)
        assert results == []

    def test_save_and_load_index(self, indexed_retriever, tmp_path):
        path = str(tmp_path / "bm25_test.pkl")
        indexed_retriever.save_index(path)

        retriever2 = BM25Retriever()
        retriever2.load_index(path)
        results = retriever2.retrieve("量子计算", top_k=3)
        assert len(results) > 0

    def test_metadata_preserved(self, indexed_retriever):
        results = indexed_retriever.retrieve("新能源", top_k=3)
        assert all(r.metadata.get("title") for r in results)

    def test_compact_country_group_expansion_recalls_authoritative_sources(self):
        from src.agents.retrieval_agent import DynamicRetrievalAgent
        from src.benchmark.loader import load_verabench
        from src.utils.data_structures import SubQuestion

        benchmark = load_verabench()
        retriever = BM25Retriever()
        retriever.index_documents([
            {
                "id": document.doc_id,
                "title": document.title,
                "text": document.content,
                "source": document.source,
                "date": document.date,
                "entities": document.entities,
            }
            for document in benchmark.corpus.values()
        ])
        agent = DynamicRetrievalAgent(retriever)
        question = "中美欧在AI监管方面分别采取了什么模式？各有什么特点？"

        results = agent.retrieve_for_subquestion(
            SubQuestion(id="sq_original", question=question),
            top_k=10,
        )

        result_ids = {result.doc_id for result in results}
        assert {"D001", "D004", "D005"} <= result_ids


class TestHybridRetriever:
    def test_reciprocal_rank_fusion_combines_weighted_ranks(self):
        retriever = HybridRetriever(sparse_weight=0.25, dense_weight=0.75)
        sparse_results = [
            RetrievalResult(doc_id="A", content="a", score=3.0),
            RetrievalResult(doc_id="B", content="b", score=2.0),
        ]
        dense_results = [
            RetrievalResult(doc_id="B", content="b", score=0.9),
            RetrievalResult(doc_id="C", content="c", score=0.8),
        ]

        scores = retriever._reciprocal_rank_fusion(
            [sparse_results, dense_results],
            weights=[retriever.sparse_weight, retriever.dense_weight],
            k=0,
        )

        assert scores["B"] == pytest.approx(0.25 / 2 + 0.75 / 1)
        assert scores["B"] > scores["C"] > scores["A"]

    def test_falls_back_to_bm25_when_dense_indexing_fails(self):
        retriever = HybridRetriever()
        docs = [
            {"id": "D1", "text": "verarag_unique_term 法案 已经 通过", "title": "欧盟AI法案"},
            {"id": "D2", "text": "量子计算使用量子比特", "title": "量子计算"},
            {"id": "D3", "text": "新能源汽车销量增长", "title": "新能源汽车"},
        ]

        assert retriever.dense_retriever is not None
        retriever.dense_retriever.index_documents = lambda _docs: (_ for _ in ()).throw(
            OSError("offline model unavailable")
        )
        retriever.index_documents(docs)
        results = retriever.retrieve("verarag_unique_term", top_k=1)

        assert retriever._dense_available is False
        assert len(results) == 1
        assert results[0].doc_id == "D1"

    def test_falls_back_to_bm25_when_dense_retrieval_fails(self):
        retriever = HybridRetriever()
        docs = [
            {"id": "D1", "text": "verarag_fallback_token appears here", "title": "fallback"},
            {"id": "D2", "text": "other content", "title": "other"},
            {"id": "D3", "text": "additional unrelated content", "title": "extra"},
        ]
        assert retriever.dense_retriever is not None
        retriever.dense_retriever.index_documents = lambda _docs: None
        retriever.dense_retriever.retrieve = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("dense service unavailable")
        )

        retriever.index_documents(docs)
        results = retriever.retrieve("verarag_fallback_token", top_k=1)

        assert retriever._dense_available is False
        assert [result.doc_id for result in results] == ["D1"]

    def test_hybrid_save_and_load_round_trips_sparse_and_dense_indexes(self, tmp_path):
        retriever = HybridRetriever()
        docs = [
            {"id": "D1", "text": "alpha beta", "title": "Alpha"},
            {"id": "D2", "text": "gamma delta", "title": "Gamma"},
        ]
        assert retriever.dense_retriever is not None
        retriever.dense_retriever._encode_texts = lambda texts: np.array(
            [[float(len(text)), float(text.count("a"))] for text in texts],
            dtype=float,
        )
        retriever.index_documents(docs)

        index_dir = tmp_path / "hybrid"
        retriever.save_index(str(index_dir))

        restored = HybridRetriever()
        assert restored.dense_retriever is not None
        restored.load_index(str(index_dir))
        restored.dense_retriever._encode_texts = lambda texts: np.array(
            [[float(len(text)), float(text.count("a"))] for text in texts],
            dtype=float,
        )

        results = restored.retrieve("alpha", top_k=1)

        assert (index_dir / "sparse_index.pkl").exists()
        assert (index_dir / "dense_index.pkl").exists()
        assert results


class TestDenseRetriever:
    def test_dense_retriever_reads_config_and_validates_batch_size(self):
        retriever = DenseRetriever(
            config={"model_name": "configured-model", "device": "cpu", "batch_size": 4},
            model_name="default-model",
            batch_size=8,
        )

        assert retriever.model_name == "configured-model"
        assert retriever.device == "cpu"
        assert retriever.batch_size == 4
        assert DenseRetriever(config={"local_files_only": True}).local_files_only is True

        with pytest.raises(ValueError, match="batch_size"):
            DenseRetriever(config={"batch_size": 0})

    def test_dense_retriever_ranks_by_cosine_similarity(self):
        retriever = DenseRetriever()
        documents = [
            {"id": "D1", "text": "alpha document", "title": "Alpha", "source": "test"},
            {"id": "D2", "text": "beta document", "title": "Beta", "source": "test"},
        ]
        vectors = {
            "Alpha alpha document": [1.0, 0.0],
            "Beta beta document": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
        }
        retriever._encode_texts = lambda texts: np.array([vectors[text] for text in texts], dtype=float)

        retriever.index_documents(documents)
        results = retriever.retrieve("alpha query", top_k=2)

        assert [result.doc_id for result in results] == ["D1", "D2"]
        assert results[0].score == pytest.approx(1.0)
        assert results[0].metadata["source"] == "test"
        assert results[0].metadata["title"] == "Alpha"

    def test_dense_retriever_handles_zero_norm_vectors(self):
        retriever = DenseRetriever()
        retriever._encode_texts = lambda texts: np.zeros((len(texts), 2), dtype=float)

        retriever.index_documents([{"id": "D1", "text": "zero", "title": "Zero"}])
        results = retriever.retrieve("zero query", top_k=1)

        assert len(results) == 1
        assert results[0].score == 0.0

    def test_dense_retriever_save_and_load_index(self, tmp_path):
        retriever = DenseRetriever(model_name="fake-model")
        retriever.embeddings = np.array([[1.0, 0.0]], dtype=float)
        retriever.doc_ids = ["D1"]
        retriever.doc_texts = ["Alpha alpha"]
        retriever.doc_metadata = [{"title": "Alpha", "source": "fixture"}]

        path = tmp_path / "dense.pkl"
        retriever.save_index(str(path))

        restored = DenseRetriever(model_name="other-model")
        restored.load_index(str(path))
        restored._encode_texts = lambda texts: np.array([[1.0, 0.0]], dtype=float)
        results = restored.retrieve("alpha", top_k=1)

        assert restored.model_name == "fake-model"
        assert restored.local_files_only is False
        assert results[0].doc_id == "D1"
        assert results[0].metadata["source"] == "fixture"

    def test_dense_retriever_empty_or_unindexed_operations_are_explicit(self, tmp_path):
        retriever = DenseRetriever()

        assert retriever.retrieve("query") == []

        with pytest.raises(ValueError, match="before indexing"):
            retriever.save_index(str(tmp_path / "dense.pkl"))

        retriever.index_documents([])

        assert retriever.retrieve("query") == []

    def test_dense_retriever_rejects_empty_documents_and_bad_embedding_shapes(self):
        retriever = DenseRetriever()

        with pytest.raises(ValueError, match="empty text"):
            retriever.index_documents([{"id": "D1", "text": "   "}])

        retriever._encode_texts = lambda texts: np.array([1.0, 2.0, 3.0], dtype=float)
        with pytest.raises(ValueError, match="row count"):
            retriever.index_documents([
                {"id": "D1", "text": "one"},
                {"id": "D2", "text": "two"},
            ])

    def test_dense_retriever_rejects_query_dimension_mismatch(self):
        retriever = DenseRetriever()
        retriever._encode_texts = lambda texts: np.array(
            [[1.0, 0.0] if text == "doc" else [1.0, 0.0, 0.0] for text in texts],
            dtype=float,
        )

        retriever.index_documents([{"id": "D1", "text": "doc"}])

        with pytest.raises(ValueError, match="dimension"):
            retriever.retrieve("query", top_k=1)

    def test_faiss_retriever_load_rebuilds_index_without_mutating_embeddings(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setitem(sys.modules, "faiss", _FakeFaissModule())
        retriever = FAISSRetriever(model_name="fake-faiss")
        retriever._encode_texts = lambda texts: np.array(
            [[1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )
        retriever.index_documents([
            {"id": "D1", "text": "alpha document", "title": "Alpha"},
            {"id": "D2", "text": "beta document", "title": "Beta"},
        ])
        assert retriever.embeddings is not None
        original_embeddings = retriever.embeddings.copy()

        path = tmp_path / "faiss.pkl"
        retriever.save_index(str(path))

        restored = FAISSRetriever()
        restored._encode_texts = retriever._encode_texts
        restored.load_index(str(path))
        results = restored.retrieve("alpha query", top_k=1)

        assert np.array_equal(retriever.embeddings, original_embeddings)
        assert restored.faiss_index is not None
        assert [result.doc_id for result in results] == ["D1"]

    def test_faiss_retriever_rejects_query_dimension_mismatch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "faiss", _FakeFaissModule())
        retriever = FAISSRetriever()
        retriever._encode_texts = lambda texts: np.array(
            [[1.0, 0.0] if text == "doc" else [1.0, 0.0, 0.0] for text in texts],
            dtype=np.float32,
        )
        retriever.index_documents([{"id": "D1", "text": "doc"}])

        with pytest.raises(ValueError, match="dimension"):
            retriever.retrieve("query", top_k=1)


class TestPipelineRetrieverConfig:
    def test_verarag_respects_bm25_retriever_type(self):
        from src.pipeline.verarag import VeraRAG

        pipeline = VeraRAG({
            "llm": {"provider": "openai", "api_key": "test-key"},
            "retriever": {"type": "bm25"},
        })

        assert isinstance(pipeline.retriever, BM25Retriever)

    def test_verarag_respects_bm25_rerank_retriever_type(self):
        from src.pipeline.verarag import VeraRAG
        from src.retriever.reranker import RerankingRetriever

        pipeline = VeraRAG({
            "llm": {"provider": "openai", "api_key": "test-key"},
            "retriever": {
                "type": "bm25_rerank",
                "reranker_model_name": "fake-reranker",
                "reranker_candidate_k": 5,
                "reranker_batch_size": 4,
                "reranker_local_files_only": True,
            },
        })

        assert isinstance(pipeline.retriever, RerankingRetriever)
        assert isinstance(pipeline.retriever.base_retriever, BM25Retriever)
        assert pipeline.retriever.candidate_k == 5
        assert pipeline.retriever.reranker.model_name == "fake-reranker"
        assert pipeline.retriever.reranker.batch_size == 4
        assert pipeline.retriever.reranker.local_files_only is True


class TestReranker:
    def test_reranker_orders_by_cross_encoder_score(self):
        from src.retriever.reranker import Reranker

        reranker = Reranker(batch_size=4)
        reranker._load_model = lambda: None
        reranker.model = _FakeCrossEncoder([0.1, 2.0, 0.5])
        results = [
            RetrievalResult(doc_id="D1", content="low", title="Low", score=0.0),
            RetrievalResult(doc_id="D2", content="high", title="High", score=0.0),
            RetrievalResult(doc_id="D3", content="mid", title="Mid", score=0.0),
        ]

        reranked = reranker.rerank("query", results, top_k=2)

        assert [result.doc_id for result in reranked] == ["D2", "D3"]
        assert [result.score for result in reranked] == [2.0, 0.5]

    def test_reranker_batch_uses_query_result_pairs(self):
        from src.retriever.reranker import Reranker

        reranker = Reranker()
        reranker._load_model = lambda: None
        reranker.model = _FakeCrossEncoder([1.0])
        batch = reranker.rerank_batch(
            ["q1", "q2"],
            [
                [RetrievalResult(doc_id="D1", content="one")],
                [RetrievalResult(doc_id="D2", content="two")],
            ],
            top_k=1,
        )

        assert [[result.doc_id for result in results] for results in batch] == [["D1"], ["D2"]]

    def test_reranker_respects_explicit_zero_top_k(self):
        from src.retriever.reranker import Reranker

        reranker = Reranker(top_k=5)
        reranker._load_model = lambda: None
        reranker.model = _FakeCrossEncoder([1.0, 0.5])
        results = [
            RetrievalResult(doc_id="D1", content="one"),
            RetrievalResult(doc_id="D2", content="two"),
        ]

        assert reranker.rerank("query", results, top_k=0) == []

    def test_evidence_aware_reranker_score_adjustment(self):
        """EvidenceAwareReranker should adjust scores based on metadata."""
        from src.retriever.reranker import EvidenceAwareReranker

        reranker = EvidenceAwareReranker(
            credibility_weight=0.3,
            recency_weight=0.2,
            relevance_weight=0.5,
        )
        reranker._load_model = lambda: None
        reranker.model = _FakeCrossEncoder([0.0, 0.0])

        results = [
            RetrievalResult(
                doc_id="D1",
                content="test1",
                title="T1",
                score=0.8,
                metadata={"credibility_score": 0.9, "recency_score": 0.8},
            ),
            RetrievalResult(
                doc_id="D2",
                content="test2",
                title="T2",
                score=0.9,
                metadata={"credibility_score": 0.5, "recency_score": 0.3},
            ),
        ]

        reranked = reranker.rerank("query", results, top_k=2)

        assert [result.doc_id for result in reranked] == ["D1", "D2"]
        assert reranked[0].score > reranked[1].score

    def test_evidence_aware_reranker_combines_relevance_and_metadata(self):
        from src.retriever.reranker import EvidenceAwareReranker

        reranker = EvidenceAwareReranker(
            credibility_weight=0.4,
            recency_weight=0.2,
            relevance_weight=0.4,
        )
        reranker._load_model = lambda: None
        reranker.model = _FakeCrossEncoder([0.0, 0.0])
        results = [
            RetrievalResult(
                doc_id="low-quality",
                content="same relevance",
                metadata={"credibility_score": 0.1, "recency_score": 0.1},
            ),
            RetrievalResult(
                doc_id="high-quality",
                content="same relevance",
                metadata={"credibility_score": 0.9, "recency_score": 0.9},
            ),
        ]

        reranked = reranker.rerank("query", results, top_k=2)

        assert [result.doc_id for result in reranked] == ["high-quality", "low-quality"]
        assert reranked[0].score > reranked[1].score

    def test_reranker_empty_results(self):
        from src.retriever.reranker import Reranker
        reranker = Reranker()
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_reranking_retriever_uses_candidate_pool_before_truncation(self):
        from src.retriever.reranker import RerankingRetriever

        base = _RecordingRetriever()
        reranker = _FakeReranker()
        retriever = RerankingRetriever(base, reranker=reranker, candidate_k=4)

        results = retriever.retrieve("query", top_k=2)

        assert base.calls == [("query", 4, {})]
        assert reranker.calls[0][0] == "query"
        assert [result.doc_id for result in reranker.calls[0][1]] == [
            "query-0",
            "query-1",
            "query-2",
            "query-3",
        ]
        assert [result.doc_id for result in results] == ["query-3", "query-2"]


class _FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return np.array(self.scores[: len(pairs)], dtype=float)


class _FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, results, top_k=None):
        self.calls.append((query, results, top_k))
        return list(reversed(results))[:top_k]


class _RecordingRetriever(BaseRetriever):
    def __init__(self):
        super().__init__()
        self.calls = []

    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        self.calls.append((query, top_k, kwargs))
        return [
            RetrievalResult(doc_id=f"{query}-{idx}", content=query, score=1.0 / (idx + 1))
            for idx in range(top_k)
        ]

    def index_documents(self, documents):
        self.documents = documents


class _DelegatingRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        return super().retrieve(query, top_k, **kwargs)

    def index_documents(self, documents):
        return super().index_documents(documents)


class _FakeFaissModule(types.SimpleNamespace):
    def __init__(self):
        super().__init__(IndexFlatIP=_FakeFaissIndex)

    @staticmethod
    def normalize_L2(array):
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array /= norms


class _FakeFaissIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.embeddings = np.empty((0, dimension), dtype=np.float32)

    def add(self, embeddings):
        assert embeddings.shape[1] == self.dimension
        self.embeddings = embeddings.astype(np.float32, copy=True)

    def search(self, query, top_k):
        scores = query @ self.embeddings.T
        order = np.argsort(scores[0])[::-1][:top_k]
        result_scores = scores[0][order].astype(np.float32)
        result_indices = order.astype(np.int64)
        if len(order) < top_k:
            pad_count = top_k - len(order)
            result_scores = np.concatenate([result_scores, np.full(pad_count, -np.inf)])
            result_indices = np.concatenate([result_indices, np.full(pad_count, -1)])
        return result_scores.reshape(1, -1), result_indices.reshape(1, -1)
