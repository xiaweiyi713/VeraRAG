"""Hybrid Retriever combining sparse and dense retrieval for VeraRAG."""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseRetriever, RetrievalResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines sparse (BM25) and dense retrieval.

    Combines scores from both retrievers using configurable weights.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
        **kwargs
    ):
        super().__init__(config)
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

        # Initialize sub-retrievers
        sparse_config = config.get('sparse', {}) if config else {}
        dense_config = config.get('dense', {}) if config else {}

        self.sparse_retriever = BM25Retriever(
            config=sparse_config,
            **{k: v for k, v in kwargs.items() if k in ['k1', 'b', 'epsilon']}
        )

        # Dense retriever is optional — may not have sentence-transformers installed
        self._dense_available = True
        try:
            self.dense_retriever = DenseRetriever(
                config=dense_config,
                **{k: v for k, v in kwargs.items() if k in ['model_name', 'device', 'batch_size']}
            )
        except ImportError:
            self._dense_available = False
            self.dense_retriever = None
            import logging
            logging.getLogger("verarag").warning("DenseRetriever unavailable (sentence-transformers not installed), using BM25 only")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.sparse_retriever.index_documents(documents)
        if self._dense_available and self.dense_retriever:
            try:
                self.dense_retriever.index_documents(documents)
            except ImportError:
                self._dense_available = False

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        fetch_k: int = 100,
        **kwargs
    ) -> List[RetrievalResult]:
        sparse_results = self.sparse_retriever.retrieve(query, top_k=fetch_k)

        if not self._dense_available or not self.dense_retriever:
            return sparse_results[:top_k]

        try:
            dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k)
        except ImportError:
            self._dense_available = False
            return sparse_results[:top_k]

        # Combine scores using reciprocal rank fusion
        combined_scores = self._reciprocal_rank_fusion(
            [sparse_results, dense_results],
            weights=[self.sparse_weight, self.dense_weight]
        )

        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Get full result objects
        results_map = {r.doc_id: r for r in sparse_results + dense_results}
        results = [
            RetrievalResult(
                doc_id=doc_id,
                content=results_map[doc_id].content,
                title=results_map[doc_id].title,
                score=score,
                metadata=results_map[doc_id].metadata
            )
            for doc_id, score in sorted_results
        ]

        return results

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        weights: List[float],
        k: int = 60
    ) -> Dict[str, float]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        Args:
            result_lists: Lists of retrieval results from different retrievers
            weights: Weights for each retriever
            k: RRF constant

        Returns:
            Dictionary mapping doc_id to combined score
        """
        scores = {}

        for results, weight in zip(result_lists, weights):
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                # RRF score: weight / (k + rank)
                rrf_score = weight / (k + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        return scores

    def save_index(self, path: str) -> None:
        """Save both indexes to disk."""
        from pathlib import Path
        path = Path(path)

        self.sparse_retriever.save_index(str(path / "sparse_index.pkl"))
        self.dense_retriever.save_index(str(path / "dense_index.pkl"))

    def load_index(self, path: str) -> None:
        """Load both indexes from disk."""
        from pathlib import Path
        path = Path(path)

        self.sparse_retriever.load_index(str(path / "sparse_index.pkl"))
        self.dense_retriever.load_index(str(path / "dense_index.pkl"))
