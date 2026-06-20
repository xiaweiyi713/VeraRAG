"""Reranker for improving retrieval results in VeraRAG."""

from typing import Any

import numpy as np

from .base import BaseRetriever, RetrievalResult


class Reranker:
    """
    Reranker for improving retrieval results using cross-encoder models.

    Uses a cross-encoder model that takes query-document pairs and outputs
    a relevance score.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 16,
        top_k: int = 20,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.top_k = top_k
        self.local_files_only = local_files_only
        self.model: Any = None

    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return (defaults to self.top_k)

        Returns:
            Reranked results
        """
        if not results:
            return results

        top_k = self.top_k if top_k is None else top_k
        top_k = min(top_k, len(results))
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        # Load model
        self._load_model()

        # Prepare query-document pairs
        pairs = [
            [query, f"{r.title} {r.content}".strip()]
            for r in results
        ]

        # Get scores from cross-encoder
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        # Sort by new scores
        scored_results = list(zip(results, scores))  # noqa: B905
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k with updated scores
        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(RetrievalResult(
                doc_id=result.doc_id,
                content=result.content,
                title=result.title,
                score=float(score),
                metadata=result.metadata
            ))

        return reranked

    def rerank_batch(
        self,
        queries: list[str],
        results_list: list[list[RetrievalResult]],
        top_k: int | None = None
    ) -> list[list[RetrievalResult]]:
        """
        Rerank multiple query-result pairs.

        Args:
            queries: List of queries
            results_list: List of retrieval result lists
            top_k: Number of top results to return

        Returns:
            List of reranked result lists
        """
        return [
            self.rerank(q, r, top_k)
            for q, r in zip(queries, results_list)  # noqa: B905
        ]


class RerankingRetriever(BaseRetriever):
    """Retriever wrapper that reranks a larger candidate pool before truncation."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Reranker | None = None,
        *,
        candidate_k: int = 20,
        preserve_base_top_k: int = 0,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.base_retriever = base_retriever
        self.reranker = reranker or Reranker(top_k=candidate_k)
        self.candidate_k = self._validate_top_k(candidate_k)
        if preserve_base_top_k < 0:
            raise ValueError("preserve_base_top_k must be non-negative")
        self.preserve_base_top_k = preserve_base_top_k

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        self.base_retriever.index_documents(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> list[RetrievalResult]:
        query = self._validate_query(query)
        top_k = self._validate_top_k(top_k)
        candidate_k = max(top_k, self.candidate_k)
        candidates = self.base_retriever.retrieve(query, top_k=candidate_k, **kwargs)
        reranked = self.reranker.rerank(query, candidates, top_k=candidate_k)
        return self._apply_recall_guard(candidates, reranked, top_k)

    def _apply_recall_guard(
        self,
        candidates: list[RetrievalResult],
        reranked: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Preserve a small base-retriever anchor set after reranking."""
        if top_k == 0:
            return []
        if self.preserve_base_top_k == 0:
            return reranked[:top_k]

        required = self._dedupe_by_doc_id(candidates[: min(self.preserve_base_top_k, top_k)])
        required_ids = {result.doc_id for result in required}
        fill = [
            result for result in self._dedupe_by_doc_id(reranked)
            if result.doc_id not in required_ids
        ]
        return (required + fill)[:top_k]

    @staticmethod
    def _dedupe_by_doc_id(results: list[RetrievalResult]) -> list[RetrievalResult]:
        deduped: list[RetrievalResult] = []
        seen: set[str] = set()
        for result in results:
            if result.doc_id in seen:
                continue
            seen.add(result.doc_id)
            deduped.append(result)
        return deduped


class EvidenceAwareReranker(Reranker):
    """
    Reranker that considers evidence quality factors.

    Takes into account:
    - Relevance score from cross-encoder
    - Source credibility
    - Recency
    - Citation count
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        credibility_weight: float = 0.2,
        recency_weight: float = 0.1,
        relevance_weight: float = 0.7
    ):
        super().__init__(model_name, device)
        self.credibility_weight = credibility_weight
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None
    ) -> list[RetrievalResult]:
        """
        Rerank with evidence quality considerations.
        """
        if not results:
            return results

        top_k = self.top_k if top_k is None else top_k
        top_k = min(top_k, len(results))
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        # Get base relevance scores
        self._load_model()
        pairs = [[query, f"{r.title} {r.content}".strip()] for r in results]
        relevance_scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Calculate combined scores
        scored_results = []
        for result, relevance in zip(results, relevance_scores):  # noqa: B905
            # Normalize relevance to 0-1
            norm_relevance = float(1 / (1 + np.exp(-relevance)))

            # Get credibility score from metadata
            credibility = (result.metadata or {}).get('credibility_score', 0.5)

            # Get recency score (could be based on date)
            recency = (result.metadata or {}).get('recency_score', 0.5)

            # Combine scores
            combined = (
                norm_relevance * self.relevance_weight +
                credibility * self.credibility_weight +
                recency * self.recency_weight
            )

            scored_results.append((result, combined))

        # Sort and return
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(RetrievalResult(
                doc_id=result.doc_id,
                content=result.content,
                title=result.title,
                score=score,
                metadata=result.metadata
            ))

        return reranked
