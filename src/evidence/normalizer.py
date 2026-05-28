"""Evidence Normalizer for VeraRAG."""

import re
import uuid
from datetime import datetime
from typing import Any

from ..retriever.base import RetrievalResult
from ..utils.data_structures import Evidence


class EvidenceNormalizer:
    """
    Normalizes retrieved documents into structured Evidence units.

    Handles:
    1. Converting RetrievalResult to Evidence
    2. Scoring evidence quality
    3. Extracting metadata
    4. Standardizing formats
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def normalize_retrieval_results(
        self,
        results: list[RetrievalResult],
        extract_claims: bool = True
    ) -> list[Evidence]:
        """
        Normalize retrieval results into Evidence objects.

        Args:
            results: List of retrieval results
            extract_claims: Whether to extract claims from evidence

        Returns:
            List of normalized Evidence objects
        """
        evidence_list = []

        for result in results:
            evidence = self._normalize_single_result(result, extract_claims)
            if evidence:
                evidence_list.append(evidence)

        return evidence_list

    def _normalize_single_result(
        self,
        result: RetrievalResult,
        extract_claims: bool
    ) -> Evidence | None:
        """Normalize a single retrieval result."""
        # Extract metadata
        meta = result.metadata or {}
        source = meta.get("source", "unknown")
        date = meta.get("date")
        author = meta.get("author")
        url = meta.get("url")

        # Create evidence object
        evidence = Evidence(
            evidence_id=f"E{uuid.uuid4().hex[:8]}",
            source=source,
            title=result.title,
            text_span=result.content,
            date=date,
            author=author,
            url=url,
            relevance_score=min(1.0, result.score),
            credibility_score=self._estimate_credibility(result),
            recency_score=self._estimate_recency(date)
        )

        return evidence

    def _estimate_credibility(self, result: RetrievalResult) -> float:
        """Estimate credibility score based on source."""
        source = (result.metadata or {}).get("source", "").lower()

        # Known high-credibility sources
        high_sources = {
            "nature", "science", "cell", "nejm", "lancet",
            "arxiv", "pnas", "acm", "ieee"
        }

        # Known medium-credibility sources
        medium_sources = {
            "wikipedia", "news", "blog"
        }

        if any(s in source for s in high_sources):
            return 0.9
        elif any(s in source for s in medium_sources):
            return 0.7
        else:
            return 0.5

    def _estimate_recency(self, date: str | None) -> float:
        """Estimate recency score based on date."""
        if not date:
            return 0.5

        try:
            # Parse date
            if isinstance(date, str):
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
                    try:
                        dt = datetime.strptime(date, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return 0.5
            else:
                dt = date

            # Calculate years since publication
            current_year = datetime.now().year
            pub_year = dt.year
            years_ago = current_year - pub_year

            # Decay score: 1.0 for current year, 0.5 for 5+ years
            return max(0.5, 1.0 - years_ago * 0.1)

        except Exception:
            return 0.5

    def filter_low_quality(
        self,
        evidence_list: list[Evidence],
        min_score: float = 0.3
    ) -> list[Evidence]:
        """
        Filter out low-quality evidence.

        Args:
            evidence_list: List of evidence objects
            min_score: Minimum combined score threshold

        Returns:
            Filtered list of evidence
        """
        return [
            ev for ev in evidence_list
            if ev.combined_score >= min_score
        ]

    def deduplicate(
        self,
        evidence_list: list[Evidence],
        similarity_threshold: float = 0.92
    ) -> list[Evidence]:
        """
        Remove duplicate evidence based on semantic similarity.

        Uses sentence-transformers for semantic similarity when available,
        falls back to exact text matching otherwise.

        Args:
            evidence_list: List of evidence objects
            similarity_threshold: Threshold for semantic dedup (0-1)

        Returns:
            Deduplicated list
        """
        if len(evidence_list) <= 1:
            return evidence_list

        # Try semantic dedup with sentence-transformers
        try:
            return self._semantic_dedup(evidence_list, similarity_threshold)
        except (ImportError, Exception):
            # Fallback: exact text match
            return self._exact_dedup(evidence_list)

    def _semantic_dedup(
        self,
        evidence_list: list[Evidence],
        threshold: float
    ) -> list[Evidence]:
        """Semantic deduplication using sentence-transformers."""
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Sort by combined score descending (keep higher-scored ones)
        sorted_ev = sorted(evidence_list, key=lambda ev: ev.combined_score, reverse=True)

        texts = [ev.text_span for ev in sorted_ev]
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        embeddings = model.encode(texts, normalize_embeddings=True)

        # Greedy dedup: keep item if not similar to any already-kept item
        kept = []
        kept_indices: list[int] = []

        for i, ev in enumerate(sorted_ev):
            is_duplicate = False
            for j in kept_indices:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(ev)
                kept_indices.append(i)

        return kept

    def _exact_dedup(self, evidence_list: list[Evidence]) -> list[Evidence]:
        """Exact text matching deduplication (fallback)."""
        # Sort by combined score descending so higher-scored evidence is kept
        sorted_ev = sorted(evidence_list, key=lambda ev: ev.combined_score, reverse=True)

        seen = set()
        deduplicated = []

        for ev in sorted_ev:
            text_key = re.sub(r'\s+', ' ', ev.text_span.lower().strip())
            if text_key not in seen:
                seen.add(text_key)
                deduplicated.append(ev)

        return deduplicated

    def rank_by_quality(
        self,
        evidence_list: list[Evidence]
    ) -> list[Evidence]:
        """
        Rank evidence by combined quality score.

        Args:
            evidence_list: List of evidence objects

        Returns:
            Sorted list of evidence
        """
        return sorted(
            evidence_list,
            key=lambda ev: ev.combined_score,
            reverse=True
        )
