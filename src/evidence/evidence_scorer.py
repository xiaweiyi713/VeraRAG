"""Evidence Scorer for VeraRAG."""

from typing import Any

from ..utils.data_structures import Evidence, EvidenceConflictGraph


class EvidenceScorer:
    """
    Scores evidence based on multiple quality dimensions.

    Dimensions:
    1. Credibility: Source reliability
    2. Recency: How recent the evidence is
    3. Relevance: How relevant to the query
    4. Support: How well supported by other evidence
    5. Conflict: How much conflict with other evidence
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.weights = self.config.get("weights", {
            "credibility": 0.3,
            "recency": 0.2,
            "relevance": 0.3,
            "support": 0.1,
            "conflict": 0.1
        })

    def score_evidence(
        self,
        evidence: Evidence,
        conflict_graph: EvidenceConflictGraph | None = None
    ) -> float:
        """
        Calculate a composite score for an evidence item.

        Args:
            evidence: The evidence to score
            conflict_graph: Optional conflict graph for context

        Returns:
            Composite score (0-1)
        """
        # Base scores from evidence object
        credibility = evidence.credibility_score
        recency = evidence.recency_score
        relevance = evidence.relevance_score

        # Calculate support score from conflict graph
        support = 0.5
        if conflict_graph:
            support = self._calculate_support_score(evidence, conflict_graph)

        # Calculate conflict penalty
        conflict = 0.0
        if conflict_graph:
            conflict = self._calculate_conflict_penalty(evidence, conflict_graph)

        # Weighted combination
        composite = (
            credibility * self.weights["credibility"] +
            recency * self.weights["recency"] +
            relevance * self.weights["relevance"] +
            support * self.weights["support"] -
            conflict * self.weights["conflict"]
        )

        return max(0.0, min(1.0, composite))

    def _calculate_support_score(
        self,
        evidence: Evidence,
        conflict_graph: EvidenceConflictGraph
    ) -> float:
        """Calculate how much this evidence is supported by others."""
        support_count = 0
        total_count = 0

        for edge in conflict_graph.edges:
            if edge.source_id in [c.claim_id for c in evidence.claims]:
                total_count += 1
                if edge.conflict_type.value == "support":
                    support_count += 1

        if total_count == 0:
            return 0.5

        return support_count / total_count

    def _calculate_conflict_penalty(
        self,
        evidence: Evidence,
        conflict_graph: EvidenceConflictGraph
    ) -> float:
        """Calculate penalty based on conflicts."""
        conflict_count = 0
        total_count = 0

        for edge in conflict_graph.get_conflicts():
            if edge.source_id in [c.claim_id for c in evidence.claims]:
                total_count += 1
                conflict_count += 1 * edge.confidence

        if total_count == 0:
            return 0.0

        return conflict_count / total_count

    def score_evidence_list(
        self,
        evidence_list: list[Evidence],
        conflict_graph: EvidenceConflictGraph | None = None
    ) -> list[float]:
        """
        Score a list of evidence items.

        Args:
            evidence_list: List of evidence to score
            conflict_graph: Optional conflict graph

        Returns:
            List of scores aligned with input list
        """
        return [
            self.score_evidence(ev, conflict_graph)
            for ev in evidence_list
        ]

    def rank_evidence(
        self,
        evidence_list: list[Evidence],
        conflict_graph: EvidenceConflictGraph | None = None
    ) -> list[tuple[Evidence, float]]:
        """
        Rank evidence by composite score.

        Args:
            evidence_list: List of evidence to rank
            conflict_graph: Optional conflict graph

        Returns:
            List of (evidence, score) tuples sorted by score
        """
        scored = [
            (ev, self.score_evidence(ev, conflict_graph))
            for ev in evidence_list
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def filter_by_threshold(
        self,
        evidence_list: list[Evidence],
        threshold: float,
        conflict_graph: EvidenceConflictGraph | None = None
    ) -> list[Evidence]:
        """
        Filter evidence by score threshold.

        Args:
            evidence_list: List of evidence to filter
            threshold: Minimum score threshold
            conflict_graph: Optional conflict graph

        Returns:
            Filtered list of evidence
        """
        return [
            ev for ev in evidence_list
            if self.score_evidence(ev, conflict_graph) >= threshold
        ]
