"""Conflict Graph Builder for VeraRAG."""

import json
from typing import Any

from ..agents.base import BaseAgent
from ..utils.data_structures import (
    Claim,
    ConflictEdge,
    ConflictGraphNode,
    ConflictType,
    Evidence,
    EvidenceConflictGraph,
)


class ConflictGraphBuilder(BaseAgent):
    """
    Builds and updates evidence conflict graphs.

    Detects and models:
    1. Support relationships between claims
    2. Refutation relationships
    3. Numerical conflicts
    4. Temporal conflicts
    5. Entity mismatches
    6. Source disagreements
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any | None = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert at detecting conflicts and relationships between claims.
Identify whether claims support, refute, or partially support each other.
Output ONLY valid JSON, no other text."""

    def build_graph(
        self,
        evidence_list: list[Evidence],
        use_llm: bool = True
    ) -> EvidenceConflictGraph:
        """
        Build a conflict graph from a list of evidence.

        Args:
            evidence_list: List of evidence objects
            use_llm: Whether to use LLM for conflict detection

        Returns:
            EvidenceConflictGraph with nodes and edges
        """
        graph = EvidenceConflictGraph()

        # Add nodes for each claim
        all_claims = []
        for ev in evidence_list:
            for claim in ev.claims:
                node = ConflictGraphNode(
                    node_id=claim.claim_id,
                    content=claim.claim,
                    node_type="claim",
                    evidence_ids=[ev.evidence_id]
                )
                graph.add_node(node)
                all_claims.append((claim, ev))

        # Add edges between claims
        for i, (claim_i, ev_i) in enumerate(all_claims):
            for j, (claim_j, ev_j) in enumerate(all_claims):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue

                edge = self._detect_relationship(
                    claim_i, ev_i,
                    claim_j, ev_j,
                    use_llm
                )

                if edge:
                    graph.add_edge(edge)

        return graph

    def _detect_relationship(
        self,
        claim_i: Claim,
        ev_i: Evidence,
        claim_j: Claim,
        ev_j: Evidence,
        use_llm: bool
    ) -> ConflictEdge | None:
        """Detect the relationship between two claims."""
        # First, check for specific conflict types with rules
        edge = self._rule_based_conflict_detection(claim_i, ev_i, claim_j, ev_j)
        if edge:
            return edge

        # Use LLM for more nuanced detection
        if use_llm and self.llm_client:
            return self._llm_conflict_detection(claim_i, claim_j)

        # Default: assume unrelated
        return None

    def _rule_based_conflict_detection(
        self,
        claim_i: Claim,
        ev_i: Evidence,
        claim_j: Claim,
        ev_j: Evidence
    ) -> ConflictEdge | None:
        """Detect conflicts using rule-based methods."""

        # Check for numerical conflicts
        if claim_i.numbers and claim_j.numbers:
            conflict = self._check_numerical_conflict(claim_i, claim_j)
            if conflict:
                return conflict

        # Check for temporal conflicts
        if claim_i.time_expressions and claim_j.time_expressions:
            conflict = self._check_temporal_conflict(claim_i, claim_j)
            if conflict:
                return conflict

        # Check for entity conflicts
        if claim_i.entities and claim_j.entities:
            conflict = self._check_entity_conflict(claim_i, claim_j)
            if conflict:
                return conflict

        return None

    def _check_numerical_conflict(
        self,
        claim_i: Claim,
        claim_j: Claim
    ) -> ConflictEdge | None:
        """Check for numerical conflicts between claims."""
        # Extract numbers
        nums_i = self._parse_numbers(claim_i.numbers)
        nums_j = self._parse_numbers(claim_j.numbers)

        if not nums_i or not nums_j:
            return None

        # Check for significant differences
        # This is a simplified check - real implementation would need more nuance
        for n_i in nums_i:
            for n_j in nums_j:
                # If numbers differ by more than 10%, flag as conflict
                if n_i > 0 and n_j > 0:
                    ratio = max(n_i, n_j) / min(n_i, n_j)
                    if ratio > 1.1:
                        return ConflictEdge(
                            source_id=claim_i.claim_id,
                            target_id=claim_j.claim_id,
                            conflict_type=ConflictType.NUMERIC_CONFLICT,
                            confidence=0.7,
                            rationale=f"Numerical values {n_i} and {n_j} differ significantly"
                        )

        return None

    def _parse_numbers(self, num_strings: list[str]) -> list[float]:
        """Parse numeric strings to floats."""
        numbers = []
        for ns in num_strings:
            try:
                # Remove % and commas
                clean = ns.replace('%', '').replace(',', '')
                numbers.append(float(clean))
            except ValueError:
                pass
        return numbers

    def _check_temporal_conflict(
        self,
        claim_i: Claim,
        claim_j: Claim
    ) -> ConflictEdge | None:
        """Check for temporal conflicts between claims."""
        # Check if claims use different time frames for the same entity
        # This is simplified - real implementation would be more sophisticated

        entities_i = set(claim_i.entities)
        entities_j = set(claim_j.entities)

        # If claims share entities but have conflicting temporal markers
        if entities_i & entities_j:  # Overlapping entities
            time_i = claim_i.time_expressions
            time_j = claim_j.time_expressions

            # Check for contradictory temporal markers
            contradictory_pairs = [
                ("before", "after"),
                ("earlier", "later"),
                ("past", "future")
            ]

            for t_i in time_i:
                for t_j in time_j:
                    for pair in contradictory_pairs:
                        if pair[0] in t_i.lower() and pair[1] in t_j.lower():
                            return ConflictEdge(
                                source_id=claim_i.claim_id,
                                target_id=claim_j.claim_id,
                                conflict_type=ConflictType.TEMPORAL_CONFLICT,
                                confidence=0.6,
                                rationale=f"Contradictory temporal markers: {t_i} vs {t_j}"
                            )

        return None

    def _check_entity_conflict(
        self,
        claim_i: Claim,
        claim_j: Claim
    ) -> ConflictEdge | None:
        """Check for entity conflicts between claims."""
        entities_i = set(claim_i.entities)
        entities_j = set(claim_j.entities)

        shared = entities_i & entities_j
        if not shared:
            return None

        diff_i = entities_i - entities_j
        diff_j = entities_j - entities_i

        # Case 1: 共享实体 + 不同实体 -> 可能属性冲突
        if diff_i and diff_j:
            text_i_lower = claim_i.claim.lower()
            text_j_lower = claim_j.claim.lower()

            for entity in shared:
                entity_lower = entity.lower()
                if entity_lower in text_i_lower and entity_lower in text_j_lower:
                    return ConflictEdge(
                        source_id=claim_i.claim_id,
                        target_id=claim_j.claim_id,
                        conflict_type=ConflictType.ENTITY_MISMATCH,
                        confidence=0.6,
                        rationale=f"Different values for shared entity '{entity}': {diff_i} vs {diff_j}"
                    )

        # Case 2: 否定冲突检测
        text_i_lower = claim_i.claim.lower()
        text_j_lower = claim_j.claim.lower()

        negation_patterns = [
            (" is ", " is not "),
            (" are ", " are not "),
            (" was ", " was not "),
            (" has ", " has no "),
            (" can ", " cannot "),
        ]
        for pos, neg in negation_patterns:
            if (pos in text_i_lower and neg in text_j_lower) or \
               (neg in text_i_lower and pos in text_j_lower):
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.ENTITY_MISMATCH,
                    confidence=0.8,
                    rationale=f"Negation conflict: '{claim_i.claim}' vs '{claim_j.claim}'"
                )

        return None

    def _llm_conflict_detection(
        self,
        claim_i: Claim,
        claim_j: Claim
    ) -> ConflictEdge | None:
        """Use LLM to detect claim relationships."""
        prompt = f"""Determine the relationship between these two claims.

Claim A: {claim_i.claim}

Claim B: {claim_j.claim}

Output JSON:
{{
    "relationship": "SUPPORT|REFUTE|PARTIAL_SUPPORT|UNRELATED",
    "confidence": <0.0-1.0>,
    "rationale": "brief explanation"
}}

Relationship types:
- SUPPORT: The claims agree with each other
- REFUTE: The claims contradict each other
- PARTIAL_SUPPORT: The claims partially agree but have differences
- UNRELATED: The claims don't address the same topic
"""

        try:
            response = self._call_llm(
                prompt,
                system_prompt=self.system_prompt,
                response_format="json"
            )
            data = json.loads(response)

            relationship = data.get("relationship", "UNRELATED")

            if relationship == "UNRELATED":
                return None

            conflict_type_map = {
                "SUPPORT": ConflictType.SUPPORT,
                "REFUTE": ConflictType.REFUTE,
                "PARTIAL_SUPPORT": ConflictType.PARTIAL_SUPPORT
            }

            return ConflictEdge(
                source_id=claim_i.claim_id,
                target_id=claim_j.claim_id,
                conflict_type=conflict_type_map.get(relationship, ConflictType.SUPPORT),
                confidence=data.get("confidence", 0.5),
                rationale=data.get("rationale", "")
            )

        except Exception:
            return None

    def update_graph(
        self,
        graph: EvidenceConflictGraph,
        new_evidence: list[Evidence],
        use_llm: bool = True
    ) -> EvidenceConflictGraph:
        """
        Update an existing graph with new evidence.

        Args:
            graph: Existing conflict graph
            new_evidence: New evidence to add
            use_llm: Whether to use LLM for conflict detection

        Returns:
            Updated graph
        """
        # Add new nodes
        for ev in new_evidence:
            for claim in ev.claims:
                if claim.claim_id not in graph.nodes:
                    node = ConflictGraphNode(
                        node_id=claim.claim_id,
                        content=claim.claim,
                        node_type="claim",
                        evidence_ids=[ev.evidence_id]
                    )
                    graph.add_node(node)

        # Check edges between new and existing nodes
        existing_claims = []
        for node in graph.nodes.values():
            if node.node_type == "claim":
                # Find corresponding claim
                for ev in new_evidence:
                    for claim in ev.claims:
                        if claim.claim_id == node.node_id:
                            existing_claims.append((claim, ev))

        new_claims = []
        for ev in new_evidence:
            for claim in ev.claims:
                new_claims.append((claim, ev))

        # Add edges
        for claim_i, ev_i in existing_claims:
            for claim_j, ev_j in new_claims:
                edge = self._detect_relationship(claim_i, ev_i, claim_j, ev_j, use_llm)
                if edge and edge.conflict_type != ConflictType.UNRELATED:
                    # Check if edge already exists
                    existing = any(
                        e.source_id == edge.source_id and e.target_id == edge.target_id
                        for e in graph.edges
                    )
                    if not existing:
                        graph.add_edge(edge)

        return graph

    def run(self, *args, **kwargs) -> Any:
        """Run the conflict graph builder."""
        return self.build_graph(*args, **kwargs)
