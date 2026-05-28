"""Unit tests for VeraRAG data structures."""

import sys

sys.path.insert(0, 'src')

import unittest

from src.utils.data_structures import (
    Claim,
    ClaimType,
    Complexity,
    ConflictEdge,
    ConflictGraphNode,
    ConflictType,
    Evidence,
    EvidenceConflictGraph,
    SubQuestion,
    TaskAnalysis,
    TaskType,
    UncertaintyBreakdown,
)


class TestClaim(unittest.TestCase):
    """Test Claim data structure."""

    def test_claim_creation(self):
        """Test creating a claim."""
        claim = Claim(
            claim_id="C1",
            claim="RAG improves factuality",
            claim_type=ClaimType.FACTUAL,
            entities=["RAG", "factuality"],
            numbers=["50"],
            confidence=0.9
        )

        self.assertEqual(claim.claim_id, "C1")
        self.assertEqual(claim.claim, "RAG improves factuality")
        self.assertEqual(claim.claim_type, ClaimType.FACTUAL)
        self.assertEqual(claim.confidence, 0.9)

    def test_claim_to_dict(self):
        """Test claim serialization."""
        claim = Claim(
            claim_id="C1",
            claim="Test claim",
            claim_type=ClaimType.FACTUAL
        )

        data = claim.to_dict()
        self.assertEqual(data["claim_id"], "C1")
        self.assertEqual(data["claim"], "Test claim")
        # Check the type value (enum returns the value when converted)
        self.assertEqual(data["claim_type"], "factual")


class TestEvidence(unittest.TestCase):
    """Test Evidence data structure."""

    def test_evidence_creation(self):
        """Test creating evidence."""
        evidence = Evidence(
            evidence_id="E1",
            source="paper",
            title="Self-RAG",
            text_span="RAG improves factuality...",
            credibility_score=0.9,
            relevance_score=0.8
        )

        self.assertEqual(evidence.evidence_id, "E1")
        self.assertEqual(evidence.source, "paper")
        self.assertAlmostEqual(evidence.combined_score, 0.84, places=1)

    def test_combined_score_calculation(self):
        """Test combined score calculation."""
        # High credibility, medium recency, high relevance
        ev1 = Evidence(
            evidence_id="E1",
            source="paper",
            title="Test",
            text_span="Content",
            credibility_score=0.9,
            recency_score=0.7,
            relevance_score=0.9
        )
        # Expected: 0.9*0.4 + 0.7*0.3 + 0.9*0.3 = 0.36 + 0.21 + 0.27 = 0.84
        self.assertAlmostEqual(ev1.combined_score, 0.84, places=2)


class TestConflictGraph(unittest.TestCase):
    """Test EvidenceConflictGraph."""

    def test_empty_graph(self):
        """Test empty conflict graph."""
        graph = EvidenceConflictGraph()

        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)
        self.assertEqual(graph.get_conflict_score(), 0.0)

    def test_add_node_and_edge(self):
        """Test adding nodes and edges."""
        graph = EvidenceConflictGraph()

        node1 = ConflictGraphNode(
            node_id="C1",
            content="Claim 1",
            node_type="claim"
        )
        node2 = ConflictGraphNode(
            node_id="C2",
            content="Claim 2",
            node_type="claim"
        )

        graph.add_node(node1)
        graph.add_node(node2)

        edge = ConflictEdge(
            source_id="C1",
            target_id="C2",
            conflict_type=ConflictType.REFUTE,
            confidence=0.8
        )
        graph.add_edge(edge)

        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.get_conflict_score(), 1.0)

    def test_conflict_vs_support(self):
        """Test distinguishing conflicts from supports."""
        graph = EvidenceConflictGraph()

        # Add some nodes
        for i in range(4):
            node = ConflictGraphNode(
                node_id=f"C{i}",
                content=f"Claim {i}",
                node_type="claim"
            )
            graph.add_node(node)

        # Add support edge
        graph.add_edge(ConflictEdge(
            source_id="C0",
            target_id="C1",
            conflict_type=ConflictType.SUPPORT,
            confidence=0.9
        ))

        # Add conflict edges
        graph.add_edge(ConflictEdge(
            source_id="C1",
            target_id="C2",
            conflict_type=ConflictType.REFUTE,
            confidence=0.8
        ))
        graph.add_edge(ConflictEdge(
            source_id="C2",
            target_id="C3",
            conflict_type=ConflictType.NUMERIC_CONFLICT,
            confidence=0.7
        ))

        supports = graph.get_supports()
        conflicts = graph.get_conflicts()

        self.assertEqual(len(supports), 1)
        self.assertEqual(len(conflicts), 2)
        self.assertAlmostEqual(graph.get_conflict_score(), 2/3, places=2)


class TestUncertaintyBreakdown(unittest.TestCase):
    """Test UncertaintyBreakdown."""

    def test_empty_uncertainty(self):
        """Test uncertainty with no components."""
        uncertainty = UncertaintyBreakdown()

        self.assertEqual(uncertainty.overall, 0.0)
        self.assertTrue(uncertainty.is_acceptable())

    def test_mixed_uncertainty(self):
        """Test uncertainty with mixed components."""
        uncertainty = UncertaintyBreakdown(
            retrieval_uncertainty=0.2,
            evidence_conflict=0.3,
            reasoning_gap=0.1,
            source_reliability=0.15,
            verification_uncertainty=0.1
        )

        # Expected: 0.2*0.25 + 0.3*0.3 + 0.1*0.2 + 0.15*0.15 + 0.1*0.1
        # = 0.05 + 0.09 + 0.02 + 0.0225 + 0.01 = 0.1925
        expected = 0.2 * 0.25 + 0.3 * 0.3 + 0.1 * 0.2 + 0.15 * 0.15 + 0.1 * 0.1
        self.assertAlmostEqual(uncertainty.overall, expected, places=3)

    def test_acceptable_threshold(self):
        """Test acceptable threshold check."""
        # Low uncertainty: 0.1 * 0.25 = 0.025 overall
        low_uncertainty = UncertaintyBreakdown(retrieval_uncertainty=0.1)

        # High uncertainty: all components at 0.8 = high overall
        high_uncertainty = UncertaintyBreakdown(
            retrieval_uncertainty=0.8,
            evidence_conflict=0.8,
            reasoning_gap=0.8,
            source_reliability=0.8,
            verification_uncertainty=0.8
        )

        self.assertTrue(low_uncertainty.is_acceptable(threshold=0.3))
        self.assertFalse(high_uncertainty.is_acceptable(threshold=0.3))


class TestSubQuestion(unittest.TestCase):
    """Test SubQuestion."""

    def test_subquestion_creation(self):
        """Test creating a sub-question."""
        sq = SubQuestion(
            id="sq1",
            question="What is RAG?",
            required_evidence_type="definition",
            status="pending"
        )

        self.assertEqual(sq.id, "sq1")
        self.assertEqual(sq.status, "pending")
        self.assertEqual(sq.coverage_score, 0.0)


class TestTaskAnalysis(unittest.TestCase):
    """Test TaskAnalysis."""

    def test_task_analysis_creation(self):
        """Test creating task analysis."""
        task = TaskAnalysis(
            task_type=TaskType.MULTI_HOP_QA,
            complexity=Complexity.HIGH,
            requires_retrieval=True,
            requires_conflict_check=True,
            estimated_hops=3
        )

        self.assertEqual(task.task_type, TaskType.MULTI_HOP_QA)
        self.assertEqual(task.complexity, Complexity.HIGH)
        self.assertTrue(task.requires_retrieval)
        self.assertTrue(task.requires_conflict_check)
        self.assertEqual(task.estimated_hops, 3)


if __name__ == "__main__":
    unittest.main()
