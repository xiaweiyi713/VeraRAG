"""Tests for entity conflict detection."""

import sys

sys.path.insert(0, 'src')

import unittest

from src.evidence.conflict_graph import ConflictGraphBuilder
from src.utils.data_structures import Claim, ClaimType, ConflictType


class TestEntityConflict(unittest.TestCase):
    """Test entity conflict detection."""

    def _make_claim(self, claim_id, text, entities=None, numbers=None, time_expressions=None):
        return Claim(
            claim_id=claim_id,
            claim=text,
            claim_type=ClaimType.FACTUAL,
            entities=entities or [],
            numbers=numbers or [],
            time_expressions=time_expressions or []
        )

    def test_negation_conflict(self):
        """检测否定冲突: 'X is Y' vs 'X is not Y'"""
        builder = ConflictGraphBuilder()
        claim_i = self._make_claim("C1", "Python is the best language", entities=["Python"])
        claim_j = self._make_claim("C2", "Python is not the best language", entities=["Python"])

        edge = builder._check_entity_conflict(claim_i, claim_j)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.ENTITY_MISMATCH)

    def test_different_entity_values(self):
        """检测同一属性不同实体值: 'capital is X' vs 'capital is Y'"""
        builder = ConflictGraphBuilder()
        claim_i = self._make_claim("C1", "The capital of France is Paris", entities=["France", "Paris"])
        claim_j = self._make_claim("C2", "The capital of France is Lyon", entities=["France", "Lyon"])

        edge = builder._check_entity_conflict(claim_i, claim_j)
        self.assertIsNotNone(edge)

    def test_no_conflict_different_entities(self):
        """不同实体不冲突"""
        builder = ConflictGraphBuilder()
        claim_i = self._make_claim("C1", "Paris is the capital of France", entities=["Paris", "France"])
        claim_j = self._make_claim("C2", "Berlin is the capital of Germany", entities=["Berlin", "Germany"])

        edge = builder._check_entity_conflict(claim_i, claim_j)
        self.assertIsNone(edge)

    def test_no_conflict_supporting(self):
        """相同实体不冲突"""
        builder = ConflictGraphBuilder()
        claim_i = self._make_claim("C1", "Einstein was born in Germany", entities=["Einstein", "Germany"])
        claim_j = self._make_claim("C2", "Einstein grew up in Germany", entities=["Einstein", "Germany"])

        edge = builder._check_entity_conflict(claim_i, claim_j)
        self.assertIsNone(edge)


if __name__ == "__main__":
    unittest.main()
