"""Unit tests for VeraRAG evaluation metrics."""

import sys

sys.path.insert(0, 'src')

import unittest

from src.evaluation.answer_metrics import AnswerMetrics
from src.evaluation.calibration_metrics import CalibrationMetrics
from src.evaluation.conflict_metrics import ConflictMetrics
from src.evaluation.evidence_metrics import EvidenceMetrics
from src.evaluation.hallucination_metrics import HallucinationMetrics


class TestAnswerMetrics(unittest.TestCase):
    """Test answer evaluation metrics."""

    def test_exact_match(self):
        """Test exact match calculation."""
        # Perfect match
        self.assertEqual(AnswerMetrics.exact_match("Paris", "Paris"), 1.0)
        self.assertEqual(AnswerMetrics.exact_match("Paris", "paris"), 1.0)
        self.assertEqual(AnswerMetrics.exact_match("Paris", "London"), 0.0)

    def test_f1_score(self):
        """Test F1 score calculation."""
        # Perfect match
        self.assertEqual(AnswerMetrics.f1_score("Paris", "Paris"), 1.0)
        # Partial match
        f1 = AnswerMetrics.f1_score("The capital is Paris", "Paris is the capital")
        self.assertGreater(f1, 0.5)
        # No match
        self.assertEqual(AnswerMetrics.f1_score("Paris", "London"), 0.0)


class TestEvidenceMetrics(unittest.TestCase):
    """Test evidence evaluation metrics."""

    def test_evidence_precision(self):
        """Test evidence precision."""
        # Perfect precision
        prec = EvidenceMetrics.evidence_precision(["E1", "E2"], ["E1", "E2"])
        self.assertEqual(prec, 1.0)

        # Half precision
        prec = EvidenceMetrics.evidence_precision(["E1", "E2"], ["E1", "E3"])
        self.assertEqual(prec, 0.5)

    def test_evidence_recall(self):
        """Test evidence recall."""
        # Perfect recall
        rec = EvidenceMetrics.evidence_recall(["E1", "E2"], ["E1", "E2"])
        self.assertEqual(rec, 1.0)

        # Half recall
        rec = EvidenceMetrics.evidence_recall(["E1"], ["E1", "E2"])
        self.assertEqual(rec, 0.5)

    def test_evidence_f1(self):
        """Test evidence F1."""
        f1 = EvidenceMetrics.evidence_f1(["E1", "E2"], ["E1", "E3"])
        expected = 2 * 0.5 * 0.5 / (0.5 + 0.5)
        self.assertEqual(f1, expected)


class TestConflictMetrics(unittest.TestCase):
    """Test conflict metrics."""

    def test_conflict_detection_f1(self):
        """Test conflict detection F1."""
        # Perfect match
        f1 = ConflictMetrics.conflict_detection_f1(
            [("A", "B"), ("C", "D")],
            [("A", "B"), ("C", "D")]
        )
        self.assertEqual(f1, 1.0)

        # Partial match
        f1 = ConflictMetrics.conflict_detection_f1(
            [("A", "B"), ("B", "C")],
            [("A", "B"), ("C", "D")]
        )
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_false_conflict_rate(self):
        """Test false conflict rate."""
        # No false conflicts
        rate = ConflictMetrics.false_conflict_rate(
            [("A", "B")],
            [("A", "B"), ("C", "D")]
        )
        self.assertEqual(rate, 0.0)

        # Some false conflicts
        rate = ConflictMetrics.false_conflict_rate(
            [("A", "B"), ("E", "F")],
            [("A", "B")]
        )
        self.assertEqual(rate, 0.5)


class TestCalibrationMetrics(unittest.TestCase):
    """Test calibration metrics."""

    def test_brier_score(self):
        """Test Brier score calculation."""
        # Perfect predictions
        brier = CalibrationMetrics.brier_score([1.0, 0.0, 1.0], [True, False, True])
        self.assertEqual(brier, 0.0)

        # Wrong predictions
        brier = CalibrationMetrics.brier_score([0.0, 1.0], [True, False])
        self.assertEqual(brier, 1.0)

    def test_ece(self):
        """Test Expected Calibration Error."""
        # Well calibrated
        ece = CalibrationMetrics.expected_calibration_error(
            [0.9, 0.1, 0.8],
            [True, False, True]
        )
        self.assertLess(ece, 0.2)

        # Poorly calibrated
        ece = CalibrationMetrics.expected_calibration_error(
            [0.5, 0.5, 0.5],
            [True, False, True]
        )
        self.assertGreaterEqual(ece, 0.0)


class TestHallucinationMetrics(unittest.TestCase):
    """Test hallucination metrics."""

    def test_unsupported_claim_rate(self):
        """Test unsupported claim rate."""
        claims = ["RAG helps", "RAG is perfect", "RAG has limits"]
        supported = {"RAG helps", "RAG has limits"}

        rate = HallucinationMetrics.unsupported_claim_rate(claims, supported)
        self.assertEqual(rate, 1/3)

    def test_extract_entities(self):
        """Test entity extraction."""
        text = "Apple and Google are tech companies"
        entities = HallucinationMetrics.extract_entities_from_text(text)

        self.assertIn("Apple", entities)
        self.assertIn("Google", entities)

    def test_extract_numbers(self):
        """Test number extraction."""
        text = "The accuracy was 85.5% and the error rate was 14.5%"
        numbers = HallucinationMetrics.extract_numbers_from_text(text)

        self.assertEqual(len(numbers), 2)
        # Note: percentages are divided by 100
        self.assertIn(0.855, numbers)
        self.assertIn(0.145, numbers)


if __name__ == "__main__":
    unittest.main()
